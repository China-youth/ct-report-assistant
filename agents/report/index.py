"""POST /report — run the brain-CT report pipeline and return the result.

Entry signature follows the Makers Python agent runtime:
``async def handler(ctx)`` returning a JSON-able dict.

Request body:
    {
      "description": "双侧大脑半球对称，脑室系统无扩大，未见明显出血及梗死灶。",
      "image": "data:image/png;base64,..."   // optional; when present the VLM node reads it
    }

Response (uniform shape across normal / refuse / error):
    {
      "build": "...",
      "report": {...} | null,
      "refuse_reason": "" | "...",
      "image_status": "valid" | "invalid" | "undetermined",
      "selfcheck": {...},
      "node_timings": [...],   // refuse path: vision/refuse only, never generate/check
      "iteration": int,
      "evidence": [...],
      "disclaimer": "..."
    }

With request header ``X-Debug: 1`` the response gains a ``_debug`` object:
    {
      "classify_inputs": {"image_check": <rule-a result>,
                          "vlm_json": <rule-b result>,
                          "text": <rule-c result>},
      "classify_result": <final image_status>,
      "state_keys": [<state keys the vision node wrote>],
      "findings_at_vision": <text the classifier received>
    }
"""
from __future__ import annotations

import base64
import json
import re

from ._graph import (
    BUILD,
    DISCLAIMER,
    parse_vlm_reply,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Rule A — request-level deterministic image check (no model involved).
# A 1x1 (or any small / non-decodable / undersized) upload must be rejected
# before it ever reaches the VLM.
# ---------------------------------------------------------------------------
_JPEG_SOI = b"\xff\xd8"


def _dimensions(data: bytes):
    """Best-effort (width, height) for PNG / JPEG / BMP, else None.

    PNG: IHDR holds big-endian uint32 width/height right after the
    8-byte signature + "IHDR" tag.
    JPEG: walk marker segments until a size-appropriate SOF.
    BMP: bytes 18-22 are little-endian int32 width/height.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if data[12:16] == b"IHDR":
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h
        return None
    if data[:2] == _JPEG_SOI:
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in (0xD8, 0xD9):
                continue
            if 0xD0 <= marker <= 0xD7:
                continue
            if marker in (0xC0, 0xC1, 0xC2, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = int.from_bytes(data[i + 5: i + 7], "big")
                w = int.from_bytes(data[i + 3: i + 5], "big")
                return w, h
            seg_len = int.from_bytes(data[i: i + 2], "big")
            i += 2 + seg_len
        return None
    if data[:2] == b"BM" and len(data) >= 26:
        w = int.from_bytes(data[18:22], "little")
        h = abs(int.from_bytes(data[22:26], "little", signed=True))
        return w, h
    return None


# 64px minimum dimension. A 1x1, 32x32, etc. upload can never be a usable CT.
_MIN_DIM = 64
# Refuse anything under 1 KB — a genuine CT slice is far larger even compressed.
_MIN_BYTES = 1024


def check_image(image: str | None) -> dict:
    """Rule A output. ``valid`` is a hard gate: False => status=invalid."""
    if not image:
        return {"valid": True, "note": "no image; text-only path"}
    # data URL: data:<mime>;base64,<payload>
    m = re.match(r"data:[^;,]+;base64,(.+)$", image, re.DOTALL)
    if m:
        raw = m.group(1)
    elif image.startswith(("http://", "https://")):
        # Remote URL: the server side would need to fetch it; we only gate the
        # in-band uploads. Let it through and let the VLM decide.
        return {"valid": True, "note": "remote url; decode gate not applicable"}
    else:
        raw = image
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        return {"valid": False, "note": "base64 decode failed"}
    if len(data) < _MIN_BYTES:
        return {"valid": False, "note": f"bytes {len(data)} < {_MIN_BYTES}"}
    dims = _dimensions(data)
    if dims is None:
        # Unknown format. We could not verify dimensions, but the byte-size
        # gate already passed. Mark as undetermined-equivalent: hard-refuse,
        # because a real CT upload will always be a decodable PNG/JPEG/BMP.
        return {"valid": False, "note": "unrecognized image format"}
    w, h = dims
    if w < _MIN_DIM or h < _MIN_DIM:
        return {"valid": False, "note": f"dimension {w}x{h} < {_MIN_DIM}px"}
    return {"valid": True, "note": f"ok {w}x{h}"}


# ---------------------------------------------------------------------------
# Rule B — forced VLM JSON shape.
# ``_VLM_SYS`` (the prompt template) and ``parse_vlm_reply`` live in
# ``_graph.py`` so the vision node and the unit tests share one parser.
# Imported at the top of this module; used here only via the shared imports.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
_EMPTY_USAGE = {
    "per_node": [], "totals": {"prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0}, "model_calls": 0,
}


def _empty_usage():
    import copy
    return copy.deepcopy(_EMPTY_USAGE)


async def handler(ctx):
    body = ctx.request.body or {}
    description = (body.get("description") or "").strip()
    image = body.get("image") or None
    # Ablation mode: "A" baseline / "B" half / "C" full (default).
    mode = (body.get("mode") or "C").strip().upper()
    if mode not in ("A", "B", "C"):
        mode = "C"
    if not description and not image:
        return {"build": BUILD, "mode": mode, "usage": _empty_usage(),
                "error": "请提供影像所见描述（description）或 CT 图像。"}, 400

    # X-Debug: 1 toggles the debug payload in the response.
    debug = (ctx.request.headers or {}).get("x-debug", "").strip() in ("1", "true", "yes")

    # Rule A: deterministic image gate.
    img_check = check_image(image)
    if not img_check["valid"]:
        # Short-circuit before touching the graph: no model call, no generation.
        return {
            "build": BUILD,
            "mode": mode,
            "report": None,
            "refuse_reason": (
                f"未识别到有效头颅 CT 影像（图像校验未通过：{img_check['note']}），"
                "请重新上传清晰、完整的头颅 CT 轴位图像。"
            ),
            "image_status": "invalid",
            "selfcheck": {"passed": None, "issues": ["图像校验未通过"]},
            "node_timings": [],
            "usage": _empty_usage(),
            "iteration": 0,
            "evidence": [],
            "disclaimer": DISCLAIMER,
            **({"_debug": {
                "rule_a": img_check,
                "rule_b": {"skipped": True, "reason": "rule A hard-blocked"},
                "rule_c": {"skipped": True, "reason": "rule A hard-blocked"},
                "classify_result": "invalid",
                "state_keys": [],
                "findings_at_vision": "",
            }} if debug else {}),
        }

    try:
        result = await run_pipeline(ctx, description, image,
                                    img_check=img_check, debug=debug,
                                    mode=mode)
    except Exception as e:
        return {
            "build": BUILD,
            "mode": mode,
            "report": None, "refuse_reason": f"处理失败：{e}",
            "image_status": "undetermined",
            "selfcheck": {"passed": False, "issues": [str(e)]},
            "node_timings": [], "usage": _empty_usage(),
            "iteration": 0, "evidence": [],
            "disclaimer": DISCLAIMER, "error": str(e),
            **({"_debug": {
                "rule_a": img_check,
                "rule_b": {"error": str(e)[:200]},
                "rule_c": {"error": str(e)[:200]},
                "classify_result": "undetermined",
                "state_keys": [], "findings_at_vision": "",
            }} if debug else {}),
        }

    if debug:
        dbg = result.get("_debug") or {}
        result = dict(result)
        result["_debug"] = {
            "rule_a": img_check,
            "rule_b": dbg.get("rule_b"),
            "rule_c": dbg.get("rule_c"),
            "classify_inputs": {
                "image_check": img_check,
                "vlm_json": dbg.get("rule_b"),
                "text": dbg.get("rule_c"),
            },
            "classify_result": result.get("image_status"),
            "state_keys": dbg.get("state_keys", []),
            "findings_at_vision": dbg.get("findings_at_vision", ""),
        }
    return result
