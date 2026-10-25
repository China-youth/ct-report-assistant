"""Image preprocessor shared by the vision Provider path and any test script.

Before an image enters the vision node it is decoded, shrunk to at most
1024px on the longest edge, and re-encoded as JPEG quality 85. This keeps
the base64 payload far below the provider's text-token accounting threshold
(the 402 root cause: an unshrunk 1.5MB upload counted as text tokens).

Usage (Provider function and tests alike):
    from preprocess_image import preprocess_image
    url = preprocess_image(data_url_or_http_url)
"""
from __future__ import annotations

import base64
import io
import re

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is a hard dep in practice
    Image = None

# Longest-edge cap chosen so a 1024px JPEG q85 stays under ~300KB in base64.
_MAX_EDGE = 1024
_JPEG_QUALITY = 85


def _shrink_and_reencode(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    w, h = img.size
    scale = _MAX_EDGE / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def preprocess_image(image: str) -> str:
    """Return a JPEG data-URL, longest edge <= 1024px.

    Accepts:
      - ``data:<mime>;base64,<payload>``  -> decoded, shrunk, re-encoded
      - ``http://...`` / ``https://...``  -> returned unchanged (remote URLs
        are not fetched here; the provider handles them or rejects them)
    """
    if not image:
        return image
    if image.startswith(("http://", "https://")):
        return image
    m = re.match(r"data:[^;,]+;base64,(.+)$", image, re.DOTALL)
    if not m:
        # Raw base64 (no data-URL wrapper) — decode and re-wrap.
        try:
            raw = base64.b64decode(image, validate=True)
        except Exception:
            return image
    else:
        try:
            raw = base64.b64decode(m.group(1), validate=True)
        except Exception:
            return image
    try:
        jpeg = _shrink_and_reencode(raw)
    except Exception:
        # Undecodable or PIL unavailable — fall back to the original so the
        # caller's existing error handling still fires.
        return image
    b64 = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
