# 脑 CT 影像报告助手（ct-report-assistant）

基于 **Python + LangGraph（官方 SDK，StateGraph 显式节点/边/条件边）** 的脑 CT 影像
报告 Agent，部署在 EdgeOne Makers 国内站，加速区域为全球可用区（不含中国大陆）。

> ⚠️ 医学免责声明：本工具由 AI 辅助生成，仅用于辅助阅片与信息整理，
> **不能替代执业医师的判断与面诊**。任何临床决策请结合完整影像资料与患者
> 实际情况，由具备资质的放射科医师或相关专科医师作出。

## 图结构

```
影像理解(vision) → 知识检索(retrieval) → 生成报告(generate) → 一致性自检(check)
                                                              │ 自检不通过且轮次<2
                                                              └──► 回退到 generate 重来
                                                              │ 通过
                                                              └──► 输出结构化报告 + 免责声明
```

节点均用 `@timed(...)` 装饰器记录单节点耗时，随结果一并返回前端展示。

### 为什么「影像理解」是文本而非 VLM 读图

EdgeOne Makers 国内站 AI Gateway 免费模型池（`hy3 / hy3-preview /
deepseek-v4.1-flash / deepseek-v4-flash / deepseek-v4-pro / minimax-m2.7 /
minimax-m3 / kimi-k2.6`）中**没有能真正读图的视觉模型**：`hy3`、
`deepseek-v4-pro`、`minimax-m2.7` 虽接受 `image_url` 参数，但对其内置图片
一律回答「未提供图片 / 无法查看图片 / Unsupported Image」，图像 token 未
进入模型上下文。因此 v1 的影像理解节点接收**放射科已给出的「所见」文本**
（或用户自述），后续迭代可替换为真实视觉模型。

## 模型

- 文本 LLM（检索、报告生成、自检）：`@makers/deepseek-v4-pro`（`_llm.py` 的
  `STRONG_MODEL`）；默认快模 `@makers/deepseek-v4-flash`（`DEFAULT_MODEL`）。
- 模型名 / base URL / key 全部由 `context.env` 注入（`AI_GATEWAY_*`），
  代码里不硬编码。

## 本地跑通 pipeline（不依赖 Makers runtime）

```bash
pip install -r requirements.txt
python -I run_smoke.py   # 用旧项目 .env 里的 gateway 凭据跑一遍整图
```

## 目录

- `agents/report/` — Python agent（`index.py` handler、`_graph.py` StateGraph、`_llm.py` 模型+本地知识库）
- `src/` — Vite + React 前端（上传 CT 图、报告展示、节点耗时面板、免责声明）
- `edgeone.json` — `agents.framework: langgraph`
- `.env.example` — 声明 `AI_GATEWAY_API_KEY` / `AI_GATEWAY_BASE_URL`（CLI 自动注入）

## 部署

```bash
export PAGES_SOURCE=skills
edgeone makers link                # 首次：创建/链接国内站项目
edgeone makers env pull
edgeone makers deploy -n ct-report-assistant --area global -e production --json
```

- `--area global` = 全球可用区（不含中国大陆）
- 子域名 `ct.hyhagent.dpdns.org`（免费域名，无需备案）待绑定
- `.edgeone/` 是构建产物，已 gitignore，**不要手改**；若部署遇到
  `EPERM ... agent-python/_platform`，删掉整个 `.edgeone/` 重新 `makers link` 即可
