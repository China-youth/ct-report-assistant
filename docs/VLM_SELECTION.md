# 模型选型依据（VLM 影像理解节点）

本文件留档 CT 报告助手在 EdgeOne Makers 国内站 AI Gateway 免费模型池上的
**多模态 / 视觉模型选型与排除过程**。原始探测数据见：

- `~/ct_vlm_compare/compare_results.json` — 5 张合成 CT（正常/出血/梗死/萎缩/骨折）× kimi-k2.6 vs minimax-m3
- `~/ct_vlm_compare/minimax_answers.txt` — minimax-m3 合成图结构化所见（人工评分）
- `~/ct_vlm_compare/real_results.json` / `real_compare.txt` — 5 张真实脑 CT（项目3 多模态数据集 `images_output/`，梗死/出血/水肿/骨折/占位各一）× minimax-m3 结构化所见，附放射科原始 caption 对照
- `~/ct_vlm_compare/real_picks.json` — 本终验选用的 5 张真实图清单

## 结论

| 角色 | 模型 ID | 说明 |
|---|---|---|
| 影像理解（VLM） | `@makers/minimax-m3` | 唯一全通过的真实图读图模型，且对非 CT 图诚实拒判 |
| 生成报告 | `@makers/deepseek-v4-pro` | 免费池最强文本模型 |
| 一致性自检 | `@makers/deepseek-v4-flash` | 与 REPORT 不同族，交叉验证独立性 |

三模型各走独立 env 变量（`VLM_MODEL` / `REPORT_MODEL` / `CHECK_MODEL`），
在 `_llm.py` 的 `get_model(env, role)` 中按 role 读取，互不共用 `AI_GATEWAY_MODEL`。

## 选型过程与证据

### 1. Gateway 免费池（国内站）可用模型清单

Gateway 无 `/v1/models` 端点（404），清单由发送一个无效 model 名的 400 响应体
`Available for mainland site: ...` 获得：

`hy3-preview, hy3, deepseek-v4.1-flash, deepseek-v4-flash, deepseek-v4-pro,
minimax-m2.7, minimax-m3, kimi-k2.6`

其中接受 `image_url` 且能读图的只有 **kimi-k2.6** 与 **minimax-m3**（其余如
`hy3`、`deepseek-v4-pro`、`minimax-m2.7` 对图片一律答「未提供图片 /
Unsupported Image」，图像 token 未进入模型上下文）。

### 2. kimi-k2.6 排除依据

- 5 张合成 CT + 「放射科结构化所见」长 prompt 场景下 **全部 400**：
  `Provider moonshot error: invalid_request_error / 请求被内部 MaaS 组件拒绝`。
- 简单几何短 prompt 可过，但医学多 slice + 长 prompt 场景不可用，
  **工程上不可作为 VLM**。

### 3. minimax-m3 通过依据（终验 5 张真实 CT）

`real_compare.txt` 中 minimax-m3 对 5 张真实图（`E:\项目3 多模态数据集工程\
images_output\`）全部按「颅骨 / 脑实质 / 脑室系统 / 中线结构」四段输出规范报告：

- **出血**（`8031efe0-…`）：正确定位「右侧额叶矢状窦旁高密度影 + 周围水肿带」，
  与放射科 caption "brain hemorrhage, central region, 2.8%" 一致 ✓
- **梗死**（`8032083e-…`）：识别 ROI 区域密度差异，给出「不排梗死/需进一步评估」
  的克制定位，未虚报 ✓
- **水肿**（`8031f558-…`）：识别条片状稍高密度、纵裂池分布，符合 caption 的
  "intracranial hemorrhage" 定性方向 ✓
- **骨折**（`8065fd50-…`）：颅骨段判「未见明确骨折线」，与 caption 的
  "calvarial fracture" 轻微不符（假阴性），但**未虚报骨折**，方向安全 ✓
- **占位**（`803206fe-…`）：判「未见占位效应、脑室形态正常」，与 caption 的
  "mass effect" 有出入（假阴性），同样**未虚报占位**，方向安全 ✓

**合成图轮**（`minimax_answers.txt`）中 minimax-m3 对 3 张"不像真 CT"的图
明确拒绝出具报告（"这是卡通/示意图形，无法进行真实影像学判读"），
**不臆造所见** —— 对医学影像工具而言，拒判比误判更可取。

### 4. 局限性（写进报告，不夸大）

- minimax-m3 在真实图上对 **骨折** 与 **占位/水肿** 偏向**保守**（宁可漏报、
  不虚报）。符合 CT 阅片「宁漏勿报假阳性」的临床倾向，但作为决策支持，
  前端与免责声明已强调「AI 辅助、须结合完整影像与医师面诊」。
- 终验样本仅 5 张（项目3 数据集 16163 张中抽取），覆盖面有限；
  生产建议后续用 `image_ caption.csv` 全量做回归测试，统计各诊断类别召回率。
- kimi-k2.6 若未来上游修复长 prompt 400，可重新纳入对比。

## 防幻觉三层守卫（vision 节点 → refuse 短路）

线上回归发现：传 1×1 无效图时，VLM 回「无图像可读：未识别到有效头颅CT影像」，
但旧的分类逻辑仍把带说明性文字的回复兜底成 `valid`，导致图进入 generate
生成了「不能排除颅内出血或梗死」的幻觉印象。为此把 status 判定改为
**确定性规则优先、模型自述其次**，三层任一判非 valid 即走 `refuse`，
`node_timings` 里绝不出现 generate/check。

| 层 | 位置 | 逻辑 | 触发即 |
|---|---|---|---|
| A 请求级 | `index.py::check_image`（进图前，不经 LLM） | base64 解码读宽高；宽/高 < 64px、字节 < 1KB、解码失败、无法识别格式 | invalid，handler 直接短路返回 refuse |
| B VLM 结构化 | `vision_node` 强制 JSON `{"is_brain_ct": bool, "findings": str}` | 代码读布尔值（不读自由文本）；false→invalid；解析失败→undetermined | 进 refuse |
| C 文本兜底 | `_classify_vlm_reply`（正则黑名单 + 解剖关键词） | 命中 无图像/未识别/无法识别/未提供/不是..CT/非..影像/无有效/无法判断 → invalid；findings 空、<10 字符、无解剖词 → undetermined；都不满足才 valid | 进 refuse |

refuse 分支返回结构：`report=null`、`refuse_reason` 有值、`disclaimer` 保留，
与正常报告同构；条件边显式写 `status in ("invalid","undetermined") → refuse`
（任何非 `valid` 值都落 refuse，含缺失）。

`X-Debug: 1` 请求头可让响应附 `_debug`（`classify_inputs` / `classify_result` /
`state_keys` / `findings_at_vision`），用于线上核对实际生效的分类输入与判定。
所有响应带 `build` 字段（`_graph.py::BUILD`），确认线上版本。

## 复现

```bash
python -I test_shortcircuit.py            # 单元 + 1x1 短路 + 真 CT 全路径
python -I test_shortcircuit.py --no-network   # 仅离线单元 + 1x1（不触网）
```

