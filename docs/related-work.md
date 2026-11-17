# 相关方案调研（GitHub）

调研日期：**2026-11-17**。star 数为当日快照，仅用于判断项目影响力，不代表质量。

目标是回答一个问题：**「视觉 → 事件 → LLM 工具调用 → 可追溯实验记录」这条链路，GitHub 上有没有同类实现？**
结论是：**各段都有成熟方案，但把五段串成一个可审计闭环的项目很少。** 下面的对比可以帮助 LabScribe 明确差异化与下一步。

---

## 1. 最接近的项目

### MannLabs/proteomics_lab_agent — 多模态实验室 Agent（26★，Apache-2.0）

- https://github.com/MannLabs/proteomics_lab_agent
- 架构：ADK root agent + 子 Agent（Lab Knowledge / Instrument / QC Memory / Protocol），工具与知识库通过 **MCP server** 暴露（Confluence、AlphaKraken、本地数据库）。
- 亮点：**从视频生成实验协议**——专家边操作边讲解，Protocol Agent 分析视频+音频生成步骤；QC Memory 保存故障排查历史；回答「这台仪器今天能不能用」。
- 与 LabScribe 的关系：同样是「实验室 + 多模态 + Agent + 工具」。差异是它偏向**蛋白质组学数据/知识/仪器 QC**，没有实时摄像头事件流、低置信度队列、`record → frame` 追溯和事件级评测。
- 可借鉴：**MCP 化工具**、QC 历史记忆、仪器 readiness 查询。

### NVIDIA/sop-monitoring-blueprints — 视频 SOP 合规监控（65★）

- https://github.com/NVIDIA/sop-monitoring-blueprints
- 架构：训练微服务（微调 VLM Cosmos-Reason + 时序动作分割 DDM）+ 推理微服务（DeepStream 低延迟实时推理）+ Agentic Skills（自动生成/评估/部署）。
- 亮点：**VLM + 时序分割做 SOP 步骤识别与合规分析**，工业级实时流水线；需要自有标注视频和模型微调。
- 与 LabScribe 的关系：SOP 监控这一层最接近。差异是它重模型训练/部署，目标是「识别步骤是否合规」，而不是把每次读数变成带 `source_frame` 的可追溯实验记录，也没有逐事件的 LLM 工具调用链。
- 可借鉴：用 VLM/时序模型做**步骤分类**，与现有确定性状态机融合（模型负责语义，规则负责门禁）。

### zbbio/lab-robot — MCP 实验室视觉机器人（小众但非常贴近）

- https://github.com/zbbio/lab-robot
- 架构：MCP 摄像头工具（ONVIF/RTSP、截图、PTZ）+ **云端 VLM 图像问答** + `opentrons-mcp` 设备控制 + 语音终端。
- 亮点：自然语言「看看当前画面」→ 拉流截图 → VLM 回答；MCP 把摄像头与移液设备统一成工具。
- 与 LabScribe 的关系：验证了「摄像头即工具、Flash 类 VLM 做视觉问答」的路线。差异是没有事件折叠、读数稳定化、SOP 顺序校验、记录追溯与评测。
- 可借鉴：**把相机/画面理解做成 MCP 工具**，以及 VLM 作为低置信度帧的二次判读。

### KRATSZ/LabScript-AI — 执行感知的移液机器人 Agent（131★）

- https://github.com/KRATSZ/LabScript-AI
- 架构：自然语言协议 → 平台脚本；运行时用控制器状态 + 摄像头 + 压力曲线做**分层传感**；失败时提出恢复动作，由确定性 **allow / ask / suspend** 门禁决定是否执行；带 case memory。
- 亮点：**证据权威分层**（控制器状态为准，视觉/压力为辅助证据）与**确定性安全门禁**，非常适合高风险实验室自动化。
- 与 LabScribe 的关系：同样强调「Agent 提议、确定性层裁决、人类兜底」。差异是它面向机器人执行与恢复，LabScribe 面向记录、SOP 校验与审计。
- 可借鉴：给写记录/推进步骤之外的未来动作工具增加 `allow/ask/suspend` 策略；在记录中标注证据来源与权威等级。

### zaixizhang/LabOS / wu-yc/LabClaw — 自进化实验室 Co-Scientist（33★ / 1070★）

- https://github.com/zaixizhang/LabOS
- https://github.com/wu-yc/LabClaw
- 架构：多 Agent 协作（Researcher / Critic / Toolmaker）、98 个生物医学工具、Gemini 3、多模态感知与 XR 人机协作；LabSuperVision 是面向实验室的 VLM 基准。
- 与 LabScribe 的关系：工具生态与多模态感知值得参考，但主体是**干实验计算**，不覆盖摄像头事件流与仪器读数记录。
- 可借鉴：工具注册表/工具自动生成、真实场景 VLM 评测基准。

### ur-whitelab/chemcrow-public（971★）与 gomesgroup/coscientist（214★）

- https://github.com/ur-whitelab/chemcrow-public
- https://github.com/gomesgroup/coscientist
- 两者都证明「LLM + 领域工具 + 多步循环」在化学实验规划中可行。ChemCrow 用 LangChain 组织 RDKit/PubChem/paper-qa 等工具；Coscientist 接入网页与机器人 API。
- 与 LabScribe 的关系：工具调用模式一致，但都**不处理摄像头/仪器读数/实验记录可追溯**。

---

## 2. 互补生态

| 方向 | 项目 | 可借鉴点 |
| --- | --- | --- |
| 实验编排/采集 | [bluesky/bluesky](https://github.com/bluesky/bluesky)（244★） | `start/event/stop` 文档流，适合把 LabScribe 事件总线标准化 |
| 硬件抽象 | [PyLabRobot/pylabrobot](https://github.com/PyLabRobot/pylabrobot)（653★）、[Uni-Lab-OS](https://github.com/deepmodeling/Uni-Lab-OS)（189★）、[Octopus](https://github.com/richardingham/octopus)（23★） | 设备状态作为更高权威的证据源；实时监控走 websocket |
| ELN/记录 | [eLabFTW](https://github.com/elabftw/elabftw)（1494★）、[chemotion ELN](https://github.com/ComPlat/chemotion_ELN)（192★） | 记录导出 API、附件（原始帧）与审计字段 |
| 视频异常检测 | [EventVAD](https://github.com/YihuaJerry/EventVAD)（543★）、[LaVAD](https://github.com/lucazanella/lavad)（154★） | training-free、LLM/VLM 驱动的事件/异常推理 |
| 手物交互 | [HOI-DETR](https://github.com/AhmadDarKhalil/HOI-DETR)（219★）、[OakInk](https://github.com/oakink/OakInk)（153★） | 更细粒度的「手-仪器」交互与遮挡判断 |
| 科研 Agent | [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)（15k★） | 长程多步循环、可复现实验管理；与 LabScribe 的「湿实验记录」互补 |

---

## 3. 差异化定位

LabScribe 的组合在调研中没有发现完全相同的开源实现：

1. **连续帧 → 稳定读数 → 离散事件**（而不是逐帧问答或纯模型分类）；
2. **低置信度观测的待确认队列**，人工确认后可回灌 Agent；
3. **事件驱动工具调用**，SOP 顺序/读数范围校验、推进步骤、异常告警；
4. **记录 → 事件 → 原始帧** 的强追溯（模型不能伪造 frame id）；
5. **可复现的 100 场景基准**，覆盖遮挡、检测丢帧、OCR 空帧，并统计事件 P/R/F1、读数准确率、端到端延迟与遮挡 recall。

---

## 4. DeepSeek Flash 端点验证（2026-11-17）

`GET https://api.deepseek.com/models` 返回：

| 模型 id | 名称 | 上下文 | 输入模态 |
| --- | --- | --- | --- |
| `deepseek-flash` | DeepSeek-V4.1-Flash | 1,048,576 | **text + image** |
| `deepseek-v4-pro` | DeepSeek-V4-Pro | 1,048,576 | text |

已实测：

- `deepseek-flash` **支持 OpenAI 风格 tool calling**（返回 `tool_calls`，可驱动 LabScribe 的 8 个工具）；
- `deepseek-flash` **支持图片输入**：对 Demo 原始帧提问，能识别仪器标签 `TITRATOR`、读数 `15.2`、有人存在且显示区被遮挡；
- 因此本项目已将默认 Agent 模型切换为 **`deepseek-flash`**，仍可通过 `LABSCRIBE_LLM_MODEL` 覆盖。

Flash 的图片能力给出一条低成本增强路线：**不必先训练视觉模型，就能对低置信度帧做 VLM 二次判读**。

---

## 5. 建议的演进路线

### P0（低成本、直接补强当前闭环）

1. **Flash VLM 二次判读**：对低置信度 pending 项提供「VLM 复核」按钮/API；VLM 返回仪器、读数、是否遮挡、是否有人，作为人工确认的辅助证据（不自动改状态）。
2. **记录证据等级**：在记录里标注 `evidence: ocr | detector | vlm | human`，并保留原始帧；这与 LabScript-AI 的「分层传感/权威证据」一致。
3. **Agent 模型已切到 Flash**，保留 fallback 与规划建议。

### P1（互操作与审计）

4. **MCP Server**：把 camera、events、records、SOP 暴露为 MCP tools/resources（参考 proteomics_lab_agent、lab-robot），让 Claude/Cursor 等外部 Agent 直接查询「仪器是否就绪」「最新读数」。
5. **ELN 导出**：对接 eLabFTW API，记录作为实验条目，`source_frame` 作为附件；支持人工签名/审计。
6. **事件文档流**：提供 Bluesky 风格的 `start/event/stop` 文档或 MQTT/WebSocket 订阅，方便接入编排系统。

### P2（更强的视觉理解）

7. **VLM 步骤分类**：用 Flash 对短帧窗口判断当前 SOP 步骤，与规则状态机融合——模型给语义判断，规则仍做确定性门禁（参考 NVIDIA SOP）。
8. **手物交互/姿态**：引入 HOI/pose 模型，改善「操作者是否真的在操作仪器」与遮挡判定。
9. **真实场景评测**：用 LabSuperVision 等真实实验室 VLM 基准校准；保留现有合成基准用于 CI。

### P3（执行扩展，若未来加入控制）

10. **allow / ask / suspend 安全门禁**：任何非只读工具（如控制仪器、弃用数据）都经过确定性策略；分层证据 + 人类兜底（参考 LabScript-AI）。
11. **数字孪生/XR** 与工具自动生成（LabOS）属于更大范围，不建议在记录 Agent 阶段引入。

---

## 6. 明确不照搬

- **一开始就微调 VLM**（NVIDIA）：标注与算力成本高，先用 Flash API 验证价值，再决定是否自训。
- **机器人控制/协议生成**（LabScript-AI、LabOS）：与「记录与审计」是不同产品边界，可通过 MCP/ELN 集成而不是重写。
- **纯 VLM 逐帧推理**：成本高、时序不稳定；保留「轻量视觉 + 稳定器 + 规则状态机」，把 VLM 用在低置信度与语义步骤上更划算。
