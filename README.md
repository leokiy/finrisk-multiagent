# 🏦 FinRisk MultiAgent

<div align="center">

**多 Agent 协作金融风险智能分析系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DashScope](https://img.shields.io/badge/LLM-DashScope(Qwen)-orange.svg)](https://dashscope.aliyun.com/)

*不是一个人工智能回答你——而是一个 AI 分析团队在帮你分析金融风险*

</div>

---

## 📌 这是什么？

FinRisk MultiAgent 是一个基于**多智能体协作（Multi-Agent Collaboration）**架构的金融文档风险分析平台。

你上传一份金融文档（年报、招股书、债券募集说明书等），提出风险相关问题，系统调度 **4 个各司其职的 AI Agent** 并行分析，最后由协调 Agent 综合成一份多视角的结构化风险评估报告。

### 核心特色

- 🧠 **Multi-Agent 协作**: 不是单模型"一问一答"，而是 4 个专业化 Agent 分工协作
- 📎 **RAG 文档检索**: 基于 FAISS 向量库 + DashScope Embedding，从上传的 PDF 中精准检索相关段落，支持全量表格提取
- 🔍 **Devil's Advocate 机制**: 内置"魔鬼代言人"Agent，专门挑战其他 Agent 的结论，防止盲点
- 🌐 **Agent 级联网搜索**: 每个 Agent 独立调用 DashScope 内置搜索（enable_search），当文档信息不足时自主联网补充最新行业数据、新闻和监管动态
- 🔑 **用户自有 API Key**: 不内置任何密钥，用户完全掌控自己的 API 使用
- 📊 **结构化输出**: 四维风险矩阵（市场/信用/流动/操作与治理）+ 合规审查清单 + 可操作建议
- ⚡ **流式输出**: 最终报告逐字实时生成，无需等待全部完成

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│              👤 Streamlit Web UI                 │
│         上传文档 · 提问 · 查看报告 · 下载          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           🎯 Orchestrator 协调 Agent              │
│    拆解用户问题 → 调度专业 Agent → 综合最终报告     │
└──────┬──────────┬──────────┬──────────┬─────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  📊 数据   │ │  ⚠️ 风险  │ │  📋 合规  │ │  🔍 深度  │
│ 提取Agent │ │ 评估Agent │ │ 审查Agent │ │ 质疑Agent │
│          │ │          │ │          │ │          │
│ 提取关键  │ │ 四个维度  │ │ 对照监管  │ │ 挑战结论  │
│ 财务指标  │ │ 逐项评分  │ │ 框架审查  │ │ 找出盲点  │
│ 结构化输出│ │ 风险定级  │ │ 合规风险  │ │ 反向叙事  │
└──────┬───┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
       │          │          │          │
       └──────────┴──────────┴──────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│             📚 RAG 知识库（底层支撑）               │
│   PDF → 文本提取 → 语义分块 → Embedding 向量化      │
│   → FAISS 索引 → 各 Agent 独立检索相关段落          │
└─────────────────────────────────────────────────┘
```

### 工作流程

```
📄 上传文档 → 📂 处理文档(PDF解析·表格提取·向量化) → 🔍 理解问题(查询转述)
   → 🤖 Agent 并行分析(3 Agent + 各自联网搜索) → ⚖️ 质疑与裁决(Devil's Advocate)
   → 📊 流式生成报告(逐字实时输出)
```

| 阶段 | 做什么 | 并行/串行 |
|------|--------|:--:|
| **第零轮** | 问题理解与转述——明确分析目标 | 串行 |
| **第一轮** | 数据提取、风险评估、合规审查 3 个 Agent 并行执行，各自可联网搜索补充信息 | 并行 ⚡ |
| **第二轮** | 深度质疑 Agent 看到前三方的输出后，进行挑战性审阅（也可联网搜索反向证据） | 串行 |
| **第三轮** | 协调 Agent 综合所有输出，处理矛盾，流式生成最终报告 | 串行 |

### 四个专业 Agent

| Agent | 角色 | 核心职责 | 输出 |
|-------|------|----------|------|
| 📊 **数据提取** | 审计式抄写员 | 只提取数字和事实，不分析不推断 | 结构化财务指标表格（标注来源页码） |
| ⚠️ **风险评估** | 风控专家 | 从市场/信用/流动/操作与治理四维度逐项评分 | 四维风险矩阵 + 详细分析 + 证据引用 |
| 📋 **合规审查** | 监管者视角 | 对照金融监管框架逐条检查合规风险 | 合规检查清单（通过/警示/违规） |
| 🔍 **深度质疑** | 魔鬼代言人 | 刻意挑战其他 Agent 的结论，寻找被遗漏的盲点 | 盲点清单 + 矛盾分析 + 反向叙事 |

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- [DashScope API Key](https://dashscope.console.aliyun.com/apiKey)（通义千问），或任何 OpenAI 兼容的 API

### 1. 克隆项目

```bash
git clone https://github.com/leokiy/finrisk-multiagent.git
cd finrisk-multiagent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
# Linux / macOS / Windows (推荐)
python -m streamlit run app.py

# 或使用启动脚本
# Windows: 双击 run.bat
# Linux/macOS: bash run.sh
```

### 4. 使用

1. 在浏览器中打开 `http://localhost:8501`
2. 在左侧边栏输入你的 DashScope API Key
3. 上传一份金融文档（PDF / TXT / MD）
4. 提问 —— 系统会自动启动 4 个 Agent 协作分析
5. 查看综合报告，可下载 Markdown 格式的完整报告

### 可选：使用 .env 文件

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

---

## 📂 项目结构

```
finrisk-multiagent/
├── app.py                      # Streamlit 主界面
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── .gitignore
├── README.md                   # 本文件
│
├── src/
│   ├── orchestrator.py         # 🎯 中央调度器：Agent 编排 + 报告合成
│   ├── agents/
│   │   ├── base.py             # Agent 基类：RAG 检索 + LLM 调用 + 后处理
│   │   ├── data_extractor.py   # 📊 数据提取 Agent
│   │   ├── risk_assessor.py    # ⚠️ 风险评估 Agent
│   │   ├── compliance_checker.py # 📋 合规审查 Agent
│   │   └── devils_advocate.py  # 🔍 深度质疑 Agent
│   ├── rag/
│   │   └── engine.py           # RAG 模块：PDF 加载 · 文本分块 · 全量表格提取 · Embedding · FAISS
│   ├── search/
│   │   └── web_search.py       # 联网搜索：DashScope enable_search（Agent 自主触发）
│   └── llm/
│       └── client.py           # LLM 客户端：DashScope + OpenAI 兼容双模式 · 流式输出
│
├── prompts/                    # 📝 Prompt 模板（Markdown 格式，易于修改）
│   ├── data_extractor.md       # 数据提取 Agent 的 system prompt
│   ├── risk_assessor.md        # 风险评估 Agent 的 system prompt
│   ├── compliance_checker.md   # 合规审查 Agent 的 system prompt
│   ├── devils_advocate.md      # 深度质疑 Agent 的 system prompt
│   └── orchestrator.md         # 协调 Agent 的 synthesis prompt
│
└── examples/                   # 示例文件
    └── sample_report.md        # 示例分析报告
```

---

## 🔧 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Streamlit | 纯 Python Web UI，零前端代码 |
| **LLM** | DashScope (Qwen) / OpenAI 兼容 | 用户自选 API，支持 qwen-turbo/plus/max |
| **RAG** | FAISS + DashScope Embedding | 本地向量存储，全量 PDF 表格提取 |
| **联网搜索** | DashScope enable_search | Agent 自主触发，文档信息不足时自动补充 |
| **文档处理** | pdfplumber + LangChain | PDF 文本提取 + 递归语义分块 + 表格结构化 |
| **Agent 框架** | 自研轻量框架 | 不依赖 LangChain Agent / AutoGen，完全可控 |
| **并行调度** | concurrent.futures | 标准库，3 Agent 并行 + 流式输出 |

### 为什么不用 LangChain Agent / AutoGen / CrewAI？

本项目选择**自研轻量 Agent 框架**，原因：
- **完全可控**: 4 个 Agent + 1 个 Orchestrator 的架构清晰简单，不依赖重型框架的黑盒抽象，每一行逻辑都可审计和定制
- **性能优势**: 无框架开销，Agent 调度延迟更低；并行检索 + 流式输出，端到端响应时间可控
- **易于扩展**: 新增 Agent 仅需 20 行代码，Prompt 外置为 Markdown 文件，业务人员可直接调整分析逻辑而无需修改代码

---

## 📊 示例输出

<details>
<summary>点击展开示例分析报告（年报风险分析）</summary>

```markdown
# 🏦 金融风险综合评估报告

> **分析对象**: XX 股份有限公司
> **综合风险评级**: 中高风险
> **综合风险评分**: 13/20

## 📌 执行摘要

### 三个最关键的信息
1. **短期偿债压力大**: 短期借款 + 一年内到期负债占总负债 62%，而货币资金覆盖率仅 0.7 倍
2. **经营现金流持续为负**: 连续 3 年经营现金流为负，利润质量存疑
3. **大股东高比例质押**: 控股股东质押比例达 78%，存在控制权变更风险

### 四维风险雷达
| 风险维度 | 等级 | 趋势 | 关键发现 |
|----------|:----:|:----:|----------|
| 市场风险 | 3/5 | ↑ | 行业竞争加剧，毛利率连续下滑 |
| 信用风险 | 4/5 | ↑ | 有息负债率偏高，利息保障倍数仅 2.1x |
| 流动性风险 | 4/5 | → | 流动比率仅 0.92，经营现金流为负 |
| 操作与治理风险 | 2/5 | → | 内控基本合规，但大股东质押比例过高 |

...
```
</details>

完整示例见 [examples/sample_report.md](examples/sample_report.md)

---

## 🎯 适用场景

| 场景 | 说明 |
|------|------|
| **投资尽调** | 快速分析目标公司的财务风险和合规状况 |
| **持仓监控** | 定期审查持仓标的的风险变化 |
| **信用评估** | 评估债券发行人的信用风险 |
| **合规自查** | 对照监管框架检查信息披露的完整性 |
| **监管科技** | 辅助监管机构进行信息披露合规检查和风险筛查 |
| **投后监控** | 定期跟踪持仓标的财务状况，结合联网搜索获取最新行业动态和风险事件 |

---

## ⚠️ 免责声明

- 本系统由 AI 驱动，分析结果**仅供参考**，不构成投资建议、法律意见或任何形式的专业建议
- 分析仅基于用户上传的文档内容，未使用实时市场数据或外部数据库
- 金融监管框架摘要仅用于合规风险提示，不构成正式的法律合规意见
- 用户应在做出任何决策前咨询持牌专业人士
- 使用本系统即表示您已理解并同意上述声明

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [DashScope (阿里云灵积)](https://dashscope.aliyun.com/) — 提供 Qwen 系列大模型 API
- [Streamlit](https://streamlit.io/) — 让 Python 脚本变成 Web 应用
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI 的高效向量检索库
