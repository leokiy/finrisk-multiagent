# 🏦 FinRisk MultiAgent

<div align="center">

**多 Agent 协作金融风险智能分析系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DashScope](https://img.shields.io/badge/LLM-DashScope(Qwen)-orange.svg)](https://dashscope.aliyun.com/)

*不是一个 AI 回答你——而是一个 AI 分析团队在帮你分析金融风险*

</div>

---

## 📌 这是什么？

FinRisk MultiAgent 是一个基于**多智能体协作（Multi-Agent Collaboration）**架构的金融文档风险分析平台。

上传一份金融文档（年报、招股书、债券募集说明书等），提出风险相关问题，系统自动**扫描文档、生成搜索策略、调度 4 个专业 Agent 并行工作**，最后综合成结构化风险评估报告。

### 核心特色

- 🧠 **Multi-Agent 协作**: 4 个专业化 Agent 各司其职，不是单模型"一问一答"
- 📎 **RAG 文档检索**: FAISS 向量库 + DashScope Embedding，精准检索 PDF 相关段落，支持全量表格提取
- 🌐 **联网搜索**: 每个 Agent 基于文档简报生成专属搜索词，通过 DashScope 内置搜索（enable_search）并行获取最新数据
- 🎯 **自适应分析深度**: 自动判断问题类型——查数据直接给答案、专项分析聚焦回答、全面评估走完整流程
- 🔍 **Devil's Advocate 机制**: 内置"魔鬼代言人"Agent，专门挑战其他 Agent 的结论（全面评估模式）
- 🔑 **用户自有 API Key**: 不内置任何密钥，用户完全掌控自己的 API 使用
- ⚡ **流式输出**: 最终报告逐字实时生成

---

## 🏗️ 系统架构

```
                         👤 用户上传文档 + 提问
                                │
                                ▼
┌──────────────────────────────────────────────────┐
│              📋 第零轮：文档简报                    │
│        扫描文档 → 提取公司名/代码/行业/报告期        │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│           🌐 联网搜索（基于简报生成搜索词）          │
│    各 Agent 用自己的搜索词 + enable_search 并行搜索  │
└──────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  📊 数据   │ │  ⚠️ 风险  │ │  📋 合规  │ │  🔍 深度  │
│ 提取Agent │ │ 评估Agent │ │ 审查Agent │ │ 质疑Agent │
│          │ │          │ │          │ │          │
│ 提取关键  │ │ 相关维度  │ │ 对照法规  │ │ 挑战结论  │
│ 财务指标  │ │ 聚焦评分  │ │ 聚焦审查  │ │ 找出盲点  │
└──────┬───┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
       │          │          │          │
       └──────────┴──────────┴──────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│            🎯 协调 Agent 综合裁决                  │
│    根据问题复杂度自适应详略——简单问题直接答，        │
│    专项分析聚焦答，全面评估完整报告                  │
└──────────────────────────────────────────────────┘
```

### 自适应分析深度

| 问题类型 | 示例 | 流程 |
|----------|------|------|
| **factual** 数据查询 | "2026年一季度利润是多少？" | RAG检索 + enable_search → 一句话回答 |
| **analytical** 专项分析 | "偿债能力怎么样？" | 3 Agent并行 + 直接综合 |
| **comprehensive** 全面评估 | "做个全面风险评估" | 4 Agent + 魔鬼代言人 + 反驳 + 完整报告 |

### 四个专业 Agent

每个 Agent 用专业能力**回答问题**，不是机械填充检查清单：

| Agent | 角色 | 工作方式 |
|-------|------|----------|
| 📊 **数据提取** | CFA+CPA 审计专家 | 先理解问题 → 只提取相关数据 → 适可而止 |
| ⚠️ **风险评估** | 18年风控经验 | 用户只问偿债就只分析偿债，不顺便打分其他维度 |
| 📋 **合规审查** | 前证监会预审员 | 聚焦用户关心的合规领域，不逐条审查所有框架 |
| 🔍 **深度质疑** | 桥水 Red Team | 只质疑与用户问题相关的结论，不是抬杠机器 |

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- [DashScope API Key](https://dashscope.console.aliyun.com/apiKey)（通义千问）

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
python -m streamlit run app.py
```

### 4. 使用

1. 浏览器打开 `http://localhost:8501`
2. 左侧边栏输入 DashScope API Key
3. 上传金融文档（PDF / TXT / MD）
4. 勾选 🌐 联网搜索
5. 提问——系统自动判断问题类型，匹配合适的分析深度

---

## 📂 项目结构

```
finrisk-multiagent/
├── app.py                      # Streamlit 主界面（中英双语）
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── README.md
│
├── src/
│   ├── orchestrator.py         # 🎯 中央调度器：文档简报→搜索→Agent编排→综合
│   ├── agents/
│   │   ├── base.py             # Agent 基类：RAG检索 + 简报注入 + LLM调用
│   │   ├── data_extractor.py   # 📊 数据提取 Agent
│   │   ├── risk_assessor.py    # ⚠️ 风险评估 Agent
│   │   ├── compliance_checker.py # 📋 合规审查 Agent
│   │   └── devils_advocate.py  # 🔍 深度质疑 Agent
│   ├── rag/
│   │   └── engine.py           # RAG 模块：PDF加载·文本分块·表格提取·Embedding·FAISS
│   ├── search/
│   │   └── web_search.py       # 联网搜索：DashScope enable_search + DDGS fallback
│   └── llm/
│       └── client.py           # LLM 客户端：DashScope + OpenAI兼容双模式·流式输出
│
├── prompts/                    # 📝 Prompt 模板（Markdown 格式）
│   ├── zh/                     # 中文 prompts
│   │   ├── data_extractor.md
│   │   ├── risk_assessor.md
│   │   ├── compliance_checker.md
│   │   ├── devils_advocate.md
│   │   └── orchestrator.md
│   └── en/                     # English prompts
│
└── examples/                   # 示例文件
    └── sample_report.md        # 示例分析报告
```

---

## 🔧 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Streamlit | 纯 Python Web UI，零前端代码 |
| **LLM** | DashScope (Qwen) / OpenAI 兼容 | qwen-turbo/plus/max |
| **RAG** | FAISS + DashScope Embedding | 本地向量存储，全量 PDF 表格提取 |
| **联网搜索** | DashScope enable_search | LLM 原生搜索，比传统搜索引擎 snippet 更精准 |
| **文档处理** | pdfplumber + LangChain | PDF 文本提取 + 递归语义分块 + 表格结构化 |
| **Agent 框架** | 自研轻量框架 | 不依赖 LangChain Agent / AutoGen，完全可控 |
| **并行调度** | concurrent.futures | 标准库，Agent 并行 + 搜索并行 |

---

## 📊 示例输出

<details>
<summary>点击展开示例</summary>

**Factual 查询**

> 问：2026年一季度利润是多少？
>
> 2026年一季度归母净利润为 **57.35 亿元**（网络来源：东方财富网、中证智能财讯）

**Analytical 专项分析**

> 问：偿债能力怎么样？
>
> 中际旭创 2025 年半年度偿债能力**整体稳健，短期流动性压力可控**。
>
> - 有息负债总额 ≈ 22.85 亿元，仅占净资产约 9.4%，杠杆极低
> - 货币资金约 112-123 亿元，覆盖短期借款 119.46 亿元（来源：文档片段）

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

---

## ⚠️ 免责声明

- 本系统由 AI 驱动，分析结果**仅供参考**，不构成投资建议、法律意见或任何形式的专业建议
- 分析基于用户上传的文档和联网搜索结果，可能存在时效性差异
- 用户应在做出任何决策前咨询持牌专业人士

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件。
