"""
FinRisk MultiAgent — 多 Agent 协作金融风险智能分析系统

启动方式:
    streamlit run app.py

然后在浏览器中打开 http://localhost:8501
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit 每次 rerun 都重新执行 app.py，但 Python 的 sys.modules 会缓存
# 已导入的模块。强制清除项目源码模块的缓存，确保每次运行都是最新代码。
_PROJECT_MODULES = [m for m in sys.modules if m.startswith("src.")]
for _m in _PROJECT_MODULES:
    del sys.modules[_m]

import streamlit as st

# ============================================================================
# 国际化文本
# ============================================================================

I18N = {
    "zh": {
        "title": "🏦 FinRisk MultiAgent",
        "subtitle": "多 Agent 协作金融风险智能分析系统 · 上传文档 → 提问 → 四个 AI 分析师并行工作 → 综合报告",
        "api_settings": "🔑 API 设置",
        "api_provider": "API 提供商",
        "api_provider_dashscope": "DashScope (通义千问)",
        "api_provider_custom": "自定义 OpenAI 兼容接口",
        "api_key": "DashScope API Key",
        "api_key_custom": "API Key",
        "api_key_help": "在 https://dashscope.console.aliyun.com/apiKey 获取。你的 Key 仅用于本次会话，不会被存储。",
        "api_base": "API Base URL",
        "model": "模型",
        "model_help": "qwen-plus: 性价比最高 | qwen-max: 最强推理 | qwen-turbo: 最快",
        "model_params": "🎛️ 模型参数",
        "temperature": "Temperature",
        "temperature_help": "越低越确定，越高越有创造性。金融分析建议 0.2-0.4",
        "max_tokens": "Max Tokens",
        "max_tokens_help": "单次生成的最大 token 数",
        "language": "🌐 界面语言",
        "about_title": "📖 关于本系统",
        "about_text": "FinRisk MultiAgent 是一个基于多 Agent 协作的金融风险智能分析平台。",
        "upload_label": "📄 上传金融文档",
        "upload_help": "支持：年报、招股说明书、债券募集说明书、风险披露报告等 PDF 或文本文件",
        "file_name": "文件名",
        "file_size": "文件大小",
        "upload_placeholder": "👈 上传一份金融文档开始分析\n\n支持 PDF / TXT / MD 格式",
        "api_key_warning": "⚠️ 请先在侧边栏输入 API Key",
        "processing": "📂 正在处理文档...",
        "extracting": "📖 提取文本...",
        "vectorizing": "🧮 向量化（Embedding + FAISS 索引构建）...",
        "processing_done": "✅ 文档处理完成！{count} 个文本块 + {tables} 个表格已索引",
        "processing_fail": "❌ 处理失败: {error}",
        "file_ready": "📄 **{name}** 已就绪（{count} 个文本块）",
        "reset": "🔄 重新上传",
        "chat_placeholder": "输入你的风险分析问题...",
        "quick_questions": [
            "对这份文件做个全面的风险评估",
            "这家公司最大的风险是什么？",
            "有哪些合规风险需要关注？",
            "偿债能力如何？有流动性危机吗？",
            "这份文件有没有刻意回避的问题？",
        ],
        "analyzing": "🤖 多 Agent 协作分析中...",
        "download": "📥 下载完整报告 (Markdown)",
        "detail_title": "🔍 查看各 Agent 的详细分析过程",
        "detail_tabs": ["📊 数据提取", "⚠️ 风险评估", "📋 合规审查", "🔍 深度质疑", "📋 执行日志"],
        "welcome_title": "🧠 多 Agent 协作分析引擎",
        "welcome_subtitle": "不是一个 AI 在回答问题——而是一个由四个专业化 Agent 组成的分析团队在并行工作，最后交由协调 Agent 综合裁决。",
        "welcome_agent_data_name": "数据提取 Agent",
        "welcome_agent_data_desc": "从 PDF 中自动提取财务数据，支持全量表格识别和结构化输出，每项数据标注来源页码。",
        "welcome_agent_risk_name": "风险评估 Agent",
        "welcome_agent_risk_desc": "四维风险框架逐项打分：市场、信用、流动性、操作与治理。每项判断强制引用原文证据。",
        "welcome_agent_comply_name": "合规审查 Agent",
        "welcome_agent_comply_desc": "以监管者视角逐条对照法规框架，区分\"违规\"与\"需关注\"，标注具体条款依据。",
        "welcome_agent_devil_name": "深度质疑 Agent",
        "welcome_agent_devil_desc": "魔鬼代言人机制——专门挑战前三方结论，发现被忽视的盲点和逻辑矛盾。",
        "welcome_pipeline_title": "⚡ 分析流程",
        "welcome_pipeline_1": "📄 上传文档",
        "welcome_pipeline_2": "📂 处理文档",
        "welcome_pipeline_3": "🔍 理解问题",
        "welcome_pipeline_4": "🤖 Agent 并行分析",
        "welcome_pipeline_5": "⚖️ 质疑与裁决",
        "welcome_pipeline_6": "📊 流式生成报告",
        "welcome_tech_title": "🛠️ 技术架构",
        "welcome_tech_items": "RAG 检索增强 · FAISS 向量索引 · 全量 PDF 表格提取 · LLM 查询改写 · DashScope Embedding · Qwen API · 流式输出 · 中英双语 · 联网搜索",
        "disclaimer": "⚠️ 免责声明：本系统由 AI 驱动，分析结果仅供参考，不构成投资建议或法律意见。使用前请确保已阅读并理解相关风险。",
        "agent_data": "📊 数据提取 Agent",
        "agent_risk": "⚠️ 风险评估 Agent",
        "agent_compliance": "📋 合规审查 Agent",
        "agent_devil": "🔍 深度质疑 Agent",
        "status_running": "⏳ 分析中...",
        "status_done": "✅ 完成",
        "status_error": "❌ 出错",
        "status_waiting": "⏸️ 等待中",
        "round1_msg": "启动第一轮分析：数据提取、风险评估、合规审查并行执行...",
        "round2_msg": "正在审阅前三方分析结果，寻找盲点和矛盾...",
        "round3_msg": "正在综合所有分析结果，生成最终报告...",
        "analysis_done": "分析完成。",
    },
    "en": {
        "title": "🏦 FinRisk MultiAgent",
        "subtitle": "Multi-Agent Financial Risk Analysis · Upload → Ask → 4 AI Analysts Work in Parallel → Synthesized Report",
        "api_settings": "🔑 API Settings",
        "api_provider": "API Provider",
        "api_provider_dashscope": "DashScope (Qwen)",
        "api_provider_custom": "Custom OpenAI-Compatible",
        "api_key": "DashScope API Key",
        "api_key_custom": "API Key",
        "api_key_help": "Get your key at https://dashscope.console.aliyun.com/apiKey. Your key is only used for this session and will not be stored.",
        "api_base": "API Base URL",
        "model": "Model",
        "model_help": "qwen-plus: Best value | qwen-max: Strongest reasoning | qwen-turbo: Fastest",
        "model_params": "🎛️ Model Parameters",
        "temperature": "Temperature",
        "temperature_help": "Lower = more deterministic. Financial analysis: 0.2-0.4 recommended.",
        "max_tokens": "Max Tokens",
        "max_tokens_help": "Maximum tokens per generation",
        "language": "🌐 Language",
        "about_title": "📖 About",
        "about_text": "FinRisk MultiAgent is a multi-agent collaborative financial risk intelligence platform.",
        "upload_label": "📄 Upload Financial Document",
        "upload_help": "Supports: Annual reports, prospectuses, bond offering circulars, risk disclosures (PDF/TXT/MD)",
        "file_name": "File Name",
        "file_size": "File Size",
        "upload_placeholder": "👈 Upload a financial document to start\n\nSupports PDF / TXT / MD formats",
        "api_key_warning": "⚠️ Please enter your API Key in the sidebar first",
        "processing": "📂 Processing document...",
        "extracting": "📖 Extracting text...",
        "vectorizing": "🧮 Building vector index (Embedding + FAISS)...",
        "processing_done": "✅ Document processed! {count} chunks + {tables} tables indexed",
        "processing_fail": "❌ Processing failed: {error}",
        "file_ready": "📄 **{name}** ready ({count} chunks)",
        "reset": "🔄 Upload New",
        "chat_placeholder": "Ask a risk analysis question...",
        "quick_questions": [
            "Give me a comprehensive risk assessment of this document",
            "What is the biggest risk for this company?",
            "Are there any compliance concerns I should know about?",
            "How is the debt servicing capacity? Any liquidity crisis?",
            "Is there anything this document is deliberately avoiding?",
        ],
        "analyzing": "🤖 Multi-Agent analysis in progress...",
        "download": "📥 Download Full Report (Markdown)",
        "detail_title": "🔍 View Detailed Agent Analysis",
        "detail_tabs": ["📊 Data Extraction", "⚠️ Risk Assessment", "📋 Compliance Check", "🔍 Devil's Advocate", "📋 Execution Log"],
        "welcome_title": "🧠 Multi-Agent Analysis Engine",
        "welcome_subtitle": "Not one AI answering your question — a team of four specialized agents working in parallel, with a coordinator adjudicating and synthesizing the final report.",
        "welcome_agent_data_name": "Data Extraction Agent",
        "welcome_agent_data_desc": "Extracts financial data from PDFs with full table recognition. Every number cited with source page.",
        "welcome_agent_risk_name": "Risk Assessment Agent",
        "welcome_agent_risk_desc": "Four-dimension risk scoring: Market, Credit, Liquidity, Operational & Governance. Evidence-grounded.",
        "welcome_agent_comply_name": "Compliance Checker Agent",
        "welcome_agent_comply_desc": "Reviews against regulatory frameworks from a regulator's perspective. Cites specific provisions.",
        "welcome_agent_devil_name": "Devil's Advocate Agent",
        "welcome_agent_devil_desc": "Systematically challenges other agents' conclusions. Finds blind spots and logical contradictions.",
        "welcome_pipeline_title": "⚡ Analysis Pipeline",
        "welcome_pipeline_1": "📄 Upload",
        "welcome_pipeline_2": "📂 Process",
        "welcome_pipeline_3": "🔍 Understand",
        "welcome_pipeline_4": "🤖 Analyze",
        "welcome_pipeline_5": "⚖️ Challenge",
        "welcome_pipeline_6": "📊 Report",
        "welcome_tech_title": "🛠️ Tech Stack",
        "welcome_tech_items": "RAG · FAISS · Full Table Extraction · LLM Query Rewriting · DashScope Embedding · Qwen API · Streaming Output · Bilingual · Web Search",
        "disclaimer": "⚠️ Disclaimer: This system is AI-powered. Analysis results are for reference only and do NOT constitute investment advice or legal opinion.",
        "agent_data": "📊 Data Extraction Agent",
        "agent_risk": "⚠️ Risk Assessment Agent",
        "agent_compliance": "📋 Compliance Checker Agent",
        "agent_devil": "🔍 Devil's Advocate Agent",
        "status_running": "⏳ Analyzing...",
        "status_done": "✅ Done",
        "status_error": "❌ Error",
        "status_waiting": "⏸️ Waiting",
        "round1_msg": "Starting Round 1: Data Extraction, Risk Assessment, Compliance Check running in parallel...",
        "round2_msg": "Reviewing outputs from all three agents, searching for blind spots and contradictions...",
        "round3_msg": "Synthesizing all analysis results into final report...",
        "analysis_done": "Analysis complete.",
    },
}


def t(key: str, **kwargs) -> str:
    """获取国际化文本，支持格式化参数。"""
    lang = st.session_state.get("language", "zh")
    text = I18N.get(lang, I18N["zh"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def _generate_doc_questions(vector_store, api_key: str, language: str,
                            filename: str) -> list[str]:
    """扫描文档，生成针对该文档的个性化推荐提问。

    取文档中最有信息量的几个文本块，让 LLM 识别：
      - 文档类型 / 主体名称 / 行业
      - 文档中最突出的风险信号
      - 3-5 个值得深入提问的方向
    """
    from src.llm.client import LLMClient, LLMConfig

    # 取前 6 个 chunk 作为文档概览上下文
    sample_chunks = vector_store.search(
        "风险 负债 收入 利润 资产 合规 担保 关联交易 现金流 行业 竞争 经营",
        top_k=6, api_key=api_key
    )
    context = "\n\n---\n\n".join(
        f"[片段 {i+1}] {r.chunk.text[:800]}" for i, r in enumerate(sample_chunks)
    )

    scan_prompt_zh = f"""你是一个金融文档分析专家。请快速浏览以下文档片段，然后完成两项任务。

## 文档信息
文件名: {filename}

## 文档片段
{context[:4000]}

## 任务

### 任务1: 文档速览（用1-2句话）
- 这是什么类型的文档？（年报/招股书/募集说明书/研报/其他？）
- 文档主体的名称和行业是什么？
- 最突出的1-2个风险信号是什么？

### 任务2: 生成3-5个针对本文档的推荐提问
基于文档的实际内容（不要编造文档中没有的信息），生成3-5个值得用户深入分析的问题。
- 问题要具体，关联文档中真实出现的数据、事件或风险
- 覆盖面要广：财务健康、风险敞口、合规、行业前景等不同角度
- 每个问题不超过25个字
- 直接输出问题列表，每行一个，以"- "开头

请只输出任务2的问题列表，格式如下：
- 问题1
- 问题2
- 问题3
- 问题4
- 问题5"""

    scan_prompt_en = f"""You are a financial document analysis expert. Quickly scan the following document excerpts and complete two tasks.

## Document Info
Filename: {filename}

## Document Excerpts
{context[:4000]}

## Tasks

### Task 1: Document Overview (1-2 sentences)
- What type of document is this? (Annual report / Prospectus / Offering circular / Research report / Other?)
- What is the entity name and industry?
- What are the 1-2 most prominent risk signals?

### Task 2: Generate 3-5 Recommended Questions
Based on the document's actual content (do NOT fabricate information), generate 3-5 questions worth investigating deeper.
- Questions should be specific, referencing real data, events, or risks found in the document
- Cover different angles: financial health, risk exposure, compliance, industry outlook, etc.
- Each question under 15 words
- Output as a list, one per line, starting with "- "

Only output the question list from Task 2, in this format:
- Question 1
- Question 2
- Question 3
- Question 4
- Question 5"""

    prompt = scan_prompt_zh if language == "zh" else scan_prompt_en

    config = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.3, max_tokens=512)
    client = LLMClient(config)
    resp = client.chat([{"role": "user", "content": prompt}])

    # 解析问题列表
    questions = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("- "):
            q = line[2:].strip()
            if q and len(q) > 3:
                questions.append(q)
        elif line and len(line) > 3 and "?" in line:
            # 兼容不规范的格式（没有 - 前缀）
            questions.append(line.strip())

    # 至少保证有 3 个问题，不够就用默认的补齐
    if len(questions) < 3:
        fallback = I18N[language]["quick_questions"]
        questions += fallback[len(questions):]

    return questions[:5]


# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="FinRisk MultiAgent — 金融风险智能分析",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 样式
# ============================================================================

st.markdown("""
<style>
    /* ── 全局 ── */
    .main-title {
        font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem;
        text-align: center; color: #1a3c5e;
    }
    .subtitle {
        font-size: 0.95rem; color: #888; margin-bottom: 2rem;
        text-align: center;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #2d6a9f;
        text-align: center; margin: 2rem 0 1rem 0;
    }

    /* ── Pipeline 流程动画 ── */
    .pipeline-container {
        display: flex; align-items: center; justify-content: center;
        gap: 0; padding: 1rem 0 1.5rem 0; flex-wrap: nowrap;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
        color: #fff; border-radius: 10px; padding: 14px 22px;
        text-align: center; min-width: 90px; font-weight: 600;
        font-size: 0.9rem; box-shadow: 0 3px 10px rgba(26,60,94,0.15);
        animation: fadeInUp 0.5s ease-out both;
        flex-shrink: 0;
    }
    .pipeline-step:nth-child(1)  { animation-delay: 0.00s; }
    .pipeline-step:nth-child(3)  { animation-delay: 0.10s; }
    .pipeline-step:nth-child(5)  { animation-delay: 0.20s; }
    .pipeline-step:nth-child(7)  { animation-delay: 0.30s; }
    .pipeline-step:nth-child(9)  { animation-delay: 0.40s; }
    .pipeline-arrow {
        font-size: 1.2rem; color: #bbb; margin: 0 6px;
        flex-shrink: 0; user-select: none;
        animation: arrowPulse 2s infinite;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes arrowPulse {
        0%, 100% { opacity: 0.25; }
        50%      { opacity: 0.70; }
    }

    /* ── Agent 卡片 ── */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px; margin: 1rem 0 1.5rem 0;
    }
    @media (max-width: 960px)  { .agent-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 520px)  { .agent-grid { grid-template-columns: 1fr; } }
    .agent-feature-card {
        border: 1px solid #e8eaed; border-radius: 12px;
        padding: 24px 18px 20px 18px; text-align: center;
        background: #fafbfc;
        transition: transform 0.2s, box-shadow 0.2s;
        display: flex; flex-direction: column;
        align-items: center; justify-content: flex-start;
        height: 100%; min-height: 200px;
    }
    .agent-feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .agent-feature-icon {
        font-size: 2rem; margin-bottom: 10px; line-height: 1;
    }
    .agent-feature-name {
        font-weight: 700; font-size: 0.95rem; color: #1a3c5e;
        margin-bottom: 8px; line-height: 1.3;
    }
    .agent-feature-desc {
        font-size: 0.82rem; color: #666; line-height: 1.55;
        flex-grow: 1;
    }

    /* ── 技术栈标签 ── */
    .tech-tag {
        display: inline-block; background: #e8f0fe; color: #1a73e8;
        border-radius: 4px; padding: 3px 10px; font-size: 0.8rem;
        margin: 3px; font-weight: 500; line-height: 1.6;
    }
    .tech-tags-wrap {
        text-align: center; margin-bottom: 2rem; line-height: 2;
    }

    /* ── 文件信息 ── */
    .file-ready-box {
        background: #f0f7f0; border: 1px solid #c8e6c9;
        border-radius: 8px; padding: 12px 16px; margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    /* ── 免责声明 ── */
    .disclaimer  { color: #aaa; font-size: 0.75rem; margin-top: 2rem; text-align: center; }

    /* ── 状态指示器 ── */
    .agent-running { border-left: 4px solid #1a73e8; animation: pulse 1.5s infinite; }
    .agent-done    { border-left: 4px solid #0f9d58; }
    .agent-error   { border-left: 4px solid #d93025; }
    @keyframes pulse {
        0%   { opacity: 1.0; }
        50%  { opacity: 0.6; }
        100% { opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State 初始化
# ============================================================================

# 分析状态初始化（重置时清除这些）
for key, default in [
    ("vector_store", None),
    ("file_processed", False),
    ("processing_file", False),
    ("chat_history", []),
    ("last_analysis", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 语言偏好（独立于分析状态，不随重置清除）
if "language" not in st.session_state:
    st.session_state.language = "zh"
if "_qkey" not in st.session_state:
    st.session_state._qkey = 0

# ============================================================================
# 侧边栏 — 配置
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("## ⚙️ " + ("系统配置" if st.session_state.language == "zh" else "Settings"))

    # -- 语言选择 --
    lang = st.selectbox(
        t("language"),
        ["中文", "English"],
        index=0 if st.session_state.language == "zh" else 1,
    )
    new_lang = "zh" if lang == "中文" else "en"
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()

    st.divider()

    # -- API 配置 --
    st.markdown("### " + t("api_settings"))

    api_provider = st.selectbox(
        t("api_provider"),
        [t("api_provider_dashscope"), t("api_provider_custom")],
    )

    is_dashscope = api_provider == t("api_provider_dashscope")

    if is_dashscope:
        api_key = st.text_input(
            t("api_key"),
            type="password",
            value=os.getenv("DASHSCOPE_API_KEY", ""),
            help=t("api_key_help"),
        )
        api_base = ""
        model = st.selectbox(
            t("model"),
            ["qwen-plus", "qwen-max", "qwen-turbo"],
            index=0,
            help=t("model_help"),
        )
    else:
        api_key = st.text_input(
            t("api_key_custom"),
            type="password",
        )
        api_base = st.text_input(t("api_base"), value="https://api.openai.com/v1")
        model = st.text_input(t("model"), value="gpt-4o")

    st.divider()

    # -- 联网搜索 --
    web_search_enabled = st.checkbox(
        "🌐 联网搜索" if st.session_state.language == "zh" else "🌐 Web Search",
        value=True,
        help=("默认开启。Agent 会在文档信息不足时自动联网搜索最新数据。"
              if st.session_state.language == "zh"
              else "On by default. Agents search the web when document info is insufficient."),
    )

    st.divider()

    # -- 模型参数 --
    st.markdown("### " + t("model_params"))
    temperature = st.slider(t("temperature"), 0.0, 1.0, 0.3, 0.05, help=t("temperature_help"))
    max_tokens = st.slider(t("max_tokens"), 1024, 8192, 4096, 256, help=t("max_tokens_help"))

    st.divider()

    # -- 关于 --
    st.markdown("### " + t("about_title"))
    st.markdown(t("about_text"))
    if st.session_state.language == "zh":
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.7; color:#555">
        <b>四个专业 Agent</b><br>
        📊 数据提取 &nbsp; ⚠️ 风险评估 &nbsp; 📋 合规审查 &nbsp; 🔍 深度质疑<br><br>
        <b>工作方式</b><br>
        上传文档 → 提问 → Agent 并行分析 → 综合报告<br><br>
        <a href="https://github.com/leokiy/finrisk-multiagent">GitHub</a> · <a href="https://github.com/leokiy/finrisk-multiagent/issues">Issues</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.7; color:#555">
        <b>Four Specialist Agents</b><br>
        📊 Data Extraction &nbsp; ⚠️ Risk Assessment &nbsp; 📋 Compliance &nbsp; 🔍 Devil's Advocate<br><br>
        <b>How it works</b><br>
        Upload → Ask → Agents analyze → Report<br><br>
        <a href="https://github.com/leokiy/finrisk-multiagent">GitHub</a> · <a href="https://github.com/leokiy/finrisk-multiagent/issues">Issues</a>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 主页面
# ============================================================================

st.markdown(f'<div style="font-size:1.8rem;font-weight:700;color:#1a3c5e;margin-bottom:0.3rem">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:0.9rem;color:#888;margin-bottom:1.2rem">{t("subtitle")}</p>', unsafe_allow_html=True)

# ============================================================================
# 文件上传区域
# ============================================================================

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        t("upload_label"),
        type=["pdf", "txt", "md"],
        help=t("upload_help"),
    )

with col_info:
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric(t("file_name"), uploaded_file.name)
        st.metric(t("file_size"), f"{file_size_mb:.1f} MB")
        # 保存文件名到 session state，防止 rerun 时丢失
        st.session_state._uploaded_filename = uploaded_file.name
    elif st.session_state.file_processed:
        # rerun 后 uploaded_file 可能为 None，用缓存的文件名
        cached_name = st.session_state.get("_uploaded_filename", "unknown")
        st.metric(t("file_name"), cached_name)
    else:
        st.info(t("upload_placeholder"))

# ============================================================================
# 处理上传的文件
# ============================================================================

if uploaded_file and not st.session_state.file_processed:
    # 文件大小限制 (20MB)
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > 20:
        st.error("文件过大（{:.1f}MB），请上传小于20MB的文件。"
                 if st.session_state.language == "zh"
                 else "File too large ({:.1f}MB). Please upload files under 20MB.".format(file_size_mb))
        st.stop()

    # 校验 API Key
    if not api_key:
        st.error(t("api_key_warning"))
        st.stop()

    st.session_state.processing_file = True
    # 缓存文件信息
    st.session_state._file_bytes = uploaded_file.getvalue()
    st.session_state._uploaded_filename = uploaded_file.name

    with st.status("Processing document...", expanded=True) as status:
        try:
            # 保存到临时文件
            suffix = Path(st.session_state._uploaded_filename).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(st.session_state._file_bytes)
                tmp_path = tmp.name

            # RAG 构建
            st.write("Extracting text...")
            from src.rag.engine import build_rag_from_file

            st.write("Building vector index...")
            vector_store = build_rag_from_file(
                tmp_path, api_key=api_key
            )

            st.session_state.vector_store = vector_store
            st.session_state.file_processed = True
            st.session_state.processing_file = False
            st.session_state.chat_history = []
            st.session_state.doc_questions = None
            st.session_state.doc_type = ""  # 触发文档类型识别

            # 识别文档类型（取前3页文本块快速判断，失败不影响主流程）
            try:
                from src.llm.client import LLMConfig, LLMClient
                front_chunks = vector_store.search(
                    "年报 半年报 季度报告 招股说明书", top_k=5, api_key=api_key
                )
                front_text = " ".join(r.chunk.text[:200] for r in front_chunks)[:1500]
                if front_text.strip():
                    doc_cfg = LLMConfig(api_key=api_key, model="qwen-turbo",
                                       temperature=0.1, max_tokens=50)
                    doc_client = LLMClient(doc_cfg)
                    lang_hint = st.session_state.get("language", "zh")
                    if lang_hint == "zh":
                        doc_prompt = f"判断文档类型（年报/半年报/季报/招股说明书/债券募集说明书/其他）。只输出类型名称：\n{front_text}"
                    else:
                        doc_prompt = f"Identify document type (Annual Report/Semi-annual/Quarterly/Prospectus/Bond/Other). Output type only:\n{front_text}"
                    doc_resp = doc_client.chat([{"role": "user", "content": doc_prompt}])
                    st.session_state.doc_type = doc_resp.strip()
            except Exception as e:
                st.session_state.doc_type = ""
                print(f"[DocType] skipped: {e}")

            # 清理临时文件
            os.unlink(tmp_path)

            status.update(
                label=f"Done: {vector_store.chunk_count} chunks, {vector_store.table_count} tables",
                state="complete",
            )

        except Exception as exc:
            # 避免 emoji 在错误消息中触发 latin-1 编码问题
            import traceback
            traceback.print_exc()
            try:
                status.update(label=f"[ERROR] {str(exc)[:200]}", state="error")
            except Exception:
                status.update(label="Processing failed. Check console for details.", state="error")
            st.session_state.processing_file = False
            st.stop()

# ============================================================================
# 重置按钮
# ============================================================================

if st.session_state.file_processed:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.success(t("file_ready", name=st.session_state.get("_uploaded_filename", ""),
                     count=st.session_state.vector_store.chunk_count))
    with col_b:
        if st.button(t("reset")):
            # 只清分析状态，保留语言和 API Key 等用户偏好
            st.session_state.vector_store = None
            st.session_state.file_processed = False
            st.session_state.chat_history = []
            st.session_state.last_analysis = None
            st.session_state.pop("_uploaded_filename", None)
            st.session_state.pop("_file_bytes", None)
            st.session_state.pop("doc_questions", None)
            st.rerun()

# ============================================================================
# 对话区域
# ============================================================================

if st.session_state.file_processed:

    # ── 初始文档扫描：自动生成个性化推荐提问（仅首次）──
    if (st.session_state.get("doc_questions") is None
            and st.session_state.vector_store is not None
            and api_key):
        with st.spinner("🔍 " + ("正在分析文档，生成推荐提问..." if st.session_state.language == "zh" else "Analyzing document to generate tailored questions...")):
            try:
                st.session_state.doc_questions = _generate_doc_questions(
                    st.session_state.vector_store, api_key,
                    st.session_state.language,
                    st.session_state.get("_uploaded_filename", "document"),
                )
            except Exception:
                st.session_state.doc_questions = t("quick_questions")

    quick_questions = st.session_state.get("doc_questions") or t("quick_questions")

    # 输入区
    user_query = st.chat_input(t("chat_placeholder"))

    if "pending_query" in st.session_state:
        user_query = st.session_state.pop("pending_query")

    # ── 聊天记录（先渲染，在上面）──
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 推荐提问（后渲染，在聊天记录下方、输入框上方）──
    st.markdown("**" + ("📋 推荐追问:" if st.session_state.language == "zh" else "📋 Follow-up Questions:") + "**")
    qkey = st.session_state.get("_qkey", 0)
    cols = st.columns(len(quick_questions))
    for i, (col, q) in enumerate(zip(cols, quick_questions)):
        with col:
            if st.button(q, key=f"quick_{qkey}_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.session_state._qkey = qkey + 1

    # 处理用户提问
    if user_query:
        # 检查 API Key
        if not api_key:
            st.error(t("api_key_warning"))
            st.stop()

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # ── 运行多 Agent 分析 ──
        from src.llm.client import LLMClient, LLMConfig
        from src.orchestrator import Orchestrator

        # 构建 LLM 客户端
        config = LLMConfig(
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_client = LLMClient(config)
        orchestrator = Orchestrator(llm_client, language=st.session_state.language)

        # ── 显示最终报告（流式输出）──
        with st.chat_message("assistant"):
            report_placeholder = st.empty()
            streamed_text = []

            def on_token(token: str):
                streamed_text.append(token)
                report_placeholder.markdown("".join(streamed_text))

            # 执行分析（synthesis 阶段流式推送）
            result = orchestrator.run(
                user_query,
                st.session_state.vector_store,
                api_key=api_key,
                on_synthesis_token=on_token,
                web_search_enabled=web_search_enabled,
                doc_type=st.session_state.get("doc_type", ""),
            )

            # 诊断：直接在报告上方显示搜索状态
            search_logs = [l for l in result.get("execution_log", [])
                          if "搜索" in l.get("content", "") or "search" in l.get("content", "").lower()
                          or "web_search" in l.get("content", "") or "results" in l.get("content", "")]
            if search_logs:
                diag = "\n\n".join(f"> {l['content'][:150]}" for l in search_logs)
            elif web_search_enabled:
                diag = "> ⚠️ 联网搜索已开启但未产生任何搜索日志"
            else:
                diag = "> 🔒 联网搜索未开启"

            report = diag + "\n\n---\n\n" + result.get("final_report", "")
            if not result.get("final_report"):
                fail_msg = ("分析失败，请检查 API Key 和网络连接。"
                           if st.session_state.language == "zh"
                           else "Analysis failed. Please check your API Key and network connection.")
                report = fail_msg
                report_placeholder.markdown(report)

            # 下载按钮
            st.download_button(
                label=t("download"),
                data=report,
                file_name=f"finrisk_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

        st.session_state.chat_history.append({"role": "assistant", "content": report})
        st.session_state.last_analysis = result

        # 用 Orchestrator 生成的追问实时刷新推荐问题
        followups = result.get("followup_questions", [])
        if followups and len(followups) >= 2:
            st.session_state.doc_questions = followups

        # 显示各 Agent 的详细输出（可折叠）
        with st.expander(t("detail_title"), expanded=False):
            tab_labels = t("detail_tabs")
            tabs = st.tabs(tab_labels)
            sections = [
                ("data_extraction", result.get("data_extraction", "")),
                ("risk_assessment", result.get("risk_assessment", "")),
                ("compliance_check", result.get("compliance_check", "")),
                ("devils_advocate", result.get("devils_advocate", "")),
            ]
            for tab, (_, content) in zip(tabs[:4], sections):
                with tab:
                    st.markdown(content)
            with tabs[4]:
                for log in result.get("execution_log", []):
                    status_icon = {"running": "⏳", "done": "✅", "error": "❌"}
                    icon = status_icon.get(log["status"], "•")
                    content = log.get("content", "")
                    if content:
                        st.markdown(f"{icon} **{log['agent']}**: {content[:200]}")
                    else:
                        st.markdown(f"{icon} **{log['agent']}**: {log['status']}")

    # ── 兜底刷新推荐问题（仅值变化时更新，避免无限rerun）──
    last = st.session_state.get("last_analysis")
    if last and last.get("followup_questions"):
        new_qs = last["followup_questions"]
        if new_qs != st.session_state.get("doc_questions"):
            st.session_state.doc_questions = new_qs

else:
    # 未上传文件时显示欢迎信息
    st.divider()

    # ── 标题 ──
    st.markdown(f'<h2 class="section-title" style="font-size:1.6rem;color:#1a3c5e">{t("welcome_title")}</h2>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t("welcome_subtitle")}</p>',
                unsafe_allow_html=True)

    # ── 分析流程动画 ──
    st.markdown(f'<div class="section-title">{t("welcome_pipeline_title")}</div>',
                unsafe_allow_html=True)
    pipeline_steps = [
        t("welcome_pipeline_1"), t("welcome_pipeline_2"), t("welcome_pipeline_3"),
        t("welcome_pipeline_4"), t("welcome_pipeline_5"),
    ]
    pipeline_html = '<div class="pipeline-container">'
    for i, step in enumerate(pipeline_steps):
        pipeline_html += f'<div class="pipeline-step">{step}</div>'
        if i < len(pipeline_steps) - 1:
            pipeline_html += '<div class="pipeline-arrow">→</div>'
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ── Agent 卡片 ──
    agent_title = ("🤖 四个专业 Agent" if st.session_state.language == "zh" else "🤖 Four Specialist Agents")
    st.markdown(f'<div class="section-title">{agent_title}</div>',
                unsafe_allow_html=True)
    agent_cards = [
        ("📊", t("welcome_agent_data_name"), t("welcome_agent_data_desc")),
        ("⚠️", t("welcome_agent_risk_name"), t("welcome_agent_risk_desc")),
        ("📋", t("welcome_agent_comply_name"), t("welcome_agent_comply_desc")),
        ("🔍", t("welcome_agent_devil_name"), t("welcome_agent_devil_desc")),
    ]
    cards_html = '<div class="agent-grid">'
    for icon, name, desc in agent_cards:
        cards_html += f'''<div class="agent-feature-card">
            <div class="agent-feature-icon">{icon}</div>
            <div class="agent-feature-name">{name}</div>
            <div class="agent-feature-desc">{desc}</div>
        </div>'''
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── 技术栈 ──
    st.markdown(f'<div class="section-title">{t("welcome_tech_title")}</div>',
                unsafe_allow_html=True)
    tech_items = t("welcome_tech_items").split(" · ")
    tags_html = '<div class="tech-tags-wrap">'
    for item in tech_items:
        tags_html += f'<span class="tech-tag">{item.strip()}</span>'
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<p class="disclaimer">{t("disclaimer")}</p>', unsafe_allow_html=True)
