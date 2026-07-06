"""
FinRisk MultiAgent — 多 Agent 协作金融风险智能分析系统

启动方式:
    streamlit run app.py

然后在浏览器中打开 http://localhost:8501
"""

import os
import sys
import re
import html
import time
import tempfile
from pathlib import Path

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

# ── 安全工具 ──

def sanitize_html(text: str) -> str:
    """对 LLM 输出做 HTML 转义，防止 XSS 注入。"""
    return html.escape(text, quote=True)

def sanitize_api_key(key: str) -> str:
    """脱敏 API Key，仅显示前 6 位 + 后 4 位。"""
    key = (key or "").strip()
    if len(key) <= 10:
        return "***"
    return key[:6] + "****" + key[-4:]

def validate_file_upload(uploaded_file, max_size_mb: int = 20) -> str | None:
    """校验上传文件。返回错误消息字符串，通过返回 None。"""
    if uploaded_file is None:
        return None
    # 大小校验
    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if size_mb > max_size_mb:
        return f"文件过大（{size_mb:.1f}MB），上限 {max_size_mb}MB。"
    # 扩展名校验
    allowed_exts = {'.pdf', '.txt', '.md'}
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in allowed_exts:
        return f"不支持的文件格式 {ext}，仅支持 PDF/TXT/MD。"
    # MIME 类型校验
    import mimetypes
    mime, _ = mimetypes.guess_type(uploaded_file.name)
    allowed_mimes = {'application/pdf', 'text/plain', 'text/markdown', 'text/x-markdown', None}
    if mime and mime not in allowed_mimes:
        return f"不支持的文件类型 {mime}。"
    return None

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit 每次 rerun 都重新执行 app.py，强制清除项目源码模块的缓存
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
        "subtitle": "Coordinator 驱动的多 Agent 协作 · 自动判断问题复杂度 · 简单查询直接答 · 复杂分析派专家",
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
        "about_text": "Coordinator 驱动的多 Agent 金融风险分析平台。自动判断问题类型——简单查询自己搜、专项分析派专家、全面评估启动完整编队。Agent 分析后自判信息是否足够，不够就告诉 Coordinator 需要搜什么。",
        "upload_label": "📄 上传金融文档",
        "upload_help": "支持：年报、招股说明书、债券募集说明书、风险披露报告等 PDF 或文本文件。也可不传文档直接提问。",
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
        "welcome_title": "🧠 Coordinator 驱动的多 Agent 分析引擎",
        "welcome_subtitle": "Coordinator 自动判断你的问题——简单查数据自己搜索回答，专项分析调度相关专家，全面评估启动完整编队。每个 Agent 分析后自判信息是否足够，不够就告诉 Coordinator 需要搜什么。",
        "welcome_agent_data_name": "数据提取 Agent",
        "welcome_agent_data_desc": "先理解问题再提取数据，不填六层框架。信息不够就告诉 Coordinator 需要搜什么。",
        "welcome_agent_risk_name": "风险评估 Agent",
        "welcome_agent_risk_desc": "只分析用户关心的风险维度。自判信息是否足够，不够就申请联网搜索。",
        "welcome_agent_comply_name": "合规审查 Agent",
        "welcome_agent_comply_desc": "聚焦用户问的合规领域，不逐条审查所有框架。需要时主动申请搜索违规记录。",
        "welcome_agent_devil_name": "深度质疑 Agent",
        "welcome_agent_devil_desc": "只质疑用户问题相关的结论，弹药不够向 Coordinator 申请联网搜索补充。",
        "welcome_pipeline_title": "⚡ 分析流程",
        "welcome_pipeline_1": "📄 上传文档",
        "welcome_pipeline_2": "🧠 Coordinator 判断",
        "welcome_pipeline_3": "🔍 混合检索+搜索",
        "welcome_pipeline_4": "🤖 按需派 Agent",
        "welcome_pipeline_5": "✅ Agent 自判完成",
        "welcome_pipeline_6": "📊 综合输出",
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
    },
    "en": {
        "title": "🏦 FinRisk MultiAgent",
        "subtitle": "Coordinator-Driven Multi-Agent · Auto Strategy Selection · Simple Query → Direct Answer · Complex → Dispatch Experts",
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
        "about_text": "Coordinator-driven multi-agent financial risk analysis. Auto-judges question type: simple queries answered directly, analytical questions dispatched to relevant agents, comprehensive assessments run the full team. Agents self-assess completeness and request searches when needed.",
        "upload_label": "📄 Upload Financial Document",
        "upload_help": "Supports: Annual reports, prospectuses, bond offering circulars, risk disclosures (PDF/TXT/MD). Or ask without uploading.",
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
        "welcome_title": "🧠 Coordinator-Driven Multi-Agent Engine",
        "welcome_subtitle": "Coordinator judges your question — simple data queries answered directly with search, analytical questions dispatched to relevant experts, comprehensive assessments run the full team. Each agent self-assesses and requests searches when needed.",
        "welcome_agent_data_name": "Data Extraction Agent",
        "welcome_agent_data_desc": "Understands the question first, extracts only relevant data. Requests web search when info is insufficient.",
        "welcome_agent_risk_name": "Risk Assessment Agent",
        "welcome_agent_risk_desc": "Focuses only on risk dimensions the user asks about. Self-assesses and requests searches.",
        "welcome_agent_comply_name": "Compliance Checker Agent",
        "welcome_agent_comply_desc": "Focuses on the compliance area asked. Proactively requests search for violation records.",
        "welcome_agent_devil_name": "Devil's Advocate Agent",
        "welcome_agent_devil_desc": "Challenges only question-relevant conclusions. Requests web search for more ammunition.",
        "welcome_pipeline_title": "⚡ Analysis Pipeline",
        "welcome_pipeline_1": "📄 Upload",
        "welcome_pipeline_2": "🧠 Coordinator",
        "welcome_pipeline_3": "🔍 Search+RAG",
        "welcome_pipeline_4": "🤖 Agents",
        "welcome_pipeline_5": "✅ Self-Check",
        "welcome_pipeline_6": "📊 Output",
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
    """扫描文档，生成针对该文档的个性化推荐提问。"""
    from src.llm.client import LLMClient, LLMConfig

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

    scan_prompt_en = f"""You are a financial document analysis expert. Quickly scan the following document excerpts.

## Document Info
Filename: {filename}

## Document Excerpts
{context[:4000]}

Generate 3-5 recommended questions based on the document's actual content.
- Under 15 words each
- Output as list starting with "- "

Only output the question list:"""

    prompt = scan_prompt_zh if language == "zh" else scan_prompt_en

    config = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.3, max_tokens=512)
    client = LLMClient(config)
    resp = client.chat([{"role": "user", "content": prompt}])

    questions = []
    for line in resp.strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            q = line[2:].strip()
            if q and len(q) > 3:
                questions.append(q)
        elif line and len(line) > 3 and "?" in line:
            questions.append(line.strip())

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
# 样式（简洁干净）
# ============================================================================

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; text-align: center; color: #1a3c5e; margin-bottom: 0.3rem; }
    .subtitle { font-size: 0.95rem; color: #888; text-align: center; margin-bottom: 1rem; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #2d6a9f; text-align: center; margin: 2rem 0 1rem 0; }

    .pipeline-container { display: flex; align-items: center; justify-content: center; gap: 0; padding: 1rem 0 1.5rem 0; }
    .pipeline-step { background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%); color: #fff; border-radius: 10px; padding: 14px 22px; text-align: center; min-width: 90px; font-weight: 600; font-size: 0.9rem; box-shadow: 0 3px 10px rgba(26,60,94,0.15); animation: fadeInUp 0.5s ease-out both; flex-shrink: 0; }
    .pipeline-step:nth-child(1)  { animation-delay: 0.00s; }
    .pipeline-step:nth-child(3)  { animation-delay: 0.10s; }
    .pipeline-step:nth-child(5)  { animation-delay: 0.20s; }
    .pipeline-step:nth-child(7)  { animation-delay: 0.30s; }
    .pipeline-step:nth-child(9)  { animation-delay: 0.40s; }
    .pipeline-arrow { font-size: 1.2rem; color: #bbb; margin: 0 6px; flex-shrink: 0; user-select: none; animation: arrowPulse 2s infinite; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes arrowPulse { 0%, 100% { opacity: 0.25; } 50% { opacity: 0.70; } }

    .agent-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 1rem 0 1.5rem 0; }
    @media (max-width: 960px) { .agent-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 520px) { .agent-grid { grid-template-columns: 1fr; } }
    .agent-feature-card { border: 1px solid #e8eaed; border-radius: 12px; padding: 24px 18px 20px 18px; text-align: center; background: #fafbfc; transition: transform 0.2s, box-shadow 0.2s; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; height: 100%; min-height: 200px; }
    .agent-feature-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
    .agent-feature-icon { font-size: 2rem; margin-bottom: 10px; line-height: 1; }
    .agent-feature-name { font-weight: 700; font-size: 0.95rem; color: #1a3c5e; margin-bottom: 8px; }
    .agent-feature-desc { font-size: 0.82rem; color: #666; line-height: 1.55; flex-grow: 1; }

    .tech-tag { display: inline-block; background: #e8f0fe; color: #1a73e8; border-radius: 4px; padding: 3px 10px; font-size: 0.8rem; margin: 3px; font-weight: 500; }
    .tech-tags-wrap { text-align: center; margin-bottom: 2rem; line-height: 2; }

    .disclaimer { color: #aaa; font-size: 0.75rem; margin-top: 2rem; text-align: center; }

    .agent-running { border-left: 4px solid #1a73e8; animation: pulse 1.5s infinite; }
    .agent-done    { border-left: 4px solid #0f9d58; }
    .agent-error   { border-left: 4px solid #d93025; }
    @keyframes pulse { 0% { opacity: 1.0; } 50% { opacity: 0.6; } 100% { opacity: 1.0; } }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State 初始化
# ============================================================================

for key, default in [
    ("vector_store", None),
    ("file_processed", False),
    ("processing_file", False),
    ("chat_history", []),
    ("last_analysis", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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

    st.markdown("### " + t("api_settings"))

    api_provider = st.selectbox(
        t("api_provider"),
        [t("api_provider_dashscope"), t("api_provider_custom")],
    )
    is_dashscope = api_provider == t("api_provider_dashscope")

    if is_dashscope:
        api_key = st.text_input(
            t("api_key"), type="password",
            value=os.getenv("DASHSCOPE_API_KEY", ""),
            help=t("api_key_help"),
        )
        api_base = ""
        model = st.selectbox(t("model"), ["qwen-plus", "qwen-max", "qwen-turbo"], index=0, help=t("model_help"))
    else:
        api_key = st.text_input(t("api_key_custom"), type="password")
        api_base = st.text_input(t("api_base"), value="https://api.openai.com/v1")
        model = st.text_input(t("model"), value="gpt-4o")

    st.divider()

    web_search_enabled = st.checkbox(
        "🌐 联网搜索" if st.session_state.language == "zh" else "🌐 Web Search",
        value=True,
        help=("默认开启。Agent 会在文档信息不足时自动联网搜索最新数据。"
              if st.session_state.language == "zh"
              else "On by default. Agents search the web when document info is insufficient."),
    )

    st.divider()

    st.markdown("### " + t("model_params"))
    temperature = st.slider(t("temperature"), 0.0, 1.0, 0.3, 0.05, help=t("temperature_help"))
    max_tokens = st.slider(t("max_tokens"), 1024, 8192, 4096, 256, help=t("max_tokens_help"))

    st.divider()

    st.markdown("### " + t("about_title"))
    st.markdown(t("about_text"))
    if st.session_state.language == "zh":
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.7; color:#555">
        <b>四个专业 Agent</b><br>
        📊 数据提取 &nbsp; ⚠️ 风险评估 &nbsp; 📋 合规审查 &nbsp; 🔍 深度质疑<br><br>
        <a href="https://github.com/leokiy/finrisk-multiagent">GitHub</a> · <a href="https://github.com/leokiy/finrisk-multiagent/issues">Issues</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.7; color:#555">
        <b>Four Specialist Agents</b><br>
        📊 Data Extraction &nbsp; ⚠️ Risk Assessment &nbsp; 📋 Compliance &nbsp; 🔍 Devil's Advocate<br><br>
        <a href="https://github.com/leokiy/finrisk-multiagent">GitHub</a> · <a href="https://github.com/leokiy/finrisk-multiagent/issues">Issues</a>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 主页面
# ============================================================================

st.markdown(f'<div class="main-title">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{t("subtitle")}</p>', unsafe_allow_html=True)

# ============================================================================
# 文件上传
# ============================================================================

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        t("upload_label"), type=["pdf", "txt", "md"], help=t("upload_help"),
    )

with col_info:
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric(t("file_name"), uploaded_file.name)
        st.metric(t("file_size"), f"{file_size_mb:.1f} MB")
        st.session_state._uploaded_filename = uploaded_file.name
    elif st.session_state.file_processed:
        cached_name = st.session_state.get("_uploaded_filename", "unknown")
        st.metric(t("file_name"), cached_name)
    else:
        st.info(t("upload_placeholder"))

# ============================================================================
# 处理上传的文件
# ============================================================================

if uploaded_file and not st.session_state.file_processed:
    # 安全校验
    validation_error = validate_file_upload(uploaded_file)
    if validation_error:
        st.error(validation_error)
        st.stop()

    if not api_key:
        st.error(t("api_key_warning"))
        st.stop()

    st.session_state.processing_file = True
    st.session_state._file_bytes = uploaded_file.getvalue()
    st.session_state._uploaded_filename = uploaded_file.name

    with st.status("Processing document...", expanded=True) as status:
        try:
            suffix = Path(st.session_state._uploaded_filename).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(st.session_state._file_bytes)
                tmp_path = tmp.name

            from src.rag.engine import build_rag_from_file
            vector_store = build_rag_from_file(tmp_path, api_key=api_key)

            st.session_state.vector_store = vector_store
            st.session_state.file_processed = True
            st.session_state.processing_file = False
            st.session_state.chat_history = []
            st.session_state.doc_questions = None
            st.session_state.doc_type = ""
            st.session_state.doc_company = ""

            try:
                from src.llm.client import LLMConfig, LLMClient
                front_chunks = vector_store.search("年报 半年报 季度报告 招股说明书 公司 股份", top_k=5, api_key=api_key)
                front_text = " ".join(r.chunk.text[:200] for r in front_chunks)[:1500]
                if front_text.strip():
                    doc_cfg = LLMConfig(api_key=api_key, model="qwen-turbo", temperature=0.1, max_tokens=80)
                    doc_client = LLMClient(doc_cfg)
                    lang_hint = st.session_state.get("language", "zh")
                    if lang_hint == "zh":
                        doc_prompt = f"输出两行：第一行=文档类型（年报/半年报/季报/招股书/其他），第二行=公司全称。只输出这两行，不要其他内容：\n{front_text}"
                    else:
                        doc_prompt = f"Output two lines: line1=document type, line2=company full name. Only these two lines:\n{front_text}"
                    doc_resp = doc_client.chat([{"role": "user", "content": doc_prompt}])
                    lines = [l.strip() for l in doc_resp.strip().split("\n") if l.strip()]
                    if len(lines) >= 1:
                        st.session_state.doc_type = lines[0]
                    if len(lines) >= 2:
                        st.session_state.doc_company = lines[1]
            except Exception as e:
                st.session_state.doc_type = ""
                st.session_state.doc_company = ""
                print(f"[DocType] skipped: {e}")

            os.unlink(tmp_path)
            status.update(label=f"Done: {vector_store.chunk_count} chunks, {vector_store.table_count} tables", state="complete")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                status.update(label=f"[ERROR] {str(exc)[:200]}", state="error")
            except Exception:
                status.update(label="Processing failed. Check console for details.", state="error")
            st.session_state.processing_file = False
            st.stop()

# ============================================================================
# 文件就绪状态 + 重置
# ============================================================================

if st.session_state.file_processed:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        fname = st.session_state.get("_uploaded_filename", "")
        fchunks = st.session_state.vector_store.chunk_count if st.session_state.vector_store else 0
        st.success(t("file_ready", name=fname, count=fchunks))
    with col_b:
        if st.button(t("reset")):
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

# ── 推荐提问 ──
last = st.session_state.get("last_analysis")
has_doc = st.session_state.file_processed and st.session_state.vector_store and not st.session_state.vector_store.is_empty

if last and last.get("followup_questions"):
    quick_questions = last["followup_questions"]
elif has_doc and not st.session_state.get("doc_questions"):
    if api_key:
        with st.spinner("🔍 " + ("正在分析文档，生成推荐提问..." if st.session_state.language == "zh" else "Analyzing document...")):
            try:
                st.session_state.doc_questions = _generate_doc_questions(
                    st.session_state.vector_store, api_key,
                    st.session_state.language,
                    st.session_state.get("_uploaded_filename", "document"),
                )
            except Exception:
                st.session_state.doc_questions = t("quick_questions")
    quick_questions = st.session_state.get("doc_questions") or t("quick_questions")
elif has_doc and st.session_state.get("doc_questions"):
    quick_questions = st.session_state["doc_questions"]
else:
    quick_questions = t("quick_questions")

# ── 聊天记录 ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 欢迎信息（无对话且无文档时显示）──
if not st.session_state.chat_history and not st.session_state.file_processed:
    st.divider()

    st.markdown(f'<h2 class="section-title" style="font-size:1.6rem;color:#1a3c5e">{t("welcome_title")}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{t("welcome_subtitle")}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">{t("welcome_pipeline_title")}</div>', unsafe_allow_html=True)
    pipeline_steps = [t("welcome_pipeline_1"), t("welcome_pipeline_2"), t("welcome_pipeline_3"), t("welcome_pipeline_4"), t("welcome_pipeline_5")]
    pipeline_html = '<div class="pipeline-container">'
    for i, step in enumerate(pipeline_steps):
        pipeline_html += f'<div class="pipeline-step">{step}</div>'
        if i < len(pipeline_steps) - 1:
            pipeline_html += '<div class="pipeline-arrow">→</div>'
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

    agent_title = ("🤖 四个专业 Agent" if st.session_state.language == "zh" else "🤖 Four Specialist Agents")
    st.markdown(f'<div class="section-title">{agent_title}</div>', unsafe_allow_html=True)
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

    st.markdown(f'<div class="section-title">{t("welcome_tech_title")}</div>', unsafe_allow_html=True)
    tech_items = t("welcome_tech_items").split(" · ")
    tags_html = '<div class="tech-tags-wrap">'
    for item in tech_items:
        tags_html += f'<span class="tech-tag">{item.strip()}</span>'
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<p class="disclaimer">{t("disclaimer")}</p>', unsafe_allow_html=True)

# ── 推荐提问按钮 ──
st.markdown("**" + ("📋 推荐问题:" if st.session_state.language == "zh" else "📋 Suggested Questions:") + "**")
qkey = st.session_state.get("_qkey", 0)
cols = st.columns(len(quick_questions))
for i, (col, q) in enumerate(zip(cols, quick_questions)):
    with col:
        if st.button(q, key=f"quick_{qkey}_{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.session_state._qkey = qkey + 1

# ── 输入区 ──
user_query = st.chat_input(t("chat_placeholder"))

if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

# ── 处理用户提问 ──
if user_query:
    if not api_key:
        st.error(t("api_key_warning"))
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # ── 运行分析：Coordinator 驱动的多 Agent 协作 ──
    from src.llm.client import LLMClient, LLMConfig
    from src.orchestrator_v2 import OrchestratorV2

    config = LLMConfig(
        api_key=api_key, api_base=api_base,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )
    llm_client = LLMClient(config)
    orchestrator = OrchestratorV2(llm_client, language=st.session_state.language)
    doc_type_str = st.session_state.get("doc_type", "")
    doc_company_str = st.session_state.get("doc_company", "")

    vs = st.session_state.vector_store
    if vs is None:
        from src.rag.engine import VectorStore
        vs = VectorStore()

    with st.chat_message("assistant"):
        report_placeholder = st.empty()
        progress_placeholder = st.empty()
        streamed_text = []
        progress_lines = []

        def on_progress(agent: str, status: str, content: str):
            icon = {"running": "⏳", "done": "✅", "error": "❌"}.get(status, "•")
            progress_lines.append(f"{icon} **{agent}**: {content[:150]}")
            progress_placeholder.markdown("> " + "\n> ".join(progress_lines[-10:]))

        def on_token(token: str):
            streamed_text.append(token)
            report_placeholder.markdown("".join(streamed_text))

        try:
            result = orchestrator.run(
                user_query=user_query,
                vector_store=vs,
                api_key=api_key,
                on_token=on_token,
                on_progress=on_progress,
                web_search_enabled=web_search_enabled,
                doc_type=doc_type_str,
                doc_company=doc_company_str,
                chat_history=st.session_state.get("chat_history", []),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                "final_report": f"分析出错: {e}",
                "execution_log": [],
                "followup_questions": [],
            }

        progress_placeholder.empty()

        report = result.get("final_report", "")
        if not report:
            fail_msg = ("分析失败，请检查 API Key 和网络连接。"
                       if st.session_state.language == "zh"
                       else "Analysis failed. Please check your API Key and network connection.")
            report = fail_msg
            report_placeholder.markdown(report)

        st.download_button(
            label=t("download"),
            data=report,
            file_name=f"finrisk_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    st.session_state.chat_history.append({"role": "assistant", "content": report})
    st.session_state.last_analysis = result
    st.rerun()

# ── Agent 详情 expander ──
if st.session_state.get("last_analysis"):
    result = st.session_state.last_analysis
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
                if content:
                    st.markdown(content)
                else:
                    st.caption("（分析过程中未调用此 Agent）" if st.session_state.language == "zh" else "(This agent was not called)")
        with tabs[4]:
            for log in result.get("execution_log", []):
                status_icon = {"running": "⏳", "done": "✅", "error": "❌"}
                icon = status_icon.get(log["status"], "•")
                content = log.get("content", "")
                if content:
                    st.markdown(f"{icon} **{log['agent']}**: {content[:200]}")
                else:
                    st.markdown(f"{icon} **{log['agent']}**: {log['status']}")

# ── 底部 ──
st.divider()
st.markdown(f'<p class="disclaimer">{t("disclaimer")}</p>', unsafe_allow_html=True)
