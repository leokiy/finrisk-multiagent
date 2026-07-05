"""
🎯 Orchestrator — 多 Agent 协作的中央调度器。

工作流:
  1. 接收用户问题 + RAG 知识库
  2. 并行启动 4 个专业 Agent (各从 RAG 中独立检索)
  3. 收集所有 Agent 的输出
  4. 让深度质疑 Agent 基于其他 Agent 的输出做第二轮分析
  5. 综合所有结果，生成最终风险报告
"""

import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient, LLMConfig
from src.rag.engine import VectorStore
from src.agents.base import AgentResult, _load_prompt
from src.agents.data_extractor import DataExtractorAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.compliance_checker import ComplianceCheckerAgent
from src.agents.devils_advocate import DevilsAdvocateAgent


# ============================================================================
# 进度回调
# ============================================================================

class ProgressReporter:
    """将 Agent 执行进度通过回调传出（供 Streamlit UI 消费）。"""

    def __init__(self):
        self.logs: list[dict] = []

    def update(self, agent_name: str, status: str, content: str = ""):
        self.logs.append({"agent": agent_name, "status": status, "content": content})

    def clear(self):
        self.logs.clear()


# ============================================================================
# Orchestrator
# ============================================================================

class Orchestrator:
    """多 Agent 协作调度器。

    设计要点:
      - 第一轮: 数据提取、风险评估、合规审查并行执行
      - 第二轮: 深度质疑 Agent 看到前三个的输出后执行
      - 第三轮: 协调 Agent（LLM）综合所有输出生成最终报告
    """

    def __init__(self, llm_client: LLMClient, language: str = "zh"):
        self.llm = llm_client
        self.language = language
        self.reporter = ProgressReporter()

        # 初始化所有 Agent
        self.data_extractor  = DataExtractorAgent(llm_client, language)
        self.risk_assessor   = RiskAssessorAgent(llm_client, language)
        self.compliance      = ComplianceCheckerAgent(llm_client, language)
        self.devils_advocate = DevilsAdvocateAgent(llm_client, language)

        # 加载协调 prompt
        self.synthesis_prompt = _load_prompt("orchestrator", language)

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def run(self, user_query: str, vector_store: VectorStore,
            api_key: str = "", on_synthesis_token=None,
            web_search_enabled: bool = False,
            doc_type: str = "") -> dict:
        """完整执行多 Agent 协作工作流，自适应分析深度。"""
        self.reporter.clear()
        lang = self.language

        # ── 第零轮：理解问题 + 判断意图 ──
        msg0 = ("正在理解你的问题，判断分析策略..."
                if lang == "zh" else
                "Understanding your question, determining analysis depth...")
        self.reporter.update("Orchestrator", "running", msg0)

        queries, question_type = self._reformulate_per_agent(user_query)

        # ── 联网搜索：为各 Agent 生成专属搜索查询并并行执行 ──
        if web_search_enabled:
            self.reporter.update(
                "Orchestrator", "running",
                ("正在为各 Agent 生成联网搜索查询..."
                 if lang == "zh" else
                 "Generating per-agent web search queries...")
            )
            agent_search_queries = self._generate_agent_search_queries(
                user_query, queries, question_type, doc_type
            )
            # 调试：打印生成的搜索词
            import json
            print(f"[DEBUG] question_type={question_type}")
            print(f"[DEBUG] search_queries={json.dumps(agent_search_queries, ensure_ascii=False, indent=2)}")

            agent_web_results = self._execute_agent_searches(
                agent_search_queries, api_key
            )
            # 调试：打印搜索结果摘要
            for key, results in agent_web_results.items():
                print(f"[DEBUG] {key}: {len(results)} results")
                for r in results:
                    print(f"[DEBUG]   - {r.title[:80]}")
                    print(f"[DEBUG]     snippet={r.snippet[:150]}...")
            searched_agents = [k for k, v in agent_web_results.items() if v]
            if searched_agents:
                self.reporter.update(
                    "Orchestrator", "done",
                    (f"联网搜索完成：{', '.join(searched_agents)}"
                     if lang == "zh" else
                     f"Web search done: {', '.join(searched_agents)}")
                )
            else:
                self.reporter.update(
                    "Orchestrator", "error",
                    ("联网搜索未返回任何结果"
                     if lang == "zh" else
                     "Web search returned no results")
                )
        else:
            agent_web_results = {}

        # ── 根据问题类型自适应分析深度 ──
        # factual: 只查数据，精简回答
        # analytical: 数据+风险+质疑，不要合规审查
        # comprehensive: 全流程
        is_factual = question_type == "factual"

        if is_factual:
            # 精简模式：只跑数据提取，直接给答案
            msg = ("问题为数据查询，直接提取数据+联网搜索..."
                   if lang == "zh" and web_search_enabled else
                   "问题为数据查询，直接提取数据..."
                   if lang == "zh" else
                   "Factual query, extracting data + web search..."
                   if web_search_enabled else
                   "Factual query, extracting data directly...")
            self.reporter.update("Orchestrator", "running", msg)

            data_query = queries.get("data_extractor", user_query)
            data_web_results = agent_web_results.get("data_extractor")
            data_result = self.data_extractor.run(
                data_query, vector_store, api_key=api_key,
                enable_search=web_search_enabled,
                web_search_results=data_web_results,
            )
            self._log_agent_result(data_result)

            # 综合：文档数据 + 网络搜索结果
            self.llm.config.enable_search = web_search_enabled
            final_report = self._synthesize_factual(
                user_query, data_result, doc_type, on_synthesis_token,
                web_search_enabled=web_search_enabled,
                web_results=data_web_results,
            )

            self.reporter.update("Orchestrator", "done",
                                 "分析完成。" if lang == "zh" else "Analysis complete.")

            return {
                "query": user_query,
                "question_type": "factual",
                "data_extraction": self._safe_result(data_result),
                "risk_assessment": "",
                "compliance_check": "",
                "devils_advocate": "",
                "rebuttals": {},
                "final_report": final_report,
                "execution_log": self.reporter.logs,
            }

        # ── 第一轮：3 Agent 并行 ──
        msg = ("启动第一轮分析：数据提取、风险评估、合规审查并行执行..."
               if lang == "zh" else
               "Round 1: Agents analyzing with role-specific instructions...")
        if web_search_enabled:
            msg += ("（各Agent使用专属联网搜索结果）"
                    if lang == "zh" else
                    " (each agent uses its own web search results)")
        self.reporter.update("Orchestrator", "running", msg)

        round1_results = self._run_parallel_with_queries(
            [self.data_extractor, self.risk_assessor, self.compliance],
            queries, vector_store, api_key,
            enable_search=web_search_enabled,
            agent_web_results=agent_web_results,
        )

        data_result, risk_result, compliance_result = round1_results

        # ── 第二轮：深度质疑 ──
        msg2 = ("正在审阅前三方分析结果，寻找盲点和矛盾..."
                if lang == "zh" else
                "Devil's Advocate reviewing all outputs for blind spots...")
        self.reporter.update("Devil's Advocate", "running", msg2)

        other_context = {
            self.data_extractor.prompt_file: data_result.content if data_result.success else f"[Failed] {data_result.error}",
            self.risk_assessor.prompt_file: risk_result.content if risk_result.success else f"[Failed] {risk_result.error}",
            self.compliance.prompt_file: compliance_result.content if compliance_result.success else f"[Failed] {compliance_result.error}",
        }

        devil_result = self.devils_advocate.run(
            queries.get("devils_advocate", user_query), vector_store,
            context_from_other_agents=other_context,
            api_key=api_key,
            enable_search=web_search_enabled,
            web_search_results=agent_web_results.get("devils_advocate"),
        )
        self._log_agent_result(devil_result)

        # ── 第三轮：Agent 回应质疑 ──
        rebuttals = {}
        if devil_result.success and devil_result.content:
            msg_rebuttal = ("Agent 正在回应质疑..."
                           if lang == "zh" else
                           "Agents responding to challenges...")
            self.reporter.update("Orchestrator", "running", msg_rebuttal)
            rebuttals = self._collect_rebuttals(
                data_result, risk_result, compliance_result,
                devil_result, vector_store, api_key, web_search_enabled,
            )

        # ── 第四轮：综合 ──
        msg4 = ("正在综合所有分析结果、质疑与回应，生成最终报告..."
                if lang == "zh" else
                "Synthesizing all analysis, challenges, and rebuttals into final report...")
        self.reporter.update("Orchestrator", "running", msg4)

        self.llm.config.enable_search = web_search_enabled
        final_report = self._synthesize_with_rebuttals(
            user_query, data_result, risk_result, compliance_result,
            devil_result, rebuttals, on_synthesis_token,
        )

        self.reporter.update("Orchestrator", "done",
                             "分析完成。" if lang == "zh" else "Analysis complete.")

        return {
            "query": user_query,
            "question_type": question_type,
            "refined_query":      queries.get("data_extractor", user_query),
            "data_extraction":    self._safe_result(data_result),
            "risk_assessment":    self._safe_result(risk_result),
            "compliance_check":   self._safe_result(compliance_result),
            "devils_advocate":    self._safe_result(devil_result),
            "rebuttals":          {k: v.content for k, v in rebuttals.items()},
            "final_report":       final_report,
            "execution_log":      self.reporter.logs,
        }

    # ----------------------------------------------------------------
    # 内部 — 问题转述
    # ----------------------------------------------------------------

    def _reformulate_per_agent(self, user_query: str) -> tuple[dict[str, str], str]:
        """为每个 Agent 生成针对其角色的专属转述问题。

        Returns:
            (queries_dict, question_type)
            question_type: "factual" | "analytical" | "comprehensive"
        """
        lang = self.language

        # 先判断问题类型
        qt_prompt = (
            f"判断以下用户问题属于哪种类型，只输出一个词（factual/analytical/comprehensive）：\n\n"
            f"factual = 查询具体数据（如\"利润多少\"\"营收增长了多少\"\"资产负债率是多少\"）\n"
            f"analytical = 分析特定方面（如\"偿债能力如何\"\"有没有合规风险\"\"为什么毛利率下降\"）\n"
            f"comprehensive = 全面评估（如\"做个风险评估\"\"全面分析这家公司\"\"有没有问题\"）\n\n"
            f"用户问题：{user_query}\n\n类型："
            if lang == "zh" else
            f"Classify this user question. Output one word:\n\n"
            f"factual = asking for specific data (e.g., 'what is the profit', 'how much did revenue grow')\n"
            f"analytical = analyzing a specific aspect (e.g., 'how is debt capacity', 'any compliance risks')\n"
            f"comprehensive = full assessment (e.g., 'do a risk assessment', 'analyze this company')\n\n"
            f"Question: {user_query}\n\nType:"
        )
        try:
            qt_resp = self.llm.chat(
                [{"role": "user", "content": qt_prompt}],
                model="qwen-turbo", temperature=0.0, max_tokens=20,
            )
            qt = qt_resp.strip().lower()
            if "factual" in qt:
                question_type = "factual"
            elif "analytical" in qt:
                question_type = "analytical"
            else:
                question_type = "comprehensive"
        except Exception:
            question_type = "comprehensive"

        role_hints = {
            "data_extractor": (
                "这个Agent的职责是从文档中提取结构化财务数据。"
                "转述时要强调：需要哪些具体数据、从哪些章节找、数据格式要求。"
                if lang == "zh" else
                "This agent extracts structured financial data. Emphasize: what data, which sections, format."
            ),
            "risk_assessor": (
                "这个Agent的职责是从四个维度评估风险。"
                "转述时要强调：关注哪些风险信号、每个维度需要什么证据、风险传导机制。"
                if lang == "zh" else
                "This agent assesses risk across four dimensions. Emphasize: risk signals, evidence needed, transmission mechanisms."
            ),
            "compliance_checker": (
                "这个Agent的职责是对照监管框架审查合规问题。"
                "转述时要强调：适用的法规框架、需要逐条检查的披露项、区分'违规'和'需关注'。"
                if lang == "zh" else
                "This agent checks regulatory compliance. Emphasize: applicable frameworks, disclosure items to check, violation vs concern."
            ),
            "devils_advocate": (
                "这个Agent的职责是挑战其他Agent的结论，寻找盲点。"
                "转述时要强调：质疑的角度（数据/假设/逻辑/遗漏/时间）、要求找出矛盾。"
                if lang == "zh" else
                "This agent challenges other agents' conclusions. Emphasize: angles of challenge, find contradictions."
            ),
        }

        if lang == "zh":
            prompt = f"""你是一个分析策略制定专家。用户向一个金融文档分析系统提出了问题。系统有4个专业Agent，每个Agent的能力不同。请你为每个Agent制定专属的分析指引。

## 用户问题
{user_query}

## 各Agent角色说明及转述要求

"""
            for key, hint in role_hints.items():
                prompt += f"### {key}\n{hint}\n\n"

            prompt += """请输出每个Agent的专属分析指引，格式：
data_extractor: [转述后的分析任务]
risk_assessor: [转述后的分析任务]
compliance_checker: [转述后的分析任务]
devils_advocate: [转述后的分析任务]

注意：每个指引要具体、与该Agent的能力匹配、明确告知Agent应该做什么和怎么做。"""
        else:
            prompt = f"""You are an analysis strategy expert. A user asked a question to a financial document analysis system with 4 specialized agents. Create tailored analysis directives for each agent.

## User Question
{user_query}

## Agent Roles
"""
            for key, hint in role_hints.items():
                prompt += f"### {key}\n{hint}\n\n"

            prompt += """Output one directive per agent:
data_extractor: [directive]
risk_assessor: [directive]
compliance_checker: [directive]
devils_advocate: [directive]"""

        resp = self.llm.chat(
            [{"role": "user", "content": prompt}],
            model="qwen-turbo", temperature=0.2, max_tokens=800,
        )

        # 解析输出
        queries = {}
        for line in resp.strip().split("\n"):
            line = line.strip()
            for key in role_hints:
                if line.startswith(f"{key}:") or line.startswith(f"{key}："):
                    queries[key] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        # fallback
        for key in role_hints:
            if key not in queries or not queries[key]:
                queries[key] = user_query
        return queries, question_type

    # ----------------------------------------------------------------
    # 内部 — 联网搜索查询生成
    # ----------------------------------------------------------------

    def _generate_agent_search_queries(self, user_query: str,
                                       per_agent_queries: dict[str, str],
                                       question_type: str,
                                       doc_type: str = "") -> dict[str, list[str]]:
        """为每个 Agent 生成两条联网搜索查询：一条找信息、一条交叉验证。

        每个 Agent 的搜索方向不同：
        - 数据提取：搜具体财务数据 → 用另一来源验证数字
        - 风险评估：搜行业风险/负面新闻 → 搜相反观点/反驳信息
        - 合规审查：搜监管政策变化 → 搜公司是否有违规记录
        - 深度质疑：搜争议事件/做空报告 → 搜公司回应/辩解

        Returns:
            {agent_key: [info_query, verify_query]}
        """
        lang = self.language

        # 数据查询模式：一条找数据、一条验证
        if question_type == "factual":
            return {"data_extractor": [user_query, f"{user_query} 最新数据 核实"]}

        if lang == "zh":
            prompt = f"""你是金融信息检索专家。4个专业Agent即将分析一份金融文档，每个Agent需要联网搜索来获取信息并验证自己判断的准确性。请为每个Agent生成2条搜索查询。

## 用户问题
{user_query}

## 文档类型
{doc_type or '未知'}

## 各Agent的搜索需求（信息 + 验证）
- 数据提取Agent：
  信息查询：搜索最新的具体财务数据、关键指标、季度/年度财报
  验证查询：搜另一个来源核实同一数据，确认数字的准确性
- 风险评估Agent：
  信息查询：搜索行业风险动态、市场环境变化、负面新闻、经营风险事件
  验证查询：搜索相反观点或公司正面信息，检验自己的风险判断是否偏颇
- 合规审查Agent：
  信息查询：搜索最新的监管政策变化、合规要求更新
  验证查询：搜索该公司是否有实际违规记录、处罚公告、监管函
- 深度质疑Agent：
  信息查询：搜索可能被忽略的风险信号、市场争议、做空报告
  验证查询：搜索公司对这些质疑的回应、澄清公告

## 要求
- 每个Agent生成2条查询，第一条找信息，第二条交叉验证
- 查询自然语言，10-30字，包含关键实体（公司名/行业/指标）
- 严格按格式输出（每行一个）：
data_extractor_info: <信息查询>
data_extractor_verify: <验证查询>
risk_assessor_info: <信息查询>
risk_assessor_verify: <验证查询>
compliance_checker_info: <信息查询>
compliance_checker_verify: <验证查询>
devils_advocate_info: <信息查询>
devils_advocate_verify: <验证查询>"""
        else:
            prompt = f"""You are a financial information retrieval expert. Generate 2 search queries for each agent: one to find information, one to cross-verify.

## User Question
{user_query}

## Document Type
{doc_type or 'Unknown'}

## Agent Search Needs (info + verification)
- Data Extractor: info query for latest financials → verify query to cross-check numbers with another source
- Risk Assessor: info query for industry risks/negative news → verify query for opposing viewpoints
- Compliance Checker: info query for regulatory changes → verify query for actual violation records
- Devil's Advocate: info query for overlooked risks/controversies → verify query for company responses

## Requirements
- 2 queries per agent: first for info, second for verification
- Natural language, include key entities
- Output format (one per line):
data_extractor_info: <info query>
data_extractor_verify: <verify query>
risk_assessor_info: <info query>
risk_assessor_verify: <verify query>
compliance_checker_info: <info query>
compliance_checker_verify: <verify query>
devils_advocate_info: <info query>
devils_advocate_verify: <verify query>"""

        key_map = {
            "data_extractor_info": "data_extractor",
            "data_extractor_verify": "data_extractor",
            "risk_assessor_info": "risk_assessor",
            "risk_assessor_verify": "risk_assessor",
            "compliance_checker_info": "compliance_checker",
            "compliance_checker_verify": "compliance_checker",
            "devils_advocate_info": "devils_advocate",
            "devils_advocate_verify": "devils_advocate",
        }

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                model="qwen-turbo", temperature=0.1, max_tokens=600,
            )
        except Exception:
            return {k: [user_query] for k in
                    ["data_extractor", "risk_assessor", "compliance_checker", "devils_advocate"]}

        # 解析：同一 agent 的两条查询合并到同一个 list
        queries: dict[str, list[str]] = {}
        for line in resp.strip().split("\n"):
            line = line.strip()
            for prefix, key in key_map.items():
                if line.lower().startswith(prefix.lower()):
                    q = line.split(":", 1)[-1].strip().strip('"').strip("'")
                    if q and len(q) > 2:
                        queries.setdefault(key, []).append(q)

        # 补全缺失的 Agent
        for key in ["data_extractor", "risk_assessor", "compliance_checker", "devils_advocate"]:
            if key not in queries or not queries[key]:
                queries[key] = [user_query]

        return queries

    def _execute_agent_searches(self, agent_search_queries: dict[str, list[str]],
                                api_key: str) -> dict[str, list]:
        """并行执行所有 Agent 的联网搜索。单个搜索失败不影响其他。"""
        from src.search.web_search import search_web

        all_results: dict[str, list] = {}

        def _search_for_agent(agent_key: str, queries: list[str]):
            results = []
            for q in queries[:2]:  # 每个 Agent 最多 2 个搜索
                try:
                    r = search_web(q, api_key=api_key, max_results=2,
                                   language=self.language)
                    if r:
                        results.extend(r)
                except Exception:
                    pass  # 单个搜索失败不阻塞
            return agent_key, results

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_search_for_agent, key, queries): key
                for key, queries in agent_search_queries.items()
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, results = future.result()
                    all_results[key] = results
                except Exception:
                    pass

        return all_results

    def _reformulate_query(self, user_query: str) -> str:
        """理解并转述用户问题：明确问什么、需要什么信息、好答案长什么样。"""
        lang = self.language
        if lang == "zh":
            prompt = f"""你是一个问题分析专家。用户在向一个金融文档分析系统提问。请你理解用户的问题，然后做三件事：

1. **问题澄清**：用户到底想知道什么？用一句话说清楚。
2. **信息需求**：要回答这个问题，需要从文档中提取哪些具体信息？（列出3-5项）
3. **分析指引**：转述为一个清晰的分析任务，指导下游的分析Agent应该做什么、怎么回答。

注意：
- 不要回答用户的问题，只做理解和转述
- 如果用户问的是具体数据（如"利润多少"），指引要强调"直接提取数据、标注来源"
- 如果用户问的是风险评估，指引要强调"从多个维度分析、引用证据"
- 转述后的任务描述要明确告诉Agent：回答要精准、简洁、每条信息标注出处

用户问题：{user_query}

请输出（格式：先一句话澄清 → 列出需要的信息 → 给出分析指引）："""

        else:
            prompt = f"""You are a question analysis expert. A user is asking a question to a financial document analysis system. Understand the question and do three things:

1. **Clarification**: What exactly does the user want to know? State in one sentence.
2. **Information needs**: What specific information from the document is needed to answer? (List 3-5 items)
3. **Analysis directive**: Reformulate as a clear analysis task, guiding downstream agents on what to do and how to answer.

Note:
- Do NOT answer the user's question. Only understand and reformulate.
- If asking for specific data (e.g., "what's the profit"), emphasize: extract data directly, cite sources.
- If asking for risk assessment, emphasize: analyze from multiple dimensions, cite evidence.
- The reformulated task should tell agents: be precise, be concise, cite sources for every fact.

User question: {user_query}

Output (clarification → information needs → analysis directive):"""

        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            model="qwen-turbo", temperature=0.1, max_tokens=600,
        )

    def _run_parallel_with_queries(self, agents: list, queries: dict,
                                    store: VectorStore, api_key: str,
                                    enable_search: bool = False,
                                    agent_web_results: dict | None = None
                                    ) -> list[AgentResult]:
        """并行执行多个 Agent，各自使用专属转述问题 + 专属联网搜索结果。"""
        results: list[AgentResult] = []
        # agent → query 映射
        agent_key_map = {
            self.data_extractor: "data_extractor",
            self.risk_assessor: "risk_assessor",
            self.compliance: "compliance_checker",
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for agent in agents:
                agent_key = agent_key_map.get(agent, "")
                agent_query = queries.get(agent_key, "")
                if not agent_query:
                    agent_query = list(queries.values())[0] if queries else ""
                web_results = (agent_web_results or {}).get(agent_key)
                futures[
                    executor.submit(
                        agent.run, agent_query, store, None, 20, api_key,
                        enable_search, web_results
                    )
                ] = agent
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                self._log_agent_result(result)

        ordered = []
        agent_map = {r.agent_name: r for r in results}
        for agent in agents:
            ordered.append(agent_map.get(agent.name,
                           AgentResult(agent.name, success=False, error="未返回结果")))
        return ordered

    def _run_parallel(self, agents: list, query: str, store: VectorStore,
                      api_key: str, enable_search: bool = False,
                      agent_web_results: dict | None = None) -> list[AgentResult]:
        """并行执行多个 Agent（使用相同 query，兼容旧调用）。"""
        results: list[AgentResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for agent in agents:
                agent_key_map = {
                    self.data_extractor: "data_extractor",
                    self.risk_assessor: "risk_assessor",
                    self.compliance: "compliance_checker",
                }
                web_results = (agent_web_results or {}).get(
                    agent_key_map.get(agent, "")
                )
                futures[
                    executor.submit(
                        agent.run, query, store, None, 20, api_key,
                        enable_search, web_results
                    )
                ] = agent
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                self._log_agent_result(result)

        # 保持原始顺序
        ordered = []
        agent_map = {r.agent_name: r for r in results}
        for agent in agents:
            ordered.append(agent_map.get(agent.name,
                           AgentResult(agent.name, success=False, error="未返回结果")))
        return ordered

    def _synthesize(self, user_query: str, refined_query: str,
                    data: AgentResult, risk: AgentResult,
                    compliance: AgentResult, devil: AgentResult) -> str:
        """调用 LLM 综合所有 Agent 的输出，生成最终报告。"""
        messages = self._build_synthesis_messages(
            user_query, refined_query, data, risk, compliance, devil)
        return self.llm.chat(messages)

    def _synthesize_stream(self, user_query: str, refined_query: str,
                           data: AgentResult, risk: AgentResult,
                           compliance: AgentResult, devil: AgentResult,
                           on_token=None) -> str:
        """流式版本——每个 token 通过 on_token 回调推送。"""
        messages = self._build_synthesis_messages(
            user_query, refined_query, data, risk, compliance, devil)
        if on_token:
            return self.llm.chat_stream(messages, on_token=on_token)
        return self.llm.chat(messages)

    def _build_synthesis_messages(self, user_query: str, refined_query: str,
                                  data: AgentResult, risk: AgentResult,
                                  compliance: AgentResult, devil: AgentResult) -> list[dict]:
        content = f"""## 用户原始问题
{user_query}

## 问题理解与转述
{refined_query}

## 📊 数据提取 Agent 输出
{data.content if data.success else f'[执行失败: {data.error}]'}

## ⚠️ 风险评估 Agent 输出
{risk.content if risk.success else f'[执行失败: {risk.error}]'}

## 📋 合规审查 Agent 输出
{compliance.content if compliance.success else f'[执行失败: {compliance.error}]'}

## 🔍 深度质疑 Agent 输出
{devil.content if devil.success else f'[执行失败: {devil.error}]'}

请综合以上所有分析，生成最终报告。
重要：你的回答必须直接回应用户的原始问题。如果用户问的是具体数据，直接给出数据。如果用户问的是风险分析，给出分析。回答要精准、简洁、每个事实标注出处。"""
        return [
            {"role": "system", "content": self.synthesis_prompt},
            {"role": "user", "content": content},
        ]

    # ── 精简模式：数据查询 ──

    def _synthesize_factual(self, user_query, data_result, doc_type, on_token=None,
                            web_search_enabled: bool = False,
                            web_results: list | None = None):
        """数据查询模式：结合文档数据和联网搜索结果，直接给出答案。

        三层策略：
        1. 有预搜索结果 → 注入结果，LLM 直接综合（不重复搜索）
        2. 开启了搜索但无预搜索结果 → LLM 自行联网搜索
        3. 未开启搜索 → 只用文档数据
        """
        from datetime import datetime

        lang = self.language
        doc_info = f"文档类型：{doc_type}" if doc_type else "未知"
        agent_output = data_result.content if data_result.success else "提取失败"
        today = datetime.now().strftime("%Y年%m月%d日" if lang == "zh" else "%B %d, %Y")

        if web_search_enabled and web_results:
            # ── 有预搜索结果：直接注入，不重复搜索 ──
            web_text_parts = []
            for i, wr in enumerate(web_results[:3], 1):
                snippet = wr.snippet[:800] if wr.snippet else ""
                title = wr.title[:120] if wr.title else ""
                url = wr.url[:200] if wr.url else ""
                if snippet:
                    web_text_parts.append(
                        f"### 搜索结果{i}: {title}\n来源: {url}\n内容: {snippet}"
                    )
            web_text = "\n\n".join(web_text_parts) if web_text_parts else "（未获取到搜索结果）"

            if lang == "zh":
                content = f"""今天是{today}。

用户想知道：{user_query}

## 文档中提取到的信息
{agent_output}

## 联网搜索结果（来自搜索引擎的真实网页内容）
{web_text}

请从以上信息中提取答案。严格按以下规则：
1. **第一句话直接给出具体数字和单位**，不要任何铺垫
2. **从搜索结果中直接复制数字**——不要修改、不要换算、不要凭记忆补充。搜索结果里有194.96亿就写194.96亿，不要改
3. 同时列出文档数据和网络数据，标注来源
4. 如果搜索结果中的数字互不一致，列出所有版本并标注哪个来源
5. 控制在10行以内"""
            else:
                content = f"""Today is {today}.

Question: {user_query}

## Data from Document
{agent_output}

## Web Search Results (real search engine content)
{web_text}

Extract the answer from the information above. Strict rules:
1. **First sentence: specific numbers with units. No preamble.**
2. **Copy numbers directly from search results** — do not modify, convert, or supplement from memory. If results say 19.496 billion, write 19.496 billion.
3. List both document data and web data, cite sources
4. If numbers differ across sources, list ALL versions with attribution
5. Keep under 10 lines."""

            messages = [{"role": "user", "content": content}]
            # 使用更强模型做综合
            if on_token:
                return self.llm.chat_stream(messages, on_token=on_token, model="qwen-max")
            return self.llm.chat(messages, model="qwen-max")

        elif web_search_enabled:
            # ── 搜索未返回结果（DDGS 失败），只用文档数据 ──
            if lang == "zh":
                content = f"""今天是{today}。

用户想知道：{user_query}

上传文档：{doc_info}
文档中提取到的信息：{agent_output}

注意：联网搜索暂时不可用，请仅基于文档数据回答。
规则：
1. 第一句话直接给答案（具体数字+单位）
2. 文档有就给数据+页码，文档没有就说未披露
3. 控制在10行以内，不展开分析"""
            else:
                content = f"""Today is {today}.

Question: {user_query}

Document: {doc_info}
Data from document: {agent_output}

Note: Web search is temporarily unavailable. Answer based on document data only.
Rules:
1. First sentence: direct answer with numbers. No preamble.
2. If in document, cite page. If not, state not disclosed.
3. Keep under 10 lines, no analysis."""

            messages = [{"role": "user", "content": content}]
            if on_token:
                return self.llm.chat_stream(messages, on_token=on_token)
            return self.llm.chat(messages)

        else:
            # ── 无联网：只用文档数据 ──
            if lang == "zh":
                content = f"""用户问题：{user_query}
文档类型：{doc_info}
文档数据：{agent_output}

直接回答。文档有就给数据+页码，文档没有就说未披露。8行以内。"""
            else:
                content = f"""Question: {user_query}
Document: {doc_info}
Data: {agent_output}

Answer directly. If in document, cite page. If not, state not disclosed. Under 8 lines."""

            messages = [{"role": "user", "content": content}]
            if on_token:
                return self.llm.chat_stream(messages, on_token=on_token)
            return self.llm.chat(messages)

    # ── 回应质疑 ──

    def _collect_rebuttals(self, data_result, risk_result, compliance_result,
                           devil_result, vector_store, api_key, enable_search):
        """让第一轮 Agent 简短回应深度质疑的挑战。"""
        rebuttals = {}
        devil_content = devil_result.content if devil_result.success else ""

        agents_to_rebut = [
            (self.data_extractor, data_result, "data_extractor"),
            (self.risk_assessor, risk_result, "risk_assessor"),
            (self.compliance, compliance_result, "compliance_checker"),
        ]

        lang = self.language
        for agent, original_result, key in agents_to_rebut:
            if not original_result.success:
                continue
            prompt = (
                f"深度质疑Agent对你的分析提出了以下质疑。请简短回应（不超过200字）：\n\n"
                f"## 你的原始分析\n{original_result.content[:1500]}\n\n"
                f"## 深度质疑Agent的质疑\n{devil_content[:1200]}\n\n"
                f"请回应：质疑是否合理？你的分析需要调整吗？如果质疑有道理，补充什么信息可以解决？"
                if lang == "zh" else
                f"The Devil's Advocate raised the following challenges to your analysis. Respond briefly (under 150 words):\n\n"
                f"## Your Analysis\n{original_result.content[:1500]}\n\n"
                f"## Devil's Advocate Challenges\n{devil_content[:1200]}\n\n"
                f"Respond: Is the challenge valid? Does your analysis need adjustment?"
            )
            try:
                resp = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    model="qwen-turbo", temperature=0.2, max_tokens=400,
                )
                rebuttals[key] = AgentResult(agent_name=agent.name, content=resp)
                self.reporter.update(agent.name, "done",
                                     "已回应质疑" if lang == "zh" else "Rebuttal complete")
            except Exception as e:
                rebuttals[key] = AgentResult(agent_name=agent.name,
                                             success=False, error=str(e))
        return rebuttals

    def _synthesize_with_rebuttals(self, user_query, data_result, risk_result,
                                   compliance_result, devil_result, rebuttals,
                                   on_token=None):
        """综合所有 Agent 输出 + 质疑 + 回应，生成最终报告。"""
        lang = self.language

        rebuttal_text = ""
        for key, result in rebuttals.items():
            if result.success:
                rebuttal_text += f"\n### {key} 对质疑的回应\n{result.content}\n"

        content = f"""## 用户原始问题
{user_query}

## 📊 数据提取 Agent 输出
{data_result.content if data_result.success else f'[执行失败: {data_result.error}]'}

## ⚠️ 风险评估 Agent 输出
{risk_result.content if risk_result.success else f'[执行失败: {risk_result.error}]'}

## 📋 合规审查 Agent 输出
{compliance_result.content if compliance_result.success else f'[执行失败: {compliance_result.error}]'}

## 🔍 深度质疑 Agent 输出
{devil_result.content if devil_result.success else f'[执行失败: {devil_result.error}]'}

## 🔄 Agent 对质疑的回应
{rebuttal_text if rebuttal_text else '（无回应）'}

请综合以上所有内容——原始分析、质疑、回应——生成最终报告。
重要：
- 如果质疑被采纳，报告中应体现调整后的结论
- 如果质疑被驳回，说明为什么
- 回应中的补充信息也应纳入报告
- 每个事实标注出处，区分事实与推理"""
        messages = [
            {"role": "system", "content": self.synthesis_prompt},
            {"role": "user", "content": content},
        ]
        if on_token:
            return self.llm.chat_stream(messages, on_token=on_token)
        return self.llm.chat(messages)

    def _log_agent_result(self, result: AgentResult):
        status = "done" if result.success else "error"
        self.reporter.update(result.agent_name, status,
                             result.content if result.success else result.error)

    @staticmethod
    def _safe_result(result: AgentResult) -> str:
        return result.content if result.success else f"[执行失败] {result.error}"
