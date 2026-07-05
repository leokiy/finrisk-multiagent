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
import json
import sys
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
        """完整执行多 Agent 协作工作流。

        新架构：
        0. 扫描文档 → 文档简报（公司名/代码/行业/报告期）
        1. 基于简报 + 用户原始问题 → 各Agent专属搜索词
        2. 并行执行搜索 → 搜索结果注入各Agent
        3. Agent 拿到：用户原始问题 + 文档简报 + 自己的RAG检索 + 自己的搜索结果
        4. 深度质疑 → 回应 → 综合报告
        """
        self.reporter.clear()
        lang = self.language
        self.reporter.update("Orchestrator", "done",
                             "=== NEW v2.0 (doc_brief + ddgs) ===")

        # ═══════════════════════════════════════════════════════════════
        # 第零轮：扫描文档 → 文档简报
        # ═══════════════════════════════════════════════════════════════
        self.reporter.update(
            "Orchestrator", "running",
            ("正在分析文档结构..."
             if lang == "zh" else "Analyzing document structure...")
        )
        doc_brief = self._scan_document_brief(vector_store, api_key, doc_type)
        question_type = self._classify_question(user_query)

        # 把关键信息写入执行日志（用户可在 Streamlit 执行日志 tab 看到）
        brief_lines = [l.strip() for l in doc_brief.split("\n") if l.strip() and "：" in l]
        brief_summary = " | ".join(l[:60] for l in brief_lines[:4])
        self.reporter.update(
            "Orchestrator", "done",
            (f"文档简报: {brief_summary}" if lang == "zh" and brief_summary
             else f"Doc brief: {brief_summary}" if brief_summary
             else ("文档分析完成" if lang == "zh" else "Document analysis done"))
        )
        self.reporter.update(
            "Orchestrator", "info",
            f"question_type={question_type} | web_search={web_search_enabled}"
        )

        # ═══════════════════════════════════════════════════════════════
        # 联网搜索：基于文档简报 + 用户原始问题 → 精准搜索词 → 并行搜索
        # ═══════════════════════════════════════════════════════════════
        if web_search_enabled:
            agent_search_queries = self._generate_agent_search_queries(
                user_query, doc_brief, question_type
            )
            # 写搜索词到执行日志
            sq_summary = {}
            for k, v in agent_search_queries.items():
                sq_summary[k] = v
            self.reporter.update(
                "Orchestrator", "running",
                ("搜索词: " + json.dumps(sq_summary, ensure_ascii=False)[:300]
                 if lang == "zh" else
                 "Search queries: " + json.dumps(sq_summary, ensure_ascii=False)[:300])
            )

            agent_web_results = self._execute_agent_searches(
                agent_search_queries, api_key
            )
            # 写搜索结果摘要到执行日志
            for key, results in agent_web_results.items():
                n = len(results)
                sample = ""
                if results:
                    r = results[0]
                    sample = f" | sample: {r.title[:50]} | {r.snippet[:80]}"
                self.reporter.update(
                    "Orchestrator", "done" if n > 0 else "error",
                    f"{key}: {n} results{sample}"
                )
            searched_agents = [k for k, v in agent_web_results.items() if v]
            if not searched_agents:
                self.reporter.update(
                    "Orchestrator", "error",
                    ("联网搜索未返回任何结果" if lang == "zh" else "Web search returned no results")
                )
        else:
            agent_web_results = {}

        # ═══════════════════════════════════════════════════════════════
        # 根据问题类型自适应分析深度
        # ═══════════════════════════════════════════════════════════════
        is_factual = question_type == "factual"

        if is_factual:
            self.reporter.update(
                "Orchestrator", "running",
                ("数据查询模式：提取文档数据+联网搜索..."
                 if lang == "zh" and web_search_enabled else
                 "数据查询模式：提取文档数据..."
                 if lang == "zh" else
                 "Factual mode: extracting document + web data..."
                 if web_search_enabled else
                 "Factual mode: extracting document data...")
            )

            data_web_results = agent_web_results.get("data_extractor")
            data_result = self.data_extractor.run(
                user_query, vector_store, api_key=api_key,
                doc_brief=doc_brief,
                web_search_results=data_web_results,
            )
            self._log_agent_result(data_result)

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

        # ═══════════════════════════════════════════════════════════════
        # 第一轮：3 Agent 并行（各自拿到原始问题+文档简报+专属搜索结果）
        # ═══════════════════════════════════════════════════════════════
        self.reporter.update(
            "Orchestrator", "running",
            ("启动第一轮分析：各Agent独立检索文档并分析..."
             if lang == "zh" else
             "Round 1: Agents independently searching document and analyzing...")
        )

        round1_results = self._run_parallel_with_queries(
            [self.data_extractor, self.risk_assessor, self.compliance],
            user_query, vector_store, api_key,
            doc_brief=doc_brief,
            agent_web_results=agent_web_results,
        )

        data_result, risk_result, compliance_result = round1_results

        # ═══════════════════════════════════════════════════════════════
        # 第二轮：深度质疑
        # ═══════════════════════════════════════════════════════════════
        self.reporter.update(
            "Devil's Advocate", "running",
            ("正在审阅前三方分析结果，寻找盲点和矛盾..."
             if lang == "zh" else
             "Devil's Advocate reviewing all outputs for blind spots...")
        )

        other_context = {
            self.data_extractor.prompt_file: data_result.content if data_result.success else f"[Failed] {data_result.error}",
            self.risk_assessor.prompt_file: risk_result.content if risk_result.success else f"[Failed] {risk_result.error}",
            self.compliance.prompt_file: compliance_result.content if compliance_result.success else f"[Failed] {compliance_result.error}",
        }

        devil_result = self.devils_advocate.run(
            user_query, vector_store,
            context_from_other_agents=other_context,
            api_key=api_key,
            doc_brief=doc_brief,
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

    # ----------------------------------------------------------------
    # 内部 — 文档扫描
    # ----------------------------------------------------------------

    def _scan_document_brief(self, vector_store: VectorStore,
                             api_key: str, doc_type: str = "") -> str:
        """第零轮：快速扫描文档，提取关键实体信息。

        不分析、不判断——只提取客观事实：公司名、代码、行业、报告期等。
        产出"文档简报"，后续搜索词生成和Agent分析都基于这份简报。
        """
        lang = self.language

        # 取文档前部最有信息量的片段
        sample = vector_store.search(
            "公司 名称 证券 简称 股票 代码 行业 业务 主营 报告期 年度 股东",
            top_k=6, api_key=api_key
        )
        context = "\n\n".join(c.chunk.text[:500] for c in sample)[:4000]

        if lang == "zh":
            prompt = f"""从以下文档片段中提取关键信息。只输出事实，不要推测。找不到的写"未知"。

## 文档片段
{context}

## 提取以下信息：
- 主体公司全称：
- 股票代码（如有）：
- 所属行业：
- 主营业务（一句话）：
- 文档类型（年报/半年报/季报/招股书/募集说明书/其他）：
- 报告期（如"截至202X年X月X日"）：
- 文档中最新的财务数据属于哪个期间："""
        else:
            prompt = f"""Extract key information from the document excerpts below. Facts only, no speculation. Write "Unknown" if not found.

## Document Excerpts
{context}

## Extract:
- Entity full name:
- Stock ticker (if any):
- Industry:
- Main business (one sentence):
- Document type (Annual/Semi-annual/Quarterly/Prospectus/Offering/Other):
- Reporting period (e.g. "As of June 30, 202X"):
- Latest financial data period mentioned in the document:"""

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                model="qwen-turbo", temperature=0.0, max_tokens=300,
            )
            brief = resp.strip()
            if doc_type and doc_type not in brief:
                brief = f"文档类型：{doc_type}\n{brief}"
            return brief
        except Exception:
            return f"文档（未能自动提取信息）"

    @staticmethod
    def _classify_question(user_query: str) -> str:
        """快速分类用户问题类型，不调用 LLM。

        用关键词规则做粗分类，避免不必要的 API 调用。
        """
        q = user_query.strip()
        # 明显的数据查询
        data_keywords = [
            "多少", "是多少", "什么时候", "何时", "哪天",
            "how much", "how many", "what is", "when",
        ]
        if any(kw in q.lower() for kw in data_keywords):
            # 进一步判断是否只是简单查数据
            if len(q) < 30 or q.endswith("？") or q.endswith("?"):
                return "factual"

        # 全面评估请求
        comprehensive_keywords = [
            "全面", "综合", "完整", "所有", "全部", "整体",
            "风险评估", "分析报告", "做个分析",
            "comprehensive", "full", "complete", "overall",
            "risk assessment", "analyze everything",
        ]
        if any(kw in q.lower() for kw in comprehensive_keywords):
            return "comprehensive"

        # 默认：分析特定方面
        return "analytical"

    # ----------------------------------------------------------------
    # 内部 — 联网搜索查询生成
    # ----------------------------------------------------------------

    def _generate_agent_search_queries(self, user_query: str,
                                       doc_brief: str,
                                       question_type: str) -> dict[str, list[str]]:
        """基于文档简报 + 用户原始问题，为每个 Agent 生成精准搜索词。

        关键改进：现在有了文档简报（公司名/代码/行业/报告期），
        搜索词不再是瞎子摸象，而是有具体实体的定向搜索。

        每个 Agent 生成 2 条：信息查询 + 交叉验证。
        """
        lang = self.language

        if question_type == "factual":
            # 从文档简报中提取公司名和股票代码，拼入搜索词
            company_name = ""
            stock_code = ""
            for line in doc_brief.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if "公司全称" in line or "Entity" in line:
                    val = line.split("：")[-1].split(":")[-1].strip()
                    if val and val != "未知":
                        company_name = val
                if "股票代码" in line or "Stock" in line:
                    val = line.split("：")[-1].split(":")[-1].strip()
                    if val and val != "未知" and val != "无":
                        stock_code = val
            entity = stock_code or company_name
            if entity:
                return {"data_extractor": [
                    f"{entity} {user_query}",
                    f"{entity} 2026 一季度 财报 数据 核实",
                ]}
            return {"data_extractor": [user_query, f"{user_query} 2026 最新 数据"]}

        if lang == "zh":
            prompt = f"""你是金融信息检索专家。以下是用户问题、以及从用户上传的文档中提取的主体信息。请基于这些信息，为4个专业Agent各生成2条精准的联网搜索查询。

## 用户原始问题
{user_query}

## 文档主体信息
{doc_brief}

## 各Agent的搜索目标
- 数据提取Agent：
  信息查询：搜索该主体的最新财务数据、关键指标（使用文档简报中的公司名/代码）
  验证查询：用另一来源核实同一数据

- 风险评估Agent：
  信息查询：搜索该主体及行业的风险动态、负面新闻、市场变化
  验证查询：搜索相反观点，检验自己的风险判断是否偏颇

- 合规审查Agent：
  信息查询：搜索该主体及行业的监管政策变化、合规要求
  验证查询：搜索该主体是否有实际违规记录、处罚公告

- 深度质疑Agent：
  信息查询：搜索该主体的争议事件、做空报告、被忽视的风险信号
  验证查询：搜索公司对这些质疑的回应和澄清

## 要求
- **必须使用文档简报中的公司全称/股票代码（如有）构造查询**
- 每条查询10-30字，自然语言
- 严格按格式输出：
data_extractor_info: <查询>
data_extractor_verify: <查询>
risk_assessor_info: <查询>
risk_assessor_verify: <查询>
compliance_checker_info: <查询>
compliance_checker_verify: <查询>
devils_advocate_info: <查询>
devils_advocate_verify: <查询>"""
        else:
            prompt = f"""You are a financial information retrieval expert. Generate 2 search queries for each agent based on the user's question and document entity information.

## User Question
{user_query}

## Document Entity Info
{doc_brief}

## Agent Search Targets
(same structure as Chinese version)

## Requirements
- **Must use entity names/tickers from the document brief**
- 10-20 words per query
- Output format:
data_extractor_info: <query>
data_extractor_verify: <query>
risk_assessor_info: <query>
risk_assessor_verify: <query>
compliance_checker_info: <query>
compliance_checker_verify: <query>
devils_advocate_info: <query>
devils_advocate_verify: <query>"""

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
            return {k: [user_query] for k in key_map.values()}

        queries: dict[str, list[str]] = {}
        for line in resp.strip().split("\n"):
            line = line.strip()
            for prefix, key in key_map.items():
                if line.lower().startswith(prefix.lower()):
                    q = line.split(":", 1)[-1].strip().strip('"').strip("'")
                    if q and len(q) > 2:
                        queries.setdefault(key, []).append(q)

        for key in set(key_map.values()):
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

    def _run_parallel_with_queries(self, agents: list,
                                    user_query: str,
                                    store: VectorStore, api_key: str,
                                    doc_brief: str = "",
                                    agent_web_results: dict | None = None
                                    ) -> list[AgentResult]:
        """并行执行多个 Agent。

        每个 Agent 拿到：
        - 用户原始问题（不转义）
        - 文档简报（公司名/代码/行业/报告期）
        - 自己专属的联网搜索结果
        - 向量库（Agent 自己搜文档）
        """
        results: list[AgentResult] = []
        agent_key_map = {
            self.data_extractor: "data_extractor",
            self.risk_assessor: "risk_assessor",
            self.compliance: "compliance_checker",
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for agent in agents:
                agent_key = agent_key_map.get(agent, "")
                web_results = (agent_web_results or {}).get(agent_key)
                futures[
                    executor.submit(
                        agent.run, user_query, store, None, 20, api_key,
                        doc_brief, web_results
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

    def _run_parallel(self, agents: list, user_query: str, store: VectorStore,
                      api_key: str, doc_brief: str = "",
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
                        agent.run, user_query, store, None, 20, api_key,
                        doc_brief, web_results
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
2. **从搜索结果中直接复制数字，区分清楚每个指标**——营收是营收、净利润是净利润、扣非净利润是扣非净利润，不要混
3. 如果用户问"利润"/"净利润"但搜索结果同时有营收和净利，只答净利润，不要答营收
4. 同时列出文档数据和网络数据，标注来源
5. 如果搜索结果中的数字互不一致，列出所有版本
6. 控制在10行以内"""
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
