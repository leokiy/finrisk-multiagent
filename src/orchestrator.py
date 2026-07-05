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
            web_search_enabled: bool = False) -> dict:
        """完整执行多 Agent 协作工作流，返回结构化的最终报告。

        改进版流程:
          0. 为每个 Agent 生成针对其角色的专属转述问题
          1. 第一轮：3 Agent 并行分析（各自用自己的转述问题）
          2. 第二轮：深度质疑 Agent 挑战前三方结论
          3. 第三轮：第一轮 Agent 回应质疑（简短二次分析）
          4. 第四轮：协调 Agent 综合所有输出 + 质疑 + 回应 → 最终报告
        """
        self.reporter.clear()
        lang = self.language

        # ── 第零轮：为每个 Agent 生成专属转述 ──
        msg0 = ("正在理解你的问题，为各 Agent 制定分析策略..."
                if lang == "zh" else
                "Understanding your question, tailoring analysis for each agent...")
        self.reporter.update("Orchestrator", "running", msg0)

        queries = self._reformulate_per_agent(user_query)

        # ── 第一轮：3 Agent 并行，各自用自己的转述问题 ──
        msg = ("启动第一轮分析：数据提取、风险评估、合规审查并行执行..."
               if lang == "zh" else
               "Round 1: Agents analyzing with role-specific instructions...")
        if web_search_enabled:
            msg += ("（联网搜索已启用）" if lang == "zh" else " (web search enabled)")
        self.reporter.update("Orchestrator", "running", msg)

        round1_results = self._run_parallel_with_queries(
            [self.data_extractor, self.risk_assessor, self.compliance],
            queries, vector_store, api_key,
            enable_search=web_search_enabled,
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
        )
        self._log_agent_result(devil_result)

        # ── 第三轮：第一轮 Agent 回应质疑 ──
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

        # ── 第四轮：协调 Agent 综合所有输出（流式）──
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

    def _reformulate_per_agent(self, user_query: str) -> dict[str, str]:
        """为每个 Agent 生成针对其角色的专属转述问题。"""
        lang = self.language

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
        return queries

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
                                    enable_search: bool = False) -> list[AgentResult]:
        """并行执行多个 Agent，各自使用专属转述问题。"""
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
                agent_query = queries.get(agent_key_map.get(agent, ""), "")
                if not agent_query:
                    agent_query = list(queries.values())[0] if queries else ""
                futures[
                    executor.submit(agent.run, agent_query, store, None, 20,
                                   api_key, enable_search)
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
                      api_key: str, enable_search: bool = False) -> list[AgentResult]:
        """并行执行多个 Agent（使用相同 query，兼容旧调用）。"""
        results: list[AgentResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(agent.run, query, store, None, 20, api_key,
                               enable_search): agent
                for agent in agents
            }
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
