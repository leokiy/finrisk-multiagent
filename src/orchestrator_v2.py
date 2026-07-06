"""
🧠 Orchestrator V2 — LLM 驱动的协调 Agent。

不再是硬编码的流水线。协调 Agent 像人一样工作：
  1. 理解问题 → 拆解需要什么信息
  2. 决定找谁 → 只派相关的 Agent，不全部跑
  3. 评估结果 → 够了吗？缺什么？
  4. 不够就补 → 告诉 Agent 具体查什么、怎么查
  5. 够了就综合 → 裁决矛盾、输出结论

这是 ReAct (Reasoning + Acting) 模式：
  Think → Act → Observe → Think → ... → Synthesize
"""

import concurrent.futures
import json
import time
from typing import Optional

from src.llm.client import LLMClient, LLMConfig
from src.rag.engine import VectorStore
from src.agents.base import AgentResult, _load_prompt
from src.agents.data_extractor import DataExtractorAgent
from src.agents.risk_assessor import RiskAssessorAgent
from src.agents.compliance_checker import ComplianceCheckerAgent
from src.agents.devils_advocate import DevilsAdvocateAgent


# ═══════════════════════════════════════════════════════════════
# 协调 Agent 的 System Prompt
# ═══════════════════════════════════════════════════════════════

COORDINATOR_PROMPT_ZH = """# 🎯 你是金融文档分析系统的总协调人

你像一个投资决策委员会的主席，手下有 4 个专业分析师和一个资料室。

## 你的工具箱

你可以调用以下工具完成任务。不是每次都要全用——根据问题按需选择：

### search_document(query)
在用户上传的 PDF 文档中搜索相关内容。返回最相关的段落。
什么时候用：问题可能在文档中有答案时。用精准关键词搜索。

### search_web(query)
联网搜索最新信息。返回网页结果（标题+链接+摘要）。
什么时候用：文档里没有的信息、需要验证的数据、行业对比、最新动态。

### run_analyst(analyst_name, instruction)
派指定分析师去完成任务。可用的分析师：
- "data_extractor": 从文档和搜索结果中提取具体数据。适合查数字、指标。
- "risk_assessor": 从市场/信用/流动/治理四维度评估风险。适合分析类问题。
- "compliance_checker": 对照法规框架审查合规问题。适合合规类问题。
- "devils_advocate": 挑战已有结论、寻找盲点和矛盾。适合需要质疑时。

每个 analyst 需要一句明确的 instruction——告诉它具体做什么、关注什么。
可以一次派多个 analyst（它们会并行工作）。

### synthesize()
当你认为已经收集了足够的信息来回答用户的问题时，调用此工具。你会输出最终的综合报告。

---

## 工作方式

对每个用户问题，按以下循环工作：

1. **Think（思考）**: 用户到底想知道什么？要回答它，我需要哪些信息？
2. **Act（行动）**: 选择合适的工具，告诉它具体做什么
3. **Observe（观察）**: 看返回的结果——信息够了吗？还缺什么？
4. **Repeat or Synthesize**: 如果够了 → synthesize。不够 → 回到 Think

## 你的角色——不仅仅是调度员

你是协调者，不是传话筒。你的职责：
1. **审阅专家输出**——专家说了什么？有矛盾吗？缺什么？
2. **指导专家**——告诉专家"你上次忽略了X，这次重点关注X"或"你的结论和网络数据矛盾，重新评估"
3. **按需搜索**——专家说需要什么就搜什么，不要替专家猜
4. **判断时机**——信息够了就综合，不够就继续

## 行动原则

- **先分析、再搜索**。先 run_analyst 让专家看文档→专家告诉你缺什么→你搜→让专家看搜索结果重新分析
- 不要把所有专家都跑一遍。只跑回答问题相关的专家。
- 如果用户只问一个数字，search_document + search_web 就够了
- 每次行动前想清楚：我为什么要做这一步？
- 每轮用简体中文输出 JSON 格式的决策。

---

## 输出格式

每轮只输出一个 JSON 对象，不要额外文字：

```json
{
  "thought": "我对问题的理解和当前思考",
  "plan": "我打算做什么、为什么",
  "actions": [
    {"tool": "search_document", "query": "..."},
    {"tool": "search_web", "query": "..."},
    {"tool": "run_analyst", "analyst": "data_extractor", "instruction": "..."}
  ],
  "need_more": true
}
```

当信息足够时：
```json
{
  "thought": "已收集足够信息",
  "need_more": false
}
```

**规则：**
- actions 可以包含多个工具，它们会并行执行
- 先 search_document/search_web，拿到结果后再决定是否需要 run_analyst
- 不要重复执行相同或相似的查询
- need_more: false 表示可以进入 synthesize 阶段"""

COORDINATOR_PROMPT_EN = """# 🎯 You are the Chief Coordinator of a financial document analysis system.

You are like the chair of an investment committee with 4 specialist analysts and a document library.

## Your Toolbox

### search_document(query)
Search the uploaded PDF. Returns most relevant passages.
Use: When the answer might be in the document.

### search_web(query)
Search the web for latest information. Returns web results.
Use: When document lacks data, needs verification, industry comparison.

### run_analyst(analyst_name, instruction)
Dispatch an analyst: "data_extractor", "risk_assessor", "compliance_checker", "devils_advocate".
Give each a clear instruction. Multiple analysts run in parallel.

### synthesize()
Call when you have enough information. Output the final report.

---

## How You Work

Think -> Act -> Observe -> Repeat or Synthesize.

Key principles:
- Don't blindly run all 4 analysts. Only dispatch relevant ones.
- For simple data queries: search_document + search_web is enough.
- For analytical questions: search first, then run analysts if needed.
- Each action should have a clear reason.

## Output Format

Each round, output ONE JSON object:
```json
{
  "thought": "my understanding and reasoning",
  "plan": "what I plan to do and why",
  "actions": [{"tool": "...", ...}],
  "need_more": true
}
```

When done: `{"thought": "...", "need_more": false}`"""


def _format_web_for_agent(web_items: list[dict]) -> list | None:
    """把累积的 web 结果格式化为 Agent 能接受的 WebResult 列表。"""
    if not web_items:
        return None
    from src.search.web_search import WebResult
    out = []
    for item in web_items[-10:]:  # 最近 10 条
        out.append(WebResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
        ))
    return out if out else None


# ═══════════════════════════════════════════════════════════════
# Orchestrator V2
# ═══════════════════════════════════════════════════════════════

class OrchestratorV2:
    """LLM 驱动的协调 Agent。"""

    def __init__(self, llm_client: LLMClient, language: str = "zh"):
        self.llm = llm_client
        self.language = language
        self._reporter_logs: list[dict] = []

        # 初始化所有 Agent
        self.data_extractor  = DataExtractorAgent(llm_client, language)
        self.risk_assessor   = RiskAssessorAgent(llm_client, language)
        self.compliance      = ComplianceCheckerAgent(llm_client, language)
        self.devils_advocate = DevilsAdvocateAgent(llm_client, language)

        self.coordinator_prompt = (
            COORDINATOR_PROMPT_ZH if language == "zh" else COORDINATOR_PROMPT_EN
        )
        self.synthesis_prompt = _load_prompt("orchestrator", language)

    # ------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------

    def run(self, user_query: str, vector_store: VectorStore,
            api_key: str = "", on_token=None,
            web_search_enabled: bool = True,
            doc_type: str = "",
            max_rounds: int = 8) -> dict:
        """ReAct 决策循环。"""
        self._reporter_logs = []
        if api_key:
            import dashscope
            dashscope.api_key = api_key

        lang = self.language

        # ── 对话历史 ──
        from datetime import datetime
        today = datetime.now().strftime("%Y年%m月%d日" if lang == "zh" else "%B %d, %Y")
        system_with_date = self.coordinator_prompt + (
            f"\n\n**当前日期: {today}**。你是在{today}进行分析。不要依赖训练数据中的日期——"
            f"今天是真实日期，所有财务数据可能已经发布。"
            if lang == "zh" else
            f"\n\n**Current date: {today}**. You are analyzing on {today}. "
            f"Don't rely on training data dates - use the real current date."
        )

        conversation = [{
            "role": "system",
            "content": system_with_date
        }, {
            "role": "user",
            "content": (f"用户上传了一份金融文档（{doc_type or '未知类型'}），"
                       f"你的任务是回答：{user_query}\n\n请开始分析。先思考你需要什么信息。")
            if lang == "zh" else
            f"Document: {doc_type or 'Unknown'}. Question: {user_query}. Start analyzing."
        }]

        # ── 累积的所有发现 ──
        all_findings: list[str] = []
        searched_queries: set[str] = set()
        accumulated_web: list[dict] = []  # 跨轮累积的网络搜索结果

        for round_num in range(1, max_rounds + 1):
            self._log("Coordinator", "running",
                      f"Round {round_num}/{max_rounds}" if lang == "en"
                      else f"第{round_num}轮思考")

            # 1. 让 LLM 思考下一步
            decision = self._think(conversation)

            if decision is None:
                self._log("Coordinator", "error",
                          "Decision parse failed" if lang == "en" else "决策解析失败")
                break

            self._log("Coordinator", "done",
                      decision.get("thought", "")[:150])

            # 2. 先执行 actions（收集数据），再判断是否结束
            actions = decision.get("actions", [])
            if actions:
                results = self._execute_actions(
                    actions, user_query, vector_store, api_key,
                    web_search_enabled, searched_queries, accumulated_web,
                    all_findings
                )
                observation = self._format_observations(results, lang)
                all_findings.append(observation)
                conversation.append({
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=False)
                })
                conversation.append({
                    "role": "user",
                    "content": f"<observation>\n{observation}\n</observation>\n\n请基于以上观察继续思考。信息够了吗？不够还需要什么？"
                })

            # 3. 判断是否结束（在 actions 执行之后）
            if not decision.get("need_more", True):
                self._log("Coordinator", "done",
                          "Information sufficient, synthesizing" if lang == "en"
                          else "信息足够，开始综合")
                break

            if not actions:
                self._log("Coordinator", "done",
                          "No actions, synthesizing" if lang == "en" else "无行动，综合")
                break

        # ── 综合 ──
        self._log("Coordinator", "running",
                  "Synthesizing final report..." if lang == "en" else "综合最终报告...")

        final_report = self._synthesize(
            user_query, all_findings, vector_store, api_key, on_token
        )

        self._log("Coordinator", "done",
                  f"Analysis complete ({round_num} rounds)" if lang == "en"
                  else f"分析完成（{round_num}轮）")

        # 生成追问
        followup = self._generate_followup_questions(user_query, final_report)

        return {
            "query": user_query,
            "final_report": final_report,
            "followup_questions": followup,
            "execution_log": self._reporter_logs,
            "rounds": round_num,
        }

    # ------------------------------------------------------------
    # 内部 — Think
    # ------------------------------------------------------------

    def _think(self, conversation: list[dict]) -> dict | None:
        """让 LLM 思考下一步行动，返回决策 JSON。"""
        try:
            resp = self.llm.chat(
                conversation,
                model="qwen-plus",
                temperature=0.2,
                max_tokens=800,
            )
        except Exception as e:
            self._log("Coordinator", "error", f"Think failed: {e}")
            return None

        return self._parse_json(resp)

    def _parse_json(self, text: str) -> dict | None:
        """从 LLM 回复中提取 JSON。"""
        text = text.strip()
        # 去掉可能的 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
            if text.endswith("```"):
                text = text[:-3]

        # 找 JSON 边界
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------
    # 内部 — Act
    # ------------------------------------------------------------

    def _execute_actions(self, actions: list[dict], user_query: str,
                         vector_store: VectorStore, api_key: str,
                         web_search_enabled: bool,
                         searched_queries: set[str],
                         accumulated_web: list[dict],
                         all_findings: list[str]) -> dict[str, list]:
        """执行一批 actions，并行跑。"""
        results: dict[str, list] = {}

        def _do(action: dict):
            tool = action.get("tool", "")
            key = tool
            try:
                if tool == "search_document":
                    q = action.get("query", user_query)
                    key = f"search_doc:{q[:60]}"
                    if q in searched_queries:
                        return key, [{"type": "cached", "text": "(重复查询，跳过)"}]
                    searched_queries.add(q)
                    chunks = self._search_document(q, vector_store, api_key)
                    return key, chunks

                elif tool == "search_web":
                    if not web_search_enabled:
                        return key, [{"type": "disabled", "text": "联网搜索未启用"}]
                    q = action.get("query", user_query)
                    key = f"search_web:{q[:60]}"
                    if q in searched_queries:
                        return key, [{"type": "cached", "text": "(重复查询，跳过)"}]
                    searched_queries.add(q)
                    web_results = self._search_web(q, api_key)
                    accumulated_web.extend(web_results)  # 累积
                    return key, web_results

                elif tool == "run_analyst":
                    name = action.get("analyst", "data_extractor")
                    instr = action.get("instruction", user_query)
                    key = f"analyst:{name}"
                    # 把累积的网络搜索结果 + 之前所有发现传给 Agent
                    web_for_agent = _format_web_for_agent(accumulated_web)
                    # 构建 Context：之前发现了什么、还需要关注什么
                    ctx = action.get("context", "") or ""
                    if not ctx and len(all_findings) > 0:
                        ctx = "此前分析中已收集的信息：\n" + "\n".join(all_findings[-3:])[:2000]
                    full_instr = instr
                    if ctx:
                        full_instr = f"{instr}\n\n## 参考背景（此前分析中发现的信息）\n{ctx}"
                    agent_result = self._run_analyst(
                        name, full_instr, vector_store, api_key, web_for_agent
                    )
                    return key, [agent_result]

                else:
                    return key, [{"type": "error", "text": f"未知工具: {tool}"}]
            except Exception as e:
                return key, [{"type": "error", "text": str(e)}]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_do, a): a for a in actions}
            for future in concurrent.futures.as_completed(futures):
                key, items = future.result()
                results.setdefault(key, []).extend(items)

        return results

    # ── 具体工具实现 ──

    def _search_document(self, query: str, store: VectorStore,
                         api_key: str) -> list[dict]:
        """搜索文档。"""
        if store.is_empty:
            return [{"type": "empty", "text": "文档为空"}]
        try:
            extra = []
            if api_key:
                from src.rag.engine import rewrite_query_for_rag
                extra = rewrite_query_for_rag(query, api_key, self.language)
            results = store.search(query, top_k=5, extra_queries=extra, api_key=api_key)
            return [{
                "type": "doc_chunk",
                "page": r.chunk.page,
                "score": round(r.score, 3),
                "text": r.chunk.text[:500]
            } for r in results]
        except Exception as e:
            return [{"type": "error", "text": str(e)}]

    def _search_web(self, query: str, api_key: str) -> list[dict]:
        """联网搜索。"""
        from src.search.web_search import search_web, WebResult
        try:
            results = search_web(query, api_key=api_key, max_results=3,
                                language=self.language)
            out = []
            for r in results:
                out.append({
                    "type": "web_result",
                    "title": r.title[:120],
                    "url": r.url[:300] if r.url else "",
                    "snippet": r.snippet[:500],
                })

            # 也跑 enable_search 获取更详实的内容
            from src.llm.client import LLMConfig
            prompt = f"搜索以下内容，返回具体数据：{query}"
            config = LLMConfig(api_key=api_key, model="qwen-plus",
                              temperature=0.1, max_tokens=1000)
            client = LLMClient(config)
            resp = client.chat([{"role": "user", "content": prompt}],
                              enable_search=True)
            if resp and len(resp.strip()) > 20:
                out.append({
                    "type": "ai_search",
                    "title": f"AI搜索: {query[:60]}",
                    "url": "",
                    "snippet": resp.strip()[:800],
                })

            return out if out else [{"type": "empty", "text": "搜索无结果"}]
        except Exception as e:
            return [{"type": "error", "text": str(e)}]

    def _run_analyst(self, name: str, instruction: str,
                     vector_store: VectorStore, api_key: str,
                     web_results: list | None = None) -> dict:
        """运行指定分析师，传入累积的网络搜索结果。"""
        agents = {
            "data_extractor": self.data_extractor,
            "risk_assessor": self.risk_assessor,
            "compliance_checker": self.compliance,
            "devils_advocate": self.devils_advocate,
        }
        agent = agents.get(name)
        if not agent:
            return {"type": "error", "text": f"未知分析师: {name}"}

        # 把累积的网络搜索结果传给 Agent
        result = agent.run(instruction, vector_store, api_key=api_key,
                          web_search_results=web_results)
        self._log(agent.name,
                  "done" if result.success else "error",
                  result.content[:200] if result.success else result.error)

        return {
            "type": "analyst_output",
            "analyst": name,
            "success": result.success,
            "content": result.content[:3000] if result.success else result.error,
        }

    # ------------------------------------------------------------
    # 内部 — Observe
    # ------------------------------------------------------------

    def _format_observations(self, results: dict[str, list],
                             lang: str) -> str:
        """把工具执行结果格式化为 LLM 可读的观察文本。"""
        parts = []
        for key, items in results.items():
            parts.append(f"\n### {key}")
            for item in items:
                t = item.get("type", "?")
                if t == "doc_chunk":
                    parts.append(
                        f"[文档 p{item['page']}, score={item['score']}] {item['text'][:300]}"
                    )
                elif t in ("web_result", "ai_search"):
                    url = f" ({item.get('url', '')})" if item.get('url') else ""
                    parts.append(f"[{item.get('title', '')}{url}] {item['snippet'][:500]}")
                elif t == "analyst_output":
                    status = "✓" if item.get("success") else "✗"
                    name = item.get("analyst", "?")
                    parts.append(f"[{name} {status}] {item['content'][:500]}")
                else:
                    parts.append(f"[{t}] {item.get('text', str(item))[:200]}")
        return "\n".join(parts)

    # ------------------------------------------------------------
    # 内部 — Synthesize
    # ------------------------------------------------------------

    def _synthesize(self, user_query: str, findings: list[str],
                    vector_store: VectorStore, api_key: str,
                    on_token=None) -> str:
        """综合所有发现，生成最终报告。"""
        lang = self.language

        # 也做一次最后的文档检索，确保没有遗漏
        final_doc = ""
        if not vector_store.is_empty:
            results = vector_store.search(user_query, top_k=3, api_key=api_key)
            if results:
                final_doc = "\n".join(
                    f"[p{r.chunk.page}] {r.chunk.text[:300]}" for r in results[:3]
                )

        findings_text = "\n\n---\n\n".join(findings[-8:])  # 最近 8 轮发现
        findings_text = findings_text[:6000]

        # 强制提示：优先采信网络数据
        priority_hint = (
            "\n\n**重要：上述信息中如果网络搜索到的数据与文档数据时间不同，"
            "以网络搜索的最新数据为准。文档只有2025年数据，网络有2026年数据时，答2026年的。**"
            if lang == "zh" else
            "\n\n**IMPORTANT: If web search found newer data than the document, use the web data.**"
        )

        content = f"""## 用户问题
{user_query}

## 文档关键段落
{final_doc}

## 分析过程中收集的信息（包含网络搜索和Agent分析结果）
{findings_text}
{priority_hint}

请综合以上所有信息，生成最终报告。

铁律：
- 第一句话直接回答用户问题
- **收集到的信息中如果有具体数字就直接用**——coordinator已经搜到了就不要再搜
- 网络搜索到的数据优先采信，尤其是年代更新的数据
- 每条事实标注来源：文档标页码，网络数据标媒体名+日期
- 禁止把2025年数据当成2026年数据来回答"""

        messages = [
            {"role": "system", "content": self.synthesis_prompt},
            {"role": "user", "content": content},
        ]

        # 不传 enable_search——coordinator 已经搜索够了，合成时只综合已有信息
        # 传 enable_search 反而会让模型忽略传入的 findings 去搜自己的
        if on_token:
            return self.llm.chat_stream(messages, on_token=on_token, model="qwen-max")
        return self.llm.chat(messages, model="qwen-max")

    # ------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------

    def _generate_followup_questions(self, user_query: str,
                                      analysis_summary: str,
                                      count: int = 4) -> list[str]:
        """基于分析结果生成追问。"""
        lang = self.language
        summary = analysis_summary[:1500] if analysis_summary else ""
        if not summary.strip():
            return []

        if lang == "zh":
            prompt = f"""基于以下对话上下文，生成{count}个值得继续追问的问题。
用户刚才问了：{user_query}
系统的分析结论（摘要）：{summary}
要求：紧跟分析中的关键发现、覆盖不同角度、每个不超过20字、直接输出列表每行一个以\"- \"开头"""
        else:
            prompt = f"""Generate {count} follow-up questions based on:
Q: {user_query}
A: {summary}
Under 12 words each, output as list starting with \"- \""""

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                model="qwen-turbo", temperature=0.4, max_tokens=300,
            )
        except Exception:
            return []

        questions = []
        for line in resp.strip().split("\n"):
            line = line.strip().lstrip("- ").lstrip("0123456789. ").strip()
            if line and len(line) > 3:
                questions.append(line)
        return questions[:count]

    def _log(self, agent: str, status: str, content: str = ""):
        self._reporter_logs.append({
            "agent": agent, "status": status, "content": content
        })
