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

## 行动原则——按问题复杂度选择策略

**第一轮必须先判断问题类型，选对策略：**

### 简单查询（查数字、查事实）
用户问"X是多少""X什么时候"等具体数据。
→ **你自己搜+答就行，不要派专家。**
Round 1: search_document + search_web
Round 2: 信息够了就 need_more: false

### 专项分析（分析某方面、评估风险、判断合理性）
用户问"怎么样""为什么""是否合理""有什么风险"。
→ **先搜文档和网络，再根据需要派专家。**
Round 1: search_document + search_web
Round 2: 看搜索结果——数据够就直接答，不够才 run_analyst
后续: 专家输出有 [NEED_MORE] 就补搜→回传→再判断

### 全面评估（用户明确要求"全面""综合""完整"评估）
→ **可以派多个专家并行，配合魔鬼代言人。**
但也不要一开始就把4个都派出去。先搜→再决定派谁。

**通用原则：**
- 永远不要第一轮就 run_analyst。先自己搜了再说。
- 能自己搜到答案的就不要派专家。
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
            max_rounds: int = 8,
            on_progress=None,
            chat_history: list[dict] | None = None) -> dict:
        """ReAct 决策循环。

        chat_history: 之前的对话历史 [{"role": "user/assistant", "content": "..."}]
                      用于支持追问和上下文连贯。
        """
        self._reporter_logs = []
        self._on_progress = on_progress
        # 保存供 _synthesize 使用
        self._chat_context = chat_history
        if api_key:
            import dashscope
            dashscope.api_key = api_key

        lang = self.language

        # ── 对话历史上下文 ──
        from datetime import datetime
        today = datetime.now().strftime("%Y年%m月%d日" if lang == "zh" else "%B %d, %Y")
        history_context = ""
        if chat_history and len(chat_history) > 0:
            # 取最近 6 轮对话（12 条消息）作为上下文
            recent = chat_history[-12:]
            history_lines = []
            for msg in recent:
                role_label = "用户" if msg["role"] == "user" else "系统"
                if lang != "zh":
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                # 截断长消息
                content = msg["content"][:500]
                history_lines.append(f"{role_label}: {content}")
            history_context = "\n".join(history_lines)

        system_with_date = self.coordinator_prompt + (
            f"\n\n**当前日期: {today}**。你是在{today}进行分析。不要依赖训练数据中的日期——"
            f"今天是真实日期，所有财务数据可能已经发布。"
            if lang == "zh" else
            f"\n\n**Current date: {today}**. You are analyzing on {today}. "
            f"Don't rely on training data dates - use the real current date."
        )

        # 注入对话上下文，让 Coordinator 理解追问的指代
        user_msg = (
            f"用户上传了一份金融文档（{doc_type or '未知类型'}）。\n\n"
            + (f"## 之前的对话\n{history_context}\n\n" if history_context else "")
            + f"## 当前问题\n{user_query}\n\n"
            + ("如果当前问题是追问（如'那…呢？''同比呢？'），请结合之前的对话理解指代。"
               if lang == "zh" else
               "If this is a follow-up question, use the conversation history to resolve references.")
            + "\n请开始分析。先思考你需要什么信息。"
        )

        conversation = [{
            "role": "system",
            "content": system_with_date
        }, {
            "role": "user",
            "content": user_msg
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
            user_query, all_findings, vector_store, api_key, on_token,
            accumulated_web
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
        """格式化观察文本，同时提取各 Agent 的 [COMPLETE]/[NEED_MORE] 声明。"""
        parts = []
        agent_statuses = []

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
                    content = item.get("content", "")

                    # 检测完整性声明
                    if "[COMPLETE]" in content:
                        agent_statuses.append(f"[{name}] COMPLETE")
                    elif "[NEED_MORE]" in content:
                        need_idx = content.find("[NEED_MORE]")
                        need_text = content[need_idx:need_idx+300]
                        agent_statuses.append(f"[{name}] NEED_MORE: {need_text}")
                    else:
                        agent_statuses.append(f"[{name}] 未声明")

                    parts.append(f"[{name} {status}] {content[:500]}")
                else:
                    parts.append(f"[{t}] {item.get('text', str(item))[:200]}")

        # 汇总 Agent 状态，引导 Coordinator 决策
        if agent_statuses:
            parts.insert(0, "\n## Agent 完整性汇总")
            all_complete = all("COMPLETE" in s and "NEED_MORE" not in s
                              for s in agent_statuses)
            if all_complete:
                parts.insert(1, "**所有 Agent 确认完成 → 你下一轮必须 need_more: false**")
            else:
                incomplete = [s for s in agent_statuses if "NEED_MORE" in s]
                parts.insert(1, f"**{len(incomplete)}个Agent需要更多信息 → 你必须继续搜索**")
                for s in agent_statuses:
                    parts.insert(2, f"- {s}")

        return "\n".join(parts)

    # ------------------------------------------------------------
    # 内部 — Synthesize（重构：结构化 claim 提取 + 冲突裁决 + 简报 → LLM 只写报告）
    # ------------------------------------------------------------

    def _synthesize(self, user_query: str, findings: list[str],
                    vector_store: VectorStore, api_key: str,
                    on_token=None,
                    accumulated_web: list[dict] | None = None) -> str:
        """综合所有发现，生成最终报告。

        架构原则：不让 LLM 做数据取证。
        1. 从所有原始数据中提取结构化 claims
        2. 算法裁决冲突（实际 > 预测、官方 > 媒体、多源 > 孤立）
        3. 构建干净简报给 LLM —— LLM 只负责写报告，不负责判断哪个数字对
        """
        lang = self.language

        # 1) 最终文档检索
        final_doc = ""
        if not vector_store.is_empty:
            results = vector_store.search(user_query, top_k=3, api_key=api_key)
            if results:
                final_doc = "\n".join(
                    f"[p{r.chunk.page}] {r.chunk.text[:300]}" for r in results[:3]
                )

        # 2) 收集所有原始数据文本
        all_raw_text = self._collect_raw_data(accumulated_web, findings)

        # 3) 最终交叉验证搜索
        fresh_text = ""
        try:
            fresh_web = self._search_web(user_query, api_key)
            if fresh_web:
                fresh_text = "\n".join(
                    f"[{w.get('title','')}] {w.get('snippet','')[:500]}"
                    for w in fresh_web[:3]
                )
        except Exception:
            pass

        # 4) 结构化 claim 提取（一次 LLM 调用，把原始文本转为结构化事实表）
        claims = self._extract_claims(all_raw_text, fresh_text, user_query, api_key)

        # 5) 冲突裁决（确定性规则，不靠 LLM）
        resolved, conflicts = self._resolve_claims(claims)

        # 6) 如果存在无法裁决的冲突，做针对性搜索
        arbitration = ""
        if conflicts:
            arb_query = self._build_arbitration_query(conflicts, user_query)
            if arb_query:
                self._log("Coordinator", "running",
                          f"Unresolved conflicts, arbitrating: {arb_query[:80]}"
                          if lang == "en" else f"存在未裁决的数据矛盾，针对性搜索: {arb_query[:80]}")
                try:
                    arb_results = self._search_web(arb_query, api_key)
                    if arb_results:
                        arb_text = "\n".join(
                            f"[{w.get('title','')}] {w.get('snippet','')[:500]}"
                            for w in arb_results[:3]
                        )
                        # 从裁决搜索中再次提取 claims 并重新裁决
                        arb_claims = self._extract_claims(arb_text, "", user_query, api_key)
                        resolved, conflicts = self._resolve_claims(claims + arb_claims)
                        arbitration = arb_text
                except Exception:
                    pass

        # 7) 构建简报（含对话上下文，让报告能引用之前的分析）
        brief = self._build_brief(user_query, resolved, conflicts,
                                  final_doc, arbitration, findings, lang,
                                  self._chat_context)

        # 8) LLM 只负责：基于已裁决的事实写报告
        messages = [
            {"role": "system", "content": self.synthesis_prompt},
            {"role": "user", "content": brief},
        ]

        if on_token:
            return self.llm.chat_stream(messages, on_token=on_token, model="qwen-max")
        return self.llm.chat(messages, model="qwen-max")

    # ------------------------------------------------------------
    # Claim 提取 / 裁决 / 简报构建
    # ------------------------------------------------------------

    def _collect_raw_data(self, accumulated_web: list[dict] | None,
                          findings: list[str]) -> str:
        """收集所有原始数据文本，去重后拼接。"""
        parts = []
        seen = set()
        if accumulated_web:
            for w in accumulated_web:
                sig = w.get("snippet", "")[:80].strip()
                if sig and sig not in seen:
                    seen.add(sig)
                    parts.append(
                        f"[{w.get('title','')}] (来源: {w.get('url','')})\n"
                        f"{w.get('snippet','')[:500]}"
                    )
        # 只取最近 4 轮 findings，避免 token 爆炸
        findings_text = "\n---\n".join(findings[-4:])[:4000]
        if findings_text.strip():
            parts.append(findings_text)
        return "\n\n".join(parts)

    def _extract_claims(self, primary_text: str, fresh_text: str,
                        user_query: str, api_key: str) -> list[dict]:
        """从原始文本中提取结构化 claims。

        一次 LLM 调用，输出 JSON 数组。每个 claim:
          {metric, value, unit, source_type, source_label, date}
        source_type: "actual" | "forecast" | "unknown"
        """
        if not primary_text.strip() and not fresh_text.strip():
            return []

        combined = primary_text[:4000]
        if fresh_text.strip():
            combined += f"\n\n[交叉验证]\n{fresh_text[:1500]}"

        prompt = f"""你是金融数据提取专家。从以下搜索结果的原始文本中，提取所有与用户问题相关的**量化主张 (claims)**。

用户问题: {user_query}

原始文本:
{combined}

请提取所有涉及具体数字的主张。每条主张输出一个 JSON 对象，整个输出为 JSON 数组。

每条主张的格式:
{{
  "metric": "指标名称（如 归母净利润、营收、同比增长率 等，用中文）",
  "value": "数值（如 57.35亿、262.28% 等，保留原始表述）",
  "source_type": "actual|forecast|unknown",
  "source_label": "来源简称（如 巨潮资讯网、新浪财经 等）",
  "date": "数据相关日期（如 2026Q1、2026-04-17 等，未知写 unknown）"
}}

source_type 判断规则:
- "actual": 文本描述的是已发布/已实现的业绩（关键词: 公告、披露、报告、发布、实现、完成）
- "forecast": 文本描述的是预测/估算（关键词: 预计、预测、预告、估计、guidance、estimate）
- "unknown": 无法判断

只输出 JSON 数组，不要其他文字。如果没有可提取的主张，输出 []。"""

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                model="qwen-turbo", temperature=0.0, max_tokens=800,
            )
            return self._parse_claims_json(resp)
        except Exception:
            return []

    def _parse_claims_json(self, text: str) -> list[dict]:
        """解析 LLM 返回的 claims JSON。"""
        import json as _json
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                claims = _json.loads(text[start:end + 1])
                if isinstance(claims, list):
                    return [c for c in claims if isinstance(c, dict) and c.get("metric")]
            except _json.JSONDecodeError:
                pass
        return []

    def _resolve_claims(self, claims: list[dict]) -> tuple[list[dict], list[dict]]:
        """裁决冲突：分组 → 排序 → 选出最佳。

        排序权重（依次比较）:
          1. source_type: actual > unknown > forecast
          2. 数据精度: 具体数字 > 范围值（"57.35亿" > "2.00-2.50亿"）
          3. 来源权威: 含 URL 的 > 无 URL 的
          4. 出现次数（通过 metric 模糊匹配计算 corroboration）

        Returns: (resolved, conflicts)
          resolved: 已裁决的事实列表（每个 metric 只保留最佳）
          conflicts: 无法确定最佳的事实对（权重相同、数值矛盾）
        """
        if not claims:
            return [], []

        # 分组: 按 metric 归一化后分组
        import re

        def normalize_metric(m: str) -> str:
            """归一化 metric 名称以便分组。"""
            m = m.strip().lower()
            # 去掉括号内容
            m = re.sub(r'\(.*?\)', '', m)
            m = re.sub(r'（.*?）', '', m)
            return m.strip()

        groups: dict[str, list[dict]] = {}
        for c in claims:
            key = normalize_metric(c.get("metric", ""))
            if key:
                groups.setdefault(key, []).append(c)

        resolved = []
        conflicts = []

        for metric_key, group in groups.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue

            # 多源 → 排序选最佳
            def claim_score(c: dict) -> tuple:
                # source_type 优先级
                type_order = {"actual": 0, "unknown": 1, "forecast": 2}
                t = type_order.get(c.get("source_type", "unknown"), 1)

                # 是否为范围值（含 "-" 或 "至" 或 "~"）
                v = c.get("value", "")
                is_range = bool(re.search(r'[\d.]+[-~至]\s*[\d.]+', v))

                # 来源是否有 URL/日期
                has_label = 1 if c.get("source_label") else 0
                has_date = 1 if c.get("date") and c["date"] != "unknown" else 0

                # 高优先级 → 小数字排在前面
                return (t, 1 if is_range else 0, -(has_label + has_date))

            sorted_group = sorted(group, key=claim_score)

            best = sorted_group[0]
            second = sorted_group[1] if len(sorted_group) > 1 else None

            # 检测冲突: 如果最佳和第二的数值显著不同
            if second and self._values_conflict(best.get("value", ""),
                                                 second.get("value", "")):
                # 如果最佳和第二的 source_type 相同且分数相同 → 无法裁决
                if claim_score(best) == claim_score(second):
                    conflicts.append(best)
                    conflicts.append(second)
                else:
                    resolved.append(best)
                    # 标注有争议
                    best["_disputed"] = True
                    best["_disputed_by"] = second.get("value", "")
            else:
                resolved.append(best)

        return resolved, conflicts

    @staticmethod
    def _values_conflict(v1: str, v2: str) -> bool:
        """判断两个数值是否存在实质性冲突（相差 2 倍以上）。"""
        import re

        def to_float(v: str) -> float | None:
            m = re.search(r'([\d,]+\.?\d*)', str(v).replace(",", ""))
            return float(m.group(1)) if m else None

        n1, n2 = to_float(v1), to_float(v2)
        if n1 and n2 and n1 > 0 and n2 > 0:
            ratio = max(n1, n2) / min(n1, n2)
            return ratio > 2.0
        return False

    def _build_arbitration_query(self, conflicts: list[dict],
                                  user_query: str) -> str | None:
        """从未裁决冲突中构建针对性验证查询。"""
        if not conflicts:
            return None
        metrics = list(set(c.get("metric", "") for c in conflicts))
        metric_str = " ".join(metrics[:3])
        return (f"{user_query} {metric_str} 官方公告 实际数据 "
                f"site:cninfo.com.cn OR site:sse.com.cn OR site:szse.cn"
                f" -预计 -预测 -预告 -forecast")

    def _build_brief(self, user_query: str, resolved: list[dict],
                     conflicts: list[dict], doc_text: str,
                     arbitration_text: str, findings: list[str],
                     lang: str,
                     chat_history: list[dict] | None = None) -> str:
        """构建结构化简报 —— 这是综合 LLM 收到的唯一输入。

        设计原则：不给 LLM 原始文本碎片，给它已经裁决好的事实表。
        LLM 的工作：基于已验证的事实写出清晰、专业的报告。
        """
        zh = lang == "zh"

        parts = []

        # ── 用户问题 + 对话上下文 ──
        parts.append(f"## {'用户问题' if zh else 'User Question'}\n{user_query}\n")

        if chat_history and len(chat_history) > 1:
            # 取最近几轮对话，帮 LLM 理解追问的指代
            recent = chat_history[-8:]
            ctx_lines = []
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "FinRisk"
                if lang != "zh":
                    role = "User" if msg["role"] == "user" else "FinRisk"
                ctx_lines.append(f"- **{role}**: {msg['content'][:300]}")
            parts.append(
                f"## {'对话上下文（当前问题是追问，请结合之前的对话理解指代）' if zh else 'Conversation Context (resolve references from prior exchanges)'}\n"
                + "\n".join(ctx_lines) + "\n"
            )

        # ── 已验证事实（核心！这是 LLM 写报告的唯一数据来源）──
        parts.append(f"## {'已验证事实（请仅使用以下数据回答）' if zh else 'Verified Facts (use ONLY these data to answer)'}")
        if resolved:
            parts.append("| # | Metric | Value | Type | Source | Date |")
            parts.append("|---|--------|-------|------|--------|------|")
            for i, c in enumerate(resolved, 1):
                disputed = " ⚠️" if c.get("_disputed") else ""
                parts.append(
                    f"| {i} | {c.get('metric','?')} | {c.get('value','?')}{disputed} | "
                    f"{c.get('source_type','?')} | {c.get('source_label','?')} | "
                    f"{c.get('date','?')} |"
                )
            # 标注有争议的事实
            disputed_claims = [c for c in resolved if c.get("_disputed")]
            if disputed_claims:
                parts.append(
                    f"\n⚠️ "
                    + ("以上标注 ⚠️ 的事实存在矛盾来源，已通过算法裁决选出最可靠值。"
                       if zh else
                       "Facts marked with ⚠️ have conflicting sources; the most reliable value was selected algorithmically.")
                )
        else:
            parts.append(
                "（未提取到结构化事实，请基于文档和搜索文本回答）"
                if zh else "(No structured facts extracted; answer based on document and search text.)"
            )
        parts.append("")

        # ── 未解决的冲突 ──
        if conflicts:
            parts.append(f"## {'未解决的冲突（以下数据存在矛盾，请在报告中指出不确定性）' if zh else 'Unresolved Conflicts (note uncertainty in report)'}")
            for c in conflicts:
                parts.append(
                    f"- {c.get('metric','?')}: {c.get('value','?')} "
                    f"({c.get('source_label','?')}, type={c.get('source_type','?')})"
                )
            parts.append("")

        # ── 文档上下文 ──
        if doc_text.strip():
            parts.append(f"## {'文档关键段落（辅助背景）' if zh else 'Document Context (background only)'}")
            parts.append(doc_text)
            parts.append("")

        # ── 裁决搜索（如有）──
        if arbitration_text.strip():
            parts.append(f"## {'矛盾裁决搜索' if zh else 'Arbitration Search'}")
            parts.append(arbitration_text)
            parts.append("")

        # ── 分析过程摘要 ──
        brief_findings = "\n---\n".join(findings[-3:])[:2000]
        if brief_findings.strip():
            parts.append(f"## {'分析过程摘要（供参考）' if zh else 'Analysis Process Summary (for reference)'}")
            parts.append(brief_findings)
            parts.append("")

        # ── 输出指令（简洁，因为数据已经裁决好了）──
        parts.append("---")
        parts.append(
            "## 输出要求\n\n"
            "1. 直接回答用户问题。第一段就给答案。\n"
            "2. 使用「已验证事实」表中的数据。如果表中有具体数字，直接引用。\n"
            "3. 标注来源时，引用表格中的 Source 和 Date 列。\n"
            "4. 如果「未解决的冲突」中有相关指标，在报告中指出不确定性和矛盾来源。\n"
            "5. 不要引入表格中没有的新数据。不要做额外的推测或分析。"
            if zh else
            "## Instructions\n\n"
            "1. Answer the user's question directly. Lead with the answer.\n"
            "2. Use ONLY data from the Verified Facts table. Cite Source and Date.\n"
            "3. If Unresolved Conflicts are relevant, note the uncertainty.\n"
            "4. Do NOT introduce data not in the table. Do NOT speculate."
        )

        return "\n".join(parts)

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
        if self._on_progress:
            try:
                self._on_progress(agent, status, content)
            except Exception:
                pass
