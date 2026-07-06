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

COORDINATOR_PROMPT_ZH = """# 你是金融分析团队的 Coordinator

## 你的团队

4 个专家。每人有明确的输入契约和输出契约。见下表：

| 成员 | 输入 | 产出 | 触发 |
|------|------|------|------|
| data_extractor | 搜索文本 + 聚焦指令 | `{status, data: [{metric, value, source_type, source}]}` | 需要提取数字时 |
| risk_assessor | 数据提取员 JSON + 搜索补充 | `{status, findings: [{dimension, fact, judgment, risk_level, evidence}]}` | 需要风险判断时 |
| compliance_checker | 数据提取员 JSON + 搜索补充 | `{status, findings: [{area, finding, verdict, evidence}]}` | 涉及合规时 |
| devils_advocate | 另外三人完整 JSON | `{status, challenges: [{target, target_conclusion, challenge, severity, evidence}]}` | 另外三人完成后，需要质疑时 |

## 你的职责

1. 判断问题类型 → 选策略
2. 写清晰的派发指令（用户问什么 + 已知什么 + 聚焦什么）
3. 审查 Agent 返回的 JSON：COMPLETE 真的够了吗？不同 Agent 结论有矛盾吗？
4. 维护任务账本：谁被派了 → 回没回来 → 结论是什么
5. 决定何时信息足够，结束循环

## 你的工具

- search_document(query) — 搜上传的 PDF。用关键词。
- search_web(query) — AI 联网搜索。用自然语言问题。不要用 site: 语法。
- run_analyst(analyst, instruction, context) — 派专家。instruction 按"用户问X。已掌握Y。聚焦Z。"格式写。

## 决策流程

### 问题类型 → 策略

**事实查询**（"...是多少""...什么时候"）→ search_web → 搜到就结束。不派专家。
**分析判断**（"怎么样""是否合理""有什么风险"）→ search_web → data_extractor → risk_assessor/compliance_checker → devils_advocate
**全面评估**（"全面""综合"）→ search_web → data_extractor → risk_assessor + compliance_checker 并行 → devils_advocate
**文档分析**（用户上传了文档且针对文档内容）→ search_document 优先 → search_web 仅补充 → 文档数据 vs 网络数据冲突时以文档为准，不把网络预测当实际数据

### 任务账本

每轮派 Agent 后，在心里记录：
- 本轮委托：派了谁，查什么
- 返回状态：COMPLETE or NEED_MORE
- 结论摘要：关键发现（1-2 句话）

下一轮决策时参考账本：已查到什么、还缺什么、谁还需要被派。

### 上下文预算

给 Agent 的 context 只传摘要（≤500 字），不传全文。Agent 的 JSON 输出只取关键字段给下一个 Agent，不整段复制。

### 回退链

搜索无结果 → 换关键词重试一次 → 仍无结果 → 向用户说明"该方向未找到信息"，继续其他方向。
Agent 返回 NEED_MORE → 补搜 → 重新派同一 Agent（最多 2 次）→ 仍 NEED_MORE → 以已有信息继续。

## JSON 决策格式

每轮一个 JSON:

```json
{
  "thought": "当前理解和进展（含任务账本摘要）",
  "plan": "下一步做什么，为什么",
  "actions": [
    {"tool": "search_web", "query": "自然语言搜索"},
    {"tool": "run_analyst", "analyst": "data_extractor", "instruction": "用户问XX。已掌握：YY。聚焦ZZ。", "context": "关键发现摘要"}
  ],
  "need_more": true
}
```

信息足够时:
```json
{"thought": "信息足够，无矛盾", "need_more": false}
```

## 铁律

1. 事实查询搜到就答，不派专家
2. devils_advocate 最后派，给三人完整输出
3. 指令按"用户问X。已掌握Y。聚焦Z。"格式
4. Agent NEED_MORE → 补搜 → 重派（最多 2 次）
5. 搜索无结果 → 换词重试一次 → 仍无 → 说明未找到，继续
6. need_more: false = 确认信息足够 + 无未解决矛盾"""

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


def _summarize_agent_output(parsed: dict | None, raw: str) -> str:
    """生成 Agent 输出的可读摘要，用于日志。"""
    if not parsed:
        return raw[:200]
    status = parsed.get("status", "?")
    if "data" in parsed:
        metrics = [d.get("metric", "?") for d in parsed["data"][:5]]
        return f"[{status}] 提取: {', '.join(metrics)}"
    if "findings" in parsed:
        items = [f.get("dimension", f.get("area", "?")) for f in parsed["findings"][:5]]
        return f"[{status}] 发现: {', '.join(items)}"
    if "challenges" in parsed:
        sevs = [c.get("severity", "?") for c in parsed["challenges"][:5]]
        return f"[{status}] 质疑: {len(parsed['challenges'])}条 ({', '.join(sevs)})"
    return raw[:200]


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
            doc_company: str = "",
            max_rounds: int = 5,
            on_progress=None,
            chat_history: list[dict] | None = None) -> dict:
        """ReAct 决策循环。

        chat_history: 之前的对话历史 [{"role": "user/assistant", "content": "..."}]
                      用于支持追问和上下文连贯。
        """
        self._reporter_logs = []
        self._on_progress = on_progress
        self._chat_context = chat_history
        if api_key:
            import dashscope
            # 清理 API key 中的不可见字符（避免 latin-1 编码错误）
            clean_key = api_key.strip().encode('ascii', errors='ignore').decode('ascii')
            dashscope.api_key = clean_key

        lang = self.language

        # ── 快速通道：简单查数据 → 直接 qwen-max + enable_search，不走编排器 ──
        if self._is_simple_lookup(user_query):
            self._log("Coordinator", "running",
                      "Fast path: direct search" if lang == "en" else "简单查询，直接搜索回答")
            answer = self._fast_answer(user_query, vector_store, api_key, on_token, web_search_enabled, doc_company)
            self._log("Coordinator", "done",
                      "Done" if lang == "en" else "完成")
            return {
                "query": user_query,
                "final_report": answer,
                "followup_questions": [],
                "execution_log": self._reporter_logs,
                "rounds": 0,
            }

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
        doc_info = f"用户上传了一份金融文档（{doc_type or '未知类型'}"
        if doc_company:
            doc_info += f"，公司：{doc_company}"
        doc_info += "）。"
        if doc_company:
            doc_info += ("\n文档是关于该公司的。所有 search_web 查询必须包含公司名，"
                        "不要搜到其他公司的数据。"
                        if lang == "zh" else
                        "\nDocument is about this company. ALL search_web queries MUST include the company name.")

        user_msg = (
            f"{doc_info}\n\n"
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
        structured_findings: list[dict] = []  # Agent 的结构化 JSON 输出
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
                          "Decision parse failed, retrying" if lang == "en" else "决策解析失败，尝试恢复")
                # 第一次恢复：提醒输出纯 JSON
                conversation.append({
                    "role": "user",
                    "content": ("上一轮输出不是有效 JSON。请只输出一行 JSON，不要任何解释。"
                               '格式: {"thought":"...","plan":"...","actions":[],"need_more":true}')
                    if lang == "zh" else
                    "Last output was not valid JSON. Output ONLY one line of JSON."
                })
                decision = self._think(conversation)
                if decision is None:
                    self._log("Coordinator", "error",
                              "Retry failed, forcing synthesis" if lang == "en" else "二次恢复失败，强制综合")
                    # 不再 break——用已收集的信息直接综合
                    decision = {"thought": "JSON解析失败，基于已收集信息综合", "need_more": False}

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
                # 收集 Agent 的结构化输出
                for items in results.values():
                    for item in items:
                        if item.get("type") == "analyst_output" and item.get("parsed"):
                            structured_findings.append({
                                "analyst": item["analyst"],
                                **item["parsed"]
                            })
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
            accumulated_web, structured_findings
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
                max_tokens=1200,
            )
        except Exception as e:
            self._log("Coordinator", "error", f"Think failed: {e}")
            return None

        return self._parse_json(resp)

    def _parse_json(self, text: str) -> dict | None:
        """从 LLM 回复中提取 JSON。增强了容错。"""
        import re
        text = text.strip()
        # 去掉 markdown 代码块
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
            if text.endswith("```"):
                text = text[:-3]
        # 去掉代码块后的残余标记
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # 找 JSON 边界
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                # 尝试修复常见问题：尾部多余逗号
                try:
                    fixed = re.sub(r',\s*}', '}', text[start:end + 1])
                    fixed = re.sub(r',\s*]', ']', fixed)
                    return json.loads(fixed)
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
                    # 把累积的网络搜索结果传给 Agent
                    web_for_agent = _format_web_for_agent(accumulated_web)
                    # Coordinator 填写的 context 字段：团队已发现的关键信息
                    ctx = action.get("context", "") or ""
                    if not ctx and len(all_findings) > 0:
                        ctx = "\n".join(all_findings[-3:])[:2000]
                    # 结构化组装：任务 + 团队背景
                    full_instr = instr
                    if ctx:
                        full_instr = f"""## 你的任务
{instr}

## 团队已掌握的信息
{ctx}"""
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
        """联网搜索。enable_search 为主（国内可用），DDGS 为辅（挂了就跳过）。"""
        import concurrent.futures
        out = []

        # ── 主搜索：enable_search（DashScope，国内直连）──
        def _do_enable_search():
            try:
                from src.llm.client import LLMConfig
                prompt = f"搜索以下内容，返回具体数据和来源：{query}"
                cfg = LLMConfig(api_key=api_key, model="qwen-turbo",
                              temperature=0.1, max_tokens=1200)
                client = LLMClient(cfg)
                resp = client.chat([{"role": "user", "content": prompt}],
                                  enable_search=True)
                if resp and len(resp.strip()) > 20:
                    return [{
                        "type": "ai_search",
                        "title": f"AI搜索: {query[:60]}",
                        "url": "",
                        "snippet": resp.strip()[:1000],
                    }]
            except Exception:
                pass
            return []

        # 只跑 enable_search（DDGS 在国内网络环境不可用）
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future_enable = pool.submit(_do_enable_search)
            try:
                out.extend(future_enable.result(timeout=30))
            except Exception:
                pass

        return out if out else [{"type": "empty", "text": "搜索无结果"}]

    def _run_analyst(self, name: str, instruction: str,
                     vector_store: VectorStore, api_key: str,
                     web_results: list | None = None) -> dict:
        """运行指定分析师，解析其 JSON 输出。"""
        agents = {
            "data_extractor": self.data_extractor,
            "risk_assessor": self.risk_assessor,
            "compliance_checker": self.compliance,
            "devils_advocate": self.devils_advocate,
        }
        agent = agents.get(name)
        if not agent:
            return {"type": "error", "text": f"未知分析师: {name}"}

        result = agent.run(instruction, vector_store, api_key=api_key,
                          web_search_results=web_results)
        raw = result.content[:3000] if result.success else result.error

        # 尝试解析 Agent 的 JSON 输出
        parsed = self._parse_agent_output(raw, name)

        self._log(agent.name,
                  "done" if result.success else "error",
                  _summarize_agent_output(parsed, raw))

        return {
            "type": "analyst_output",
            "analyst": name,
            "success": result.success,
            "content": raw,
            "parsed": parsed,  # 结构化数据，供 Coordinator 和 synthesis 使用
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
                    accumulated_web: list[dict] | None = None,
                    structured_findings: list[dict] | None = None) -> str:
        """综合所有发现，生成最终报告。

        优先使用 Agent 已产出的结构化 JSON。仅当无 Agent 输出时回退到 claim 提取管道。
        """
        lang = self.language

        # 1) 最终文档检索
        final_doc = ""
        if not vector_store.is_empty:
            try:
                results = vector_store.search(user_query, top_k=3, api_key=api_key)
                if results:
                    final_doc = "\n".join(
                        f"[p{r.chunk.page}] {r.chunk.text[:300]}" for r in results[:3]
                    )
            except Exception as e:
                self._log("Coordinator", "warning", f"Final doc search failed: {e}")

        # 2) 选择路径：有 Agent 结构化数据 → 直接用；无 → claim 提取管道
        if structured_findings:
            brief = self._build_brief_from_structured(
                user_query, structured_findings, final_doc, findings, lang
            )
        else:
            brief = self._build_brief_from_raw(
                user_query, accumulated_web, findings, vector_store, api_key, lang, final_doc
            )

        # 3) LLM 写报告
        messages = [
            {"role": "system", "content": self.synthesis_prompt},
            {"role": "user", "content": brief},
        ]

        try:
            if on_token:
                return self.llm.chat_stream(messages, on_token=on_token, model="qwen-max")
            return self.llm.chat(messages, model="qwen-max")
        except Exception as e:
            self._log("Coordinator", "error",
                      f"Synthesis LLM failed: {e}" if lang == "en" else f"综合报告生成失败: {e}")
            return ("报告生成失败，请稍后重试。分析数据已收集。"
                    if lang == "zh" else "Report generation failed. Analysis data collected.")

    # ------------------------------------------------------------
    # 简报构建（两条路径）
    # ------------------------------------------------------------

    def _build_brief_from_structured(
        self, user_query: str, structured: list[dict],
        doc_text: str, findings: list[str], lang: str
    ) -> str:
        """路径 A：直接用 Agent JSON 输出构建简报。"""
        zh = lang == "zh"
        parts = []
        parts.append(f"## {'用户问题' if zh else 'User Question'}\n{user_query}\n")

        # 对话上下文
        chat = self._chat_context
        if chat and len(chat) > 1:
            parts.append(f"## {'对话上下文' if zh else 'Conversation Context'}")
            for msg in chat[-6:]:
                role = "用户" if msg["role"] == "user" else "系统"
                parts.append(f"- **{role}**: {msg['content'][:200]}")
            parts.append("")

        # Agent 结构化输出 → 事实表
        parts.append(f"## {'团队分析结果（唯一数据来源）' if zh else 'Team Analysis (sole data source)'}")
        for s in structured:
            aname = s.get("analyst", "?")
            if "data" in s:
                parts.append(f"### {aname}")
                parts.append("| 指标 | 数值 | 类型 | 来源 |")
                parts.append("|------|------|------|------|")
                for d in s["data"]:
                    parts.append(f"| {d.get('metric','?')} | {d.get('value','?')} | {d.get('source_type','?')} | {d.get('source','?')} |")
                parts.append("")
            if "findings" in s:
                parts.append(f"### {aname}")
                for f in s["findings"]:
                    dim = f.get("dimension", f.get("area", "?"))
                    verdict = f.get("risk_level", f.get("verdict", ""))
                    parts.append(f"- **{dim}** [{verdict}]: {f.get('judgment', f.get('finding','?'))} (证据: {f.get('evidence','?')})")
                parts.append("")
            if "challenges" in s:
                parts.append(f"### {aname}")
                for c in s["challenges"]:
                    parts.append(f"- [{c.get('severity','?')}] **{c.get('target','?')}**: {c.get('challenge','?')}")
                parts.append("")
        parts.append("")

        # 文档上下文
        if doc_text.strip():
            parts.append(f"## {'文档关键段落' if zh else 'Document Context'}")
            parts.append(doc_text)
            parts.append("")

        # 分析日志摘要
        brief_log = "\n---\n".join(findings[-3:])[:1500]
        if brief_log.strip():
            parts.append(f"## {'分析过程摘要' if zh else 'Analysis Log Summary'}")
            parts.append(brief_log)
            parts.append("")

        # 输出指令
        parts.append("---")
        parts.append(
            "## 输出要求\n\n"
            "1. 上面「团队分析结果」是你唯一允许使用的数据来源。**禁止使用表中没有的数字。**\n"
            "2. 第一句直接给答案。结构跟问题走，不用模板。\n"
            "3. 标注来源用表中的 Source 列。\n"
            "4. 不同 Agent 结论有矛盾时必须指出并裁决。"
            if zh else
            "## Instructions\n\n"
            "1. ONLY use data from the Team Analysis table. No external numbers.\n"
            "2. Lead with answer. No templates.\n"
            "3. Cite sources. Resolve contradictions explicitly."
        )
        return "\n".join(parts)

    def _build_brief_from_raw(
        self, user_query: str, accumulated_web: list[dict] | None,
        findings: list[str], vector_store: VectorStore, api_key: str,
        lang: str, doc_text: str
    ) -> str:
        """路径 B：无 Agent 输出时的回退——claim 提取管道。"""
        all_raw_text = self._collect_raw_data(accumulated_web, findings)

        # 最终交叉验证搜索
        fresh_text = ""
        try:
            fresh_web = self._search_web(user_query, api_key)
            if fresh_web:
                fresh_text = "\n".join(
                    f"[{w.get('title','')}] {w.get('snippet','')[:500]}"
                    for w in fresh_web[:3]
                )
        except Exception as e:
            self._log("Coordinator", "warning", f"Fresh search failed: {e}")

        claims = self._extract_claims(all_raw_text, fresh_text, user_query, api_key)
        if not claims and all_raw_text:
            claims = self._fallback_extract(all_raw_text)

        try:
            resolved, conflicts = self._resolve_claims(claims)
        except Exception as e:
            self._log("Coordinator", "warning", f"Claim resolution failed: {e}")
            resolved, conflicts = claims, []

        arbitration = ""
        if conflicts:
            arb_query = self._build_arbitration_query(conflicts, user_query)
            if arb_query:
                try:
                    arb_results = self._search_web(arb_query, api_key)
                    if arb_results:
                        arb_text = "\n".join(
                            f"[{w.get('title','')}] {w.get('snippet','')[:500]}"
                            for w in arb_results[:3]
                        )
                        arb_claims = self._extract_claims(arb_text, "", user_query, api_key)
                        resolved, conflicts = self._resolve_claims(claims + arb_claims)
                        arbitration = arb_text
                except Exception as e:
                    self._log("Coordinator", "warning", f"Arbitration search failed: {e}")

        return self._build_brief(user_query, resolved, conflicts,
                                  doc_text, arbitration, findings, lang,
                                  self._chat_context)

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

    def _fallback_extract(self, raw_text: str) -> list[dict]:
        """暴力兜底：从原始文本中提取含数字的行，构建基础 claims。"""
        import re
        claims = []
        # 匹配 "数字 + 单位" 模式
        pattern = re.compile(
            r'([\d,]+\.?\d*)\s*(亿元|亿|万元|万|元|%|％)\s*[^\n]{0,80}'
        )
        seen_values = set()
        for m in pattern.finditer(raw_text):
            value = m.group(0).strip()[:120]
            sig = value[:60]
            if sig in seen_values:
                continue
            seen_values.add(sig)
            # 简单分类
            surrounding = raw_text[max(0, m.start()-60):m.end()+60]
            is_forecast = any(kw in surrounding for kw in
                            ['预计', '预测', '预告', '估计'])
            claims.append({
                "metric": "数据",
                "value": value,
                "source_type": "forecast" if is_forecast else "unknown",
                "source_label": "分析记录",
                "date": "unknown",
            })
        return claims[:15]

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
        return (f"{user_query} {metric_str} 的官方公告实际数据，排除业绩预告和预测数据")

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

        # ── 输出指令（强制使用已验证数据）──
        parts.append("---")
        if resolved:
            parts.append(
                "## 输出要求\n\n"
                "1. 上面「已验证事实」表是你唯一允许使用的数据来源。禁止使用表中没有的数字。\n"
                "2. 如果你看到搜索结果中有数字但表里没有 → 说明该数字在冲突裁决中被淘汰了，不要使用。\n"
                "3. 第一句直接给答案。结构跟问题走，不用模板。\n"
                "4. 标注来源用表中的 Source 和 Date 列。\n"
                "5. 有未解决的冲突时指出来。"
                if zh else
                "## Instructions\n\n"
                "1. ONLY use data from the Verified Facts table above. Do NOT use numbers not in the table.\n"
                "2. If search results have numbers not in the table → they were removed during conflict resolution. Do not use them.\n"
                "3. Lead with the answer. Structure follows the question, not a template.\n"
                "4. Cite Source and Date from the table.\n"
                "5. Note unresolved conflicts if any."
            )
        else:
            parts.append(
                "## 输出要求\n\n"
                "1. 第一句直接给答案。结构跟问题走，不用模板。\n"
                "2. 不确定的事就说\"不确定\"或\"信息不足\"，不要用确定语气描述不确定的事。\n"
                "3. 标注来源（文档页码或网络来源媒体名+日期）。"
                if zh else
                "## Instructions\n\n"
                "1. Lead with the answer. Do not use templates.\n"
                "2. Say \"uncertain\" or \"insufficient information\" when unsure — don't sound certain about uncertain things.\n"
                "3. Cite sources."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 快速通道：简单查数据 bypass 编排器
    # ------------------------------------------------------------

    @staticmethod
    def _is_simple_lookup(query: str) -> bool:
        """判断是否为简单事实查询——明确问一个数字/事实，不需要多 Agent。"""
        q = query.strip()
        # 只有明确查事实关键词 + 短问题才走快速通道
        fact_keywords = ['多少', '什么时候', '代码', '股价', '市值', '市盈率',
                        '地址', '电话', '利率', '汇率', '几点', '在哪',
                        'what is', 'how many', 'how much', 'when did']
        analysis_keywords = ['分析', '评估', '风险', '怎么样', '为什么', '是否',
                           '全面', '综合', '完整', '对比', '比较', '有哪些',
                           '哪些', '名单', '列表', '排名', '走势', '趋势',
                           '如何', '怎样', '增长', '下降', '变化', '影响',
                           'analyze', 'assess', 'risk', 'comprehensive', 'compare',
                           'evaluate', 'why', 'which companies', 'list of']

        has_fact = any(kw in q.lower() for kw in fact_keywords)
        has_analysis = any(kw in q.lower() for kw in analysis_keywords)

        if has_analysis:
            return False
        if has_fact and len(q) < 30:
            return True
        return False

    def _fast_answer(self, user_query: str, vector_store: VectorStore,
                     api_key: str, on_token, web_search_enabled: bool,
                     doc_company: str = "") -> str:
        """简单查询：直接 qwen-max + enable_search，不走编排器。"""
        from datetime import datetime
        today = datetime.now().strftime("%Y年%m月%d日")

        # 文档上下文（如有）
        doc_context = ""
        if not vector_store.is_empty:
            try:
                results = vector_store.search(user_query, top_k=2, api_key=api_key)
                if results:
                    doc_context = "\n".join(
                        f"[第{r.chunk.page}页] {r.chunk.text[:300]}" for r in results[:2]
                    )
            except Exception:
                pass

        company_hint = ""
        if doc_company:
            company_hint = f"\n注意：你正在分析的公司是{doc_company}。只回答关于该公司的数据，忽略其他公司的信息。\n"

        prompt = f"""今天是{today}。请联网搜索最新信息后回答。

{f'## 用户上传的文档（仅供参考）\n{doc_context[:1500]}' if doc_context else ''}
{company_hint}
## 用户问题
{user_query}

请联网搜索后直接回答。有具体数据列出并标注来源。不确定说"不确定"。不要说你没有访问权限——你已经联网了。"""

        messages = [{"role": "user", "content": prompt}]

        if on_token and web_search_enabled:
            return self.llm.chat_stream(
                messages, on_token=on_token, model="qwen-max", enable_search=True
            )
        elif web_search_enabled:
            return self.llm.chat(
                messages, model="qwen-max", enable_search=True
            )
        elif on_token:
            return self.llm.chat_stream(
                messages, on_token=on_token, model="qwen-max"
            )
        return self.llm.chat(messages, model="qwen-max")

    # ------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------

    @staticmethod
    def _parse_agent_output(raw: str, agent_name: str) -> dict | None:
        """尝试从 Agent 的原始输出中解析 JSON 结构。失败返回 None。"""
        import json as _json
        text = raw.strip()
        # 找 JSON 边界
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = _json.loads(text[start:end + 1])
            if isinstance(obj, dict) and "status" in obj:
                return obj
        except _json.JSONDecodeError:
            pass
        return None

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
