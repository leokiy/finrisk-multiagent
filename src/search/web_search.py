"""
联网搜索模块 — 基于 DashScope 内置搜索能力（enable_search=True）。

使用用户已有的 DashScope API Key，无需额外申请搜索服务。
搜索作为独立步骤在 Orchestrator 中执行，结果注入所有 Agent 上下文。
"""

from dataclasses import dataclass


@dataclass
class WebResult:
    """单条网络搜索结果。"""
    title: str
    url: str
    snippet: str


def search_web(query: str, api_key: str = "", max_results: int = 3,
               language: str = "zh") -> list[WebResult]:
    """使用 DashScope 内置搜索（enable_search=True）搜索最新信息。

    同一 API Key，无需额外开通。搜索失败返回空列表，不影响主流程。
    """
    if not api_key:
        return []

    try:
        import dashscope

        search_prompt = (
            f"请搜索以下内容，返回具体数据、来源和日期：{query}"
            if language == "zh"
            else f"Search for the following and return specific data, sources and dates: {query}"
        )

        resp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": search_prompt}],
            result_format="message",
            api_key=api_key,
            enable_search=True,
            max_tokens=800,
            temperature=0.1,
        )

        if resp.status_code == 200:
            text = resp.output.choices[0].message.content
            if text and len(text.strip()) > 20:
                # 尝试提取搜索引用信息
                search_info = ""
                if hasattr(resp, "output") and resp.output:
                    try:
                        search_info = str(
                            resp.output.get("search_info", "")
                            if isinstance(resp.output, dict)
                            else ""
                        )
                    except Exception:
                        pass

                return [WebResult(
                    title=f"DashScope 搜索结果: {query[:60]}",
                    url=search_info if search_info else "",
                    snippet=text.strip(),
                )]
        return []
    except Exception:
        return []


def build_search_queries(user_query: str, document_context: str = "",
                         language: str = "zh") -> list[str]:
    """根据用户问题生成 1-2 个精准搜索查询。

    DashScope 搜索是 LLM 驱动的，查询越具体结果越精准。
    不需要像传统搜索引擎那样拼接关键词——直接用自然语言更有效。
    """
    base = user_query.strip()

    if language == "zh":
        queries = [
            f"{base}。请搜索并返回最新的具体数据、时间和来源。",
        ]
        # 如果问题涉及财务数据，追加一个行业/对比查询
        if any(kw in base for kw in ["营收", "利润", "收入", "毛利率", "净利率",
                                       "资产负债", "现金流", "增长", "下滑"]):
            queries.append(
                f"{base} 同行业对比 2025 最新"
            )
        return queries[:2]

    queries = [
        f"{base}. Please search and return the latest specific data, dates and sources.",
    ]
    if any(kw in base.lower() for kw in ["revenue", "profit", "margin", "growth",
                                           "ratio", "cash flow"]):
        queries.append(f"{base} industry comparison 2025 latest")
    return queries[:2]
