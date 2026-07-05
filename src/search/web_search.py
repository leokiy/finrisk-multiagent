"""
联网搜索模块 — 基于 DuckDuckGo 真实搜索引擎。

每个 Agent 使用专属搜索查询并行执行搜索，获取真实的标题、URL 和摘要。
搜索结果注入 Agent 上下文，由 LLM 综合文档 + 网络信息后产出分析。

不再依赖 DashScope 内置搜索（enable_search），那个是黑盒且经常不生效。
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
    """使用 DuckDuckGo 搜索引擎获取真实网络搜索结果。

    api_key 参数保留以兼容旧接口，实际不使用。
    搜索失败返回空列表，不阻塞主流程。

    Args:
        query: 搜索查询（自然语言）
        api_key: 保留参数（兼容性）
        max_results: 最大返回条数
        language: 查询语言
    Returns:
        WebResult 列表
    """
    if not query or len(query.strip()) < 3:
        return []

    # 中文查询加时间限定，确保搜到最新数据
    if language == "zh" and not any(kw in query for kw in ["2025", "2026", "2024"]):
        query = f"{query} 2025 2026"

    try:
        # 优先使用新版 ddgs 包，fallback 到旧版 duckduckgo_search
        try:
            from ddgs import DDGS
            engine = DDGS()
            raw_results = list(engine.text(query, max_results=max_results))
        except ImportError:
            from duckduckgo_search import DDGS
            engine = DDGS()
            raw_results = list(engine.text(query, max_results=max_results))

        results = []
        for r in raw_results:
            body = r.get("body", "")
            href = r.get("href", "")
            title = r.get("title", "")
            if body and len(body) > 20:
                results.append(WebResult(
                    title=str(title)[:120],
                    url=str(href)[:500],
                    snippet=str(body)[:800],
                ))

        return results[:max_results]

    except Exception:
        return []
