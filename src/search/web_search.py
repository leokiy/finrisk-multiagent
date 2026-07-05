"""
联网搜索模块 — 基于 DashScope 内置搜索能力（enable_search=True）。

每个 Agent 使用专属搜索查询并行执行搜索，结果注入 Agent 上下文。
搜索作为独立步骤在 Orchestrator 中执行，与 Agent 分析解耦。
"""

import re
from datetime import datetime
from dataclasses import dataclass


@dataclass
class WebResult:
    """单条网络搜索结果。"""
    title: str
    url: str
    snippet: str


def search_web(query: str, api_key: str = "", max_results: int = 2,
               language: str = "zh") -> list[WebResult]:
    """使用 DashScope 内置搜索（enable_search=True）获取最新信息。

    同一 API Key，无需额外开通搜索服务。
    搜索失败返回空列表，不阻塞主流程。

    Args:
        query: 搜索查询（自然语言）
        api_key: DashScope API Key
        max_results: 保留参数，当前由 DashScope 内部控制
        language: 查询语言
    Returns:
        WebResult 列表（通常 1 条，包含 LLM 搜索+综合后的结果）
    """
    if not api_key:
        return []

    try:
        import dashscope

        today = datetime.now().strftime("%Y年%m月%d日" if language == "zh" else "%B %d, %Y")

        if language == "zh":
            search_prompt = (
                f"今天是{today}。请联网搜索以下内容，返回具体数据、来源和日期。\n"
                f"重要：一定要搜索最新数据，不要依赖你的训练数据。你的知识可能已过时。\n"
                f"搜索内容：{query}"
            )
        else:
            search_prompt = (
                f"Today is {today}. Search the web for the following. "
                f"Return specific data, sources and dates.\n"
                f"IMPORTANT: Search for the LATEST data. Your training data may be outdated.\n"
                f"Search query: {query}"
            )

        resp = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": search_prompt}],
            result_format="message",
            api_key=api_key,
            enable_search=True,
            max_tokens=1000,
            temperature=0.1,
        )

        if resp.status_code == 200:
            try:
                text = resp.output.choices[0].message.content
            except (AttributeError, IndexError):
                return []

            if text and len(text.strip()) > 20:
                # 提取搜索来源信息
                search_refs = _extract_search_info(resp)

                return [WebResult(
                    title=f"搜索: {query[:80]}",
                    url=search_refs if search_refs else "",
                    snippet=text.strip(),
                )]
        return []
    except Exception:
        return []


def _extract_search_info(resp) -> str:
    """从 DashScope 响应中提取搜索引用/URL 信息。"""
    try:
        output = resp.output
        # 尝试多种方式提取 search_info
        if hasattr(output, 'search_info') and output.search_info:
            return str(output.search_info)
        if isinstance(output, dict):
            search_info = output.get('search_info', '')
            if search_info:
                return str(search_info)
    except Exception:
        pass

    # Fallback：从响应文本中提取 URL
    try:
        text = resp.output.choices[0].message.content
        urls = re.findall(r'https?://[^\s)\]）】】　]+', text)
        if urls:
            return '\n'.join(urls[:5])
    except Exception:
        pass

    return ""
