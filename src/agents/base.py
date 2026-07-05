"""
Agent 基类 — 所有专业 Agent 共享的公共逻辑。

每个 Agent:
  1. 加载自己的 system prompt（来自 prompts/ 目录）
  2. 从 RAG 知识库检索相关上下文
  3. 调用 LLM 生成结构化的分析结果
  4. 返回结果给 Orchestrator
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.llm.client import LLMClient, LLMConfig
from src.rag.engine import VectorStore, RetrievalResult


# Prompt 目录（项目根 / prompts）
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _load_prompt(name: str, language: str = "zh") -> str:
    """从 prompts/{language}/ 目录加载 markdown 格式的 prompt 文件。"""
    path = _PROMPT_DIR / language / f"{name}.md"
    if not path.exists():
        # fallback: try zh if the requested language doesn't have this prompt
        path = _PROMPT_DIR / "zh" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


# ============================================================================
# Agent 输出数据结构
# ============================================================================

class AgentResult:
    """单个 Agent 的运行结果。"""

    __slots__ = ("agent_name", "content", "success", "error")

    def __init__(self, agent_name: str, content: str = "", success: bool = True,
                 error: str = ""):
        self.agent_name = agent_name
        self.content = content
        self.success = success
        self.error = error

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"<{self.agent_name} {status}>"


# ============================================================================
# Base Agent
# ============================================================================

class BaseAgent(ABC):
    """所有专业 Agent 的抽象基类。

    子类只需实现:
      - name: str                       — Agent 名称
      - prompt_file: str                — prompt 文件名（不含 .md 后缀）
      - _postprocess(raw: str) -> str   — 可选，对 LLM 输出做后处理

    然后调用 self.run(query, vector_store) 即可。
    """

    def __init__(self, llm_client: LLMClient, language: str = "zh"):
        self.llm = llm_client
        self.language = language
        self.system_prompt = _load_prompt(self.prompt_file, language)

    # ----------------------------------------------------------------
    # 子类必须定义
    # ----------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def prompt_file(self) -> str:
        ...

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def run(self, user_query: str, vector_store: VectorStore,
            context_from_other_agents: dict[str, str] | None = None,
            top_k: int = 20, api_key: str = "") -> AgentResult:
        """
        执行 Agent：
          1. RAG 检索相关文档
          2. 构建消息（system prompt + 检索上下文 + 用户问题）
          3. 调用 LLM
          4. 后处理 + 返回
        """
        try:
            # 1. RAG 检索 + 表格匹配
            rag_results, tables = self._retrieve(user_query, vector_store, top_k, api_key)

            # 2. 构建消息
            messages = self._build_messages(user_query, rag_results, tables,
                                            context_from_other_agents)

            # 3. 调用 LLM
            raw_output = self.llm.chat(messages)

            # 4. 后处理
            final = self._postprocess(raw_output)

            return AgentResult(agent_name=self.name, content=final)

        except Exception as exc:
            return AgentResult(agent_name=self.name, success=False, error=str(exc))

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _retrieve(self, query: str, store: VectorStore, top_k: int,
                  api_key: str = "") -> tuple[list[RetrievalResult], list]:
        """返回 (RAG文本检索结果, 相关表格列表)。"""
        if store.is_empty:
            return [], []
        # LLM 查询改写 → 多路精准检索
        extra_queries = []
        if api_key and len(query) > 10:
            try:
                from src.rag.engine import rewrite_query_for_search
                extra_queries = rewrite_query_for_search(query, api_key, self.language)
            except Exception:
                pass
        rag_results = store.search(query, top_k=top_k, api_key=api_key,
                                   extra_queries=extra_queries)
        # 表格关键词匹配
        table_results = store.search_tables(query, top_k=8)
        return rag_results, table_results

    def _build_messages(self, query: str, rag_results: list[RetrievalResult],
                        tables: list, other_context: dict[str, str] | None) -> list[dict]:
        parts = [self.system_prompt]

        # 注入 RAG 检索到的文档片段
        if rag_results:
            ctx_lines = ["\n\n## 检索到的文档片段（以下内容来自用户上传的文件）\n"]
            for i, rr in enumerate(rag_results, 1):
                ctx_lines.append(
                    f"### 片段 {i}  {rr.chunk.citation()}\n{rr.chunk.text}\n"
                )
            parts.append("".join(ctx_lines))

        # 注入相关表格
        if tables:
            tbl_lines = ["\n\n## 文档中的相关表格（从PDF原始表格提取）\n"]
            for i, t in enumerate(tables, 1):
                tbl_lines.append(f"### 表格 {i}  {t.citation()}\n")
                tbl_lines.append(t.to_markdown())
                tbl_lines.append("\n")
            parts.append("".join(tbl_lines))

        # 注入其他 Agent 的分析结果
        if other_context:
            ctx_lines = ["\n\n## 其他分析师的输出（供参考）\n"]
            for agent_name, content in other_context.items():
                ctx_lines.append(f"### {agent_name} 的输出\n{content}\n")
            parts.append("".join(ctx_lines))

        # 用户问题
        parts.append(f"\n\n## 用户问题\n{query}")

        system_content = "".join(parts)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

    def _postprocess(self, raw: str) -> str:
        """默认不做后处理。子类可 override。"""
        return raw
