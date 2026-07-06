"""
RAG Triad 评估器 — 基于 LLM-as-Judge 的量化评估体系。

三维度打分：
  - Context Relevance:  检索到的文档段落对回答问题有帮助吗？
  - Faithfulness:      回答严格基于文档/搜索结果，没有编造吗？
  - Answer Relevance:  回答正面回应用户问题了吗？

用法：
  python eval/evaluator.py --api-key sk-xxx --pdf path/to/file.pdf
  python eval/evaluator.py --api-key sk-xxx --pdf path/to/file.pdf --judge-model qwen-max
"""

import json
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient, LLMConfig
from src.rag.engine import build_rag_from_file, VectorStore
from src.orchestrator import Orchestrator


# ═══════════════════════════════════════════════════════════════
# LLM-as-Judge Prompts
# ═══════════════════════════════════════════════════════════════

JUDGE_CONTEXT_RELEVANCE = """你是 RAG 系统的评估裁判。请对文档检索结果的相关性打分。

## 用户问题
{question}

## 检索到的文档段落
{context}

## 评分标准
- 1.0: 检索段落直接包含回答问题所需的关键信息
- 0.7: 检索段落部分相关，但缺少一些关键数据
- 0.3: 检索段落只有边缘相关性
- 0.0: 检索段落与问题完全无关（如用户问利润，返回了目录页）

只输出一个 0.0 到 1.0 之间的数字和一句话理由。
格式: <分数> | <一句话理由>"""

JUDGE_FAITHFULNESS = """你是 RAG 系统的评估裁判。请对回答的忠实度打分——回答是否严格基于提供的文档和搜索结果，没有编造。

## 用户问题
{question}

## 系统回答
{answer}

## 参考依据（文档段落 + 搜索结果）
{context}

## 评分标准
- 1.0: 回答中所有事实性陈述都能在参考依据中找到支撑
- 0.7: 大部分有依据，个别细节无法验证但不是关键信息
- 0.3: 存在明显的编造或幻觉（如引用不存在的页码、编造数字）
- 0.0: 回答与参考依据完全矛盾或大篇幅编造

只输出一个 0.0 到 1.0 之间的数字和一句话理由。
格式: <分数> | <一句话理由>"""

JUDGE_ANSWER_RELEVANCE = """你是 RAG 系统的评估裁判。请对回答的切题度打分——回答是否正面、直接地回应用户的问题。

## 用户问题
{question}

## 系统回答
{answer}

## 评分标准
- 1.0: 第一句话直接给答案，后续内容紧扣问题，不跑题不炫技
- 0.7: 回答了问题但夹杂了一些不相关的内容
- 0.3: 绕了一大圈才触及问题核心，或大量内容与问题无关
- 0.0: 完全答非所问（如用户问利润，系统背诵了公司简介）

只输出一个 0.0 到 1.0 之间的数字和一句话理由。
格式: <分数> | <一句话理由>"""


# ═══════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════

class RAGEvaluator:
    """RAG Triad 评估器。"""

    def __init__(self, api_key: str, judge_model: str = "qwen-turbo"):
        self.api_key = api_key
        self.judge_model = judge_model
        self.judge_config = LLMConfig(
            api_key=api_key, model=judge_model,
            temperature=0.0, max_tokens=200
        )

    # ------------------------------------------------------------
    # 评分
    # ------------------------------------------------------------

    def _judge(self, prompt_template: str, **kwargs) -> tuple[float, str]:
        """调用裁判模型打分，返回 (分数, 理由)。"""
        prompt = prompt_template.format(**kwargs)
        client = LLMClient(self.judge_config)
        try:
            resp = client.chat([{"role": "user", "content": prompt}])
            # 解析 "0.85 | 理由..."
            parts = resp.strip().split("|", 1)
            score = float(parts[0].strip())
            reason = parts[1].strip() if len(parts) > 1 else ""
            return max(0.0, min(1.0, score)), reason
        except Exception as e:
            return 0.0, f"Judge error: {e}"

    def score_context_relevance(self, question: str, rag_chunks: list[str]) -> tuple[float, str]:
        """检索到的文档段落是否相关。"""
        context = "\n---\n".join(rag_chunks[:5]) if rag_chunks else "（无检索结果）"
        context = context[:3000]
        return self._judge(JUDGE_CONTEXT_RELEVANCE, question=question, context=context)

    def score_faithfulness(self, question: str, answer: str,
                           rag_chunks: list[str], web_results: str = "") -> tuple[float, str]:
        """回答是否忠于文档和搜索结果。"""
        ctx_parts = []
        if rag_chunks:
            ctx_parts.append("## 文档段落\n" + "\n---\n".join(rag_chunks[:3]))
        if web_results:
            ctx_parts.append("## 搜索结果\n" + web_results)
        context = "\n\n".join(ctx_parts)[:3000] if ctx_parts else "（无参考依据）"
        return self._judge(JUDGE_FAITHFULNESS, question=question, answer=answer, context=context)

    def score_answer_relevance(self, question: str, answer: str) -> tuple[float, str]:
        """回答是否切题。"""
        return self._judge(JUDGE_ANSWER_RELEVANCE, question=question, answer=answer[:2000])

    # ------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------

    def evaluate_one(self, test_case: dict, vector_store: VectorStore,
                     orchestrator: Orchestrator) -> dict:
        """评估单个测试用例。"""
        qid = test_case["id"]
        question = test_case["query"]
        qtype = test_case["type"]

        t0 = time.time()

        # 1. 跑 RAG 检索（独立于 Orchestrator，拿到原始 chunk）
        rag_chunks = []
        if not vector_store.is_empty:
            results = vector_store.search(question, top_k=5, api_key=self.api_key)
            rag_chunks = [f"[第{r.chunk.page}页] {r.chunk.text[:500]}" for r in results]

        # 2. 跑 Orchestrator（拿到系统回答）
        try:
            orch_result = orchestrator.run(
                question, vector_store,
                api_key=self.api_key,
                web_search_enabled=True,
                doc_type="半年报",
            )
            answer = orch_result.get("final_report", "")
        except Exception as e:
            answer = f"[系统异常] {e}"

        elapsed = time.time() - t0

        # 3. 三维打分
        cr_score, cr_reason = self.score_context_relevance(question, rag_chunks)

        # 收集搜索结果文本（用于 doc_has_answer=false 的题目）
        web_text = ""
        if not test_case.get("doc_has_answer", True):
            # 从执行日志中提取搜索结果
            for log in orch_result.get("execution_log", []):
                content = log.get("content", "")
                if "results" in content.lower() or "sample" in content.lower():
                    web_text += content + "\n"
            # 也从 final_report 本身提取
            web_text += "\n[Final Report]\n" + answer[:1000]

        faith_score, faith_reason = self.score_faithfulness(
            question, answer, rag_chunks, web_text
        )
        ar_score, ar_reason = self.score_answer_relevance(question, answer)

        return {
            "id": qid,
            "type": qtype,
            "query": question,
            "answer": answer[:500],
            "context_relevance": cr_score,
            "context_relevance_reason": cr_reason,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "answer_relevance": ar_score,
            "answer_relevance_reason": ar_reason,
            "triad_avg": round((cr_score + faith_score + ar_score) / 3, 3),
            "elapsed_sec": round(elapsed, 1),
        }

    def evaluate_all(self, test_cases: list[dict], vector_store: VectorStore,
                     orchestrator: Orchestrator, parallel: bool = True) -> list[dict]:
        """批量评估全部测试用例。"""
        results = []

        if parallel:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    pool.submit(self.evaluate_one, tc, vector_store, orchestrator): tc
                    for tc in test_cases
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for tc in test_cases:
                results.append(self.evaluate_one(tc, vector_store, orchestrator))

        # 按 id 排序
        results.sort(key=lambda r: r["id"])
        return results

    # ------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------

    def report(self, results: list[dict]) -> str:
        """生成评估报告。"""
        lines = []
        lines.append("=" * 65)
        lines.append("  FinRisk MultiAgent — RAG Triad 评估报告")
        lines.append("=" * 65)
        lines.append(f"  裁判模型: {self.judge_model}")
        lines.append(f"  测试用例: {len(results)} 个")
        lines.append(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 逐条明细
        lines.append(f"{'ID':<5} {'类型':<14} {'CR':>5} {'Faith':>5} {'AR':>5} {'Triad':>6} {'耗时':>6}")
        lines.append("-" * 55)
        for r in results:
            lines.append(
                f"{r['id']:<5} {r['type']:<14} "
                f"{r['context_relevance']:>5.2f} {r['faithfulness']:>5.2f} "
                f"{r['answer_relevance']:>5.2f} {r['triad_avg']:>6.2f} "
                f"{r['elapsed_sec']:>5.1f}s"
            )
        lines.append("")

        # 统计
        cr_avg = sum(r["context_relevance"] for r in results) / len(results)
        faith_avg = sum(r["faithfulness"] for r in results) / len(results)
        ar_avg = sum(r["answer_relevance"] for r in results) / len(results)
        triad_avg = sum(r["triad_avg"] for r in results) / len(results)

        lines.append("─" * 55)
        lines.append("  综合统计:")
        lines.append(f"    Context Relevance 平均: {cr_avg:.3f}")
        lines.append(f"    Faithfulness      平均: {faith_avg:.3f}")
        lines.append(f"    Answer Relevance  平均: {ar_avg:.3f}")
        lines.append(f"    Triad Avg         平均: {triad_avg:.3f}")
        lines.append("")

        # 按类型统计
        for qtype in ["factual", "analytical", "comprehensive"]:
            typed = [r for r in results if r["type"] == qtype]
            if typed:
                avg = sum(r["triad_avg"] for r in typed) / len(typed)
                lines.append(f"    {qtype} ({len(typed)}个): Triad Avg = {avg:.3f}")

        lines.append("")
        lines.append("─" * 55)
        lines.append("  评分维度说明:")
        lines.append("    CR  = Context Relevance (检索相关性)")
        lines.append("    Faith = Faithfulness (忠实度/反幻觉)")
        lines.append("    AR  = Answer Relevance (切题度)")
        lines.append("    Triad = (CR + Faith + AR) / 3")
        lines.append("=" * 65)

        # 问题详情
        lines.append("\n\n## 各题评分详情\n")
        for r in results:
            lines.append(f"### {r['id']} ({r['type']}) — Triad: {r['triad_avg']:.2f}")
            lines.append(f"> {r['query']}")
            lines.append(f"")
            lines.append(f"- Context Relevance ({r['context_relevance']:.2f}): {r['context_relevance_reason']}")
            lines.append(f"- Faithfulness ({r['faithfulness']:.2f}): {r['faithfulness_reason']}")
            lines.append(f"- Answer Relevance ({r['answer_relevance']:.2f}): {r['answer_relevance_reason']}")
            lines.append(f"- 回答摘要: {r['answer'][:200]}...")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FinRisk RAG Triad Evaluator")
    parser.add_argument("--api-key", required=True, help="DashScope API Key")
    parser.add_argument("--pdf", required=True, help="测试用 PDF 文件路径")
    parser.add_argument("--judge-model", default="qwen-turbo",
                       help="裁判模型 (default: qwen-turbo)")
    parser.add_argument("--output", default=None, help="报告输出路径 (default: stdout)")
    parser.add_argument("--golden-set", default=None,
                       help="Golden set 路径 (default: eval/golden_set.json)")
    parser.add_argument("--model", default="qwen-plus",
                       help="Orchestrator 用模型 (default: qwen-plus)")
    args = parser.parse_args()

    # 加载 Golden Set
    golden_path = args.golden_set or str(ROOT / "eval" / "golden_set.json")
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    test_cases = golden["queries"]

    print(f"[1/4] 加载 Golden Set: {len(test_cases)} 条测试用例")
    print(f"[2/4] 构建 RAG (PDF: {args.pdf})...")
    t0 = time.time()
    vector_store = build_rag_from_file(args.pdf, api_key=args.api_key)
    print(f"      完成: {vector_store.chunk_count} chunks, {vector_store.table_count} tables ({time.time()-t0:.1f}s)")

    print(f"[3/4] 初始化 Orchestrator (model={args.model})...")
    config = LLMConfig(api_key=args.api_key, model=args.model,
                       temperature=0.3, max_tokens=4096)
    orch = Orchestrator(LLMClient(config), language="zh")

    print(f"[4/4] 开始评估 (judge={args.judge_model})...")
    evaluator = RAGEvaluator(args.api_key, judge_model=args.judge_model)
    results = evaluator.evaluate_all(test_cases, vector_store, orch, parallel=True)

    report = evaluator.report(results)

    # 保存报告
    out_path = args.output or str(ROOT / "eval" / "report_latest.md")
    Path(out_path).write_text(report, encoding="utf-8")
    print(f"\n[DONE] 报告: {out_path}")

    # 摘要
    cr_avg = sum(r["context_relevance"] for r in results) / len(results)
    faith_avg = sum(r["faithfulness"] for r in results) / len(results)
    ar_avg = sum(r["answer_relevance"] for r in results) / len(results)
    triad_avg = (cr_avg + faith_avg + ar_avg) / 3
    print(f"CR={cr_avg:.2f} Faith={faith_avg:.2f} AR={ar_avg:.2f} Triad={triad_avg:.2f}")
    for r in results:
        print(f"  {r['id']} {r['type']:<14} Triad={r['triad_avg']:.2f} ({r['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
