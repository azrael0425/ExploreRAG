"""
ExploreRAG Evaluation Script — local RAG quality assessment.

Uses DeepEval metrics (LLM-as-judge) + custom rule-based checks.
The optional LLM judge uses the project's existing Qwen API configuration.

Usage:
    cd ExploreRAG/backend
    source ../venv/bin/activate
    python scripts/eval_rag.py --workspace 11 [--judge qwen]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8080/api/v1/rag"
TIMEOUT = 120  # seconds per request


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    category: str  # fact_extraction, table_data, cross_doc, anti_hallucination, history, citation
    question: str
    language: str  # vi, en
    history: list[dict] = field(default_factory=list)
    # Ground truth (optional — for reference-based metrics)
    expected_answer: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    expected_refuse: bool = False  # Should the system refuse to answer?
    # Results (filled after evaluation)
    answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    source_count: int = 0
    latency_ms: float = 0


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    score: float  # 0.0 - 1.0
    passed: bool
    reason: str = ""


@dataclass
class TestResult:
    """Full evaluation result for a test case."""
    test_id: str
    category: str
    question: str
    language: str
    answer_preview: str
    source_count: int
    latency_ms: float
    metrics: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0


# ── Test Dataset ──────────────────────────────────────────────────────────────

def build_test_cases(workspace_id: int) -> list[TestCase]:
    """
    Hand-crafted test cases for KBG9 workspace (id=11).
    Documents:
      - doc 11: TechVina annual report 2025 (Vietnamese)
      - doc 12: DeepSeek-V3.2 technical paper (English)
    """
    cases = [
        # ── Fact Extraction (Vietnamese doc) ──
        TestCase(
            id="FACT-VI-01",
            category="fact_extraction",
            question="TechVina được thành lập năm nào và hoạt động ở bao nhiêu quốc gia?",
            language="vi",
            expected_keywords=["2010", "12"],
        ),
        TestCase(
            id="FACT-VI-02",
            category="fact_extraction",
            question="Doanh thu của TechVina năm 2025 là bao nhiêu và tăng trưởng bao nhiêu phần trăm?",
            language="vi",
            expected_keywords=["4.850", "4850", "23,4", "23.4"],
        ),
        TestCase(
            id="FACT-VI-03",
            category="fact_extraction",
            question="TechVina có bao nhiêu nhân sự và phân bổ theo trình độ như thế nào?",
            language="vi",
            expected_keywords=["3.200", "3200"],
        ),
        # ── Fact Extraction (English doc) ──
        TestCase(
            id="FACT-EN-01",
            category="fact_extraction",
            question="What are the key technical breakthroughs of DeepSeek-V3.2?",
            language="en",
            expected_keywords=["DSA", "Sparse Attention", "reinforcement", "RL"],
        ),
        TestCase(
            id="FACT-EN-02",
            category="fact_extraction",
            question="What competitions did DeepSeek-V3.2 achieve gold-medal performance in?",
            language="en",
            expected_keywords=["IMO", "IOI"],
        ),
        # ── Table Data Extraction ──
        TestCase(
            id="TABLE-01",
            category="table_data",
            question="Cho tôi biết doanh thu thuần và EBITDA của TechVina từ 2023-2025?",
            language="vi",
            expected_keywords=["3.180", "3180", "4.850", "4850", "890"],
        ),
        TestCase(
            id="TABLE-02",
            category="table_data",
            question="Biên lợi nhuận gộp và ROE của TechVina qua các năm 2023-2025 là bao nhiêu?",
            language="vi",
            expected_keywords=["40", "42", "44", "12,8", "15,6", "18,7"],
        ),
        TestCase(
            id="TABLE-03",
            category="table_data",
            question="DeepSeek-V3.2 đạt kết quả bao nhiêu trên AIME 2025 và HMMT Feb 2025?",
            language="vi",
            expected_keywords=["93.1", "AIME"],
        ),
        # ── Cross-Document Reasoning ──
        TestCase(
            id="CROSS-01",
            category="cross_doc",
            question="TechVina có mảng AI Platform không? Doanh thu mảng này là bao nhiêu? Và DeepSeek-V3.2 có những khả năng AI gì nổi bật?",
            language="vi",
            expected_keywords=["AI Platform", "900", "DSA"],
        ),
        TestCase(
            id="CROSS-02",
            category="cross_doc",
            question="TechVina đầu tư bao nhiêu cho R&D? DeepSeek-V3.2 có đóng góp gì cho cộng đồng open-source?",
            language="vi",
            expected_keywords=["R&D", "12"],
        ),
        # ── Anti-Hallucination (should refuse) ──
        TestCase(
            id="ANTI-01",
            category="anti_hallucination",
            question="Elon Musk sinh năm bao nhiêu?",
            language="vi",
            expected_refuse=True,
        ),
        TestCase(
            id="ANTI-02",
            category="anti_hallucination",
            question="Cách nấu phở Hà Nội ngon nhất?",
            language="vi",
            expected_refuse=True,
        ),
        TestCase(
            id="ANTI-03",
            category="anti_hallucination",
            question="Bitcoin giá bao nhiêu hôm nay?",
            language="vi",
            expected_refuse=True,
        ),
        # ── History / Follow-up ──
        TestCase(
            id="HIST-01",
            category="history",
            question="Mảng nào tăng trưởng mạnh nhất?",
            language="vi",
            history=[
                {"role": "user", "content": "TechVina có những mảng kinh doanh chính nào?"},
                {"role": "assistant", "content": "TechVina có 4 mảng kinh doanh chính:\n1. Giải pháp phần mềm: 1.890 tỷ VNĐ\n2. Dịch vụ Cloud: 1.520 tỷ VNĐ\n3. AI Platform: 900 tỷ VNĐ\n4. Tư vấn & Triển khai: 540 tỷ VNĐ"},
            ],
            expected_keywords=["AI Platform", "66", "67"],
        ),
        TestCase(
            id="HIST-02",
            category="history",
            question="Giải thích chi tiết hơn về điểm đầu tiên",
            language="vi",
            history=[
                {"role": "user", "content": "DeepSeek-V3.2 có những đặc điểm kỹ thuật nào nổi bật?"},
                {"role": "assistant", "content": "DeepSeek-V3.2 có 3 đặc điểm kỹ thuật nổi bật:\n1. DeepSeek Sparse Attention (DSA) - cơ chế attention hiệu quả\n2. Scalable RL framework - mở rộng tính toán post-training\n3. Agentic Task Synthesis - pipeline tạo dữ liệu cho agent"},
            ],
            expected_keywords=["DSA", "Sparse Attention", "lightning", "indexer"],
        ),
        # ── Citation Quality ──
        TestCase(
            id="CITE-01",
            category="citation",
            question="TechVina thực hiện thương vụ M&A nào năm 2025 và giá trị bao nhiêu?",
            language="vi",
            expected_keywords=["DataStream", "Singapore", "45"],
        ),
    ]
    return cases


# ── Rule-based metrics (no LLM needed) ────────────────────────────────────

def eval_keyword_coverage(tc: TestCase) -> MetricResult:
    """Check if expected keywords appear in the answer."""
    if not tc.expected_keywords:
        return MetricResult("keyword_coverage", 1.0, True, "No keywords to check")

    found = 0
    missing = []
    for kw in tc.expected_keywords:
        if kw.lower() in tc.answer.lower():
            found += 1
        else:
            missing.append(kw)

    score = found / len(tc.expected_keywords) if tc.expected_keywords else 1.0
    passed = score >= 0.5  # At least half the keywords
    reason = f"{found}/{len(tc.expected_keywords)} keywords found"
    if missing:
        reason += f". Missing: {missing}"
    return MetricResult("keyword_coverage", score, passed, reason)


def eval_refusal_accuracy(tc: TestCase) -> MetricResult:
    """Check if the system correctly refused (or didn't refuse) to answer.

    Distinguishes between:
    - Full refusal: entire answer is a refusal (e.g., "Tài liệu không chứa thông tin này.")
    - Partial gap noting: answer provides data but notes some gaps (acceptable)
    """
    refusal_phrases = [
        "không chứa thông tin",
        "không có thông tin",
        "tài liệu không",
        "not contain",
        "no relevant information",
    ]
    answer_lower = tc.answer.lower()

    # Count refusal phrase occurrences
    refusal_hits = sum(1 for p in refusal_phrases if p in answer_lower)

    # Check if answer is MOSTLY a refusal (short + refusal phrase)
    word_count = len(tc.answer.split())
    is_full_refusal = refusal_hits > 0 and word_count < 20
    # Partial gap: answer has substance + notes some gaps
    is_partial_gap = refusal_hits > 0 and word_count >= 20

    if tc.expected_refuse:
        if refusal_hits > 0:
            return MetricResult("refusal_accuracy", 1.0, True, "Correctly refused")
        else:
            return MetricResult("refusal_accuracy", 0.0, False,
                                "Should have refused but answered")
    else:
        if is_full_refusal:
            return MetricResult("refusal_accuracy", 0.0, False,
                                "Over-refusal: entire answer is a refusal")
        elif is_partial_gap:
            # Answer provides some data but notes gaps — this is acceptable behavior
            return MetricResult("refusal_accuracy", 0.8, True,
                                "Partial answer with noted gaps (acceptable)")
        else:
            return MetricResult("refusal_accuracy", 1.0, True, "Correctly answered")


def eval_phantom_citations(tc: TestCase) -> MetricResult:
    """Check for phantom citations when refusing to answer.

    Only flags as phantom if:
    - The answer is a FULL refusal (short, no substantive content) AND has citations
    - OR a refusal sentence itself contains citations (e.g., "Không có thông tin [1]")

    Does NOT flag if the answer provides useful data with citations + notes some gaps.
    """
    refusal_phrases = ["không chứa", "không có thông tin", "not contain", "no information"]
    answer_lower = tc.answer.lower()
    word_count = len(tc.answer.split())

    # Full refusal with citations = phantom
    is_full_refusal = any(p in answer_lower for p in refusal_phrases) and word_count < 20
    all_citations = re.findall(r'\[(?:IMG-)?\d+\]', tc.answer)

    if is_full_refusal and all_citations:
        return MetricResult("no_phantom_citations", 0.0, False,
                            f"Phantom citations on full refusal: {all_citations}")

    # Check for citations IN refusal sentences specifically
    sentences = re.split(r'[.!?\n]', tc.answer)
    phantom_in_sentence = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if any(p in sent_lower for p in refusal_phrases):
            sent_citations = re.findall(r'\[(?:IMG-)?\d+\]', sent)
            if sent_citations:
                phantom_in_sentence.extend(sent_citations)

    if phantom_in_sentence:
        return MetricResult("no_phantom_citations", 0.3, False,
                            f"Citations in refusal sentences: {phantom_in_sentence}")

    return MetricResult("no_phantom_citations", 1.0, True, "No phantom citations")


def eval_citation_format(tc: TestCase) -> MetricResult:
    """Check citation format: [1] [2] not [1, 2] or [1][2]."""
    # Check for grouped citations (bad: [1, 2] or [1,2])
    grouped = re.findall(r'\[\d+[,\s]+\d+\]', tc.answer)
    if grouped:
        return MetricResult("citation_format", 0.0, False,
                            f"Grouped citations found: {grouped}")
    return MetricResult("citation_format", 1.0, True, "Citations properly formatted")


def eval_token_artifacts(tc: TestCase) -> MetricResult:
    """Check for provider token artifacts like <unusedNNN>."""
    artifacts = re.findall(r'<unused\d+>:?\s*', tc.answer)
    if artifacts:
        return MetricResult("no_token_artifacts", 0.0, False,
                            f"Token artifacts: {artifacts}")
    return MetricResult("no_token_artifacts", 1.0, True, "No token artifacts")


def eval_language_match(tc: TestCase) -> MetricResult:
    """Check if answer language matches question language."""
    # Simple heuristic: Vietnamese has many diacritical marks
    vn_chars = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    vn_count = sum(1 for c in tc.answer.lower() if c in vn_chars)
    total_alpha = sum(1 for c in tc.answer if c.isalpha())

    if total_alpha == 0:
        return MetricResult("language_match", 1.0, True, "No text to check")

    vn_ratio = vn_count / total_alpha

    if tc.language == "vi":
        # Vietnamese question should get Vietnamese answer
        # Allow some English (technical terms), but at least 5% VN chars
        if vn_ratio > 0.03:
            return MetricResult("language_match", 1.0, True,
                                f"Vietnamese content detected ({vn_ratio:.1%})")
        else:
            return MetricResult("language_match", 0.0, False,
                                f"Expected Vietnamese but got mostly English ({vn_ratio:.1%})")
    else:
        return MetricResult("language_match", 1.0, True, "English response OK")


def eval_answer_completeness(tc: TestCase) -> MetricResult:
    """Check if answer has substance (not just a one-liner refusal for non-refusal cases)."""
    if tc.expected_refuse:
        return MetricResult("answer_completeness", 1.0, True, "Refusal case — skip")

    word_count = len(tc.answer.split())
    if word_count < 10:
        return MetricResult("answer_completeness", 0.2, False,
                            f"Answer too short ({word_count} words)")
    elif word_count < 30:
        return MetricResult("answer_completeness", 0.6, True,
                            f"Brief answer ({word_count} words)")
    else:
        return MetricResult("answer_completeness", 1.0, True,
                            f"Detailed answer ({word_count} words)")


def eval_context_utilization(tc: TestCase) -> MetricResult:
    """Check if retrieved contexts are actually being cited in the answer."""
    if tc.expected_refuse or tc.source_count == 0:
        return MetricResult("context_utilization", 1.0, True, "Skip — refusal or no sources")

    citations = re.findall(r'\[(\d+)\]', tc.answer)
    cited_indices = set(int(c) for c in citations)
    if not cited_indices:
        return MetricResult("context_utilization", 0.0, False,
                            "Answer uses sources but cites none")

    ratio = len(cited_indices) / max(tc.source_count, 1)
    score = min(ratio, 1.0)
    return MetricResult("context_utilization", score,
                        score >= 0.2,
                        f"Cited {len(cited_indices)}/{tc.source_count} sources")


# ── LLM-as-judge metrics via DeepEval ─────────────────────────────────────

def get_deepeval_model(judge: str):
    """Get DeepEval model wrapper for the judge LLM."""
    if judge == "qwen":
        from deepeval.models import DeepEvalBaseLLM

        class QwenJudge(DeepEvalBaseLLM):
            def __init__(self):
                from dotenv import load_dotenv

                load_dotenv(Path(__file__).resolve().parents[2] / ".env")
                self.model_name = os.getenv("LLM_MODEL_FAST", "qwen-plus")
                self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
                self.base_url = os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                if not self.api_key:
                    raise ValueError("DASHSCOPE_API_KEY is required for --judge qwen")

            def load_model(self):
                from openai import OpenAI

                return OpenAI(api_key=self.api_key, base_url=self.base_url)

            def generate(self, prompt: str, schema=None) -> str:
                client = self.load_model()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content or ""

            async def a_generate(self, prompt: str, schema=None) -> str:
                return self.generate(prompt, schema)

            def get_model_name(self):
                return self.model_name

        return QwenJudge()

    else:
        raise ValueError(f"Unknown judge: {judge}")


def run_deepeval_metrics(tc: TestCase, judge_model) -> list[MetricResult]:
    """Run DeepEval LLM-as-judge metrics on a test case."""
    results = []

    if tc.expected_refuse:
        return results  # Skip LLM metrics for refusal cases

    if not tc.retrieved_contexts:
        return results

    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
        )

        deepeval_tc = LLMTestCase(
            input=tc.question,
            actual_output=tc.answer,
            retrieval_context=tc.retrieved_contexts[:5],  # Limit to avoid token overflow
        )

        metrics = [
            ("faithfulness", FaithfulnessMetric(model=judge_model, threshold=0.7)),
            ("answer_relevancy", AnswerRelevancyMetric(model=judge_model, threshold=0.7)),
            ("context_relevancy", ContextualRelevancyMetric(model=judge_model, threshold=0.5)),
        ]

        for name, metric in metrics:
            try:
                metric.measure(deepeval_tc)
                results.append(MetricResult(
                    name=name,
                    score=metric.score or 0.0,
                    passed=metric.is_successful(),
                    reason=str(metric.reason)[:200] if metric.reason else "",
                ))
            except Exception as e:
                results.append(MetricResult(
                    name=name, score=0.0, passed=False,
                    reason=f"Error: {str(e)[:150]}",
                ))

    except Exception as e:
        results.append(MetricResult(
            name="deepeval_error", score=0.0, passed=False,
            reason=f"DeepEval setup error: {str(e)[:200]}",
        ))

    return results


# ── Main evaluation runner ────────────────────────────────────────────────

def call_debug_chat(workspace_id: int, tc: TestCase) -> dict:
    """Call the debug-chat endpoint and return full response."""
    payload = {"message": tc.question}
    if tc.history:
        payload["history"] = tc.history

    start = time.time()
    r = requests.post(
        f"{BASE_URL}/debug-chat/{workspace_id}",
        json=payload,
        timeout=TIMEOUT,
    )
    latency = (time.time() - start) * 1000
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = latency
    return data


def evaluate_test_case(tc: TestCase, judge_model=None) -> TestResult:
    """Run all metrics on a single test case."""
    # Rule-based metrics (always run)
    rule_metrics = [
        eval_keyword_coverage(tc),
        eval_refusal_accuracy(tc),
        eval_phantom_citations(tc),
        eval_citation_format(tc),
        eval_token_artifacts(tc),
        eval_language_match(tc),
        eval_answer_completeness(tc),
        eval_context_utilization(tc),
    ]

    # LLM-as-judge metrics (optional)
    llm_metrics = []
    if judge_model and not tc.expected_refuse and tc.retrieved_contexts:
        llm_metrics = run_deepeval_metrics(tc, judge_model)

    all_metrics = rule_metrics + llm_metrics

    # Overall score = weighted average
    scores = [m.score for m in all_metrics if m.score >= 0]
    overall = sum(scores) / len(scores) if scores else 0.0

    return TestResult(
        test_id=tc.id,
        category=tc.category,
        question=tc.question[:60] + "..." if len(tc.question) > 60 else tc.question,
        language=tc.language,
        answer_preview=tc.answer[:80] + "..." if len(tc.answer) > 80 else tc.answer,
        source_count=tc.source_count,
        latency_ms=tc.latency_ms,
        metrics=all_metrics,
        overall_score=overall,
    )


def print_results_table(results: list[TestResult], show_llm: bool = False):
    """Print detailed evaluation results as formatted table."""

    # ── Per-test results ──
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for r in results:
        status = "PASS" if r.overall_score >= 0.7 else "PARTIAL" if r.overall_score >= 0.5 else "FAIL"
        icon = "✓" if status == "PASS" else "~" if status == "PARTIAL" else "✗"

        print(f"\n{icon} [{r.test_id}] ({r.category}) score={r.overall_score:.2f} | {r.latency_ms:.0f}ms | {r.source_count} sources")
        print(f"  Q: {r.question}")
        print(f"  A: {r.answer_preview}")

        for m in r.metrics:
            m_icon = "✓" if m.passed else "✗"
            suffix = f" — {m.reason}" if m.reason and not m.passed else ""
            if not m.passed or m.score < 1.0:
                print(f"    {m_icon} {m.name}: {m.score:.2f}{suffix}")

    # ── Summary by category ──
    print("\n" + "=" * 120)
    print("SUMMARY BY CATEGORY")
    print("=" * 120)

    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print(f"\n{'Category':<22} {'Tests':>5} {'Pass':>5} {'Avg Score':>10} {'Avg Latency':>12}")
    print("-" * 60)

    for cat, cat_results in sorted(categories.items()):
        total = len(cat_results)
        passed = sum(1 for r in cat_results if r.overall_score >= 0.7)
        avg_score = sum(r.overall_score for r in cat_results) / total
        avg_latency = sum(r.latency_ms for r in cat_results) / total
        print(f"{cat:<22} {total:>5} {passed:>5} {avg_score:>9.2f} {avg_latency:>10.0f}ms")

    # ── Summary by metric ──
    print("\n" + "=" * 120)
    print("SUMMARY BY METRIC")
    print("=" * 120)

    metric_scores: dict[str, list[float]] = {}
    metric_passes: dict[str, list[bool]] = {}
    for r in results:
        for m in r.metrics:
            metric_scores.setdefault(m.name, []).append(m.score)
            metric_passes.setdefault(m.name, []).append(m.passed)

    print(f"\n{'Metric':<25} {'Avg Score':>10} {'Pass Rate':>10} {'Count':>6}")
    print("-" * 55)

    for name in sorted(metric_scores.keys()):
        scores = metric_scores[name]
        passes = metric_passes[name]
        avg = sum(scores) / len(scores)
        pass_rate = sum(passes) / len(passes)
        print(f"{name:<25} {avg:>9.2f} {pass_rate:>9.0%} {len(scores):>6}")

    # ── Overall ──
    print("\n" + "=" * 120)
    all_scores = [r.overall_score for r in results]
    avg_overall = sum(all_scores) / len(all_scores)
    pass_count = sum(1 for s in all_scores if s >= 0.7)
    total_tests = len(results)

    print(f"OVERALL SCORE: {avg_overall:.2f} | PASS: {pass_count}/{total_tests} | "
          f"AVG LATENCY: {sum(r.latency_ms for r in results) / total_tests:.0f}ms")
    print("=" * 120)

    # ── Final verdict ──
    if avg_overall >= 0.85:
        print("\nVerdict: EXCELLENT — Strong result for the measured local benchmark")
    elif avg_overall >= 0.7:
        print("\nVerdict: GOOD — Acceptable for the measured local benchmark")
    elif avg_overall >= 0.5:
        print("\nVerdict: FAIR — Needs improvement against the benchmark gates")
    else:
        print("\nVerdict: POOR — Significant issues to address")


def main():
    parser = argparse.ArgumentParser(description="ExploreRAG Evaluation")
    parser.add_argument("--workspace", type=int, default=11, help="Workspace ID")
    parser.add_argument("--judge", choices=["qwen", "none"], default="none",
                        help="LLM judge for DeepEval metrics (default: none = rule-based only)")
    parser.add_argument("--test-ids", nargs="*", help="Run specific test IDs only")
    args = parser.parse_args()

    print(f"ExploreRAG Evaluation — Workspace {args.workspace}")
    print(f"Judge: {args.judge}")
    print(f"Endpoint: {BASE_URL}")

    # Verify server is running
    try:
        r = requests.get("http://localhost:8080/health", timeout=5)
        r.raise_for_status()
        print("Server: OK\n")
    except Exception:
        print("ERROR: Server not reachable at localhost:8080")
        sys.exit(1)

    # Build test cases
    test_cases = build_test_cases(args.workspace)
    if args.test_ids:
        test_cases = [tc for tc in test_cases if tc.id in args.test_ids]

    print(f"Running {len(test_cases)} test cases...\n")

    # Setup judge model (if requested)
    judge_model = None
    if args.judge != "none":
        try:
            judge_model = get_deepeval_model(args.judge)
            print(f"Judge model: {judge_model.get_model_name()}")
        except Exception as e:
            print(f"WARNING: Failed to init judge model: {e}")
            print("Falling back to rule-based only.\n")

    # Run evaluation
    results: list[TestResult] = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc.id}: {tc.question[:50]}...", end=" ", flush=True)

        try:
            data = call_debug_chat(args.workspace, tc)
            tc.answer = data.get("answer", "")
            tc.source_count = data.get("total_sources", 0)
            tc.latency_ms = data.get("_latency_ms", 0)

            # Extract retrieved contexts for DeepEval
            tc.retrieved_contexts = [
                s.get("content_preview", "")
                for s in data.get("retrieved_sources", [])
            ]

            result = evaluate_test_case(tc, judge_model)
            results.append(result)

            icon = "✓" if result.overall_score >= 0.7 else "✗"
            print(f"{icon} {result.overall_score:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append(TestResult(
                test_id=tc.id, category=tc.category,
                question=tc.question[:60], language=tc.language,
                answer_preview=f"ERROR: {e}", source_count=0,
                latency_ms=0, overall_score=0.0,
            ))

    # Print results
    print_results_table(results, show_llm=(args.judge != "none"))

    # Save JSON results
    output_path = Path(__file__).parent / "eval_results.json"
    json_results = []
    for r in results:
        json_results.append({
            "test_id": r.test_id,
            "category": r.category,
            "question": r.question,
            "language": r.language,
            "overall_score": r.overall_score,
            "source_count": r.source_count,
            "latency_ms": r.latency_ms,
            "metrics": [{"name": m.name, "score": m.score, "passed": m.passed, "reason": m.reason} for m in r.metrics],
        })
    output_path.write_text(json.dumps(json_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
