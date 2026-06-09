from app.evaluation.metrics import calculate_fast_metrics, verdict_for


def test_fast_metrics_accept_namespaced_alphanumeric_citations() -> None:
    metrics, status = calculate_fast_metrics(
        answer="结论见 [KB-a3x9]，附件补充见 [ATT-z9q8]。",
        sources=[
            {"index": "KB-a3x9", "chunk_id": "doc_1_chunk_0"},
            {"index": "ATT-z9q8", "chunk_id": "attachment_1_chunk_0"},
        ],
        reference_chunk_ids=["doc_1_chunk_0"],
        reference_answer="结论和附件补充",
    )

    assert metrics["citation_validity"] == 1.0
    assert metrics["context_recall"] == 1.0
    assert status == {
        "citation_validity": "ok",
        "context_recall": "ok",
        "answer_token_recall": "ok",
    }


def test_missing_ground_truth_is_skipped_not_awarded() -> None:
    metrics, status = calculate_fast_metrics(
        answer="没有引用的回答",
        sources=[],
        reference_chunk_ids=[],
        reference_answer=None,
    )

    assert metrics == {
        "citation_validity": None,
        "context_recall": None,
        "answer_token_recall": None,
    }
    assert verdict_for(metrics, status) == "needs_review"


def test_fast_metrics_treats_b_and_billion_as_the_same_numeric_unit() -> None:
    metrics, status = calculate_fast_metrics(
        answer="The sparse stage trains 943.7 billion tokens [KB-a3x9].",
        sources=[{"index": "KB-a3x9", "chunk_id": "doc_1_chunk_0"}],
        reference_chunk_ids=["doc_1_chunk_0"],
        reference_answer="943.7B tokens",
    )

    assert metrics["answer_token_recall"] == 1.0
    assert status["answer_token_recall"] == "ok"


def test_low_ragas_score_requests_review_but_not_a_hard_failure() -> None:
    metrics = {
        "citation_validity": 1.0,
        "context_recall": 1.0,
        "answer_token_recall": 1.0,
        "faithfulness": 0.5,
        "factual_correctness": 0.0,
    }
    statuses = {name: "ok" for name in metrics}

    assert verdict_for(metrics, statuses) == "needs_review"


def test_low_deterministic_signal_is_a_hard_failure() -> None:
    metrics = {
        "citation_validity": 0.2,
        "faithfulness": 1.0,
        "factual_correctness": 1.0,
    }
    statuses = {name: "ok" for name in metrics}

    assert verdict_for(metrics, statuses) == "fail"


def test_ragas_error_requires_review_instead_of_a_pass() -> None:
    metrics = {
        "citation_validity": 1.0,
        "context_recall": 1.0,
        "answer_token_recall": 1.0,
        "faithfulness": None,
        "factual_correctness": 1.0,
    }
    statuses = {
        "citation_validity": "ok",
        "context_recall": "ok",
        "answer_token_recall": "ok",
        "faithfulness": "error:IncompleteOutputException",
        "factual_correctness": "ok",
    }

    assert verdict_for(metrics, statuses) == "needs_review"


def test_retrieval_metrics_use_pre_and_post_rerank_order() -> None:
    metrics, statuses = calculate_fast_metrics(
        answer="",
        sources=[],
        reference_chunk_ids=["doc_2_chunk_7"],
        reference_answer=None,
        retrieval_trace={
            "pre_rerank_candidates": [
                {"chunk_id": "doc_1_chunk_0"},
                {"chunk_id": "doc_2_chunk_7"},
            ],
            "final_candidates": [
                {"chunk_id": "doc_2_chunk_7"},
                {"chunk_id": "doc_1_chunk_0"},
            ],
        },
        expected_behavior="answer",
        include_answer_metrics=False,
    )

    assert metrics["recall_at_20"] == 1.0
    assert metrics["hit_at_4"] == 1.0
    assert metrics["mrr_at_4"] == 1.0
    assert metrics["ndcg_at_4"] == 1.0
    assert statuses["ndcg_at_4"] == "ok"


def test_graph_citations_do_not_count_as_vector_chunk_false_positives() -> None:
    metrics, statuses = calculate_fast_metrics(
        answer="Vector evidence [KB-a3x9] and graph evidence [KG-g7h8].",
        sources=[
            {
                "index": "KB-a3x9",
                "chunk_id": "doc_1_chunk_0",
                "source_type": "vector",
            },
            {
                "index": "KG-g7h8",
                "chunk_id": "kg-fact:abc123",
                "source_type": "kg",
            },
        ],
        reference_chunk_ids=["doc_1_chunk_0"],
        reference_answer="Vector and graph evidence",
        expected_behavior="answer",
    )

    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert statuses["citation_precision"] == "ok"
