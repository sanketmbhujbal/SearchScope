from eval.run_qa_eval import build_test_sets, compute_latency_percentiles, top_k_passages


def test_compute_latency_percentiles_empty_input():
    stats = compute_latency_percentiles([])
    assert stats == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "max": 0.0}


def test_compute_latency_percentiles_basic_ordering():
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    stats = compute_latency_percentiles(latencies)
    # p50 should be roughly in the middle, p99 near the top, p95 above p50
    assert stats["p50"] <= stats["p95"] <= stats["p99"]
    assert stats["max"] == 10.0
    assert stats["avg"] == 5.5


def test_compute_latency_percentiles_single_value():
    stats = compute_latency_percentiles([2.5])
    assert stats["p50"] == stats["p95"] == stats["p99"] == stats["avg"] == stats["max"] == 2.5


def test_compute_latency_percentiles_p99_sensitive_to_outlier():
    """A single large outlier should show up in p99/max but shouldn't
    drag p50 up much — this is the whole point of using percentiles
    instead of avg/max alone."""
    latencies = [1.0] * 99 + [100.0]
    stats = compute_latency_percentiles(latencies)
    assert stats["p50"] == 1.0
    assert stats["max"] == 100.0
    assert stats["avg"] > 1.0  # the outlier does drag avg up, unlike p50


def test_top_k_passages_sorts_by_cross_encoder_score_descending():
    examples = [
        {"doc_id": "d1", "doc_text": "text1", "features": {"cross_encoder_score": 0.5}},
        {"doc_id": "d2", "doc_text": "text2", "features": {"cross_encoder_score": 0.9}},
        {"doc_id": "d3", "doc_text": "text3", "features": {"cross_encoder_score": 0.1}},
    ]
    top2 = top_k_passages(examples, 2)
    assert [p["doc_id"] for p in top2] == ["d2", "d1"]


def test_top_k_passages_respects_k():
    examples = [
        {"doc_id": f"d{i}", "doc_text": f"text{i}", "features": {"cross_encoder_score": float(i)}}
        for i in range(10)
    ]
    assert len(top_k_passages(examples, 3)) == 3


def _make_examples_by_query():
    return {
        "q1": [{"doc_id": "d1", "doc_text": "t1", "features": {"cross_encoder_score": 0.9}}],
        "q2": [{"doc_id": "d2", "doc_text": "t2", "features": {"cross_encoder_score": 0.8}}],
        "q3": [{"doc_id": "d3", "doc_text": "t3", "features": {"cross_encoder_score": 0.7}}],
    }


def test_build_test_sets_answerable_cases_use_own_query_passages():
    examples_by_query = _make_examples_by_query()
    queries = {"q1": "query one", "q2": "query two", "q3": "query three"}

    answerable, _unanswerable = build_test_sets(examples_by_query, queries, top_k=1, seed=42)

    own_doc_id = {"q1": "d1", "q2": "d2", "q3": "d3"}
    for case in answerable:
        assert case["passages"][0]["doc_id"] == own_doc_id[case["query_id"]]


def test_build_test_sets_unanswerable_cases_never_use_own_query_passages():
    examples_by_query = _make_examples_by_query()
    queries = {"q1": "query one", "q2": "query two", "q3": "query three"}

    _answerable, unanswerable = build_test_sets(examples_by_query, queries, top_k=1, seed=42)

    own_doc_id = {"q1": "d1", "q2": "d2", "q3": "d3"}
    for case in unanswerable:
        assert case["passages"][0]["doc_id"] != own_doc_id[case["query_id"]]
        assert case["context_from_query_id"] != case["query_id"]


def test_build_test_sets_covers_every_query_in_both_sets():
    examples_by_query = _make_examples_by_query()
    queries = {"q1": "query one", "q2": "query two", "q3": "query three"}

    answerable, unanswerable = build_test_sets(examples_by_query, queries, top_k=1, seed=42)

    assert {c["query_id"] for c in answerable} == {"q1", "q2", "q3"}
    assert {c["query_id"] for c in unanswerable} == {"q1", "q2", "q3"}


def test_build_test_sets_is_reproducible_with_same_seed():
    examples_by_query = _make_examples_by_query()
    queries = {"q1": "query one", "q2": "query two", "q3": "query three"}

    _a1, unanswerable_1 = build_test_sets(examples_by_query, queries, top_k=1, seed=7)
    _a2, unanswerable_2 = build_test_sets(examples_by_query, queries, top_k=1, seed=7)

    mapping_1 = {c["query_id"]: c["context_from_query_id"] for c in unanswerable_1}
    mapping_2 = {c["query_id"]: c["context_from_query_id"] for c in unanswerable_2}
    assert mapping_1 == mapping_2


def test_build_test_sets_skips_queries_with_no_candidates():
    examples_by_query = _make_examples_by_query()
    examples_by_query["q_empty"] = []
    queries = {"q1": "query one", "q2": "query two", "q3": "query three", "q_empty": "empty query"}

    answerable, unanswerable = build_test_sets(examples_by_query, queries, top_k=1, seed=42)

    assert "q_empty" not in {c["query_id"] for c in answerable}
    assert "q_empty" not in {c["query_id"] for c in unanswerable}
