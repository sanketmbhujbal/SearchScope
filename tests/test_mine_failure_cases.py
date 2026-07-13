from eval.mine_failure_cases import find_best_vocabulary_mismatch_case, score_vocabulary_mismatch


def test_score_vocabulary_mismatch_detects_doc_missing_from_bm25():
    examples = [
        {"doc_id": "d_relevant", "features": {"bm25_rank": 0.0, "dense_rank": 2, "term_overlap": 0.1}},
        {"doc_id": "d_bm25_top1", "features": {"bm25_rank": 1, "dense_rank": 50, "term_overlap": 0.8}},
    ]
    result = score_vocabulary_mismatch(examples, {"d_relevant": 3})

    assert result is not None
    assert result["relevant_doc_id"] == "d_relevant"
    assert result["relevant_doc_bm25_rank"] == 0.0
    assert result["bm25_top1_doc_id"] == "d_bm25_top1"


def test_score_vocabulary_mismatch_returns_none_when_bm25_did_fine():
    examples = [
        {"doc_id": "d_relevant", "features": {"bm25_rank": 1, "dense_rank": 1, "term_overlap": 0.9}},
    ]
    result = score_vocabulary_mismatch(examples, {"d_relevant": 3})
    assert result is None


def test_score_vocabulary_mismatch_returns_none_with_no_judged_candidate():
    examples = [
        {"doc_id": "d1", "features": {"bm25_rank": 1, "dense_rank": 1, "term_overlap": 0.5}},
    ]
    result = score_vocabulary_mismatch(examples, {})
    assert result is None


def test_score_vocabulary_mismatch_detects_much_worse_bm25_rank():
    examples = [
        {"doc_id": "d_relevant", "features": {"bm25_rank": 30, "dense_rank": 2, "term_overlap": 0.2}},
    ]
    result = score_vocabulary_mismatch(examples, {"d_relevant": 2})
    assert result is not None
    assert result["relevant_doc_bm25_rank"] == 30


def test_score_vocabulary_mismatch_picks_highest_graded_doc_when_multiple_judged():
    examples = [
        {"doc_id": "d_low_grade", "features": {"bm25_rank": 0.0, "dense_rank": 5, "term_overlap": 0.1}},
        {"doc_id": "d_high_grade", "features": {"bm25_rank": 0.0, "dense_rank": 3, "term_overlap": 0.1}},
    ]
    result = score_vocabulary_mismatch(examples, {"d_low_grade": 1, "d_high_grade": 3})
    assert result["relevant_doc_id"] == "d_high_grade"
    assert result["relevant_doc_grade"] == 3


def test_find_best_vocabulary_mismatch_case_sorts_by_gap_descending():
    examples_by_query = {
        "q_missing": [
            {"doc_id": "d1", "features": {"bm25_rank": 0.0, "dense_rank": 2, "term_overlap": 0.1}},
        ],
        "q_fine": [
            {"doc_id": "d2", "features": {"bm25_rank": 1, "dense_rank": 1, "term_overlap": 0.9}},
        ],
        "q_moderate_gap": [
            {"doc_id": "d3", "features": {"bm25_rank": 30, "dense_rank": 2, "term_overlap": 0.2}},
        ],
    }
    qrels = {"q_missing": {"d1": 3}, "q_fine": {"d2": 3}, "q_moderate_gap": {"d3": 2}}

    ranked = find_best_vocabulary_mismatch_case(examples_by_query, qrels)

    assert len(ranked) == 2  # q_fine excluded, nothing to report
    assert ranked[0]["query_id"] == "q_missing"  # starkest gap ranks first
    assert ranked[1]["query_id"] == "q_moderate_gap"


def test_find_best_vocabulary_mismatch_case_empty_when_no_queries_qualify():
    examples_by_query = {
        "q1": [{"doc_id": "d1", "features": {"bm25_rank": 1, "dense_rank": 1, "term_overlap": 0.9}}],
    }
    qrels = {"q1": {"d1": 3}}
    ranked = find_best_vocabulary_mismatch_case(examples_by_query, qrels)
    assert ranked == []
