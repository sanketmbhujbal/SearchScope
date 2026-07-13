from retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_boosts_doc_present_in_both_lists():
    bm25_hits = [("d1", 5.0), ("d2", 4.0), ("d3", 3.0)]
    dense_hits = [("d3", 0.9), ("d1", 0.8), ("d4", 0.7)]

    fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=60, top_k=10)
    fused_ids = [doc_id for doc_id, _ in fused]

    # d1 is rank 1 in bm25 and rank 2 in dense -> should outrank d3
    # (rank 3 bm25, rank 1 dense) and everything appearing in only one list.
    assert fused_ids[0] == "d1"
    assert set(fused_ids) == {"d1", "d2", "d3", "d4"}


def test_rrf_respects_top_k_truncation():
    bm25_hits = [(f"d{i}", float(10 - i)) for i in range(10)]
    dense_hits = [(f"d{i}", float(10 - i)) for i in range(10)]

    fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_scores_are_score_distribution_agnostic():
    """A doc ranked #1 by a source with huge raw scores shouldn't dominate
    a doc ranked #1 by a source with tiny raw scores — RRF only cares
    about rank position, not score magnitude."""
    bm25_hits = [("d1", 1000.0), ("d2", 1.0)]
    dense_hits = [("d2", 0.001), ("d1", 0.0009)]

    fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=60, top_k=10)
    fused_scores = dict(fused)
    # d1 is #1 in bm25 and #2 in dense; d2 is #2 in bm25 and #1 in dense.
    # Symmetric ranks -> equal fused scores, regardless of raw magnitude.
    assert fused_scores["d1"] == fused_scores["d2"]
