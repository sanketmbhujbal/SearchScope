"""
Hybrid fusion via Reciprocal Rank Fusion (DESIGN.md §7.3).

RRF is score-distribution agnostic (BM25 and cosine similarity live on
completely different scales) and requires no normalization or tuning
beyond the damping constant k. We also expose the raw per-source ranks
so a learned linear interpolation weight can be computed as an LTR
feature later (DESIGN.md §9.1, "RRF fusion score").
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    ranked_lists: one list per retrieval source, each a list of
                  [(doc_id, score), ...] already sorted by descending score.
    k: RRF damping constant (60 is the standard default from the original
       RRF paper — larger k flattens the influence of top ranks).
    Returns: [(doc_id, rrf_score), ...] sorted by descending fused score.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


class HybridRetriever:
    """Combines a BM25Retriever and DenseRetriever via RRF."""

    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        bm25_depth: int = 100,
        dense_depth: int = 100,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        bm25_hits = self.bm25.search(query, k=bm25_depth)
        dense_hits = self.dense.search(query, k=dense_depth)
        return reciprocal_rank_fusion([bm25_hits, dense_hits], k=self.rrf_k, top_k=top_k)

    def search_with_components(self, query: str, bm25_depth: int = 100, dense_depth: int = 100):
        """
        Like search(), but also returns the individual BM25 and dense hit
        lists — useful for the LTR feature layer, which needs BM25 rank/
        score and dense rank/score as separate signals (DESIGN.md §9.1),
        not just the fused result.
        """
        bm25_hits = self.bm25.search(query, k=bm25_depth)
        dense_hits = self.dense.search(query, k=dense_depth)
        fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=self.rrf_k)
        return {"bm25": bm25_hits, "dense": dense_hits, "hybrid": fused}
