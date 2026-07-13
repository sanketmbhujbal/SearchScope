"""
Cross-encoder reranking (DESIGN.md, "NEW IN V2 Layer 2").

Day 3. After hybrid retrieval returns top-100, this scores each
query-passage pair jointly with a cross-encoder and keeps the top-20 for
the LTR layer — the now-standard 3-stage stack (retrieve -> rerank -> LTR).

Cross-encoders see query and doc together (not encoded independently like
the bi-encoder in retrieval/dense.py), capturing fine-grained interaction
signals a bi-encoder's fixed-vector similarity misses — too slow to run
over a full corpus, but fast enough over a top-100 candidate set.

See eval/run_reranked_eval.py for the 4-way comparison (BM25 / Dense /
Hybrid / +Reranker) this pipeline stage feeds into.
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        from config import CROSS_ENCODER_NAME

        self.model_name = model_name or CROSS_ENCODER_NAME
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],  # [(doc_id, doc_text), ...]
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Returns [(doc_id, cross_encoder_score), ...] sorted descending."""
        model = self._load_model()
        pairs = [(query, text) for _doc_id, text in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip([doc_id for doc_id, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k]]
