from eval.run_reranked_eval import build_reranked_run
from reranking.cross_encoder import CrossEncoderReranker


class _StubCrossEncoderModel:
    """Fake CrossEncoder that scores by string length instead of running a
    real model — lets us test rerank()'s sorting/truncation logic without
    network access or a downloaded model."""

    def predict(self, pairs):
        return [float(len(doc_text)) for _query, doc_text in pairs]


def test_rerank_sorts_by_score_descending():
    reranker = CrossEncoderReranker()
    reranker._model = _StubCrossEncoderModel()  # bypass _load_model()

    candidates = [
        ("d1", "short"),
        ("d2", "a much longer passage of text here"),
        ("d3", "medium length text"),
    ]
    results = reranker.rerank("irrelevant query", candidates, top_k=10)
    result_ids = [doc_id for doc_id, _score in results]

    # Longest text (d2) should score highest with the stub model
    assert result_ids == ["d2", "d3", "d1"]


def test_rerank_respects_top_k_truncation():
    reranker = CrossEncoderReranker()
    reranker._model = _StubCrossEncoderModel()

    candidates = [(f"d{i}", "x" * i) for i in range(10)]
    results = reranker.rerank("query", candidates, top_k=3)
    assert len(results) == 3


class _StubReranker:
    """Stands in for CrossEncoderReranker in build_reranked_run tests —
    reverses candidate order, so we can verify the orchestration wiring
    (candidate gathering, text lookup, per-query dispatch) independent of
    any actual reranking model."""

    def rerank(self, query, candidates, top_k):
        reversed_candidates = list(reversed(candidates))[:top_k]
        return [(doc_id, 1.0) for doc_id, _text in reversed_candidates]


def test_build_reranked_run_dispatches_per_query_with_correct_candidates():
    hybrid_run = {
        "q1": [("d1", 0.9), ("d2", 0.8)],
        "q2": [("d3", 0.7)],
    }
    passage_texts = {"d1": "text one", "d2": "text two", "d3": "text three"}
    queries = {"q1": "query one", "q2": "query two"}

    reranked = build_reranked_run(hybrid_run, passage_texts, queries, _StubReranker(), top_k=10)

    assert set(reranked.keys()) == {"q1", "q2"}
    assert [doc_id for doc_id, _ in reranked["q1"]] == ["d2", "d1"]  # reversed by stub
    assert [doc_id for doc_id, _ in reranked["q2"]] == ["d3"]


def test_build_reranked_run_skips_candidates_missing_from_passage_texts():
    """If a doc_id's text couldn't be found in the corpus (shouldn't
    normally happen, but defends against a partial/mismatched lookup),
    it should be silently excluded rather than crashing."""
    hybrid_run = {"q1": [("d1", 0.9), ("d_missing", 0.8)]}
    passage_texts = {"d1": "text one"}  # d_missing intentionally absent
    queries = {"q1": "query one"}

    reranked = build_reranked_run(hybrid_run, passage_texts, queries, _StubReranker(), top_k=10)
    result_ids = [doc_id for doc_id, _ in reranked["q1"]]
    assert result_ids == ["d1"]
    assert "d_missing" not in result_ids
