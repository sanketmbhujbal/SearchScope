from retrieval.bm25 import BM25Retriever
from tests.fixtures import TOY_CORPUS, TOY_QUERIES


def test_bm25_rank_bm25_backend_builds_and_searches():
    retriever = BM25Retriever(backend="rank_bm25")
    retriever.build_index(TOY_CORPUS)
    assert retriever.backend == "rank_bm25"

    results = retriever.search(TOY_QUERIES["q1"], k=3)
    assert len(results) <= 3
    assert all(isinstance(doc_id, str) and isinstance(score, float) for doc_id, score in results)


def test_bm25_finds_lexically_matching_doc_top1():
    """'reset service credentials' should surface d1 ('reset your service
    account credentials') over the HR password doc — pure term overlap
    favors d1 here even before any semantic layer is involved."""
    retriever = BM25Retriever(backend="rank_bm25")
    retriever.build_index(TOY_CORPUS)

    top1_doc_id, _score = retriever.search(TOY_QUERIES["q1"], k=1)[0]
    assert top1_doc_id == "d1"


def test_bm25_vocabulary_mismatch_fails_on_deployment_query():
    """Reproduces the Story 2 failure mode from DESIGN.md §4: 'deployment
    process' should NOT reliably beat the HR doc on pure lexical overlap,
    since 'process' appears in both — this is exactly the gap dense
    retrieval and the cross-encoder are meant to close."""
    retriever = BM25Retriever(backend="rank_bm25")
    retriever.build_index(TOY_CORPUS)

    results = retriever.search(TOY_QUERIES["q2"], k=6)
    doc_ids = [doc_id for doc_id, _ in results]
    # Both candidates should at least be retrieved (this is a recall check,
    # not a ranking-quality assertion — BM25's ranking mistake here is the
    # point, not a bug).
    assert "d3" in doc_ids
    assert "d4" in doc_ids
