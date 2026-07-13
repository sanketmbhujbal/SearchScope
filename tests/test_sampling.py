import json
import tempfile
from pathlib import Path

import pytest

from data.sampling import build_scoped_corpus, load_judged_doc_ids
from tests.fixtures import TOY_CORPUS, TOY_QRELS


def _write_toy_qrels(tmpdir: Path) -> Path:
    qrels_path = tmpdir / "qrels.jsonl"
    with open(qrels_path, "w") as f:
        for qid, doc_scores in TOY_QRELS.items():
            for doc_id, relevance in doc_scores.items():
                f.write(json.dumps({"query_id": qid, "doc_id": doc_id, "relevance": relevance}) + "\n")
    return qrels_path


def test_load_judged_doc_ids_includes_grade_zero_docs():
    """Grade-0 (judged nonrelevant) docs matter too — a metric needs to
    know a doc was judged, not just that it was relevant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qrels_path = _write_toy_qrels(Path(tmpdir))
        judged_ids = load_judged_doc_ids(qrels_path)
        # TOY_QRELS references all 6 toy docs across its two queries (d1-d6),
        # including several with relevance 0.
        assert judged_ids == {"d1", "d2", "d3", "d4", "d5", "d6"}


def test_scoped_corpus_always_includes_all_judged_docs():
    with tempfile.TemporaryDirectory() as tmpdir:
        qrels_path = _write_toy_qrels(Path(tmpdir))
        # target_size equal to the full toy corpus -> should just return everything
        scoped = build_scoped_corpus(TOY_CORPUS, qrels_path, target_size=6, seed=1)
        scoped_ids = {d["doc_id"] for d in scoped}
        assert scoped_ids == {"d1", "d2", "d3", "d4", "d5", "d6"}


def test_scoped_corpus_raises_if_target_smaller_than_judged_set():
    """This is the failure mode the function exists to prevent: silently
    dropping judged passages, which would quietly break NDCG/MAP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qrels_path = _write_toy_qrels(Path(tmpdir))
        with pytest.raises(ValueError, match="smaller than the"):
            build_scoped_corpus(TOY_CORPUS, qrels_path, target_size=2, seed=1)


def test_scoped_corpus_is_reproducible_with_same_seed():
    with tempfile.TemporaryDirectory() as tmpdir:
        qrels_path = _write_toy_qrels(Path(tmpdir))
        # Add extra unjudged docs so there's something to actually sample from
        extended_corpus = TOY_CORPUS + [
            {"doc_id": f"extra{i}", "text": f"filler passage {i}"} for i in range(20)
        ]
        scoped_a = build_scoped_corpus(extended_corpus, qrels_path, target_size=10, seed=7)
        scoped_b = build_scoped_corpus(extended_corpus, qrels_path, target_size=10, seed=7)
        assert {d["doc_id"] for d in scoped_a} == {d["doc_id"] for d in scoped_b}
