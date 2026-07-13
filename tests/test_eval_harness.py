import pytest

from eval.harness import evaluate, format_results_table
from tests.fixtures import TOY_QRELS


def test_perfect_run_scores_maximum_ndcg():
    """A run that puts the only relevant doc at rank 1 for every query
    should score NDCG@10 == 1.0 exactly."""
    perfect_run = {
        "q1": [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)],
        "q2": [("d3", 3.0), ("d4", 2.0), ("d1", 1.0)],
    }
    results = evaluate(perfect_run, TOY_QRELS, metrics={"ndcg_cut_10"})
    assert results["ndcg_cut_10"] == pytest.approx(1.0)


def test_worst_case_run_scores_zero_ndcg():
    """A run with no relevant docs retrieved at all should score 0."""
    bad_run = {
        "q1": [("d2", 3.0), ("d4", 2.0), ("d5", 1.0)],
        "q2": [("d1", 3.0), ("d2", 2.0), ("d5", 1.0)],
    }
    results = evaluate(bad_run, TOY_QRELS, metrics={"ndcg_cut_10"})
    assert results["ndcg_cut_10"] == pytest.approx(0.0)


def test_evaluate_raises_on_no_overlapping_queries():
    run = {"nonexistent_query": [("d1", 1.0)]}
    with pytest.raises(ValueError, match="No overlapping query IDs"):
        evaluate(run, TOY_QRELS)


def test_format_results_table_is_valid_markdown():
    results_by_stage = {
        "BM25": {"ndcg_cut_10": 0.43, "recip_rank": 0.35, "recall_100": 0.75, "map": 0.30},
        "Hybrid": {"ndcg_cut_10": 0.55, "recip_rank": 0.47, "recall_100": 0.83, "map": 0.40},
    }
    table = format_results_table(results_by_stage)
    lines = table.splitlines()
    assert lines[0].startswith("| Stage |")
    assert "BM25" in table
    assert "Hybrid" in table
    assert "0.4300" in table
