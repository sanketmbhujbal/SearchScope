"""
Offline evaluation harness (DESIGN.md §11.1).

Thin, well-tested wrapper around pytrec_eval so every pipeline stage
(BM25, hybrid, reranked, LTR) is scored identically against the official
TREC DL judgments. Keeping this in one place is what makes the
before/after story in DESIGN.md §3 trustworthy — every stage is compared
on the same harness, not ad-hoc scoring per script.

Metrics (DESIGN.md §11.1):
    NDCG@10      — primary metric, graded relevance, position-discounted
    MRR@10       — first relevant result position (navigational queries)
    Recall@100   — retrieval ceiling; what the ranker can work with
    MAP          — mean average precision over the full ranking
"""
from __future__ import annotations

import json
from pathlib import Path

import pytrec_eval

METRIC_KEYS = {
    "ndcg_cut_10": "ndcg@10",
    "recip_rank": "mrr@10",   # pytrec_eval's recip_rank is unbounded by default;
                              # we pass a run truncated the same way for all stages
                              # so it's comparable to a conventional MRR@10.
    "recall_100": "recall@100",
    "map": "map",
}


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    """Loads qrels.jsonl (as written by data/download_msmarco.py) into the
    {query_id: {doc_id: relevance}} format pytrec_eval expects."""
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path) as f:
        for line in f:
            row = json.loads(line)
            qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return qrels


def run_to_pytrec_format(
    run: dict[str, list[tuple[str, float]]]
) -> dict[str, dict[str, float]]:
    """run: {query_id: [(doc_id, score), ...]} -> pytrec_eval's expected format."""
    return {qid: {doc_id: float(score) for doc_id, score in hits} for qid, hits in run.items()}


def evaluate(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    metrics: set[str] | None = None,
) -> dict[str, float]:
    """
    Scores a run against qrels and returns metric averages across all
    queries present in both the run and the qrels.

    run: {query_id: [(doc_id, score), ...]}, unsorted is fine — pytrec_eval
         sorts internally, but callers should pass results already ranked
         by descending relevance score for anything relying on top-k depth.
    qrels: {query_id: {doc_id: relevance}}, from load_qrels().
    metrics: pytrec_eval metric names, defaults to config.METRICS.
    """
    if metrics is None:
        from config import METRICS

        metrics = METRICS

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query_results = evaluator.evaluate(run_to_pytrec_format(run))

    if not per_query_results:
        raise ValueError(
            "No overlapping query IDs between run and qrels — check that "
            "query IDs match between your retrieval run and the qrels file."
        )

    # Average each metric across queries
    aggregated: dict[str, float] = {}
    for metric in metrics:
        values = [q_result[metric] for q_result in per_query_results.values() if metric in q_result]
        aggregated[metric] = sum(values) / len(values) if values else 0.0

    return aggregated


def format_results_table(results_by_stage: dict[str, dict[str, float]]) -> str:
    """
    results_by_stage: {"BM25": {...}, "Hybrid": {...}, "After Reranker": {...}, ...}
    Renders a markdown table matching the DESIGN.md §3 success-metrics format,
    so results can be pasted straight into the README.
    """
    metric_order = ["ndcg_cut_10", "recip_rank", "recall_100", "map"]
    headers = ["Stage"] + [METRIC_KEYS[m] for m in metric_order]

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for stage, results in results_by_stage.items():
        row = [stage] + [f"{results.get(m, 0.0):.4f}" for m in metric_order]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def save_results(results_by_stage: dict[str, dict[str, float]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_by_stage, f, indent=2)
