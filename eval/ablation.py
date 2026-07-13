"""
Ablation study (DESIGN.md §9.3).

Day 7. Removes each signal category in turn, retrains via the same
leave-one-query-out CV used for the Day 5-6 headline number, and measures
the NDCG@10 delta — the table structure below is pre-committed
specifically to force honest measurement (no cherry-picking which
ablations to report).

Ablations run (DESIGN.md §9.3, plus one addition):
    Full model              - baseline, all 20 signals
    No lexical               - drop bm25_score, bm25_rank, term_overlap, title_match
    No semantic               - drop dense_cosine, cross_encoder_score, dense_rank, rrf_score
    No query                  - drop query_entropy, query_idf_mean, query_intent_class, query_length
    No behavior                - drop simulated_ctr, simulated_dwell_time
    No freshness/authority      - drop doc_recency, source_authority
    No personalization           - drop role_doc_affinity
    No synthetic (bonus)          - drop all four synthetic signals together
    BM25 only                      - keep ONLY bm25_score
    Cross-encoder only               - keep ONLY cross_encoder_score

"No synthetic" isn't in the original DESIGN.md table — it's added because
results/day5-6_findings.md's LTR-vs-reranker comparison generated a
specific, testable hypothesis: that the four purely synthetic features
(doc_recency, source_authority, simulated_ctr, simulated_dwell_time —
random per doc_id, uncorrelated with true relevance) might be diluting
the model's limited training signal from only ~40 queries. This ablation
tests that directly: if dropping them recovers or exceeds the reranker's
NDCG@10, that confirms the mechanism.

"BM25 only" and "Cross-encoder only" are sanity checks on the training
pipeline itself, not just interesting ablations: an LTR model trained on
nothing but bm25_score should track close to plain BM25 retrieval, and
one trained on nothing but cross_encoder_score should track close to the
reranker. If either doesn't, that points at a training bug, not a
feature-importance finding.

Usage:
    python -m eval.ablation --dataset trec-dl-2019 --sample 150000
"""
from __future__ import annotations

import argparse

import config
from ranking.features import FEATURE_NAMES

ABLATION_SPECS = {
    "Full model": {"mode": "drop", "features": []},
    "No lexical": {"mode": "drop", "features": ["bm25_score", "bm25_rank", "term_overlap", "title_match"]},
    "No semantic": {"mode": "drop", "features": ["dense_cosine", "cross_encoder_score", "dense_rank", "rrf_score"]},
    "No query": {"mode": "drop", "features": ["query_entropy", "query_idf_mean", "query_intent_class", "query_length"]},
    "No behavior": {"mode": "drop", "features": ["simulated_ctr", "simulated_dwell_time"]},
    "No freshness/authority": {"mode": "drop", "features": ["doc_recency", "source_authority"]},
    "No personalization": {"mode": "drop", "features": ["role_doc_affinity"]},
    "No synthetic (bonus)": {
        "mode": "drop",
        "features": ["simulated_ctr", "simulated_dwell_time", "doc_recency", "source_authority"],
    },
    "BM25 only": {"mode": "keep_only", "features": ["bm25_score"]},
    "Cross-encoder only": {"mode": "keep_only", "features": ["cross_encoder_score"]},
}
for _name, _spec in ABLATION_SPECS.items():
    assert all(f in FEATURE_NAMES for f in _spec["features"]), f"Unknown feature in ablation '{_name}'"


def resolve_feature_order(spec: dict) -> list[str]:
    """Computes the feature_order list for a given ablation spec, always
    preserving FEATURE_NAMES's canonical ordering."""
    if spec["mode"] == "drop":
        dropped = set(spec["features"])
        return [f for f in FEATURE_NAMES if f not in dropped]
    elif spec["mode"] == "keep_only":
        kept = set(spec["features"])
        return [f for f in FEATURE_NAMES if f in kept]
    raise ValueError(f"Unknown ablation mode: {spec['mode']}")


def run_ablation_study(examples_by_query: dict, qrels: dict, specs: dict | None = None) -> dict:
    """
    For each ablation spec, retrains via leave-one-query-out CV (reusing
    eval.run_ltr_eval.run_loqo_cv with a restricted feature_order) and
    scores against qrels. Returns {ablation_name: {metric: value}} ready
    for eval.harness.format_results_table().
    """
    from eval.harness import evaluate
    from eval.run_ltr_eval import run_loqo_cv

    specs = specs or ABLATION_SPECS
    results_by_ablation = {}
    for name, spec in specs.items():
        feature_order = resolve_feature_order(spec)
        print(f"Running ablation '{name}' ({len(feature_order)}/{len(FEATURE_NAMES)} features)...")
        run = run_loqo_cv(examples_by_query, feature_order=feature_order)
        results_by_ablation[name] = evaluate(run, qrels)
    return results_by_ablation


def main(dataset_key: str, sample: int | None, rebuild_cache: bool = False) -> None:
    from eval.build_ltr_dataset import assemble_ltr_examples_cached
    from eval.harness import format_results_table, load_qrels, save_results

    print("Loading LTR dataset (cached after first Day 5-6/7 run)...")
    examples_by_query = assemble_ltr_examples_cached(dataset_key, sample, force_rebuild=rebuild_cache)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    print(f"Running {len(ABLATION_SPECS)} ablation variants via LOQO CV "
          f"(~0.5 min each based on Day 5-6 timing)...")
    results_by_ablation = run_ablation_study(examples_by_query, qrels)

    print()
    print(format_results_table(results_by_ablation))
    if sample:
        print(f"\n(scoped corpus, sample={sample} — see DESIGN.md §5 for why)")

    full_ndcg = results_by_ablation["Full model"]["ndcg_cut_10"]
    print("\nNDCG@10 delta vs. Full model:")
    for name, metrics in results_by_ablation.items():
        if name == "Full model":
            continue
        delta = metrics["ndcg_cut_10"] - full_ndcg
        print(f"  {name:35s} {delta:+.4f}")

    suffix = f"_sample_{sample}" if sample else ""
    out_path = config.RESULTS_DIR / f"{dataset_key}_ablation_results{suffix}.json"
    save_results(results_by_ablation, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force rebuilding the LTR dataset instead of using the cached version.")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, rebuild_cache=args.rebuild_cache)
