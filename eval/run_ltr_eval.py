"""
Day 5-6 — feature engineering (20 signals) + XGBoost LTR training.

Uses leave-one-query-out cross-validation across the 43 TREC DL 2019
judged queries rather than MS MARCO's separate training-triples set
(hundreds of thousands of examples — a multi-GB download and hours of
additional compute, out of scope for this timeline; see DESIGN.md §5 for
the same kind of scoping decision applied elsewhere in this project).
LOQO is a standard, defensible methodology for a small labeled set: every
query's predictions come from a model that never saw that query's labels
during training, so the eval numbers aren't inflated by the model having
memorized the held-out query's relevant docs.

Usage:
    python -m eval.run_ltr_eval --dataset trec-dl-2019 --sample 150000

Assumes data/preprocess.py has already built the indexes (same --sample
value, if any). This reuses the persisted BM25/dense indexes but
re-runs retrieval + reranking for every query (same cost as Day 3) to
assemble the candidate pool with features attached — expect this to take
a similar amount of time to eval/run_reranked_eval.py, plus the (fast)
LOQO training loop on top.
"""
import argparse
import time

import config
from eval.build_ltr_dataset import assemble_ltr_examples_cached
from eval.harness import evaluate, format_results_table, load_qrels, save_results
from ranking.train_ltr import examples_to_xgb_inputs, train_ltr_model


def build_run_from_examples(examples_by_query: dict, score_key: str) -> dict:
    """Builds a {qid: [(doc_id, score), ...]} run directly from a stored
    feature (e.g. cross_encoder_score) rather than re-running retrieval.
    Used for the +Reranker sanity-check row: computed on the *identical*
    candidate pool as +LTR, so it should reproduce Day 3's recorded
    number almost exactly — confirming the two scripts are looking at
    the same thing before trusting the +LTR comparison next to it. See
    results/day3_findings.md for why candidate-pool consistency matters
    (that's exactly the mismatch that caused the Day 3 metric-depth bug)."""
    run = {}
    for qid, examples in examples_by_query.items():
        hits = [(ex["doc_id"], ex["features"][score_key]) for ex in examples]
        run[qid] = sorted(hits, key=lambda x: x[1], reverse=True)
    return run


def run_loqo_cv(
    examples_by_query: dict,
    params: dict | None = None,
    feature_order: list[str] | None = None,
) -> dict:
    """Leave-one-query-out CV. Returns a {qid: [(doc_id, score), ...]} run
    where every query's scores came from a model trained on all *other*
    queries' candidates.

    feature_order: restricts training/scoring to a subset of features (in
    this order) — used by eval/ablation.py to retrain with a signal
    category dropped, reusing this exact CV loop rather than duplicating
    it."""
    import xgboost as xgb

    ltr_run = {}
    qids = [q for q, ex in examples_by_query.items() if ex]  # skip queries with no candidates
    for i, held_out_qid in enumerate(qids):
        train_examples = {q: ex for q, ex in examples_by_query.items() if q != held_out_qid and ex}
        X_train, y_train, groups, _, _ = examples_to_xgb_inputs(train_examples, feature_order=feature_order)
        model = train_ltr_model(X_train, y_train, groups, params=params)

        test_examples = {held_out_qid: examples_by_query[held_out_qid]}
        X_test, _, _, _, doc_ids_test = examples_to_xgb_inputs(test_examples, feature_order=feature_order)
        dtest = xgb.DMatrix(X_test)
        scores = model.predict(dtest)

        hits = sorted(zip(doc_ids_test, scores), key=lambda x: x[1], reverse=True)
        ltr_run[held_out_qid] = [(doc_id, float(score)) for doc_id, score in hits]

        if (i + 1) % 10 == 0 or (i + 1) == len(qids):
            print(f"  LOQO fold {i + 1}/{len(qids)} done")
    return ltr_run


def main(dataset_key: str, sample: int | None, rebuild_cache: bool = False) -> None:
    print("Assembling candidate pool + 20-signal features for all queries "
          "(cached after first run — see eval/build_ltr_dataset.py)...")
    start = time.time()
    examples_by_query = assemble_ltr_examples_cached(dataset_key, sample, force_rebuild=rebuild_cache)
    n_candidates = sum(len(v) for v in examples_by_query.values())
    print(f"  done in {(time.time() - start) / 60:.1f} min "
          f"({len(examples_by_query)} queries, {n_candidates} total candidates)")

    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    print("Sanity check: recomputing +Reranker from the same candidate pool used for LTR "
          "(should match Day 3's recorded number in results/day3_findings.md)...")
    reranker_run = build_run_from_examples(examples_by_query, "cross_encoder_score")

    n_queries = len([q for q, ex in examples_by_query.items() if ex])
    print(f"Training XGBoost LambdaRank via leave-one-query-out CV ({n_queries} folds)...")
    start = time.time()
    ltr_run = run_loqo_cv(examples_by_query)
    print(f"  LOQO CV done in {(time.time() - start) / 60:.1f} min")

    results_by_stage = {
        "+ Cross-Encoder Reranker (sanity check)": evaluate(reranker_run, qrels),
        "+ LTR (LOQO, 20 signals)": evaluate(ltr_run, qrels),
    }

    print()
    print(format_results_table(results_by_stage))
    if sample:
        print(f"\n(scoped corpus, sample={sample} — see DESIGN.md §5 for why)")
    print(
        "\nNOTE: the sanity-check +Reranker row above should be very close to "
        "results/day3_findings.md's recorded number. If it isn't, that points to a "
        "candidate-pool inconsistency between the two scripts, not a real ranking "
        "difference — worth checking before trusting the +LTR row next to it."
    )

    suffix = f"_sample_{sample}" if sample else ""
    out_path = config.RESULTS_DIR / f"{dataset_key}_ltr_results{suffix}.json"
    save_results(results_by_stage, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force rebuilding the LTR dataset instead of using the cached version "
                              "(needed if ranking/features.py has changed since the cache was built).")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, rebuild_cache=args.rebuild_cache)
