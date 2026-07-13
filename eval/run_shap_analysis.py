"""
Day 7 — SHAP analysis (DESIGN.md §12, first-class deliverable — three
committed artifacts, not optional).

Trains ONE model on ALL 43 queries' candidates (not leave-one-query-out)
— deliberately different from eval/run_ltr_eval.py's LOQO CV. LOQO exists
to produce honest, non-inflated *evaluation numbers* (Day 5-6). SHAP
analysis has a different goal: explaining the model that would actually
be deployed, which is trained on all available labeled data, the same
way a real system's final production model would be after CV had already
validated the approach. Using a LOQO-fold model here would just pick one
arbitrary fold's model to explain, which isn't more "honest," just
arbitrary.

Produces three artifacts under results/day7_shap/:
    global_importance.png   — mean |SHAP value| per feature, all candidates
    per_intent_breakdown.png — same, grouped by query_intent_class
    single_query_waterfall.png — per-feature contribution for one prediction

Usage:
    python -m eval.run_shap_analysis --dataset trec-dl-2019 --sample 150000
"""
import argparse

import config
from eval.build_ltr_dataset import assemble_ltr_examples_cached
from ranking.features import FEATURE_NAMES
from ranking.train_ltr import (
    compute_shap_values,
    examples_to_xgb_inputs,
    plot_global_importance,
    plot_per_intent_breakdown,
    plot_single_query_waterfall,
    train_ltr_model,
)


def main(dataset_key: str, sample: int | None, rebuild_cache: bool = False) -> None:
    print("Loading LTR dataset (cached after first Day 5-6/7 run)...")
    examples_by_query = assemble_ltr_examples_cached(dataset_key, sample, force_rebuild=rebuild_cache)

    print("Training one model on ALL queries' candidates (not LOQO — see module docstring)...")
    X, y, groups, qid_per_row, doc_id_per_row = examples_to_xgb_inputs(examples_by_query)
    model = train_ltr_model(X, y, groups)
    print(f"  trained on {X.shape[0]} candidates across {len(groups)} queries")

    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X, FEATURE_NAMES)

    out_dir = config.RESULTS_DIR / "day7_shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Artifact 1/3: global feature importance...")
    plot_global_importance(shap_values, out_dir / "global_importance.png")

    print("Artifact 2/3: per-query-intent-class breakdown...")
    intent_idx = FEATURE_NAMES.index("query_intent_class")
    intent_labels = X[:, intent_idx]
    intent_counts = {int(v): int((intent_labels == v).sum()) for v in set(intent_labels)}
    print(f"  query_intent_class distribution across all candidates: {intent_counts} "
          f"(0=informational, 1=navigational, 2=transactional)")
    plot_per_intent_breakdown(shap_values, intent_labels, FEATURE_NAMES, out_dir / "per_intent_breakdown.png")

    print("Artifact 3/3: single-query waterfall (highest-label candidate, for a clear example)...")
    # Pick the candidate with the highest label (most clearly "relevant")
    # so the waterfall shows a meaningful example, not an arbitrary one.
    best_row_index = int(y.argmax())
    example_qid = qid_per_row[best_row_index]
    example_doc_id = doc_id_per_row[best_row_index]
    print(f"  showing query_id={example_qid}, doc_id={example_doc_id}, label={y[best_row_index]}")
    plot_single_query_waterfall(shap_values, best_row_index, out_dir / "single_query_waterfall.png")

    # Also print the global ranking as text, so the finding is readable
    # without opening an image.
    import numpy as np

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print("\nGlobal feature importance (mean |SHAP value|), ranked:")
    for name, value in ranked:
        print(f"  {name:24s} {value:.4f}")

    print(f"\nSaved 3 artifacts -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force rebuilding the LTR dataset instead of using the cached version.")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, rebuild_cache=args.rebuild_cache)
