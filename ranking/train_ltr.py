"""
XGBoost LambdaRank training (DESIGN.md §9.2) + SHAP analysis (DESIGN.md §12).

Day 5-6: train the ranker. See eval/build_ltr_dataset.py for how the
labeled dataset is assembled, and eval/run_ltr_eval.py for the
leave-one-query-out cross-validation loop this feeds.

Day 7: SHAP global importance, per-query-type breakdown, single-query
waterfall — all three are committed first-class deliverables, not optional.
"""
from __future__ import annotations

import numpy as np


def examples_to_xgb_inputs(
    examples_by_query: dict[str, list[dict]],
    feature_order: list[str] | None = None,
):
    """
    Converts the {query_id: [{"doc_id", "features", "label"}, ...]}
    structure from eval/build_ltr_dataset.py into XGBoost's expected
    inputs: a flat feature matrix with all of one query's rows
    contiguous (required for set_group()), plus parallel label/qid/doc_id
    arrays so predictions can be mapped back to (query_id, doc_id) pairs.

    Returns: (X, y, groups, qid_per_row, doc_id_per_row)
        X: (n_rows, n_features) float array
        y: (n_rows,) label array
        groups: list of candidate counts per query, in row order
        qid_per_row / doc_id_per_row: parallel lists identifying each row
    """
    if feature_order is None:
        from ranking.features import FEATURE_NAMES as feature_order

    rows, labels, groups, qid_per_row, doc_id_per_row = [], [], [], [], []
    for qid, examples in examples_by_query.items():
        if not examples:
            continue
        groups.append(len(examples))
        for ex in examples:
            rows.append([ex["features"][f] for f in feature_order])
            labels.append(ex["label"])
            qid_per_row.append(qid)
            doc_id_per_row.append(ex["doc_id"])

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y, groups, qid_per_row, doc_id_per_row


def train_ltr_model(feature_matrix, labels, query_groups, params: dict | None = None):
    """
    feature_matrix: (n_samples, 20) array, columns ordered per
                     ranking.features.FEATURE_NAMES
    labels: relevance labels (from MS MARCO training triples / TREC qrels)
    query_groups: number of candidates per query, in matrix order
                  (required by XGBoost's ranking objective)
    """
    import xgboost as xgb
    from config import XGBOOST_PARAMS

    params = params or XGBOOST_PARAMS
    dtrain = xgb.DMatrix(feature_matrix, label=labels)
    dtrain.set_group(query_groups)
    return xgb.train(params, dtrain, num_boost_round=200)


def compute_shap_values(model, feature_matrix, feature_names):
    """
    Returns a shap.Explanation for the global bar chart / waterfall plots.

    feature_names must be attached via a pandas DataFrame — TreeExplainer's
    __call__ does NOT accept a feature_names kwarg directly (verified
    against the installed shap version; passing it raises a TypeError).
    Wrapping the matrix in a DataFrame is the supported way to carry
    column names through to the resulting Explanation and its plots.
    """
    import pandas as pd
    import shap

    X_df = pd.DataFrame(feature_matrix, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    return explainer(X_df)


def plot_global_importance(shap_values, out_path):
    """Global feature importance bar chart — mean |SHAP value| per
    feature across all candidates. DESIGN.md §12, artifact 1 of 3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(shap_values, shap_values.data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_intent_breakdown(shap_values, intent_labels, feature_names, out_path):
    """
    Per-query-intent-class breakdown — mean |SHAP value| per feature,
    grouped by query_intent_class (informational/navigational/transactional).
    DESIGN.md §12, artifact 2 of 3.

    intent_labels: array of the query_intent_class value for each row
        (same length/order as shap_values), used to group rows rather
        than re-deriving intent from raw query text here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    intent_names = {0: "informational", 1: "navigational", 2: "transactional"}
    unique_intents = sorted(set(int(v) for v in intent_labels))

    abs_shap = np.abs(shap_values.values)
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_width = 0.8 / max(len(unique_intents), 1)
    x = np.arange(len(feature_names))

    for i, intent in enumerate(unique_intents):
        mask = intent_labels == intent
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        mean_importance = abs_shap[mask].mean(axis=0)
        label = f"{intent_names.get(intent, intent)} (n={n_rows})"
        ax.barh(x + i * bar_width, mean_importance, height=bar_width, label=label)

    ax.set_yticks(x + bar_width * (len(unique_intents) - 1) / 2)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Feature importance by query intent class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_single_query_waterfall(shap_values, row_index, out_path):
    """Single-prediction SHAP waterfall — shows exactly how each feature
    pushed one specific candidate's score up or down. DESIGN.md §12,
    artifact 3 of 3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.waterfall(shap_values[row_index], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
