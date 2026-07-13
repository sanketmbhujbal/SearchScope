import random

from eval.ablation import ABLATION_SPECS, resolve_feature_order, run_ablation_study
from ranking.features import FEATURE_NAMES


def test_all_ablation_specs_resolve_without_error():
    for name, spec in ABLATION_SPECS.items():
        order = resolve_feature_order(spec)
        assert len(order) > 0, f"'{name}' resolved to zero features"
        assert all(f in FEATURE_NAMES for f in order)


def test_drop_mode_excludes_specified_features():
    spec = {"mode": "drop", "features": ["bm25_score", "bm25_rank"]}
    order = resolve_feature_order(spec)
    assert "bm25_score" not in order
    assert "bm25_rank" not in order
    assert len(order) == len(FEATURE_NAMES) - 2


def test_keep_only_mode_includes_only_specified_features():
    spec = {"mode": "keep_only", "features": ["cross_encoder_score"]}
    order = resolve_feature_order(spec)
    assert order == ["cross_encoder_score"]


def test_full_model_spec_keeps_all_features():
    order = resolve_feature_order(ABLATION_SPECS["Full model"])
    assert set(order) == set(FEATURE_NAMES)


def test_bm25_only_and_cross_encoder_only_are_single_feature():
    assert resolve_feature_order(ABLATION_SPECS["BM25 only"]) == ["bm25_score"]
    assert resolve_feature_order(ABLATION_SPECS["Cross-encoder only"]) == ["cross_encoder_score"]


def test_no_synthetic_drops_exactly_the_four_synthetic_signals():
    order = resolve_feature_order(ABLATION_SPECS["No synthetic (bonus)"])
    dropped = set(FEATURE_NAMES) - set(order)
    assert dropped == {"simulated_ctr", "simulated_dwell_time", "doc_recency", "source_authority"}


def _make_synthetic_dataset(n_queries=6, n_docs_per_query=6, seed=0):
    rng = random.Random(seed)
    examples_by_query = {}
    qrels = {}
    for qi in range(n_queries):
        qid = f"q{qi}"
        examples = []
        qrels[qid] = {}
        for di in range(n_docs_per_query):
            doc_id = f"d{qi}_{di}"
            features = {f: rng.random() for f in FEATURE_NAMES}
            label = rng.choice([0, 0, 1, 2, 3])
            examples.append({"doc_id": doc_id, "features": features, "label": label})
            qrels[qid][doc_id] = label
        examples_by_query[qid] = examples
    return examples_by_query, qrels


def test_run_ablation_study_returns_metrics_for_each_spec():
    examples_by_query, qrels = _make_synthetic_dataset()
    specs = {"Full model": ABLATION_SPECS["Full model"], "BM25 only": ABLATION_SPECS["BM25 only"]}

    results = run_ablation_study(examples_by_query, qrels, specs=specs)

    assert set(results.keys()) == {"Full model", "BM25 only"}
    for name, metrics in results.items():
        assert "ndcg_cut_10" in metrics
        assert 0.0 <= metrics["ndcg_cut_10"] <= 1.0
