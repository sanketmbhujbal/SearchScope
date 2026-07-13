import random

from eval.run_ltr_eval import build_run_from_examples, run_loqo_cv
from ranking.features import FEATURE_NAMES
from ranking.train_ltr import examples_to_xgb_inputs, train_ltr_model


def _make_synthetic_examples(n_queries=5, n_docs_per_query=8, seed=0):
    rng = random.Random(seed)
    examples_by_query = {}
    for qi in range(n_queries):
        qid = f"q{qi}"
        examples = []
        for di in range(n_docs_per_query):
            doc_id = f"d{qi}_{di}"
            features = {f: rng.random() for f in FEATURE_NAMES}
            label = rng.choice([0, 0, 0, 1, 2, 3])
            examples.append({"doc_id": doc_id, "features": features, "label": label})
        examples_by_query[qid] = examples
    return examples_by_query


def test_examples_to_xgb_inputs_shapes_and_group_sizes():
    examples_by_query = _make_synthetic_examples(n_queries=5, n_docs_per_query=8)
    X, y, groups, qid_per_row, doc_id_per_row = examples_to_xgb_inputs(examples_by_query)

    assert X.shape == (40, 20)
    assert y.shape == (40,)
    assert groups == [8, 8, 8, 8, 8]
    assert len(qid_per_row) == 40
    assert len(doc_id_per_row) == 40


def test_examples_to_xgb_inputs_skips_queries_with_no_candidates():
    examples_by_query = _make_synthetic_examples(n_queries=3, n_docs_per_query=4)
    examples_by_query["q_empty"] = []  # e.g. a query whose retrieval found nothing

    X, y, groups, qid_per_row, _ = examples_to_xgb_inputs(examples_by_query)
    assert "q_empty" not in qid_per_row
    assert sum(groups) == X.shape[0] == 12  # 3 queries * 4 docs, empty one excluded


def test_examples_to_xgb_inputs_feature_column_order_matches_feature_names():
    examples_by_query = {
        "q0": [{"doc_id": "d0", "features": {f: i for i, f in enumerate(FEATURE_NAMES)}, "label": 1}]
    }
    X, _y, _groups, _q, _d = examples_to_xgb_inputs(examples_by_query)
    # Feature at column i should equal its index (per the synthetic features dict above)
    for i in range(len(FEATURE_NAMES)):
        assert X[0][i] == i


def test_train_ltr_model_runs_end_to_end():
    examples_by_query = _make_synthetic_examples(n_queries=5, n_docs_per_query=8)
    X, y, groups, _, _ = examples_to_xgb_inputs(examples_by_query)
    model = train_ltr_model(X, y, groups)

    import xgboost as xgb
    preds = model.predict(xgb.DMatrix(X[:8]))
    assert len(preds) == 8


def test_build_run_from_examples_sorts_descending_by_given_score_key():
    examples_by_query = {
        "q0": [
            {"doc_id": "d1", "features": {"cross_encoder_score": 0.2}, "label": 0},
            {"doc_id": "d2", "features": {"cross_encoder_score": 0.9}, "label": 1},
            {"doc_id": "d3", "features": {"cross_encoder_score": 0.5}, "label": 0},
        ]
    }
    run = build_run_from_examples(examples_by_query, "cross_encoder_score")
    assert [doc_id for doc_id, _score in run["q0"]] == ["d2", "d3", "d1"]


def test_run_loqo_cv_returns_all_queries_with_correct_candidate_sets():
    examples_by_query = _make_synthetic_examples(n_queries=6, n_docs_per_query=5)
    ltr_run = run_loqo_cv(examples_by_query)

    assert set(ltr_run.keys()) == set(examples_by_query.keys())
    for qid, hits in ltr_run.items():
        result_doc_ids = {doc_id for doc_id, _score in hits}
        expected_doc_ids = {ex["doc_id"] for ex in examples_by_query[qid]}
        assert result_doc_ids == expected_doc_ids
        assert len(hits) == 5


def test_run_loqo_cv_skips_empty_queries():
    examples_by_query = _make_synthetic_examples(n_queries=4, n_docs_per_query=5)
    examples_by_query["q_empty"] = []

    ltr_run = run_loqo_cv(examples_by_query)
    assert "q_empty" not in ltr_run
    assert len(ltr_run) == 4
