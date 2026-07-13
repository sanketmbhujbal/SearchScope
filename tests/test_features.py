import pytest

from ranking.features import (
    FEATURE_NAMES,
    CorpusStats,
    FeatureBuilder,
    doc_length_log,
    doc_recency,
    query_entropy,
    query_intent_class,
    query_length,
    role_doc_affinity,
    section_importance,
    simulated_ctr,
    simulated_dwell_time,
    source_authority,
    term_overlap,
    title_match,
)


def test_term_overlap_full_match():
    assert term_overlap("reset password", "how to reset your password today") == 1.0


def test_term_overlap_partial_match():
    assert term_overlap("reset password now", "how to reset your password today") == pytest.approx(2 / 3)


def test_term_overlap_empty_query_returns_zero():
    assert term_overlap("", "some text") == 0.0


def test_title_match_only_checks_leading_words():
    doc = "reset password instructions " + " ".join(["filler"] * 20) + " unrelated_term_at_end"
    assert title_match("reset password", doc, title_words=3) == 1.0
    # A term appearing only past the pseudo-title window shouldn't count
    assert title_match("unrelated_term_at_end", doc, title_words=3) == 0.0


def test_doc_length_log_increases_with_length():
    short = doc_length_log("one two three")
    long = doc_length_log(" ".join(["word"] * 100))
    assert long > short


def test_query_length_counts_tokens():
    assert query_length("what is the capital of France") == 6


def test_query_entropy_zero_for_repeated_single_term():
    assert query_entropy("test test test") == pytest.approx(0.0)


def test_query_entropy_positive_for_varied_terms():
    assert query_entropy("what is deployment process") > 0.0


def test_query_intent_class_transactional():
    from ranking.features import _INTENT_CLASSES
    assert query_intent_class("buy cheap laptop") == _INTENT_CLASSES["transactional"]


def test_query_intent_class_navigational():
    from ranking.features import _INTENT_CLASSES
    assert query_intent_class("official website login") == _INTENT_CLASSES["navigational"]


def test_query_intent_class_informational_default():
    from ranking.features import _INTENT_CLASSES
    assert query_intent_class("what is photosynthesis") == _INTENT_CLASSES["informational"]


def test_section_importance_short_opener_scores_higher():
    heading_like = "Overview. This document explains the deployment process in detail."
    body_like = "This is a long opening sentence that goes on for quite a while before any punctuation."
    assert section_importance(heading_like) > section_importance(body_like)


def test_simulated_ctr_is_deterministic_for_same_doc_id():
    a = simulated_ctr("d1")
    b = simulated_ctr("d1")
    assert a == b


def test_simulated_ctr_differs_across_doc_ids():
    a = simulated_ctr("d1")
    b = simulated_ctr("d2")
    assert a != b  # extremely unlikely to collide by chance


def test_simulated_ctr_bounded_unit_interval():
    for doc_id in ["d1", "d2", "d3", "some_longer_doc_id_123"]:
        value = simulated_ctr(doc_id)
        assert 0.0 <= value <= 1.0


def test_simulated_ctr_and_dwell_time_are_not_identical():
    """Different hash keys — shouldn't produce identical values for the
    same doc_id even though both are synthetic priors on it."""
    ctr = simulated_ctr("d1")
    dwell = simulated_dwell_time("d1")
    assert ctr != dwell


def test_simulated_ctr_takes_no_relevance_grade_argument():
    """Regression test for the label-leakage bug: simulated_ctr must be a
    function of doc_id alone. If this signature ever grows a
    relevance/label parameter again, that's the leak coming back —
    verified here by confirming the function is callable with exactly
    one positional argument."""
    import inspect

    sig = inspect.signature(simulated_ctr)
    params = list(sig.parameters.keys())
    assert params == ["doc_id"], (
        f"simulated_ctr signature changed to {params} — if this now includes "
        "anything label/relevance-derived, that reintroduces the Day 5-6 "
        "leakage bug (see ranking/features.py module docstring)."
    )


def test_doc_recency_deterministic_and_bounded():
    a = doc_recency("d12345")
    b = doc_recency("d12345")
    assert a == b
    assert 0.0 <= a <= 1.0


def test_source_authority_differs_across_doc_ids():
    a = source_authority("d1")
    b = source_authority("d2")
    assert a != b  # extremely unlikely to collide by chance


def test_role_doc_affinity_neutral_without_role():
    assert role_doc_affinity("some text", None, None) == 0.0


class _StubAffinityScorer:
    def affinity(self, doc_text, role):
        return 0.75


def test_role_doc_affinity_delegates_to_scorer():
    assert role_doc_affinity("some text", "engineer", _StubAffinityScorer()) == 0.75


def test_corpus_stats_idf_zero_for_oov_term():
    stats = CorpusStats()
    stats.fit(["the quick brown fox", "jumps over the lazy dog"])
    assert stats.idf("nonexistent_term_xyz") == 0.0


def test_corpus_stats_idf_positive_for_known_term():
    stats = CorpusStats()
    stats.fit(["the quick brown fox", "jumps over the lazy dog", "fox and dog play"])
    assert stats.idf("fox") > 0.0


def test_feature_builder_produces_all_20_named_features():
    stats = CorpusStats()
    stats.fit(["reset your password today", "deployment pipeline runbook", "leave policy overview"])
    builder = FeatureBuilder(corpus_stats=stats)

    candidate = {
        "doc_id": "d1",
        "doc_text": "reset your password today",
        "bm25_score": 5.2,
        "bm25_rank": 1,
        "dense_cosine": 0.8,
        "dense_rank": 2,
    }
    features = builder.build("reset password", candidate)

    assert set(features.keys()) == set(FEATURE_NAMES)
    assert len(features) == 20
    assert features["bm25_score"] == 5.2
    assert features["term_overlap"] == 1.0


def test_feature_builder_defaults_missing_upstream_signals_to_zero():
    """A candidate that only came from BM25 (never scored by dense/
    cross-encoder) shouldn't crash — those features should default to 0."""
    stats = CorpusStats()
    stats.fit(["some corpus text", "more corpus text"])
    builder = FeatureBuilder(corpus_stats=stats)

    candidate = {"doc_id": "d1", "doc_text": "some corpus text", "bm25_score": 3.0, "bm25_rank": 1}
    features = builder.build("query", candidate)

    assert features["dense_cosine"] == 0.0
    assert features["cross_encoder_score"] == 0.0


def test_feature_builder_has_no_relevance_grade_parameter():
    """Regression test for the label-leakage bug (see ranking/features.py
    module docstring): build() must not accept anything label-derived,
    or a future edit could quietly reintroduce the leak that caused
    NDCG@10 to jump to an implausible 0.97 with LOQO CV."""
    import inspect

    sig = inspect.signature(FeatureBuilder.build)
    params = list(sig.parameters.keys())
    assert "relevance_grade" not in params, (
        "FeatureBuilder.build() has a relevance_grade parameter again — "
        "this is exactly the Day 5-6 label leakage bug. Do not thread a "
        "label into any feature computation."
    )


def test_feature_builder_output_identical_for_same_candidate_regardless_of_context():
    """The concrete guarantee the leakage fix provides: building features
    for the exact same (query, candidate) twice must produce identical
    output — nothing external (like a label that varies by whether this
    candidate is currently in a training or held-out-eval role) can leak
    in and change the result."""
    stats = CorpusStats()
    stats.fit(["reset your password today", "deployment pipeline runbook"])
    builder = FeatureBuilder(corpus_stats=stats)
    candidate = {
        "doc_id": "d1", "doc_text": "reset your password today",
        "bm25_score": 5.0, "dense_cosine": 0.7,
    }

    features_a = builder.build("reset password", candidate)
    features_b = builder.build("reset password", candidate)
    assert features_a == features_b
