from eval.run_personalization_demo import (
    _min_max_normalize,
    build_role_ranking_diff,
    format_role_diff_table,
)


def test_min_max_normalize_scales_to_unit_range():
    result = _min_max_normalize([10.0, 20.0, 30.0])
    assert result == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_all_equal_values():
    """All-equal input has no meaningful range — should return a neutral
    midpoint rather than dividing by zero."""
    result = _min_max_normalize([5.0, 5.0, 5.0])
    assert result == [0.5, 0.5, 0.5]


def test_min_max_normalize_handles_empty_list():
    assert _min_max_normalize([]) == []


class _StubAffinityScorer:
    """Fake RoleAffinityScorer — each role has a fixed preference for a
    specific doc_id-length-coded text, letting us verify the blending and
    per-role dispatch logic without a real TF-IDF fit."""

    def __init__(self, preferred_role_to_text: dict[str, str]):
        self.preferred_role_to_text = preferred_role_to_text

    def affinity(self, doc_text: str, role: str) -> float:
        return 1.0 if doc_text == self.preferred_role_to_text.get(role) else 0.0


def test_build_role_ranking_diff_reflects_role_specific_preference():
    """With affinity_weight=1.0 (pure affinity, ignore base relevance),
    each role should surface its own preferred document at rank 1 — this
    is the core claim of Day 4: same query, different top result by role."""
    candidates = [
        ("d1", "engineering doc about deployment", 0.5),
        ("d2", "sales doc about discounts", 0.5),
        ("d3", "hr doc about leave policy", 0.5),
    ]
    scorer = _StubAffinityScorer({
        "engineer": "engineering doc about deployment",
        "sales": "sales doc about discounts",
        "hr": "hr doc about leave policy",
    })

    results = build_role_ranking_diff(
        "policy", candidates, scorer, roles=["engineer", "sales", "hr"],
        top_n=3, affinity_weight=1.0,
    )

    assert results["engineer"][0][0] == "d1"
    assert results["sales"][0][0] == "d2"
    assert results["hr"][0][0] == "d3"


def test_build_role_ranking_diff_respects_top_n_truncation():
    candidates = [(f"d{i}", f"text {i}", float(i)) for i in range(10)]
    scorer = _StubAffinityScorer({})  # no preferences -> affinity always 0

    results = build_role_ranking_diff(
        "query", candidates, scorer, roles=["engineer"], top_n=3, affinity_weight=0.5,
    )
    assert len(results["engineer"]) == 3


def test_build_role_ranking_diff_falls_back_to_relevance_when_affinity_zero_weight():
    """affinity_weight=0.0 should reduce to pure base-relevance ranking,
    ignoring role entirely — a sanity check that the blend formula is
    correct at the boundary."""
    candidates = [
        ("d1", "text one", 0.9),
        ("d2", "text two", 0.1),
    ]
    scorer = _StubAffinityScorer({"engineer": "text two"})  # would prefer d2, but weight=0

    results = build_role_ranking_diff(
        "query", candidates, scorer, roles=["engineer"], top_n=2, affinity_weight=0.0,
    )
    assert results["engineer"][0][0] == "d1"  # higher base relevance wins


def test_format_role_diff_table_includes_all_roles_and_query():
    results = {
        "engineer": [("d1", "some engineering text", 0.9)],
        "hr": [("d2", "some hr text", 0.8)],
    }
    table = format_role_diff_table(results, query="policy")
    assert "policy" in table
    assert "Engineer" in table
    assert "Hr" in table  # .title() capitalization
    assert "d1" in table
    assert "d2" in table
