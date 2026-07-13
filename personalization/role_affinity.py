"""
Role-based personalization layer (DESIGN.md, "NEW IN V2 Layer 3b").

Day 4. Four simulated user roles (Engineer, Sales, HR, Legal), each with a
topic affinity vector (config.ROLE_AFFINITY_TOPICS). At query time, cosine
similarity between the role vector and a document's topic distribution
(TF-IDF at index time) becomes feature 20 in the LTR model — the ranker
learns when to weight it vs. when pure relevance dominates.

Explicitly documented as synthetic per DESIGN.md §5 ("User roles are
simulated"). This is the conceptual prototype for what would run on
Glean's Personal Knowledge Graph in production (DESIGN.md §15).

TODO (Day 4):
    - Fit a TF-IDF vectorizer over the corpus at index time
    - Build each role's affinity vector (via config.ROLE_AFFINITY_TOPICS
      seed terms, projected into the same TF-IDF space)
    - Implement affinity(doc_text, role) -> cosine similarity in [0, 1]
    - Produce the ranking-diff table: same query ('policy'), top-5 per role
"""
from __future__ import annotations


class RoleAffinityScorer:
    def __init__(self, roles: dict[str, list[str]] | None = None):
        from config import ROLE_AFFINITY_TOPICS

        self.roles = roles or ROLE_AFFINITY_TOPICS
        self._vectorizer = None      # sklearn TfidfVectorizer, fit at index time
        self._role_vectors: dict[str, "np.ndarray"] = {}

    def fit(self, corpus_texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectorizer.fit(corpus_texts)

        # Represent each role as the TF-IDF vector of its seed-topic terms
        for role, topics in self.roles.items():
            seed_text = " ".join(topics)
            self._role_vectors[role] = self._vectorizer.transform([seed_text])

    def affinity(self, doc_text: str, role: str) -> float:
        """Cosine similarity between a document and a user role's affinity vector."""
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None:
            raise RuntimeError("Call fit() with the corpus before scoring affinity.")
        if role not in self._role_vectors:
            raise ValueError(f"Unknown role '{role}'. Choose from {list(self.roles)}")

        doc_vec = self._vectorizer.transform([doc_text])
        sim = cosine_similarity(doc_vec, self._role_vectors[role])
        return float(sim[0][0])

    def rerank_by_role(
        self, candidates: list[tuple[str, str, float]], role: str
    ) -> list[tuple[str, float]]:
        """
        candidates: [(doc_id, doc_text, base_relevance_score), ...]
        TODO (Day 4): decide the blend of base relevance vs. affinity here,
        or (preferred, per DESIGN.md) leave the blend to the LTR model and
        just expose affinity() as feature 20 instead of hand-tuning a blend.
        """
        raise NotImplementedError(
            "Per DESIGN.md §9, role_doc_affinity should be an LTR feature, "
            "not a hand-tuned re-ranking blend — implement the feature path "
            "in ranking/features.py instead of blending here."
        )
