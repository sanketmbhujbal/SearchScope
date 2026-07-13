"""
Feature engineering for the LTR layer (DESIGN.md §9.1) — the core of this
project and the core of the Glean role.

Signal inventory (20 total; see DESIGN.md §9.1 for full descriptions):
    Lexical:        bm25_score, bm25_rank, term_overlap, title_match
    Semantic:       dense_cosine, dense_rank, cross_encoder_score, rrf_score
    Query:          query_length, query_entropy, query_idf_mean, query_intent_class
    Document:       doc_length_log, section_importance, metadata_overlap
    Behavior:       simulated_ctr, simulated_dwell_time
    Freshness:      doc_recency
    Authority:      source_authority
    Personalization: role_doc_affinity   (feature 20 — from personalization/role_affinity.py)

IMPORTANT — which signals are real vs. synthetic (DESIGN.md §5):
    Real, computed directly from this project's actual pipeline:
        bm25_score, bm25_rank, dense_cosine, dense_rank, cross_encoder_score,
        rrf_score, term_overlap, doc_length_log, query_length, query_idf_mean,
        role_doc_affinity
    Heuristic proxies (real computation, but standing in for something MS
    MARCO passages don't natively carry — titles, sections, metadata):
        title_match, section_importance, metadata_overlap, query_entropy,
        query_intent_class
    Synthetic (no real signal exists in this dataset; deterministic
    hash-based stand-ins, clearly flagged so they're never mistaken for
    real behavioral/organizational data):
        simulated_ctr, simulated_dwell_time, doc_recency, source_authority

simulated_ctr and simulated_dwell_time are deterministic functions of
doc_id only — a document-level prior standing in for a historical
aggregate CTR/dwell-time a real system would pull from click logs. They
are deliberately NOT derived from the current query's relevance label.
An earlier version of this file made that mistake (correlating these
features with relevance_grade, with noise) and it caused genuine label
leakage: leave-one-query-out CV correctly held the *model* out of
training on each held-out query, but the *features* for that query's own
candidates were still computed from their true labels, handing the model
a near-direct readout of the answer at prediction time too. That
produced an implausibly large jump (NDCG@10 0.76 -> 0.97, MRR@10 ->
1.0000) that turned out to be leakage, not genuine ranking improvement.
See results/day5-6_findings.md for the full story — kept here as a
reminder of why these two functions must never take a label or
label-derived value as input.
"""
from __future__ import annotations

import hashlib
import math

FEATURE_NAMES = [
    # lexical
    "bm25_score", "bm25_rank", "term_overlap", "title_match",
    # semantic
    "dense_cosine", "dense_rank", "cross_encoder_score", "rrf_score",
    # query
    "query_length", "query_entropy", "query_idf_mean", "query_intent_class",
    # document
    "doc_length_log", "section_importance", "metadata_overlap",
    # behavior
    "simulated_ctr", "simulated_dwell_time",
    # freshness / authority
    "doc_recency", "source_authority",
    # personalization
    "role_doc_affinity",
]
assert len(FEATURE_NAMES) == 20, "DESIGN.md §9.1 specifies 18-20 signals"

_INTENT_CLASSES = {"informational": 0, "navigational": 1, "transactional": 2}
_TRANSACTIONAL_TERMS = {"buy", "price", "cost", "discount", "order", "purchase", "cheap", "deal"}
_NAVIGATIONAL_TERMS = {"login", "website", "sign", "homepage", "official"}
_INFORMATIONAL_STARTERS = {"what", "how", "why", "when", "where", "who", "which", "is", "are", "does"}


class CorpusStats:
    """Lightweight corpus-level statistics (IDF table, TF-IDF term
    weights) backing query_idf_mean and metadata_overlap. Reuses
    sklearn's TfidfVectorizer rather than hand-rolling document-frequency
    counting — same tool already used in personalization/role_affinity.py."""

    def __init__(self):
        self._vectorizer = None

    def fit(self, corpus_texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectorizer.fit(corpus_texts)

    def idf(self, term: str) -> float:
        """IDF of a single term; 0.0 for out-of-vocabulary terms (rather
        than raising, since query terms routinely won't all appear in a
        150K-passage scoped corpus)."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before idf().")
        vocab = self._vectorizer.vocabulary_
        idx = vocab.get(term.lower())
        if idx is None:
            return 0.0
        return float(self._vectorizer.idf_[idx])

    def top_terms(self, doc_text: str, k: int = 10) -> set[str]:
        """The doc's top-k TF-IDF-weighted terms — used as a pseudo-
        metadata/tag proxy by metadata_overlap, since MS MARCO passages
        don't carry real tags."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before top_terms().")
        vec = self._vectorizer.transform([doc_text])
        row = vec.tocoo()
        if row.nnz == 0:
            return set()
        feature_names = self._vectorizer.get_feature_names_out()
        pairs = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)[:k]
        return {feature_names[col] for col, _weight in pairs}


# --- Lexical ---

def term_overlap(query: str, doc_text: str) -> float:
    """Fraction of query terms present in the document."""
    q_terms = set(query.lower().split())
    if not q_terms:
        return 0.0
    d_terms = set(doc_text.lower().split())
    return len(q_terms & d_terms) / len(q_terms)


def title_match(query: str, doc_text: str, title_words: int = 10) -> float:
    """Heuristic proxy: MS MARCO passages don't carry real titles, so the
    first `title_words` tokens stand in for one. Overlap fraction of
    query terms found in that pseudo-title."""
    q_terms = set(query.lower().split())
    if not q_terms:
        return 0.0
    pseudo_title_terms = set(doc_text.lower().split()[:title_words])
    return len(q_terms & pseudo_title_terms) / len(q_terms)


# --- Query ---

def query_length(query: str) -> int:
    return len(query.split())


def query_entropy(query: str) -> float:
    """Shannon entropy of the query's own token frequency distribution.
    A simplified proxy — a corpus-informed specificity measure would be
    richer, but query_idf_mean already partially covers that ground."""
    terms = query.lower().split()
    if not terms:
        return 0.0
    counts: dict[str, int] = {}
    for t in terms:
        counts[t] = counts.get(t, 0) + 1
    n = len(terms)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def query_idf_mean(query: str, corpus_stats: CorpusStats) -> float:
    terms = query.lower().split()
    if not terms:
        return 0.0
    return sum(corpus_stats.idf(t) for t in terms) / len(terms)


def query_intent_class(query: str) -> int:
    """Rule-based classifier -> {0: informational, 1: navigational, 2: transactional}.
    A real system would use a trained classifier or query-log-derived
    signals; this is an explicit, documented heuristic standing in for one."""
    q_lower = query.lower()
    terms = set(q_lower.split())
    if terms & _TRANSACTIONAL_TERMS:
        return _INTENT_CLASSES["transactional"]
    if terms & _NAVIGATIONAL_TERMS:
        return _INTENT_CLASSES["navigational"]
    return _INTENT_CLASSES["informational"]  # default, including explicit wh-question starters


# --- Document ---

def doc_length_log(doc_text: str) -> float:
    """log-scaled passage length; penalizes very short/long passages."""
    n_tokens = len(doc_text.split())
    return math.log1p(n_tokens)


def section_importance(doc_text: str) -> float:
    """Heuristic proxy: MS MARCO passages have no real section/heading
    markup. Approximates it by checking whether the passage's opening
    looks heading-like (short first sentence) — a weak but explicit
    stand-in, not a claim of real document structure."""
    first_sentence = doc_text.split(".")[0] if doc_text else ""
    n_words = len(first_sentence.split())
    if 0 < n_words <= 6:
        return 1.0  # looks heading-like
    return 0.5  # default "body text" weight


def metadata_overlap(query: str, doc_text: str, corpus_stats: CorpusStats) -> float:
    """Overlap between query terms and the doc's top TF-IDF terms, used
    as a pseudo-metadata/tag proxy — MS MARCO passages carry no real
    metadata or tags."""
    q_terms = set(query.lower().split())
    if not q_terms:
        return 0.0
    doc_top_terms = corpus_stats.top_terms(doc_text)
    return len(q_terms & doc_top_terms) / len(q_terms)


# --- Behavior (synthetic — see module docstring for why these are
# doc_id-only, never label-derived) ---

def simulated_ctr(doc_id: str) -> float:
    """Synthetic click-through proxy: a deterministic per-document prior,
    standing in for a historical aggregate CTR a real system would pull
    from click logs. Deliberately independent of the current query's
    relevance label — see the note above for why that independence
    matters."""
    return _deterministic_unit_value(f"ctr:{doc_id}")


def simulated_dwell_time(doc_id: str) -> float:
    """Synthetic engagement proxy, same rationale as simulated_ctr but a
    different hash key so the two aren't identical to each other."""
    return _deterministic_unit_value(f"dwell:{doc_id}")


# --- Freshness / Authority (synthetic — see module docstring) ---

def _deterministic_unit_value(key: str) -> float:
    """Deterministic pseudo-random value in [0, 1] from a hash of key —
    used so the same doc_id always gets the same synthetic recency/
    authority value across runs, without needing to actually store it."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def doc_recency(doc_id: str) -> float:
    """Synthetic: MS MARCO passages carry no real timestamps (DESIGN.md
    §5 flags this explicitly). Deterministic pseudo-random value per
    doc_id standing in for "how recent" a document is."""
    return _deterministic_unit_value(f"recency:{doc_id}")


def source_authority(doc_id: str) -> float:
    """Synthetic: MS MARCO passages carry no real domain/source metadata.
    Deterministic pseudo-random value per doc_id standing in for source
    reliability."""
    return _deterministic_unit_value(f"authority:{doc_id}")


# --- Personalization ---

def role_doc_affinity(doc_text: str, role: str | None, role_affinity_scorer) -> float:
    """Delegates to personalization/role_affinity.py's RoleAffinityScorer.
    Returns 0.0 (neutral) if no role/scorer is provided — e.g. when
    building features outside a personalization context."""
    if role is None or role_affinity_scorer is None:
        return 0.0
    return role_affinity_scorer.affinity(doc_text, role)


class FeatureBuilder:
    """Assembles the full 20-dim feature vector for a (query, candidate) pair."""

    def __init__(self, corpus_stats: CorpusStats, role_affinity_scorer=None):
        self.corpus_stats = corpus_stats
        self.role_affinity_scorer = role_affinity_scorer

    def build(
        self,
        query: str,
        candidate: dict,
        user_role: str | None = None,
    ) -> dict:
        """
        candidate: expected keys — doc_id, doc_text, and whatever upstream
            retrieval signals are available (bm25_score, bm25_rank,
            dense_cosine, dense_rank, cross_encoder_score, rrf_score).
            Missing upstream signals default to 0.0 — candidates aren't
            guaranteed to have passed through every pipeline stage (e.g. a
            BM25-only candidate that dense retrieval never surfaced).

        No relevance_grade parameter — deliberately. See the module
        docstring: an earlier version threaded the true label into
        simulated_ctr/simulated_dwell_time, causing label leakage during
        LOQO evaluation. Every feature here is computable from the
        query/candidate alone, with no access to the answer.

        Returns a dict of {feature_name: value} — deliberately a dict, not
        a bare vector, so SHAP plots (Day 7) can label features by name.
        """
        doc_id = candidate["doc_id"]
        doc_text = candidate["doc_text"]

        features = {
            "bm25_score": float(candidate.get("bm25_score", 0.0)),
            "bm25_rank": float(candidate.get("bm25_rank", 0.0)),
            "term_overlap": term_overlap(query, doc_text),
            "title_match": title_match(query, doc_text),
            "dense_cosine": float(candidate.get("dense_cosine", 0.0)),
            "dense_rank": float(candidate.get("dense_rank", 0.0)),
            "cross_encoder_score": float(candidate.get("cross_encoder_score", 0.0)),
            "rrf_score": float(candidate.get("rrf_score", 0.0)),
            "query_length": float(query_length(query)),
            "query_entropy": query_entropy(query),
            "query_idf_mean": query_idf_mean(query, self.corpus_stats),
            "query_intent_class": float(query_intent_class(query)),
            "doc_length_log": doc_length_log(doc_text),
            "section_importance": section_importance(doc_text),
            "metadata_overlap": metadata_overlap(query, doc_text, self.corpus_stats),
            "simulated_ctr": simulated_ctr(doc_id),
            "simulated_dwell_time": simulated_dwell_time(doc_id),
            "doc_recency": doc_recency(doc_id),
            "source_authority": source_authority(doc_id),
            "role_doc_affinity": role_doc_affinity(doc_text, user_role, self.role_affinity_scorer),
        }
        assert set(features.keys()) == set(FEATURE_NAMES), "feature dict must match FEATURE_NAMES exactly"
        return features
