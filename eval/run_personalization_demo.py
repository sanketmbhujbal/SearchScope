"""
Day 4 — role-based personalization demo (DESIGN.md, "NEW IN V2 Layer 3b").

Produces the ranking-diff table the Day 4 build-plan deliverable calls
for (DESIGN.md §14): same query, top-5 results per simulated role
(Engineer / Sales / HR / Legal), showing that role context changes which
passages surface even though the query text is identical.

IMPORTANT — this script's role/relevance blend is illustrative only, not
the production design. RoleAffinityScorer.rerank_by_role() deliberately
raises NotImplementedError to steer away from a hand-tuned blend in the
library code — the actual plan (DESIGN.md §9) is for role_doc_affinity to
be feature 20 in the Day 5-6 LTR model, letting the ranker learn when
affinity should matter vs. when raw relevance should dominate, rather
than a fixed formula guessing that upfront. This script's blend exists
only to produce a visible demo before the LTR layer is built.

Usage:
    python -m eval.run_personalization_demo --dataset trec-dl-2019 --sample 150000 --query policy
"""
import argparse

import config
from data.corpus_lookup import lookup_passage_texts
from data.preprocess import load_corpus
from data.sampling import build_scoped_corpus
from personalization.role_affinity import RoleAffinityScorer
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever


def _min_max_normalize(values: list[float]) -> list[float]:
    """Scales a list of scores to [0, 1] so relevance (RRF scores, which
    live on an arbitrary small scale) and affinity (cosine similarity,
    [0, 1] but usually clustered near 0 for short queries) can be blended
    on comparable footing for this demo."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]  # all-equal input -> neutral midpoint
    return [(v - lo) / (hi - lo) for v in values]


def build_role_ranking_diff(
    query: str,
    candidates: list[tuple[str, str, float]],  # [(doc_id, text, base_relevance), ...]
    scorer: RoleAffinityScorer,
    roles: list[str],
    top_n: int = 5,
    affinity_weight: float = 0.5,
) -> dict[str, list[tuple[str, str, float]]]:
    """
    Pure logic, factored out for testing without a real TF-IDF fit — see
    tests/test_personalization.py.

    Returns {role: [(doc_id, text, demo_score), ...]} sorted descending,
    truncated to top_n per role.
    """
    doc_ids = [c[0] for c in candidates]
    texts = [c[1] for c in candidates]
    base_scores = [c[2] for c in candidates]
    norm_relevance = _min_max_normalize(base_scores)

    results: dict[str, list[tuple[str, str, float]]] = {}
    for role in roles:
        affinities = [scorer.affinity(text, role) for text in texts]
        norm_affinity = _min_max_normalize(affinities)

        demo_scores = [
            (1 - affinity_weight) * rel + affinity_weight * aff
            for rel, aff in zip(norm_relevance, norm_affinity)
        ]
        ranked = sorted(
            zip(doc_ids, texts, demo_scores), key=lambda x: x[2], reverse=True
        )
        results[role] = ranked[:top_n]
    return results


def format_role_diff_table(results: dict[str, list[tuple[str, str, float]]], query: str) -> str:
    lines = [f"# Personalization demo — query: \"{query}\"\n"]
    for role, hits in results.items():
        lines.append(f"## {role.title()}\n")
        for rank, (doc_id, text, score) in enumerate(hits, start=1):
            preview = text[:150] + ("..." if len(text) > 150 else "")
            lines.append(f"{rank}. [{doc_id}] (score={score:.3f}) {preview}")
        lines.append("")
    return "\n".join(lines)


def main(dataset_key: str, sample: int | None, query: str, candidate_depth: int, top_n: int) -> None:
    bm25_index_dir = config.get_bm25_index_dir(sample)
    faiss_index_path = config.get_faiss_index_path(sample)

    print("Loading BM25 index...")
    bm25 = BM25Retriever()
    if not BM25Retriever.index_exists(bm25_index_dir):
        raise FileNotFoundError(f"No BM25 index at {bm25_index_dir}. Run data/preprocess.py first.")
    bm25.load(bm25_index_dir)

    print(f"Loading dense index ({config.DENSE_ENCODER_NAME})...")
    dense = DenseRetriever()
    if not faiss_index_path.exists():
        raise FileNotFoundError(f"No dense index at {faiss_index_path}. Run data/preprocess.py first.")
    dense.load(faiss_index_path)

    hybrid = HybridRetriever(bm25, dense, rrf_k=config.RRF_K)

    print(f"Retrieving top-{candidate_depth} candidates for query: '{query}'...")
    hits = hybrid.search(query, top_k=candidate_depth)
    doc_ids = {doc_id for doc_id, _score in hits}
    passage_texts = lookup_passage_texts(dataset_key, doc_ids)
    candidates = [(doc_id, passage_texts[doc_id], score) for doc_id, score in hits if doc_id in passage_texts]
    print(f"  {len(candidates)} candidates with resolved passage text")

    print("Fitting role affinity vectorizer on the scoped corpus (this may take a moment)...")
    corpus_docs = load_corpus(dataset_key)
    if sample:
        qrels_path = config.RAW_DATA_DIR / dataset_key / "qrels.jsonl"
        corpus_docs = build_scoped_corpus(corpus_docs, qrels_path, target_size=sample, seed=config.RANDOM_SEED)
    scorer = RoleAffinityScorer()
    scorer.fit([d["text"] for d in corpus_docs])

    print(f"Scoring candidates for roles: {list(config.USER_ROLES)}...")
    results = build_role_ranking_diff(query, candidates, scorer, config.USER_ROLES, top_n=top_n)

    table = format_role_diff_table(results, query)
    print()
    print(table)

    out_path = config.RESULTS_DIR / "day4_personalization_demo.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--query", type=str, default="policy", help="Demo query — DESIGN.md uses 'policy'.")
    parser.add_argument("--candidate-depth", type=int, default=50, help="Candidates retrieved before role scoring.")
    parser.add_argument("--top-n", type=int, default=5, help="Results shown per role.")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, query=args.query, candidate_depth=args.candidate_depth, top_n=args.top_n)
