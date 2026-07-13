"""
Day 3 — cross-encoder reranking on top of hybrid retrieval.

Produces the 4-way comparison table (BM25 / Dense / Hybrid / +Reranker)
called for in the Day 3 build-plan deliverable (DESIGN.md §14). Loads the
persisted BM25 + dense indexes from Day 1-2 (no rebuilding), takes each
query's hybrid top-100, reranks with a cross-encoder, and scores every
stage on the same eval harness so the comparison is apples-to-apples.

IMPORTANT — reranking depth: this script reranks at full HYBRID_TOP_K
depth (100), not the smaller RERANK_TOP_K (20) that DESIGN.md specifies
as what actually feeds the Day 5-6 LTR layer downstream. That's
deliberate: Recall@100/MAP need up to 100 ranked docs per query to score
meaningfully, and truncating the *eval* run to 20 would silently starve
those metrics of anything sitting at rank 21-100 — making it look like
reranking hurt retrieval, when actually the reranker was just never asked
to return that many results. NDCG@10/MRR@10 wouldn't show this (they only
look at top-10), which is exactly what made the artifact easy to miss the
first time around. See results/day3_findings.md for the full story.

Usage:
    python -m eval.run_reranked_eval --dataset trec-dl-2019 --sample 150000

Assumes data/preprocess.py has already built the indexes (same --sample
value, if any).
"""
import argparse

import config
from data.corpus_lookup import lookup_passage_texts
from eval.harness import evaluate, format_results_table, load_qrels, save_results
from eval.run_baseline_eval import load_queries
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever


def build_reranked_run(
    hybrid_run: dict[str, list[tuple[str, float]]],
    passage_texts: dict[str, str],
    queries: dict[str, str],
    reranker: CrossEncoderReranker,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """
    Pure orchestration logic, factored out so it's testable with a stub
    reranker (no real model / network needed) — see tests/test_cross_encoder.py.

    hybrid_run: {query_id: [(doc_id, score), ...]} — the Day 1-2 hybrid
        retrieval output, used as the reranker's candidate pool.
    passage_texts: {doc_id: text} for every doc_id appearing anywhere in
        hybrid_run — the cross-encoder needs actual text, not just doc_ids.
    """
    reranked_run: dict[str, list[tuple[str, float]]] = {}
    for qid, hits in hybrid_run.items():
        candidates = [
            (doc_id, passage_texts[doc_id]) for doc_id, _score in hits if doc_id in passage_texts
        ]
        reranked_run[qid] = reranker.rerank(queries[qid], candidates, top_k=top_k)
    return reranked_run


def main(dataset_key: str, sample: int | None = None) -> None:
    queries = load_queries(dataset_key)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    bm25_index_dir = config.get_bm25_index_dir(sample)
    faiss_index_path = config.get_faiss_index_path(sample)

    print(f"Dataset: {dataset_key} | {len(queries)} queries" + (f" | sample={sample}" if sample else ""))

    print("Loading BM25 index...")
    bm25 = BM25Retriever()
    if not BM25Retriever.index_exists(bm25_index_dir):
        raise FileNotFoundError(
            f"No BM25 index at {bm25_index_dir}. Run data/preprocess.py first (same --sample value)."
        )
    bm25.load(bm25_index_dir)

    print(f"Loading dense index ({config.DENSE_ENCODER_NAME})...")
    dense = DenseRetriever()
    if not faiss_index_path.exists():
        raise FileNotFoundError(
            f"No dense index at {faiss_index_path}. Run data/preprocess.py first (same --sample value)."
        )
    dense.load(faiss_index_path)

    hybrid = HybridRetriever(bm25, dense, rrf_k=config.RRF_K)

    print("Running BM25 / Dense / Hybrid retrieval for all queries...")
    bm25_run, dense_run, hybrid_run = {}, {}, {}
    for qid, qtext in queries.items():
        components = hybrid.search_with_components(
            qtext, bm25_depth=config.BM25_TOP_K, dense_depth=config.DENSE_TOP_K
        )
        bm25_run[qid] = components["bm25"]
        dense_run[qid] = components["dense"]
        hybrid_run[qid] = components["hybrid"]

    # Gather passage text for every candidate doc_id across all queries in
    # one pass over the corpus, rather than per-query lookups.
    all_doc_ids = {doc_id for hits in hybrid_run.values() for doc_id, _score in hits}
    print(f"Looking up passage text for {len(all_doc_ids)} candidate passages...")
    passage_texts = lookup_passage_texts(dataset_key, all_doc_ids)

    print(f"Reranking with {config.CROSS_ENCODER_NAME} (full depth, for a fair comparison "
          f"against Hybrid's top-{config.HYBRID_TOP_K})...")
    reranker = CrossEncoderReranker()
    # NOTE: reranks at full HYBRID_TOP_K depth (100), not RERANK_TOP_K (20),
    # specifically so Recall@100/MAP are comparable to the other three
    # stages. Truncating to top-20 here (matching what actually feeds the
    # Day 5-6 LTR layer downstream) silently starves Recall@100/MAP of any
    # relevant doc sitting at rank 21-100 — the reranker never gets a
    # chance to lose them, it's just never asked to return them, which
    # made an unaffected top-10 metric (NDCG@10, MRR@10) look fine right
    # next to a collapsed one (Recall@100, MAP) for reasons that had
    # nothing to do with reranking quality. See results/day3_findings.md.
    reranked_run = build_reranked_run(
        hybrid_run, passage_texts, queries, reranker, top_k=config.HYBRID_TOP_K
    )

    print("Scoring all four stages...")
    results_by_stage = {
        "BM25 Baseline": evaluate(bm25_run, qrels),
        "Dense": evaluate(dense_run, qrels),
        "Hybrid (RRF)": evaluate(hybrid_run, qrels),
        "+ Cross-Encoder Reranker": evaluate(reranked_run, qrels),
    }

    print()
    print(format_results_table(results_by_stage))
    if sample:
        print(f"\n(scoped corpus, sample={sample} — see DESIGN.md §5 for why)")

    suffix = f"_sample_{sample}" if sample else ""
    out_path = config.RESULTS_DIR / f"{dataset_key}_reranked_results{suffix}.json"
    save_results(results_by_stage, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Must match the --sample value used in data/preprocess.py, if any.",
    )
    args = parser.parse_args()
    main(args.dataset, sample=args.sample)
