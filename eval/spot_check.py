"""
Spot-check dense retrieval's top-1 results against actual query text.

MRR@10 = 1.0000 exactly means every judged query had a relevant passage
at rank 1 — plausible on a 150K-passage scoped corpus (TREC DL 2019
queries often have many relevant passages, and a smaller haystack makes
rank-1 hits easier), but "exactly perfect" is worth eyeballing rather
than taking on faith. This prints query text next to the actual top-1
passage text and its TREC relevance grade (if judged) so you can
confirm the hits are genuine, not an artifact of the scoped corpus or a
bug in the eval wiring.

Usage:
    python -m eval.spot_check --dataset trec-dl-2019 --sample 150000 --n 5
"""
import argparse
import json
import random

import config
from data.corpus_lookup import lookup_passage_texts
from eval.harness import load_qrels
from retrieval.dense import DenseRetriever


def load_queries(dataset_key: str) -> dict[str, str]:
    queries_path = config.RAW_DATA_DIR / dataset_key / "queries.jsonl"
    queries = {}
    with open(queries_path) as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]
    return queries


def main(dataset_key: str, sample: int | None, n: int, seed: int) -> None:
    queries = load_queries(dataset_key)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    faiss_index_path = config.get_faiss_index_path(sample)
    if not faiss_index_path.exists():
        raise FileNotFoundError(
            f"No dense index found at {faiss_index_path}. Run data/preprocess.py first "
            f"with the same --sample value ({sample})."
        )

    print(f"Loading dense index ({config.DENSE_ENCODER_NAME})...")
    dense = DenseRetriever()
    dense.load(faiss_index_path)

    random.seed(seed)
    query_ids = random.sample(list(queries.keys()), min(n, len(queries)))

    # Run top-1 search for each sampled query first, then batch-lookup all
    # the resulting passage texts in a single corpus pass.
    top1_by_query: dict[str, tuple[str, float]] = {}
    for qid in query_ids:
        hits = dense.search(queries[qid], k=1)
        if hits:
            top1_by_query[qid] = hits[0]

    doc_ids_needed = {doc_id for doc_id, _score in top1_by_query.values()}
    passage_texts = lookup_passage_texts(dataset_key, doc_ids_needed)

    print(f"\nSpot-checking {len(query_ids)} random judged queries (seed={seed}):\n")
    print("=" * 100)
    for qid in query_ids:
        query_text = queries[qid]
        print(f"Query [{qid}]: {query_text}")

        if qid not in top1_by_query:
            print("  (no results returned)")
            print("=" * 100)
            continue

        doc_id, score = top1_by_query[qid]
        relevance = qrels.get(qid, {}).get(doc_id)
        relevance_str = f"grade {relevance}" if relevance is not None else "UNJUDGED (not in qrels)"
        passage = passage_texts.get(doc_id, "(passage text not found)")
        passage_preview = passage[:250] + ("..." if len(passage) > 250 else "")

        print(f"  Top-1 doc_id: {doc_id}  |  cosine score: {score:.4f}  |  relevance: {relevance_str}")
        print(f"  Passage: {passage_preview}")
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--n", type=int, default=5, help="Number of random queries to spot-check")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, n=args.n, seed=args.seed)
