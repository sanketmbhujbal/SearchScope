"""
Runs the Day 1-2 baseline: BM25 vs. dense vs. hybrid retrieval, scored
against TREC DL judgments, printed as the comparison table from
DESIGN.md §3.

Usage:
    python -m eval.run_baseline_eval --dataset trec-dl-2019
    python -m eval.run_baseline_eval --dataset trec-dl-2019 --sample 1000000

--sample must match whatever value (if any) was passed to
data/preprocess.py, so this loads the matching scoped index instead of
the full one. See DESIGN.md §5 for why the corpus is deliberately scoped
(all judged passages guaranteed included via data/sampling.py) rather
than always using the full 8.8M passages.

Assumes data/preprocess.py has already been run to build the indexes
(with the same --sample value, if any).
"""
import argparse
import json

import config
from data.sampling import build_scoped_corpus
from eval.harness import evaluate, format_results_table, load_qrels, save_results
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever


def load_queries(dataset_key: str) -> dict[str, str]:
    queries_path = config.RAW_DATA_DIR / dataset_key / "queries.jsonl"
    queries = {}
    with open(queries_path) as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]
    return queries


def load_corpus(dataset_key: str, sample: int | None = None) -> list[dict]:
    corpus_path = config.RAW_DATA_DIR / dataset_key / "corpus.jsonl"
    with open(corpus_path) as f:
        docs = [json.loads(line) for line in f]
    if sample:
        qrels_path = config.RAW_DATA_DIR / dataset_key / "qrels.jsonl"
        docs = build_scoped_corpus(docs, qrels_path, target_size=sample, seed=config.RANDOM_SEED)
    return docs


def main(dataset_key: str, sample: int | None = None) -> None:
    queries = load_queries(dataset_key)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    bm25_index_dir = config.get_bm25_index_dir(sample)
    faiss_index_path = config.get_faiss_index_path(sample)

    print(f"Dataset: {dataset_key} | {len(queries)} queries" + (f" | sample={sample}" if sample else ""))

    # --- Build/load retrievers (lazy-load the corpus only if an index still needs building) ---
    print("Loading BM25 index...")
    bm25 = BM25Retriever()
    if BM25Retriever.index_exists(bm25_index_dir):
        bm25.load(bm25_index_dir)
    else:
        print("  no persisted index found; building one now (run data/preprocess.py first to avoid this)...")
        bm25.build_index(load_corpus(dataset_key, sample), index_dir=bm25_index_dir)

    print(f"Loading dense index ({config.DENSE_ENCODER_NAME})...")
    dense = DenseRetriever()
    if faiss_index_path.exists():
        dense.load(faiss_index_path)
    else:
        print("  no persisted index found; building one now (run data/preprocess.py first to avoid this)...")
        dense.build_index(load_corpus(dataset_key, sample))
        dense.save(faiss_index_path)

    hybrid = HybridRetriever(bm25, dense, rrf_k=config.RRF_K)

    # --- Run each stage over all queries ---
    bm25_run, dense_run, hybrid_run = {}, {}, {}
    for qid, qtext in queries.items():
        components = hybrid.search_with_components(
            qtext, bm25_depth=config.BM25_TOP_K, dense_depth=config.DENSE_TOP_K
        )
        bm25_run[qid] = components["bm25"]
        dense_run[qid] = components["dense"]
        hybrid_run[qid] = components["hybrid"]

    # --- Score each stage on the same harness ---
    results_by_stage = {
        "BM25 Baseline": evaluate(bm25_run, qrels),
        "Dense": evaluate(dense_run, qrels),
        "Hybrid (RRF)": evaluate(hybrid_run, qrels),
    }

    print()
    print(format_results_table(results_by_stage))
    if sample:
        print(f"\n(scoped corpus, sample={sample} — see DESIGN.md §5 for why)")

    suffix = f"_sample_{sample}" if sample else ""
    out_path = config.RESULTS_DIR / f"{dataset_key}_baseline_results{suffix}.json"
    save_results(results_by_stage, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(config.EVAL_DATASETS),
        default="trec-dl-2019",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Must match the --sample value used in data/preprocess.py, if any.",
    )
    args = parser.parse_args()
    main(args.dataset, sample=args.sample)
