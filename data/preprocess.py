"""
Build the BM25 (Pyserini/Lucene) and dense (FAISS) indexes from the raw
corpus produced by download_msmarco.py.

Usage:
    python -m data.preprocess --dataset trec-dl-2019                  # full 8.8M corpus
    python -m data.preprocess --dataset trec-dl-2019 --sample 1000000 # scoped corpus (recommended)

--sample builds indexes from a deliberately scoped subset of the corpus
rather than the full 8.8M passages (see DESIGN.md §5 for the reasoning:
full-corpus dense encoding turned out to be a genuine multi-hour
CPU/GPU-quota problem, not just a slow-but-fine one). Scoping is
qrels-aware (data/sampling.py) — every passage referenced in the TREC DL
qrels is guaranteed included, so NDCG@10/MRR@10/MAP stay fully valid;
Recall@100 is measured against a smaller but still real "needle in
haystack" pool rather than the full 8.8M. This is a permanent, documented
project decision, not a throwaway dev shortcut, though the same mechanism
is also handy for fast iteration at an even smaller target_size.

This is intentionally a thin orchestration script — the actual index-
building logic lives with the retriever it belongs to (retrieval/bm25.py,
retrieval/dense.py), so each retriever stays independently testable and
swappable per DESIGN.md §6.
"""
import argparse
import json

import config
from data.sampling import build_scoped_corpus
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever


def load_corpus(dataset_key: str) -> list[dict]:
    corpus_path = config.RAW_DATA_DIR / dataset_key / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run download_msmarco.py --dataset {dataset_key} first."
        )
    docs = []
    with open(corpus_path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main(dataset_key: str, sample: int | None = None, onnx: bool = False) -> None:
    docs = load_corpus(dataset_key)
    print(f"Loaded {len(docs)} passages for '{dataset_key}'")

    if sample:
        if sample > len(docs):
            raise ValueError(f"--sample {sample} exceeds corpus size {len(docs)}")
        qrels_path = config.RAW_DATA_DIR / dataset_key / "qrels.jsonl"
        docs = build_scoped_corpus(docs, qrels_path, target_size=sample, seed=config.RANDOM_SEED)
        print(
            "  NOTE: this is a deliberately scoped corpus (all judged passages "
            "guaranteed included, see DESIGN.md §5) — not a throwaway dev sample."
        )

    bm25_index_dir = config.get_bm25_index_dir(sample)
    faiss_index_path = config.get_faiss_index_path(sample)

    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build_index(docs, index_dir=bm25_index_dir)
    print(f"  BM25 index ready ({bm25.backend} backend) -> {bm25_index_dir}")

    print(f"Building dense index with {config.DENSE_ENCODER_NAME}...")
    dense = DenseRetriever(backend="onnx" if onnx else "torch")
    dense.build_index(docs)
    dense.save(faiss_index_path)
    print(f"  Dense index saved -> {faiss_index_path}")


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
        help="Build a deliberately scoped corpus of this size (all judged passages "
             "guaranteed included; see DESIGN.md §5). Recommended: 1000000. "
             "Omit for the full 8.8M corpus.",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX Runtime for dense encoding (typically 2-3x faster on CPU, free/open-source). "
             "Requires: pip install optimum[onnxruntime]. Falls back to torch if unavailable.",
    )
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, onnx=args.onnx)
