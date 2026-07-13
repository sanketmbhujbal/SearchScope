"""
Download the MS MARCO passage corpus and TREC DL 2019/2020 queries/qrels.

This deliberately does NOT use ir_datasets. ir_datasets' download manager
writes to a .tmp file and then renames it into place, and on some Windows
machines that rename step gets blocked (antivirus real-time scanning,
Windows Search indexing, OneDrive, etc. briefly locking the new file) —
this happens even for tiny 4KB files, so it's not a large-file problem,
it's that specific tmp-then-rename pattern. Downloading directly with
urllib straight to the destination path (no separate rename step) avoids
it entirely.

Usage:
    # 1. Manually download + extract the passage collection once:
    #      https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
    #    tar -xzf collection.tar.gz   -> produces collection.tsv (pid \t text)
    #
    # 2. Then run, pointing at that file:
    python -m data.download_msmarco --dataset trec-dl-2019 --corpus-tsv path/to/collection.tsv

    # Queries + qrels download automatically (direct from Microsoft/NIST,
    # no ir_datasets involved) — no extra flags needed for those.

Writes three JSONL files under data/raw/<dataset>/:
    corpus.jsonl   {"doc_id": ..., "text": ...}
    queries.jsonl  {"query_id": ..., "text": ...}
    qrels.jsonl    {"query_id": ..., "doc_id": ..., "relevance": ...}

Note: TREC DL 2019/2020 judges only ~43/54 queries, but the retriever
still indexes against the FULL passage collection — that's what makes
Recall@100 a meaningful number instead of a trivially high one.
"""
import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import config

# Official source URLs (Microsoft Azure blob storage + NIST) — no ir_datasets involved.
QUERIES_URLS = {
    "trec-dl-2019": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
    "trec-dl-2020": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
}
QRELS_URLS = {
    "trec-dl-2019": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
    "trec-dl-2020": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (SearchScope data downloader)"}


def _download_file(url: str, dest_path: Path) -> None:
    """Streams a URL directly to dest_path — no tmp file, no rename, so
    there's nothing for antivirus/indexing to lock mid-write."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)


def _write_queries(dataset_key: str, out_dir: Path, judged_query_ids: set[str]) -> None:
    url = QUERIES_URLS[dataset_key]
    gz_path = out_dir / "_raw_queries.tsv.gz"
    print(f"Downloading queries from {url} ...")
    _download_file(url, gz_path)

    queries_path = out_dir / "queries.jsonl"
    n_queries = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(queries_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            # Restrict to the judged subset, matching ir_datasets' "/judged"
            # variant — unjudged queries can't be scored by the eval harness
            # anyway, so keeping them just adds noise to the run.
            if qid in judged_query_ids:
                fout.write(json.dumps({"query_id": qid, "text": text}) + "\n")
                n_queries += 1
    gz_path.unlink()
    print(f"  wrote {n_queries} judged queries -> {queries_path}")


def _write_qrels(dataset_key: str, out_dir: Path) -> set[str]:
    url = QRELS_URLS[dataset_key]
    raw_path = out_dir / "_raw_qrels.txt"
    print(f"Downloading qrels from {url} ...")
    _download_file(url, raw_path)

    qrels_path = out_dir / "qrels.jsonl"
    n_qrels = 0
    judged_query_ids: set[str] = set()
    with open(raw_path, encoding="utf-8") as fin, open(qrels_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # TREC qrels format: qid  iteration(unused)  doc_id  relevance
            qid, _iteration, doc_id, relevance = line.split()
            fout.write(
                json.dumps({"query_id": qid, "doc_id": doc_id, "relevance": int(relevance)}) + "\n"
            )
            judged_query_ids.add(qid)
            n_qrels += 1
    raw_path.unlink()
    print(f"  wrote {n_qrels} qrels ({len(judged_query_ids)} judged queries) -> {qrels_path}")
    return judged_query_ids


def _write_corpus_from_local_tsv(corpus_tsv: Path, out_dir: Path) -> None:
    """Reads a manually downloaded collection.tsv (pid \\t passage text,
    tab-separated, no header) and converts it to corpus.jsonl."""
    if not corpus_tsv.exists():
        raise FileNotFoundError(
            f"--corpus-tsv path not found: {corpus_tsv}\n"
            "Download + extract it first:\n"
            "  Invoke-WebRequest -Uri "
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz "
            "-OutFile collection.tar.gz\n"
            "  tar -xzf collection.tar.gz"
        )

    corpus_path = out_dir / "corpus.jsonl"
    n_docs = 0
    with open(corpus_tsv, encoding="utf-8") as fin, open(corpus_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            pid, text = line.split("\t", 1)
            fout.write(json.dumps({"doc_id": pid, "text": text}) + "\n")
            n_docs += 1
            if n_docs % 500_000 == 0:
                print(f"  ...{n_docs} passages written")
    print(f"  wrote {n_docs} passages -> {corpus_path}")


def download(dataset_key: str, corpus_tsv: str | None = None) -> None:
    if dataset_key not in config.EVAL_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. Choose from {list(config.EVAL_DATASETS)}"
        )

    out_dir = config.RAW_DATA_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    judged_query_ids = _write_qrels(dataset_key, out_dir)
    _write_queries(dataset_key, out_dir, judged_query_ids)

    if corpus_tsv:
        print(f"Reading corpus from local file: {corpus_tsv}")
        _write_corpus_from_local_tsv(Path(corpus_tsv), out_dir)
    else:
        print(
            "No --corpus-tsv given — skipping corpus. Download + extract "
            "collection.tsv and re-run with --corpus-tsv to build the "
            "retrieval indexes."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(config.EVAL_DATASETS),
        default="trec-dl-2019",
    )
    parser.add_argument(
        "--corpus-tsv",
        default=None,
        help=(
            "Path to a manually downloaded collection.tsv (pid<TAB>text). "
            "Get it from: "
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
        ),
    )
    args = parser.parse_args()
    download(args.dataset, corpus_tsv=args.corpus_tsv)
