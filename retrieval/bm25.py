"""
BM25 retrieval baseline (DESIGN.md §7.1).

Primary backend: Pyserini/Anserini (Lucene under the hood) — the standard
for reproducible, published BM25 numbers on MS MARCO, and what's specified
in DESIGN.md. It requires a JVM (Java 11+) and building a Lucene index on
disk, which is the right tradeoff for a "lexical ceiling" baseline that
needs to match published numbers.

Fallback backend: rank_bm25 (pure Python, in-memory). This exists purely
so the pipeline logic (indexing → search → fusion → eval) can be unit
tested in any environment without a JVM. It should never be used to
produce the reported baseline numbers in results/ — those must come from
the pyserini backend so they're comparable to published figures.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, backend: str = "auto"):
        """
        backend: "pyserini" | "rank_bm25" | "auto"
            "auto" tries pyserini first and falls back to rank_bm25 with a
            warning if Java / the pyserini package isn't available.
        """
        self.backend = backend
        self._searcher = None          # pyserini LuceneSearcher
        self._bm25 = None              # rank_bm25.BM25Okapi
        self._doc_ids: list[str] = []  # index position -> doc_id (rank_bm25 path)
        self._index_dir: Path | None = None
        self._is_temp_dir: bool = True

    def build_index(self, docs: list[dict], index_dir: Path | str | None = None) -> None:
        """
        docs: list of {"doc_id": str, "text": str}
        index_dir: where to persist the Lucene index. If given, the index
            is written there permanently (survives across script runs) and
            can be reopened with load() instead of rebuilt. If omitted,
            builds into a temp dir that's discarded on close().
        """
        if self.backend in ("auto", "pyserini"):
            try:
                self._build_pyserini_index(docs, index_dir=index_dir)
                self.backend = "pyserini"
                return
            except ImportError as e:
                if self.backend == "pyserini":
                    raise
                print(f"[BM25Retriever] pyserini unavailable ({e}); falling back to rank_bm25")

        self._build_rank_bm25_index(docs)
        self.backend = "rank_bm25"

    def load(self, index_dir: Path | str) -> None:
        """Reopens a previously built Pyserini/Lucene index without rebuilding."""
        from pyserini.search.lucene import LuceneSearcher

        lucene_dir = Path(index_dir) / "lucene"
        if not lucene_dir.exists():
            raise FileNotFoundError(
                f"No Lucene index found at {lucene_dir}. Build it first with build_index(docs, index_dir=...)."
            )
        self._searcher = LuceneSearcher(str(lucene_dir))
        self._searcher.set_bm25(k1=0.9, b=0.4)
        self.backend = "pyserini"

    @staticmethod
    def index_exists(index_dir: Path | str) -> bool:
        return (Path(index_dir) / "lucene").exists()

    def _build_pyserini_index(self, docs: list[dict], index_dir: Path | str | None = None) -> None:
        """
        Writes the corpus to Pyserini's expected JsonCollection format, then
        indexes it via `python -m pyserini.index.lucene` as a subprocess.

        This deliberately does NOT use LuceneIndexer.add_batch_dict() with
        the full corpus in one call — passing millions of Python objects
        across the JNI boundary in a single call reliably segfaults the JVM
        (WinError -1073741819 / 0xC0000005 access violation) once the
        corpus gets into the millions of documents. The CLI indexer runs
        Lucene as its own process and streams documents from disk, which is
        also the documented, standard way every public MS MARCO/Pyserini
        BM25 reproduction indexes this corpus.
        """
        import json
        import subprocess
        import sys

        if index_dir is not None:
            self._index_dir = Path(index_dir)
            self._index_dir.mkdir(parents=True, exist_ok=True)
            self._is_temp_dir = False
        else:
            self._index_dir = Path(tempfile.mkdtemp(prefix="bm25_index_"))
            self._is_temp_dir = True

        corpus_dir = self._index_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        print(f"[BM25Retriever] writing {len(docs)} docs to JsonCollection format...")
        with open(corpus_dir / "docs.jsonl", "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"id": doc["doc_id"], "contents": doc["text"]}) + "\n")

        lucene_dir = self._index_dir / "lucene"
        print(f"[BM25Retriever] indexing via Pyserini CLI -> {lucene_dir} (this takes a while for large corpora)...")
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "-collection", "JsonCollection",
            "-input", str(corpus_dir),
            "-index", str(lucene_dir),
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "4",
            "-storePositions", "-storeDocvectors", "-storeRaw",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Pyserini CLI indexing failed.\n"
                f"--- stdout ---\n{result.stdout[-3000:]}\n"
                f"--- stderr ---\n{result.stderr[-3000:]}"
            )
        print("[BM25Retriever] indexing complete.")

        from pyserini.search.lucene import LuceneSearcher

        self._searcher = LuceneSearcher(str(lucene_dir))
        self._searcher.set_bm25(k1=0.9, b=0.4)  # DESIGN.md §7.1: tuned on TREC DL 2019 dev

    def _build_rank_bm25_index(self, docs: list[dict]) -> None:
        self._doc_ids = [d["doc_id"] for d in docs]
        tokenized = [d["text"].lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Returns [(doc_id, score), ...] sorted by descending score."""
        if self.backend == "pyserini":
            hits = self._searcher.search(query, k=k)
            return [(hit.docid, hit.score) for hit in hits]

        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def close(self) -> None:
        """Only removes the index if it was built into a temp dir (no
        index_dir was passed to build_index). Persistent indexes built with
        an explicit index_dir are left on disk so they can be reused via
        load() without rebuilding."""
        if self._is_temp_dir and self._index_dir and self._index_dir.exists():
            shutil.rmtree(self._index_dir, ignore_errors=True)
