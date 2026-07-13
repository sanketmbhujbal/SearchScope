"""
Dense retrieval baseline (DESIGN.md §7.2).

Encoder: BAAI/bge-small-en-v1.5 by default (see DESIGN.md §5 — encoder
size tradeoff). bge-base scores ~a few NDCG points higher on BEIR, but at
full CPU load on constrained hardware it can run 15-25s per batch, which
turns a 1M-passage encode into many hours and real thermal load.
bge-small (384-dim vs. bge-base's 768-dim, roughly 3x fewer parameters)
is meaningfully faster per batch with a modest, documented quality cost.
Pass model_name="BAAI/bge-base-en-v1.5" explicitly if you have real
CPU/GPU headroom and want the higher-quality encoder.

Index: FAISS flat (exact L2 / cosine search). At this corpus scale, an
approximate index (HNSW) buys nothing — flat search is fast enough and
guarantees reproducible recall numbers, which matters more than latency
here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class DenseRetriever:
    def __init__(self, model_name: str | None = None, backend: str = "torch"):
        """
        backend: "torch" (default) or "onnx".
            "onnx" uses optimum's ONNX Runtime backend for CPU inference,
            which is typically 2-3x faster than plain PyTorch on CPU with
            no quality loss (same weights, just a faster runtime) — free
            and open-source, no cloud dependency. Requires:
                pip install optimum[onnxruntime]
            Falls back to "torch" with a warning if that's not installed,
            rather than failing the whole run.
        """
        from config import DENSE_ENCODER_NAME

        self.model_name = model_name or DENSE_ENCODER_NAME
        self.backend = backend
        self._model = None       # lazy-loaded SentenceTransformer
        self._index = None       # faiss.IndexFlatIP
        self._doc_ids: list[str] = []

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # If this isn't already the number of physical cores, encoding
            # is silently running on a fraction of the machine — this is
            # the single most common cause of "CPU is at 100% but this is
            # taking forever": one thread pegged at 100% while N-1 cores
            # sit idle still shows as "the CPU is hot and busy."
            n_threads = os.cpu_count() or 4
            torch.set_num_threads(n_threads)
            print(f"[DenseRetriever] torch using {torch.get_num_threads()} threads "
                  f"(os.cpu_count()={os.cpu_count()})")

            if self.backend == "onnx":
                try:
                    self._model = SentenceTransformer(self.model_name, backend="onnx")
                    print("[DenseRetriever] using ONNX Runtime backend (CPU-optimized)")
                except Exception as e:
                    print(
                        f"[DenseRetriever] ONNX backend unavailable ({e}); "
                        "falling back to torch backend. To use ONNX: "
                        "pip install optimum[onnxruntime] sentence-transformers>=3.2"
                    )
                    self._model = SentenceTransformer(self.model_name)
            else:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(self, docs: list[dict], batch_size: int = 32) -> None:
        """docs: list of {"doc_id": str, "text": str}

        batch_size defaults lower than before (32, was 64) — on memory-
        constrained machines, an oversized batch can push the process into
        swapping, which shows up as exactly this symptom: high CPU/fan
        activity with very slow real progress, since most of the "work"
        is disk I/O for paged-out memory, not actual computation. If
        you have headroom (16GB+ free RAM), bumping this back up to
        64-128 will improve throughput.
        """
        import faiss
        import time

        model = self._load_model()
        self._doc_ids = [d["doc_id"] for d in docs]
        texts = [d["text"] for d in docs]

        print(f"[DenseRetriever] encoding {len(texts)} passages, batch_size={batch_size}...")
        start = time.time()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # so inner product == cosine similarity
            show_progress_bar=True,
        )
        elapsed = time.time() - start
        print(f"[DenseRetriever] encoded {len(texts)} passages in {elapsed/60:.1f} min "
              f"({len(texts)/max(elapsed, 1e-9):.1f} passages/sec)")
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Returns [(doc_id, score), ...] sorted by descending cosine similarity."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        model = self._load_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, indices = self._index.search(q_emb, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._doc_ids[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        import faiss
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        with open(path.with_suffix(".docids.json"), "w") as f:
            json.dump(self._doc_ids, f)

    def load(self, path: Path) -> None:
        import faiss
        import json

        path = Path(path)
        self._index = faiss.read_index(str(path))
        with open(path.with_suffix(".docids.json")) as f:
            self._doc_ids = json.load(f)
