"""
Qrels-aware corpus scoping (DESIGN.md §5 — corpus scoping decision).

Indexing the full 8.8M-passage MS MARCO collection turned out to be a
genuine infrastructure problem, not just a slow-but-fine one: CPU
encoding is a multi-hour thermal load, and GPU encoding (Colab/Kaggle)
either exhausted free quota or, in one case, projected 30+ hours for a
workload a T4 should clear in 1-3 hours — almost certainly an
environment issue with that specific setup, not something intrinsic to
the corpus size.

Rather than keep chasing infrastructure, we scope the corpus down
deliberately: every passage referenced in the TREC DL 2019/2020 qrels is
guaranteed to be included (so NDCG@10 / MRR@10 / MAP stay fully valid —
those metrics only ever look at judged passages), plus a random sample of
the remainder so Recall@100 still means something (retrieval still has
to find the right passage among a large, mostly-irrelevant pool, just a
smaller one than 8.8M).

This is a permanent, documented scoping decision (see DESIGN.md §5), not
a throwaway dev shortcut — the same mechanism also happens to be useful
for fast local iteration at a smaller target_size.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def load_judged_doc_ids(qrels_path: Path) -> set[str]:
    """All doc_ids referenced anywhere in the qrels file, regardless of
    relevance grade (including grade-0 judged-nonrelevant docs — those
    are also load-bearing for NDCG/MAP, since a metric needs to know a
    doc was judged, not just that it was relevant)."""
    doc_ids: set[str] = set()
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.add(row["doc_id"])
    return doc_ids


def build_scoped_corpus(
    docs: list[dict],
    qrels_path: Path,
    target_size: int,
    seed: int = 42,
) -> list[dict]:
    """
    Returns a subset of `docs` of size `target_size` (or the full corpus,
    if target_size >= len(docs)), guaranteeing every judged doc_id from
    qrels_path is included. The remainder is filled with a random sample
    of unjudged passages, seeded for reproducibility.

    Raises if target_size is smaller than the number of judged docs —
    that would silently drop judged passages and quietly break the eval
    metrics, which is exactly the failure mode this function exists to
    prevent.
    """
    judged_ids = load_judged_doc_ids(qrels_path)

    judged_docs = [d for d in docs if d["doc_id"] in judged_ids]
    unjudged_docs = [d for d in docs if d["doc_id"] not in judged_ids]

    n_judged = len(judged_docs)
    if target_size < n_judged:
        raise ValueError(
            f"target_size={target_size} is smaller than the {n_judged} judged "
            f"passages in {qrels_path} — this would silently drop judged "
            "passages and invalidate the eval metrics. Use target_size >= "
            f"{n_judged}."
        )

    n_fill = target_size - n_judged
    random.seed(seed)
    fill_docs = random.sample(unjudged_docs, min(n_fill, len(unjudged_docs)))

    scoped = judged_docs + fill_docs
    random.seed(seed)
    random.shuffle(scoped)  # avoid judged docs all being contiguous / first

    print(
        f"Scoped corpus: {len(scoped)} passages total "
        f"({n_judged} judged, guaranteed included; {len(fill_docs)} random fill, seed={seed})"
    )
    return scoped
