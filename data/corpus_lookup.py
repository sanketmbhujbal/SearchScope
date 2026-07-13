"""
Shared corpus text lookup by doc_id.

Retrieval stages (BM25/dense/hybrid) return (doc_id, score) pairs, not
passage text — but the cross-encoder reranker (Day 3) and spot-checking
(eval/spot_check.py) both need the actual text for a given doc_id. This
streams corpus.jsonl once and collects only the requested doc_ids, rather
than loading the full (possibly 8.8M-line) corpus into memory just to
resolve a handful of lookups.
"""
import json

import config


def lookup_passage_texts(dataset_key: str, doc_ids: set[str]) -> dict[str, str]:
    """Streams corpus.jsonl once, collecting text only for the requested
    doc_ids — stops early once every requested id has been found."""
    corpus_path = config.RAW_DATA_DIR / dataset_key / "corpus.jsonl"
    found: dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["doc_id"] in doc_ids:
                found[row["doc_id"]] = row["text"]
                if len(found) == len(doc_ids):
                    break
    return found
