import json
import tempfile
from pathlib import Path

import config
from eval.spot_check import lookup_passage_texts


def test_lookup_passage_texts_finds_requested_docs_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_key = "trec-dl-2019"
        out_dir = Path(tmpdir) / dataset_key
        out_dir.mkdir(parents=True)

        original_raw_dir = config.RAW_DATA_DIR
        config.RAW_DATA_DIR = Path(tmpdir)
        try:
            with open(out_dir / "corpus.jsonl", "w") as f:
                f.write(json.dumps({"doc_id": "d1", "text": "first passage"}) + "\n")
                f.write(json.dumps({"doc_id": "d2", "text": "second passage"}) + "\n")
                f.write(json.dumps({"doc_id": "d3", "text": "third passage"}) + "\n")

            result = lookup_passage_texts(dataset_key, {"d2", "d3"})
            assert result == {"d2": "second passage", "d3": "third passage"}
            assert "d1" not in result
        finally:
            config.RAW_DATA_DIR = original_raw_dir


def test_lookup_passage_texts_handles_missing_doc_id_gracefully():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_key = "trec-dl-2019"
        out_dir = Path(tmpdir) / dataset_key
        out_dir.mkdir(parents=True)

        original_raw_dir = config.RAW_DATA_DIR
        config.RAW_DATA_DIR = Path(tmpdir)
        try:
            with open(out_dir / "corpus.jsonl", "w") as f:
                f.write(json.dumps({"doc_id": "d1", "text": "first passage"}) + "\n")

            # Requesting a doc_id that doesn't exist shouldn't crash — the
            # caller (main()) falls back to a placeholder string for it.
            result = lookup_passage_texts(dataset_key, {"d1", "nonexistent"})
            assert result == {"d1": "first passage"}
        finally:
            config.RAW_DATA_DIR = original_raw_dir
