#!/usr/bin/env bash
# Convenience setup for local development.
# Note: pyserini requires Java 11+ on PATH; check `java -version` first.
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment ready. Next steps:"
echo "  1. Download + extract the corpus (see README.md for Windows PowerShell command):"
echo "     curl -L -o collection.tar.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
echo "     tar -xzf collection.tar.gz"
echo "  2. python -m data.download_msmarco --dataset trec-dl-2019 --corpus-tsv collection.tsv"
echo "  3. python -m data.preprocess --dataset trec-dl-2019"
echo "  4. python -m eval.run_baseline_eval --dataset trec-dl-2019"
echo "  5. pytest tests/ -v"
