# SearchScope

A hybrid retrieval, learning-to-rank, and grounded QA pipeline built end
to end against MS MARCO / TREC DL 2019, with every design decision and
every bug documented as it happened.

Most search projects show a clean pipeline and a results
table. This one also shows the four times a result looked wrong (or
suspiciously right) and what that turned out to mean. The findings below
are the actual point of the project, not an afterthought.

## Key findings

- **Naive hybrid fusion made retrieval worse, not better.** Reciprocal
  rank fusion at equal weight pulled the ranking toward the weaker of
  two retrievers (BM25) instead of the stronger one (dense), dropping
  NDCG@10 below dense retrieval alone. Cross-encoder reranking recovered
  the loss and then exceeded dense retrieval on its own. →
  [`results/day1-2_findings.md`](results/day1-2_findings.md),
  [`results/day3_findings.md`](results/day3_findings.md)
- **A too-good LTR result turned out to be a label leak.** A synthetic
  training feature was quietly built from the answer it was supposed to
  predict. Leave-one-query-out cross-validation caught the *model*
  leaking across folds but missed that the *feature* itself was
  computed from ground truth. Fixed, re-run, and the corrected number is
  far less exciting and far more honest. →
  [`results/day5-6_findings.md`](results/day5-6_findings.md)
- **SHAP and ablation disagreed, and the disagreement was the finding.**
  Four purely synthetic features showed real importance in SHAP but
  almost no impact when ablated. The model was mildly overfitting to
  noise it had variance in, not to noise it had zero variance in, a
  distinction only visible by running both analyses and comparing them. →
  [`results/day7_findings.md`](results/day7_findings.md)
- **Two bugs in the QA layer stayed invisible until someone read the
  actual output.** A citation-parsing bug flagged two accurate answers
  as hallucinations, caused by the model formatting a citation
  differently than the parser expected, fixed by switching to structured
  JSON output. Separately, a 100% rejection rate on out-of-scope queries
  looked perfect until reading the 7 cases where it wrongly rejected an
  answerable query, three of which had the answer quoted directly in the
  context it was given. →
  [`results/day8_findings.md`](results/day8_findings.md)

See [`DESIGN.md`](DESIGN.md) for the full technical design doc and every
scoping decision (corpus size, encoder choice, training methodology) with
its reasoning. See [`Substack article`](https://sanketbhujbal.substack.com/p/why-i-stopped-trusting-my-own-evaluation?r=5p0kv6&utm_campaign=post-expanded-share&utm_medium=post%20viewer&triedRedirect=true) for the narrative writeup, and
[`streamlit_app`](https://searchscope.streamlit.app/) for an interactive demo of these
findings.

## Status

✅ All 10 planned days complete.

| Day(s) | Milestone | Status |
|---|---|---|
| 1-2 | Corpus + retrieval baseline (BM25 + dense + hybrid RRF) | ✅ done. See `results/day1-2_findings.md` |
| 3 | Cross-encoder reranking | ✅ done. See `results/day3_findings.md` |
| 4 | Personalization layer | ✅ done. See `results/day4_findings.md` (real finding: HR differentiated, Eng/Sales/Legal largely collapsed, diagnosed rather than hidden) |
| 5-6 | Feature engineering (20 signals) + XGBoost LTR | ✅ done. See `results/day5-6_findings.md` (leakage bug found + fixed; corrected result: LTR ≈ reranker, expected given ~40-query training set) |
| 7 | SHAP analysis + ablation study | ✅ done. See `results/day7_findings.md` (semantic signals dominate; role_doc_affinity provably inert; synthetic-noise hypothesis refined) |
| 8 | Domain-adapted QA + rejection gate | ✅ done. See `results/day8_findings.md` (citation-parsing bug found+fixed; rejection gate correct 100% on mismatched context but over-cautious on 3/7 answerable false-rejections) |
| 9 | Failure analysis + customer complaint cases | ✅ done. See `results/day9_failure_analysis.md` (5/5 cases, all real) and `results/day9_customer_complaints.md` |
| 10 | README polish + blog post outline | ✅ done. This README + `BLOG.md` |

## Repo structure

```
searchscope/
├── data/              # MS MARCO download + preprocessing + corpus scoping
├── retrieval/         # BM25, dense (FAISS), hybrid RRF fusion
├── reranking/         # Cross-encoder (bge-reranker-base)
├── ranking/           # Feature engineering (20 signals), XGBoost LTR, SHAP
├── personalization/   # Role embeddings, affinity scoring
├── qa/                # Domain-adapted QA pipeline, rejection gate
├── eval/              # All orchestration scripts, one per day's deliverable
├── results/           # Findings docs (real evidence, per day) + saved metrics
├── notebooks/         # Colab notebook for GPU-accelerated dense encoding
├── tests/             # 95 tests, one file per module
├── config.py           # All paths, model names, and hyperparameters in one place
├── DESIGN.md            # PRD + technical design doc, updated as decisions were made
└── BLOG.md               # Narrative writeup of the project
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Important: always run scripts from the repo root as modules (`-m`)**,
not as bare file paths, so `config.py` resolves correctly:

```bash
python -m data.download_msmarco --dataset trec-dl-2019   # correct
python data/download_msmarco.py --dataset trec-dl-2019   # breaks: ModuleNotFoundError: config
```

### Downloading the corpus

Queries and qrels download automatically and directly from Microsoft
(Azure blob storage) and NIST. No `ir_datasets` is involved. This sidesteps
a Windows issue some users hit where `ir_datasets`' own tmp-file-then-
rename download step gets blocked (antivirus/indexing locking the file
mid-write), even on small files.

The full passage corpus (~8.8M passages / ~2.9GB) still needs to be
downloaded manually and passed in via `--corpus-tsv`, since a 2.9GB
in-Python download is worth doing with a purpose-built downloader (browser,
`curl`, `Invoke-WebRequest`) rather than reinventing one:

```powershell
# Windows PowerShell: do NOT use curl -L (that's aliased to
# Invoke-WebRequest and doesn't understand curl flags)
Invoke-WebRequest -Uri "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz" -OutFile "collection.tar.gz"
tar -xzf collection.tar.gz   # -> collection.tsv
```

```bash
# macOS / Linux
curl -L -o collection.tar.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz
```

Then:

```bash
python -m data.download_msmarco --dataset trec-dl-2019 --corpus-tsv collection.tsv
```

Running without `--corpus-tsv` downloads queries + qrels only and skips
the corpus, useful for inspecting the eval set before committing to the
big download.

## Running the Day 1-2 baseline

**This project runs against a scoped corpus, not the full 8.8M. This is
a deliberate, documented decision (see `DESIGN.md` §5), not a shortcut.**
Dense-encoding the full collection turned out to be a genuine
infrastructure bottleneck (hours of CPU/GPU time, thermal load, free-tier
GPU quotas exhausted). Scoping is qrels-aware (`data/sampling.py`): every
passage referenced in the TREC DL qrels is guaranteed included, so
NDCG@10/MRR@10/MAP stay fully valid. Only Recall@100 reads differently
in absolute terms vs. published full-corpus numbers, since it's measured
against a smaller (but still real) pool.

```bash
python -m data.preprocess --dataset trec-dl-2019 --sample 150000
python -m eval.run_baseline_eval --dataset trec-dl-2019 --sample 150000
```

150,000 was chosen based on measured CPU throughput after tuning (see
below), roughly ~3 hours on a typical machine, overnight-viable without
needing a GPU. Adjust up or down based on your own hardware; `config.py`'s
`RECOMMENDED_CORPUS_SCOPE` documents the reasoning.

**Speed levers already applied, in order of impact:**
1. `retrieval/dense.py` forces full CPU thread usage at load time and
   prints the actual thread count. Check this first if encoding feels
   slow; one thread pegged at 100% while the rest sit idle still looks
   like "the CPU is clearly working."
2. Default encoder is `bge-small-en-v1.5` rather than `bge-base-en-v1.5`
   (~3x fewer parameters, a documented quality/speed tradeoff, see
   `DESIGN.md` §5/§7.2).
3. Batch size defaults to 32 rather than 64, to avoid swap-driven
   slowdowns on memory-constrained machines.

**Optional further speedup: ONNX Runtime (free, open-source, no cloud
dependency):**
```bash
pip install optimum[onnxruntime] "sentence-transformers>=3.2"
python -m data.preprocess --dataset trec-dl-2019 --sample 150000 --onnx
```
Typically 2-3x faster than plain PyTorch on CPU with identical weights
and quality: same model, faster runtime. Falls back to the regular
torch backend automatically (with a printed message) if `optimum` isn't
installed, so this is safe to try without committing to it.

**If you want to index the full 8.8M anyway** (e.g. you have real GPU
infrastructure available), omit `--sample`:

```bash
python -m data.preprocess --dataset trec-dl-2019
python -m eval.run_baseline_eval --dataset trec-dl-2019
```

For dense encoding specifically, `notebooks/searchscope_dense_encode.ipynb`
runs on a free Colab GPU and downloads the corpus directly inside Colab
(no local upload). See the notebook for details. Given the scoped-corpus
approach above is now the primary path, this is only useful if you're
deliberately doing a full-corpus run.

## Running the Day 3 baseline (cross-encoder reranking)

Reuses the persisted BM25 + dense indexes from Day 1-2. No rebuilding.
Takes each query's hybrid top-100, reranks with `BAAI/bge-reranker-base`
to a top-20, and produces the 4-way comparison table (BM25 / Dense /
Hybrid / +Reranker):

```bash
python -m eval.run_reranked_eval --dataset trec-dl-2019 --sample 150000
```

Given the Day 1-2 finding that naive RRF underperformed Dense alone (see
`results/day1-2_findings.md`), the interesting question here isn't just
"does reranking help" but specifically whether the cross-encoder recovers
the ranking quality that equal-weighted fusion gave away.

## Running the Day 4 personalization demo

Produces the "same query, different top-5 per role" table (Engineer /
Sales / HR / Legal) that DESIGN.md's Layer 3b calls for. Reuses the
persisted BM25 + dense indexes; fits a TF-IDF affinity model on the
scoped corpus (a few seconds, not a heavy step):

```bash
python -m eval.run_personalization_demo --dataset trec-dl-2019 --sample 150000 --query policy
```

**Note on the blend used here:** this script combines base retrieval
relevance with role affinity using a simple fixed weight, purely to make
the demo visible before the LTR layer exists. That's explicitly *not*
the production design. `personalization/role_affinity.py`'s
`rerank_by_role()` deliberately raises `NotImplementedError` to keep a
hand-tuned blend out of the library code. The actual plan (DESIGN.md §9)
is for `role_doc_affinity` to be feature 20 in the Day 5-6 XGBoost LTR
model, letting the ranker learn when affinity should matter vs. when raw
relevance should dominate, rather than a fixed formula guessing that
upfront.

## Running the Day 5-6 baseline (feature engineering + XGBoost LTR)

Builds all 20 signals per candidate (`ranking/features.py`) and trains
via leave-one-query-out cross-validation across the 43 judged queries.
See `DESIGN.md` §9.2 for why (MS MARCO's full training-triples set is
out of scope for this timeline, same kind of decision as the corpus
scoping in Days 1-2):

```bash
python -m eval.run_ltr_eval --dataset trec-dl-2019 --sample 150000
```

This re-runs retrieval + cross-encoder reranking for every query to
assemble the candidate pool with features attached (similar cost to Day
3), then trains 43 separate models (one per LOQO fold, fast, seconds
each) and evaluates. It prints a sanity-check `+Reranker` row computed on
the identical candidate pool used for `+LTR`, which should closely match
Day 3's recorded number. If it doesn't, that flags a candidate-pool
inconsistency between the scripts rather than a real ranking difference
(exactly the kind of mismatch that caused the Day 3 metric-depth bug, so
it's checked automatically here rather than trusted on faith).

## Running Day 7 (SHAP analysis + ablation study)

Both scripts reuse a cached candidate+feature dataset (built once by
whichever of Day 5-6/7's scripts runs first, see
`eval/build_ltr_dataset.py`'s caching note) so you only pay the ~30 min
retrieval+reranking cost once, not on every script.

**SHAP analysis** produces the three committed artifacts (DESIGN.md
§12): global feature importance, per-query-intent-class breakdown, and a
single-query waterfall, saved as PNGs under `results/day7_shap/`:

```bash
python -m eval.run_shap_analysis --dataset trec-dl-2019 --sample 150000
```

Trains one model on *all* queries' candidates (not LOQO). LOQO exists
for honest eval numbers (Day 5-6), but SHAP is explaining the model that
would actually be deployed, which is trained on everything available.

**Ablation study** tests 10 variants (drop each signal category in
turn, plus `BM25 only`/`Cross-encoder only` as training-pipeline sanity
checks) via the same LOQO CV as Day 5-6, and prints the NDCG@10 delta
vs. the full model:

```bash
python -m eval.ablation --dataset trec-dl-2019 --sample 150000
```

This includes a `"No synthetic (bonus)"` variant beyond DESIGN.md's
original table, added specifically to test the hypothesis from
`results/day5-6_findings.md`: that the purely synthetic features
(`doc_recency`, `source_authority`, `simulated_ctr`, `simulated_dwell_time`)
might be diluting the model's limited training signal. If dropping them
recovers or exceeds the reranker's NDCG@10, that confirms it.

If you change `ranking/features.py` after the cache is built, pass
`--rebuild-cache` to either script (or delete `cache/*.pkl` directly).
Otherwise you'll silently evaluate on stale features, the same class of
mistake that caused the Day 5-6 leakage bug.

## Running Day 8 (domain-adapted QA + rejection gate)

**Provider note:** uses the OpenAI API (`gpt-4o-mini`), not AWS Bedrock
as originally specified in `DESIGN.md`. This is a documented scoping decision
(see `qa/grounded_qa.py`'s module docstring). Requires `OPENAI_API_KEY`
set in your environment:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

Reuses the cached LTR dataset from Days 5-7 (no new retrieval cost), then
builds two test sets from the same 43 judged queries: each query with its
own top-5 passages ("answerable"), and each query paired with a different,
randomly-selected query's top-5 passages ("unanswerable", mismatched
context that shouldn't support an answer). See the module docstring for
why this substitutes for DESIGN.md's "50 manually identified unanswerable
queries," which this project's fully-judged query set doesn't naturally
have:

```bash
python -m eval.run_qa_eval --dataset trec-dl-2019 --sample 150000
```

Uses OpenAI's structured-output mode (`client.chat.completions.parse()`
with a Pydantic schema) rather than free-text parsing. A real citation-
format bug in the first run (`[doc_id: 12345]` vs. the expected
`[12345]`) showed free-text parsing is fragile in a way structured
output isn't. See `qa/grounded_qa.py`'s module docstring and
`results/day8_findings.md` for the full story.

Latency is reported as P50/P95/P99 (not just avg/max). Percentiles are
the standard way to reason about an SLA, since a single outlier or the
mean alone hides how bad the worst 5-1% of requests actually get.

Cheap smoke test first (5-10 queries instead of all 43, to sanity-check
before spending more API calls):

```bash
python -m eval.run_qa_eval --dataset trec-dl-2019 --sample 150000 --limit 5
```

Produces:
- `results/{dataset}_qa_metrics{suffix}.json`: Answer Rejection Rate
  (fully automated) and a citation-hygiene proxy rate
- `results/day8_qa_review{suffix}.md`: every answerable-set Q&A pair
  formatted for manual review. Answer Supported Rate is deliberately
  **not** computed automatically. DESIGN.md specifies it as a human
  spot-check, and faking an automated proxy for "is this answer actually
  correct" would be a weaker, less honest number than doing the
  10-minute manual read.

## Running Day 9 (failure analysis + customer complaint cases)

Four of the five failure cases already have complete, real evidence in
earlier findings files (`results/day1-2_findings.md`,
`results/day3_findings.md`, `results/day4_findings.md`,
`results/day7_findings.md`) and needed no new code. See
`results/day9_failure_analysis.md` for how they're organized into the
failure-case format.

The fifth (vocabulary mismatch) needed a real example mined from actual
retrieval output rather than an invented one:

```bash
python -m eval.mine_failure_cases --dataset trec-dl-2019 --sample 150000
```

Reuses the cached LTR dataset (no new retrieval cost). Ranks judged
queries by how starkly BM25 missed the true relevant document relative
to dense retrieval, and prints the top candidates with both the correct
answer's text and BM25's actual (wrong) top-1 pick. Pick the clearest
one and fill in `results/day9_failure_analysis.md`'s Case 2 template.

Customer complaint cases (5, reframing the same failures as product
issues) are in `results/day9_customer_complaints.md`.

## Testing

```bash
pytest tests/ -v
```

Tests use a small synthetic corpus (see `tests/fixtures.py`) so the
pipeline logic can be validated without downloading the full MS MARCO
corpus or any models.
