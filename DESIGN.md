# SearchScope: PRD & Design Document (v2.0)

| | |
|---|---|
| **Author** | Sanket Bhujbal |
| **Status** | Ready for implementation |
| **Target corpus** | MS MARCO Passage Ranking (TREC DL 2019/2020) |
| **Target timeline** | 10 working days |

This project demonstrates signal engineering, cross-encoder reranking,
role-based personalization, domain-adapted QA, and rigorous offline
evaluation.

## 1. Problem Statement

Enterprise search has a fundamental relevance problem. Keyword systems
miss semantic intent. Dense retrievers lose lexical precision. Neither
adapts to a specific customer's vocabulary, and neither personalizes
results by who is searching. The result: employees waste time
reformulating queries, miss relevant documents, and lose trust in the
product.

SearchScope replicates these challenges at a reproducible, benchmarkable
scale and demonstrates a principled engineering path through them, from
retrieval baseline to personalized, LLM-grounded answers.

## 2. Goals & Non-Goals

**Goals**
- Build a 4-stage pipeline: hybrid retrieval → cross-encoder reranking → LTR ranking → domain-adapted QA
- Measure NDCG@10, MRR@10, Recall@100 at each stage with a full before/after story
- Engineer 18-20 signals and run leave-one-out ablations to quantify each signal's contribution
- Add a role-based personalization layer that re-ranks results by user context (Engineer / Sales / HR / Legal)
- Produce SHAP feature importance plots as a first-class deliverable
- Write 5 visual failure case studies and 3-5 customer complaint cases with root cause and signal response
- Publish a clean GitHub repo and a 6-minute engineering blog post

**Non-Goals**
- Real-time serving infrastructure or production deployment
- Multi-modal retrieval (images, tables)
- Synthetic enterprise corpus (Slack/Confluence/GitHub mix), deferred to v3
- Fine-tuning a custom embedding model (adapter/prompt-based domain adaptation only)
- Signal Suggestion Engine (auto-detect failure patterns → retrain loop), documented as Future Work

## 3. Success Metrics

| Metric | BM25 Baseline | Hybrid | After Reranker | After LTR | Target |
|---|---|---|---|---|---|
| NDCG@10 | ~0.43 | > 0.52 | > 0.56 | > 0.60 | 0.60+ |
| MRR@10 | ~0.35 | > 0.44 | > 0.48 | > 0.52 | 0.52+ |
| Recall@100 | ~0.75 | > 0.82 | > 0.84 | > 0.86 | 0.86+ |
| Answer Supported Rate | N/A | N/A | N/A | N/A | > 85% |
| Answer Rejection Rate | N/A | N/A | N/A | N/A | > 90% |
| Retrieval Latency | < 50ms | < 150ms | < 400ms | < 500ms | < 500ms |

Baseline figures are published Pyserini BM25 numbers on the full TREC DL
2019 collection. Intermediate targets are informed by published hybrid +
cross-encoder results on MS MARCO. **Note:** this project runs against a
~1M-passage scoped corpus (see §5) rather than the full 8.8M passages, so
Recall@100 in particular will read differently in absolute terms than
these published full-corpus baselines. NDCG@10/MRR@10/MAP remain
directly comparable since they only evaluate judged passages, which are
guaranteed present at any scope size.

## 4. User Stories

**Story 1: The Overloaded Engineer**
Query: *"how do I set up the auth service"* → Expected: recent GitHub issue with setup steps. BM25 returns a 3-year-old Confluence page (freshness signal absent). Fix: document freshness + source authority features in LTR.

**Story 2: The New Employee**
Query: *"deployment process"* → Expected: release pipeline runbook. BM25 returns an unrelated HR onboarding doc (vocabulary mismatch: internal term is "release pipeline", not "deployment process"). Fix: dense retrieval captures semantic equivalence; cross-encoder reranks by intent.

**Story 3: The Executive**
Query: *"refund policy"* → Expected: direct answer with citation. BM25 returns a list of 10 policy documents, no synthesized answer (no QA layer). Fix: domain-adapted QA layer with grounded answer synthesis and source citations.

**Story 4 (v2): Role-Based Re-ranking**
Query: *"policy"* → Engineer sees code review / deployment / incident response policies; Sales sees discount / refund / SLA policies; HR sees leave / benefits / performance review policies. Without personalization, the same query returns the same docs regardless of who's searching. Fix: user role embedding + team-document affinity score in the LTR re-ranking layer.

## 5. Constraints & Assumptions

- CPU-only inference throughout, no GPU in the demo environment
- MS MARCO TREC DL 2019/2020 (~43 queries, 9k judged passages) for rigorous eval; full dev set for LTR training
- Cross-encoder: BAAI/bge-reranker-base, CPU-viable, strong MARCO performance
- Dense encoder: BAAI/bge-small-en-v1.5, switched down from bge-base-en-v1.5 during implementation for CPU feasibility (see the corpus-scope entry below for the full story: same root cause, same kind of tradeoff)
- LLM calls via OpenAI (gpt-4o-mini), minimized to control latency. Switched from the originally-planned AWS Bedrock/Claude Haiku during implementation: existing OpenAI credits meant zero new account/access-request setup, versus Bedrock's AWS account + explicit Anthropic-model access request. Same scoping-decision pattern as the corpus size and encoder choice above. See `qa/grounded_qa.py`'s module docstring.
- User roles are simulated, clearly documented as synthetic in the repo
- All relevance judgments from TREC annotations, no manual labeling
- **Corpus scope (added during implementation): the project indexes a
  qrels-aware scoped subset (~150K passages, tuned to measured CPU
  throughput) rather than the full 8.8M MS MARCO passage collection.**
  Dense-encoding the full collection turned out to be a genuine
  infrastructure bottleneck, multi-hour CPU load with real thermal cost,
  and GPU attempts either exhausted free Colab quota or (on one Kaggle
  attempt) projected 30+ hours for a workload a T4 should clear in 1-3
  hours, suggesting an environment issue rather than something intrinsic
  to the corpus size. Scoping is implemented in `data/sampling.py`: every
  passage referenced anywhere in the TREC DL qrels is guaranteed
  included, so NDCG@10/MRR@10/MAP, which only ever evaluate judged
  passages, stay fully valid at any scope size. Recall@100 is measured
  against a smaller pool, so its absolute value isn't directly comparable
  to published full-corpus numbers, but it still measures the same
  thing: can retrieval find the right passage among a large,
  mostly-irrelevant set. This is a deliberate, documented scoping
  decision, not a silent shortcut. Full-corpus indexing remains possible
  (`--sample` is optional) for anyone with the infrastructure to run it.
  Even at reduced scope, CPU encoding with bge-base ran unacceptably
  slowly (~20s per 256-passage batch, which is itself a sign of
  underutilized threads or memory pressure, not just a slow model),
  `retrieval/dense.py` now forces `torch.set_num_threads()` to the
  machine's full core count at load time and defaults to a smaller batch
  size to avoid swap-driven slowdowns; the default encoder was switched
  to bge-small-en-v1.5 (see above); and an optional ONNX Runtime backend
  (`--onnx`, free/open-source, typically 2-3x faster on CPU with
  identical weights) is available as a further lever without introducing
  any cloud dependency.

---

# Technical Design

## 6. System Architecture

Five independently-evaluable layers, each swappable without affecting the others, mirroring how a production search quality team ships incremental improvements.

| Layer | Component | Key Technology |
|---|---|---|
| 1 | Retrieval: BM25 + Dense + Hybrid Fusion | Pyserini, FAISS, bge-small-en-v1.5, RRF |
| 2 (new) | Cross-Encoder Reranking | BAAI/bge-reranker-base |
| 3 | Learning-to-Rank with 18-20 signals | XGBoost LambdaRank, SHAP |
| 3b (new) | Role-Based Personalization | User role embeddings, affinity scoring |
| 4 | Domain-Adapted QA + Rejection Gate | OpenAI (gpt-4o-mini), grounded prompting |
| 5 | Evaluation Harness + Failure Analysis | pytrec_eval, NDCG/MRR/MAP, ablation runner |

## 7. Layer 1: Retrieval

**7.1 BM25 Baseline.** Pyserini + Anserini index. k1/b tuned on TREC DL 2019 dev queries. This is the lexical ceiling. Everything above is measured as lift over BM25.

**7.2 Dense Retrieval.** Encoder: bge-small-en-v1.5 (switched down from bge-base-en-v1.5 during implementation, see §5 for the reasoning; bge-base remains available as an explicit override for anyone with more CPU/GPU headroom). Passages pre-encoded, indexed in FAISS flat (exact L2/cosine). Top-100 by cosine similarity at query time.
- Why bge-small over bge-base in practice: bge-base scores ~a few NDCG points higher on BEIR, but at full CPU load on constrained hardware it ran multi-hour to encode even a 1M-passage scoped corpus (and even GPU attempts on free-tier Colab/Kaggle hit quota limits or anomalously slow projected runtimes). bge-small (384-dim vs. 768-dim, ~3x fewer parameters) trades a modest amount of retrieval quality for CPU feasibility: a deliberate, documented tradeoff, not a silent regression.
- Why FAISS flat: at demo scale, HNSW adds index complexity without latency benefit; exact search guarantees reproducible recall numbers.

**7.3 Hybrid Fusion.** Reciprocal Rank Fusion (k=60) blending BM25 and dense ranked lists. Score-distribution agnostic, no normalization needed. A learned linear interpolation weight is also computed as an LTR feature.

## 8. Layer 2 (new): Cross-Encoder Reranking

After hybrid retrieval returns top-100, a cross-encoder (BAAI/bge-reranker-base) scores each query-passage pair jointly. Top-20 pass to the LTR layer.
- Cross-encoders see query and doc together, capturing fine-grained interaction signals bi-encoders miss, too slow for full-corpus retrieval, fast enough for top-100 reranking.
- Pipeline position: Retrieve (BM25 + dense) → Rerank (cross-encoder) → LTR → QA, the industry-standard 3-stage search stack.

## 9. Layer 3: Learning-to-Rank

### 9.1 Signal Engineering (18-20 features)

| Category | Signal | Description |
|---|---|---|
| Lexical | BM25 score | Baseline term overlap relevance |
| Lexical | BM25 rank position | Rank-based, less sensitive to score scale |
| Lexical | Query-doc term overlap | Fraction of query terms present in doc |
| Lexical | Title match score | Query term overlap with doc title specifically |
| Semantic | Dense cosine similarity | Intent-level match via bi-encoder |
| Semantic | Dense rank position | Rank-based semantic signal |
| Semantic | Cross-encoder score (new) | Fine-grained joint query-doc interaction score |
| Semantic | RRF fusion score | Combined hybrid signal from retrieval layer |
| Query | Query length (tokens) | Short vs. verbose queries behave differently |
| Query | Query entropy (new) | Measures query specificity; low = ambiguous |
| Query | Query IDF mean (new) | Avg IDF of query terms; high = rare/specific |
| Query | Query intent class | Navigational / informational / transactional |
| Document | Doc length (log) | Penalize very short/long passages |
| Document | Section importance (new) | Heading vs. body vs. footer position |
| Document | Metadata overlap (new) | Query terms in doc metadata/tags |
| Behavior | Simulated CTR (new) | Click-through proxy from training triples |
| Behavior | Simulated dwell time (new) | Engagement proxy; positive passages score higher |
| Freshness | Document recency | Log-scaled days since creation |
| Authority | Source authority | Domain-level reliability (docs > forums) |
| Personalization | Role-document affinity (new) | Cosine sim between user role embedding and doc |

### 9.2 Model

XGBoost LambdaRank (`rank:pairwise`), evaluated on TREC DL 2019/2020 via pytrec_eval.

**Training methodology (added during implementation, another documented scoping decision, see §5):** MS MARCO's full training-triples set is hundreds of thousands of examples, a multi-GB download and hours of additional compute beyond this project's timeline. Instead, the ranker is trained and evaluated via **leave-one-query-out (LOQO) cross-validation** on the 43 TREC DL 2019 judged queries: for each query, a model is trained on the other 42 queries' candidates and used to score the held-out query, so every reported prediction comes from a model that never saw that query's labels. This is a standard, defensible approach for a small labeled set. It just means the training signal comes from ~40 queries' worth of candidates (~4,000 labeled examples across ~100 candidates each) rather than MS MARCO's full triples set, which will show up as a real ceiling on how much the ranker can learn versus a production system trained on the full training set.
- Why XGBoost: SHAP values directly answer "which signal matters for which query type": interpretability > marginal accuracy gain from neural rankers at this scale.
- Why LambdaRank: directly optimizes NDCG rather than pointwise regression.

### 9.3 Ablation Study

| Ablation | Signals removed | Expected NDCG delta |
|---|---|---|
| Full model | None | Baseline |
| No lexical | BM25 score/rank, term overlap, title match | TBD |
| No semantic | Dense cosine/rank, cross-encoder, RRF | TBD |
| No query | Query entropy, IDF mean, intent class, length | TBD |
| No behavior | Simulated CTR, dwell time | TBD |
| No freshness/authority | Recency, source authority | TBD |
| No personalization | Role-doc affinity | TBD |
| BM25 only | All except BM25 score | TBD |
| Cross-encoder only | All except cross-encoder score | TBD |

TBD values filled post-training. Table structure pre-committed to force honest measurement.

## Layer 3b (new): Role-Based Personalization

Four simulated roles (Engineer, Sales, HR, Legal), each with a pre-defined topic affinity vector. At query time, cosine similarity between the role vector and a document's TF-IDF topic distribution becomes feature 20 in the LTR model. The ranker learns when to weight it vs. when pure relevance dominates. Evaluated by running the same TREC DL 2019 queries under each role and comparing top-5 diffs (e.g. "policy" for Engineer vs. HR).

## 10. Layer 4: Domain-Adapted QA

**10.1 Domain Adaptation.** Two lightweight techniques, no fine-tuning: corpus-aware system prompt (vocabulary/acronyms/topic distribution injected) and retrieval-grounded prompting (LLM sees only top-5 LTR-ranked passages, must cite sources, cannot hallucinate facts it wasn't given).

**10.2 Rejection Gate.** If no supporting evidence is detected, return "No confident answer found from the retrieved documents" rather than fabricate.

**Implementation note (added during Day 8):** rather than 50 manually identified unanswerable queries, the rejection gate is tested using the project's 43 TREC DL 2019 judged queries (all genuinely answerable) paired with a different, randomly-selected query's top-5 passages as deliberately mismatched context, a reproducible, explainable substitute testing the same underlying behavior (does the model refuse when the evidence doesn't support an answer). See `eval/run_qa_eval.py`'s module docstring.

**10.3 QA Evaluation.** Answer Supported Rate (human spot-check, 50 samples), Answer Rejection Rate (% of unanswerable queries correctly rejected), end-to-end latency < 2s on CPU.

## 11. Layer 5: Evaluation & Failure Analysis

**11.1 Offline Metrics.** NDCG@10 (primary), MRR@10, Recall@100, MAP, all via pytrec_eval against TREC DL 2019/2020 official judgments.

**Visual Failure Analysis (5 fully worked examples)**, fully written up with real evidence in `results/day9_failure_analysis.md`: naive hybrid fusion underperforming Dense alone, vocabulary mismatch (a real mined case: query "what are the three percenters?", BM25 rank 84 vs. dense rank 2, with BM25's actual top-1 pick being an unrelated Jamaica weather/Microsoft Project page matching only on the token "percent"), cross-encoder recovering the fusion loss, personalization gap (real for one role, absent for others), and synthetic-feature SHAP-vs-ablation overfitting. Note this replaces the originally-planned "freshness failure" and "authority failure" as two separate cases. DESIGN.md §5 documents `doc_recency`/`source_authority` as synthetic hash-based values with no real dates to demonstrate a genuine freshness bug against, so the honest, evidenced version of that story is the SHAP-vs-ablation finding instead (see `results/day9_failure_analysis.md`, Case 5, for the full reasoning).

**QA-layer failure taxonomy (added during Day 9, adapted from external review feedback):** the retrieval/ranking failure cases above are naturally 5 different *causes* within one pipeline stage. The Day 8 QA layer's failures are better organized by *where in the QA pipeline* the failure occurred, since retrieval, rejection-gate calibration, and generation-faithfulness are genuinely separable failure modes with different fixes:

| Type | Symptom | Root cause | Example from this project |
|---|---|---|---|
| A: Retrieval-induced | Rejected or wrong answer, but the true answer exists in the corpus | The relevant passage never made it into the QA layer's top-K context | N/A observed directly. Day 8's context always came from this project's own top-K, so a genuine "answer exists but never retrieved" case would need checking against the full 150K corpus, not just the QA test set |
| B: Rejection-gate failure | Model rejects (or answers) when it shouldn't | Rejection-gate calibration: over- or under-cautious relative to what's actually in context | Day 8: 3/7 false rejections were clear misses. A directly quotable answer was in context and the model rejected anyway (see `results/day8_findings.md`) |
| C: Generation/citation failure | Answer is right, but citations are wrong, missing, or malformed | Output-format drift under an ambiguous instruction | Day 8: the `[doc_id: 12345]` vs `[12345]` citation-format bug, fixed by switching to structured JSON output rather than free-text parsing (see `qa/grounded_qa.py`) |

This table is a live cross-reference back to `results/day8_findings.md`'s case-by-case analysis, not a duplicate of it.

**Customer Complaint Cases (3-5, new in v2)** reframe failure analysis as product thinking. Full table with 5 cases in `results/day9_customer_complaints.md`, each tied to a specific failure case above with an evidence citation rather than a hypothetical.

## 12. SHAP Analysis: First-Class Deliverable

Three committed artifacts: global feature importance bar chart, per-query-type SHAP breakdown (navigational/informational/transactional), and a single-query SHAP waterfall. These validate the signal engineering decisions and directly prepare answers to "How does your ranker decide?"

## 13. Repository Structure

```
searchscope/
├── data/            # MS MARCO download + preprocessing
├── retrieval/       # BM25 (Pyserini), dense (FAISS), hybrid RRF fusion
├── reranking/        # Cross-encoder (bge-reranker-base)
├── ranking/          # Feature engineering (20 signals), XGBoost LTR, SHAP
├── personalization/  # Role embeddings, affinity scoring, re-ranking
├── qa/                # Domain-adapted QA pipeline, rejection gate
├── eval/              # pytrec_eval harness, ablation runner, failure logger
├── notebooks/         # Results viz, ablation tables, failure cases, SHAP plots
├── tests/             # Unit tests for each pipeline stage
├── README.md          # Project overview, results table, setup guide
├── DESIGN.md           # This document
└── BLOG.md             # Engineering blog post draft
```

## 14. Build Plan (10 Days)

| Day(s) | Milestone | Deliverable | JD Bullet Covered |
|---|---|---|---|
| 1-2 | Corpus + Retrieval Baseline | BM25 vs. dense vs. hybrid eval table | Evaluation |
| 3 | Cross-Encoder Reranking | 4-way comparison table with reranker | Combine LLMs + search |
| 4 | Personalization Layer | Role-based re-ranking demo | Invent personalization signals |
| 5-6 | Feature Eng + LTR Training | 20 signals, XGBoost trained, NDCG lift | Train signal interaction model |
| 7 | SHAP + Ablation Study | 3 SHAP plots, ablation table | Signal engineering story |
| 8 | Domain-Adapted QA | Answer synthesis + rejection gate | Combine LLMs + search |
| 9 | Failure Analysis + Customer Cases | 5 visual failures + 3 complaint cases | Customer interaction thinking |
| 10 | README + Blog Post Outline | Repo polished, blog draft started | Productionization writeup |

## 15. How This Would Productionize 

### 15.1 Ingestion architecture

This project starts from a static, pre-downloaded corpus (`corpus.jsonl`). A real enterprise search system ingests continuously from live sources, and the ingestion design differs sharply by data type:

- **Unstructured content** (documents, wikis, chat: Confluence, Google Drive, Slack): pulled via per-source connectors, run through content extraction (strip HTML/markdown/layout), permission mapping (mirror each source's ACLs at user/group level), and activity-log mining (click/edit/view history, the real-world counterpart to this project's `simulated_ctr`/`simulated_dwell_time`). Initial sync is a deep historical crawl; ongoing sync switches to webhooks/incremental APIs to pick up edits within minutes, not on the next full reindex.
- **Structured data** (Salesforce, Databricks, Snowflake): not ingested/copied at all. A natural-language query gets translated on the fly into native query syntax (SOQL, SQL, JQL) against the live system, and the result is rendered directly. This avoids the staleness and duplication problem of copying relational data into a search index.

### 15.2 Security: permission-aware retrieval, not permission-filtered display

This project has no concept of per-user access control. Every query sees the full scoped corpus. A production system's defining security property is the reverse of "filter results after retrieval": **the query never touches documents a user can't see in the first place.** ACLs are ingested alongside content, user identity is resolved across systems (the same person's Slack ID, GitHub username, and SSO identity are one entity), and an early metadata filter restricts the searchable set to authorized doc_ids *before* the vector index or reranker ever runs, not as a post-hoc filter on results that were already scored. Retrofitting permissions after the fact (scoring first, filtering second) is both a security bug risk and a wasted-compute problem; production systems build the filter into the retrieval query itself.

### 15.3 Pipeline-stage productionization

- Retrieval layer would be replaced by connector-aware index spanning 100+ SaaS sources with per-document permissions (§15.2). Hybrid fusion remains valid; BM25 and dense signals still apply per-connector.
- Cross-encoder reranking would need accelerated hardware at enterprise query volume. The model architecture stays identical, the serving infrastructure changes (see §15.4).
- LTR signal set would extend with real click-through data, dwell time, and real Enterprise Graph signals (co-viewer, co-searcher, team affinity). This demo's simulated signals are the scaffolding for those, and Day 7's ablation study already showed the current simulated versions carry close to no real signal at this project's scale, which is itself informative about how much *real* behavioral data would be needed before these features earn their place in a production model.
- Personalization would run per-employee using a Personal Knowledge Graph (actual interaction history, not simulated role vectors). The role simulation here is the conceptual prototype. Day 4's finding (TF-IDF affinity showed real signal for HR, but collapsed for other roles on this general corpus) is direct evidence for why: role signal needs role-labeled interaction data, not just seed keywords against generic content.
- Domain adaptation would run per-customer at index time: extract vocabulary distribution, update system prompts, optionally fine-tune an adapter on customer query logs.
- Offline evaluation with TREC judgments shifts to online A/B experiments with interleaving. This harness is the pre-launch gate before any online experiment.

### 15.4 Production Considerations: latency, SLAs, CPU vs. GPU, and multi-modal input

This project runs entirely on CPU against a 150K-passage scoped corpus, with QA latency measured in the single-digit seconds (see `results/day8_findings.md` for P50/P95/P99). None of that generalizes directly to production. Worth being explicit about what changes and why, rather than assuming the same numbers would just scale down.

**Latency budgets, not single numbers.** A production system doesn't have one latency target. It has a *budget per pipeline stage*, with defined fallback behavior when a stage runs over: e.g. "if reranking exceeds 200ms, skip it and serve raw hybrid results" rather than blocking the whole response. This project's stages are evaluated independently (Days 1-8's separate eval scripts) but never composed into one latency-budgeted request path, which is the real production requirement. SLA/SLO design also means tracking P95/P99 as the primary signal, not average. A single slow outlier matters far more to user experience than the mean (this is exactly why Day 8's eval script now reports full percentiles rather than avg/max alone).

**CPU vs. GPU is a hard boundary, not a slider.** CPU inference here was the right call for a solo, budget-conscious project at 150K passages, not a preview of how this would run at scale. Dense encoding and cross-encoder reranking would move to GPU-backed serving (batched, fp16, likely behind a framework like Triton or a managed embedding/reranking endpoint) once query volume or corpus size grows past what CPU can sustain. This is roughly an order-of-magnitude throughput change, not a marginal one. CPU stays reasonable for offline batch jobs (nightly reindexing) even in a production system; it stops being viable for the synchronous hot query path well before enterprise scale.

**The QA layer's latency floor is largely provider round-trip, not local compute.** An external hosted LLM API call has an unavoidable floor of a few hundred ms to a couple seconds regardless of prompt size. A strict sub-second QA SLA typically isn't solved by a faster model alone. It requires making generation *non-blocking*: return ranked results immediately, stream the AI-generated answer in as a secondary UI element rather than gating the whole response on it. That's the actual pattern most production search+AI products use for this specific tension, and it's a UX/architecture decision as much as a model choice.

**Multi-modal input is a genuine scope gap, not a small addition.** This project only handles plain text passages. Real enterprise content is PDFs with embedded tables/images, slide decks, spreadsheets, code, and audio/video transcripts. Supporting that needs content-type routing at ingestion, OCR/layout-aware parsing for documents, and either true multi-modal embeddings or multiple specialized indexes fused at retrieval time. Worth stating plainly as unattempted scope rather than implying it's a minor follow-on; the retrieval/ranking/QA architecture in this project would still be the right shape underneath, but the ingestion and embedding layers would need real rework, not a parameter change.

## 16. Future Work

- **Signal Suggestion Engine**: auto-detect patterns in failure cases (e.g. "many failures caused by freshness") and suggest signal reweighting, the iterative search quality improvement loop enterprise runs internally.
- **Synthetic enterprise corpus**: multi-source dataset (Slack, Confluence, GitHub issues, Google Docs) with simulated permissions and team ownership.
- **Personalization v2**: collaborative filtering over user-document interaction history to replace role-affinity simulation with learned user preference vectors.
- **Fine-tuned domain encoder**: adapter fine-tuning of bge-base on domain-specific query-passage pairs to close the vocabulary gap further.
- **Latency-optimized QA backend**: evaluate a faster-inference hosted provider (e.g. LPU-based serving of an open 8B-class model) against the current OpenAI-based QA layer specifically on the rejection-gate calibration issue found in Day 8, not just on raw speed. A faster model that's *more* miscalibrated isn't a win. Not pursued in this project's timeline; noted here as the concrete next experiment rather than assumed to be an obvious improvement.

---
*SearchScope: PRD & Design Doc v2.0*
