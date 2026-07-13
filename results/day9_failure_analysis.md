# Day 9: Failure Analysis (5 Fully Worked Cases)

Every case below cites real, already-run evidence from this project.
no fabricated examples. All five cases are complete, including Case 2
(vocabulary mismatch), mined directly from this project's own persisted
retrieval indexes via `eval/mine_failure_cases.py` rather than invented.

---

## Case 1: Naive hybrid fusion underperforms the stronger individual retriever

**Stage:** Retrieval (Hybrid RRF, Day 1-2)

**What happened:** Reciprocal Rank Fusion combined BM25 and Dense
retrieval at equal weight. Because Dense (NDCG@10 0.7337) dramatically
outperformed BM25 (NDCG@10 0.4839) on this corpus, the equal-weighted
fusion actively pulled the ranking *toward* BM25's weaker judgments,
Hybrid (NDCG@10 0.6845) scored *worse* than Dense alone on every metric
except a razor-thin Recall@100 edge.

**Why it happened:** RRF is rank-position-only. It has no mechanism to
recognize "this ranker is systematically stronger than that one." It
treats a rank-1 BM25 result and a rank-1 Dense result as equally
trustworthy, which is false whenever the two rankers have very different
underlying quality.

**Signal insight / fix:** This is the direct justification for treating
fusion weight as a *learned* signal (`rrf_score` alongside individual
`bm25_score`/`bm25_rank`/`dense_cosine`/`dense_rank`, DESIGN.md §9.1)
rather than a fixed heuristic. Day 3's cross-encoder reranking already
recovers this loss and exceeds Dense alone (see Case 3).

**Evidence:** `results/day1-2_findings.md`

---

## Case 2: Vocabulary mismatch: BM25 misses a real answer that dense retrieval finds

**Stage:** Retrieval (BM25, lexical-only)

**What happened:**
- Query: `"what are the three percenters?"`
- Relevant document (TREC grade 3, `doc_id` 3423067): *"A loose affiliation
  of like minded Americans who vow to refuse to comply with laws that
  violate the second amendment right to keep (own) and bear (carry)
  firearms. The name 3 percenter comes from the fact that only 3 percent
  of colonial British subjects..."* BM25 rank: **84**, dense rank: **2**.
- BM25's actual top-1 pick (`doc_id` 6630430): *"Microsoft Project
  contains three measures of %Complete...Jamaica weather, Falmouth
  Jamaica weather, Ocho Rios Jamaica weather..."*, a Jamaica weather
  page and Microsoft Project documentation, matching the query on
  essentially one token ("percent"/"%") and nothing else.

**Why it happens:** the query spells out "three percenters," while the
relevant document uses the numeral form "3 percenter", a trivial
surface variation to a human reader, but BM25's exact-term matching
treats "three" and "3" as unrelated tokens. With the correct lexical
match unavailable, BM25 fell back to whatever else in a 150K-passage
corpus happened to contain the word "percent," which produced a result
with zero topical relevance to the query. Dense retrieval, working on
semantic meaning rather than surface tokens, found the actual relevant
document easily (rank 2).

**Signal insight / fix:** this is about as clean an illustration as
exists of why BM25 alone (NDCG@10 0.48) trails Dense (0.73) by such a
wide margin in this project's Day 1-2 baseline, not a subtle case, an
almost absurd one, which makes it a good illustration precisely because
the failure is instantly legible without domain expertise.

**Runner-up case, also real and worth keeping in reserve:** query
`"medicare's definition of mechanical ventilation"`, BM25 rank 39 vs.
dense rank 2, BM25's top-1 pick discusses Medicare billing/payment
claims that happen to mention "mechanical ventilation" in passing,
while the actually relevant document is a direct definitional sentence
that shares fewer exact query terms. A subtler version of the same
mechanism, worth using if a second vocabulary-mismatch example is
ever needed (e.g. for the blog post's illustration set).

**Evidence:** `eval/mine_failure_cases.py` output, run against the real
150K scoped corpus + persisted indexes, 2026-07 (see
`tests/test_mine_failure_cases.py` for the mining logic's test coverage)

---

## Case 3: Cross-encoder reranking recovers what naive fusion gave away, and exceeds the best individual retriever

**Stage:** Reranking (Cross-Encoder, Day 3)

**What happened:** Following directly from Case 1, reranking Hybrid's
top-100 candidates with `bge-reranker-base` didn't just recover Hybrid's
lost ground. It pushed past Dense alone on both NDCG@10 (0.7576 vs.
0.7337) and MAP (0.5607 vs. 0.5398).

**Why it happened:** The cross-encoder jointly encodes query and passage
together (rather than comparing independently-encoded vectors like the
bi-encoder in Dense retrieval), capturing fine-grained query-passage
interaction signal that neither BM25's lexical overlap nor Dense's
fixed-vector similarity can see on their own.

**Signal insight:** This validates the retrieve → fuse → rerank
architecture as genuinely more than the sum of its parts, but only once
each stage's actual contribution is measured honestly (Case 1 wouldn't
have been visible without checking Hybrid against Dense specifically,
not just against BM25).

**Evidence:** `results/day3_findings.md`

---

## Case 4: Role-based personalization shows real signal for one role, collapses for others

**Stage:** Personalization (Day 4)

**What happened:** For the query "policy," HR's top-5 results were
genuinely distinct and plausible (a Group Policy/Active Directory doc, a
concussion-management policy, a benefits-coordination policy). Engineer,
Sales, and Legal's top-5s largely collapsed into the same shared list,
only rank 1 differed between them, and even that was questionable
(Engineer's #1 result was an HIV/AIDS workplace policy document, with no
real connection to the engineering seed terms).

**Why it happened:** TF-IDF cosine similarity against a short seed-term
list is a blunt signal on a general web corpus (MS MARCO) that was never
built with role-segmented enterprise content. HR's seed list happened to
overlap unusually well with this particular corpus and with the query
itself ("policy" is a literal HR seed term), a seed-vocabulary-richness
artifact more than genuine role modeling. "Group Policy" scoring the
single highest affinity of the whole demo (0.804, for HR) is TF-IDF
rewarding shared surface words ("policy," "group"), not real
understanding.

**Signal insight / fix:** Directly explains why Day 7's ablation study
found `role_doc_affinity` had *exactly* 0.0000 NDCG@10 impact. TREC DL
queries carry no persona label, so the feature was a constant 0.0 for
every training example, and both SHAP and ablation correctly found
nothing there. A production system would need real role-segmented
interaction data (Glean's Personal Knowledge Graph, DESIGN.md §15), not
seed keywords against generic content.

**Evidence:** `results/day4_findings.md`, `results/day7_findings.md`

---

## Case 5: The LTR model shows real internal reliance on synthetic noise features, without a real ranking benefit

**Stage:** Learning-to-Rank / Feature Importance (Day 7)

**What happened:** SHAP analysis showed `source_authority` (0.0819) and
`doc_recency` (0.0608), both purely synthetic, deterministic hash values
per doc_id with zero relationship to true relevance, carrying
non-trivial importance in the fitted model. But the ablation study
showed dropping the freshness/authority feature pair cost only −0.0063
NDCG@10, well inside the noise band established by every other non-dominant
ablation category.

**Why it happened:** SHAP measures how much the *fitted model* relies on
a feature over its *training* distribution; ablation measures whether
that reliance actually helps predictions *generalize* to a held-out
query. With only ~40 queries of training signal and a model with real
capacity, some mild overfitting to noise variables is expected. The
model finds spurious splits in features that have variance (different
values per doc_id) but no true signal. `role_doc_affinity` (Case 4) is
the clean control: it has *zero variance* (a literal constant), so
there's nothing to even spuriously fit, and both SHAP and ablation
correctly report exactly zero for it.

**Signal insight:** This is a more precise, more defensible version of a
"freshness/authority is a fake feature" story, not "the model is wrong
to use it," but "the model mildly overfits to it internally without that
overfitting costing real ranking quality at this data scale," which is a
genuinely different and more interesting claim, only visible by running
both SHAP and ablation and comparing them rather than trusting either
alone.

**Evidence:** `results/day7_findings.md`
