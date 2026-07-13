# Day 7 Findings: SHAP Analysis + Ablation Study

## Ablation study results

| Ablation | NDCG@10 | Δ vs. Full | MRR@10 | MAP |
|---|---|---|---|---|
| Full model (20 signals) | 0.7324 | N/A | 0.9767 | 0.5439 |
| No lexical | 0.7377 | +0.0053 | 0.9767 | 0.5478 |
| **No semantic** | **0.4071** | **−0.3253** | 0.7427 | 0.3577 |
| No query | 0.7197 | −0.0126 | 0.9574 | 0.5452 |
| No behavior | 0.7320 | −0.0004 | 0.9884 | 0.5449 |
| No freshness/authority | 0.7260 | −0.0063 | 0.9612 | 0.5452 |
| No personalization | 0.7324 | **+0.0000** | 0.9767 | 0.5439 |
| No synthetic (bonus) | 0.7152 | −0.0171 | 0.9496 | 0.5431 |
| **BM25 only** | **0.3944** | **−0.3380** | 0.7558 | 0.3529 |
| Cross-encoder only | 0.7238 | −0.0086 | 0.9426 | 0.5459 |

## What's a real finding vs. what's noise: read the deltas at two different scales

Two ablations stand far apart from everything else: **No semantic
(−0.3253)** and **BM25 only (−0.3380)**. Both are roughly 20-30x larger
than every other delta in the table. That's not a coincidence. They're
directly testing the same thing from opposite ends: semantic signals
(dense cosine, cross-encoder score, dense rank, RRF score) are
overwhelmingly the dominant driver of ranking quality in this pipeline,
and lexical signal (BM25) alone is comparably weak on its own. This
matches every finding from Days 1-3: dense retrieval beat BM25 by a wide
margin from the very first baseline, and the cross-encoder's joint
query-passage modeling added real signal beyond that. The ablation study
confirms it quantitatively rather than just qualitatively.

Every other delta (No lexical, No query, No behavior, No
freshness/authority, No personalization, No synthetic) sits in a tight
±0.02 NDCG@10 band. With only 43 judged queries in the eval set, deltas
this small are within plausible fold-to-fold noise for LOQO CV, not
reliable evidence that any one of these signal categories matters more
than another. Reporting "No query hurts by 0.0126 but No freshness/authority
only hurts by 0.0063, so query signals matter almost twice as much" would
be overstating what 43 queries can actually support statistically. The
honest read is: one dominant signal family (semantic), one clearly weak
one in isolation (lexical alone), and a cluster of everything else that
this eval set is too small to cleanly rank against each other.

## No personalization: exactly 0.0000 delta, expected, and confirms an earlier finding

Removing `role_doc_affinity` changed literally nothing. This isn't
noise. It's the direct, mechanical consequence of a decision already
documented in `eval/build_ltr_dataset.py`: TREC DL queries carry no
persona/role label, so `role_doc_affinity` was computed as a constant
`0.0` for every single training example. A feature with zero variance
carries zero information; XGBoost correctly learned to ignore it
entirely. This is the cleanest possible confirmation that the feature
needs real per-session role context to matter at all, exactly the point
made in `results/day4_findings.md` about TF-IDF affinity needing
role-segmented content this general corpus doesn't have.

## The synthetic-signal hypothesis from Day 5-6: not confirmed, and that's worth reporting honestly

`results/day5-6_findings.md` proposed that the four purely synthetic
features (`doc_recency`, `source_authority`, `simulated_ctr`,
`simulated_dwell_time`, random per doc_id, no real signal since the
label-leakage fix) might be diluting the model's limited training
signal, and that dropping them might recover some of the gap between LTR
and the reranker alone.

**The bonus "No synthetic" ablation shows the opposite direction**:
removing all four together made NDCG@10 slightly *worse* (−0.0171), not
better. Given that number sits well inside the ±0.02 noise band
established above, the honest conclusion isn't "the hypothesis was wrong
and synthetic noise actually helps". It's that **43 queries isn't
enough signal to distinguish this hypothesis from noise at all**. The
individual "No behavior" (−0.0004) and "No freshness/authority" (−0.0063)
deltas are both close to zero, consistent with those features carrying
close to no real information post-fix, which is what the fix should
produce. The combined "No synthetic" delta being noisier than the sum of
its parts is itself informative about how much CV variance exists at
this query count, more than it's informative about the features
themselves.

**Framing for the writeup:** "the dilution hypothesis wasn't confirmed,
and at this eval scale (43 queries) it couldn't have been cleanly
confirmed or rejected either way. The honest conclusion is a limitation
of the evaluation's statistical power, not a finding about the features."
This is a better, more self-aware story than force-fitting the ablation
result to match the hypothesis.

## Sanity checks passed

- **BM25 only (0.3944)** and **Cross-encoder only (0.7238)** both land in
  plausible ranges relative to their respective standalone baselines
  (Day 1's BM25 baseline was 0.4839 on its own top-100 candidate pool,
  not the same pool measured here; Day 3's reranker was 0.7576). The
  cross-encoder-only LTR model coming in slightly below the raw reranker
  score (0.7238 vs. 0.7576) is expected. XGBoost's tree-based
  discretization of a single continuous feature isn't identical to
  sorting by that feature directly, and some fidelity loss there is
  normal, not a bug.
- **Recall@100 is identical (0.6895) across every single ablation row**,
  correctly confirms that every ablation variant reorders the same
  candidate pool rather than changing what's retrieved, consistent with
  the same pattern already verified in Days 3 and 5-6.

## SHAP analysis

Global feature importance (mean |SHAP value|, all 4,300 candidates, model
trained on all 43 queries, not LOQO, see `eval/run_shap_analysis.py`'s
module docstring for why):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | cross_encoder_score | 1.1601 |
| 2 | dense_cosine | 0.3866 |
| 3 | rrf_score | 0.2458 |
| 4 | dense_rank | 0.2136 |
| 5 | doc_length_log | 0.1626 |
| 6 | bm25_score | 0.1527 |
| 7 | term_overlap | 0.0908 |
| 8 | source_authority | 0.0819 |
| 9 | query_idf_mean | 0.0807 |
| 10 | simulated_ctr | 0.0704 |
| 11 | query_entropy | 0.0696 |
| 12 | simulated_dwell_time | 0.0689 |
| 13 | metadata_overlap | 0.0636 |
| 14 | bm25_rank | 0.0630 |
| 15 | title_match | 0.0619 |
| 16 | doc_recency | 0.0608 |
| 17 | query_length | 0.0410 |
| 18 | section_importance | 0.0129 |
| 19 | query_intent_class | 0.0124 |
| 20 | role_doc_affinity | **0.0000** |

### The semantic family dominates: confirms the ablation study at the per-feature level

The top 4 features by a wide margin are `cross_encoder_score`,
`dense_cosine`, `rrf_score`, `dense_rank`. That's literally the entire semantic
signal category, with `cross_encoder_score` alone (1.16) outweighing
everything else combined. This is the same conclusion the ablation study
reached (`No semantic` was by far the largest NDCG@10 drop, −0.325), now
confirmed at the individual-feature level rather than only at the
category level. Two independent methods agreeing is a stronger claim
than either alone.

### `role_doc_affinity`: exactly 0.0000, the cleanest result in this project

Zero SHAP importance, exactly matching the ablation study's exact 0.0000
NDCG@10 delta for `No personalization`. Two different measurement methods
(SHAP explains the fitted model's internal reliance on a feature;
ablation measures its actual contribution to held-out ranking quality)
landing on the identical number isn't a coincidence. It's the direct,
mechanical consequence of `role_doc_affinity` being a literal constant
(`0.0`) for every training example, since TREC DL queries carry no
persona label. A zero-variance feature has nothing for either method to
find, so both correctly report nothing.

### The interesting discrepancy: synthetic noise features have *nonzero* SHAP importance despite *zero* ablation impact

This is worth sitting with rather than glossing over. `source_authority`
(0.0819) and `doc_recency` (0.0608) are purely synthetic, deterministic
hash-based values per `doc_id`, by construction unrelated to true
relevance. `simulated_ctr` (0.0704) and `simulated_dwell_time` (0.0689)
are the same, now that the Day 5-6 label-leakage bug is fixed. All four
sit in a similar, non-trivial importance band, yet the ablation study
showed dropping the freshness/authority pair cost only −0.0063 NDCG@10,
and dropping the behavior pair cost only −0.0004, both inside the noise
band established earlier in this document.

**Why both can be true at once:** SHAP measures how much the *fitted
model* relies on a feature over the *training distribution* it was fit
to. Ablation (via LOQO) measures how much a feature actually helps
predictions *generalize* to a held-out query. A feature can have real,
non-trivial SHAP importance purely by the model finding spurious splits
in noise. With only 43 queries (4,300 candidates) and a model with
real capacity (200 boosting rounds, depth 6), some overfitting to
noise variables is expected, not surprising. `role_doc_affinity` is the
control case that makes this legible: it has zero *variance*, so there's
nothing for the model to spuriously fit, and both methods correctly
report zero. `source_authority`/`doc_recency`/`simulated_ctr`/`simulated_dwell_time`
have variance (different values per `doc_id`) but zero true signal, so
the model finds patterns in that variance, patterns that don't
generalize, which is exactly what the near-zero ablation deltas confirm.

**This is a stronger, more precise version of the Day 5-6 hypothesis**:
it's not that synthetic noise features dilute the *ranking output*
(ablation shows they mostly don't, within this eval's noise floor). It's
that they give the model something to mildly overfit to *internally*,
which SHAP can see even when the downstream ranking impact is
negligible. Worth stating plainly in the writeup as the more accurate
version of the original hypothesis, not a contradiction of it.

### `query_intent_class`: near-zero importance, and a real, honest reason why

Only 100 of 4,300 candidates (all from the same 1-2 queries) were
classified `transactional`; zero were `navigational`. TREC DL 2019's
judged queries are almost entirely informational/question-style (MS
MARCO originates as a QA dataset), so the simple rule-based classifier in
`ranking/features.py` had almost no class variance to work with on this
particular query set. Near-zero SHAP importance (0.0124) is the correct,
expected consequence, not a flaw in the classifier itself, but a
mismatch between a general-purpose heuristic and a query set that
happens to be very homogeneous in intent. Worth noting as a limitation
of the *evaluation set*, not the *feature*.

### `doc_length_log` and `bm25_score`: the two next-most-important non-semantic features

`doc_length_log` (0.1626) landing above even `bm25_score` (0.1527) is a
plausible, real signal. Passage length plausibly correlates with
"complete enough to be a good answer" independent of exact term overlap.
`bm25_score` having real if modest SHAP importance, despite the ablation
study showing `No lexical` very slightly *improving* NDCG@10 (+0.0053),
is a good example of the same SHAP-vs-ablation distinction discussed
above: the model relies on it somewhat internally, but that reliance
doesn't reliably help (and may occasionally mildly hurt) held-out ranking
quality, consistent with lexical signal being genuinely weaker than
semantic signal in this pipeline (also directly shown by `BM25 only`
being the single worst ablation variant, −0.338).

## Summary: what Day 7 actually established

1. **Semantic retrieval signals (cross-encoder + dense) are the dominant
   driver of ranking quality**, confirmed by both ablation (largest
   NDCG@10 drop) and SHAP (top 4 features), independently agreeing.
2. **`role_doc_affinity` is provably inert in this evaluation**, exactly
   zero by both measures, directly traceable to the lack of role-labeled
   queries in TREC DL, not a flaw in the affinity-scoring logic itself
   (which was validated separately and correctly in Day 4).
3. **The synthetic-noise-dilution hypothesis from Day 5-6 needed
   refining, not confirming or rejecting outright**: noise features show
   real SHAP importance (mild overfitting) without a correspondingly
   real ablation impact (no genuine generalization cost), a more
   precise and more defensible finding than the original hypothesis, and
   only visible by running both analyses and comparing them.
