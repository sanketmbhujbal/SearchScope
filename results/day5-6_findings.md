# Day 5-6 Findings: Feature Engineering + XGBoost LTR

## A real bug, caught by the numbers being implausible

The first `eval/run_ltr_eval.py` run produced:

| Stage | NDCG@10 | MRR@10 | Recall@100 | MAP |
|---|---|---|---|---|
| + Cross-Encoder Reranker (sanity check) | 0.7576 | 0.9457 | 0.6895 | 0.5607 |
| + LTR (LOQO, 20 signals) | **0.9731** | **1.0000** | 0.6895 | 0.6747 |

The sanity-check `+Reranker` row matched Day 3's recorded number exactly.
This confirms the candidate pool was consistent between scripts, which
was the point of that check. But `+LTR` jumping to NDCG@10=0.97 with a
perfect MRR@10=1.0000, trained on leave-one-query-out CV across only ~40
queries, is not a plausible result. Real LTR gains from that little
training signal should be modest, not near-perfect.

## Root cause: label leakage in the synthetic behavior features

`simulated_ctr` and `simulated_dwell_time` were originally implemented as
noisy functions of `relevance_grade`, the actual TREC judgment for that
(query, doc) pair. The intent was reasonable (weak-supervision-style
proxies for "would a user have engaged with this," standing in for real
click logs that don't exist in this project's scope), and it was even
flagged in advance in the original feature docstring as something to
watch in Day 7's SHAP analysis.

But the actual bug was more fundamental than "this feature will look
suspiciously important." Leave-one-query-out CV correctly held the
*model* out of training on each held-out query, but the *features* for
that held-out query's own candidates were still computed using their
true relevance labels, because `assemble_ltr_examples` passed
`relevance_grade=label` into feature construction for every candidate,
train or test. That handed the model a near-direct, lightly-noised
readout of the answer at prediction time, not just training time. LOQO's
train/test split was working correctly; the feature pipeline was quietly
bypassing it.

## The fix

Real CTR/dwell-time signals are historical, document-level aggregates
from past behavior across *many* queries, never derived from the
specific relevance label of the query currently being scored.
`simulated_ctr(doc_id)` and `simulated_dwell_time(doc_id)` are now
deterministic functions of `doc_id` alone (the same pattern already used
correctly by `doc_recency`/`source_authority`), with no path for a label
to reach them. `FeatureBuilder.build()` no longer accepts a
`relevance_grade` parameter at all, removing the parameter, not just
changing how it's used, so a future edit can't quietly reintroduce the
leak by accident.

Four regression tests now exist specifically to catch this class of bug
if it recurs (`tests/test_features.py`): two check the function
signatures directly refuse a label-shaped argument, one checks
`FeatureBuilder.build()` output is identical for the same candidate
regardless of context, and one confirms `simulated_ctr`/`simulated_dwell_time`
values are genuinely independent of relevance.

## Corrected results

| Stage | NDCG@10 | MRR@10 | Recall@100 | MAP |
|---|---|---|---|---|
| + Cross-Encoder Reranker (sanity check) | 0.7576 | 0.9457 | 0.6895 | 0.5607 |
| + LTR (LOQO, 20 signals) | 0.7324 | **0.9767** | 0.6895 | 0.5439 |

**Recall@100 sanity check (again):** identical to the reranker row, as
expected. LTR only reorders the same 100-candidate pool, it doesn't
retrieve new documents. Same confirmation pattern as Day 3.

**The actual result:** LTR is essentially flat-to-slightly-worse than the
cross-encoder reranker alone on NDCG@10 (-0.025) and MAP (-0.017), with a
modest MRR@10 improvement (+0.031). This is the honest, expected outcome
for training on ~40 queries' worth of candidates rather than MS MARCO's
full training-triples set (documented in `DESIGN.md` §9.2), not a
disappointing result, and importantly, not another bug. The magnitude is
plausible this time, unlike the leakage-driven 0.97 from before.

**A concrete hypothesis this sets up for Day 7's ablation study:**
`cross_encoder_score` is itself one of the 20 input features. In
principle, XGBoost should be able to at least reproduce reranker-level
ranking by learning to weight that one feature heavily, then improve on
it using the other 19. That it doesn't quite manage to *improve* on it
suggests the model may be spending some of its limited training signal
partially discounting noise from the synthetic features (`doc_recency`,
`source_authority`, and the now-correctly-independent
`simulated_ctr`/`simulated_dwell_time` are literally random per doc_id)
rather than purely sharpening the real signals. Day 7's ablation study
(dropping the synthetic-signal group specifically) is a direct test of
this hypothesis: if removing the synthetic features recovers or exceeds
the reranker's NDCG@10, that confirms noise dilution from a small
training set as the mechanism, rather than e.g. `role_doc_affinity`
being uninformative (already known from Day 4) or a genuine ceiling in
the real signals themselves.

## Why this is worth keeping in the writeup

This is a case where an unglamorous result (LTR roughly ties the
reranker, doesn't clearly beat it) is more useful and more credible than
a flattering one. It's directly explained by the small-training-set
caveat that was documented *before* running the experiment, and it
generates a specific, testable hypothesis for the next stage of work.
"My LTR model matched the reranker rather than beating it, and I can
explain exactly why given the training data size, and here's the
ablation that tests that explanation" is a stronger technical story than
an unexplained win would have been.

## Why this is worth keeping in the writeup

This is the most substantial finding in the project so far, more so than
the RRF or metric-depth findings: a two-line change (a features function
taking a label as input) produced a result that looked like a genuine
breakthrough and was actually a data leak. Catching it required treating
an unusually good result with the same scrutiny as an unusually bad
one, noticing that NDCG@10=0.97 and MRR@10=1.0000 from ~40 training
queries was too good to be true, then tracing the mechanism precisely
rather than accepting the number. That instinct, leakage is often
disguised as success, not failure, is exactly the kind of judgment a
search quality or ML engineering role is testing for.
