# Day 3 Findings: Cross-Encoder Reranking

## A metric-depth artifact, caught before it became a false conclusion

The first `eval/run_reranked_eval.py` run reranked at `RERANK_TOP_K` depth
(20, the depth DESIGN.md specifies for what actually feeds the Day 5-6
LTR layer downstream) and produced this table:

| Stage | NDCG@10 | MRR@10 | Recall@100 | MAP |
|---|---|---|---|---|
| BM25 Baseline | 0.4839 | 0.8037 | 0.6125 | 0.3923 |
| Dense | 0.7337 | 1.0000 | 0.6812 | 0.5398 |
| Hybrid (RRF) | 0.6845 | 0.9564 | 0.6895 | 0.5056 |
| + Cross-Encoder Reranker | 0.7576 | 0.9457 | **0.3228** | **0.2855** |

At a glance this looks like the reranker improved top-10 quality
(NDCG@10, MRR@10 both fine or better) while somehow collapsing Recall@100
and MAP. That split is the tell that something's wrong with the
*comparison*, not the *reranker*: Recall@100 and MAP need up to 100
ranked docs per query to score meaningfully. A run truncated to 20 docs
can't credit any relevant passage sitting at rank 21-100 in Hybrid's
list, not because the reranker lost it, but because it was never asked
to return that many results in the first place. NDCG@10/MRR@10 only look
at the top 10, so the truncation is invisible to them, which is exactly
what made the artifact easy to mistake for a real finding.

**Fix:** `eval/run_reranked_eval.py` now reranks at full `HYBRID_TOP_K`
depth (100) for this comparison table specifically, so all four stages
are evaluated on equal footing. The smaller `RERANK_TOP_K` (20) remains
the intended depth for what actually feeds the Day 5-6 LTR layer. That's
a separate, deliberate pipeline-efficiency decision (LTR doesn't need
100 candidates, it needs a strong top-20), not something the eval
harness should silently inherit.

## Corrected results

| Stage | NDCG@10 | MRR@10 | Recall@100 | MAP |
|---|---|---|---|---|
| BM25 Baseline | 0.4839 | 0.8037 | 0.6125 | 0.3923 |
| Dense | 0.7337 | 1.0000 | 0.6812 | 0.5398 |
| Hybrid (RRF) | 0.6845 | 0.9564 | 0.6895 | 0.5056 |
| + Cross-Encoder Reranker | **0.7576** | 0.9457 | 0.6895 | **0.5607** |

**Recall@100 sanity check:** Reranker's Recall@100 (0.6895) is *exactly*
equal to Hybrid's. This is expected and confirms the pipeline is wired
correctly, not a coincidence. Reranking only reorders Hybrid's existing
top-100 candidate set, it can't retrieve documents that weren't already
in it. Recall@100 only asks "is a relevant doc anywhere in the top-100,"
not caring about rank, so an identical candidate set guarantees an
identical score regardless of how the reranker reorders it.

**The actual finding:** the cross-encoder recovers what naive RRF gave
away (the Day 1-2 finding: Hybrid underperformed Dense alone) *and*
pushes past Dense alone on both NDCG@10 (0.7576 vs. 0.7337) and MAP
(0.5607 vs. 0.5398). That's evidence the cross-encoder's joint
query-passage modeling, seeing query and doc together, rather than
comparing independently-encoded vectors, is adding real signal beyond
what bi-encoder similarity plus rank fusion captures on its own.

MRR@10 dipped very slightly versus Hybrid (0.9457 vs. 0.9564), small
enough to note honestly rather than explain away; the reranker isn't
strictly better on every single metric, just clearly better on the two
that matter most for ranking quality (NDCG@10, MAP).

**Framing for the README / blog post:** "cross-encoder reranking not only
recovered the ranking quality naive fusion gave away, it exceeded the
best individual retriever, validating the retrieve, fuse, then rerank
architecture as more than the sum of its parts."

## Why this is worth keeping in the writeup

"I caught a metric-depth mismatch between my reranking cutoff and a
depth-sensitive eval metric, before it became a false 'reranking hurts
retrieval' conclusion" is a stronger, more specific engineering story
than a clean run would have been. It's the same kind of habit as the
Day 1-2 RRF finding: read the metric split, not just the headline number,
before drawing a conclusion.
