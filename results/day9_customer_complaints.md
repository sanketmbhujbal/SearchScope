# Day 9: Customer Complaint Cases

Same failure cases as `results/day9_failure_analysis.md`, reframed as
product thinking: what a user would actually say, and what signal
change addresses it. Five cases (DESIGN.md §11 asks for 3-5).

| # | Customer said | Expected | Actual | Root cause | Signal / fix |
|---|---|---|---|---|---|
| 1 | "I search deployment but get HR docs" | Release pipeline runbook | HR onboarding doc | Vocabulary mismatch (Failure Case 2, real example: "what are the three percenters?" returned an unrelated Jamaica weather page at BM25 rank 1, matching only on the token "percent") | Dense retrieval + cross-encoder reranking (Days 1-3) directly address this; query intent classification is a secondary signal |
| 2 | "Combining our two search systems made results worse, not better" | Hybrid search improving on either system alone | Hybrid actually scored below the stronger of the two individual systems | Naive equal-weight rank fusion (Failure Case 1), RRF has no notion of "this ranker is more trustworthy" | Cross-encoder reranking on top of fusion (already recovers this, Failure Case 3); longer-term, a learned fusion weight as an LTR feature rather than a fixed heuristic |
| 3 | "Policy search is useless for my team" | Team-relevant policies | Generic, mostly-identical policies regardless of team | Personalization signal has real content for some teams and none for others on this corpus (Failure Case 4) | Role-doc affinity as a learned LTR feature (already implemented as feature 20), but genuinely needs role-labeled interaction data to matter, not just seed keywords against generic content |
| 4 | "The assistant said it couldn't find an answer, but I know it's in there" | Direct answer with citation | Rejection despite supporting evidence being present in the passages it was given | Rejection-gate over-caution (QA Type B, Day 8), 3 of 7 false rejections were clear misses where a directly quotable answer was in context | Candidate follow-up: a second-pass prompt when high-relevance context is present but the model still rejects, or lowering the rejection threshold, not attempted in this project's timeline, noted as a concrete next experiment |
| 5 | "Why does the system trust some random-looking factors so much?" | Ranking driven only by genuinely relevant signals | SHAP shows the model leaning on synthetic recency/authority features with no real informational content | Model overfitting to noise-variance features at a small training-data scale (Failure Case 5) | More training data would shrink this naturally; alternatively, drop synthetic features with near-zero ablation impact from the production feature set entirely once real behavioral data isn't available to replace them |

## Notes on framing

Case 2 (hybrid-worse-than-either-alone) is deliberately included even
though it's an unusual complaint to picture a real user voicing directly.
It's more realistic as an internal engineering-team complaint ("why did
combining our systems make this worse?") than an end-user one, which is
worth being upfront about rather than forcing an artificial end-user
quote onto an internal-facing finding.

Case 4 is the only case in this table without a proposed fix that's
already implemented and measured elsewhere in this project, flagged
explicitly as "not attempted" rather than implied to be solved, matching
the standard set by every other honestly-scoped decision in this
project (DESIGN.md §5, §16).
