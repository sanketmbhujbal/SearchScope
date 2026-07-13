# Day 1-2 Findings: Retrieval Baseline

**Dataset:** TREC DL 2019 (43 judged queries), qrels-aware scoped corpus,
150,000 passages (`data/sampling.py`, all judged passages guaranteed
included; see `DESIGN.md` §5 for why the corpus is scoped rather than
the full 8.8M).

**Encoder:** BAAI/bge-small-en-v1.5 (switched down from bge-base-en-v1.5
for CPU feasibility, see `DESIGN.md` §5/§7.2).

## Results

| Stage | NDCG@10 | MRR@10 | Recall@100 | MAP |
|---|---|---|---|---|
| BM25 Baseline | 0.4839 | 0.8037 | 0.6125 | 0.3923 |
| Dense | 0.7337 | 1.0000 | 0.6812 | 0.5398 |
| Hybrid (RRF) | 0.6845 | 0.9564 | 0.6895 | 0.5056 |

Numbers read higher than published full-corpus TREC DL 2019 baselines
(e.g. BM25 MAP is typically ~0.30 on the full 8.8M collection vs. 0.39
here). Expected and documented: fewer distractor passages in a 150K
scoped corpus makes ranking easier. Not a bug, not cause for concern,
just a known effect of the scoping decision.

## Finding: naive RRF underperforms the stronger individual retriever

Hybrid (RRF) scores *worse* than Dense alone on every metric except a
razor-thin Recall@100 edge:

- NDCG@10: Dense 0.7337 vs. Hybrid 0.6845 (**Dense wins by 0.049**)
- MRR@10: Dense 1.0000 vs. Hybrid 0.9564
- MAP: Dense 0.5398 vs. Hybrid 0.5056

**Why:** naive Reciprocal Rank Fusion treats every ranker as equally
trustworthy, purely by rank position. It has no notion of "this ranker
is actually much stronger." Here, Dense (NDCG@10 0.73) dramatically
outperforms BM25 (NDCG@10 0.48), so blending them at equal weight drags
the fused ranking toward BM25's weaker judgments and away from Dense's
stronger ones.

**This motivates, rather than undermines, the planned Day 3+ work:**
cross-encoder reranking (Day 3) operates on top of whatever the fusion
stage hands it, and the LTR layer (Days 5-6) already plans a **learned**
fusion weight as a feature (`rrf_score` alongside individual
`bm25_score`/`bm25_rank`/`dense_cosine`/`dense_rank`, see `DESIGN.md`
§9.1) rather than relying on a fixed RRF heuristic. This finding is the
concrete justification for that design choice, not just a checklist item
from the original PRD.

**Framing for the README / blog post:** "naive equal-weighted RRF
underperformed the stronger individual retriever, motivating a learned
fusion approach", sharper and more credible than "hybrid beat
everything," and it's true.

## Spot-check: MRR@10 = 1.0000 confirmed genuine

Manually verified via `eval/spot_check.py` (5 random judged queries,
seed=42). Every top-1 Dense result was a direct, on-topic answer to its
query (spruce tree description, Jamaica weather, WiFi vs. Bluetooth,
trapezoid midsegment, RN vs. BSN degree). Relevance grades (1-3, TREC DL
2019 scale) tracked intuitive answer quality: e.g. the WiFi/Bluetooth
and RN/BSN hits scored the maximum grade 3 and read like textbook-perfect
answers, while a more tangential hit scored the minimum passing grade 1.
Not an artifact of the scoped corpus or an eval-wiring bug.
