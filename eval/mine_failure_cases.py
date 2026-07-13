"""
Day 9 — mine the real cached eval data for genuine failure cases.

This project's failure analysis needs to be evidence-backed, the same
standard applied to every other day's findings — not illustrative
examples invented for the writeup. Two of DESIGN.md's five planned
failure cases (vocabulary mismatch, and a workable "freshness/authority"
story) don't have a documented real case yet:

    - Vocabulary mismatch: real and findable, but nobody had gone looking
      for a specific example in the actual retrieval output yet. This
      script mines for one.
    - Freshness/authority failure: DESIGN.md originally envisioned "old
      doc ranked above new doc," but doc_recency/source_authority are
      synthetic hash-based values (DESIGN.md §5) — there are no real
      dates to demonstrate a genuine freshness bug with. Day 7's SHAP-vs-
      ablation finding (results/day7_findings.md) is the honest
      replacement: the model shows real internal reliance on these noise
      features (nonzero SHAP) without a corresponding real ranking
      benefit (near-zero ablation impact) — a documented, evidenced
      finding, just a different and more precise one than originally
      planned. No mining needed for this one; it's already fully written
      up.

The other three of the five failure cases (hybrid underperforming dense,
cross-encoder recovering the loss, personalization collapse) already
have complete, real evidence in results/day1-2_findings.md,
results/day3_findings.md, and results/day4_findings.md — this script
doesn't touch those.

Usage:
    python -m eval.mine_failure_cases --dataset trec-dl-2019 --sample 150000
"""
import argparse

import config
from data.corpus_lookup import lookup_passage_texts
from eval.build_ltr_dataset import assemble_ltr_examples_cached
from eval.harness import load_qrels
from eval.run_baseline_eval import load_queries


def score_vocabulary_mismatch(examples: list[dict], qrels_for_query: dict) -> dict | None:
    """
    Pure scoring logic, factored out for testing with synthetic data —
    see tests/test_mine_failure_cases.py.

    Looks for the classic vocabulary-mismatch signature: the actual
    relevant document (highest qrels grade among this query's candidates)
    ranked poorly by BM25 despite ranking well by dense retrieval, AND
    BM25's own top-1 pick has low term overlap with the relevant doc's
    vocabulary (i.e. BM25 wasn't just narrowly missing — it was pulled
    toward a lexically-similar but semantically wrong candidate, or found
    nothing lexically similar to the right answer at all).

    Returns None if this query has no judged-relevant candidate in the
    pool, or if BM25 actually did fine (nothing to report).
    """
    judged = [(ex, qrels_for_query.get(ex["doc_id"], 0)) for ex in examples]
    judged = [(ex, grade) for ex, grade in judged if grade > 0]
    if not judged:
        return None

    best_ex, best_grade = max(judged, key=lambda pair: pair[1])
    bm25_rank_of_best = best_ex["features"]["bm25_rank"]
    dense_rank_of_best = best_ex["features"]["dense_rank"]

    # "BM25 failed to find it" = either absent from BM25's ranked results
    # (rank 0.0, the FeatureBuilder default) or ranked much worse than dense.
    bm25_effectively_missing = bm25_rank_of_best == 0.0
    bm25_much_worse_than_dense = (
        dense_rank_of_best > 0
        and bm25_rank_of_best > 0
        and bm25_rank_of_best > dense_rank_of_best * 3
    )
    if not (bm25_effectively_missing or bm25_much_worse_than_dense):
        return None

    bm25_top1 = next((ex for ex in examples if ex["features"]["bm25_rank"] == 1), None)

    return {
        "relevant_doc_id": best_ex["doc_id"],
        "relevant_doc_grade": best_grade,
        "relevant_doc_bm25_rank": bm25_rank_of_best,
        "relevant_doc_dense_rank": dense_rank_of_best,
        "relevant_doc_term_overlap": best_ex["features"]["term_overlap"],
        "bm25_top1_doc_id": bm25_top1["doc_id"] if bm25_top1 else None,
        "bm25_top1_term_overlap": bm25_top1["features"]["term_overlap"] if bm25_top1 else None,
    }


def find_best_vocabulary_mismatch_case(examples_by_query: dict, qrels: dict) -> list[dict]:
    """Scores every query, returns candidates sorted by how stark the
    mismatch is (dense_rank - bm25_rank, treating 'missing' as rank 1000
    so it sorts to the top) — the best candidate for the writeup is
    whichever has the starkest, most legible gap."""
    candidates = []
    for qid, examples in examples_by_query.items():
        result = score_vocabulary_mismatch(examples, qrels.get(qid, {}))
        if result is None:
            continue
        bm25_rank = result["relevant_doc_bm25_rank"] or 1000
        dense_rank = result["relevant_doc_dense_rank"] or 1000
        result["query_id"] = qid
        result["gap"] = bm25_rank - dense_rank
        candidates.append(result)

    return sorted(candidates, key=lambda c: c["gap"], reverse=True)


def main(dataset_key: str, sample: int | None, rebuild_cache: bool = False, top_n: int = 5) -> None:
    print("Loading LTR dataset (cached after first Day 5-6/7/8 run)...")
    examples_by_query = assemble_ltr_examples_cached(dataset_key, sample, force_rebuild=rebuild_cache)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")
    queries = load_queries(dataset_key)

    print("Mining for vocabulary-mismatch cases (BM25 missed or badly ranked "
          "the true relevant doc vs. dense retrieval)...")
    candidates = find_best_vocabulary_mismatch_case(examples_by_query, qrels)
    print(f"  found {len(candidates)} candidate queries with a real BM25-vs-dense gap")

    if not candidates:
        print("  No qualifying cases found — try a different --sample or check qrels overlap.")
        return

    top_candidates = candidates[:top_n]
    doc_ids_needed = set()
    for c in top_candidates:
        doc_ids_needed.add(c["relevant_doc_id"])
        if c["bm25_top1_doc_id"]:
            doc_ids_needed.add(c["bm25_top1_doc_id"])
    passage_texts = lookup_passage_texts(dataset_key, doc_ids_needed)

    print(f"\nTop {len(top_candidates)} vocabulary-mismatch candidates (starkest gap first):\n")
    print("=" * 100)
    for c in top_candidates:
        qtext = queries.get(c["query_id"], "(query text unavailable)")
        print(f"Query [{c['query_id']}]: {qtext}")
        print(f"  Relevant doc (grade {c['relevant_doc_grade']}): [{c['relevant_doc_id']}] "
              f"BM25 rank={c['relevant_doc_bm25_rank'] or 'not in top-100'}, "
              f"dense rank={c['relevant_doc_dense_rank']}, "
              f"query-term-overlap={c['relevant_doc_term_overlap']:.2f}")
        print(f"    text: {passage_texts.get(c['relevant_doc_id'], '(not found)')[:250]}")
        if c["bm25_top1_doc_id"]:
            print(f"  BM25's actual top-1 pick: [{c['bm25_top1_doc_id']}] "
                  f"query-term-overlap={c['bm25_top1_term_overlap']:.2f}")
            print(f"    text: {passage_texts.get(c['bm25_top1_doc_id'], '(not found)')[:250]}")
        print("=" * 100)

    print(
        "\nPick the clearest case above for results/day9_failure_analysis.md — look for one "
        "where BM25's top-1 pick shares surface vocabulary with the query but is topically "
        "wrong, and the relevant doc uses different wording for the same concept. That's the "
        "legible 'vocabulary mismatch' story; a case where BM25 just found nothing at all is "
        "real but less visually clear for a writeup."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force rebuilding the LTR dataset instead of using the cached version.")
    parser.add_argument("--top-n", type=int, default=5, help="How many candidate cases to print.")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, rebuild_cache=args.rebuild_cache, top_n=args.top_n)
