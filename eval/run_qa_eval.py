"""
Day 8 — domain-adapted QA + rejection gate evaluation (DESIGN.md §10.3).

Builds two test sets from the 43 TREC DL 2019 judged queries:

    "Answerable" set: each query paired with its own top-K passages,
        ranked by cross_encoder_score (reusing the cached Day 5-6/7 LTR
        dataset — no new retrieval needed).
    "Unanswerable" set: each query paired with a DIFFERENT, randomly
        selected query's top-K passages (seeded, reproducible) — context
        that's guaranteed not to answer the question. This project's 43
        judged queries are all genuinely answerable, so there's no
        natural "unanswerable" set to draw from the way DESIGN.md's "50
        manually identified unanswerable queries" envisioned; mismatched
        context is a clean, reproducible, explainable substitute that
        tests the same thing the rejection gate is actually for — does
        the model correctly refuse when the evidence doesn't support an
        answer, rather than fabricating one anyway.

Metrics:
    Answer Rejection Rate — fully automated: % of the unanswerable set
        where the model correctly rejected. Computed here directly.
    Answer Supported Rate — DESIGN.md specifies this as a human
        spot-check (50 samples), and that's not something a script can
        honestly automate (grading "is this answer actually supported by
        the passages" requires reading both). This script produces a
        formatted review file instead of fabricating an automated proxy
        number for something the methodology says needs a human.
    Citation hygiene — fully automated proxy worth reporting alongside
        the human-reviewed number: % of non-rejected answerable-set
        answers with zero hallucinated citations (citing a doc_id that
        wasn't actually provided). This doesn't confirm an answer is
        *correct*, but it's a real, cheap signal that the grounding
        constraint is being respected mechanically.

Usage:
    python -m eval.run_qa_eval --dataset trec-dl-2019 --sample 150000

Requires OPENAI_API_KEY to be set in the environment.
"""
import argparse
import os
import random

import config
from data.preprocess import load_corpus
from data.sampling import build_scoped_corpus
from eval.build_ltr_dataset import assemble_ltr_examples_cached
from eval.run_baseline_eval import load_queries
from qa.grounded_qa import GroundedQA, extract_corpus_vocabulary


def compute_latency_percentiles(latencies: list[float]) -> dict:
    """
    P50/P95/P99 latency, not just avg/max. Avg hides tail behavior
    entirely, and max is a single outlier that could be one fluke call —
    percentiles are the standard way to talk about latency against an
    SLA, since "how bad does it get for the worst 5% of requests" is
    usually the number that actually matters for a latency target.

    Uses a simple nearest-rank method (no interpolation) — fine at this
    dataset size (dozens to low hundreds of calls); a production
    monitoring system would use a proper streaming percentile estimator
    (e.g. t-digest) rather than sorting an in-memory list, but that's not
    needed at this scale.
    """
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "max": 0.0}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return sorted_lat[idx]

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "avg": sum(latencies) / n,
        "max": max(latencies),
    }


def top_k_passages(examples: list[dict], k: int) -> list[dict]:
    """Sorts a query's cached candidates by cross_encoder_score and
    returns the top-k as {"doc_id", "text"} passages for QA context."""
    ranked = sorted(examples, key=lambda ex: ex["features"]["cross_encoder_score"], reverse=True)
    return [{"doc_id": ex["doc_id"], "text": ex["doc_text"]} for ex in ranked[:k] if "doc_text" in ex]


def build_test_sets(examples_by_query: dict, queries: dict, top_k: int, seed: int) -> tuple[list, list]:
    """
    Returns (answerable_cases, unanswerable_cases), each a list of
    {"query_id", "query_text", "passages"} dicts.

    NOTE: examples_by_query entries from eval/build_ltr_dataset.py don't
    currently carry doc_text in the stored example dict (only doc_id,
    features, label) — see the caller for how this is resolved via a
    fresh doc_id -> text lookup, since the cached dataset was built for
    feature vectors, not for reproducing passage text downstream.
    """
    qids = [q for q, ex in examples_by_query.items() if ex]
    rng = random.Random(seed)

    answerable_cases = []
    for qid in qids:
        passages = top_k_passages(examples_by_query[qid], top_k)
        answerable_cases.append({"query_id": qid, "query_text": queries[qid], "passages": passages})

    unanswerable_cases = []
    for qid in qids:
        other_qids = [q for q in qids if q != qid]
        mismatched_qid = rng.choice(other_qids)
        passages = top_k_passages(examples_by_query[mismatched_qid], top_k)
        unanswerable_cases.append({
            "query_id": qid,
            "query_text": queries[qid],
            "passages": passages,
            "context_from_query_id": mismatched_qid,  # kept for transparency in the output
        })

    return answerable_cases, unanswerable_cases


def main(dataset_key: str, sample: int | None, rebuild_cache: bool = False, limit: int | None = None) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before running this script:\n"
            "  Windows (PowerShell): $env:OPENAI_API_KEY = \"sk-...\"\n"
            "  macOS/Linux: export OPENAI_API_KEY=\"sk-...\""
        )

    print("Loading LTR dataset (cached after first Day 5-6/7 run)...")
    examples_by_query = assemble_ltr_examples_cached(dataset_key, sample, force_rebuild=rebuild_cache)

    # examples_by_query entries carry doc_id/features/label but not
    # doc_text (that's only held transiently during assembly) — rebuild
    # a doc_id -> text lookup for the doc_ids we'll actually use as QA
    # context, same efficient single-pass pattern used elsewhere.
    from data.corpus_lookup import lookup_passage_texts

    needed_doc_ids = {ex["doc_id"] for examples in examples_by_query.values() for ex in examples}
    print(f"Looking up passage text for {len(needed_doc_ids)} candidate passages...")
    passage_texts = lookup_passage_texts(dataset_key, needed_doc_ids)
    for examples in examples_by_query.values():
        for ex in examples:
            ex["doc_text"] = passage_texts.get(ex["doc_id"], "")

    queries = load_queries(dataset_key)

    print("Fitting corpus vocabulary for the domain-adapted system prompt...")
    corpus_docs = load_corpus(dataset_key)
    if sample:
        qrels_path = config.RAW_DATA_DIR / dataset_key / "qrels.jsonl"
        corpus_docs = build_scoped_corpus(corpus_docs, qrels_path, target_size=sample, seed=config.RANDOM_SEED)
    vocabulary = extract_corpus_vocabulary([d["text"] for d in corpus_docs], top_k=config.QA_VOCABULARY_SIZE)
    print(f"  vocabulary: {vocabulary}")

    print("Building answerable + unanswerable test sets...")
    answerable_cases, unanswerable_cases = build_test_sets(
        examples_by_query, queries, top_k=config.QA_CONTEXT_TOP_K, seed=config.RANDOM_SEED
    )
    if limit:
        answerable_cases = answerable_cases[:limit]
        unanswerable_cases = unanswerable_cases[:limit]
    print(f"  {len(answerable_cases)} answerable cases, {len(unanswerable_cases)} unanswerable cases")

    qa = GroundedQA()
    latencies = []

    print(f"Running QA on {len(answerable_cases)} answerable cases...")
    answerable_results = []
    for i, case in enumerate(answerable_cases):
        result = qa.answer(case["query_text"], case["passages"], vocabulary)
        latencies.append(result["latency_seconds"])
        answerable_results.append({**case, **result})
        if (i + 1) % 10 == 0 or (i + 1) == len(answerable_cases):
            print(f"  {i + 1}/{len(answerable_cases)} done")

    print(f"Running QA on {len(unanswerable_cases)} unanswerable (mismatched-context) cases...")
    unanswerable_results = []
    for i, case in enumerate(unanswerable_cases):
        result = qa.answer(case["query_text"], case["passages"], vocabulary)
        latencies.append(result["latency_seconds"])
        unanswerable_results.append({**case, **result})
        if (i + 1) % 10 == 0 or (i + 1) == len(unanswerable_cases):
            print(f"  {i + 1}/{len(unanswerable_cases)} done")

    # --- Metrics ---
    n_rejected_unanswerable = sum(1 for r in unanswerable_results if r["rejected"])
    rejection_rate = n_rejected_unanswerable / len(unanswerable_results) if unanswerable_results else 0.0

    non_rejected_answerable = [r for r in answerable_results if not r["rejected"]]
    n_clean_citations = sum(1 for r in non_rejected_answerable if not r["hallucinated_doc_ids"])
    citation_hygiene_rate = n_clean_citations / len(non_rejected_answerable) if non_rejected_answerable else 0.0
    n_answerable_incorrectly_rejected = sum(1 for r in answerable_results if r["rejected"])

    latency_stats = compute_latency_percentiles(latencies)

    print("\n=== Day 8 QA Metrics ===")
    print(f"Answer Rejection Rate (unanswerable set):  {rejection_rate:.1%} "
          f"({n_rejected_unanswerable}/{len(unanswerable_results)})")
    print(f"Answerable queries incorrectly rejected:    {n_answerable_incorrectly_rejected}/{len(answerable_results)} "
          f"(false rejections — worth checking manually if nonzero)")
    print(f"Citation hygiene rate (no hallucinated cites): {citation_hygiene_rate:.1%} "
          f"({n_clean_citations}/{len(non_rejected_answerable)})")
    print(f"Latency: p50 {latency_stats['p50']:.2f}s, p95 {latency_stats['p95']:.2f}s, "
          f"p99 {latency_stats['p99']:.2f}s, avg {latency_stats['avg']:.2f}s, max {latency_stats['max']:.2f}s "
          f"(target: <{config.QA_MAX_LATENCY_SECONDS}s, DESIGN.md §10.3)")
    print("\nAnswer Supported Rate: NOT computed automatically — see the review file below. "
          "DESIGN.md specifies this as a human spot-check; automating a proxy for "
          "'is this answer actually correct' would be a weaker, less honest number "
          "than just doing the spot-check.")

    # --- Save results + human-review artifact ---
    import json

    suffix = f"_sample_{sample}" if sample else ""
    metrics_path = config.RESULTS_DIR / f"{dataset_key}_qa_metrics{suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "answer_rejection_rate": rejection_rate,
            "answerable_incorrectly_rejected": n_answerable_incorrectly_rejected,
            "citation_hygiene_rate": citation_hygiene_rate,
            "latency_seconds": latency_stats,
            "n_answerable": len(answerable_results),
            "n_unanswerable": len(unanswerable_results),
        }, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    review_path = config.RESULTS_DIR / f"day8_qa_review{suffix}.md"
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("# Day 8 QA — Human Review\n\n")
        f.write(
            "For each answerable-set case: read the passages and the answer, and judge "
            "whether the answer is actually supported by them. This is the Answer Supported "
            "Rate DESIGN.md §10.3 specifies as a human spot-check — tally your own yes/no "
            "count after reviewing.\n\n"
        )
        for r in answerable_results:
            f.write(f"## Query [{r['query_id']}]: {r['query_text']}\n\n")
            f.write("**Passages:**\n\n")
            for p in r["passages"]:
                f.write(f"- [{p['doc_id']}] {p['text'][:200]}\n")
            f.write(f"\n**Answer:** {r['answer']}\n\n")
            f.write(f"**Rejected:** {r['rejected']} | **Cited:** {r['cited_doc_ids']} | "
                    f"**Hallucinated citations:** {r['hallucinated_doc_ids']}\n\n")
            f.write("**Supported? (fill in):** \n\n---\n\n")
    print(f"Saved human-review artifact -> {review_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(config.EVAL_DATASETS), default="trec-dl-2019")
    parser.add_argument("--sample", type=int, default=None, help="Must match data/preprocess.py's --sample value.")
    parser.add_argument("--rebuild-cache", action="store_true",
                         help="Force rebuilding the LTR dataset instead of using the cached version.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N queries per test set (useful for a cheap smoke test "
                              "before spending API calls on the full 43-query set).")
    args = parser.parse_args()
    main(args.dataset, sample=args.sample, rebuild_cache=args.rebuild_cache, limit=args.limit)
