"""
Assembles the labeled LTR training/eval dataset: for every judged query,
gathers its hybrid top-100 candidates with all upstream retrieval signals
(BM25/dense/RRF/cross-encoder) attached, builds the full 20-signal
feature vector for each, and labels each candidate from qrels.

See eval/run_ltr_eval.py for why this uses leave-one-query-out CV on the
43 TREC DL 2019 judged queries rather than MS MARCO's separate training-
triples set (hundreds of thousands of examples — out of scope; see
DESIGN.md §5).

CAVEAT — unjudged candidates default to label 0: TREC DL judges only the
docs assessors actually reviewed; a candidate absent from qrels for a
query is "never assessed," not necessarily "confirmed irrelevant." This
is standard LTR practice (excluding unjudged candidates entirely would
throw away most of the negative examples a ranker needs to learn from),
but it's worth stating plainly rather than letting it pass as an
unexamined assumption.
"""
import config
from data.corpus_lookup import lookup_passage_texts
from data.preprocess import load_corpus
from data.sampling import build_scoped_corpus
from eval.harness import load_qrels
from eval.run_baseline_eval import load_queries
from ranking.features import CorpusStats, FeatureBuilder
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever


def assemble_ltr_examples(dataset_key: str, sample: int | None, candidate_depth: int = 100) -> dict:
    queries = load_queries(dataset_key)
    qrels = load_qrels(config.RAW_DATA_DIR / dataset_key / "qrels.jsonl")

    bm25_index_dir = config.get_bm25_index_dir(sample)
    faiss_index_path = config.get_faiss_index_path(sample)

    print("Loading BM25 index...")
    bm25 = BM25Retriever()
    if not BM25Retriever.index_exists(bm25_index_dir):
        raise FileNotFoundError(f"No BM25 index at {bm25_index_dir}. Run data/preprocess.py first.")
    bm25.load(bm25_index_dir)

    print(f"Loading dense index ({config.DENSE_ENCODER_NAME})...")
    dense = DenseRetriever()
    if not faiss_index_path.exists():
        raise FileNotFoundError(f"No dense index at {faiss_index_path}. Run data/preprocess.py first.")
    dense.load(faiss_index_path)

    hybrid = HybridRetriever(bm25, dense, rrf_k=config.RRF_K)

    # --- Pass 1: retrieval for every query, collecting components + the
    # union of all candidate doc_ids, so passage text lookup is a single
    # corpus pass rather than one pass per query. ---
    print(f"Retrieving top-{candidate_depth} candidates for all {len(queries)} queries...")
    components_by_query = {}
    for qid, qtext in queries.items():
        components_by_query[qid] = hybrid.search_with_components(
            qtext, bm25_depth=candidate_depth, dense_depth=candidate_depth
        )

    all_doc_ids = {
        doc_id
        for components in components_by_query.values()
        for doc_id, _score in components["hybrid"]
    }
    print(f"Looking up passage text for {len(all_doc_ids)} candidate passages (single corpus pass)...")
    passage_texts = lookup_passage_texts(dataset_key, all_doc_ids)

    # --- Fit feature-layer corpus stats (TF-IDF/IDF table) on the same
    # scoped corpus used for indexing. ---
    print("Fitting feature CorpusStats on the scoped corpus...")
    corpus_docs = load_corpus(dataset_key)
    if sample:
        qrels_path = config.RAW_DATA_DIR / dataset_key / "qrels.jsonl"
        corpus_docs = build_scoped_corpus(corpus_docs, qrels_path, target_size=sample, seed=config.RANDOM_SEED)
    corpus_stats = CorpusStats()
    corpus_stats.fit([d["text"] for d in corpus_docs])
    # No role_affinity_scorer here — TREC DL queries carry no persona
    # label, so role_doc_affinity is structurally 0.0 (neutral) for every
    # training example. See ranking/features.py's role_doc_affinity() and
    # results/day5-6_findings.md for what this means for that feature's
    # SHAP importance in Day 7.
    builder = FeatureBuilder(corpus_stats=corpus_stats)

    # --- Pass 2: rerank each query's candidates, then build full feature
    # vectors + labels. ---
    reranker = CrossEncoderReranker()
    print(f"Reranking + building 20-signal features for {len(queries)} queries...")
    examples_by_query = {}
    for i, (qid, qtext) in enumerate(queries.items()):
        components = components_by_query[qid]
        bm25_rank_map = {doc_id: rank for rank, (doc_id, _s) in enumerate(components["bm25"], start=1)}
        bm25_score_map = dict(components["bm25"])
        dense_rank_map = {doc_id: rank for rank, (doc_id, _s) in enumerate(components["dense"], start=1)}
        dense_score_map = dict(components["dense"])
        rrf_score_map = dict(components["hybrid"])
        hybrid_doc_ids = [doc_id for doc_id, _score in components["hybrid"]]

        rerank_candidates = [
            (doc_id, passage_texts[doc_id]) for doc_id in hybrid_doc_ids if doc_id in passage_texts
        ]
        reranked = reranker.rerank(qtext, rerank_candidates, top_k=len(rerank_candidates))
        cross_encoder_score_map = dict(reranked)

        examples = []
        for doc_id in hybrid_doc_ids:
            if doc_id not in passage_texts:
                continue
            candidate = {
                "doc_id": doc_id,
                "doc_text": passage_texts[doc_id],
                "query_id": qid,
                "bm25_score": bm25_score_map.get(doc_id, 0.0),
                "bm25_rank": bm25_rank_map.get(doc_id, 0.0),
                "dense_cosine": dense_score_map.get(doc_id, 0.0),
                "dense_rank": dense_rank_map.get(doc_id, 0.0),
                "rrf_score": rrf_score_map.get(doc_id, 0.0),
                "cross_encoder_score": cross_encoder_score_map.get(doc_id, 0.0),
            }
            label = qrels.get(qid, {}).get(doc_id, 0)
            features = builder.build(qtext, candidate, user_role=None)
            examples.append({"doc_id": doc_id, "features": features, "label": label})

        examples_by_query[qid] = examples
        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  processed {i + 1}/{len(queries)} queries")

    return examples_by_query


def _cache_path(dataset_key: str, sample: int | None, candidate_depth: int):
    suffix = f"_sample_{sample}" if sample else ""
    cache_dir = config.ROOT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"ltr_examples_{dataset_key}{suffix}_depth{candidate_depth}.pkl"


def assemble_ltr_examples_cached(
    dataset_key: str,
    sample: int | None,
    candidate_depth: int = 100,
    force_rebuild: bool = False,
) -> dict:
    """
    Wraps assemble_ltr_examples() with a disk cache — retrieval +
    cross-encoder reranking for all 43 queries takes ~30 min, and Day 7's
    SHAP analysis and ablation study both need this same dataset. Without
    caching, every script that needs it would re-pay that cost.

    IMPORTANT: the cache is keyed on (dataset_key, sample, candidate_depth)
    only, NOT on the feature-computation code in ranking/features.py. If
    you change how features are computed (as happened with the Day 5-6
    label-leakage fix), delete the cache file or pass force_rebuild=True —
    otherwise you'll silently keep evaluating on stale features.
    """
    cache_path = _cache_path(dataset_key, sample, candidate_depth)
    if cache_path.exists() and not force_rebuild:
        import pickle

        print(f"Loading cached LTR dataset from {cache_path}")
        print("  (delete this file, or pass --rebuild-cache, if ranking/features.py has changed since it was built)")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    examples_by_query = assemble_ltr_examples(dataset_key, sample, candidate_depth)

    import pickle

    with open(cache_path, "wb") as f:
        pickle.dump(examples_by_query, f)
    print(f"Cached LTR dataset -> {cache_path}")
    return examples_by_query
