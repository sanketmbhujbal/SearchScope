"""
Central configuration for SearchScope.

Keeping paths, model names, and pipeline constants in one place makes each
layer swappable without touching the others — the design principle called
out in DESIGN.md §6 (System Architecture).
"""
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT_DIR / "indexes"
BM25_INDEX_DIR = INDEX_DIR / "bm25"
FAISS_INDEX_PATH = INDEX_DIR / "dense_flat.faiss"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_bm25_index_dir(sample: int | None = None) -> Path:
    """Sampled runs get their own index dir so they never collide with (or
    silently overwrite) a full-corpus index you spent an hour building."""
    if sample:
        return INDEX_DIR / f"bm25_sample_{sample}"
    return BM25_INDEX_DIR


def get_faiss_index_path(sample: int | None = None) -> Path:
    if sample:
        return INDEX_DIR / f"dense_flat_sample_{sample}.faiss"
    return FAISS_INDEX_PATH

# --- Dataset (DESIGN.md §5) ---
DATASET_NAME = "msmarco-passage/trec-dl-2019/judged"  # dataset identifier (informational)
EVAL_DATASETS = {
    "trec-dl-2019": "msmarco-passage/trec-dl-2019/judged",
    "trec-dl-2020": "msmarco-passage/trec-dl-2020/judged",
}

# Recommended corpus scope (DESIGN.md §5 — corpus scoping decision).
# Pass as `--sample` to data/preprocess.py and eval/run_baseline_eval.py.
# All judged passages are guaranteed included regardless of this number
# (see data/sampling.py) — this only controls how much unjudged "filler"
# is added so Recall@100 still means something.
# At measured CPU throughput (~0.07s/passage with bge-small + full thread
# utilization), 1,000,000 passages is ~18+ hours — too long for iteration.
# 150,000 is ~3 hours (overnight-viable); adjust based on your hardware.
RECOMMENDED_CORPUS_SCOPE = 150_000

# --- Retrieval (DESIGN.md §7) ---
# bge-small-en-v1.5 (384-dim) rather than bge-base-en-v1.5 (768-dim) —
# see DESIGN.md §5. bge-base scores a few NDCG points higher on BEIR, but
# proved impractical to encode at scale on available CPU/free-GPU
# infrastructure. Pass model_name="BAAI/bge-base-en-v1.5" to DenseRetriever
# explicitly if you have real compute headroom.
DENSE_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
BM25_TOP_K = 100          # depth pulled from BM25 before fusion
DENSE_TOP_K = 100         # depth pulled from dense retrieval before fusion
RRF_K = 60                # standard RRF damping constant
HYBRID_TOP_K = 100        # depth passed forward to the reranker

# --- Reranking (DESIGN.md, "NEW IN V2 Layer 2") ---
CROSS_ENCODER_NAME = "BAAI/bge-reranker-base"
RERANK_TOP_K = 20         # depth passed forward to the LTR layer

# --- Ranking / LTR (DESIGN.md §9) ---
LTR_FINAL_TOP_K = 10      # final ranked list length (matches NDCG@10 metric)
XGBOOST_PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@10",
    "max_depth": 6,
    "eta": 0.1,
    "seed": 42,
}

# --- Personalization (DESIGN.md §9, Layer 3b) ---
USER_ROLES = ["engineer", "sales", "hr", "legal"]
ROLE_AFFINITY_TOPICS = {
    "engineer": ["code", "deployment", "debugging", "infra", "api", "auth"],
    "sales": ["discount", "pricing", "sla", "contract", "renewal", "quota"],
    "hr": ["policy", "benefits", "onboarding", "leave", "payroll", "review"],
    "legal": ["compliance", "contract", "liability", "nda", "regulation"],
}

# --- QA (DESIGN.md §10) ---
# Uses OpenAI (gpt-4o-mini) rather than AWS Bedrock/Claude Haiku as
# DESIGN.md originally specified — a documented scoping decision, see
# qa/grounded_qa.py's module docstring for why.
OPENAI_MODEL_NAME = "gpt-4o-mini"
QA_CONTEXT_TOP_K = 5       # number of ranked passages shown to the LLM
QA_MAX_LATENCY_SECONDS = 2.0
QA_VOCABULARY_SIZE = 50    # number of corpus terms injected into the system prompt

# --- Eval (DESIGN.md §11) ---
METRICS = {"ndcg_cut_10", "recip_rank", "recall_100", "map"}
RANDOM_SEED = 42
