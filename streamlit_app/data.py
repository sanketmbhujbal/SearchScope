"""
Real, precomputed data backing the SearchScope demo app.

Every number here comes from an actual run recorded in this project's
results/ files, referenced in each section below. This app does not run
the live retrieval/reranking/LTR pipeline (that needs Pyserini + a JVM,
FAISS, PyTorch, and XGBoost, which is exactly the kind of heavy,
environment-fragile stack this project spent real time fighting through
during development). Deploying that stack inside a free-tier hosted demo
would reintroduce the same risk for a component whose whole point is to
be reliable and fast to load. Instead, this app presents real results
that were already computed and verified, interactively.

If a number in this file doesn't match its cited source file, that's a
bug in this file, not a new finding. The results/ files are the source
of truth.
"""

# --- Day 1-2, Day 3, Day 5-6: pipeline-stage metrics ---
# Source: results/day1-2_findings.md, results/day3_findings.md,
# results/day5-6_findings.md (corrected, post label-leak-fix numbers)
PIPELINE_METRICS = [
    {"stage": "BM25", "ndcg10": 0.4839, "mrr10": 0.8037, "recall100": 0.6125, "map": 0.3923},
    {"stage": "Dense", "ndcg10": 0.7337, "mrr10": 1.0000, "recall100": 0.6812, "map": 0.5398},
    {"stage": "Hybrid (RRF)", "ndcg10": 0.6845, "mrr10": 0.9564, "recall100": 0.6895, "map": 0.5056},
    {"stage": "+ Cross-Encoder", "ndcg10": 0.7576, "mrr10": 0.9457, "recall100": 0.6895, "map": 0.5607},
    {"stage": "+ LTR", "ndcg10": 0.7324, "mrr10": 0.9767, "recall100": 0.6895, "map": 0.5439},
]

# The leak, before and after the Day 5-6 fix.
# Source: results/day5-6_findings.md
LEAK_COMPARISON = [
    {"label": "Reranker (reference)", "ndcg10": 0.7576},
    {"label": "LTR (first run, leaking)", "ndcg10": 0.9731},
    {"label": "LTR (after fix)", "ndcg10": 0.7324},
]

# --- Day 7: SHAP global feature importance ---
# Source: results/day7_findings.md
SHAP_IMPORTANCE = [
    {"feature": "cross_encoder_score", "importance": 1.1601},
    {"feature": "dense_cosine", "importance": 0.3866},
    {"feature": "rrf_score", "importance": 0.2458},
    {"feature": "dense_rank", "importance": 0.2136},
    {"feature": "doc_length_log", "importance": 0.1626},
    {"feature": "bm25_score", "importance": 0.1527},
    {"feature": "term_overlap", "importance": 0.0908},
    {"feature": "source_authority", "importance": 0.0819},
    {"feature": "query_idf_mean", "importance": 0.0807},
    {"feature": "simulated_ctr", "importance": 0.0704},
    {"feature": "query_entropy", "importance": 0.0696},
    {"feature": "simulated_dwell_time", "importance": 0.0689},
    {"feature": "metadata_overlap", "importance": 0.0636},
    {"feature": "bm25_rank", "importance": 0.0630},
    {"feature": "title_match", "importance": 0.0619},
    {"feature": "doc_recency", "importance": 0.0608},
    {"feature": "query_length", "importance": 0.0410},
    {"feature": "section_importance", "importance": 0.0129},
    {"feature": "query_intent_class", "importance": 0.0124},
    {"feature": "role_doc_affinity", "importance": 0.0000},
]

# Which features are synthetic (no real signal exists in TREC DL for these)
# vs real, computed directly from the pipeline. Source: ranking/features.py
# module docstring.
SYNTHETIC_FEATURES = {"simulated_ctr", "simulated_dwell_time", "doc_recency", "source_authority"}

# --- Day 7: ablation study ---
# Source: results/day7_findings.md
ABLATION_RESULTS = [
    {"ablation": "Full model", "ndcg10": 0.7324, "delta": 0.0},
    {"ablation": "No lexical", "ndcg10": 0.7377, "delta": 0.0053},
    {"ablation": "No semantic", "ndcg10": 0.4071, "delta": -0.3253},
    {"ablation": "No query", "ndcg10": 0.7197, "delta": -0.0126},
    {"ablation": "No behavior", "ndcg10": 0.7320, "delta": -0.0004},
    {"ablation": "No freshness/authority", "ndcg10": 0.7260, "delta": -0.0063},
    {"ablation": "No personalization", "ndcg10": 0.7324, "delta": 0.0000},
    {"ablation": "No synthetic (bonus)", "ndcg10": 0.7152, "delta": -0.0171},
    {"ablation": "BM25 only", "ndcg10": 0.3944, "delta": -0.3380},
    {"ablation": "Cross-encoder only", "ndcg10": 0.7238, "delta": -0.0086},
]

# --- Day 9, Case 2: the vocabulary mismatch worked example ---
# Source: results/day9_failure_analysis.md, mined directly from the real
# persisted indexes via eval/mine_failure_cases.py.
VOCAB_MISMATCH_CASE = {
    "query": "what are the three percenters?",
    "relevant_doc_id": "3423067",
    "relevant_doc_grade": 3,
    "relevant_doc_bm25_rank": 84,
    "relevant_doc_dense_rank": 2,
    "relevant_doc_text": (
        "A loose affiliation of like minded Americans who vow to refuse to "
        "comply with laws that violate the second amendment right to keep "
        "(own) and bear (carry) firearms. The name 3 percenter comes from "
        "the fact that only 3 percent of colonial British subjects..."
    ),
    "bm25_top1_doc_id": "6630430",
    "bm25_top1_text": (
        "Microsoft Project contains three measures of %Complete...Jamaica "
        "weather, Falmouth Jamaica weather, Ocho Rios Jamaica weather, "
        "Negril Jamaica weather, Runaway Bay Jamaica Weather..."
    ),
}

# Runner-up case, also real, kept as a second illustration.
VOCAB_MISMATCH_RUNNER_UP = {
    "query": "medicare's definition of mechanical ventilation",
    "relevant_doc_id": "8390518",
    "relevant_doc_grade": 3,
    "relevant_doc_bm25_rank": 39,
    "relevant_doc_dense_rank": 2,
    "relevant_doc_text": (
        "Mechanical ventilation is the medical term for artificial "
        "ventilation where mechanical means is used to assist or replace "
        "spontaneous breathing."
    ),
    "bm25_top1_doc_id": "2176453",
    "bm25_top1_text": (
        "For 363 of the selected claims, Medicare payments to hospitals "
        "were incorrect. For example, for one beneficiary, the "
        "documentation (e.g., time log for the mechanical ventilation) "
        "showed that the beneficiary had received 73 hours of mechanical "
        "ventilation..."
    ),
}

# --- Day 4: personalization demo, real output for query "policy" ---
# Source: results/day4_findings.md. Scores are the demo script's blended
# relevance+affinity score (see eval/run_personalization_demo.py), not a
# production ranking formula, just what made the real difference visible.
PERSONALIZATION_QUERY = "policy"
PERSONALIZATION_RESULTS = {
    "Engineer": [
        {"doc_id": "7598958", "score": 0.703,
         "text": "The foundation of the National Workplace Policy on HIV/AIDS, like the foundation of the Workplace Policy of the Office of the Services Commissions..."},
        {"doc_id": "1169578", "score": 0.500,
         "text": "Group Policy. Policies linked to Active Directory domains, organizational units, or groups, which are applied to the child objects within..."},
        {"doc_id": "5589306", "score": 0.432,
         "text": "Cite public policies, also referred to as government policy, as the organization, the publication date, the title, the filing number..."},
    ],
    "Sales": [
        {"doc_id": "1242835", "score": 0.698,
         "text": "(1) Read Your Policy Carefully: This Outline of Coverage provides a very brief description of some of the important features of your policy..."},
        {"doc_id": "1169578", "score": 0.500,
         "text": "Group Policy. Policies linked to Active Directory domains, organizational units, or groups, which are applied to the child objects within..."},
        {"doc_id": "5589306", "score": 0.432,
         "text": "Cite public policies, also referred to as government policy, as the organization, the publication date, the title, the filing number..."},
    ],
    "HR": [
        {"doc_id": "1169578", "score": 0.804,
         "text": "Group Policy. Policies linked to Active Directory domains, organizational units, or groups, which are applied to the child objects within..."},
        {"doc_id": "2350582", "score": 0.796,
         "text": "Please click about to review Keystone's policy on Concussion Management. All Players, Parents, Coaches, and Managers shall review and abide by this policy..."},
        {"doc_id": "5794872", "score": 0.750,
         "text": "Policy formulation is, therefore, comprised of analysis that identifies the most effective policies and political authorization..."},
    ],
    "Legal": [
        {"doc_id": "1242835", "score": 0.698,
         "text": "(1) Read Your Policy Carefully: This Outline of Coverage provides a very brief description of some of the important features of your policy..."},
        {"doc_id": "1582199", "score": 0.520,
         "text": "Why our HIPAA Security Rule Policies and Procedures Templates / forms. We have created 71 HIPAA security policies whereby 60 of them are the security..."},
        {"doc_id": "1169578", "score": 0.500,
         "text": "Group Policy. Policies linked to Active Directory domains, organizational units, or groups, which are applied to the child objects within..."},
    ],
}

# --- Day 8: the 7 real false-rejection cases ---
# Source: results/day8_findings.md
QA_FALSE_REJECTIONS = [
    {
        "query": "medicare's definition of mechanical ventilation",
        "verdict": "Correct rejection",
        "verdict_color": "#2f9e44",
        "explanation": (
            "Passages define mechanical ventilation generically. None mention "
            "Medicare specifically. The query asks for Medicare's definition, "
            "which genuinely isn't supported."
        ),
    },
    {
        "query": "what is an aml surveillance analyst",
        "verdict": "Borderline / defensible",
        "verdict_color": "#f08c00",
        "explanation": (
            "Passages describe \"AML Analyst\" and \"BSA/AML Analyst\" roles but "
            "never the exact phrase \"AML surveillance analyst.\" Reasonable "
            "caution about a term not literally present."
        ),
    },
    {
        "query": "what is the daily life of thai people",
        "verdict": "Likely over-cautious",
        "verdict_color": "#e8590c",
        "explanation": (
            "Passages directly describe SANUK and JAI YEN as everyday-life "
            "concepts, real, on-topic content the model could have "
            "synthesized into an answer."
        ),
    },
    {
        "query": "cost of interior concrete flooring",
        "verdict": "Clear miss",
        "verdict_color": "#e03131",
        "explanation": (
            "A passage states directly: \"concrete floors can cost as little "
            "as $2 to $6 a square foot or be as expensive as $15 to $30 a "
            "square foot.\" A direct, unambiguous answer was in context and "
            "the model rejected anyway."
        ),
    },
    {
        "query": "causes of military suicide",
        "verdict": "Clear miss",
        "verdict_color": "#e03131",
        "explanation": (
            "A passage states directly: \"We suggest that moral injury is "
            "likely one of the most important factors in military suicide "
            "rates.\" A stated cause was explicitly present."
        ),
    },
    {
        "query": "anthropological definition of environment",
        "verdict": "Borderline / defensible",
        "verdict_color": "#f08c00",
        "explanation": (
            "Passages define environmental anthropology and ecological "
            "anthropology (the sub-fields) but none give a clean definition "
            "of \"environment\" itself, a subtly different ask."
        ),
    },
    {
        "query": "is cdg airport in main paris",
        "verdict": "Clear miss",
        "verdict_color": "#e03131",
        "explanation": (
            "A passage states directly: \"Charles de Gaulle airport (CDG) is "
            "the main international airport for Paris.\" Directly answers "
            "the yes/no question."
        ),
    },
]

# --- Day 8: QA layer summary metrics ---
# Source: results/day8_findings.md (corrected citation hygiene after the
# parsing-bug fix)
QA_METRICS = {
    "rejection_rate_unanswerable": 1.00,
    "false_rejections_on_answerable": 7,
    "total_answerable": 43,
    "citation_hygiene_corrected": 1.00,
    "citation_hygiene_original": 0.944,
}

# --- Key findings summary, mirrors README.md's Key Findings section ---
KEY_FINDINGS = [
    {
        "title": "Naive hybrid fusion made retrieval worse, not better",
        "body": (
            "Reciprocal rank fusion at equal weight pulled the ranking toward "
            "the weaker of two retrievers instead of the stronger one, "
            "dropping NDCG@10 below dense retrieval alone. Cross-encoder "
            "reranking recovered the loss and then exceeded dense retrieval "
            "on its own."
        ),
        "source": "results/day1-2_findings.md, results/day3_findings.md",
    },
    {
        "title": "A too-good LTR result turned out to be a label leak",
        "body": (
            "A synthetic training feature was built from the answer it was "
            "supposed to predict. Cross-validation caught the model leaking "
            "across folds, but missed that the feature itself was already "
            "computed from ground truth. Fixed, re-run, and the corrected "
            "number is far less exciting and far more honest."
        ),
        "source": "results/day5-6_findings.md",
    },
    {
        "title": "SHAP and ablation disagreed, and the disagreement was the finding",
        "body": (
            "Four synthetic features showed real importance in SHAP but "
            "almost no impact when ablated. The model was mildly overfitting "
            "to noise it had variance in, not to noise it had zero variance "
            "in, a distinction only visible by running both analyses."
        ),
        "source": "results/day7_findings.md",
    },
    {
        "title": "Two QA bugs stayed invisible until someone read the output",
        "body": (
            "A citation-parsing bug flagged two accurate answers as "
            "hallucinations. A 100% rejection rate on out-of-scope queries "
            "looked perfect until reading the 7 cases where it wrongly "
            "rejected an answerable query, three with the answer quoted "
            "directly in context."
        ),
        "source": "results/day8_findings.md",
    },
]
