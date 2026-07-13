"""
Small synthetic corpus + qrels for testing pipeline logic without a JVM,
downloaded models, or the real MS MARCO corpus. Deliberately tiny (6
passages, 2 queries) — this validates wiring and correctness, not
retrieval quality.
"""

TOY_CORPUS = [
    {"doc_id": "d1", "text": "how to reset your service account credentials"},
    {"doc_id": "d2", "text": "resetting your personal password for HR portal access"},
    {"doc_id": "d3", "text": "release pipeline runbook for deployment engineers"},
    {"doc_id": "d4", "text": "HR onboarding process for new employees"},
    {"doc_id": "d5", "text": "refund policy for enterprise customers"},
    {"doc_id": "d6", "text": "leave policy and benefits overview for staff"},
]

TOY_QUERIES = {
    "q1": "reset service credentials",
    "q2": "deployment process",
}

# Graded relevance judgments (0-3), matching TREC-style qrels
TOY_QRELS = {
    "q1": {"d1": 3, "d2": 0, "d3": 0, "d4": 0, "d5": 0, "d6": 0},
    "q2": {"d3": 3, "d4": 0, "d1": 0, "d2": 0, "d5": 0, "d6": 0},
}
