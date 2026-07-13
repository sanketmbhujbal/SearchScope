"""
Domain-adapted QA + rejection gate (DESIGN.md §10).

Day 8. Two lightweight domain-adaptation techniques, no fine-tuning:
    1. Corpus-aware system prompt — vocabulary/acronyms/topic distribution
       extracted from the corpus, injected into the system prompt.
    2. Retrieval-grounded prompting — LLM sees only the top-5 LTR-ranked
       passages and must answer exclusively from them, citing sources.

Rejection gate: if no supporting evidence, return "No confident answer
found from the retrieved documents" instead of fabricating — measured via
Answer Rejection Rate on 50 manually identified unanswerable queries
(DESIGN.md §10.2).

TODO (Day 8):
    - Extract corpus vocabulary/acronym list for the system prompt
    - Implement build_grounded_prompt(query, passages) -> system + user prompt
    - Call Bedrock (config.BEDROCK_MODEL_ID) via boto3
    - Implement the rejection gate (parse "no confident answer" signal,
      or add an explicit confidence-scoring step before answering)
    - Measure Answer Supported Rate (human spot-check, 50 samples) and
      Answer Rejection Rate (50 unanswerable queries) per DESIGN.md §10.3
"""
from __future__ import annotations

REJECTION_PHRASE = "No confident answer found from the retrieved documents."

SYSTEM_PROMPT_TEMPLATE = """You are a search assistant answering questions using ONLY the \
provided passages. You must not use outside knowledge.

Domain vocabulary for this corpus: {vocabulary}

Rules:
- Answer only if the passages directly support an answer.
- Cite the passage(s) you used by their doc_id.
- If the passages do not contain enough evidence to answer confidently, \
respond with exactly: "{rejection_phrase}"
"""


class DomainAdaptedQA:
    def __init__(self, model_id: str | None = None, region: str | None = None):
        from config import BEDROCK_MODEL_ID, BEDROCK_REGION

        self.model_id = model_id or BEDROCK_MODEL_ID
        self.region = region or BEDROCK_REGION
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def build_prompt(self, query: str, passages: list[dict], vocabulary: list[str]) -> dict:
        """passages: [{"doc_id": ..., "text": ...}, ...], top-5 per DESIGN.md §10.1"""
        system = SYSTEM_PROMPT_TEMPLATE.format(
            vocabulary=", ".join(vocabulary[:50]),
            rejection_phrase=REJECTION_PHRASE,
        )
        context = "\n\n".join(f"[{p['doc_id']}] {p['text']}" for p in passages)
        user = f"Passages:\n{context}\n\nQuestion: {query}"
        return {"system": system, "user": user}

    def answer(self, query: str, passages: list[dict], vocabulary: list[str]) -> dict:
        """
        TODO (Day 8): wire up the actual Bedrock invoke_model call, parse
        the response, and detect the rejection phrase to compute the
        Answer Rejection Rate metric.
        Returns: {"answer": str, "rejected": bool, "cited_doc_ids": [...]}
        """
        raise NotImplementedError("Implement Bedrock call in Day 8.")
