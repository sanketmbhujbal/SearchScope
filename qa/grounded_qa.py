"""
Domain-adapted QA + rejection gate (DESIGN.md §10).

Day 8. Two lightweight domain-adaptation techniques, no fine-tuning:
    1. Corpus-aware system prompt — vocabulary extracted from the corpus,
       injected into the system prompt.
    2. Retrieval-grounded prompting — LLM sees only the top-K ranked
       passages and must answer exclusively from them, citing sources.

Rejection gate: if no supporting evidence, the model sets a boolean
`rejected` field rather than us string-matching a rejection phrase in
free text — see the structured-output note below for why.

PROVIDER NOTE — this deliberately uses the OpenAI API (gpt-4o-mini), not
AWS Bedrock as DESIGN.md originally specified. This is a documented
scoping decision, the same pattern as the corpus-size and encoder-choice
decisions elsewhere in this project: Bedrock requires AWS account setup
and an explicit Anthropic-model access request that can introduce real
setup friction, while existing OpenAI credits meant zero new setup cost.

STRUCTURED OUTPUT NOTE (added after the first real Day 8 run): the
original implementation had the model write free-text with inline
`[doc_id]` citations, parsed afterward with a regex. That approach broke
in a real run — the model wrote `[doc_id: 12345]` instead of `[12345]`
following a genuinely ambiguous prompt instruction, which caused two
accurate, well-grounded answers to be misclassified as hallucinating
citations (see results/day8_findings.md for the full story). Rather than
just patch the regex further, this now uses OpenAI's structured-output
mode (`client.chat.completions.parse()` with a Pydantic schema,
`QAResponse` below) so the model returns `rejected: bool` and
`citations: list[str]` as actual typed fields the API guarantees conform
to the schema, not values buried in prose that need to be re-extracted.
This eliminates the whole class of citation-format parsing bugs by
construction rather than by adding more defensive regex patterns.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

REJECTION_REASON = "No confident answer found from the retrieved documents."

SYSTEM_PROMPT_TEMPLATE = """You are a search assistant answering questions using ONLY the \
provided passages. You must not use outside knowledge.

Domain vocabulary for this corpus: {vocabulary}

Rules:
- Answer only if the passages directly support an answer.
- List every doc_id you actually relied on in the `citations` field. Use \
the exact doc_id values shown in the passages — do not modify, prefix, \
or reformat them.
- Do not cite a doc_id that was not provided in the passages.
- If the passages do not contain enough evidence to answer confidently, \
set `rejected` to true, leave `answer` empty, and leave `citations` empty.
"""


class QAResponse(BaseModel):
    """Structured output schema — the API guarantees the model's response
    conforms to this shape, so downstream code reads typed fields
    directly instead of re-parsing free text."""

    rejected: bool = Field(
        description="True if the passages do not contain enough evidence to answer confidently."
    )
    answer: str = Field(
        description="The grounded answer, or an empty string if rejected is true."
    )
    citations: list[str] = Field(
        description="doc_id values (exactly as shown in the passages) that support the answer. "
        "Empty if rejected is true."
    )


def extract_corpus_vocabulary(corpus_texts: list[str], top_k: int = 50) -> list[str]:
    """
    Domain adaptation technique 1 (DESIGN.md §10.1): extract corpus
    vocabulary for the system prompt, so the model has a lightweight
    signal about this domain's terminology without any fine-tuning.

    Uses raw term frequency (not TF-IDF) deliberately — the goal here is
    "what words define this domain to a reader," which is closer to
    common/frequent domain terms than to rare/distinctive ones. A TF-IDF
    top-k would surface rare technical terms instead, which is a
    reasonable alternative design but a different one; CorpusStats
    (ranking/features.py) already covers the TF-IDF use case for ranking
    features, so this is intentionally a separate, simpler mechanism
    suited to what a system prompt actually needs.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", max_features=top_k)
    vectorizer.fit(corpus_texts)
    return list(vectorizer.get_feature_names_out())


class GroundedQA:
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        from config import OPENAI_MODEL_NAME

        self.model_name = model_name or OPENAI_MODEL_NAME
        self.api_key = api_key  # falls back to OPENAI_API_KEY env var if None
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        return self._client

    def build_prompt(self, query: str, passages: list[dict], vocabulary: list[str]) -> dict:
        """passages: [{"doc_id": ..., "text": ...}, ...], top-K per DESIGN.md §10.1"""
        system = SYSTEM_PROMPT_TEMPLATE.format(vocabulary=", ".join(vocabulary[:50]))
        context = "\n\n".join(f"[{p['doc_id']}] {p['text']}" for p in passages)
        user = f"Passages:\n{context}\n\nQuestion: {query}"
        return {"system": system, "user": user}

    def parse_response(self, parsed: QAResponse, provided_doc_ids: set[str]) -> dict:
        """
        Pure post-processing logic, factored out so it's testable without
        a real API call — see tests/test_grounded_qa.py.

        Splits the model's reported citations into cited_doc_ids (valid,
        actually among the provided passages) and hallucinated_doc_ids
        (claimed but not actually provided) — a citation to a doc_id NOT
        in the context is a hallucinated citation, surfaced separately
        rather than silently trusted, even though structured output
        guarantees the *shape* of citations (a list of strings), not that
        every value is a real, provided doc_id.
        """
        cited_raw = set(parsed.citations)
        cited_valid = cited_raw & provided_doc_ids
        cited_hallucinated = cited_raw - provided_doc_ids

        return {
            "answer": parsed.answer,
            "rejected": parsed.rejected,
            "cited_doc_ids": sorted(cited_valid),
            "hallucinated_doc_ids": sorted(cited_hallucinated),
        }

    def answer(self, query: str, passages: list[dict], vocabulary: list[str]) -> dict:
        """
        Returns: {"answer": str, "rejected": bool, "cited_doc_ids": [...],
                   "hallucinated_doc_ids": [...], "latency_seconds": float}
        """
        prompt = self.build_prompt(query, passages, vocabulary)
        client = self._get_client()

        start = time.time()
        completion = client.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=0.0,  # deterministic-as-possible for reproducible eval
            response_format=QAResponse,
        )
        latency = time.time() - start

        parsed = completion.choices[0].message.parsed
        provided_doc_ids = {p["doc_id"] for p in passages}
        result = self.parse_response(parsed, provided_doc_ids)
        result["latency_seconds"] = latency
        return result
