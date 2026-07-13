from qa.grounded_qa import GroundedQA, QAResponse, extract_corpus_vocabulary


def test_extract_corpus_vocabulary_returns_at_most_top_k_terms():
    corpus = [
        "how to reset your service account credentials",
        "resetting your personal password for HR portal access",
        "release pipeline runbook for deployment engineers",
    ]
    vocab = extract_corpus_vocabulary(corpus, top_k=5)
    assert len(vocab) <= 5


def test_extract_corpus_vocabulary_excludes_stopwords():
    corpus = ["the quick brown fox jumps over the lazy dog and the cat"]
    vocab = extract_corpus_vocabulary(corpus, top_k=20)
    assert "the" not in vocab
    assert "and" not in vocab
    assert "over" not in vocab


def test_build_prompt_includes_all_passages():
    qa = GroundedQA()
    passages = [
        {"doc_id": "d1", "text": "reset your password here"},
        {"doc_id": "d2", "text": "deployment guide"},
    ]
    prompt = qa.build_prompt("how to reset password", passages, ["password", "deployment"])

    assert "[d1]" in prompt["user"]
    assert "[d2]" in prompt["user"]
    assert "reset your password here" in prompt["user"]
    assert "how to reset password" in prompt["user"]


def test_build_prompt_truncates_vocabulary_to_50_terms():
    qa = GroundedQA()
    huge_vocab = [f"term{i}" for i in range(200)]
    prompt = qa.build_prompt("query", [], huge_vocab)
    assert "term49" in prompt["system"]
    assert "term50" not in prompt["system"]


def test_parse_response_splits_valid_and_hallucinated_citations():
    """Structured output guarantees citations is a list of strings, but
    not that every value is a real, provided doc_id — parse_response
    still needs to validate that against the actual passage set."""
    qa = GroundedQA()
    parsed = QAResponse(rejected=False, answer="Some answer.", citations=["d1", "d99"])
    result = qa.parse_response(parsed, provided_doc_ids={"d1", "d2"})

    assert result["rejected"] is False
    assert result["cited_doc_ids"] == ["d1"]
    assert result["hallucinated_doc_ids"] == ["d99"]
    assert result["answer"] == "Some answer."


def test_parse_response_all_valid_citations():
    qa = GroundedQA()
    parsed = QAResponse(rejected=False, answer="Answer here.", citations=["d1", "d2"])
    result = qa.parse_response(parsed, provided_doc_ids={"d1", "d2", "d3"})

    assert result["cited_doc_ids"] == ["d1", "d2"]
    assert result["hallucinated_doc_ids"] == []


def test_parse_response_rejected_case():
    qa = GroundedQA()
    parsed = QAResponse(rejected=True, answer="", citations=[])
    result = qa.parse_response(parsed, provided_doc_ids={"d1"})

    assert result["rejected"] is True
    assert result["answer"] == ""
    assert result["cited_doc_ids"] == []
    assert result["hallucinated_doc_ids"] == []


def test_parse_response_no_citations_no_rejection():
    qa = GroundedQA()
    parsed = QAResponse(rejected=False, answer="A plain answer with no citations.", citations=[])
    result = qa.parse_response(parsed, provided_doc_ids={"d1"})

    assert result["rejected"] is False
    assert result["cited_doc_ids"] == []
    assert result["hallucinated_doc_ids"] == []


def test_qa_response_schema_rejects_missing_required_fields():
    """Sanity check that the Pydantic schema actually enforces its shape
    — this is the guarantee the structured-output switch is buying us."""
    import pydantic

    try:
        QAResponse(rejected=False)  # missing answer, citations
        assert False, "should have raised a validation error"
    except pydantic.ValidationError:
        pass
