import pytest

from local_document_rag.chunker import TextChunk
from local_document_rag.prompting import (
    build_grounded_prompt,
    format_retrieved_context,
)
from local_document_rag.retriever import RetrievedChunk


def make_result(source, page, text, score):
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=f"{source}-{page}",
            page_number=page,
            text=text,
            start_char=0,
            end_char=len(text),
            source=source,
        ),
        score=score,
    )


def test_context_contains_ranked_source_labels_and_metadata():
    results = [
        make_result("first.pdf", 2, "first evidence", 0.91234),
        make_result("second.pdf", 5, "second evidence", 0.8),
    ]

    context = format_retrieved_context(results)

    assert "[Source 1] file=first.pdf, page=2, score=0.9123" in context
    assert "[Source 2] file=second.pdf, page=5, score=0.8000" in context
    assert "first evidence" in context
    assert "second evidence" in context


def test_empty_results_produce_explicit_missing_context_message():
    prompt = build_grounded_prompt("What is the answer?", [])

    assert "No relevant document context was retrieved" in prompt.user
    assert "cannot be determined" in prompt.system


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="question must not be empty"):
        build_grounded_prompt("   ", [])
