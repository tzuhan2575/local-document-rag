import pytest

from local_document_rag.chunker import TextChunk
from local_document_rag.evaluation import (
    RetrievalExample,
    evaluate_retrieval,
)
from local_document_rag.retriever import RetrievedChunk


def make_result(chunk_id):
    return RetrievedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            page_number=1,
            text=chunk_id,
            start_char=0,
            end_char=len(chunk_id),
        ),
        score=1.0,
    )


def test_evaluate_retrieval_computes_recall_and_mrr():
    rankings = {
        "question-1": [
            make_result("relevant-a"),
            make_result("other"),
        ],
        "question-2": [
            make_result("other"),
            make_result("relevant-c"),
        ],
    }
    examples = [
        RetrievalExample(
            "question-1",
            frozenset({"relevant-a", "relevant-b"}),
        ),
        RetrievalExample(
            "question-2",
            frozenset({"relevant-c"}),
        ),
    ]

    metrics = evaluate_retrieval(
        examples,
        retrieve=lambda question, top_k: rankings[question],
        top_k=2,
    )

    assert metrics.recall_at_k == pytest.approx(0.75)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)
    assert metrics.evaluated_queries == 2


def test_results_below_top_k_are_not_counted():
    metrics = evaluate_retrieval(
        [
            RetrievalExample(
                "question",
                frozenset({"relevant"}),
            )
        ],
        retrieve=lambda question, top_k: [
            make_result("other"),
            make_result("relevant"),
        ],
        top_k=1,
    )

    assert metrics.recall_at_k == 0.0
    assert metrics.mean_reciprocal_rank == 0.0


@pytest.mark.parametrize(
    ("examples", "top_k", "message"),
    [
        ([], 1, "at least one"),
        (
            [
                RetrievalExample(
                    "question",
                    frozenset({"relevant"}),
                )
            ],
            0,
            "top_k",
        ),
    ],
)
def test_invalid_evaluation_configuration_is_rejected(
    examples,
    top_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(
            examples,
            retrieve=lambda question, limit: [],
            top_k=top_k,
        )


@pytest.mark.parametrize(
    ("example", "message"),
    [
        (
            RetrievalExample(
                "   ",
                frozenset({"relevant"}),
            ),
            "question",
        ),
        (
            RetrievalExample(
                "question",
                frozenset(),
            ),
            "relevant chunk",
        ),
    ],
)
def test_invalid_examples_are_rejected(example, message):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(
            [example],
            retrieve=lambda question, top_k: [],
        )
