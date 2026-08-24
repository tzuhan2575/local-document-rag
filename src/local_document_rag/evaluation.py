"""Retrieval evaluation metrics for labeled RAG questions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from local_document_rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class RetrievalExample:
    """One question with known relevant chunk IDs."""

    question: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate retrieval quality metrics."""

    recall_at_k: float
    mean_reciprocal_rank: float
    evaluated_queries: int


def evaluate_retrieval(
    examples: Sequence[RetrievalExample],
    retrieve: Callable[[str, int], Sequence[RetrievedChunk]],
    top_k: int = 3,
) -> RetrievalMetrics:
    """Evaluate average Recall@k and mean reciprocal rank."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not examples:
        raise ValueError("at least one evaluation example is required")

    recall_total = 0.0
    reciprocal_rank_total = 0.0

    for example in examples:
        if not example.question.strip():
            raise ValueError("evaluation question must not be empty")

        if not example.relevant_chunk_ids:
            raise ValueError(
                "each example must contain a relevant chunk ID"
            )

        results = list(retrieve(example.question, top_k))
        retrieved_ids = [
            result.chunk.chunk_id
            for result in results[:top_k]
        ]

        relevant_retrieved = (
            set(retrieved_ids) & example.relevant_chunk_ids
        )
        recall_total += (
            len(relevant_retrieved)
            / len(example.relevant_chunk_ids)
        )

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in example.relevant_chunk_ids:
                reciprocal_rank_total += 1.0 / rank
                break

    query_count = len(examples)

    return RetrievalMetrics(
        recall_at_k=recall_total / query_count,
        mean_reciprocal_rank=(
            reciprocal_rank_total / query_count
        ),
        evaluated_queries=query_count,
    )
