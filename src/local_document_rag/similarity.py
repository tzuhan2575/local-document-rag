"""Exact cosine similarity search using NumPy."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SearchResult:
    """One ranked vector search result."""

    index: int
    score: float


def cosine_similarity_search(
    query_vector: NDArray[np.floating],
    document_vectors: NDArray[np.floating],
    top_k: int = 3,
) -> list[SearchResult]:
    """Return the top-k document vectors ranked by cosine similarity."""

    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_vectors, dtype=np.float32)

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional")

    if documents.ndim != 2:
        raise ValueError("document_vectors must be two-dimensional")

    if documents.shape[1] != query.shape[0]:
        raise ValueError("query and document dimensions must match")

    if len(documents) == 0:
        return []

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)

    if query_norm == 0:
        raise ValueError("query_vector must not be a zero vector")

    if np.any(document_norms == 0):
        raise ValueError("document_vectors must not contain zero vectors")

    scores = (documents @ query) / (document_norms * query_norm)
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    return [
        SearchResult(index=int(index), score=float(scores[index]))
        for index in ranked_indices
    ]
