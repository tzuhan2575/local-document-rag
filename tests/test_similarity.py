import numpy as np
import pytest

from local_document_rag.similarity import cosine_similarity_search


def test_returns_top_k_results_in_descending_similarity_order():
    documents = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ]
    )
    query = np.array([1.0, 0.0])

    results = cosine_similarity_search(query, documents, top_k=2)

    assert [result.index for result in results] == [0, 1]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.9701425)


def test_empty_document_matrix_returns_no_results():
    results = cosine_similarity_search(
        np.array([1.0, 0.0]),
        np.empty((0, 2)),
    )

    assert results == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        cosine_similarity_search(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    ("query", "documents", "message"),
    [
        (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), "one-dimensional"),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), "two-dimensional"),
        (np.array([1.0, 0.0]), np.array([[1.0, 0.0, 0.0]]), "dimensions must match"),
        (np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), "zero vector"),
        (np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), "zero vectors"),
    ],
)
def test_invalid_vectors_are_rejected(query, documents, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity_search(query, documents)
