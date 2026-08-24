import numpy as np
import pytest

from local_document_rag.embedder import LocalEmbedder


class FakeEmbeddingModel:
    def __init__(self):
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.encode_calls.append((texts, kwargs))
        return np.array(
            [[1.0, 2.0, 3.0] for _ in texts],
            dtype=np.float64,
        )


def test_embed_texts_returns_float32_matrix_and_requests_normalization():
    model = FakeEmbeddingModel()
    embedder = LocalEmbedder(model=model)

    embeddings = embedder.embed_texts(["first", "second"])

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert model.encode_calls == [
        (
            ["first", "second"],
            {
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            },
        )
    ]


def test_embed_empty_texts_returns_empty_matrix_without_model_call():
    model = FakeEmbeddingModel()
    embedder = LocalEmbedder(model=model)

    embeddings = embedder.embed_texts([])

    assert embeddings.shape == (0, 3)
    assert model.encode_calls == []


def test_embed_query_returns_one_vector_and_rejects_empty_query():
    embedder = LocalEmbedder(model=FakeEmbeddingModel())

    assert embedder.embed_query("retrieval question").shape == (3,)

    with pytest.raises(ValueError, match="query must not be empty"):
        embedder.embed_query("   ")
