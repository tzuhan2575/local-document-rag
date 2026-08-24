"""Local text embedding with Sentence Transformers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """Encode documents and queries as normalized dense vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

        self._model = model

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding model did not report its dimension")

        return dimension

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed multiple texts as a two-dimensional normalized array."""

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed one non-empty search query."""

        if not query.strip():
            raise ValueError("query must not be empty")

        return self.embed_texts([query])[0]
