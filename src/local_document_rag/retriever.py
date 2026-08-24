"""In-memory retrieval over embedded text chunks."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from local_document_rag.chunker import TextChunk
from local_document_rag.similarity import cosine_similarity_search


class Embedder(Protocol):
    """Embedding interface required by the retriever."""

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        ...

    def embed_query(self, query: str) -> NDArray[np.float32]:
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by semantic search."""

    chunk: TextChunk
    score: float


class InMemoryRetriever:
    """Embed chunks once and perform exact cosine similarity search."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        embedder: Embedder,
    ) -> None:
        self._chunks = list(chunks)
        self._embedder = embedder
        self._document_vectors = embedder.embed_texts(
            [chunk.text for chunk in self._chunks]
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed_query(query)

        results = cosine_similarity_search(
            query_vector=query_vector,
            document_vectors=self._document_vectors,
            top_k=top_k,
        )

        return [
            RetrievedChunk(
                chunk=self._chunks[result.index],
                score=result.score,
            )
            for result in results
        ]
