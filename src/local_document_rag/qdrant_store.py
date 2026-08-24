"""Qdrant-backed storage and retrieval for embedded text chunks."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from local_document_rag.chunker import TextChunk
from local_document_rag.retriever import RetrievedChunk


class QdrantChunkStore:
    """Store chunk vectors and retrieve them with cosine similarity."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be greater than 0")

        self._client = client
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def add(
        self,
        chunks: Sequence[TextChunk],
        vectors: NDArray[np.floating],
    ) -> None:
        """Upsert chunks and their vectors into the collection."""

        vector_array = np.asarray(vectors, dtype=np.float32)

        if vector_array.ndim != 2:
            raise ValueError("vectors must be two-dimensional")

        if len(chunks) != len(vector_array):
            raise ValueError("chunks and vectors must have the same length")

        if vector_array.shape[1] != self.vector_dimension:
            raise ValueError("vector dimension does not match the collection")

        if not chunks:
            return

        points = []

        for chunk, vector in zip(chunks, vector_array, strict=True):
            stable_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{chunk.source}|{chunk.chunk_id}|{chunk.page_number}|{chunk.text}",
                )
            )

            points.append(
                models.PointStruct(
                    id=stable_id,
                    vector=vector.tolist(),
                    payload={
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: NDArray[np.floating],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query vector."""

        query = np.asarray(query_vector, dtype=np.float32)

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        if query.shape[0] != self.vector_dimension:
            raise ValueError("query dimension does not match the collection")

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query.tolist(),
            limit=top_k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            chunk = TextChunk(
                chunk_id=str(payload["chunk_id"]),
                page_number=int(payload["page_number"]),
                text=str(payload["text"]),
                start_char=int(payload["start_char"]),
                end_char=int(payload["end_char"]),
                source=str(payload.get("source", "")),
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(point.score),
                )
            )

        return results
