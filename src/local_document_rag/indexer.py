"""End-to-end document indexing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from local_document_rag.chunker import TextChunk, chunk_pages
from local_document_rag.pdf_loader import extract_pdf_pages


class IndexingEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        ...


class ChunkStore(Protocol):
    def add(
        self,
        chunks: list[TextChunk],
        vectors: NDArray[np.floating],
    ) -> None:
        ...


@dataclass(frozen=True)
class IndexingResult:
    """Summary of one indexed PDF."""

    source: str
    page_count: int
    chunk_count: int


class DocumentIndexer:
    """Coordinate PDF extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        embedder: IndexingEmbedder,
        store: ChunkStore,
    ) -> None:
        self._embedder = embedder
        self._store = store

    def index_pdf(
        self,
        pdf_path: str | Path,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> IndexingResult:
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        vectors = self._embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )
        self._store.add(chunks, vectors)

        return IndexingResult(
            source=Path(pdf_path).name,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
