"""Baseline character-based text chunking."""

from dataclasses import dataclass
from typing import Sequence

from local_document_rag.pdf_loader import PageText


@dataclass(frozen=True)
class TextChunk:
    """A retrievable text chunk with source position metadata."""

    chunk_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int


def chunk_pages(
    pages: Sequence[PageText],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """Split page text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []

    for page in pages:
        if not page.text:
            continue

        start = 0
        chunk_index = 0

        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            text = page.text[start:end]

            chunks.append(
                TextChunk(
                    chunk_id=f"page-{page.page_number}-chunk-{chunk_index}",
                    page_number=page.page_number,
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )

            if end == len(page.text):
                break

            start = end - overlap
            chunk_index += 1

    return chunks
