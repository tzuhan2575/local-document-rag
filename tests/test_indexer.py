import numpy as np

import local_document_rag.indexer as indexer_module
from local_document_rag.indexer import DocumentIndexer
from local_document_rag.pdf_loader import PageText


class FakeEmbedder:
    def __init__(self):
        self.texts = None

    def embed_texts(self, texts):
        self.texts = list(texts)
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeStore:
    def __init__(self):
        self.chunks = None
        self.vectors = None

    def add(self, chunks, vectors):
        self.chunks = list(chunks)
        self.vectors = vectors


def test_index_pdf_coordinates_extraction_chunking_embedding_and_storage(
    monkeypatch,
):
    pages = [
        PageText(
            page_number=1,
            text="abcdefghij",
            source="document.pdf",
        )
    ]
    monkeypatch.setattr(
        indexer_module,
        "extract_pdf_pages",
        lambda _: pages,
    )
    embedder = FakeEmbedder()
    store = FakeStore()
    indexer = DocumentIndexer(embedder, store)

    result = indexer.index_pdf(
        "document.pdf",
        chunk_size=6,
        overlap=2,
    )

    assert result.source == "document.pdf"
    assert result.page_count == 1
    assert result.chunk_count == 2
    assert embedder.texts == ["abcdef", "efghij"]
    assert [chunk.source for chunk in store.chunks] == [
        "document.pdf",
        "document.pdf",
    ]
    assert store.vectors.shape == (2, 2)
