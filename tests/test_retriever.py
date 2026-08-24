import numpy as np

from local_document_rag.chunker import TextChunk
from local_document_rag.retriever import InMemoryRetriever


class FakeEmbedder:
    def __init__(self):
        self.embedded_texts = None
        self.queries = []

    def embed_texts(self, texts):
        self.embedded_texts = list(texts)

        if not texts:
            return np.empty((0, 2), dtype=np.float32)

        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def embed_query(self, query):
        self.queries.append(query)
        return np.array([1.0, 0.0], dtype=np.float32)


def make_chunk(chunk_id, page_number, text):
    return TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_retriever_embeds_chunks_and_returns_ranked_original_chunks():
    chunks = [
        make_chunk("chunk-a", 1, "relevant text"),
        make_chunk("chunk-b", 2, "unrelated text"),
    ]
    embedder = FakeEmbedder()

    retriever = InMemoryRetriever(chunks, embedder)
    results = retriever.search("retrieval question", top_k=1)

    assert embedder.embedded_texts == ["relevant text", "unrelated text"]
    assert embedder.queries == ["retrieval question"]
    assert len(results) == 1
    assert results[0].chunk is chunks[0]
    assert results[0].score == 1.0


def test_empty_retriever_returns_no_results():
    retriever = InMemoryRetriever([], FakeEmbedder())

    assert retriever.search("question") == []
