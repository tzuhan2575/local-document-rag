import numpy as np
import pytest
from qdrant_client import QdrantClient

from local_document_rag.chunker import TextChunk
from local_document_rag.qdrant_store import QdrantChunkStore


def make_chunk(chunk_id, page_number, text):
    return TextChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def make_store():
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(
        client=client,
        collection_name="test_chunks",
        vector_dimension=2,
    )
    return client, store


def test_add_and_search_preserve_chunk_payload_and_ranking():
    _, store = make_store()
    chunks = [
        make_chunk("chunk-a", 1, "relevant text"),
        make_chunk("chunk-b", 2, "unrelated text"),
    ]
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    store.add(chunks, vectors)
    results = store.search(np.array([1.0, 0.0]), top_k=1)

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].score == pytest.approx(1.0)


def test_adding_same_chunk_twice_is_idempotent():
    client, store = make_store()
    chunk = make_chunk("chunk-a", 1, "same text")
    vector = np.array([[1.0, 0.0]], dtype=np.float32)

    store.add([chunk], vector)
    store.add([chunk], vector)

    count = client.count(
        collection_name="test_chunks",
        exact=True,
    ).count

    assert count == 1


def test_add_rejects_invalid_vector_input():
    _, store = make_store()
    chunk = make_chunk("chunk-a", 1, "text")

    with pytest.raises(ValueError, match="two-dimensional"):
        store.add([chunk], np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="same length"):
        store.add([chunk], np.empty((0, 2)))

    with pytest.raises(ValueError, match="dimension"):
        store.add([chunk], np.array([[1.0, 0.0, 0.0]]))


def test_search_rejects_invalid_query_input():
    _, store = make_store()

    with pytest.raises(ValueError, match="top_k"):
        store.search(np.array([1.0, 0.0]), top_k=0)

    with pytest.raises(ValueError, match="one-dimensional"):
        store.search(np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="dimension"):
        store.search(np.array([1.0, 0.0, 0.0]))
