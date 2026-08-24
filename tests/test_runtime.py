from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from local_document_rag.runtime import create_runtime_app
from local_document_rag.settings import AppSettings


class FakeEmbedder:
    dimension = 2

    def embed_query(self, _):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_texts(self, texts):
        return np.empty((len(texts), 2), dtype=np.float32)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "No indexed evidence is available."


class TrackingQdrantClient(QdrantClient):
    def __init__(self):
        super().__init__(":memory:")
        self.close_calls = 0

    def close(self, **kwargs):
        self.close_calls += 1
        return super().close(**kwargs)


def make_settings(openai_enabled):
    return AppSettings(
        qdrant_path=Path("unused"),
        qdrant_collection="documents",
        embedding_model="fake",
        openai_model="fake",
        openai_enabled=openai_enabled,
    )


def test_runtime_without_openai_keeps_query_disabled_and_closes_qdrant():
    qdrant = TrackingQdrantClient()
    app = create_runtime_app(
        settings=make_settings(openai_enabled=False),
        embedder=FakeEmbedder(),
        qdrant_client=qdrant,
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post(
            "/query",
            json={"question": "question"},
        ).status_code == 503

    assert qdrant.close_calls == 1


def test_runtime_with_openai_enables_query():
    qdrant = TrackingQdrantClient()
    llm = FakeLLM()
    app = create_runtime_app(
        settings=make_settings(openai_enabled=True),
        embedder=FakeEmbedder(),
        qdrant_client=qdrant,
        llm=llm,
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"question": "question"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "No indexed evidence is available.",
        "sources": [],
    }
    assert len(llm.prompts) == 1
    assert qdrant.close_calls == 1
