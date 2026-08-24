import pytest
from fastapi.testclient import TestClient

from local_document_rag.api import create_app
from local_document_rag.assistant import AnswerResult
from local_document_rag.chunker import TextChunk
from local_document_rag.retriever import RetrievedChunk


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def ask(self, question, top_k=3):
        self.calls.append((question, top_k))
        source = RetrievedChunk(
            chunk=TextChunk(
                chunk_id="chunk-0",
                page_number=2,
                text="Citation metadata is tested on page two.",
                start_char=0,
                end_char=40,
                source="sample.pdf",
            ),
            score=0.92,
        )
        return AnswerResult(
            answer="It is tested on page two. [Source 1]",
            sources=(source,),
        )


def test_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "local-document-rag",
    }


def test_query_endpoint_returns_answer_and_sources():
    assistant = FakeAssistant()
    client = TestClient(create_app(assistant))

    response = client.post(
        "/query",
        json={
            "question": "  Where is it tested?  ",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    assert assistant.calls == [("Where is it tested?", 2)]
    assert response.json() == {
        "answer": "It is tested on page two. [Source 1]",
        "sources": [
            {
                "source": "sample.pdf",
                "page_number": 2,
                "score": 0.92,
                "text": "Citation metadata is tested on page two.",
            }
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   ", "top_k": 2},
        {"question": "valid question", "top_k": 0},
    ],
)
def test_query_endpoint_rejects_invalid_request(payload):
    response = TestClient(create_app(FakeAssistant())).post(
        "/query",
        json=payload,
    )

    assert response.status_code == 422


def test_query_endpoint_returns_503_when_assistant_is_not_configured():
    response = TestClient(create_app()).post(
        "/query",
        json={"question": "question"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "RAG assistant is not configured"
    }
