import numpy as np
import pytest

from local_document_rag.assistant import RAGAssistant
from local_document_rag.chunker import TextChunk
from local_document_rag.retriever import RetrievedChunk


def make_source():
    return RetrievedChunk(
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


class FakeEmbedder:
    def __init__(self):
        self.questions = []

    def embed_query(self, query):
        self.questions.append(query)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, query_vector, top_k=3):
        self.calls.append((query_vector, top_k))
        return [make_source()]


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def test_assistant_coordinates_retrieval_prompting_and_generation():
    embedder = FakeEmbedder()
    store = FakeStore()
    llm = FakeLLM("Supported answer. [Source 1]")
    assistant = RAGAssistant(embedder, store, llm)

    result = assistant.ask("Where is it tested?", top_k=2)

    assert embedder.questions == ["Where is it tested?"]
    assert store.calls[0][1] == 2
    assert "[Source 1]" in llm.prompts[0].user
    assert "sample.pdf" in llm.prompts[0].user
    assert result.answer == "Supported answer. [Source 1]"
    assert result.sources == (make_source(),)


def test_assistant_rejects_empty_llm_answer():
    assistant = RAGAssistant(
        FakeEmbedder(),
        FakeStore(),
        FakeLLM("   "),
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        assistant.ask("question")
