"""Provider-independent RAG answer orchestration."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from local_document_rag.prompting import GroundedPrompt, build_grounded_prompt
from local_document_rag.retriever import RetrievedChunk


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> NDArray[np.float32]:
        ...


class SearchStore(Protocol):
    def search(
        self,
        query_vector: NDArray[np.floating],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        ...


class LLMClient(Protocol):
    def generate(self, prompt: GroundedPrompt) -> str:
        ...


@dataclass(frozen=True)
class AnswerResult:
    """Generated answer together with its retrieved evidence."""

    answer: str
    sources: tuple[RetrievedChunk, ...]


class RAGAssistant:
    """Retrieve evidence, build a grounded prompt, and call an LLM."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        store: SearchStore,
        llm: LLMClient,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def ask(self, question: str, top_k: int = 3) -> AnswerResult:
        query_vector = self._embedder.embed_query(question)
        sources = self._store.search(query_vector, top_k=top_k)
        prompt = build_grounded_prompt(question, sources)
        answer = self._llm.generate(prompt).strip()

        if not answer:
            raise RuntimeError("LLM returned an empty answer")

        return AnswerResult(
            answer=answer,
            sources=tuple(sources),
        )
