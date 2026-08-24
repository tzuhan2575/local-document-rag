"""Production dependency wiring for the FastAPI application."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from qdrant_client import QdrantClient

from local_document_rag.api import create_app
from local_document_rag.assistant import RAGAssistant
from local_document_rag.embedder import LocalEmbedder
from local_document_rag.indexer import DocumentIndexer
from local_document_rag.openai_llm import OpenAILLMClient
from local_document_rag.qdrant_store import QdrantChunkStore
from local_document_rag.settings import AppSettings, load_settings


def create_runtime_app(
    settings: AppSettings | None = None,
    *,
    embedder: Any | None = None,
    qdrant_client: QdrantClient | None = None,
    llm: Any | None = None,
) -> FastAPI:
    """Build the runnable API with local RAG dependencies."""

    runtime_settings = settings or load_settings()

    if embedder is None:
        embedder = LocalEmbedder(
            model_name=runtime_settings.embedding_model
        )

    if qdrant_client is None:
        qdrant_client = QdrantClient(
            path=str(runtime_settings.qdrant_path)
        )

    store = QdrantChunkStore(
        client=qdrant_client,
        collection_name=runtime_settings.qdrant_collection,
        vector_dimension=embedder.dimension,
    )
    indexer = DocumentIndexer(embedder, store)

    assistant = None

    if runtime_settings.openai_enabled:
        if llm is None:
            llm = OpenAILLMClient(
                model=runtime_settings.openai_model
            )

        assistant = RAGAssistant(
            embedder=embedder,
            store=store,
            llm=llm,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            qdrant_client.close()

    return create_app(
        assistant=assistant,
        indexer=indexer,
        lifespan=lifespan,
    )
