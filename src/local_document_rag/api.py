"""FastAPI application for the Local Document RAG service."""

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


def create_app() -> FastAPI:
    """Create the API application without loading ML models."""

    application = FastAPI(
        title="Local Document RAG API",
        version="0.1.0",
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="local-document-rag",
        )

    return application


app = create_app()
