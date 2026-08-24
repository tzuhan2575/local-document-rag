"""FastAPI application for the Local Document RAG service."""

from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from local_document_rag.assistant import AnswerResult


class AnsweringAssistant(Protocol):
    def ask(self, question: str, top_k: int = 3) -> AnswerResult:
        ...


class HealthResponse(BaseModel):
    status: str
    service: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        clean_value = value.strip()

        if not clean_value:
            raise ValueError("question must not be blank")

        return clean_value


class SourceResponse(BaseModel):
    source: str
    page_number: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


def create_app(
    assistant: AnsweringAssistant | None = None,
) -> FastAPI:
    """Create the API, optionally injecting a configured RAG assistant."""

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

    @application.post(
        "/query",
        response_model=QueryResponse,
        tags=["rag"],
    )
    def query(request: QueryRequest) -> QueryResponse:
        if assistant is None:
            raise HTTPException(
                status_code=503,
                detail="RAG assistant is not configured",
            )

        result = assistant.ask(
            request.question,
            top_k=request.top_k,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceResponse(
                    source=item.chunk.source,
                    page_number=item.chunk.page_number,
                    score=item.score,
                    text=item.chunk.text,
                )
                for item in result.sources
            ],
        )

    return application


app = create_app()
