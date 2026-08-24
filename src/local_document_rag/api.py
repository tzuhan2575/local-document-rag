"""FastAPI application for the Local Document RAG service."""

from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator
from pypdf.errors import PdfReadError

from local_document_rag.assistant import AnswerResult
from local_document_rag.indexer import IndexingResult


class AnsweringAssistant(Protocol):
    def ask(self, question: str, top_k: int = 3) -> AnswerResult:
        ...


class DocumentIndexingService(Protocol):
    def index_pdf(self, pdf_path: str | Path) -> IndexingResult:
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


class IndexResponse(BaseModel):
    source: str
    page_count: int
    chunk_count: int


def create_app(
    assistant: AnsweringAssistant | None = None,
    indexer: DocumentIndexingService | None = None,
    lifespan: Any | None = None,
) -> FastAPI:
    """Create the API with optional RAG dependencies."""

    application = FastAPI(
        title="Local Document RAG API",
        version="0.1.0",
        lifespan=lifespan,
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
        "/documents",
        response_model=IndexResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["rag"],
    )
    def index_document(
        file: Annotated[
            UploadFile,
            File(description="PDF document to index"),
        ],
    ) -> IndexResponse:
        if indexer is None:
            raise HTTPException(
                status_code=503,
                detail="Document indexer is not configured",
            )

        filename = Path(file.filename or "").name

        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported",
            )

        with TemporaryDirectory(
            prefix="local-document-rag-",
        ) as temporary_directory:
            pdf_path = Path(temporary_directory) / filename

            with pdf_path.open("wb") as destination:
                copyfileobj(file.file, destination)

            try:
                result = indexer.index_pdf(pdf_path)
            except PdfReadError as error:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a readable PDF",
                ) from error

        return IndexResponse(
            source=result.source,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
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
