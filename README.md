# Local Document RAG / AI Knowledge Assistant

A local document question-answering system that retrieves relevant document
content and generates answers grounded in the retrieved evidence.

## Problem

General-purpose language models do not automatically know the contents of a
user's private or local documents. This project explores how retrieval-augmented
generation (RAG) can provide relevant document context before generating an
answer.

## Planned Pipeline

```text
Document -> Text Extraction -> Chunking -> Embedding -> Vector Search
         -> Retrieved Context + Question -> LLM -> Grounded Answer
Project Goals
Build the RAG pipeline as understandable, testable components
Compare retrieval and chunking strategies
Return source-aware answers
Evaluate retrieval and answer quality
Provide a reproducible API-based application
Status
End-to-end local document RAG implemented with persistent retrieval, grounded prompting, OpenAI integration, FastAPI, and retrieval evaluation.
Environment
Python 3.12
macOS
Virtual environment: venv

## Embedding Baseline

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: 384
- Local inference with Sentence Transformers
- L2-normalized vectors for cosine similarity search

## Retrieval Baseline

- Exact cosine similarity search implemented with NumPy
- In-memory document embedding index
- Ranked retrieval with similarity scores and page metadata
- Current limitation: embeddings are recomputed after each restart

## Vector Database

- Qdrant local mode with on-disk persistence
- Cosine-distance collections with explicit vector dimensions
- Chunk text and page positions stored as payload metadata
- Stable UUID-based point IDs for idempotent upserts
- In-memory Qdrant integration tests require no Docker or network

## Ingestion Pipeline

```text
PDF
 -> page-level text with filename and page number
 -> overlapping chunks with character positions
 -> normalized 384-dimensional embeddings
 -> persistent Qdrant points with metadata payloads
The current system can index a PDF and retrieve ranked evidence with its
source filename, page number, character positions, and similarity score.

## Grounded Prompting

- Retrieved chunks are labeled as `[Source N]`
- Filename, page number, and retrieval score are included
- The LLM is instructed to answer only from retrieved evidence
- Missing context produces an explicit insufficient-evidence instruction
- Retrieved text is delimited and treated as evidence, not instructions

## Answer Generation Architecture

The generation layer depends on a small `LLMClient` protocol rather than a
specific provider SDK. This keeps retrieval and prompting testable without API
keys, network access, or usage cost.

`AnswerResult` returns both the generated answer and the exact retrieved chunks
used as evidence.

## OpenAI Integration

- Uses the OpenAI Responses API through the official Python SDK
- Maps the grounded system prompt to `instructions`
- Maps the question and retrieved context to `input`
- Limits generated output tokens
- Reads credentials through the SDK environment configuration
- Fully tested with a fake client; live API verification is pending billing setup

## API

The project exposes a FastAPI application with automatic OpenAPI documentation.

Run locally:

```bash
python -m uvicorn local_document_rag.runtime:create_runtime_app --factory --app-dir src --port 8000
Available endpoints:
GET /health — service health check
GET /docs — interactive Swagger UI

Example query request:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?","top_k":3}'
The default exported app does not initialize ML or API dependencies and returns
HTTP 503 for /query until a configured RAGAssistant is injected.

Upload and index a PDF:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@/absolute/path/to/document.pdf"
Uploaded files are copied into a temporary directory for extraction. The
temporary file is removed after indexing; persistent chunks, embeddings, and
source metadata remain in Qdrant.

## Runtime Configuration

Copy `.env.example` to `.env` only when local overrides are needed.

- `QDRANT_PATH` — local persistent vector database directory
- `QDRANT_COLLECTION` — Qdrant collection name
- `EMBEDDING_MODEL` — local Sentence Transformers model
- `OPENAI_MODEL` — OpenAI generation model
- `OPENAI_API_KEY` — optional secret; never commit this value

When `OPENAI_API_KEY` is absent, PDF indexing remains available and `/query`
returns HTTP 503. The application does not store the API key in its settings
object.

### Local Qdrant Concurrency Note

The current API keeps local Qdrant operations on the FastAPI event-loop thread
because its embedded SQLite storage is thread-affine. This is appropriate for
the current single-user local application. A production deployment should use a
Qdrant server or asynchronous storage boundary for concurrent workloads.

## Retrieval Evaluation

Retrieval is evaluated separately from answer generation to distinguish
retrieval failures from generation failures.

Current metrics:

- Recall@k — fraction of labeled relevant chunks retrieved in the top-k
- Mean Reciprocal Rank — average inverse rank of the first relevant chunk

The evaluation implementation supports multiple relevant chunks per question
and validates empty labels, invalid questions, and top-k configuration.

### Baseline Retrieval Result

| Dataset | Queries | Chunks | Top-k | Recall@k | MRR |
|---|---:|---:|---:|---:|---:|
| Synthetic two-page PDF | 2 | 2 | 1 | 1.000 | 1.000 |

Run the baseline:

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  evaluation/retrieval_examples.json \
  --top-k 1
This synthetic baseline is a pipeline sanity check, not evidence of general
retrieval quality. A meaningful experiment requires more documents, harder
questions, negative examples, and comparison across chunking configurations.
