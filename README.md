# Local Document RAG / AI Knowledge Assistant

[![CI](https://github.com/tzuhan2575/local-document-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tzuhan2575/local-document-rag/actions/workflows/ci.yml)

A local document question-answering system that indexes PDFs, retrieves relevant evidence, and builds source-grounded answers with filename and page citations.

## Problem

General-purpose language models do not automatically know the contents of private or local documents. This project implements retrieval-augmented generation (RAG) as explicit, testable components instead of hiding the pipeline behind a high-level framework.

## System Design

~~~text
PDF upload
  -> page-level text extraction
  -> overlapping text chunks
  -> normalized local embeddings
  -> persistent Qdrant vector store

User question
  -> query embedding
  -> semantic similarity search
  -> ranked evidence with filename and page
  -> grounded prompt
  -> optional OpenAI Responses API
  -> answer plus retrieved sources
~~~

## Current Features

- Page-level PDF extraction with filename and page metadata
- Configurable overlapping character chunking with position tracking
- Local 384-dimensional Sentence Transformers embeddings
- Exact NumPy retrieval baseline and persistent Qdrant storage
- FastAPI upload/query endpoints, evaluation metrics, and 58 automated tests

## Technology

| Component | Implementation |
|---|---|
| Language | Python 3.12 |
| PDF extraction | pypdf |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | Qdrant local mode |
| API | FastAPI + Uvicorn |
| Optional LLM | OpenAI Responses API |
| Testing | pytest + GitHub Actions |

## Quick Start

### Flexible development install

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
pytest -q
~~~

### Reproduce the verified environment

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps .
pytest -q
~~~

The lock file was verified with Python 3.12.9 on macOS x86_64. GitHub Actions also installs the project and runs the full test suite on Ubuntu.

## Configuration

Copy the public template only when local overrides are needed:

~~~bash
cp .env.example .env
~~~

| Variable | Purpose |
|---|---|
| `QDRANT_PATH` | Persistent local vector database directory |
| `QDRANT_COLLECTION` | Qdrant collection name |
| `EMBEDDING_MODEL` | Local Sentence Transformers model |
| `OPENAI_MODEL` | OpenAI generation model |
| `OPENAI_API_KEY` | Optional secret; never commit this value |

When `OPENAI_API_KEY` is absent, PDF indexing remains available and `/query` returns HTTP 503. Live OpenAI generation has not yet been verified because API billing is not configured; the adapter is tested with a fake Responses client.

## Run the API

~~~bash
python -m uvicorn \
  local_document_rag.runtime:create_runtime_app \
  --factory \
  --app-dir src \
  --port 8000
~~~

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Upload and index a PDF

~~~bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@/absolute/path/to/document.pdf"
~~~

### Ask a question

~~~bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the document say?","top_k":3}'
~~~

The query response contains the answer and the retrieved evidence text, source filename, page number, and similarity score.

## Retrieval Evaluation

Retrieval is evaluated separately from generation so retrieval failures can be distinguished from answer-generation failures.

- Recall@k: fraction of labeled relevant chunks retrieved in the top-k
- Mean Reciprocal Rank: inverse rank of the first relevant chunk, averaged across questions

Run the reproducible synthetic baseline:

~~~bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  evaluation/retrieval_examples.json \
  --top-k 1
~~~

| Dataset | Queries | Chunks | Top-k | Recall@k | MRR |
|---|---:|---:|---:|---:|---:|
| Synthetic two-page PDF | 2 | 2 | 1 | 1.000 | 1.000 |

This result is a pipeline sanity check, not evidence of general retrieval quality. A meaningful experiment requires more documents, harder questions, negative examples, and chunking comparisons.

## Engineering Decisions

- The RAG stages are separate modules with dependency injection for fast offline tests.
- Page and character metadata are retained from ingestion through retrieval.
- Stable UUID-based Qdrant point IDs make repeated indexing idempotent.
- Uploaded PDFs are removed after temporary processing; vectors and metadata persist.
- Secrets, local documents, model caches, and vector databases are excluded from Git.

## Current Limitations

- Text-based PDFs only; scanned documents require an OCR pipeline.
- Character chunking is a transparent baseline and may split semantic units.
- Embedded Qdrant is intended for the current single-user local application.
- OpenAI live generation remains pending API billing and credential setup.
- The evaluation corpus is intentionally small and synthetic.

For concurrent production traffic, use a Qdrant server or an asynchronous storage boundary instead of the embedded SQLite-backed local mode.

## Project Structure

~~~text
src/local_document_rag/   RAG components, API, settings, and runtime wiring
tests/                    Unit and local integration tests
evaluation/               Synthetic PDF, labels, and baseline results
scripts/                  Reproducible evaluation command
.github/workflows/        GitHub Actions CI
~~~

## License

This project is licensed under the [MIT License](LICENSE).
