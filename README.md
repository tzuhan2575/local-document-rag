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
End-to-end PDF ingestion, persistent semantic retrieval, source-grounded prompting, and provider-independent answer orchestration implemented with automated tests.
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
