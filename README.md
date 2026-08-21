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
PDF extraction and overlapping character-based chunking implemented with source-position metadata and automated tests.
Environment
Python 3.12
macOS
Virtual environment: venv
