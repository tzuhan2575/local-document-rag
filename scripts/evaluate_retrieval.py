"""Run retrieval evaluation on a labeled JSON dataset."""

import argparse
import json
from pathlib import Path

from local_document_rag.chunker import chunk_pages
from local_document_rag.embedder import (
    DEFAULT_MODEL_NAME,
    LocalEmbedder,
)
from local_document_rag.evaluation import (
    RetrievalExample,
    evaluate_retrieval,
)
from local_document_rag.pdf_loader import extract_pdf_pages
from local_document_rag.retriever import InMemoryRetriever


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval quality."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the labeled evaluation JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per question.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset.resolve()
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )
    document_path = (
        dataset_path.parent / dataset["document"]
    ).resolve()

    pages = extract_pdf_pages(document_path)
    chunks = chunk_pages(
        pages,
        chunk_size=int(dataset["chunk_size"]),
        overlap=int(dataset["overlap"]),
    )

    actual_chunk_ids = {chunk.chunk_id for chunk in chunks}
    examples = [
        RetrievalExample(
            question=item["question"],
            relevant_chunk_ids=frozenset(
                item["relevant_chunk_ids"]
            ),
        )
        for item in dataset["examples"]
    ]
    labeled_chunk_ids = {
        chunk_id
        for example in examples
        for chunk_id in example.relevant_chunk_ids
    }

    missing_ids = labeled_chunk_ids - actual_chunk_ids

    if missing_ids:
        raise ValueError(
            f"Dataset references missing chunk IDs: "
            f"{sorted(missing_ids)}"
        )

    embedder = LocalEmbedder()
    retriever = InMemoryRetriever(chunks, embedder)
    metrics = evaluate_retrieval(
        examples,
        retrieve=lambda question, top_k: retriever.search(
            question,
            top_k=top_k,
        ),
        top_k=args.top_k,
    )

    output = {
        "dataset": str(args.dataset),
        "document": dataset["document"],
        "embedding_model": DEFAULT_MODEL_NAME,
        "chunk_size": dataset["chunk_size"],
        "overlap": dataset["overlap"],
        "chunk_count": len(chunks),
        "query_count": metrics.evaluated_queries,
        "top_k": args.top_k,
        "recall_at_k": metrics.recall_at_k,
        "mean_reciprocal_rank": (
            metrics.mean_reciprocal_rank
        ),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
