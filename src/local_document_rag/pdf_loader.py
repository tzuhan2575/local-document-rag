"""PDF text extraction with page-level metadata."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page."""

    page_number: int
    text: str
    source: str = ""


def extract_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract text from each page of a text-based PDF."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.suffix or 'no extension'}")

    reader = PdfReader(path)
    pages: list[PageText] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            PageText(
                page_number=page_number,
                text=text.strip(),
                source=path.name,
            )
        )

    return pages
