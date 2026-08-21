import pytest

from local_document_rag.pdf_loader import extract_pdf_pages


def test_missing_pdf_raises_file_not_found(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_pages(missing_pdf)


def test_non_pdf_file_is_rejected(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF file"):
        extract_pdf_pages(text_file)
