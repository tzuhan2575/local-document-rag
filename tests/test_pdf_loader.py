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


def test_extract_pdf_pages_includes_filename_source(tmp_path, monkeypatch):
    import local_document_rag.pdf_loader as pdf_loader_module

    class FakePage:
        def extract_text(self):
            return "page text"

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]

    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    monkeypatch.setattr(pdf_loader_module, "PdfReader", FakeReader)

    pages = pdf_loader_module.extract_pdf_pages(pdf_path)

    assert pages == [
        pdf_loader_module.PageText(
            page_number=1,
            text="page text",
            source="document.pdf",
        )
    ]
