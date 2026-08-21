import pytest

from local_document_rag.chunker import chunk_pages
from local_document_rag.pdf_loader import PageText


def test_chunks_include_expected_overlap_and_positions():
    pages = [PageText(page_number=3, text="abcdefghij")]

    chunks = chunk_pages(pages, chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.page_number for chunk in chunks] == [3, 3]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 6),
        (4, 10),
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        "page-3-chunk-0",
        "page-3-chunk-1",
    ]


def test_empty_pages_are_skipped():
    pages = [PageText(page_number=1, text="")]

    assert chunk_pages(pages) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)
