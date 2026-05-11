"""Tests for chunk_pdf() chunking logic."""
from unittest.mock import MagicMock, patch

import pytest

from apps.records.management.commands.import_supporting_documents import chunk_pdf


def make_mock_reader(page_texts):
    """Build a mock PdfReader whose .pages yields mock pages with extract_text()."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.pages = pages
    return reader


@patch("apps.records.management.commands.import_supporting_documents.PdfReader")
def test_skips_near_blank_page(mock_reader_cls):
    """Pages with fewer than min_tokens words are skipped."""
    mock_reader_cls.return_value = make_mock_reader(["too short"])
    chunks = list(chunk_pdf("fake.pdf", min_tokens=50))
    assert chunks == []


@patch("apps.records.management.commands.import_supporting_documents.PdfReader")
def test_single_page_under_limit(mock_reader_cls):
    """A single page under max_tokens produces one chunk with correct page_number."""
    words = " ".join(["word"] * 60)
    mock_reader_cls.return_value = make_mock_reader([words])
    chunks = list(chunk_pdf("fake.pdf", max_tokens=800, min_tokens=50))
    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0


@patch("apps.records.management.commands.import_supporting_documents.PdfReader")
def test_long_page_produces_multiple_chunks(mock_reader_cls):
    """A page longer than max_tokens is split into multiple chunks each ≤ max_tokens."""
    words = " ".join(["word"] * 900)
    mock_reader_cls.return_value = make_mock_reader([words])
    chunks = list(chunk_pdf("fake.pdf", max_tokens=400, overlap_tokens=50, min_tokens=50))
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] <= 400


@patch("apps.records.management.commands.import_supporting_documents.PdfReader")
def test_overlap_carried_across_pages(mock_reader_cls):
    """Words from page 1 appear in page 2's first chunk due to overlap."""
    page1_words = ["alpha"] * 60
    page2_words = ["beta"] * 60
    mock_reader_cls.return_value = make_mock_reader([
        " ".join(page1_words),
        " ".join(page2_words),
    ])
    chunks = list(chunk_pdf("fake.pdf", max_tokens=800, overlap_tokens=10, min_tokens=5))
    # page 2 chunk text should contain "alpha" from the overlap
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert any("alpha" in c["text"] for c in page2_chunks)


@patch("apps.records.management.commands.import_supporting_documents.PdfReader")
def test_none_page_text_is_skipped(mock_reader_cls):
    """Pages where extract_text() returns None are skipped without error."""
    mock_reader_cls.return_value = make_mock_reader([None, " ".join(["word"] * 60)])
    chunks = list(chunk_pdf("fake.pdf", min_tokens=50))
    # Only page 2 should produce a chunk
    assert all(c["page_number"] == 2 for c in chunks)
