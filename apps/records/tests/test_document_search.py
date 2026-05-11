"""Tests for document_search() vector similarity function."""
import pytest
from asgiref.sync import sync_to_async

from apps.records.search import document_search
from apps.records.tests.factories import DocumentChunkFactory, SupportingDocumentFactory

DUMMY_EMBEDDING = [0.1] * 1536


@pytest.mark.django_db(transaction=True)
class TestDocumentSearch:
    def _make_chunk_with_embedding(self):
        chunk = DocumentChunkFactory(embedding=None)
        chunk.embedding = DUMMY_EMBEDDING
        chunk.save(update_fields=["embedding"])
        return chunk

    def test_returns_results(self):
        self._make_chunk_with_embedding()
        results = document_search(query_embedding=DUMMY_EMBEDDING, limit=10)
        assert len(results) >= 1

    def test_result_has_expected_keys(self):
        self._make_chunk_with_embedding()
        results = document_search(query_embedding=DUMMY_EMBEDDING, limit=10)
        assert len(results) >= 1
        row = results[0]
        for key in ("id", "text", "page_number", "document_title", "similarity_score"):
            assert key in row, f"Missing key: {key}"

    def test_respects_limit(self):
        for _ in range(5):
            self._make_chunk_with_embedding()
        results = document_search(query_embedding=DUMMY_EMBEDDING, limit=2)
        assert len(results) <= 2

    def test_skips_chunks_without_embedding(self):
        # chunk with no embedding
        DocumentChunkFactory(embedding=None)
        # Ensure there are no embedded chunks
        from apps.records.models import DocumentChunk
        DocumentChunk.objects.filter(embedding__isnull=False).delete()

        results = document_search(query_embedding=DUMMY_EMBEDDING, limit=10)
        assert results == []
