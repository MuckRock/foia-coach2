"""Tests for hybrid_search()."""
import pytest

from apps.records.search import hybrid_search
from apps.records.tests.factories import RetentionRecordFactory, SourceDocumentFactory

DUMMY_EMBEDDING = [0.1] * 1536


@pytest.mark.django_db
class TestHybridSearch:
    def test_returns_results(self):
        doc = SourceDocumentFactory(jurisdiction="Colorado")
        RetentionRecordFactory(
            source_document=doc,
            record_title="Building Permits",
            record_description="Construction permits for buildings.",
            embedding=DUMMY_EMBEDDING,
        )
        RetentionRecordFactory(
            source_document=doc,
            record_title="Budget Documents",
            record_description="Annual budget worksheets.",
            embedding=DUMMY_EMBEDDING,
        )
        RetentionRecordFactory(
            source_document=doc,
            record_title="Audit Reports",
            record_description="Independent audit findings.",
            embedding=DUMMY_EMBEDDING,
        )
        results = hybrid_search("building permits", DUMMY_EMBEDDING)
        assert len(results) >= 1

    def test_result_has_expected_keys(self):
        doc = SourceDocumentFactory(jurisdiction="Colorado")
        RetentionRecordFactory(
            source_document=doc,
            record_title="Building Permits",
            record_description="Construction permits for buildings.",
            embedding=DUMMY_EMBEDDING,
        )
        results = hybrid_search("building permits", DUMMY_EMBEDDING)
        assert len(results) >= 1
        keys = results[0].keys()
        for expected_key in ["record_title", "rrf_score", "record_number", "minimum_retention_period"]:
            assert expected_key in keys

    def test_excludes_cross_references_by_default(self):
        doc = SourceDocumentFactory(jurisdiction="Colorado")
        RetentionRecordFactory(
            source_document=doc,
            record_title="Building Permits",
            record_description="Construction permits for buildings.",
            embedding=DUMMY_EMBEDDING,
            is_cross_reference=False,
        )
        cross_ref = RetentionRecordFactory(
            source_document=doc,
            record_title="See Schedule 7",
            record_description="See Schedule 7 for building records.",
            embedding=DUMMY_EMBEDDING,
            is_cross_reference=True,
        )
        results = hybrid_search("building", DUMMY_EMBEDDING, exclude_cross_references=True)
        result_ids = [r["id"] for r in results]
        assert cross_ref.id not in result_ids

    def test_includes_cross_references_when_flag_false(self):
        doc = SourceDocumentFactory(jurisdiction="Colorado")
        cross_ref = RetentionRecordFactory(
            source_document=doc,
            record_title="See Schedule 7 Building Records",
            record_description="See Schedule 7 for building records.",
            embedding=DUMMY_EMBEDDING,
            is_cross_reference=True,
        )
        results = hybrid_search(
            "building schedule",
            DUMMY_EMBEDDING,
            exclude_cross_references=False,
        )
        result_ids = [r["id"] for r in results]
        assert cross_ref.id in result_ids

    def test_respects_limit(self):
        doc = SourceDocumentFactory(jurisdiction="Colorado")
        for i in range(5):
            RetentionRecordFactory(
                source_document=doc,
                record_title=f"Record {i} permits building",
                record_description=f"Description for record {i} about permits.",
                embedding=DUMMY_EMBEDDING,
            )
        results = hybrid_search("permits", DUMMY_EMBEDDING, limit=2)
        assert len(results) <= 2

    def test_jurisdiction_filter(self):
        co_doc = SourceDocumentFactory(jurisdiction="Colorado")
        ca_doc = SourceDocumentFactory(jurisdiction="California")
        co_record = RetentionRecordFactory(
            source_document=co_doc,
            record_title="Colorado Building Permits",
            record_description="Building permits for Colorado.",
            embedding=DUMMY_EMBEDDING,
        )
        ca_record = RetentionRecordFactory(
            source_document=ca_doc,
            record_title="California Building Permits",
            record_description="Building permits for California.",
            embedding=DUMMY_EMBEDDING,
        )
        results = hybrid_search("building permits", DUMMY_EMBEDDING, jurisdiction="Colorado")
        result_ids = [r["id"] for r in results]
        assert co_record.id in result_ids
        assert ca_record.id not in result_ids
