"""Tests for management commands."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from django.core.management import call_command

from apps.records.models import RetentionRecord, SourceDocument
from apps.records.tests.factories import RetentionRecordFactory, SourceDocumentFactory

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JSON = FIXTURES_DIR / "sample_records.json"


@pytest.mark.django_db
class TestImportRetentionRecords:
    def test_import_creates_source_documents(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        assert SourceDocument.objects.count() == 2

    def test_import_creates_retention_records(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        assert RetentionRecord.objects.count() == 5

    def test_import_idempotent(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        assert SourceDocument.objects.count() == 2
        assert RetentionRecord.objects.count() == 5

    def test_import_sets_cross_reference_flag(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        cross_ref = RetentionRecord.objects.get(record_number="1.20")
        assert cross_ref.is_cross_reference is True
        # Others should not be cross-references
        normal = RetentionRecord.objects.get(record_number="1.10")
        assert normal.is_cross_reference is False

    def test_import_sets_permanent_flag(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        permanent = RetentionRecord.objects.get(record_number="1.30")
        assert permanent.is_permanent is True
        normal = RetentionRecord.objects.get(record_number="1.10")
        assert normal.is_permanent is False

    def test_import_updates_record_count(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        schedule1 = SourceDocument.objects.get(
            document_title="SCHEDULE NO. 1 - BUILDING AND STRUCTURE RECORDS (Colorado Special Districts)"
        )
        assert schedule1.record_count == 3

    def test_import_maps_field_names(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        record = RetentionRecord.objects.get(record_number="1.10")
        assert record.custodian_requirement == "Destroy after retention period."
        assert record.regulatory_citations == "C.R.S. 24-80-101"

    def test_import_parses_schedule_number(self):
        call_command("import_retention_records", str(SAMPLE_JSON), "--jurisdiction", "Colorado")
        doc = SourceDocument.objects.get(
            document_title="SCHEDULE NO. 1 - BUILDING AND STRUCTURE RECORDS (Colorado Special Districts)"
        )
        assert doc.schedule_number == "1"
        assert doc.entity_type == "Colorado Special Districts"

    def test_import_with_custom_filename(self, tmp_path):
        """Test that --filename sets the source document filename."""
        tmp_file = tmp_path / "records.json"
        tmp_file.write_text(json.dumps([{
            "record_number": "1.10",
            "record_title": "Test Record",
            "record_description": "A test record.",
            "record_custodian_preservation_destruction_requirement": "",
            "minimum_retention_period": "5 years",
            "regulatory_citation_statutes_rules_notations": "",
            "page_number": 1,
            "document_title": "SCHEDULE NO. 1 - TEST RECORDS (Colorado Special Districts)",
        }]))
        call_command(
            "import_retention_records",
            str(tmp_file),
            "--jurisdiction", "Colorado",
            "--filename", "Custom_Name.pdf",
        )
        doc = SourceDocument.objects.get(schedule_number="1")
        assert doc.filename == "Custom_Name.pdf"


@pytest.mark.django_db
class TestGenerateEmbeddings:
    def _make_embedding_response(self, texts):
        """Build a mock OpenAI embeddings response."""
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=[0.1] * 1536)
            for _ in texts
        ]
        return mock_resp

    @patch("apps.records.management.commands.generate_embeddings.openai.OpenAI")
    def test_generate_embeddings_skips_already_embedded(self, mock_openai_cls):
        record = RetentionRecordFactory(embedding=[0.1] * 1536)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        call_command("generate_embeddings")

        mock_client.embeddings.create.assert_not_called()

    @patch("apps.records.management.commands.generate_embeddings.openai.OpenAI")
    def test_generate_embeddings_skips_cross_references(self, mock_openai_cls):
        RetentionRecordFactory(is_cross_reference=True, embedding=None)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        call_command("generate_embeddings")

        mock_client.embeddings.create.assert_not_called()

    @patch("apps.records.management.commands.generate_embeddings.openai.OpenAI")
    def test_generate_embeddings_stores_vector(self, mock_openai_cls):
        record = RetentionRecordFactory(embedding=None)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_embedding_response([record])
        mock_openai_cls.return_value = mock_client

        call_command("generate_embeddings")

        record.refresh_from_db()
        assert record.embedding is not None

    @patch("apps.records.management.commands.generate_embeddings.openai.OpenAI")
    def test_generate_embeddings_force_flag_re_embeds(self, mock_openai_cls):
        record = RetentionRecordFactory(embedding=[0.1] * 1536)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_embedding_response([record])
        mock_openai_cls.return_value = mock_client

        call_command("generate_embeddings", force=True)

        mock_client.embeddings.create.assert_called_once()

    @patch("apps.records.management.commands.generate_embeddings.openai.OpenAI")
    def test_generate_embeddings_batches_100_at_a_time(self, mock_openai_cls):
        records = RetentionRecordFactory.create_batch(150, embedding=None)
        mock_client = MagicMock()

        def side_effect(model, input):
            return self._make_embedding_response(input)

        mock_client.embeddings.create.side_effect = side_effect
        mock_openai_cls.return_value = mock_client

        call_command("generate_embeddings")

        assert mock_client.embeddings.create.call_count == 2
        first_call_inputs = mock_client.embeddings.create.call_args_list[0][1]["input"]
        assert len(first_call_inputs) <= 100
