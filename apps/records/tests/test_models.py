import pytest

from apps.records.models import SystemPrompt
from apps.records.tests.factories import (
    RetentionRecordFactory,
    SourceDocumentFactory,
    SystemPromptFactory,
)


@pytest.mark.django_db
class TestSystemPrompt:
    def test_only_one_active(self):
        p1 = SystemPromptFactory(is_active=True)
        p2 = SystemPromptFactory(is_active=True)
        p1.refresh_from_db()
        assert not p1.is_active
        assert p2.is_active

    def test_get_active_raises_when_none(self):
        SystemPrompt.objects.all().delete()
        with pytest.raises(RuntimeError, match="No active system prompt"):
            SystemPrompt.get_active()

    def test_get_active_returns_content(self):
        prompt = SystemPromptFactory(is_active=True, content="Hello world")
        assert SystemPrompt.get_active() == "Hello world"

    def test_str_active(self):
        p = SystemPromptFactory(name="My Prompt", is_active=True)
        assert str(p) == "My Prompt (active)"

    def test_str_inactive(self):
        p = SystemPromptFactory(name="My Prompt", is_active=False)
        assert str(p) == "My Prompt"


@pytest.mark.django_db
class TestSourceDocument:
    def test_str(self):
        doc = SourceDocumentFactory(document_title="SCHEDULE NO. 1 - BUILDING RECORDS")
        assert str(doc) == "SCHEDULE NO. 1 - BUILDING RECORDS"


@pytest.mark.django_db
class TestRetentionRecord:
    def test_str(self):
        record = RetentionRecordFactory(record_number="1.50", record_title="Building Permits")
        assert str(record) == "1.50 — Building Permits"

    def test_to_chunk_text_includes_all_fields(self):
        record = RetentionRecordFactory(
            record_title="Building Permits",
            record_description="Permits for construction",
            minimum_retention_period="20 years",
            custodian_requirement="Destroy after period",
            regulatory_citations="C.R.S. 24-80-101",
        )
        text = record.to_chunk_text()
        assert "Building Permits" in text
        assert "Permits for construction" in text
        assert "20 years" in text
        assert "Destroy after period" in text
        assert "C.R.S. 24-80-101" in text
        assert record.source_document.document_title in text

    def test_to_chunk_text_skips_empty_optional_fields(self):
        record = RetentionRecordFactory(
            custodian_requirement="",
            regulatory_citations="",
        )
        text = record.to_chunk_text()
        assert "Disposition:" not in text
        assert "Legal citations:" not in text
