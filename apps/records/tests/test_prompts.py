"""Tests for prompt assembly functions."""
import pytest
from asgiref.sync import sync_to_async

from apps.records.prompts import build_messages, format_retrieved_records
from apps.records.tests.factories import SystemPromptFactory


def make_record(**kwargs):
    defaults = {
        "id": 1,
        "record_number": "1.10",
        "record_title": "Building Permits",
        "record_description": "Construction permits for buildings.",
        "minimum_retention_period": "20 years",
        "custodian_requirement": "Destroy after period.",
        "regulatory_citations": "C.R.S. 24-80-101",
        "page_number": 5,
        "document_title": "SCHEDULE NO. 1 - BUILDING RECORDS",
        "is_permanent": False,
        "jurisdiction": "Colorado",
        "entity_type": "Special Districts",
        "rrf_score": 0.9,
    }
    defaults.update(kwargs)
    return defaults


class TestFormatRetrievedRecords:
    def test_empty_returns_no_results_string(self):
        result = format_retrieved_records([])
        assert "No relevant" in result

    def test_includes_title_and_period(self):
        record = make_record()
        result = format_retrieved_records([record])
        assert "Building Permits" in result
        assert "20 years" in result

    def test_skips_empty_custodian(self):
        record = make_record(custodian_requirement="")
        result = format_retrieved_records([record])
        assert "Disposition:" not in result

    def test_skips_empty_citations(self):
        record = make_record(regulatory_citations="")
        result = format_retrieved_records([record])
        assert "Citations:" not in result

    def test_includes_source_and_page(self):
        record = make_record()
        result = format_retrieved_records([record])
        assert "SCHEDULE NO. 1 - BUILDING RECORDS" in result
        assert "page 5" in result


@pytest.mark.django_db(transaction=True)
class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_structure(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are a helpful assistant.")
        messages = await build_messages("How long for building permits?", [], [])
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_context_injected(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are a helpful assistant.")
        records = [make_record()]
        messages = await build_messages("How long for building permits?", records, [])
        system_messages = [m for m in messages if m["role"] == "system"]
        combined = " ".join(m["content"] for m in system_messages)
        assert "Building Permits" in combined

    @pytest.mark.asyncio
    async def test_excludes_duplicate_user_message(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are a helpful assistant.")
        user_msg = "How long for building permits?"
        history = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "20 years."},
            {"role": "user", "content": user_msg},
        ]
        messages = await build_messages(user_msg, [], history)
        user_messages = [m for m in messages if m["role"] == "user"]
        # The duplicate should be deduplicated; only one user message with that content
        assert user_messages.count({"role": "user", "content": user_msg}) == 1

    @pytest.mark.asyncio
    async def test_last_message_is_user(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are helpful.")
        messages = await build_messages("What is the retention period?", [], [])
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is the retention period?"
