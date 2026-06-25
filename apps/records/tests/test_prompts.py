"""Tests for prompt assembly functions."""
import pytest
from asgiref.sync import sync_to_async

from apps.records.prompts import build_messages, format_retrieved_chunks, format_retrieved_records, postprocess_citations
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


def make_chunk(**kwargs):
    defaults = {
        "id": 1,
        "chunk_index": 0,
        "page_number": 3,
        "text": "You may request records by submitting a written request.",
        "token_count": 10,
        "document_title": "Colorado CORA Guide",
        "document_type": "FOIA Guide",
        "jurisdiction": "Colorado",
        "similarity_score": 0.85,
    }
    defaults.update(kwargs)
    return defaults


class TestFormatRetrievedChunks:
    def test_empty_returns_no_results_string(self):
        result, citation_map = format_retrieved_chunks([])
        assert "No relevant" in result
        assert citation_map == {}

    def test_includes_title_and_page(self):
        chunk = make_chunk()
        result, citation_map = format_retrieved_chunks([chunk])
        assert "Colorado CORA Guide" in result
        assert "page 3" in result

    def test_includes_text(self):
        chunk = make_chunk()
        result, _ = format_retrieved_chunks([chunk])
        assert "written request" in result

    def test_uses_namespaced_keys(self):
        chunk = make_chunk()
        result, citation_map = format_retrieved_chunks([chunk])
        assert "[G1]" in result
        assert "G1" in citation_map
        assert citation_map["G1"]["title"] == "Colorado CORA Guide"


class TestFormatRetrievedRecords:
    def test_empty_returns_no_results_string(self):
        result, citation_map = format_retrieved_records([])
        assert "No relevant" in result
        assert citation_map == {}

    def test_includes_title_and_period(self):
        record = make_record()
        result, _ = format_retrieved_records([record])
        assert "Building Permits" in result
        assert "20 years" in result

    def test_skips_empty_custodian(self):
        record = make_record(custodian_requirement="")
        result, _ = format_retrieved_records([record])
        assert "Disposition:" not in result

    def test_skips_empty_citations(self):
        record = make_record(regulatory_citations="")
        result, _ = format_retrieved_records([record])
        assert "Citations:" not in result

    def test_includes_source_and_page(self):
        record = make_record()
        result, _ = format_retrieved_records([record])
        assert "SCHEDULE NO. 1 - BUILDING RECORDS" in result
        assert "page 5" in result

    def test_uses_namespaced_keys(self):
        record = make_record()
        result, citation_map = format_retrieved_records([record])
        assert "[R1]" in result
        assert "R1" in citation_map
        assert "Building Permits" in citation_map["R1"]["label"]


@pytest.mark.django_db(transaction=True)
class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_structure(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are a helpful assistant.")
        messages, citation_map = await build_messages("How long for building permits?", [], [], [])
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_context_injected(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are a helpful assistant.")
        records = [make_record()]
        messages, citation_map = await build_messages("How long for building permits?", records, [], [])
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
        messages, _ = await build_messages(user_msg, [], [], history)
        user_messages = [m for m in messages if m["role"] == "user"]
        # The duplicate should be deduplicated; only one user message with that content
        assert user_messages.count({"role": "user", "content": user_msg}) == 1

    @pytest.mark.asyncio
    async def test_last_message_is_user(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are helpful.")
        messages, _ = await build_messages("What is the retention period?", [], [], [])
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is the retention period?"

    @pytest.mark.asyncio
    async def test_citation_map_populated(self):
        await sync_to_async(SystemPromptFactory)(is_active=True, content="You are helpful.")
        records = [make_record()]
        chunks = [make_chunk()]
        _, citation_map = await build_messages("test", records, chunks, [])
        assert "G1" in citation_map
        assert "R1" in citation_map


class TestPostprocessCitations:
    def test_renumbers_in_order_of_appearance(self):
        citation_map = {
            "G1": {"title": "Guide A", "url": "https://example.com/a", "page": 1},
            "G2": {"title": "Guide B", "url": "https://example.com/b", "page": 2},
            "R1": {"label": "Record X (Record 1.10)", "source_title": "Schedule 1", "url": "https://example.com/r1", "page": 5},
        }
        text = "Some fact [G2]. Another fact [R1]. More info [G1]."
        result = postprocess_citations(text, citation_map)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        # G2 appears first → [1], R1 second → [2], G1 third → [3]
        body = result.split("---")[0]
        assert body.index("[1]") < body.index("[2]") < body.index("[3]")

    def test_strips_llm_generated_footnotes(self):
        citation_map = {
            "G1": {"title": "Guide A", "url": "https://example.com/a", "page": 1},
        }
        text = "Some fact [G1].\n\n---\n[G1] Guide A: https://example.com/a"
        result = postprocess_citations(text, citation_map)
        # Should not have duplicate footnotes
        assert result.count("[1]") == 2  # one inline, one in generated footnotes

    def test_no_citations_returns_unchanged(self):
        result = postprocess_citations("No citations here.", {})
        assert result == "No citations here."

    def test_footnotes_include_urls(self):
        citation_map = {
            "R1": {"label": "Fire Records (Record 100.50.F)", "source_title": "SCHEDULE 100", "url": "https://example.com/s100", "page": 2},
        }
        text = "Request fire records [R1]."
        result = postprocess_citations(text, citation_map)
        assert "[SCHEDULE 100](https://example.com/s100)" in result

    def test_repeated_citation_uses_same_number(self):
        citation_map = {
            "G1": {"title": "Guide A", "url": None, "page": 1},
        }
        text = "Fact one [G1]. Fact two [G1]."
        result = postprocess_citations(text, citation_map)
        body = result.split("---")[0]
        assert body == "Fact one [1]. Fact two [1].\n\n"
