from asgiref.sync import sync_to_async

from apps.records.models import SystemPrompt


def format_retrieved_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant supporting documents were found."
    parts = ["Relevant supporting document excerpts:"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[{i}] Source: {c['document_title']}, page {c['page_number']}\n{c['text']}")
    return "\n".join(parts)


def format_retrieved_records(records: list[dict]) -> str:
    """Format retrieved records as context for the LLM prompt."""
    if not records:
        return "No relevant retention schedule entries were found."

    parts = ["Relevant retention schedule entries:"]
    for i, r in enumerate(records, 1):
        entry = [
            f"\n[{i}] {r['record_title']} (Record {r['record_number']})",
            f"Source: {r['document_title']}, page {r['page_number']}",
            f"Description: {r['record_description']}",
            f"Retention period: {r['minimum_retention_period']}",
        ]
        if r["custodian_requirement"]:
            entry.append(f"Disposition: {r['custodian_requirement']}")
        if r["regulatory_citations"]:
            entry.append(f"Citations: {r['regulatory_citations']}")
        parts.append("\n".join(entry))

    return "\n".join(parts)


async def build_messages(
    user_message: str,
    records: list[dict],
    doc_chunks: list[dict],
    conversation_history: list[dict],
) -> list[dict]:
    """
    Assemble the full message list to send to the LLM.

    Structure:
    - System prompt (fetched from DB, always first)
    - Prior conversation history (all turns except the last user message)
    - Retrieved context injected as a system message immediately before the latest user turn
    - The latest user message last
    """
    system_prompt = await sync_to_async(SystemPrompt.get_active)()
    combined_context = (
        "=== RETENTION SCHEDULE RECORDS ===\n"
        f"{format_retrieved_records(records)}\n\n"
        "=== SUPPORTING DOCUMENTS ===\n"
        f"{format_retrieved_chunks(doc_chunks)}"
    )

    prior_history = [
        m for m in conversation_history
        if not (m["role"] == "user" and m["content"] == user_message)
    ]

    return [
        {"role": "system", "content": system_prompt},
        *prior_history,
        {"role": "system", "content": f"Retrieved context:\n\n{combined_context}"},
        {"role": "user", "content": user_message},
    ]
