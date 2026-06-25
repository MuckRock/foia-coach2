import re

from asgiref.sync import sync_to_async

from apps.records.models import NFOICChapter, SystemPrompt


def format_retrieved_chunks(chunks: list[dict]) -> tuple[str, dict]:
    """Format guidance doc chunks for the LLM context.

    Returns (formatted_text, citation_map_entries) where citation_map_entries
    maps keys like "G1" to source metadata for post-processing.
    """
    if not chunks:
        return "No relevant supporting documents were found.", {}
    citation_map = {}
    parts = ["Relevant supporting document excerpts:"]
    for i, c in enumerate(chunks, 1):
        key = f"G{i}"
        url = None
        if c.get("documentcloud_url"):
            url = f"{c['documentcloud_url']}#document/p{c['page_number']}"
        citation_map[key] = {
            "title": c["document_title"],
            "url": url,
            "page": c["page_number"],
        }
        parts.append(f"\n[{key}] Source: {c['document_title']}, page {c['page_number']}\n{c['text']}")
    return "\n".join(parts), citation_map


def format_retrieved_records(records: list[dict]) -> tuple[str, dict]:
    """Format retrieved records for the LLM context.

    Returns (formatted_text, citation_map_entries) where citation_map_entries
    maps keys like "R1" to source metadata for post-processing.
    """
    if not records:
        return "No relevant retention schedule entries were found.", {}

    citation_map = {}
    parts = ["Relevant retention schedule entries:"]
    for i, r in enumerate(records, 1):
        key = f"R{i}"
        url = None
        if r.get("documentcloud_url"):
            url = f"{r['documentcloud_url']}#document/p{r['page_number']}"
        citation_map[key] = {
            "label": f"{r['record_title']} (Record {r['record_number']})",
            "source_title": r["document_title"],
            "url": url,
            "page": r["page_number"],
        }
        entry = [
            f"\n[{key}] {r['record_title']} (Record {r['record_number']})",
            f"Source: {r['document_title']}, page {r['page_number']}",
            f"Description: {r['record_description']}",
            f"Retention period: {r['minimum_retention_period']}",
        ]
        if r["custodian_requirement"]:
            entry.append(f"Disposition: {r['custodian_requirement']}")
        if r["regulatory_citations"]:
            entry.append(f"Citations: {r['regulatory_citations']}")
        parts.append("\n".join(entry))

    return "\n".join(parts), citation_map


def postprocess_citations(text: str, citation_map: dict) -> str:
    """Renumber [G1]/[R1] citation keys to sequential [1], [2], ... and append footnotes."""
    key_pattern = re.compile(r'\[([GR]\d+)\]')

    # Collect unique keys in order of first appearance
    seen: list[str] = []
    for match in key_pattern.finditer(text):
        key = match.group(1)
        if key not in seen and key in citation_map:
            seen.append(key)

    if not seen:
        return text

    renumber = {key: str(i) for i, key in enumerate(seen, 1)}

    # Strip any LLM-generated footnotes/sources section at the end.
    # Look for a trailing block of lines that start with [G or [R keys.
    stripped = re.sub(
        r'\n---\s*\n(?:\s*\[(?:[GR]\d+)\].*\n?)+\s*$', '', text
    )
    # Also catch "Sources:" / "**Sources:**" / "References:" headers
    stripped = re.sub(
        r'\n+(?:\*{0,2}(?:Sources|References|Citations)\*{0,2}:?\s*)\n(?:\s*\[(?:[GR]\d+)\].*\n?)+\s*$',
        '', stripped, flags=re.IGNORECASE,
    )

    # Replace all citation keys with sequential numbers
    def replace_key(match):
        key = match.group(1)
        if key in renumber:
            return f'[{renumber[key]}]'
        return match.group(0)

    processed = key_pattern.sub(replace_key, stripped)

    # Build clean footnotes section
    footnotes = []
    for key in seen:
        info = citation_map[key]
        num = renumber[key]
        if "label" in info:
            # Retention record
            if info.get("url"):
                footnotes.append(
                    f'[{num}] {info["label"]} — [{info["source_title"]}]({info["url"]})'
                )
            else:
                footnotes.append(f'[{num}] {info["label"]} — {info["source_title"]}')
        else:
            # Guidance document
            if info.get("url"):
                footnotes.append(
                    f'[{num}] [{info["title"]}]({info["url"]})'
                )
            else:
                footnotes.append(f'[{num}] {info["title"]}, page {info["page"]}')

    processed = processed.rstrip() + "\n\n---\n" + "\n\n".join(footnotes)
    return processed


async def build_messages(
    user_message: str,
    records: list[dict],
    doc_chunks: list[dict],
    conversation_history: list[dict],
    state: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Assemble the full message list to send to the LLM.

    Returns (messages, citation_map) where citation_map maps keys like "G1", "R1"
    to source metadata for post-processing citations.

    Structure:
    - System prompt (fetched from DB, always first)
    - Prior conversation history (all turns except the last user message)
    - Retrieved context injected as a system message immediately before the latest user turn
    - The latest user message last
    """
    system_prompt = await sync_to_async(SystemPrompt.get_active)()

    chapter = None
    if state:
        chapter = await sync_to_async(
            NFOICChapter.objects.filter(jurisdiction=state).first
        )()

    state_instruction = (
        f"You are answering questions about {state} public records. "
        f"Begin your response by stating 'Working with {state} data.' on its own line, "
        "then continue with your answer.\n\n"
        if state else ""
    )

    if chapter:
        chapter_lines = [f"=== {state.upper()} NFOIC CHAPTER (legal referral contact) ==="]
        chapter_lines.append(f"Organization: {chapter.name}")
        if chapter.website:
            chapter_lines.append(f"Website: {chapter.website}")
        if chapter.email:
            chapter_lines.append(f"Email: {chapter.email}")
        if chapter.phone:
            chapter_lines.append(f"Phone: {chapter.phone}")
        if chapter.description:
            chapter_lines.append(f"About: {chapter.description}")
        chapter_section = "\n".join(chapter_lines)
    else:
        chapter_section = ""

    chunks_text, chunks_citations = format_retrieved_chunks(doc_chunks)
    records_text, records_citations = format_retrieved_records(records)
    citation_map = {**chunks_citations, **records_citations}

    combined_context = (
        f"{state_instruction}"
        "=== GUIDANCE DOCUMENTS (check these first) ===\n"
        f"{chunks_text}\n\n"
        "=== RETENTION SCHEDULE RECORDS ===\n"
        f"{records_text}"
        + (f"\n\n{chapter_section}" if chapter_section else "")
    )

    prior_history = [
        m for m in conversation_history
        if not (m["role"] == "user" and m["content"] == user_message)
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        *prior_history,
        {"role": "system", "content": f"Retrieved context:\n\n{combined_context}"},
        {"role": "user", "content": user_message},
    ]
    return messages, citation_map
