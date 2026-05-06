# FOIA Coach — Django RAG Service Spec

## Overview

This document specifies the Django service that powers the Retrieval-Augmented Generation (RAG)
backend for FOIA Coach Prototype 2. The service exposes an OpenAI-compatible
`/v1/chat/completions` endpoint that LibreChat uses as a custom backend. It owns document
ingestion, storage, retrieval, prompt assembly, and Language Model (LM) response streaming.
The service is deployed on Render alongside LibreChat.

---

## Data Model

### Source fields from parsed JSON

Each parsed record from the retention schedule JSON has the following fields:

| Field | Type | Notes |
|---|---|---|
| `record_number` | string | e.g. `"1.50"`, `"2-1"`, `"General Description"` |
| `record_title` | string | Short name for the record type |
| `record_description` | string | Full narrative description |
| `record_custodian_preservation_destruction_requirement` | string | Disposition instructions |
| `minimum_retention_period` | string | Free-text period, e.g. `"20 years"`, `"Permanent"`, `"See Schedule 7"` |
| `regulatory_citation_statutes_rules_notations` | string | Legal citations and cross-references |
| `page_number` | integer | Page in source PDF |
| `document_title` | string | e.g. `"SCHEDULE NO. 1 - BUILDING AND STRUCTURE RECORDS (Colorado Special Districts)"` |

### Django models

```python
from django.db import models
from pgvector.django import VectorField


class SystemPrompt(models.Model):
    """
    Editable system prompt for the FOIA Coach assistant.
    Only one prompt is active at a time.
    """
    name = models.CharField(max_length=255)
    content = models.TextField()
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name}{' (active)' if self.is_active else ''}"

    def save(self, *args, **kwargs):
        if self.is_active:
            # Ensure only one prompt is active at a time
            SystemPrompt.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

    @classmethod
    def get_active(cls) -> str:
        prompt = cls.objects.filter(is_active=True).first()
        if prompt is None:
            raise RuntimeError("No active system prompt configured.")
        return prompt.content


class SourceDocument(models.Model):
    """A source retention schedule PDF."""
    filename = models.CharField(max_length=512)
    document_title = models.CharField(max_length=512)
    jurisdiction = models.CharField(max_length=255)  # e.g. "Colorado"
    entity_type = models.CharField(max_length=255)   # e.g. "Special Districts", "State Government Agencies"
    schedule_number = models.CharField(max_length=50, blank=True)  # e.g. "1", "2", "3A"
    uploaded_at = models.DateTimeField(auto_now_add=True)
    record_count = models.IntegerField(default=0)

    class Meta:
        ordering = ["jurisdiction", "document_title"]

    def __str__(self):
        return self.document_title


class RetentionRecord(models.Model):
    """A single parsed record entry from a retention schedule."""

    source_document = models.ForeignKey(
        SourceDocument, on_delete=models.CASCADE, related_name="records"
    )

    # Core fields from parsed JSON
    record_number = models.CharField(max_length=50, blank=True)
    record_title = models.CharField(max_length=512)
    record_description = models.TextField()
    custodian_requirement = models.TextField(blank=True)  # maps from record_custodian_preservation_destruction_requirement
    minimum_retention_period = models.CharField(max_length=512)
    regulatory_citations = models.TextField(blank=True)  # maps from regulatory_citation_statutes_rules_notations
    page_number = models.IntegerField(null=True, blank=True)

    # Derived / normalized fields
    is_cross_reference = models.BooleanField(default=False)  # True when record just points to another schedule
    is_permanent = models.BooleanField(default=False)        # True when retention period is "Permanent"

    # Search fields
    embedding = VectorField(dimensions=1536, null=True)      # text-embedding-3-small
    search_vector = models.GeneratedField(           # populated via PostgreSQL trigger or signal
        expression=...,                              # see migrations note below
        output_field=models.TextField(),
        db_persist=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["source_document", "record_number"]
        indexes = [
            models.Index(fields=["source_document"]),
            models.Index(fields=["is_cross_reference"]),
            models.Index(fields=["is_permanent"]),
        ]

    def __str__(self):
        return f"{self.record_number} — {self.record_title}"

    def to_chunk_text(self) -> str:
        """
        Render the record as a natural language string for embedding.
        This is the text that gets embedded, not raw JSON.
        """
        parts = [
            f"Title: {self.record_title}",
            f"Description: {self.record_description}",
            f"Retention period: {self.minimum_retention_period}",
        ]
        if self.custodian_requirement:
            parts.append(f"Disposition: {self.custodian_requirement}")
        if self.regulatory_citations:
            parts.append(f"Legal citations: {self.regulatory_citations}")
        parts.append(f"Source: {self.source_document.document_title}, page {self.page_number}")
        return "\n".join(parts)
```

**Migration note on `search_vector`:** Django's `GeneratedField` for `tsvector` requires a raw SQL
expression. Use a `RunSQL` migration to create it:

```sql
ALTER TABLE records_retentionrecord
ADD COLUMN search_vector tsvector
GENERATED ALWAYS AS (
    to_tsvector('english',
        coalesce(record_title, '') || ' ' ||
        coalesce(record_description, '') || ' ' ||
        coalesce(minimum_retention_period, '') || ' ' ||
        coalesce(regulatory_citations, '')
    )
) STORED;

CREATE INDEX retention_record_search_vector_idx
ON records_retentionrecord USING GIN(search_vector);
```

---

## Ingestion Pipeline

Ingestion is a two-step process: load parsed JSON into the database, then generate and store
embeddings. These are separate management commands so they can be run independently (e.g.
re-embed without re-parsing, or load a new batch without re-embedding everything).

### Management command: `import_retention_records`

```
python manage.py import_retention_records <json_file> --jurisdiction "Colorado" --filename "Colorado_Schedules_1_-_3.pdf"
```

**Behavior:**

1. Parse `document_title` from each record to extract `schedule_number` and `entity_type`.
   - `"SCHEDULE NO. 1 - BUILDING AND STRUCTURE RECORDS (Colorado Special Districts)"`
   - → `schedule_number="1"`, `entity_type="Colorado Special Districts"`
2. Group records by `document_title` and upsert one `SourceDocument` per unique title.
3. For each record, upsert a `RetentionRecord` keyed on `(source_document, record_number)`.
4. Set `is_cross_reference=True` if `minimum_retention_period` starts with `"See "`.
5. Set `is_permanent=True` if `minimum_retention_period` is `"Permanent"` (case-insensitive).
6. Update `SourceDocument.record_count` after import.
7. Print a summary: records created, updated, skipped.

**Upsert behavior:** Re-running the command on the same file should be safe and idempotent —
existing records are updated in place, not duplicated.

### Management command: `generate_embeddings`

```
python manage.py generate_embeddings [--source-document <id>] [--force]
```

**Behavior:**

1. Query `RetentionRecord` objects where `embedding IS NULL` (or all records if `--force`).
2. For each record, call `record.to_chunk_text()` to produce the embedding input string.
3. Batch records into groups of 100 and call the OpenAI embeddings API
   (`text-embedding-3-small`, 1536 dimensions).
4. Store the returned vector in `record.embedding`.
5. Print progress and a final summary.

**Notes:**
- Skip cross-reference records (`is_cross_reference=True`) — they contain no substantive
  content to embed, only pointers to other schedules.
- Rate-limit awareness: add a short sleep between batches to avoid hitting OpenAI rate limits.
- Embeddings should be regenerated if `record_description` or other content fields change.
  The `--force` flag handles this.

---

## Hybrid Search

Search combines dense vector similarity (via pgvector) with sparse full-text search (via
PostgreSQL Full-Text Search) using Reciprocal Rank Fusion (RRF) to merge the ranked result
lists. This is implemented as a raw SQL query since Django's ORM does not support this pattern
natively.

```python
from django.db import connection
from pgvector.django import VectorField
import numpy as np


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    jurisdiction: str | None = None,
    exclude_cross_references: bool = True,
    limit: int = 8,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run hybrid pgvector + FTS search with RRF fusion.

    Returns a list of dicts with record fields and rrf_score.
    """
    params = [
        query_embedding,          # dense ORDER BY
        query_embedding,          # dense ORDER BY (subquery)
        query_text,               # sparse ts_rank
        query_text,               # sparse plainto_tsquery
        query_text,               # sparse plainto_tsquery (WHERE)
        rrf_k,                    # RRF constant (dense)
        rrf_k,                    # RRF constant (sparse)
        limit,
    ]

    jurisdiction_filter = ""
    if jurisdiction:
        jurisdiction_filter = "AND sd.jurisdiction = %s"
        params.insert(2, jurisdiction)
        params.insert(5, jurisdiction)

    cross_ref_filter = ""
    if exclude_cross_references:
        cross_ref_filter = "AND r.is_cross_reference = FALSE"

    sql = f"""
    WITH dense AS (
        SELECT
            r.id,
            ROW_NUMBER() OVER (ORDER BY r.embedding <=> %s::vector) AS rank
        FROM records_retentionrecord r
        JOIN records_sourcedocument sd ON r.source_document_id = sd.id
        WHERE r.embedding IS NOT NULL
          {jurisdiction_filter}
          {cross_ref_filter}
        ORDER BY r.embedding <=> %s::vector
        LIMIT 60
    ),
    sparse AS (
        SELECT
            r.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(r.search_vector, plainto_tsquery('english', %s)) DESC
            ) AS rank
        FROM records_retentionrecord r
        JOIN records_sourcedocument sd ON r.source_document_id = sd.id
        WHERE r.search_vector @@ plainto_tsquery('english', %s)
          {jurisdiction_filter}
          {cross_ref_filter}
        LIMIT 60
    ),
    fused AS (
        SELECT
            COALESCE(d.id, s.id) AS id,
            (
                COALESCE(1.0 / (%s + d.rank), 0) +
                COALESCE(1.0 / (%s + s.rank), 0)
            ) AS rrf_score
        FROM dense d
        FULL OUTER JOIN sparse s ON d.id = s.id
    )
    SELECT
        r.id,
        r.record_number,
        r.record_title,
        r.record_description,
        r.minimum_retention_period,
        r.custodian_requirement,
        r.regulatory_citations,
        r.page_number,
        r.is_permanent,
        sd.document_title,
        sd.jurisdiction,
        sd.entity_type,
        f.rrf_score
    FROM fused f
    JOIN records_retentionrecord r ON f.id = r.id
    JOIN records_sourcedocument sd ON r.source_document_id = sd.id
    ORDER BY f.rrf_score DESC
    LIMIT %s
    """

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

---

## OpenAI-Compatible Endpoint

### Route

```
POST /v1/chat/completions
```

LibreChat is configured to point at this endpoint via its custom endpoint configuration.

### Request format

The endpoint accepts the standard OpenAI chat completions request body:

```json
{
    "model": "foia-coach",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "How long do I need to keep building permits?"}
    ],
    "stream": true
}
```

The `model` field is ignored — the service always uses the configured LLM. The `stream` field
should always be `true` from LibreChat; the endpoint supports both streaming and non-streaming
for completeness.

### View

```python
import json
import asyncio
from django.http import StreamingHttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import openai


@method_decorator(csrf_exempt, name="dispatch")
class ChatCompletionsView(View):

    async def post(self, request):
        body = json.loads(request.body)
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Extract the latest user message
        user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        if not user_message:
            return JsonResponse({"error": "No user message found"}, status=400)

        # Run retrieval
        records = await retrieve(user_message)

        # Build augmented prompt (fetches active system prompt from DB)
        try:
            augmented_messages = await build_messages(user_message, records, messages)
        except RuntimeError as e:
            return JsonResponse({"error": str(e)}, status=500)

        if stream:
            return StreamingHttpResponse(
                stream_completion(augmented_messages),
                content_type="text/event-stream",
            )
        else:
            return await complete(augmented_messages)
```

### Retrieval

```python
from asgiref.sync import sync_to_async
import openai


async def retrieve(query: str, jurisdiction: str | None = None) -> list[dict]:
    """Embed the query and run hybrid search."""
    client = openai.AsyncOpenAI()

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = response.data[0].embedding

    # hybrid_search is a synchronous DB call — wrap it
    records = await sync_to_async(hybrid_search)(
        query_text=query,
        query_embedding=query_embedding,
        jurisdiction=jurisdiction,
        limit=8,
    )
    return records
```

### Prompt assembly

```python
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
    conversation_history: list[dict],
) -> list[dict]:
    """
    Assemble the full message list to send to the LLM.

    Structure:
    - System prompt (fetched from DB, always first)
    - Retrieval context injected as a system message immediately before the latest user turn
    - Prior conversation history (all turns except the last user message)
    - The latest user message last
    """
    system_prompt = await sync_to_async(SystemPrompt.get_active)()
    context = format_retrieved_records(records)

    prior_history = [
        m for m in conversation_history
        if not (m["role"] == "user" and m["content"] == user_message)
    ]

    return [
        {"role": "system", "content": system_prompt},
        *prior_history,
        {"role": "system", "content": f"Retrieved context:\n\n{context}"},
        {"role": "user", "content": user_message},
    ]
```

### Streaming response

```python
import uuid
import time


async def stream_completion(messages: list[dict]):
    """
    Yield Server-Sent Events (SSE) in OpenAI streaming format.
    LibreChat expects this exact wire format.
    """
    client = openai.AsyncOpenAI()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    async with client.chat.completions.stream(
        model="gpt-5.2",  # configurable via settings
        messages=messages,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "foia-coach",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": delta.role or None,
                        "content": delta.content or "",
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }],
            }
            yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"
```

---

## Django Admin

The admin provides corpus management and prompt iteration for the team without requiring
database access or a service redeploy.

### `SystemPromptAdmin`

- List view: name, is_active, updated_at
- List filter: is_active
- Read-only fields: created_at, updated_at
- Only one prompt may be active at a time — saving a prompt with `is_active=True`
  automatically deactivates all others.
- A data migration seeds the initial prompt on first deploy so the service is never
  in a state where no active prompt exists. If `get_active()` raises a `RuntimeError`
  (i.e. someone deactivated the last prompt without activating another), the endpoint
  returns a clean HTTP 500 with a descriptive error rather than silently falling back
  to a hardcoded default.

```python
@admin.register(SystemPrompt)
class SystemPromptAdmin(admin.ModelAdmin):
    list_display = ["name", "is_active", "updated_at"]
    list_filter = ["is_active"]
    readonly_fields = ["created_at", "updated_at"]
    fields = ["name", "content", "is_active", "created_at", "updated_at"]
```

### `SourceDocumentAdmin`

- List view: filename, document_title, jurisdiction, entity_type, record_count, uploaded_at
- Actions: `regenerate_embeddings` (queues embedding generation for all records in this document)
- Read-only fields: record_count, uploaded_at

### `RetentionRecordAdmin`

- List view: record_number, record_title, source_document, minimum_retention_period,
  is_permanent, is_cross_reference, has_embedding (derived: `embedding IS NOT NULL`)
- List filters: source_document, is_permanent, is_cross_reference, has_embedding
- Search: record_title, record_description, record_number
- Read-only fields: embedding (shown as "Embedded" / "Not embedded"), search_vector,
  created_at, updated_at
- Actions: `regenerate_embeddings` for selected records

---

## Configuration

All sensitive values and environment-specific settings via environment variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Used for embeddings and LLM completions |
| `LLM_MODEL` | Model to use for completions, default `gpt-5.2` |
| `EMBEDDING_MODEL` | Model for embeddings, default `text-embedding-3-small` |
| `DATABASE_URL` | PostgreSQL connection string (Render provides this) |
| `DJANGO_SECRET_KEY` | Standard Django secret key |
| `ALLOWED_HOSTS` | Render service hostname |
| `DEBUG` | `False` in production |

---

## URL Configuration

```python
urlpatterns = [
    path("admin/", admin.site.urls),
    path("v1/chat/completions", ChatCompletionsView.as_view()),
    path("v1/models", ModelsView.as_view()),  # returns a minimal model list so LibreChat's model selector works
]
```

---

## Project Structure

```
foia_coach_api/
├── config/
│   ├── settings/
│   │   ├── base.py
│   │   ├── local.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py / asgi.py
├── apps/
│   └── records/
│       ├── admin.py
│       ├── models.py          # SourceDocument, RetentionRecord, SystemPrompt
│       ├── search.py          # hybrid_search()
│       ├── prompts.py         # build_messages(), format_retrieved_records()
│       ├── views.py           # ChatCompletionsView, ModelsView
│       ├── management/
│       │   └── commands/
│       │       ├── import_retention_records.py
│       │       └── generate_embeddings.py
│       └── migrations/
│           ├── 0001_initial.py
│           └── 0002_seed_system_prompt.py  # data migration with initial prompt
├── requirements.txt
└── manage.py
```

---

## Dependencies

```
django>=4.2
psycopg[binary]
pgvector
openai>=1.0
```

`asgiref` is included with Django and provides `sync_to_async`. No additional async framework
is needed.

---

## Open Questions

These items are intentionally deferred to avoid over-specifying before implementation:

- **Jurisdiction filtering:** The retrieval function accepts a `jurisdiction` parameter, but
  LibreChat has no mechanism to pass metadata alongside a chat message. For the prototype, the
  endpoint will search across all jurisdictions. A follow-up option is to parse jurisdiction
  intent from the query itself using a lightweight classifier or a quick LLM call before retrieval.

- **Token budget:** The prompt assembly does not currently enforce a token limit on retrieved
  context. With 8 records averaging ~200 tokens each, context stays well within GPT-5.2's
  context window for the prototype. This should be revisited if records become longer or the
  retrieval limit increases.

- **Cross-reference handling:** Cross-reference records (e.g. "See Schedule 7") are excluded
  from search. A future improvement would follow the reference and include the target record
  in the context, but this requires schedule cross-linking that is out of scope for the prototype.

- **Authentication on the endpoint:** The `/v1/chat/completions` endpoint has no authentication
  in this spec. Since LibreChat handles auth and the Django service should be on an internal
  Render network, this is acceptable for the prototype. A shared secret header should be added
  before any public exposure.
