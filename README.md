# Agent Moss — FOIA Coach RAG Backend

Django RAG service powering Agent Moss, a public records retention assistant. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint consumed by LibreChat. On every query it retrieves relevant retention schedule records and supporting documents, injects them as context, and streams an LLM response.

---

## How it works

1. **Query rewriting** — multi-turn conversation history is collapsed into a standalone search query
2. **Dual retrieval** — runs in parallel:
   - Hybrid search (dense vector + full-text with RRF fusion) over structured retention records
   - Pure vector similarity search over supporting document chunks
3. **Prompt assembly** — retrieved context injected as a system message before the user turn
4. **LLM completion** — streamed via OpenAI API in SSE format

---

## Project structure

```
apps/records/
├── models.py          # SystemPrompt, SourceDocument, RetentionRecord,
│                      # SupportingDocument, DocumentChunk
├── search.py          # hybrid_search(), document_search()
├── prompts.py         # build_messages(), format_retrieved_records(), format_retrieved_chunks()
├── views.py           # ChatCompletionsView, ModelsView
├── admin.py           # Admin for all models
├── migrations/
│   ├── 0001_initial.py                  # schema, pgvector, tsvector generated column, HNSW index
│   ├── 0002_seed_system_prompt.py       # seeds initial system prompt
│   └── 0003_supporting_documents.py    # SupportingDocument + DocumentChunk + HNSW index
└── management/commands/
    ├── import_retention_records.py      # load structured JSON into RetentionRecord rows
    ├── generate_embeddings.py           # embed RetentionRecords via OpenAI
    ├── import_supporting_documents.py   # chunk PDFs into DocumentChunk rows
    └── generate_document_embeddings.py  # embed DocumentChunks via OpenAI

config/settings/
├── base.py        # shared settings
├── local.py       # development
└── production.py  # Render deployment
```

---

## Local development

### Prerequisites

- Docker + Docker Compose
- OpenAI API key

### Setup

```bash
# Copy and fill in env files
cp .envs/.local/.django.example .envs/.local/.django
cp .envs/.local/.postgres.example .envs/.local/.postgres

# Build and start
docker compose up --build

# Run migrations
docker compose run --rm django python manage.py migrate

# Create a superuser for admin access
docker compose run --rm django python manage.py createsuperuser
```

### Running tests

```bash
docker compose run --rm django pytest
docker compose run --rm django pytest --cov=apps/records --cov-report=term-missing
```

### Interactive shell

```bash
docker compose run --rm django python manage.py shell_plus
```

---

## Data ingestion

### Retention records (structured JSON)

```bash
docker compose run --rm django python manage.py import_retention_records \
  /path/to/schedule.json --jurisdiction "Colorado"

docker compose run --rm django python manage.py generate_embeddings
```

Re-running `import_retention_records` on the same file is safe — records are upserted, not duplicated. Use `--force` with `generate_embeddings` to re-embed existing records.

### Supporting documents (PDFs)

```bash
docker compose run --rm django python manage.py import_supporting_documents \
  /path/to/guide.pdf \
  --title "Colorado CORA Guide" \
  --document-type "FOIA Guide" \
  --jurisdiction "Colorado"

docker compose run --rm django python manage.py generate_document_embeddings
```

Use `--replace` to re-import a document that has already been loaded.

---

## Configuration

All settings are via environment variables:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `DATABASE_URL` | PostgreSQL connection string | — |
| `DJANGO_SECRET_KEY` | Django secret key | — |
| `LLM_MODEL` | Model for completions | `gpt-4o` |
| `QUERY_REWRITE_MODEL` | Model for query rewriting | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` |
| `LLM_TEMPERATURE` | Completion temperature | `0.2` |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated allowed hosts | `*` (local) |

---

## API

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Accepts standard request body; the `model` field is ignored.

```json
{
  "model": "agent-moss",
  "messages": [{"role": "user", "content": "How long must building permits be kept?"}],
  "stream": true
}
```

### `GET /v1/models`

Returns the model list (used by LibreChat's model selector).

---

## Deployment (Render)

The repo includes `render.yaml` for one-click Blueprint deployment.

**Steps:**
1. Push to GitHub
2. Render → New → Blueprint → select repo
3. Set `OPENAI_API_KEY` in the Render dashboard
4. First deploy will run migrations automatically via `preDeployCommand`

**Loading data into production:**

Run import commands locally, pointed at the Render external database URL:

```bash
DATABASE_URL="postgresql://..." docker compose run --rm django \
  python manage.py import_retention_records /path/to/schedule.json --jurisdiction "Colorado"

DATABASE_URL="postgresql://..." docker compose run --rm django \
  python manage.py generate_embeddings
```

Always deploy first (so migrations are applied) before running imports against production.

---

## System prompt

The active system prompt is managed via Django Admin at `/admin/`. Only one prompt can be active at a time — saving a new prompt as active automatically deactivates the previous one. The service returns HTTP 500 if no active prompt is configured.
