# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two deployable services in one repo:

1. **Django RAG service** (`config/` + `apps/records/`) — "Agent Moss", a public-records
   retention assistant. It exposes an OpenAI-compatible `POST /v1/chat/completions`
   endpoint (consumed by LibreChat and by the MCP server) that retrieves retention
   schedule records + guidance-document chunks from Postgres/pgvector and streams an
   LLM answer.
2. **MCP server** (`mcp_server/`) — a FastMCP streamable-HTTP server that wraps the Django
   service (`search_records`, `chat`) plus MuckRock FOIA-filing tools, behind a Squarelet
   OAuth proxy.

Both deploy to Render via `render.yaml` (separate Dockerfiles: `Dockerfile`, `Dockerfile.mcp`).

## Commands

Everything runs through Docker Compose (Postgres uses the `pgvector/pgvector:pg16` image;
the pgvector and pg_trgm extensions are created by migration `0001_initial`).

```bash
docker compose up --build
docker compose run --rm django python manage.py migrate
docker compose run --rm django python manage.py shell_plus

# Tests
docker compose run --rm django pytest
docker compose run --rm django pytest apps/records/tests/test_search.py::TestHybridSearch::test_returns_results
docker compose run --rm django pytest --cov=apps/records --cov-report=term-missing
```

`pytest.ini` sets `DJANGO_SETTINGS_MODULE=config.settings.local`, `asyncio_mode=auto`, and
only collects `tests/test_*.py`. Tests hit a real Postgres — the hybrid search is raw SQL
(tsvector + pgvector), so it cannot be tested against sqlite or with mocks.

`.envs/.local/.django` and `.envs/.local/.postgres` are gitignored and not checked in;
`docker-compose.yml` expects both to exist before anything will start.

### Evals

```bash
OPENAI_API_KEY=... python eval/run_eval.py --base-url http://localhost:8000 --system-name v2-agent-moss
python eval/run_eval.py --category discovery legal_advice
```

Cases live in `eval/cases.json`, grouped into five categories (`legal_advice`, `discovery`,
`multi_hop`, `no_state`, `unknown_state`) each with its own GPT-4o judge prompt in
`eval/run_eval.py`. Reports (markdown + self-contained HTML) are gitignored. The `--adapter v1`
path targets the older Gemini-backed FOIA Coach for baseline comparison.

### MCP server

```bash
python -m mcp_server   # streamable-http on $PORT (default 8001)
```

## Request flow (`apps/records/views.py`)

`ChatCompletionsView.post` is fully async and does, in order:

1. `detect_state()` — an LLM call that resolves the jurisdiction from the conversation.
   **If it returns None the view short-circuits** with a canned clarifying question and never
   retrieves or calls the main LLM. Every downstream search is filtered by this jurisdiction,
   so state detection is the single biggest lever on answer quality.
2. In parallel: `rewrite_query()` (collapse multi-turn history into a standalone query) and
   `generate_hyde_query()` (HyDE — a hypothetical retention-schedule passage).
3. Two embeddings in parallel, then two searches in parallel:
   - **HyDE embedding → `hybrid_search()`** over `RetentionRecord` (structured schedule rows)
   - **rewritten-query embedding → `document_search()`** over `DocumentChunk` (guidance PDFs)
     The two searches deliberately use _different_ query texts; don't collapse them.
4. `build_messages()` assembles: active system prompt → prior history → a system message with
   retrieved context → the latest user turn. The NFOIC chapter for the state is injected here.
5. Streamed or non-streamed OpenAI-format response, with citation post-processing.

The whole pipeline logs at INFO under the `apps.records` logger (retrieved records with RRF
scores, the HyDE text, the full assembled prompt) — that log is the primary debugging tool.

### Retrieval (`apps/records/search.py`)

`hybrid_search()` is one hand-written SQL statement: a dense CTE (`embedding <=> %s::vector`,
top 60) FULL OUTER JOINed with a sparse CTE (`ts_rank` over the generated `search_vector`
column, top 60), fused with RRF where **dense is weighted 1.5× sparse**. The sparse side
rewrites `plainto_tsquery` output from `&` to `|` so it behaves as OR, not AND.

Parameter order in that query is positional and fragile — the `jurisdiction` filter is spliced
into two places via `params.insert(1, ...)` and `params.insert(5, ...)`. Any edit to the SQL
must keep the comment block documenting param order in sync.

`document_search()` is plain cosine-nearest-neighbour over `DocumentChunk`, no fusion.

### Citations (`apps/records/prompts.py`)

Retrieved context is labelled `[R1]…` (records) and `[G1]…` (guidance chunks); the LLM is
expected to cite those keys. `postprocess_citations()` then renumbers them to `[1], [2], …`
in order of first appearance, strips any footnote/"Sources:" block the model generated itself,
and appends clean footnotes with DocumentCloud deep links (`{url}#document/p{page}`).

Because of this, **streaming with a citation_map is not actually incremental**: `stream_completion()`
buffers the whole response, post-processes, and emits it as a single SSE chunk. That's intentional
— citations can't be renumbered until the full text is known.

## Data model notes (`apps/records/models.py`)

- `RetentionRecord.search_vector` is a **generated tsvector column added via `RunSQL` in
  migration 0001 and deliberately not declared as a Django field.** Adding it to the model
  will break the ORM. HNSW indexes on both `embedding` columns are likewise raw SQL.
- `SystemPrompt` — only one row may be `is_active`; `save()` deactivates the others.
  `get_active()` raises `RuntimeError` if none is active, which the view turns into a 500.
  Prompts are edited in Django Admin at `/admin/`; `update_system_prompt` (run by
  `bin/pre_deploy.sh` on every deploy) creates and activates the canonical prompt in code.
  **Editing the assistant's behaviour usually means editing `PROMPT_CONTENT` in
  `apps/records/management/commands/update_system_prompt.py`, not Python logic.**
- `NFOICChapter` supplies the per-state legal-referral contact the prompt requires when it
  triages a question as legal advice.

## Ingestion

Two ingestion paths exist. The DocumentCloud ones are current; the file-based ones remain for
local/one-off imports.

- `import_retention_schedules_dc --project <id> --jurisdiction <name>` — pulls a DC project,
  runs LLM extraction (`EXTRACTION_MODEL`, always OpenAI) into `RetentionRecord` rows.
- `import_guides_dc --project <id> --jurisdiction <name>` — chunks DC guidance docs into
  `DocumentChunk`.
- `import_retention_records <file.json> --jurisdiction <name>` — upsert from normalized JSON
  (safe to re-run). `normalize_records.py` converts raw Colorado JSON dumps into that format.
- `import_supporting_documents <file.pdf> --title … --document-type … --jurisdiction …`
  (`--replace` to re-import).
- `generate_embeddings` / `generate_document_embeddings` — must be run after any import;
  rows with `embedding IS NULL` are invisible to search. `--force` re-embeds.

Loading production data means running these locally with `DATABASE_URL` pointed at the Render
external database, after the deploy has applied migrations.

## LLM configuration

`LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` configure the _answer_ model and can point at
Anthropic (production currently runs `claude-opus-4-8` through the OpenAI-compatible client).
`QUERY_REWRITE_MODEL`, `EXTRACTION_MODEL`, and `EMBEDDING_MODEL` always go to OpenAI via
`OPENAI_API_KEY` — query rewriting, state detection, HyDE, and embeddings are hardcoded to
`openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)`, not `_llm_client()`.

Set `LLM_TEMPERATURE_ENABLED=False` for models that reject a `temperature` parameter; the
views omit the kwarg entirely rather than sending a default.

## MCP server notes

- `SquareletOAuthProvider` stores clients, codes, and tokens **in process memory** — it is
  documented as staging-only and needs Redis before it can scale past one instance.
- Two token layers: the MCP access token issued to the client, and the MuckRock/Squarelet JWT
  stored in its claims. `_muckrock_call()` reloads the token from the provider on every call
  (so a refresh earlier in the session is picked up) and, on a 401, calls
  `refresh_muckrock_token()` and retries once — no browser round-trip.
- `file_request` submits a real FOIA request; its docstring instructs the client to confirm
  with the user first.
