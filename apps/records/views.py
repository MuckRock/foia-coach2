import asyncio
import json
import logging
import time
import uuid

from asgiref.sync import sync_to_async
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

import openai
from apps.records.prompts import build_messages
from apps.records.search import document_search, hybrid_search

logger = logging.getLogger(__name__)


async def rewrite_query(user_message: str, conversation_history: list[dict]) -> str:
    """
    Rewrite the latest user message as a standalone search query using conversation context.
    If there is no prior conversation, returns the original message unchanged.
    """
    prior = [
        m for m in conversation_history
        if not (m["role"] == "user" and m["content"] == user_message)
    ]
    if not prior:
        return user_message

    recent = prior[-6:]  # last 3 turns
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in recent
        if m.get("content")
    )

    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.QUERY_REWRITE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a conversation history and the user's latest message, "
                    "rewrite the latest message as a complete, standalone search query "
                    "that captures all relevant context. "
                    "Output only the rewritten query, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nLatest message: {user_message}",
            },
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


async def embed_query(query: str) -> list[float]:
    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.embeddings.create(model=settings.EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


async def retrieve(query: str, query_embedding: list[float], jurisdiction: str | None = None) -> list[dict]:
    """Run hybrid search using a pre-computed embedding."""
    return await sync_to_async(hybrid_search)(
        query_text=query,
        query_embedding=query_embedding,
        jurisdiction=jurisdiction,
        limit=8,
    )


async def retrieve_documents(query_embedding: list[float]) -> list[dict]:
    return await sync_to_async(document_search)(query_embedding=query_embedding, limit=6)


async def stream_completion(messages: list[dict]):
    """Yield Server-Sent Events (SSE) in OpenAI streaming format."""
    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    stream = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

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


async def complete(messages: list[dict]) -> JsonResponse:
    """Non-streaming completion."""
    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=settings.LLM_TEMPERATURE,
    )
    return JsonResponse(response.model_dump())


@method_decorator(csrf_exempt, name="dispatch")
class ChatCompletionsView(View):

    async def post(self, request):
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        if not user_message:
            return JsonResponse({"error": "No user message found"}, status=400)

        retrieval_query = await rewrite_query(user_message, messages)
        if retrieval_query != user_message:
            logger.info("=== QUERY REWRITE === %r → %r", user_message, retrieval_query)

        query_embedding = await embed_query(retrieval_query)
        records, doc_chunks = await asyncio.gather(
            retrieve(retrieval_query, query_embedding),
            retrieve_documents(query_embedding),
        )

        logger.info("=== RETRIEVED RECORDS (%d) ===", len(records))
        logger.info("query: %s", retrieval_query)
        for i, r in enumerate(records, 1):
            logger.info(
                "  [%d] (rrf=%.4f dense=%.4f sparse=%.4f) %s — %s | retention: %s",
                i,
                r.get("rrf_score", 0),
                r.get("dense_score", 0),
                r.get("sparse_score", 0),
                r.get("record_title", ""),
                r.get("document_title", ""),
                r.get("minimum_retention_period", ""),
            )

        logger.info("=== RETRIEVED DOCUMENT CHUNKS (%d) ===", len(doc_chunks))
        for i, c in enumerate(doc_chunks, 1):
            logger.info(
                "  [%d] (sim=%.4f) %s — page %s",
                i,
                c.get("similarity_score", 0),
                c.get("document_title", ""),
                c.get("page_number", ""),
            )

        try:
            augmented_messages = await build_messages(user_message, records, doc_chunks, messages)
        except RuntimeError as e:
            return JsonResponse({"error": str(e)}, status=500)

        logger.info("=== FULL PROMPT (%d messages) ===", len(augmented_messages))
        for msg in augmented_messages:
            role = msg["role"].upper()
            content = msg["content"] or ""
            logger.info("  [%s] %s", role, content)

        if stream:
            return StreamingHttpResponse(
                stream_completion(augmented_messages),
                content_type="text/event-stream",
            )
        else:
            return await complete(augmented_messages)

    async def get(self, request):
        return JsonResponse({"error": "Method not allowed"}, status=405)


class ModelsView(View):
    async def get(self, request):
        return JsonResponse({
            "object": "list",
            "data": [
                {
                    "id": "foia-coach",
                    "object": "model",
                    "created": 0,
                    "owned_by": "foia-coach",
                }
            ],
        })
