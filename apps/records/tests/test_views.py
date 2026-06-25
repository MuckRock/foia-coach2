"""Tests for the chat completions endpoint."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from django.test import AsyncClient


@pytest.fixture
def async_client():
    return AsyncClient()


@pytest.mark.django_db
class TestModelsView:
    @pytest.mark.asyncio
    async def test_returns_model_list(self, async_client):
        response = await async_client.get("/v1/models")
        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "agent-moss" in ids


@pytest.mark.django_db
class TestChatCompletionsView:
    @pytest.mark.asyncio
    async def test_get_returns_405(self, async_client):
        response = await async_client.get("/v1/chat/completions")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, async_client):
        payload = {
            "model": "agent-moss",
            "messages": [{"role": "system", "content": "You are helpful."}],
            "stream": False,
        }
        response = await async_client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    @patch("apps.records.views.retrieve", new_callable=AsyncMock)
    @patch("apps.records.views.build_messages", new_callable=AsyncMock)
    @patch("apps.records.views.stream_completion")
    async def test_streaming_returns_event_stream(
        self, mock_stream, mock_build, mock_retrieve, async_client
    ):
        mock_retrieve.return_value = []
        mock_build.return_value = ([{"role": "user", "content": "test"}], {})

        async def fake_stream(messages, citation_map=None):
            yield "data: {}\n\n"
            yield "data: [DONE]\n\n"

        mock_stream.return_value = fake_stream([])

        payload = {
            "model": "agent-moss",
            "messages": [{"role": "user", "content": "How long for permits?"}],
            "stream": True,
        }
        response = await async_client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.get("Content-Type", "")

    @pytest.mark.asyncio
    @patch("apps.records.views.retrieve", new_callable=AsyncMock)
    @patch("apps.records.views.build_messages", new_callable=AsyncMock)
    @patch("apps.records.views.stream_completion")
    async def test_streaming_yields_done_sentinel(
        self, mock_stream, mock_build, mock_retrieve, async_client
    ):
        mock_retrieve.return_value = []
        mock_build.return_value = ([{"role": "user", "content": "test"}], {})

        async def fake_stream(messages, citation_map=None):
            yield "data: {}\n\n"
            yield "data: [DONE]\n\n"

        mock_stream.return_value = fake_stream([])

        payload = {
            "model": "agent-moss",
            "messages": [{"role": "user", "content": "How long for permits?"}],
            "stream": True,
        }
        response = await async_client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
        )
        content = b"".join([chunk async for chunk in response.streaming_content])
        assert b"[DONE]" in content

    @pytest.mark.asyncio
    @patch("apps.records.views.retrieve", new_callable=AsyncMock)
    @patch("apps.records.views.build_messages", new_callable=AsyncMock)
    @patch("apps.records.views.complete", new_callable=AsyncMock)
    async def test_non_streaming_returns_json(
        self, mock_complete, mock_build, mock_retrieve, async_client
    ):
        from django.http import JsonResponse

        mock_retrieve.return_value = []
        mock_build.return_value = ([{"role": "user", "content": "test"}], {})
        mock_complete.return_value = JsonResponse({"id": "chatcmpl-123", "choices": []})

        payload = {
            "model": "agent-moss",
            "messages": [{"role": "user", "content": "How long for permits?"}],
            "stream": False,
        }
        response = await async_client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.content)
        assert "id" in data
