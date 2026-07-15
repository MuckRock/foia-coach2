import httpx

from .config import MOSS_API_URL


class MossClient:
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=MOSS_API_URL, timeout=120.0)

    async def chat(self, messages: list[dict]) -> dict:
        """Send messages to the Moss Django API and return the completion response."""
        response = await self._client.post(
            "/v1/chat/completions",
            json={"messages": messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()
