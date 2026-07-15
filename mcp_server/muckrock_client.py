import httpx

from .config import MUCKROCK_API_URL


class MuckRockClient:
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=MUCKROCK_API_URL, timeout=30.0)

    async def _request(self, method: str, path: str, access_token: str, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await self._client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    async def get_jurisdiction_id(self, access_token: str, jurisdiction: str) -> int | None:
        """Resolve a jurisdiction name like 'Denver, CO' or 'Colorado' to a MuckRock jurisdiction ID."""
        parts = [p.strip() for p in jurisdiction.split(",")]

        if len(parts) == 1:
            resp = await self._request("GET", "/jurisdictions/", access_token, params={"name": parts[0], "level": "s"})
            results = resp.json().get("results", [])
            return results[0]["id"] if results else None
        else:
            local_name, state_abbrev = parts[0], parts[1].strip()
            state_resp = await self._request("GET", "/jurisdictions/", access_token, params={"abbrev": state_abbrev, "level": "s"})
            state_results = state_resp.json().get("results", [])
            if not state_results:
                return None
            state_id = state_results[0]["id"]
            local_resp = await self._request("GET", "/jurisdictions/", access_token, params={"name": local_name, "level": "l", "parent": state_id})
            local_results = local_resp.json().get("results", [])
            exact = [r for r in local_results if r["name"].lower() == local_name.lower()]
            return exact[0]["id"] if exact else None

    async def search_agencies(self, access_token: str, query: str | None = None, jurisdiction_id: int | None = None) -> dict:
        params = {}
        if query:
            params["search"] = query
        if jurisdiction_id:
            params["jurisdiction__id"] = jurisdiction_id
        response = await self._request("GET", "/agencies/", access_token, params=params)
        return response.json()

    async def get_agency(self, access_token: str, agency_id: int) -> dict:
        response = await self._request("GET", f"/agencies/{agency_id}/", access_token)
        return response.json()

    async def file_request(
        self, access_token: str, agency_id: int, title: str, document_request: str, full_text: str = ""
    ) -> dict:
        payload = {
            "agencies": [agency_id],
            "title": title,
            "requested_docs": document_request,
        }
        if full_text:
            payload["full_text"] = full_text
        response = await self._request("POST", "/requests/", access_token, json=payload)
        return response.json()

    async def get_me(self, access_token: str) -> dict:
        response = await self._request("GET", "/users/me/", access_token)
        return response.json()

    async def _get_user_id(self, access_token: str) -> int:
        me = await self.get_me(access_token)
        return me["id"]

    async def my_requests(self, access_token: str, user_id: int | None = None, status: str | None = None, page_size: int = 10) -> dict:
        if user_id is None:
            user_id = await self._get_user_id(access_token)
        params = {"page_size": page_size, "user": user_id}
        if status:
            params["status"] = status
        response = await self._request("GET", "/requests/", access_token, params=params)
        return response.json()

    async def get_request(self, access_token: str, request_id: int) -> dict:
        response = await self._request("GET", f"/requests/{request_id}/", access_token)
        return response.json()

    async def search_requests(
        self,
        access_token: str,
        query: str,
        agency: int | None = None,
        jurisdiction: str | None = None,
        status: str | None = None,
    ) -> dict:
        params = {"search": query}
        if agency:
            params["agency"] = agency
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        if status:
            params["status"] = status
        response = await self._request("GET", "/requests/", access_token, params=params)
        return response.json()
