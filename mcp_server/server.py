import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent
from starlette.requests import Request
from starlette.responses import Response

from .config import MCP_PORT, MCP_SERVER_URL
from .moss_client import MossClient
from .muckrock_client import MuckRockClient
from .oauth_provider import SquareletOAuthProvider

provider = SquareletOAuthProvider()

mcp = FastMCP(
    "Agent Moss",
    instructions=(
        "Agent Moss helps journalists identify public records to request and file "
        "FOIA requests through MuckRock. Start with search_records to find relevant "
        "records, then use MuckRock tools to look up agencies and file requests."
    ),
    host="0.0.0.0",
    port=MCP_PORT,
    auth_server_provider=provider,
    auth=AuthSettings(
        issuer_url=MCP_SERVER_URL,
        resource_server_url=f"{MCP_SERVER_URL}/mcp",
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["read", "write"],
            default_scopes=["read", "write"],
        ),
    ),
)

moss = MossClient()
muckrock = MuckRockClient()


async def _muckrock_call(fn: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> Any:
    """Call a MuckRock client method with the current session token.

    On 401, refreshes the MuckRock JWT via the provider and retries once —
    no OAuth browser flow required.
    """
    mcp_token_obj = get_access_token()
    if mcp_token_obj is None:
        raise RuntimeError("Authentication required")

    # Load fresh token from provider — reflects any refresh that happened earlier
    # in this session (e.g. a previous tool call already refreshed it)
    fresh_at = await provider.load_access_token(mcp_token_obj.token)
    if fresh_at is None:
        raise RuntimeError("MCP token expired")
    token = (fresh_at.claims or {}).get("squarelet_access_token")
    if not token:
        raise RuntimeError("No MuckRock token in auth claims")

    try:
        return await fn(token, *args, **kwargs)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 401:
            raise

    # MuckRock JWT expired — refresh and retry
    token = await provider.refresh_muckrock_token(mcp_token_obj.token)
    return await fn(token, *args, **kwargs)


@mcp.custom_route("/oauth/callback", methods=["GET"])
async def oauth_callback(request: Request) -> Response:
    return await provider.handle_callback(request)


# --- Moss tools (record discovery) ---

@mcp.tool()
async def search_records(query: str, jurisdiction: str) -> str:
    """Search government retention schedule records for a specific jurisdiction.

    Use this to find what public records exist for a topic in a given US state.
    The query should describe what records or information the user is looking for.
    Jurisdiction should be a US state name (e.g., 'Colorado', 'Texas').

    Returns an AI-generated response with a table of matching records including
    record titles, custodians, and retention periods.
    """
    messages = [
        {"role": "user", "content": f"[{jurisdiction}] {query}"},
    ]
    result = await moss.chat(messages)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content or json.dumps(result)


@mcp.tool()
async def chat(messages: list[dict]) -> str:
    """Multi-turn conversation with Agent Moss for follow-up questions and clarifications.

    Use this for follow-up questions after an initial search_records call.
    Pass the full conversation history in OpenAI message format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Returns the assistant's response text.
    """
    result = await moss.chat(messages)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content or json.dumps(result)


# --- MuckRock tools (FOIA workflow) ---

@mcp.tool()
async def search_agencies(query: str, jurisdiction: str | None = None, ctx: Context = None) -> str:
    """Search for government agencies on MuckRock.

    Use this to find the agency you need to send a FOIA request to.
    Provide a jurisdiction name (e.g., 'Denver, CO' or 'Colorado') to narrow results.

    Returns a list of matching agencies with their IDs, names, and jurisdictions.
    """
    jurisdiction_id = None
    if jurisdiction:
        jurisdiction_id = await _muckrock_call(muckrock.get_jurisdiction_id, jurisdiction)

    result = await _muckrock_call(muckrock.search_agencies, query, jurisdiction_id)

    if result.get("count", 0) == 0:
        if ctx is not None:
            try:
                await ctx.info(f"No results for '{query}', asking client for alternatives via sampling")
                sampling_result = await ctx.session.create_message(
                    messages=[SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                f"A user is searching for a government agency on MuckRock related to: \"{query}\"."
                                " Suggest 3 alternative agency name search terms that might match "
                                "(e.g. synonyms or common naming conventions used by government agencies). "
                                "Reply with just the terms, one per line, nothing else."
                            ),
                        ),
                    )],
                    max_tokens=100,
                )
                alternatives_text = sampling_result.content.text.strip()
                await ctx.info(f"Sampling suggested alternatives: {alternatives_text!r}")
                alternatives = [t.strip() for t in alternatives_text.splitlines() if t.strip()]

                for term in alternatives:
                    await ctx.info(f"Trying alternative term: '{term}'")
                    result = await _muckrock_call(muckrock.search_agencies, term, jurisdiction_id)
                    if result.get("count", 0) > 0:
                        result["search_term_used"] = term
                        break
            except Exception as e:
                await ctx.warning(f"Sampling failed: {e}")

        if result.get("count", 0) == 0 and jurisdiction_id:
            result = await _muckrock_call(muckrock.search_agencies, None, jurisdiction_id)

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_agency(agency_id: int) -> str:
    """Get detailed information about a specific agency on MuckRock.

    Returns contact info, average response time, fee information, and status.
    Use search_agencies first to find the agency ID.
    """
    result = await _muckrock_call(muckrock.get_agency, agency_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def file_request(agency_id: int, title: str, document_request: str, full_text: str = "") -> str:
    """File a FOIA/public records request through MuckRock.

    IMPORTANT: Always confirm with the user before calling this tool.
    This will submit a real public records request to the specified agency.

    Args:
        agency_id: MuckRock agency ID (use search_agencies to find it)
        title: Short title for the request
        document_request: The specific records being requested
        full_text: Optional full request letter text (MuckRock generates one if omitted)

    Returns the created request details including tracking URL.
    """
    result = await _muckrock_call(muckrock.file_request, agency_id, title, document_request, full_text)
    return json.dumps(result, indent=2)


@mcp.tool()
async def my_requests(user_id: int | None = None, status: str | None = None, page_size: int = 10) -> str:
    """List FOIA requests on MuckRock for the authenticated user.

    Optionally filter by user_id and/or status: 'submitted', 'ack', 'processed',
    'appealing', 'fix', 'payment', 'rejected', 'no_docs', 'done', 'partial', 'abandoned'.
    Use page_size to control how many results are returned (default 10).

    Returns a list of requests with IDs, titles, agencies, and statuses.
    """
    result = await _muckrock_call(muckrock.my_requests, user_id, status, page_size)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_request(request_id: int) -> str:
    """Get detailed status of a specific FOIA request on MuckRock.

    Returns full request details including communications and documents received.
    Use my_requests first to find the request ID.
    """
    result = await _muckrock_call(muckrock.get_request, request_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_requests(
    query: str,
    agency: int | None = None,
    jurisdiction: str | None = None,
    status: str | None = None,
) -> str:
    """Search all public FOIA requests on MuckRock.

    Useful for finding examples of past requests to similar agencies or for
    similar records. Can filter by agency ID, jurisdiction, or status.

    Returns a list of matching public requests.
    """
    result = await _muckrock_call(muckrock.search_requests, query, agency, jurisdiction, status)
    return json.dumps(result, indent=2)
