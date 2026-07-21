import secrets
import time
from urllib.parse import urlencode

import httpx
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from .config import MCP_SERVER_URL, SQUARELET_BASE, SQUARELET_CLIENT_ID, SQUARELET_CLIENT_SECRET


class SquareletAuthorizationCode(AuthorizationCode):
    squarelet_access_token: str
    squarelet_refresh_token: str | None = None


class SquareletOAuthProvider:
    """OAuth proxy: issues short-lived MCP tokens backed by Squarelet user tokens."""

    def __init__(self):
        # In-memory storage (staging only; swap for Redis in production)
        self._clients: dict[str, OAuthClientInformationFull] = {}
        # squarelet_state → {client_id, redirect_uri, mcp_state, code_challenge, ...}
        self._pending_state: dict[str, dict] = {}
        # mcp_code → SquareletAuthorizationCode
        self._auth_codes: dict[str, SquareletAuthorizationCode] = {}
        # mcp_token → AccessToken
        self._access_tokens: dict[str, AccessToken] = {}
        # mcp_refresh_token → {token: RefreshToken, squarelet_refresh_token: str}
        self._refresh_tokens: dict[str, dict] = {}
        # mcp_access_token → muckrock_refresh_token (for mid-session JWT refresh)
        self._muckrock_refresh_by_mcp_token: dict[str, str] = {}
        self._http = httpx.AsyncClient(timeout=30.0)

    # --- Client registry ---

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    # --- MuckRock token exchange ---

    async def _get_muckrock_tokens(self, oidc_access_token: str) -> tuple[str, str | None]:
        """Exchange a Squarelet OIDC token for a MuckRock API JWT."""
        resp = await self._http.post(
            f"{SQUARELET_BASE}/api/jwt/",
            data={"oidc_token": oidc_access_token},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data.get("refresh_token")

    # --- Authorization ---

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Redirect to Squarelet's authorization endpoint."""
        squarelet_state = secrets.token_urlsafe(32)
        self._pending_state[squarelet_state] = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "mcp_state": params.state,
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or [],
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "resource": params.resource,
        }
        query = urlencode({
            "response_type": "code",
            "client_id": SQUARELET_CLIENT_ID,
            "redirect_uri": f"{MCP_SERVER_URL}/oauth/callback",
            "scope": "read write",
            "state": squarelet_state,
        })
        return f"{SQUARELET_BASE}/openid/authorize?{query}"

    async def handle_callback(self, request: Request) -> Response:
        """Handle Squarelet's redirect back; exchange code and redirect to Claude."""
        code = request.query_params.get("code")
        squarelet_state = request.query_params.get("state")
        error = request.query_params.get("error")

        if error:
            return HTMLResponse(f"Authorization error: {error}", status_code=400)
        if not code or not squarelet_state:
            return HTMLResponse("Missing code or state", status_code=400)

        pending = self._pending_state.pop(squarelet_state, None)
        if pending is None:
            return HTMLResponse("Invalid or expired state", status_code=400)

        # Exchange Squarelet code for Squarelet tokens
        try:
            resp = await self._http.post(
                f"{SQUARELET_BASE}/openid/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": SQUARELET_CLIENT_ID,
                    "client_secret": SQUARELET_CLIENT_SECRET,
                    "redirect_uri": f"{MCP_SERVER_URL}/oauth/callback",
                },
            )
            resp.raise_for_status()
            tokens = resp.json()
        except Exception as exc:
            return HTMLResponse(f"Token exchange failed: {exc}", status_code=500)

        oidc_access_token = tokens.get("access_token")

        if not oidc_access_token:
            return HTMLResponse("No access token returned by Squarelet", status_code=500)

        # Exchange OIDC token for MuckRock API JWT
        try:
            muckrock_access_token, muckrock_refresh_token = await self._get_muckrock_tokens(oidc_access_token)
        except Exception as exc:
            return HTMLResponse(f"MuckRock token exchange failed: {exc}", status_code=500)

        # Generate MCP authorization code
        mcp_code = secrets.token_urlsafe(32)
        self._auth_codes[mcp_code] = SquareletAuthorizationCode(
            code=mcp_code,
            scopes=pending["scopes"],
            expires_at=time.time() + 300,  # 5-minute window
            client_id=pending["client_id"],
            code_challenge=pending["code_challenge"],
            redirect_uri=AnyUrl(pending["redirect_uri"]),
            redirect_uri_provided_explicitly=pending["redirect_uri_provided_explicitly"],
            resource=pending["resource"],
            squarelet_access_token=muckrock_access_token,
            squarelet_refresh_token=muckrock_refresh_token,
        )

        redirect_url = construct_redirect_uri(
            pending["redirect_uri"],
            code=mcp_code,
            state=pending["mcp_state"],
        )
        return RedirectResponse(redirect_url, status_code=302)

    # --- Authorization code exchange ---

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> SquareletAuthorizationCode | None:
        entry = self._auth_codes.get(authorization_code)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._auth_codes[authorization_code]
            return None
        return entry

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: SquareletAuthorizationCode
    ) -> OAuthToken:
        # Single-use: remove code immediately
        self._auth_codes.pop(authorization_code.code, None)

        mcp_token = secrets.token_urlsafe(32)
        mcp_refresh = secrets.token_urlsafe(32)

        self._access_tokens[mcp_token] = AccessToken(
            token=mcp_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + 3600,
            claims={"squarelet_access_token": authorization_code.squarelet_access_token},
        )
        self._refresh_tokens[mcp_refresh] = {
            "token": RefreshToken(
                token=mcp_refresh,
                client_id=client.client_id,
                scopes=authorization_code.scopes,
            ),
            "squarelet_refresh_token": authorization_code.squarelet_refresh_token,
        }
        if authorization_code.squarelet_refresh_token:
            self._muckrock_refresh_by_mcp_token[mcp_token] = authorization_code.squarelet_refresh_token

        return OAuthToken(
            access_token=mcp_token,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(authorization_code.scopes),
            refresh_token=mcp_refresh,
        )

    # --- Token management ---

    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access_tokens.get(token)
        if at is None:
            return None
        if at.expires_at and time.time() > at.expires_at:
            del self._access_tokens[token]
            return None
        return at

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        entry = self._refresh_tokens.get(refresh_token)
        if entry is None:
            return None
        return entry["token"]

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        entry = self._refresh_tokens.pop(refresh_token.token, None)
        if entry is None:
            raise TokenError(error="invalid_grant", error_description="Refresh token not found")

        muckrock_refresh = entry.get("squarelet_refresh_token")
        if not muckrock_refresh:
            raise TokenError(error="invalid_grant", error_description="No MuckRock refresh token available")

        try:
            resp = await self._http.post(
                f"{SQUARELET_BASE}/api/refresh/",
                json={"refresh": muckrock_refresh},
            )
            resp.raise_for_status()
            tokens = resp.json()
        except Exception as exc:
            raise TokenError(error="invalid_grant", error_description=str(exc))

        new_squarelet_access = tokens.get("access_token")
        new_squarelet_refresh = tokens.get("refresh_token", muckrock_refresh)

        if not new_squarelet_access:
            raise TokenError(error="invalid_grant", error_description="No access token returned by MuckRock")

        use_scopes = scopes or refresh_token.scopes
        mcp_token = secrets.token_urlsafe(32)
        mcp_refresh = secrets.token_urlsafe(32)

        self._access_tokens[mcp_token] = AccessToken(
            token=mcp_token,
            client_id=client.client_id,
            scopes=use_scopes,
            expires_at=int(time.time()) + 3600,
            claims={"squarelet_access_token": new_squarelet_access},
        )
        self._refresh_tokens[mcp_refresh] = {
            "token": RefreshToken(
                token=mcp_refresh,
                client_id=client.client_id,
                scopes=use_scopes,
            ),
            "squarelet_refresh_token": new_squarelet_refresh,
        }
        self._muckrock_refresh_by_mcp_token[mcp_token] = new_squarelet_refresh

        return OAuthToken(
            access_token=mcp_token,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(use_scopes),
            refresh_token=mcp_refresh,
        )

    async def refresh_muckrock_token(self, mcp_token: str) -> str:
        """Refresh the MuckRock JWT mid-session without requiring a new OAuth browser flow."""
        at = self._access_tokens.get(mcp_token)
        if at is None:
            raise RuntimeError("MCP token not found")

        muckrock_refresh = self._muckrock_refresh_by_mcp_token.get(mcp_token)
        if not muckrock_refresh:
            raise RuntimeError("No MuckRock refresh token available for this session")

        resp = await self._http.post(
            f"{SQUARELET_BASE}/api/refresh/",
            json={"refresh": muckrock_refresh},
        )
        resp.raise_for_status()
        data = resp.json()

        new_access = data["access_token"]
        new_refresh = data.get("refresh_token", muckrock_refresh)

        # Update claims in place so subsequent load_access_token calls see the new token
        if at.claims is None:
            at.claims = {}
        at.claims["squarelet_access_token"] = new_access
        self._muckrock_refresh_by_mcp_token[mcp_token] = new_refresh

        return new_access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
            self._muckrock_refresh_by_mcp_token.pop(token.token, None)
        else:
            self._refresh_tokens.pop(token.token, None)
