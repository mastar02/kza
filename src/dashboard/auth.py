"""
Authentication for private HTTP and WebSocket dashboard routes.

Modelo: token compartido único guardado en `KZA_DASHBOARD_TOKEN` env var.
Comparado con `secrets.compare_digest` para evitar timing attacks. El token
se acepta vía header `Authorization: Bearer <token>` o cookie `kza_dashboard_token`.

Esto NO es OAuth ni multi-user — es lock mínimo para evitar que cualquier
device en la LAN haga acciones imperativas (borrar usuarios, reiniciar
servicios). Para single-user on-prem es suficiente.
"""

import logging
import os
import secrets
from dataclasses import dataclass
from typing import NoReturn, Optional
from urllib.parse import urlsplit

from fastapi import Cookie, Depends, Header, HTTPException, WebSocketException, status
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

logger = logging.getLogger(__name__)

ENV_VAR = "KZA_DASHBOARD_TOKEN"
COOKIE_NAME = "kza_dashboard_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 días


@dataclass(frozen=True)
class DashboardPrincipal:
    """Server-owned identity and permissions for a dashboard session."""

    subject: str
    permissions: frozenset[str]


def _expected_token() -> Optional[str]:
    return os.environ.get(ENV_VAR) or None


def auth_configured() -> bool:
    """Return whether the dashboard credential is configured."""
    return _expected_token() is not None


def check_token(header_token: Optional[str], cookie_token: Optional[str]) -> bool:
    expected = _expected_token()
    if expected is None:
        return False
    received = header_token
    if received and received.lower().startswith("bearer "):
        received = received.split(" ", 1)[1].strip()
    if received and secrets.compare_digest(received, expected):
        return True
    if cookie_token and secrets.compare_digest(cookie_token, expected):
        return True
    return False


def _deny_session(
    request: HTTPConnection, status_code: int, detail: str, headers: dict[str, str] | None = None,
) -> NoReturn:
    if request.scope["type"] == "websocket":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=detail)
    raise HTTPException(status_code=status_code, detail=detail, headers=headers)


async def require_dashboard_origin(
    request: HTTPConnection,
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> None:
    """Reject an invalid or foreign Origin, including reads and bearer clients.

    Native clients may omit Origin. Browsers that send one must match the
    scheme, host and effective port as seen through the trusted local proxy.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return
    scheme = {"ws": "http", "wss": "https"}.get(request.url.scheme, request.url.scheme)
    expected_origin = f"{scheme}://{request.url.netloc}"

    def identity(value: str) -> tuple[str, str, int] | None:
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
            ):
                return None
            port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
            return parsed.scheme, parsed.hostname.lower(), port
        except ValueError:
            return None

    if identity(origin) is None or identity(origin) != identity(expected_origin):
        _deny_session(request, status.HTTP_403_FORBIDDEN, "origin denied")


async def require_dashboard_session(
    request: HTTPConnection,
    authorization: Optional[str] = Header(None),
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> DashboardPrincipal:
    """Require a configured, valid credential and return its fixed principal."""
    if not auth_configured():
        _deny_session(request, status.HTTP_503_SERVICE_UNAVAILABLE, "dashboard_auth_not_configured")
    if not check_token(authorization, kza_dashboard_token):
        _deny_session(request, status.HTTP_401_UNAUTHORIZED, "auth required", {"WWW-Authenticate": "Bearer"})
    await require_dashboard_origin(request, kza_dashboard_token)
    return DashboardPrincipal(
        subject="dashboard-admin",
        permissions=frozenset({"home:read", "home:control", "routines:manage", "memory:review"}),
    )


async def require_dashboard_access(
    request: HTTPConnection,
    authorization: Optional[str] = Header(None),
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> DashboardPrincipal | None:
    """Protect application routes by default, with exact public entry points.

    Only the home HTML and token login are public application routes. Static
    mounts contain UI assets and have no private data. New API/WS routes inherit
    this dependency even when their handler omits a session parameter.
    """
    public = request.scope["type"] == "http" and (
        (request.scope["method"] == "GET" and request.url.path == "/")
        or (request.scope["method"] == "POST" and request.url.path == "/api/admin/auth/login")
    )
    if public:
        await require_dashboard_origin(request, kza_dashboard_token)
        return None
    return await require_dashboard_session(request, authorization, kza_dashboard_token)


async def require_admin(
    principal: DashboardPrincipal = Depends(require_dashboard_session),
) -> DashboardPrincipal:
    """Apply the same strict session boundary to existing admin endpoints."""
    return principal


def login_response(token: str, *, secure: bool = False) -> JSONResponse:
    """Setea cookie httpOnly tras un POST /api/admin/auth/login válido."""
    if not auth_configured():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ok": False, "detail": "dashboard_auth_not_configured"},
        )
    if not check_token(None, token):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"ok": False, "detail": "token inválido"},
        )
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key=COOKIE_NAME, value=token,
        max_age=COOKIE_MAX_AGE, httponly=True,
        samesite="strict", secure=secure,
    )
    return resp


def logout_response() -> JSONResponse:
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(key=COOKIE_NAME)
    return resp


def generate_token() -> str:
    """Helper para usuarios que quieren generar un token al deploy."""
    return secrets.token_urlsafe(32)
