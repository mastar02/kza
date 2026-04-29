"""
Auth para endpoints mutativos del dashboard.

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
from typing import Optional

from fastapi import Cookie, Header, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

ENV_VAR = "KZA_DASHBOARD_TOKEN"
COOKIE_NAME = "kza_dashboard_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 días


def _expected_token() -> Optional[str]:
    return os.environ.get(ENV_VAR) or None


def auth_configured() -> bool:
    """True si hay token configurado en env. Si False, todo público."""
    return _expected_token() is not None


def check_token(header_token: Optional[str], cookie_token: Optional[str]) -> bool:
    expected = _expected_token()
    if expected is None:
        return True  # sin token configurado, no chequeamos
    received = header_token
    if received and received.lower().startswith("bearer "):
        received = received.split(" ", 1)[1].strip()
    if received and secrets.compare_digest(received, expected):
        return True
    if cookie_token and secrets.compare_digest(cookie_token, expected):
        return True
    return False


async def require_admin(
    authorization: Optional[str] = Header(None),
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> None:
    """FastAPI dependency. Levanta 401 si el token falla."""
    if check_token(authorization, kza_dashboard_token):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="auth required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def login_response(token: str) -> JSONResponse:
    """Setea cookie httpOnly tras un POST /api/admin/auth/login válido."""
    if not check_token(None, token):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"ok": False, "detail": "token inválido"},
        )
    resp = JSONResponse(content={"ok": True})
    resp.set_cookie(
        key=COOKIE_NAME, value=token,
        max_age=COOKIE_MAX_AGE, httponly=True,
        samesite="strict", secure=False,  # secure=False porque dashboard es HTTP en LAN
    )
    return resp


def logout_response() -> JSONResponse:
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(key=COOKIE_NAME)
    return resp


def generate_token() -> str:
    """Helper para usuarios que quieren generar un token al deploy."""
    return secrets.token_urlsafe(32)
