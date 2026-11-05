"""
Authentication for private HTTP and WebSocket dashboard routes.

El runtime comparte una autoridad privada para roles y sesiones individuales.
Cada operación conserva su lease original durante lecturas, trabajo nativo y
entrega. Las integraciones standalone sin autoridad admiten únicamente el bearer
administrativo; no emiten sesiones ni aceptan cookies con el secreto compartido.
"""

import logging
import os
import secrets
from typing import Callable, NoReturn, Optional
from urllib.parse import urlsplit

from fastapi import Cookie, Depends, Header, HTTPException, WebSocketException, status
from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

from src.dashboard.authority import DashboardPrincipal as DashboardPrincipal
from src.dashboard.authority import DashboardAuthority, DashboardLease, DashboardPermission, legacy_lease
from src.core.reader_binding import ReaderBinding

logger = logging.getLogger(__name__)

ENV_VAR = "KZA_DASHBOARD_TOKEN"
COOKIE_NAME = "kza_dashboard_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 días





def _expected_token() -> Optional[str]:
    value = os.environ.get(ENV_VAR, "").strip()
    return value if value and not value.startswith("${") else None


def auth_configured(authority: DashboardAuthority | None = None) -> bool:
    """Inspect the selected authority, without a fallback from a failed owner."""
    if authority is None:
        return _expected_token() is not None
    return isinstance(authority, DashboardAuthority) and authority.configured


def check_token(header_token: Optional[str], cookie_token: Optional[str]) -> bool:
    """Compatibility for bearer-only callers; retained routes use bound leases."""
    return legacy_lease(header_token) is not None


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


def dashboard_authority(request: HTTPConnection, *, required: bool = False) -> DashboardAuthority | None:
    """Select the injected owner; an invalid or unavailable owner fails closed."""
    authority = getattr(request.app.state, "dashboard_authority", None)
    if authority is None:
        if required:
            _deny_session(request, 503, "dashboard_authority_required")
        return None
    if not isinstance(authority, DashboardAuthority) or not authority.available:
        _deny_session(request, 503, "dashboard_authority_unavailable")
    return authority


def current_lease(request: HTTPConnection) -> DashboardLease | None:
    """Return only the original admitted binding, without reauthentication."""
    lease = request.scope.get("kza.dashboard_lease")
    return lease if type(lease) is DashboardLease else None


def session_reader(request: HTTPConnection, permissions: frozenset[str]) -> ReaderBinding | None:
    """Detach original ownership before handing off any private transport work."""
    lease = current_lease(request)
    return None if lease is None else lease.reader_binding(permissions)


def session_guard(request: HTTPConnection, permissions: frozenset[str] = frozenset(),
                  principal: DashboardPrincipal | None = None) -> Callable[[], bool]:
    """Detach a native/delivery guard from request headers and connection state."""
    lease = current_lease(request)
    if lease is None or principal is not None and lease.principal is not principal:
        return lambda: False
    return lambda: lease.current(permissions)


def voice_person_scope_requested(request: HTTPConnection) -> bool:
    """Recognize the closed metadata surface that requires extra server permission."""
    path,method = request.url.path,request.scope.get('method','GET')
    if (method == 'PUT' and path.startswith('/api/voices/profiles/')
            and path.endswith('/person') and path.count('/') == 5):
        return True
    if path == '/api/voices/profiles' and method in {'GET','HEAD','OPTIONS'}:
        value = request.query_params.get('include_person')
        return 'person_id' in request.query_params or value is not None and value.lower() not in {'false','f','0','off','no','n'}
    return False


def _route_permissions(request: HTTPConnection) -> frozenset[str]:
    path, method = request.url.path, request.scope.get("method", "GET")
    if path in {"/api/admin/auth/whoami", "/api/admin/auth/logout"}:
        return frozenset()
    if path in {"/api/admin/auth/credentials","/api/admin/auth/credential-attempts"} or path.startswith(("/api/admin/auth/credentials/","/api/admin/auth/credential-attempts/")):
        return frozenset({DashboardPermission.AUTH_MANAGE})
    read = method in {"GET", "HEAD", "OPTIONS"}
    if path == "/api/admin/users" and read:
        return frozenset({DashboardPermission.ADMIN_MANAGE})
    if path.startswith("/api/admin/users/") and path.count("/") == 4 and method == "DELETE":
        return frozenset({DashboardPermission.ADMIN_MANAGE, DashboardPermission.AUTH_MANAGE})
    if path == "/api/home/microphones":
        return frozenset({DashboardPermission.DIAGNOSTICS_READ})
    if path.startswith("/api/home/"):
        return frozenset({DashboardPermission.HOME_READ if read else DashboardPermission.HOME_CONTROL})
    if not read and path.startswith("/api/entities/") and path.endswith("/call") and path.count("/") == 4:
        return frozenset({DashboardPermission.HOME_CONTROL})
    if path == "/api/routines" or path.startswith("/api/routines/"):
        if read and ("/runs/" in path or path.endswith("/attempts")):
            return frozenset({DashboardPermission.ROUTINES_MANAGE})
        return frozenset({DashboardPermission.HOME_READ if read else DashboardPermission.ROUTINES_MANAGE})
    if path == "/ws/evaluation" or (method == "GET" and (path == "/api/evaluation/grants"
            or path.startswith("/api/evaluation/samples/") and path.count("/") == 4)):
        return frozenset({DashboardPermission.EVALUATION_READ})
    if path == "/api/voices" or path.startswith("/api/voices/"):
        if voice_person_scope_requested(request):
            return frozenset({DashboardPermission.VOICES_REVIEW,DashboardPermission.ADMIN_MANAGE,
                              *({DashboardPermission.VOICES_READ} if read else set())})
        return frozenset({DashboardPermission.VOICES_READ if read else DashboardPermission.VOICES_REVIEW})
    if path == "/api/memory" or path.startswith("/api/memory/"):
        return frozenset({DashboardPermission.MEMORY_READ if read else DashboardPermission.MEMORY_REVIEW})
    if path == "/api/capture" or path.startswith("/api/capture/"):
        return frozenset({DashboardPermission.ADMIN_MANAGE})
    if path == "/api/ambient" or path.startswith("/api/ambient/"):
        return frozenset({DashboardPermission.AMBIENT_CONTROL})
    if path == "/ws/voices":
        return frozenset({DashboardPermission.VOICES_READ, DashboardPermission.VOICES_REVIEW})
    if path == "/api/conversations" or path == "/ws/live":
        return frozenset({DashboardPermission.CONVERSATIONS_READ})
    if path == "/ws":
        return frozenset({DashboardPermission.HOME_READ})
    # Unknown surfaces inherit an operator permission, including new routes.
    return frozenset({DashboardPermission.DIAGNOSTICS_READ if read else DashboardPermission.ADMIN_MANAGE})


async def require_dashboard_session(
    request: HTTPConnection,
    authorization: Optional[str] = Header(None),
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> DashboardPrincipal:
    """Bind one server identity and its baseline route authority for this scope."""
    authority = dashboard_authority(request)
    lease = current_lease(request)
    if lease is None:
        if authority is not None:
            try:
                await authority.synchronize()
            except Exception:
                _deny_session(request, 503, "dashboard_authority_unavailable")
            if not auth_configured(authority):
                _deny_session(request, 503, "dashboard_auth_not_configured")
            lease = authority.resolve(authorization, kza_dashboard_token)
        else:
            if not auth_configured():
                _deny_session(request, 503, "dashboard_auth_not_configured")
            lease = legacy_lease(authorization)
        if lease is None:
            _deny_session(request, 401, "auth required", {"WWW-Authenticate": "Bearer"})
        request.scope["kza.dashboard_lease"] = lease
    if lease._authority is not authority or not lease.current():
        _deny_session(request, 401, "auth required", {"WWW-Authenticate": "Bearer"})
    await require_dashboard_origin(request, kza_dashboard_token)
    permissions = _route_permissions(request)
    if not lease.current(permissions):
        _deny_session(request, 403, "dashboard_permission_denied")
    request.scope["kza.dashboard_delivery_guard"] = lambda: lease.current(permissions)
    return lease.principal


async def require_dashboard_access(
    request: HTTPConnection,
    authorization: Optional[str] = Header(None),
    kza_dashboard_token: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> DashboardPrincipal | None:
    """Protect every route except the exact public and retire-only surfaces.

    Recorder retirement accepts its opaque capability after logout. Its route
    checks that capability; the exception here never authorizes capture or unmute.
    """
    public = request.scope["type"] == "http" and (
        (request.scope["method"] == "GET" and request.url.path == "/")
        or (request.scope["method"] == "POST" and request.url.path in {
            "/api/admin/auth/login", "/api/capture/recorders/retire",
            "/api/capture/native/calibration/retire",
            "/api/capture/native/wakeword-training/retire",
            "/api/capture/native/wakeword-diagnostics/retire",
            "/api/capture/native/zone-setup/retire",
            "/api/capture/native/manual-voice-command/retire",
        })
    )
    if public:
        await require_dashboard_origin(request, kza_dashboard_token)
        return None
    return await require_dashboard_session(request, authorization, kza_dashboard_token)


async def require_admin(
    principal: DashboardPrincipal = Depends(require_dashboard_session),
) -> DashboardPrincipal:
    """Require explicit administrative authority independently of authentication."""
    if DashboardPermission.ADMIN_MANAGE not in principal.permissions:
        raise HTTPException(status_code=403, detail="dashboard_admin_denied")
    return principal


async def login_response(token: str, request: HTTPConnection, *, authority: DashboardAuthority | None = None,
                         secure: bool = False) -> JSONResponse:
    """Set an opaque cookie only after the selected owner commits its session."""
    headers = {"Cache-Control": "no-store"}
    if not isinstance(authority, DashboardAuthority) or not authority.available:
        return JSONResponse(status_code=503, content={"ok": False, "detail": "dashboard_authority_required"},
                            headers=headers)
    previous = authority.resolve(None, request.cookies.get(COOKIE_NAME))
    try:
        issued = await authority.login(token, replacing=previous) if previous is not None else await authority.login(token)
    except PermissionError:
        return JSONResponse(status_code=401, content={"ok": False, "detail": "token inválido"}, headers=headers)
    except Exception:
        return JSONResponse(status_code=503, content={"ok": False, "detail": "dashboard_authority_unavailable"},
                            headers=headers)
    lease = issued.lease
    lifetime = lease.remaining_ttl()
    if not lifetime or not lease.current():
        return JSONResponse(status_code=401, content={"ok": False, "detail": "sesión vencida"}, headers=headers)
    request.scope["kza.dashboard_delivery_guard"] = lambda: lease.current()
    response = JSONResponse(content={"ok": True}, headers=headers)
    response.set_cookie(key=COOKIE_NAME, value=issued.token, max_age=lifetime,
                        httponly=True, samesite="strict", secure=secure)
    return response


async def logout_response(lease: DashboardLease, *, authority: DashboardAuthority | None = None) -> JSONResponse:
    """Confirm individual revocation; a static bearer remains a bearer."""
    headers = {"Cache-Control": "no-store"}
    try:
        revoked = await authority.logout(lease) if authority is not None else False
    except Exception:
        return JSONResponse(status_code=503, content={"ok": False, "detail": "session_revocation_unavailable"},
                            headers=headers)
    response = JSONResponse(content={"ok": True, "session_revoked": revoked}, headers=headers)
    response.delete_cookie(key=COOKIE_NAME)
    return response


def generate_token() -> str:
    """Helper para usuarios que quieren generar un token al deploy."""
    return secrets.token_urlsafe(32)
