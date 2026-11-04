"""Own bounded enrollment request ingestion before multipart parsing starts."""
from __future__ import annotations

import asyncio
import re
from contextvars import Context
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.dashboard.auth import COOKIE_NAME, require_admin, require_dashboard_session, session_guard

if TYPE_CHECKING:
    from src.memory.capture_control import CaptureControl


MAX_ENROLLMENT_BODY_BYTES = 16 * 4 * 1024 * 1024 + 64 * 1024


class CaptureIngressMiddleware:
    """Seal uploads before parsing and retain actual parser/form/native cleanup.

    Args:
        app: Actual ASGI application, including its form cleanup stack.
        control: Actual global admission owner.
    """

    def __init__(self, app: ASGIApp, control: CaptureControl) -> None:
        self.app = app
        self.control = control
        self._tasks: set[asyncio.Task] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate before the first receive and shield actual owned work."""
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or re.fullmatch(r"/api/admin/users/[^/]+/enroll", scope.get("path", "")) is None):
            await self.app(scope, receive, send)
            return
        try:
            request = Request(scope)
            principal = await require_dashboard_session(
                request, request.headers.get("authorization"), request.cookies.get(COOKIE_NAME))
            await require_admin(principal)
            current = session_guard(request, frozenset({"admin:manage"}), principal)
            lease = self.control.admit_upload(current)
        except HTTPException as error:
            await JSONResponse({"detail": error.detail}, error.status_code,
                               headers=error.headers)(scope, receive, send)
            return
        except (ValueError, RuntimeError):
            await JSONResponse({"detail": "capture_muted"}, 409)(scope, receive, send)
            return
        scope["kza_capture_upload"] = lease

        async def own() -> None:
            lease.attach_task(asyncio.current_task())
            data = bytearray()
            replay_body = None
            try:
                while True:
                    if lease.stop.is_set() or lease.current() is not True:
                        raise HTTPException(409, "capture_muted")
                    reading = asyncio.create_task(receive(), context=Context())
                    stopped = asyncio.create_task(lease.stop.wait(), context=Context())
                    try:
                        await asyncio.wait((reading, stopped), return_when=asyncio.FIRST_COMPLETED)
                        if lease.stop.is_set() or lease.current() is not True:
                            raise HTTPException(409, "capture_muted")
                        message = reading.result()
                    finally:
                        for child in (reading, stopped):
                            if not child.done():
                                child.cancel()
                        await asyncio.gather(reading, stopped, return_exceptions=True)
                    if message["type"] != "http.request":
                        raise HTTPException(400, "enrollment_upload_disconnected")
                    chunk = message.get("body", b"")
                    if len(data) + len(chunk) > MAX_ENROLLMENT_BODY_BYTES:
                        raise HTTPException(413, "enrollment_upload_too_large")
                    data.extend(chunk)
                    if not message.get("more_body", False):
                        break
                if lease.current() is not True:
                    raise HTTPException(409, "capture_muted")
                replay_body = bytes(data)
                data.clear()
                delivered = False

                async def replay() -> dict:
                    nonlocal delivered, replay_body
                    if delivered:
                        return {"type": "http.disconnect"}
                    delivered = True
                    body, replay_body = replay_body, None
                    return {"type": "http.request", "body": body, "more_body": False}

                # FastAPI's actual middleware AsyncExitStack closes FormData
                # after its router returns. Never cancel this parser/native path.
                await self.app(scope, replay, send)
            except HTTPException as error:
                await JSONResponse({"detail": error.detail}, error.status_code,
                                   headers=error.headers)(scope, receive, send)
            finally:
                data.clear()
                data = replay_body = None
                lease.retire()

        task = asyncio.create_task(own(), context=Context())
        self._tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished, context=Context())
        await asyncio.shield(task)
