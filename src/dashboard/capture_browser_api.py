"""Two-phase browser capture admission and retirement-only acknowledgements."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from src.dashboard.auth import (
    DashboardPrincipal, require_admin, require_dashboard_origin, session_guard,
)

if TYPE_CHECKING:
    from src.memory.capture_control import CaptureControl, CaptureController


class RecorderToken(BaseModel):
    """A retirement-only secret, never administrative unmute authority."""

    model_config = ConfigDict(extra="forbid")
    token: StrictStr = Field(min_length=43, max_length=43)


def _actor(request: Request, principal: DashboardPrincipal) -> CaptureController:
    from src.memory.capture_control import CaptureController

    return CaptureController.from_dashboard(
        principal, session_guard(request, frozenset({"admin:manage"}), principal))


def register_capture_browser_routes(app: FastAPI, control: CaptureControl) -> None:
    """Bind browser reservations to actual global control and original actors."""
    app.state.capture_control = control
    router = APIRouter(prefix="/api/capture/recorders", tags=["capture"])

    async def perform(function, *args):
        try:
            return await function(*args)
        except ValueError as error:
            raise HTTPException(409, "capture_recorder_denied") from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "capture_recorder_unavailable") from error

    @router.post("/register")
    async def register(request: Request,
                       principal: DashboardPrincipal = Depends(require_admin)) -> dict:
        return await perform(control.recorders.register, _actor(request, principal))

    @router.post("/activate")
    async def activate(body: RecorderToken, request: Request,
                       principal: DashboardPrincipal = Depends(require_admin)) -> dict:
        return await perform(control.recorders.activate, body.token, _actor(request, principal))

    @router.post("/poll")
    async def poll(body: RecorderToken, request: Request,
                   principal: DashboardPrincipal = Depends(require_admin)) -> dict:
        return await perform(control.recorders.poll, body.token, _actor(request, principal))

    @router.post("/retire", dependencies=[Depends(require_dashboard_origin)])
    async def retire(body: RecorderToken, request: Request) -> dict:
        # The exact route's auth exemption permits disposal after logout only.
        # This token cannot call any administrative policy mutation.
        if not request.headers.get("origin"):
            raise HTTPException(403, "capture_recorder_origin_required")
        return await perform(control.recorders.retire, body.token)

    app.include_router(router)
