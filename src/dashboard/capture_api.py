"""Administrative, revision-bound control of durable capture admission."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from src.dashboard.auth import require_dashboard_session, session_guard
from src.dashboard.authority import DashboardPrincipal
from src.memory.capture_control import CaptureController


class CaptureBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    muted: bool
    expected_revision: int = Field(ge=0)


class _CaptureRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def no_store(request: Request):
            try:
                response = await original(request)
            except HTTPException as error:
                error.headers = {**(error.headers or {}), "Cache-Control": "no-store"}
                raise
            except Exception:
                return JSONResponse(
                    status_code=503, content={"detail": "capture_control_unavailable"},
                    headers={"Cache-Control": "no-store"},
                )
            response.headers["Cache-Control"] = "no-store"
            return response

        return no_store


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def register_capture_routes(app, *, control=None) -> None:
    router = APIRouter(prefix="/api/capture/control", route_class=_CaptureRoute)

    async def controller(
        request: Request,
        principal: DashboardPrincipal = Depends(require_dashboard_session),
    ):
        try:
            return CaptureController.from_dashboard(
                principal, session_guard(request, frozenset({"admin:manage"}), principal),
            )
        except PermissionError:
            raise HTTPException(403, detail="capture_control_denied") from None

    def service():
        if control is None:
            raise HTTPException(503, detail="capture_control_unavailable")
        return control

    @router.get("")
    async def status(actor: CaptureController = Depends(controller)):
        try:
            actor.require_current()
            return service().snapshot()
        except PermissionError:
            raise HTTPException(403, detail="capture_control_denied") from None
        except RuntimeError:
            raise HTTPException(503, detail="capture_control_unavailable") from None

    @router.post("", status_code=202)
    async def change(request: Request, actor: CaptureController = Depends(controller)):
        if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json":
            raise HTTPException(415, detail="capture_control_json_required")
        payload = body = None
        try:
            try:
                payload = json.loads(await request.body(), object_pairs_hook=_unique_object)
                body = CaptureBody.model_validate(payload)
            except ValueError:
                raise HTTPException(422, detail="invalid_capture_control_body") from None
            try:
                actor.require_current()
                return service().request(body.muted, body.expected_revision, actor)
            except PermissionError:
                raise HTTPException(403, detail="capture_control_denied") from None
            except ValueError:
                raise HTTPException(409, detail="capture_control_revision_or_transition_conflict") from None
            except RuntimeError:
                raise HTTPException(503, detail="capture_control_unavailable") from None
        finally:
            payload = body = None

    app.include_router(router)
