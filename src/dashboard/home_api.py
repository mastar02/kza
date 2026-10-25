"""Authenticated home routes backed exclusively by injected real services."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.base import RequestResponseEndpoint

from src.dashboard.auth import DashboardPrincipal, require_dashboard_session
from src.audio.microphone_status import MicrophoneRoom, MicrophonesSnapshot
from src.dashboard.home_catalog import HomeCatalogService, HomeCatalogSnapshot
from src.home_assistant.home_actions import (
    HomeActionConflict, HomeActionRequest, HomeActionResult, HomeActionService,
    validate_home_action,
)
from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant.control_policy import ActorKind, HomeActor
from src.home_assistant.operations import PARAMETER_FIELDS, validate_parameters
from src.rooms.room_context import RoomConfig


@dataclass(frozen=True)
class HomeServices:
    """Shared home dependencies; the caller owns action shutdown and HA."""

    catalog: HomeCatalogService
    actions: HomeActionService | None
    microphone_snapshot_provider: Callable[[], MicrophonesSnapshot | None] | None


def build_microphone_rooms(rooms: Mapping[str, RoomConfig]) -> tuple[MicrophoneRoom, ...]:
    """Adapt every configured room, retaining those without a capture device."""
    return tuple(
        MicrophoneRoom(
            room.room_id, room.name, room.mic_usb_port, device_index=room.mic_device_index,
        ) for room in rooms.values()
    )


def build_home_services(
    ha_client: HomeAssistantClient | None,
    rooms: Mapping[str, RoomConfig],
    microphone_snapshot_provider: Callable[[], MicrophonesSnapshot | None] | None,
    *, actions: HomeActionService | None = None,
) -> HomeServices:
    """Construct home services without reads, tasks, discovery, or HA ownership.

    Args:
        ha_client: Shared HA transport, never closed by the constructed services.
        rooms: All configured rooms, including rooms without a microphone.
        microphone_snapshot_provider: Cached-memory getter, not called here.

    Returns:
        A single catalog/action pair for dashboard and subsequent consumers.
    """
    controls = actions if actions is not None else (HomeActionService(ha_client) if ha_client is not None else None)
    return HomeServices(
        HomeCatalogService(ha_client, rooms, policy=controls.policy if controls is not None else None),
        controls,
        microphone_snapshot_provider,
    )


def register_home_frontend(app: FastAPI) -> None:
    """Register the exact home root and assets before a catch-all static mount."""
    home = Path(__file__).parent / "frontend" / "home"

    @app.get("/", include_in_schema=False)
    async def home_index() -> FileResponse:
        return FileResponse(home / "index.html")

    app.mount("/home", StaticFiles(directory=home, html=True), name="home_assets")


class HomeActionBody(BaseModel):
    """Strict HTTP intent; identity is always supplied by the server session."""

    model_config = ConfigDict(strict=True, extra="forbid")

    entity_id: str
    operation: str
    request_id: str
    brightness_pct: int | None = Field(default=None, ge=0, le=100)
    color_temp_kelvin: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, allow_inf_nan=False)
    hvac_mode: str | None = None
    position: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def validate_intent(self) -> "HomeActionBody":
        """Apply the same allowlist used by non-HTTP callers."""
        validate_parameters(self.entity_id, self.operation, {name: getattr(self, name) for name in PARAMETER_FIELDS if name in self.model_fields_set})
        validate_home_action(HomeActionRequest(**self.model_dump()))
        return self


class LegacyHomeActionBody(BaseModel):
    """Legacy path and service remain outside a restricted idempotent body."""

    model_config = ConfigDict(strict=True, extra="forbid")

    request_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    brightness_pct: int | None = Field(default=None, ge=0, le=100)


async def execute_home_action(
    actions: HomeActionService | None, request: HomeActionRequest,
    principal: DashboardPrincipal,
) -> HomeActionResult:
    """Enforce session permissions and translate intent conflicts to HTTP."""
    if "home:control" not in principal.permissions:
        raise HTTPException(status_code=403, detail="home_control_denied")
    if actions is None:
        raise HTTPException(status_code=503, detail="Home Assistant no disponible")
    try:
        validate_home_action(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        return await actions.execute(request, actor=home_actor(principal))
    except HomeActionConflict as error:
        raise HTTPException(status_code=409, detail="request_id_body_conflict") from error


def home_actor(principal: DashboardPrincipal) -> HomeActor:
    """Adapt a validated dashboard session without accepting identity in JSON."""
    return HomeActor(ActorKind.DASHBOARD, principal.subject, principal.permissions)


def register_home_routes(
    app: FastAPI, *, catalog: HomeCatalogService | None, actions: HomeActionService | None = None,
    microphone_snapshot_provider: Callable[[], MicrophonesSnapshot | None] | None = None,
) -> None:
    """Register home routes before the dashboard's static mount.

    Args:
        app: Dashboard FastAPI application.
        catalog: Real catalog provider, or None when not configured.
        actions: Optional control provider, integrated by the action task.
        microphone_snapshot_provider: Synchronous in-memory telemetry getter.
    """

    @app.middleware("http")
    async def private_response_policy(request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Prevent storage of private responses, including handled HTTP errors."""
        response = await call_next(request)
        if request.url.path.startswith(("/api/home/", "/api/routines/")) or request.url.path == "/api/routines":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/home/catalog", response_model=HomeCatalogSnapshot)
    async def home_catalog(
        principal: DashboardPrincipal = Depends(require_dashboard_session),
    ) -> HomeCatalogSnapshot:
        if "home:read" not in principal.permissions:
            raise HTTPException(status_code=403, detail="home_read_denied")
        if catalog is None:
            raise HTTPException(status_code=503, detail="home_catalog_unavailable")
        snapshot = await catalog.snapshot()
        if snapshot.observed_at is None:
            raise HTTPException(status_code=503, detail=snapshot.reason or "home_catalog_unavailable")
        return snapshot

    @app.get("/api/home/microphones", response_model=MicrophonesSnapshot)
    async def home_microphones(
        response: Response,
        principal: DashboardPrincipal = Depends(require_dashboard_session),
    ) -> MicrophonesSnapshot:
        """Return current cached evidence without discovering or opening audio."""
        no_store = {"Cache-Control": "no-store"}
        if "home:read" not in principal.permissions:
            raise HTTPException(status_code=403, detail="home_read_denied", headers=no_store)
        try:
            snapshot = microphone_snapshot_provider() if microphone_snapshot_provider is not None else None
        except Exception:
            raise HTTPException(
                status_code=503, detail="microphone_snapshot_unavailable", headers=no_store,
            ) from None
        if snapshot is None:
            raise HTTPException(status_code=503, detail="microphone_snapshot_unavailable", headers=no_store)
        response.headers["Cache-Control"] = "no-store"
        return snapshot

    @app.post("/api/home/actions", response_model=HomeActionResult)
    async def home_action(
        body: HomeActionBody,
        principal: DashboardPrincipal = Depends(require_dashboard_session),
    ) -> HomeActionResult:
        return await execute_home_action(actions, HomeActionRequest(**body.model_dump()), principal)

    @app.get("/api/home/actions", response_model=HomeActionResult)
    async def home_action_receipt(
        request_id: str = Query(pattern=r"^[A-Za-z0-9_.:-]{1,128}$"),
        principal: DashboardPrincipal = Depends(require_dashboard_session),
    ) -> HomeActionResult:
        if "home:read" not in principal.permissions:
            raise HTTPException(status_code=403, detail="home_read_denied")
        if actions is None:
            raise HTTPException(status_code=503, detail="home_actions_unavailable")
        receipt = actions.get_receipt(actor=home_actor(principal), request_id=request_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="home_action_receipt_not_found")
        return receipt
