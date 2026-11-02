"""
Observability endpoints — sirven al frontend del dashboard wireframe.

Contrato (consumido por src/dashboard/frontend/obs/src/app.jsx hydration):
    GET  /api/zones, /api/conversations, /api/ha/entities, /api/ha/actions,
         /api/llm/endpoints, /api/users, /api/alerts,
         /api/system/gpus, /api/system/services
    POST /api/llm/endpoints/{id}/clear-cooldown
    WS   /ws/live  → frames {type, payload, ts}

WS frames por type (ver src/dashboard/live_event_bus.py:LiveEventType):
    turn:     {id, user, zone, stt, intent, tts, latency_ms, success, path}
    wake:     {zone, confidence}
    alert:    {id, priority, type, zone, title, body}
    tts:      {zone, text, duration_ms}
    cooldown: {endpoint_id, until, step}

Cada colección declara fuente, estado de observación y caducidad. El modo demo
requiere use_mocks=True; ausencia o fallo en modo operativo nunca devuelve ejemplos.
"""

import asyncio
import logging
from functools import partial
from datetime import UTC, datetime
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, WebSocketException

from src.analytics.event_logger import EventLogger
from src.dashboard import observability_mocks as mocks
from src.dashboard.monitor_service import MonitorService
from src.dashboard.observation_http import ObservationHeaders
from src.core.observation import Observation, ObservationSource, ObservationReason
from src.home_assistant.capability_service import HomeCapabilityService
from src.home_assistant.ha_client import HAStatesSnapshot, STATE_OBSERVATION_TTL_S
from src.dashboard.auth import COOKIE_NAME, require_dashboard_session, session_guard, session_reader
from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.dashboard.native_work import NativeAuthorityRetired, ensure_admin_work, require_current, run_admin_work
from src.dashboard.live_event_bus import LiveEventBus

logger = logging.getLogger(__name__)


def _zones_adapter(zone_manager) -> list[dict]:
    zones_dict = zone_manager.get_all_zones()
    out = []
    for z in zones_dict.values():
        ma = z.ma1260_zone
        ma_letter = chr(ord("A") + (ma - 1)) if isinstance(ma, int) and ma > 0 else str(ma)
        out.append({
            "id": z.id, "name": z.name, "icon": "◰",
            "user": None, "lastUtterance": None,
            "audioState": z.state.value if hasattr(z.state, "value") else str(z.state),
            "volume": z.volume, "ma1260_zone": ma_letter,
            "spotify": {"playing": None},
        })
    return out


def _llm_endpoints_adapter(llm_router, metrics_tracker=None, *, before_read=None) -> list[dict]:
    if before_read is not None:
        before_read()
    endpoints = getattr(llm_router, "_endpoints", None)
    cd = getattr(llm_router, "_cd", None)
    if not endpoints:
        return []
    out = []
    for ep in endpoints:
        ep_id = getattr(ep, "id", "?")
        in_cd = (cd is not None) and (not cd.is_available(ep_id, before_read=before_read))
        cooldown_ends = None
        if in_cd and cd is not None:
            try:
                eta = cd.next_attempt_at(ep_id, before_read=before_read)
                cooldown_ends = datetime.fromtimestamp(eta).strftime("%H:%M:%S")
            except NativeAuthorityRetired:
                raise
            except Exception:
                pass
        kind = getattr(ep, "kind", None)
        kind_val = kind.value if hasattr(kind, "value") else str(kind)
        # Métricas reales rolling 5min (tps, ttft_ms p50, calls)
        tps, ttft_ms, calls, last_check_ts = None, None, None, None
        if metrics_tracker is not None:
            snap = metrics_tracker.snapshot(ep_id, before_read=before_read)
            if snap:
                tps = snap.get("tps")
                ttft_ms = snap.get("ttft_ms")
                calls = snap.get("calls")
                last_check_ts = snap.get("last_call_ts")
        if before_read is not None:
            before_read()
        out.append({
            "id": ep_id, "name": ep_id,
            "url": getattr(getattr(ep, "client", None), "base_url", "local"),
            "priority": getattr(ep, "priority", 0),
            "role": kind_val,
            "state": "cooldown" if in_cd else "unknown",
            "tps": tps, "ttft_ms": ttft_ms, "calls_5min": calls,
            "last_check": None,
            "last_call_at": datetime.fromtimestamp(last_check_ts, tz=UTC).isoformat() if type(last_check_ts) in (int, float) else None,
            "failures_7d": {},
            "cooldown_ends": cooldown_ends,
        })
    if before_read is not None:
        before_read()
    return out


def _users_adapter(user_manager, *, before_read=None) -> list[dict]:
    """Project active account scalars without loading voice vectors or preferences."""
    users = user_manager.project_accounts(before_read=before_read)
    try:
        if before_read is not None:
            before_read()
        result = [{
            "id": user.id, "name": user.name, "samples": None, "lastEnroll": None,
            "emotions": {}, "topCommands": [], "permissions": {}, "pca": [],
        } for user in users if user.is_active]
        if before_read is not None:
            before_read()
        return result
    finally:
        users = None


def _alerts_adapter(alert_manager, status: str | None) -> list[dict]:
    history = alert_manager.get_history(limit=100)
    out = []
    for a in history:
        acked = a.processed
        if status == "active" and acked:
            continue
        if status == "acked" and not acked:
            continue
        prio_name = a.priority.name.lower() if hasattr(a.priority, "name") else str(a.priority)
        prio_map = {"low": "info", "medium": "warn", "high": "warn", "critical": "critical"}
        out.append({
            "id": a.alert_id,
            "ts": a.timestamp.strftime("%H:%M:%S") if hasattr(a.timestamp, "strftime") else str(a.timestamp),
            "priority": prio_map.get(prio_name, prio_name),
            "type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
            "zone": a.details.get("zone", "—"),
            "title": a.message,
            "body": a.details.get("body", ""),
            "acked": acked,
        })
    return out


def _ha_actions_adapter(event_logger, *, limit: int = 50) -> list[dict] | None:
    """EventLogger eventos type=COMMAND → log de acciones HA."""
    try:
        events = event_logger.get_events(limit=limit * 2)
    except Exception as e:
        logger.warning(f"event_logger.get_events failed: {e}")
        return None
    out = []
    for ev in events:
        if not ev.entity_id or "." not in ev.entity_id:
            continue
        domain = ev.entity_id.split(".", 1)[0]
        ctx = ev.context or {}
        out.append({
            "id": f"act_{int(ev.timestamp * 1000)}",
            "ts": ev.datetime.strftime("%H:%M:%S"),
            "idem": ctx.get("idem", "—"),
            "user": ev.user_name or ev.user_id or "—",
            "service": f"{domain}.{ev.action}",
            "target": ev.entity_id,
            "args": str(ctx["args"]) if "args" in ctx else None,
            "ok": ctx.get("success") if type(ctx.get("success")) is bool else None,
            "lat_ms": ctx.get("latency_ms"),
        })
        if len(out) >= limit:
            break
    return out


async def _ha_entities_adapter(ha_client, *, snapshot: HAStatesSnapshot | None = None) -> list[dict] | None:
    """Project an actual typed read; failed reads cannot masquerade as empty homes."""
    if snapshot is None:
        read = getattr(ha_client, "get_state_snapshot", None)
        if not callable(read):
            return None
        snapshot = await read()
    if not isinstance(snapshot, HAStatesSnapshot) or not snapshot.available:
        return None
    return [{
        "id": state["entity_id"], "domain": state["entity_id"].partition(".")[0],
        "name": state.get("attributes", {}).get("friendly_name", state["entity_id"]),
        "state": state["state"], "attrs": state.get("attributes", {}),
        "score": None,
        "lastSeen": snapshot.observed_at.isoformat() if snapshot.observed_at else None,
        "state_changed_at": state.get("last_changed"),
    } for state in snapshot.states[:200]]


SOURCE_HEADER = "X-KZA-Source"  # values: "real", "mock", "degraded"


def register_observability_routes(
    app: FastAPI,
    *,
    event_bus: LiveEventBus | None = None,
    ha_client=None,
    llm_router=None,
    user_manager=None,
    alert_manager=None,
    zone_manager=None,
    event_logger=None,
    llm_metrics=None,
    use_mocks: bool = False,
    capabilities: HomeCapabilityService | None = None,
    admin_work: OwnedBlockingWork | None = None,
) -> None:
    """Registrar todas las rutas observability en `app`.

    Importante: llamar ANTES de `app.mount("/", StaticFiles(...))`. FastAPI
    matchea rutas en orden; un mount catch-all en "/" engulle los /api/*.

    Headers de respuesta: cada endpoint setea `X-KZA-Source` con
        - "real": lectura real, incluidas colecciones vacías o parciales
        - "mock": demo explícita con use_mocks=True, sin fecha de observación
        - "degraded": fuente no disponible o vencida, sin ejemplos de reemplazo
    Los headers de observación conservan los instantes reales y campos desconocidos.
    """
    router = APIRouter(dependencies=[Depends(require_dashboard_session)])
    work = ensure_admin_work(app, admin_work)
    observations = ObservationHeaders()
    monitors = MonitorService()
    app.state.observability_monitors = monitors
    app.add_event_handler("shutdown", monitors.aclose)
    real_services = {
        "ha_client": ha_client, "llm_router": llm_router,
        "user_manager": user_manager, "alert_manager": alert_manager,
        "zone_manager": zone_manager, "event_logger": event_logger,
    }
    injected = [k for k, v in real_services.items() if v is not None]
    if use_mocks and injected:
        logger.warning(
            f"observability use_mocks=True con servicios reales inyectados "
            f"({injected}) — el dashboard servirá mocks. Flippeá "
            f"dashboard.observability_use_mocks=false en settings.yaml para "
            f"datos reales."
        )

    def _adapt(adapter_call, mock_data, response, *, component, source, unobserved_fields=()):
        """Return real data or explicit unavailability; demo requires configuration."""
        if use_mocks:
            _set_source(response, component, source, real=False)
            return mock_data
        try:
            data = adapter_call()
        except NativeAuthorityRetired:
            raise
        except Exception as error:
            logger.warning("Observation unavailable [%s]: %s", component, type(error).__name__)
            data = None
        _set_source(response, component, source, real=data is not None, unobserved_fields=unobserved_fields)
        return data if data is not None else []

    def _set_source(response, component, source, *, real=False, observation=None, unobserved_fields=(), reason=None):
        return observations.apply(response, component, source, available=real, observation=observation,
            demo=use_mocks, unobserved_fields=unobserved_fields, reason=reason)

    async def _adapt_events(adapter_call, mock_data, response, *, component, source, unobserved_fields=()):
        if use_mocks:
            return _adapt(lambda: None, mock_data, response, component=component, source=source)
        if isinstance(event_logger, EventLogger):
            attempted = datetime.now(UTC)
            def read():
                data = adapter_call()
                return data, datetime.now(UTC)
            try:
                data, observed = await event_logger.acall(read)
            except Exception as error:
                logger.warning("Event observation unavailable: %s", type(error).__name__)
                data, observed = None, None
            observation = Observation.from_read(source, available=data is not None, observed_at=observed,
                ttl_s=6, attempted_at=attempted, unobserved_fields=unobserved_fields)
            _set_source(response, component, source, observation=observation)
            return data if data is not None else []
        return _adapt(adapter_call, mock_data, response, component=component, source=source,
                      unobserved_fields=unobserved_fields)

    @router.get("/api/zones")
    async def get_zones(response: Response):
        return _adapt(lambda: _zones_adapter(zone_manager) if zone_manager is not None else None,
            mocks.ZONES, response, component="zones", source=ObservationSource.ZONE_MANAGER,
            unobserved_fields=("user", "last_utterance", "spotify_playback"))

    @router.get("/api/conversations")
    async def get_conversations(
        response: Response,
        request: Request,
        user: str | None = None, zone: str | None = None,
        from_: str | None = None, to: str | None = None, path: str | None = None,
    ):
        if not use_mocks and event_logger is not None:
            if isinstance(event_logger,EventLogger):
                connection = request.headers.get('x-kza-live-connection')
                if event_bus is None or connection is None:
                    raise HTTPException(428,detail='conversation_receipts_required')
                available = session_guard(request, frozenset({'conversations:read'}))
                try:
                    return event_bus.conversation_response(connection,event_logger,
                        {'user':user,'zone':zone,'path':path},available,
                        reader=session_reader(request,frozenset({'conversations:read'})))
                except RuntimeError:
                    raise HTTPException(503,detail='conversation_receipts_unavailable') from None
            raise HTTPException(503,detail='conversation_receipts_unavailable')
        _set_source(response, "conversations", ObservationSource.EVENT_STORE)
        items = mocks.CONVERSATIONS if use_mocks else []
        if user:
            items = [i for i in items if i["user"] == user]
        if zone:
            items = [i for i in items if i["zone"] == zone]
        if path:
            items = [i for i in items if i["path"] == path]
        return items

    @router.get("/api/ha/entities")
    async def get_ha_entities(response: Response, domain: str | None = None):
        source = ObservationSource.HA_STATES
        if use_mocks:
            _set_source(response, "ha_entities", source)
            items = mocks.HA_ENTITIES
        elif ha_client is None:
            _set_source(response, "ha_entities", source, reason=ObservationReason.UNCONFIGURED)
            items = []
        else:
            try:
                if capabilities is not None:
                    snapshot = await capabilities.snapshot()
                    states, observation = snapshot.states, snapshot.observation
                else:
                    states = await ha_client.get_state_snapshot()
                    if not isinstance(states, HAStatesSnapshot):
                        raise ValueError("invalid_state_observation")
                    reason = ObservationReason.DENIED if states.reason == "ha_states_denied" else None
                    observation = Observation.from_read(source, available=states.available,
                        observed_at=states.observed_at, expires_at=states.expires_at,
                        attempted_at=states.attempted_at, ttl_s=STATE_OBSERVATION_TTL_S, reason=reason)
                _set_source(response, "ha_entities", source, observation=observation,
                            unobserved_fields=("semantic_score",))
                items = await _ha_entities_adapter(ha_client, snapshot=states) if observation.current() else []
                items = items if items is not None else []
            except Exception as error:
                logger.warning("HA entity observation unavailable: %s", type(error).__name__)
                _set_source(response, "ha_entities", source)
                items = []
        if domain:
            items = [value for value in items if value["domain"] == domain]
        return items

    @router.get("/api/ha/actions")
    async def get_ha_actions(response: Response):
        return await _adapt_events(lambda: _ha_actions_adapter(event_logger) if event_logger is not None else None,
            mocks.HA_ACTIONS, response, component="ha_actions", source=ObservationSource.EVENT_STORE,
            unobserved_fields=("action_effect",))

    @router.get("/api/llm/endpoints")
    async def get_llm_endpoints(response: Response, request: Request):
        current = session_guard(request, frozenset({'diagnostics:read'}))
        if not use_mocks and llm_router is not None:
            try:
                data = await run_admin_work(work, current, _llm_endpoints_adapter, llm_router, llm_metrics,
                    before_read=partial(require_current, current))
            except NativeAuthorityRetired as error:
                raise HTTPException(403, "dashboard_authority_retired") from error
            except RuntimeError as error:
                raise HTTPException(503, "endpoint_state_unavailable") from error
        else:
            data = None
        return _adapt(lambda: data, mocks.LLM_ENDPOINTS, response, component="llm_endpoints",
            source=ObservationSource.ROUTER_STATE,
            unobserved_fields=("backend_health", "last_check", "failures_7d"))

    @router.post("/api/llm/endpoints/{endpoint_id}/clear-cooldown")
    async def clear_cooldown(endpoint_id: str, request: Request):
        if use_mocks:
            return {"ok": True, "endpoint_id": endpoint_id, "mocked": True}
        cooldown = getattr(llm_router, "_cd", None)
        if cooldown is None:
            raise HTTPException(status_code=503, detail="cooldown manager no disponible")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            await run_admin_work(work, current, cooldown.record_success, endpoint_id,
                before_save=partial(require_current, current))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        except KeyError:
            raise HTTPException(404, "endpoint_not_found")
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "cooldown_persistence_unavailable") from error
        return {"ok": True, "endpoint_id": endpoint_id}

    @router.get("/api/users")
    async def get_users(response: Response, request: Request):
        current = session_guard(request, frozenset({'diagnostics:read'}))
        if not use_mocks and user_manager is not None:
            validator = partial(require_current, current)
            try:
                data = await run_admin_work(work, current, _users_adapter, user_manager, before_read=validator)
            except NativeAuthorityRetired as error:
                raise HTTPException(403, "dashboard_authority_retired") from error
            except RuntimeError as error:
                raise HTTPException(503, "user_catalog_unavailable") from error
        else:
            data = None
        return _adapt(lambda: data, mocks.USERS, response, component="users",
            source=ObservationSource.USER_REGISTRY,
            unobserved_fields=("samples", "enrollment_time", "emotions", "pca", "effective_permissions", "top_commands"))

    @router.get("/api/alerts")
    async def get_alerts(response: Response, status: str | None = None):
        items = _adapt(lambda: _alerts_adapter(alert_manager, status) if alert_manager is not None else None,
            mocks.ALERTS, response, component="alerts", source=ObservationSource.ALERT_MANAGER)
        if use_mocks and status in {"active", "acked"}:
            items = [value for value in items if value["acked"] == (status == "acked")]
        return items

    @router.get("/api/system/gpus")
    async def get_gpus(response: Response, request: Request):
        source = ObservationSource.GPU_PROBE
        available = session_guard(request, frozenset({'diagnostics:read'}))
        if use_mocks:
            _set_source(response, "gpus", source)
            return mocks.GPUS
        try:
            sample = await monitors.sample("gpus")
            data, observation = sample.data, sample.observation
        except Exception as error:
            logger.warning("Monitor observation unavailable [gpus]: %s", type(error).__name__)
            data, observation = None, None
        if not available():
            data = None
            raise HTTPException(401, detail="dashboard_session_expired", headers={"Cache-Control": "no-store"})
        _set_source(response, "gpus", source, real=data is not None, observation=observation,
                    unobserved_fields=("role_assignment", "process_probe"))
        return data if data is not None else []

    @router.get("/api/system/services")
    async def get_services(response: Response, request: Request):
        source = ObservationSource.SERVICE_PROBE
        available = session_guard(request, frozenset({'diagnostics:read'}))
        if use_mocks:
            _set_source(response, "services", source)
            return mocks.SERVICES
        try:
            sample = await monitors.sample("services")
            data, observation = sample.data, sample.observation
        except Exception as error:
            logger.warning("Monitor observation unavailable [services]: %s", type(error).__name__)
            data, observation = None, None
        if not available():
            data = None
            raise HTTPException(401, detail="dashboard_session_expired", headers={"Cache-Control": "no-store"})
        _set_source(response, "services", source, real=data is not None, observation=observation,
                    unobserved_fields=("in_process_health", "process_memory", "uptime"))
        return data if data is not None else []

    if event_bus is not None:
        @router.websocket("/ws/live")
        async def ws_live(ws: WebSocket):
            bound_guard = session_guard(ws, frozenset({'conversations:read'}))
            reader = session_reader(ws, frozenset({'conversations:read'}))
            if reader is None:
                await ws.close(code=1008)
                return
            try:
                sub_id, _ = await event_bus.subscribe(reader=reader)
            except RuntimeError:
                await ws.close(code=1013)
                return
            async def send(frame):
                try:
                    await require_dashboard_session(ws, ws.headers.get('authorization'), ws.cookies.get(COOKIE_NAME))
                except WebSocketException:
                    await ws.close(code=1008)
                    raise
                await ws.send_json(frame)
            def available():
                return bound_guard()
            async def close():
                await ws.close(code=1000 if available() else 1008)
            try:
                await ws.accept()
                await event_bus.browser_stream(sub_id, send, close, ws.receive, available,reader=reader)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.warning('Live subscriber unavailable')
                try:
                    await ws.close(code=1000 if available() else 1008)
                except (Exception, asyncio.CancelledError):
                    pass
            finally:
                try:
                    await event_bus.unsubscribe(sub_id)
                except WorkCloseTimeout:
                    logger.warning('Live subscriber cleanup pending')

    app.include_router(router)
