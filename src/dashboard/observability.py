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

Cada endpoint setea header `X-KZA-Source: real|mock|degraded`. Cae en mocks si
no hay servicio inyectado, falla, o devuelve vacío.
"""

import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect

from src.dashboard import observability_mocks as mocks
from src.dashboard import system_monitor
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
            "spotify": {"playing": False},
        })
    return out


def _llm_endpoints_adapter(llm_router, metrics_tracker=None) -> list[dict]:
    endpoints = getattr(llm_router, "_endpoints", None)
    cd = getattr(llm_router, "_cd", None)
    if not endpoints:
        return []
    now = time.time()
    out = []
    for ep in endpoints:
        ep_id = getattr(ep, "id", "?")
        in_cd = (cd is not None) and (not cd.is_available(ep_id))
        cooldown_ends = None
        if in_cd and cd is not None:
            try:
                eta = cd.next_attempt_at(ep_id)
                cooldown_ends = datetime.fromtimestamp(eta).strftime("%H:%M:%S")
            except Exception:
                pass
        kind = getattr(ep, "kind", None)
        kind_val = kind.value if hasattr(kind, "value") else str(kind)
        # Métricas reales rolling 5min (tps, ttft_ms p50, calls)
        tps, ttft_ms, calls, last_check_ts = None, None, 0, now
        if metrics_tracker is not None:
            snap = metrics_tracker.snapshot(ep_id)
            if snap:
                tps = snap.get("tps")
                ttft_ms = snap.get("ttft_ms")
                calls = snap.get("calls", 0)
                last_check_ts = snap.get("last_call_ts", now)
        out.append({
            "id": ep_id, "name": ep_id,
            "url": getattr(getattr(ep, "client", None), "base_url", "local"),
            "priority": getattr(ep, "priority", 0),
            "role": kind_val,
            "state": "cooldown" if in_cd else "healthy",
            "tps": tps, "ttft_ms": ttft_ms, "calls_5min": calls,
            "last_check": datetime.fromtimestamp(last_check_ts).strftime("%H:%M:%S"),
            "failures_7d": {"timeout": 0, "billing": 0, "rate_limit": 0, "idle": 0},
            "cooldown_ends": cooldown_ends,
        })
    return out


def _users_adapter(user_manager) -> list[dict]:
    users = user_manager.get_all_users(active_only=True)
    out = []
    for u in users:
        lvl = u.permission_level.value if hasattr(u.permission_level, "value") else int(u.permission_level)
        out.append({
            "id": u.user_id, "name": u.name,
            "samples": 1 if u.voice_embedding is not None else 0,
            "lastEnroll": datetime.fromtimestamp(u.created_at).strftime("%Y-%m-%d"),
            "emotions": {"neutral": 100},
            "topCommands": [],
            "permissions": {
                "climate": lvl >= 1, "lights": lvl >= 1,
                "security": lvl >= 3, "music": lvl >= 1, "scenes": lvl >= 2,
            },
            "pca": [[0.0, 0.0]],
        })
    return out


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


def _conversations_adapter(
    event_logger,
    *,
    user: str | None = None,
    zone: str | None = None,
    path: str | None = None,
    limit: int = 50,
) -> list[dict] | None:
    """EventLogger.get_events → shape mocks.CONVERSATIONS.

    Cada Event con `trigger_phrase` no-vacío es un turn de voz: lo proyectamos
    al shape esperado por la vista Conversaciones del dashboard.
    """
    try:
        events = event_logger.get_events(limit=limit * 3)
    except Exception as e:
        logger.warning(f"event_logger.get_events failed: {e}")
        return None
    out = []
    for ev in events:
        if not ev.trigger_phrase:
            continue  # automaciones/sensores no son charlas
        ctx = ev.context or {}
        ev_zone = ctx.get("zone") or ctx.get("room_id")
        ev_path = ctx.get("path", "fast")
        if user and (ev.user_name or "").lower() != user.lower():
            continue
        if zone and ev_zone != zone:
            continue
        if path and ev_path != path:
            continue
        domain = ev.entity_id.split(".", 1)[0] if "." in ev.entity_id else "—"
        out.append({
            "id": f"turn_{int(ev.timestamp * 1000)}",
            "ts": ev.datetime.strftime("%H:%M:%S"),
            "user": ev.user_name or ev.user_id or "—",
            "zone": ev_zone or "—",
            "path": ev_path,
            "stt": ev.trigger_phrase,
            "intent": f"{domain}.{ev.action}",
            "target": ev.entity_id,
            "args": {k: v for k, v in ctx.items() if k not in ("zone", "room_id", "path", "latency_ms", "tts")},
            "tts": ctx.get("tts", ""),
            "latency_ms": ctx.get("latency_ms", 0),
            "success": ctx.get("success", True),
        })
        if len(out) >= limit:
            break
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
            "args": str(ctx.get("args", "{}")),
            "ok": ctx.get("success", True),
            "lat_ms": ctx.get("latency_ms", 0),
        })
        if len(out) >= limit:
            break
    return out


async def _ha_entities_adapter(ha_client) -> list[dict] | None:
    """HA convención: async/await para I/O.

    Probamos varios method names que HA clients suelen exponer:
    `get_all_entities` (KZA HomeAssistantClient), `get_states`, `states`.
    """
    import inspect
    fn = (getattr(ha_client, "get_all_entities", None)
          or getattr(ha_client, "get_states", None)
          or getattr(ha_client, "states", None))
    if not callable(fn):
        return None
    try:
        states = await fn() if inspect.iscoroutinefunction(fn) else fn()
        if states is None:
            return None
        out = []
        for s in states[:200]:
            entity_id = s.get("entity_id", "")
            domain = entity_id.split(".", 1)[0] if "." in entity_id else "—"
            out.append({
                "id": entity_id, "domain": domain,
                "name": s.get("attributes", {}).get("friendly_name", entity_id),
                "state": s.get("state", "unknown"),
                "attrs": s.get("attributes", {}),
                "score": 1.0,
                "lastSeen": (s.get("last_changed") or "")[:5],
            })
        return out
    except Exception as e:
        logger.warning(f"ha_client adapter failed: {e}")
        return None


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
    use_mocks: bool = True,
) -> None:
    """Registrar todas las rutas observability en `app`.

    Importante: llamar ANTES de `app.mount("/", StaticFiles(...))`. FastAPI
    matchea rutas en orden; un mount catch-all en "/" engulle los /api/*.

    Headers de respuesta: cada endpoint setea `X-KZA-Source` con
        - "real": datos del manager inyectado
        - "mock": no había manager o use_mocks=True
        - "degraded": adapter falló y se cae al mock
    Esto permite a la UI mostrar un banner "datos simulados" cuando aplica.
    """
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

    def _adapt(adapter_call, mock_data, response):
        """Devuelve datos reales si `adapter_call` produce un resultado.

        Setea SOURCE_HEADER:
          - "real" si el adapter devolvió un valor (incluso lista vacía — el
            usuario quiere ver "0 usuarios enrolados", no demos falsos).
          - "degraded" si el adapter retornó None (no aplicable) o lanzó.
          - "mock" si use_mocks=True.
        """
        if use_mocks:
            response.headers[SOURCE_HEADER] = "mock"
            return mock_data
        try:
            data = adapter_call()
            if data is not None:
                response.headers[SOURCE_HEADER] = "real"
                return data
            response.headers[SOURCE_HEADER] = "degraded"
            logger.warning("adapter returned None, serving mocks (degraded)")
        except Exception as e:
            response.headers[SOURCE_HEADER] = "degraded"
            logger.warning(f"adapter raised, serving mocks (degraded): {e}")
        return mock_data

    def _set_source(response: Response, real: bool) -> None:
        response.headers[SOURCE_HEADER] = "real" if real else "mock"

    @app.get("/api/zones")
    async def get_zones(response: Response):
        if zone_manager is None:
            _set_source(response, real=False)
            return mocks.ZONES
        return _adapt(lambda: _zones_adapter(zone_manager), mocks.ZONES, response)

    @app.get("/api/conversations")
    async def get_conversations(
        response: Response,
        user: str | None = None, zone: str | None = None,
        from_: str | None = None, to: str | None = None, path: str | None = None,
    ):
        if not use_mocks and event_logger is not None:
            return _adapt(
                lambda: _conversations_adapter(
                    event_logger, user=user, zone=zone, path=path),
                mocks.CONVERSATIONS, response,
            )
        response.headers[SOURCE_HEADER] = "mock"
        items = mocks.CONVERSATIONS
        if user:
            items = [i for i in items if i["user"] == user]
        if zone:
            items = [i for i in items if i["zone"] == zone]
        if path:
            items = [i for i in items if i["path"] == path]
        return items

    @app.get("/api/ha/entities")
    async def get_ha_entities(response: Response, domain: str | None = None):
        if use_mocks or ha_client is None:
            response.headers[SOURCE_HEADER] = "mock"
            items = mocks.HA_ENTITIES
        else:
            try:
                real = await _ha_entities_adapter(ha_client)
                if real is not None:
                    response.headers[SOURCE_HEADER] = "real"
                    items = real
                else:
                    response.headers[SOURCE_HEADER] = "degraded"
                    items = mocks.HA_ENTITIES
            except Exception as e:
                response.headers[SOURCE_HEADER] = "degraded"
                logger.warning(f"ha entities adapter raised, fallback: {e}")
                items = mocks.HA_ENTITIES
        if domain:
            items = [e for e in items if e["domain"] == domain]
        return items

    @app.get("/api/ha/actions")
    async def get_ha_actions(response: Response):
        if not use_mocks and event_logger is not None:
            return _adapt(lambda: _ha_actions_adapter(event_logger),
                          mocks.HA_ACTIONS, response)
        response.headers[SOURCE_HEADER] = "mock"
        return mocks.HA_ACTIONS

    @app.get("/api/llm/endpoints")
    async def get_llm_endpoints(response: Response):
        if llm_router is None:
            _set_source(response, real=False)
            return mocks.LLM_ENDPOINTS
        return _adapt(
            lambda: _llm_endpoints_adapter(llm_router, llm_metrics),
            mocks.LLM_ENDPOINTS, response,
        )

    @app.post("/api/llm/endpoints/{endpoint_id}/clear-cooldown")
    async def clear_cooldown(endpoint_id: str):
        if use_mocks or llm_router is None:
            return {"ok": True, "endpoint_id": endpoint_id, "mocked": True}
        cd = getattr(llm_router, "_cd", None)
        if cd is None:
            raise HTTPException(status_code=503, detail="cooldown manager no disponible")
        try:
            cd.record_success(endpoint_id)
            return {"ok": True, "endpoint_id": endpoint_id}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"endpoint {endpoint_id} not found")
        except Exception as e:
            logger.error(f"clear_cooldown {endpoint_id} failed: {type(e).__name__}: {e}")
            raise

    @app.get("/api/users")
    async def get_users(response: Response):
        if user_manager is None:
            _set_source(response, real=False)
            return mocks.USERS
        return _adapt(lambda: _users_adapter(user_manager), mocks.USERS, response)

    @app.get("/api/alerts")
    async def get_alerts(response: Response, status: str | None = None):
        if use_mocks or alert_manager is None:
            response.headers[SOURCE_HEADER] = "mock"
            items = mocks.ALERTS
            if status == "active":
                items = [a for a in items if not a["acked"]]
            elif status == "acked":
                items = [a for a in items if a["acked"]]
            return items
        try:
            real = _alerts_adapter(alert_manager, status)
            if real is not None:
                response.headers[SOURCE_HEADER] = "real"
                return real
            response.headers[SOURCE_HEADER] = "degraded"
            return mocks.ALERTS
        except Exception as e:
            response.headers[SOURCE_HEADER] = "degraded"
            logger.warning(f"alerts adapter failed: {e}")
            return mocks.ALERTS

    @app.get("/api/system/gpus")
    async def get_gpus(response: Response):
        real = system_monitor.gpu_snapshot()
        _set_source(response, real=real is not None)
        return real if real else mocks.GPUS

    @app.get("/api/system/services")
    async def get_services(response: Response):
        real = system_monitor.services_snapshot()
        _set_source(response, real=real is not None)
        return real if real else mocks.SERVICES

    if event_bus is not None:
        @app.websocket("/ws/live")
        async def ws_live(ws: WebSocket):
            await ws.accept()
            sub_id, queue = await event_bus.subscribe()
            try:
                while True:
                    event = await queue.get()
                    await ws.send_json(event.to_frame())
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.warning(f"[ws/live] subscriber {sub_id} error: {e}")
            finally:
                await event_bus.unsubscribe(sub_id)
