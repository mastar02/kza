"""
Observability endpoints — sirven al frontend del dashboard wireframe.

Contrato (README del prototipo):
    GET  /api/zones, /api/conversations, /api/ha/entities, /api/ha/actions,
         /api/llm/endpoints, /api/users, /api/alerts,
         /api/system/gpus, /api/system/services
    POST /api/llm/endpoints/{id}/clear-cooldown
    WS   /ws/live  → frames {type, payload, ts}

Cada endpoint usa el servicio real si está inyectado; si falla o devuelve vacío,
cae en mocks. Permite al dashboard ser funcional desde el primer arranque y
mostrar datos reales conforme los servicios se levantan.
"""

import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.dashboard import observability_mocks as mocks
from src.dashboard import system_monitor
from src.dashboard.live_event_bus import LiveEventBus

logger = logging.getLogger(__name__)


# ---------------- Adapters real → shape del dashboard ----------------

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


def _llm_endpoints_adapter(llm_router) -> list[dict]:
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
        out.append({
            "id": ep_id, "name": ep_id,
            "url": getattr(getattr(ep, "client", None), "base_url", "local"),
            "priority": getattr(ep, "priority", 0),
            "role": kind_val,
            "state": "cooldown" if in_cd else "healthy",
            "tps": None, "ttft_ms": None,
            "last_check": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
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


def _ha_entities_adapter(ha_client) -> list[dict] | None:
    states_fn = getattr(ha_client, "get_states", None)
    if not callable(states_fn):
        return None
    try:
        import inspect
        if inspect.iscoroutinefunction(states_fn):
            return None  # caller no es async-aware en este path
        states = states_fn()
        if not states:
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
        logger.debug(f"ha_client.get_states failed: {e}")
        return None


# ---------------- Registro de rutas ----------------

def register_observability_routes(
    app: FastAPI,
    *,
    event_bus: LiveEventBus | None = None,
    ha_client=None,
    llm_router=None,
    user_manager=None,
    alert_manager=None,
    zone_manager=None,
    use_mocks: bool = True,
) -> None:
    """Registrar todas las rutas observability en `app`. Llamar antes de StaticFiles."""

    def _real_or_mock(adapter_call, mock_data):
        if use_mocks:
            return mock_data
        try:
            data = adapter_call()
            if data:
                return data
        except Exception as e:
            logger.warning(f"adapter failed, fallback to mock: {e}")
        return mock_data

    @app.get("/api/zones")
    async def get_zones():
        if zone_manager is None:
            return mocks.ZONES
        return _real_or_mock(lambda: _zones_adapter(zone_manager), mocks.ZONES)

    @app.get("/api/conversations")
    async def get_conversations(
        user: str | None = None, zone: str | None = None,
        from_: str | None = None, to: str | None = None, path: str | None = None,
    ):
        items = mocks.CONVERSATIONS
        if user:
            items = [i for i in items if i["user"] == user]
        if zone:
            items = [i for i in items if i["zone"] == zone]
        if path:
            items = [i for i in items if i["path"] == path]
        return items

    @app.get("/api/ha/entities")
    async def get_ha_entities(domain: str | None = None):
        if use_mocks or ha_client is None:
            items = mocks.HA_ENTITIES
        else:
            items = _ha_entities_adapter(ha_client) or mocks.HA_ENTITIES
        if domain:
            items = [e for e in items if e["domain"] == domain]
        return items

    @app.get("/api/ha/actions")
    async def get_ha_actions():
        return mocks.HA_ACTIONS

    @app.get("/api/llm/endpoints")
    async def get_llm_endpoints():
        if llm_router is None:
            return mocks.LLM_ENDPOINTS
        return _real_or_mock(lambda: _llm_endpoints_adapter(llm_router), mocks.LLM_ENDPOINTS)

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

    @app.get("/api/users")
    async def get_users():
        if user_manager is None:
            return mocks.USERS
        return _real_or_mock(lambda: _users_adapter(user_manager), mocks.USERS)

    @app.get("/api/alerts")
    async def get_alerts(status: str | None = None):
        if use_mocks or alert_manager is None:
            items = mocks.ALERTS
            if status == "active":
                items = [a for a in items if not a["acked"]]
            elif status == "acked":
                items = [a for a in items if a["acked"]]
            return items
        try:
            return _alerts_adapter(alert_manager, status) or mocks.ALERTS
        except Exception as e:
            logger.warning(f"alerts adapter failed: {e}")
            return mocks.ALERTS

    @app.get("/api/system/gpus")
    async def get_gpus():
        real = system_monitor.gpu_snapshot()
        return real if real else mocks.GPUS

    @app.get("/api/system/services")
    async def get_services():
        real = system_monitor.services_snapshot()
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
