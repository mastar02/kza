# S6 — HA state prefetch cache

**Effort**: 🔴 4-6 h
**Depende**: nada (pero es el más invasivo)
**Branch sugerido**: `feat/s6-ha-state-prefetch`

## Objetivo

Eliminar los ~30-50ms de round-trip a Home Assistant por cada comando manteniendo
un cache local del estado de las entities, actualizado via WebSocket push.

## Arquitectura

```
HomeAssistant ──WebSocket(state_changed)──► HAClient
                                             │
                                             ├─ _state_cache: dict[entity_id, State]
                                             │   (actualizado en tiempo real)
                                             │
                                             ├─ get_state(entity_id) → cache[id]
                                             │                        O REST fallback
                                             │
                                             └─ call_service_ws() normal
                                                  + consulta cache post-call
                                                    para verificación rápida
                                                  + LastActionTracker usa cache
```

## Archivos a modificar

### `src/home_assistant/ha_client.py`

Agregar cache + subscribe:

```python
class HomeAssistantClient:
    def __init__(self, ...):
        ...
        self._state_cache: dict[str, dict] = {}
        self._state_subscribe_task: asyncio.Task | None = None
        self._state_callbacks: list = []  # observers opcionales

    async def start_state_sync(self) -> None:
        """
        Subscribe al event state_changed de HA via WebSocket. Cada push
        actualiza el cache. Llamar después de test_connection().
        """
        self._state_subscribe_task = asyncio.create_task(
            self._state_sync_loop()
        )

    async def _state_sync_loop(self) -> None:
        """Loop resiliente: reconnect si el WS cae."""
        while True:
            try:
                await self._subscribe_and_sync()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"HA state sync error: {e}, reconnectando en 5s")
                await asyncio.sleep(5)

    async def _subscribe_and_sync(self) -> None:
        """
        1. Hit REST /api/states para poblar el cache inicial.
        2. Subscribe WS state_changed para updates incrementales.
        """
        # 1. Snapshot inicial via REST
        states = await self._fetch_all_states_rest()
        for st in states:
            self._state_cache[st["entity_id"]] = st
        logger.info(f"HA state cache inicial: {len(self._state_cache)} entities")

        # 2. WS subscribe state_changed
        async with self._open_ws() as ws:
            await ws.send_json({
                "id": self._next_ws_id(),
                "type": "subscribe_events",
                "event_type": "state_changed",
            })
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "event":
                    data = msg.get("event", {}).get("data", {})
                    entity_id = data.get("entity_id")
                    new_state = data.get("new_state")
                    if entity_id and new_state is not None:
                        self._state_cache[entity_id] = new_state
                        for cb in self._state_callbacks:
                            try:
                                cb(entity_id, new_state)
                            except Exception:
                                pass

    def get_entity_state_cached(self, entity_id: str) -> dict | None:
        """Cache-first state lookup. Fallback a REST si no está en cache."""
        return self._state_cache.get(entity_id)

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """Si existe en cache lo devuelve; else REST."""
        cached = self.get_entity_state_cached(entity_id)
        if cached is not None:
            return cached
        # fallback REST (lo existente)
        return await self._get_entity_state_rest(entity_id)
```

### `src/main.py`

Al startup, post-test_connection:

```python
if not await ha_client.test_connection():
    ...
    sys.exit(1)
logger.info(f"Conectado a Home Assistant: {ha_url}")

# NEW: prefetch state cache
ha_prefetch_cfg = ha_config.get("state_prefetch", {})
if ha_prefetch_cfg.get("enabled", True):
    await ha_client.start_state_sync()
```

Al shutdown:
```python
finally:
    ...
    if ha_client._state_subscribe_task:
        ha_client._state_subscribe_task.cancel()
```

### `src/orchestrator/dispatcher.py`

Usar cache en decisiones toggle implícito:

```python
# Antes de call_service_ws:
current_state = self.ha.get_entity_state_cached(entity_id)
if current_state:
    # Ej: si intent=turn_on y ya está "on", skip (o confirm "ya está prendida")
    if intent == "turn_on" and current_state.get("state") == "on":
        return {"already_in_state": True, "state": current_state}
```

### `src/orchestrator/action_context.py`

`LastActionTracker` puede consultar cache en vez de guardar el state localmente
(quedaría más consistente con HA).

### `config/settings.yaml`

```yaml
home_assistant:
  ...
  state_prefetch:
    enabled: true
    # refresh full snapshot periódicamente para atrapar drift si WS cae
    full_refresh_interval_s: 300
```

## Validación

1. Startup → log `HA state cache inicial: 142 entities`.
2. Prender una luz desde la UI de HA (no por voz) → el cache se actualiza en <1s.
3. Medir latencia end-to-end de comando: antes ~250ms, después ~200ms.
4. Matar el WS manualmente (bloquear por iptables 5s) → log de reconnect → cache sigue consistent.

## Test unitario

`tests/unit/home_assistant/test_state_cache.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_cache_populated_on_subscribe():
    ha = HomeAssistantClient(url="http://mock", token="t")
    ha._fetch_all_states_rest = AsyncMock(return_value=[
        {"entity_id": "light.escritorio", "state": "on"},
    ])
    # simular subscribe sin WS real
    states = await ha._fetch_all_states_rest()
    for st in states:
        ha._state_cache[st["entity_id"]] = st
    assert ha.get_entity_state_cached("light.escritorio")["state"] == "on"

@pytest.mark.asyncio
async def test_state_changed_event_updates_cache():
    ha = HomeAssistantClient(url="http://mock", token="t")
    ha._state_cache["light.x"] = {"entity_id": "light.x", "state": "off"}
    # simular event push
    event = {"data": {"entity_id": "light.x", "new_state": {"state": "on"}}}
    # call _handle_state_changed(event)
    # verificar cache["light.x"]["state"] == "on"
```

## Edge cases

- **WS reconnect loop**: backoff exponencial (5s, 10s, 20s, max 60s).
- **Cache divergence**: refresh full snapshot cada 5 min para atrapar cualquier
  event perdido.
- **Race conditions**: el cache es single-writer (el loop), multi-reader. Python
  GIL + dict mutations atómicas → safe para read-during-write del mismo key.
- **Volumen de events**: HA con 150 entities genera ~10-50 events/min. Trivial
  para el cache.
- **HA LocalTuya o integraciones lentas**: pueden tener drift state vs real.
  El cache refleja lo que HA sabe, no el estado físico. Documentar.

## Commit message sugerido

```
feat(ha): state prefetch cache con WebSocket push subscribe

HomeAssistantClient mantiene un cache local del estado de todas las
entities. Ahorra ~30-50ms por comando al evitar round-trip REST.

- _state_cache: dict[entity_id, state], populated via REST snapshot al
  startup + updates incrementales vía WS subscribe_events(state_changed).
- _state_sync_loop: resiliente a desconexiones con backoff + full refresh
  periódico.
- get_entity_state_cached() cache-first, get_entity_state() con REST
  fallback si miss.
- dispatcher usa el cache para toggle implícito / skip si ya estaba en
  target state.

settings.yaml: home_assistant.state_prefetch.{enabled, full_refresh_interval_s}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Checklist

- [ ] `HomeAssistantClient`: _state_cache + start_state_sync + reconnect loop
- [ ] `get_entity_state_cached` + `get_entity_state` cache-first
- [ ] REST snapshot inicial
- [ ] WS subscribe state_changed + cache update on event
- [ ] `dispatcher.py`: usar cache para toggle/skip decisions
- [ ] `settings.yaml` home_assistant.state_prefetch
- [ ] Tests cache init + event handler + reconnect
- [ ] Regression tests
- [ ] Commit + push
