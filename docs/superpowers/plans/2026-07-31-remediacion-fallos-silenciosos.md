# Remediación de fallos silenciosos — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que KZA deje de reportar éxito cuando no hizo nada, y que la sordera del micrófono se detecte sola en minutos en vez de en 27 horas.

**Architecture:** Tres frentes independientes que comparten un mismo defecto de raíz — señales que miden un proxy barato en vez de la cosa real. (1) Se desbloquea la suite de tests, que hoy no ejecuta ni un test y por lo tanto impide TDD en todo lo demás. (2) Se cierra el lazo con Home Assistant: chequeo de disponibilidad antes de llamar y confirmación de efecto real por `context.id` después, todo fuera del camino crítico. (3) Se agrega un poller externo que vigila el heartbeat de audio que ya existe pero que hoy nadie puede leer.

**Tech Stack:** Python 3.13, pytest, asyncio, Home Assistant REST + WebSocket API, ChromaDB, systemd --user.

## Global Constraints

- Python del proyecto: `/Users/yo/Documents/kza/.venv/bin/python` (el `python3` del sistema es 3.9 y revienta en `dataclass slots=True`).
- Imports absolutos siempre: `from src.modulo import Clase`. Nunca relativos.
- Todo I/O es `async/await`. Nunca bloquear el event loop.
- Inyección de dependencias por constructor. Composición, nunca herencia profunda.
- `@dataclass` para DTOs, `Enum` para estados.
- `logger = logging.getLogger(__name__)`. Nunca `print()`.
- Mensajes de voz y UI en español; código, logs y commits en inglés (salvo el asunto del commit, que este repo escribe en español).
- Toda config nueva va a `config/settings.yaml`. No crear archivos de configuración nuevos.
- El server `kza@192.168.1.2` es **producción**: la casa lo usa a diario. Ningún paso de este plan hace deploy ni reinicia servicios. El deploy se coordina aparte.
- Presupuesto de latencia del fast path: nada de lo que se agregue puede sumar tiempo al camino crítico. Todo lo nuevo corre en background o fuera del proceso.

## Fuera de alcance (decidido 2026-07-31)

Las 5 entidades `light.grupo_{balcon,bano,cuarto,escalera,pasillo}` están `unavailable` porque sus miembros son **bombillas Philips Hue todavía no migradas a Zigbee2MQTT** (`/opt/homeassistant/config/packages/luces_usabilidad.yaml`: los grupos vivos apuntan a `light.{cocina,escritorio,living}_2`, que son grupos Z2M; los caídos apuntan a `light.b1`, `light.cu1`, `light.hue_white_lamp_1`, etc.). Esa migración vive en el repo `homelab-domo`, no en KZA, y se hace cuando el usuario decida.

**Este plan no arregla esas luces.** Lo que arregla es que KZA se comporte correctamente **mientras** están caídas — que es exactamente la condición que la migración va a producir habitación por habitación.

---

## File Structure

| Archivo | Responsabilidad | Acción |
|---|---|---|
| `tests/unit/pipeline/test_model_manager.py` | Aísla el mock de torch para que no contamine la sesión | Modificar |
| `tests/conftest.py` | Fixture que garantiza un `torch` importable con `__spec__` válido | Modificar |
| `src/home_assistant/ha_client.py` | Expone disponibilidad y devuelve el `context` de cada llamada | Modificar |
| `src/orchestrator/dispatcher.py` | Pre-chequeo de disponibilidad + reconciliación por `context.id` | Modificar |
| `src/monitoring/audio_health.py` | Estado de salud del audio serializable hacia afuera | Crear |
| `src/pipeline/multi_room_audio_loop.py` | Publica el heartbeat no-cero al archivo de estado | Modificar |
| `tools/audio_watchdog_alert.py` | Poller externo: lee el estado y notifica por HA | Crear |
| `scripts/sync_ha_to_chroma.py` | Aborta si va a indexar entidades muertas | Modificar |
| `tools/smoke_test.py` + `src/monitoring/smoke_check.py` | Cobertura derivada del índice real, no de una lista a mano | Modificar |

---

### Task 1: Desbloquear la colección de pytest

Hoy `pytest tests/` **no ejecuta ni un solo test**. Colecta 2668 y aborta con
`Interrupted: 4 errors during collection`. Sin esto, ningún otro task de este plan puede
hacer TDD, así que va primero.

Causa raíz: `tests/unit/pipeline/test_model_manager.py:21-22` hace
`sys.modules['torch'] = MagicMock()` **a nivel de módulo**. Eso se ejecuta durante la
colección y deja un `torch` falso en `sys.modules` para toda la sesión. Cualquier archivo
colectado después que importe `transformers` llama `importlib.util.find_spec("torch")`,
que exige `torch.__spec__`, y un `MagicMock` no lo tiene → `ValueError: torch.__spec__ is
not set`.

**Files:**
- Modify: `tests/unit/pipeline/test_model_manager.py:21-22`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_collection_health.py` (crear)

**Interfaces:**
- Produces: una suite que corre entera. Todos los tasks siguientes dependen de esto.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/test_collection_health.py`:

```python
"""Guard against sys.modules pollution breaking collection for other tests."""
import importlib.util
import sys


def test_torch_module_has_valid_spec():
    """Any torch in sys.modules must be introspectable by importlib.

    transformers calls importlib.util.find_spec("torch"), which raises
    ValueError if torch.__spec__ is unset. A bare MagicMock has no __spec__,
    so a module-level sys.modules patch breaks collection for every test
    file collected afterwards.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return  # torch not loaded in this run: nothing to guard
    assert getattr(torch, "__spec__", None) is not None, (
        "torch.__spec__ is unset — something replaced sys.modules['torch'] "
        "with a mock that importlib cannot introspect"
    )
    assert importlib.util.find_spec("torch") is not None
```

- [ ] **Step 2: Correr el test para verificar que falla**

```bash
cd /Users/yo/Documents/kza
.venv/bin/python -m pytest tests/unit/pipeline/test_model_manager.py tests/unit/test_collection_health.py -q ; echo "EXIT=$?"
```

Esperado: FAIL con `torch.__spec__ is unset`. Se corren los dos archivos juntos y en ese
orden a propósito: el fallo depende de que `test_model_manager` se importe primero.

⚠️ Nunca uses `pytest ... | tail` para juzgar esto — el pipe se traga el exit code.

- [ ] **Step 3: Aislar el mock con una fixture de ámbito acotado**

En `tests/unit/pipeline/test_model_manager.py`, borrar las líneas 21-22
(`sys.modules['torch'] = MagicMock()` y `sys.modules['torch.cuda'] = MagicMock()`) y
reemplazarlas por una fixture con `autouse` limitado al módulo, que además le pone un
`__spec__` válido y restaura el estado anterior:

```python
import importlib.machinery
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_torch():
    """Stub torch for this module only, leaving sys.modules as we found it.

    __spec__ must be set or importlib.util.find_spec("torch") raises, which
    breaks collection of every module importing transformers.
    """
    saved = {name: sys.modules.get(name) for name in ("torch", "torch.cuda")}

    torch_stub = MagicMock()
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    cuda_stub = MagicMock()
    cuda_stub.__spec__ = importlib.machinery.ModuleSpec("torch.cuda", loader=None)

    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = cuda_stub
    try:
        yield torch_stub
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
```

- [ ] **Step 4: Verificar que el test pasa y que la suite entera colecta**

```bash
.venv/bin/python -m pytest tests/unit/pipeline/test_model_manager.py tests/unit/test_collection_health.py -q ; echo "EXIT=$?"
.venv/bin/python -m pytest tests/ -q --collect-only 2>&1 | tail -3 ; echo "COLLECT_EXIT=${PIPESTATUS[0]}"
```

Esperado: el primero PASS. El segundo debe decir `2668 tests collected` **sin** la línea
`Interrupted: N errors during collection`, y `COLLECT_EXIT=0`.

- [ ] **Step 5: Correr la suite completa y registrar el baseline REAL**

```bash
.venv/bin/python -m pytest tests/ -q --tb=line > /tmp/baseline.txt 2>&1 ; echo "EXIT=$?"
tail -25 /tmp/baseline.txt
```

Esto produce, por primera vez, el número honesto de tests que fallan. Guardá esa lista: es
el baseline contra el que se juzgan los tasks siguientes. Si un test estaba rojo acá, no lo
arregles en este task — anotalo.

⚠️ **Un test rojo con trampa:** hay uno cuyo "arreglo" obvio reintroduce las acciones
**invertidas** de luces (prender donde va apagar). El código está bien y el test es el que
quedó viejo. No lo toques sin leer su historial con `git log -p`.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/pipeline/test_model_manager.py tests/unit/test_collection_health.py
git commit -m "test: el mock de torch ya no rompe la colección de toda la suite"
```

---

### Task 2: Pre-chequeo de disponibilidad antes de llamar a HA

Home Assistant **filtra las entidades `unavailable` en silencio** y devuelve
`success: true` (`homeassistant/helpers/service.py:720` →
`entity_candidates = [e for e in entity_candidates if e.available]`). Combinado con la
regla de "domótica silenciosa en éxito", el usuario dice *"prendé la luz del cuarto"*,
KZA loguea `success=True took=3ms`, no dice nada, y no pasa nada.

Hoy el único código de todo `src/` que menciona `unavailable` como condición de entidad es
la herramienta de diagnóstico `src/monitoring/smoke_check.py:68`. Las dos rutas reales de
ejecución no consultan estado.

Esto es lo que hace que la migración Hue→Z2M sea silenciosa habitación por habitación.

**Files:**
- Modify: `src/home_assistant/ha_client.py` (agregar `is_entity_available`)
- Modify: `src/orchestrator/dispatcher.py:1316` (dentro de `_fire_and_reconcile_ha`)
- Test: `tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py` (crear)

**Interfaces:**
- Consumes: `HomeAssistantClient.get_entity_state_cached(entity_id) -> dict | None`
  (`ha_client.py:318`), ya existente, sin I/O, alimentado por push WS.
- Produces: `HomeAssistantClient.is_entity_available(entity_id: str) -> bool | None`
  — `True` disponible, `False` `unavailable`/`unknown`, `None` sin dato en cache.
  El task 3 lo reutiliza.

⚠️ **Esto NO es el skip idempotente que se revirtió el 2026-04-28** (ver el comentario en
`dispatcher.py:812-820`). Aquel comparaba `on`/`off`, que llega con lag de segundos vía WS
y producía falsos skips. `unavailable` es una condición **durable de días**, no sufre ese
lag. Leé ese comentario antes de tocar nada acá.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py`:

```python
"""An unavailable entity must fail loudly, not be silently swallowed by HA."""
import pytest

from src.home_assistant.ha_client import HomeAssistantClient


def test_is_entity_available_true_for_live_entity():
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {"light.grupo_living": {"state": "off"}}
    assert ha.is_entity_available("light.grupo_living") is True


@pytest.mark.parametrize("state", ["unavailable", "unknown"])
def test_is_entity_available_false_for_dead_entity(state):
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {"light.grupo_cuarto": {"state": state}}
    assert ha.is_entity_available("light.grupo_cuarto") is False


def test_is_entity_available_none_when_not_cached():
    """No data is not the same as unavailable — the caller must fail open."""
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {}
    assert ha.is_entity_available("light.desconocida") is None
```

- [ ] **Step 2: Correr para verificar que fallan**

```bash
.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py -v ; echo "EXIT=$?"
```

Esperado: FAIL con `AttributeError: 'HomeAssistantClient' object has no attribute 'is_entity_available'`.

- [ ] **Step 3: Implementar `is_entity_available`**

En `src/home_assistant/ha_client.py`, justo debajo de `get_entity_state_cached` (línea 318):

```python
    def is_entity_available(self, entity_id: str) -> bool | None:
        """¿La entidad puede actuar, según el último estado conocido?

        HA acepta un service_call contra una entidad `unavailable` y lo filtra
        en silencio (helpers/service.py), devolviendo success=true. Este chequeo
        existe para no confundir "HA aceptó el mensaje" con "el dispositivo hizo
        algo".

        Returns:
            True si está disponible, False si está `unavailable`/`unknown`,
            None si no hay entry en cache — el caller debe fallar ABIERTO
            (llamar igual), porque "no sé" no es "está rota".
        """
        entry = self._state_cache.get(entity_id)
        if entry is None:
            return None
        return entry.get("state") not in ("unavailable", "unknown")
```

- [ ] **Step 4: Verificar que pasan**

```bash
.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py -v ; echo "EXIT=$?"
```

Esperado: 4 passed.

- [ ] **Step 5: Cablear el pre-chequeo en el dispatcher**

En `src/orchestrator/dispatcher.py`, dentro de `_fire_and_reconcile_ha`, **inmediatamente
antes** del bloque `t0 = time.perf_counter()`:

```python
        # HA acepta la llamada a una entidad unavailable y la filtra en
        # silencio (helpers/service.py:720), devolviendo success=true. Sin este
        # chequeo, "prendé la luz del cuarto" con la bombilla caída produce
        # silencio absoluto: ni voz, ni earcon, ni luz.
        # Falla ABIERTO ante None: sin dato en cache, llamamos igual.
        if entity_id and self.ha.is_entity_available(entity_id) is False:
            logger.warning(
                f"[HA-UNAVAILABLE] {domain}.{service}@{entity_id} "
                f"({description}) — no se envía la llamada"
            )
            if self.response_handler is not None:
                try:
                    self.response_handler.play_earcon(zone_id=command.get("zone_id"))
                except Exception as e:
                    logger.warning(f"No pude reproducir earcon de entidad caída: {e}")
            return
```

- [ ] **Step 6: Test de integración del cableado**

Agregar al mismo archivo de test:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock


def _dispatcher_with(ha_available, dispatcher_cls):
    d = dispatcher_cls.__new__(dispatcher_cls)
    d.ha = MagicMock()
    d.ha.is_entity_available = MagicMock(return_value=ha_available)
    d.ha.call_service_ws = AsyncMock(return_value=True)
    d.response_handler = MagicMock()
    d.hooks = None
    return d


def test_unavailable_entity_is_not_called_and_plays_earcon():
    from src.orchestrator.dispatcher import RequestDispatcher

    d = _dispatcher_with(False, RequestDispatcher)
    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.grupo_cuarto", "service_data": {},
        "description": "luz del cuarto", "zone_id": "cocina",
    }))
    d.ha.call_service_ws.assert_not_awaited()
    d.response_handler.play_earcon.assert_called_once()


def test_unknown_availability_fails_open_and_calls():
    """None means 'no data', not 'broken'. We must still try."""
    from src.orchestrator.dispatcher import RequestDispatcher

    d = _dispatcher_with(None, RequestDispatcher)
    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.nueva", "service_data": {},
        "description": "luz nueva", "zone_id": "cocina",
    }))
    d.ha.call_service_ws.assert_awaited_once()
```

⚠️ `_fire_and_reconcile_ha` tiene más dependencias (`self.hooks`, el rewriter de
`service_data`). Si el `__new__` desnudo revienta, agregá solo los atributos que el
traceback pida — no construyas el dispatcher entero.

- [ ] **Step 7: Correr y commitear**

```bash
.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py -v ; echo "EXIT=$?"
git add src/home_assistant/ha_client.py src/orchestrator/dispatcher.py tests/unit/orchestrator/test_dispatcher_unavailable_precheck.py
git commit -m "fix(ha): una entidad caída ya no se traga el comando en silencio"
```

---

### Task 3: Alerta de sordera

**Este es el task que devuelve más valor de todo el plan.** Hubo dos episodios de sordera
(27 h y 7 h) detectados porque el usuario preguntó *"¿por qué no contesta?"*, con el
servicio en `active` todo el tiempo.

El dato ya existe y es correcto: `multi_room_audio_loop.py:873-874` estampa
`rs.last_frame_ts` **solo si `indata.any()`** — el fix del 2026-07-30, que mide audio real
y no invocaciones del callback. El problema es que **ese dato nunca sale del proceso**.
`/api/health` no mira audio y su veredicto es constante.

**Files:**
- Create: `src/monitoring/audio_health.py`
- Modify: `src/pipeline/multi_room_audio_loop.py` (publicar el snapshot)
- Create: `tools/audio_watchdog_alert.py`
- Test: `tests/unit/monitoring/test_audio_health.py` (crear)

**Interfaces:**
- Consumes: `RoomStream.last_frame_ts` / `.opened_ts` (monotónicos),
  `detect_stale_streams` (`multi_room_audio_loop.py:74`, ya pura y testeada).
- Produces: `write_audio_health(path, rooms, now_wall, now_mono)` y
  `evaluate_health(snapshot, now_wall, deaf_after_s)` → `list[str]` de rooms sordas.

**Decisiones de diseño, con su porqué:**

1. **Criterio no-cero, NO umbral de RMS.** El piso de ruido real es 0.0104 y un stream sano
   hoy mide 0.011-0.015 — demasiado cerca para discriminar. Se reutiliza el criterio que ya
   está en el callback.
2. **El poller es un proceso EXTERNO**, no una task dentro de `kza-voice`. Un watchdog
   interno muere con el proceso que vigila, y el modo de falla observado es justamente
   "el proceso vive y no entrega audio". `systemd OnFailure` tampoco sirve: **el servicio
   no falla**.
3. **El canal de salida es Home Assistant** (`persistent_notification.create`), nunca el
   TTS del propio sistema — una alerta locutada por el pipeline sordo no llega. El sistema
   de notificaciones interno tampoco sirve: reporta `delivered=True` sin ningún canal
   conectado.
4. **Timestamps wall-clock en el archivo, no monotónicos.** `time.monotonic()` no tiene
   sentido entre procesos distintos. Se escribe la *edad* ya calculada más un wall-clock de
   referencia.
5. **Gracia de arranque de 180 s.** Arranques medidos: 1.5-2 s lo normal, **135 s el peor
   caso observado**. Un margen corto haría que el poller grite en cada reinicio.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/monitoring/test_audio_health.py`:

```python
"""The deafness signal must distinguish a silent room from a dead mic."""
import json

from src.monitoring.audio_health import evaluate_health, write_audio_health


def test_room_delivering_audio_is_healthy():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 2.0, "ever": True}}}
    assert evaluate_health(snap, now_wall=1002.0, deaf_after_s=120.0) == []


def test_room_silent_past_threshold_is_deaf():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 300.0, "ever": True}}}
    assert evaluate_health(snap, now_wall=1002.0, deaf_after_s=120.0) == ["cocina"]


def test_stale_snapshot_counts_as_deaf():
    """If the writer stopped writing, the process itself is wedged."""
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 1.0, "ever": True}}}
    # el snapshot tiene 10 minutos: nadie lo actualiza
    assert evaluate_health(snap, now_wall=1600.0, deaf_after_s=120.0) == ["cocina"]


def test_room_that_never_delivered_is_not_deaf_within_grace():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 30.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == []


def test_room_that_never_delivered_past_grace_is_deaf():
    snap = {"wall": 1000.0, "rooms": {"cocina": {"age_s": 200.0, "ever": False}}}
    assert evaluate_health(
        snap, now_wall=1002.0, deaf_after_s=120.0, first_frame_grace_s=180.0
    ) == ["cocina"]


def test_write_audio_health_is_atomic_and_readable(tmp_path):
    path = tmp_path / "audio_health.json"
    write_audio_health(
        str(path),
        rooms={"cocina": (500.0, 100.0), "escritorio": (0.0, 100.0)},
        now_wall=1000.0,
        now_mono=520.0,
    )
    data = json.loads(path.read_text())
    assert data["wall"] == 1000.0
    assert data["rooms"]["cocina"]["age_s"] == 20.0
    assert data["rooms"]["cocina"]["ever"] is True
    assert data["rooms"]["escritorio"]["ever"] is False
    assert not list(tmp_path.glob("*.tmp"))  # sin temporales huérfanos
```

- [ ] **Step 2: Correr para verificar que fallan**

```bash
.venv/bin/python -m pytest tests/unit/monitoring/test_audio_health.py -v ; echo "EXIT=$?"
```

Esperado: FAIL con `ModuleNotFoundError: No module named 'src.monitoring.audio_health'`.

- [ ] **Step 3: Implementar el módulo**

Crear `src/monitoring/audio_health.py`:

```python
"""Publica el heartbeat de audio para que un proceso externo pueda leerlo.

El watchdog interno (multi_room_audio_loop._stream_watchdog) sabe si un mic
entrega audio, pero ese dato nunca sale del proceso: durante 27h de sordera
systemd mostró `active` y nadie tuvo cómo enterarse. Este módulo escribe un
snapshot que un poller externo consume — externo a propósito, porque el modo
de falla observado es "el proceso vive y no entrega audio".
"""
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def write_audio_health(
    path: str,
    rooms: dict[str, tuple[float, float]],
    now_wall: float,
    now_mono: float,
) -> None:
    """Escribir el snapshot de salud de audio de forma atómica.

    Args:
        path: destino del JSON.
        rooms: room_id -> (last_frame_ts, opened_ts), ambos monotónicos.
            last_frame_ts == 0.0 significa "todavía no entregó ningún frame".
        now_wall: time.time() — referencia para que el lector, que corre en
            otro proceso, pueda detectar un snapshot viejo.
        now_mono: time.monotonic() — para convertir los ts a edades.
    """
    payload = {
        "wall": now_wall,
        "rooms": {
            room_id: {
                "age_s": (now_mono - last_ts) if last_ts > 0.0 else (now_mono - opened_ts),
                "ever": last_ts > 0.0,
            }
            for room_id, (last_ts, opened_ts) in rooms.items()
        },
    }
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def evaluate_health(
    snapshot: dict,
    now_wall: float,
    deaf_after_s: float,
    first_frame_grace_s: float = 180.0,
    snapshot_stale_after_s: float = 120.0,
) -> list[str]:
    """Devolver las rooms consideradas sordas.

    Un snapshot viejo implica que el escritor dejó de escribir — el proceso
    está trabado — y eso cuenta como sordera de TODAS las rooms: el silencio
    del vigilante no puede significar "todo bien".

    La gracia de primer frame es 180s porque los arranques medidos van de
    1.5-2s (lo normal) a 135s (el peor caso observado).
    """
    rooms = snapshot.get("rooms", {})
    snapshot_age = now_wall - snapshot.get("wall", 0.0)
    if snapshot_age > snapshot_stale_after_s:
        return sorted(rooms)

    deaf = []
    for room_id, info in rooms.items():
        threshold = deaf_after_s if info.get("ever") else first_frame_grace_s
        if info.get("age_s", 0.0) > threshold:
            deaf.append(room_id)
    return sorted(deaf)
```

Agregar `import contextlib` arriba junto al resto de los imports de stdlib.

- [ ] **Step 4: Verificar que pasan**

```bash
.venv/bin/python -m pytest tests/unit/monitoring/test_audio_health.py -v ; echo "EXIT=$?"
```

Esperado: 6 passed.

- [ ] **Step 5: Publicar el snapshot desde el watchdog**

En `src/pipeline/multi_room_audio_loop.py`, dentro de `_stream_watchdog` (línea 682),
inmediatamente después de construir `states` y **antes** de `detect_stale_streams`:

```python
            # Publicar el heartbeat para el poller externo. El watchdog interno
            # recupera, pero no puede avisar: si el proceso se traba, se traba
            # con él. Un fallo acá jamás debe romper la recuperación.
            try:
                from src.monitoring.audio_health import write_audio_health

                write_audio_health(
                    self._audio_health_path,
                    {room_id: (rs.last_frame_ts, rs.opened_ts)
                     for room_id, rs in self.room_streams.items()},
                    now_wall=time.time(),
                    now_mono=now,
                )
            except Exception as e:
                logger.debug(f"No pude publicar audio_health: {e}")
```

Agregar el parámetro al `__init__` (junto a los otros `stream_watchdog_*`, línea ~199):

```python
        audio_health_path: str = "./data/audio_health.json",
```

y en el cuerpo: `self._audio_health_path = audio_health_path`.

En `config/settings.yaml`, dentro del bloque `rooms.stream_watchdog`, agregar:

```yaml
    # Snapshot del heartbeat para el poller externo (tools/audio_watchdog_alert.py).
    # Externo a propósito: el modo de falla observado (27h, 7h) es "el proceso
    # vive y no entrega audio", que un watchdog interno no puede reportar.
    health_path: "./data/audio_health.json"
```

y cablearlo en `src/main.py` donde se construye el `MultiRoomAudioLoop`, leyendo
`rooms.stream_watchdog.health_path`.

- [ ] **Step 6: Escribir el poller externo**

Crear `tools/audio_watchdog_alert.py`:

```python
#!/usr/bin/env python3
"""Poller externo: avisa por Home Assistant si un micrófono quedó sordo.

Corre FUERA de kza-voice a propósito. Los dos incidentes de sordera (27h y 7h)
tuvieron el servicio en `active` todo el tiempo, así que ni systemd ni una task
interna podían detectarlos.

Uso:
    python tools/audio_watchdog_alert.py --once     # un chequeo, para cron
    python tools/audio_watchdog_alert.py            # bucle
"""
import argparse
import json
import os
import sys
import time

import requests

DEFAULT_HEALTH = "/home/kza/app/data/audio_health.json"


def notify_ha(base_url: str, token: str, title: str, message: str) -> bool:
    """Crear una notificación persistente en HA. Devuelve si HA la aceptó."""
    resp = requests.post(
        f"{base_url}/api/services/persistent_notification/create",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": title,
            "message": message,
            "notification_id": "kza_audio_deaf",
        },
        timeout=10,
    )
    return resp.status_code == 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-path", default=os.environ.get("KZA_AUDIO_HEALTH", DEFAULT_HEALTH))
    ap.add_argument("--deaf-after-s", type=float, default=300.0)
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    base_url = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123").rstrip("/")
    token = os.environ.get("HOME_ASSISTANT_TOKEN", "")
    if not token:
        print("HOME_ASSISTANT_TOKEN no está seteado", file=sys.stderr)
        return 2

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.monitoring.audio_health import evaluate_health

    while True:
        try:
            with open(args.health_path) as fh:
                snapshot = json.load(fh)
            deaf = evaluate_health(snapshot, time.time(), args.deaf_after_s)
        except FileNotFoundError:
            # El archivo no existe: kza-voice nunca arrancó o no llegó a
            # escribirlo. Eso también es una anomalía que hay que reportar.
            deaf = ["(sin snapshot de audio)"]
        except Exception as e:
            print(f"error leyendo {args.health_path}: {e}", file=sys.stderr)
            deaf = []

        if deaf:
            msg = "Sin audio de: " + ", ".join(deaf)
            print(msg)
            notify_ha(base_url, token, "KZA quedó sordo", msg)

        if args.once:
            return 1 if deaf else 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Verificar el poller sin tocar producción**

```bash
# snapshot sintético con una room sorda
mkdir -p /tmp/kzatest && .venv/bin/python -c "
import json,time
json.dump({'wall': time.time(), 'rooms': {'cocina': {'age_s': 900.0, 'ever': True}}},
          open('/tmp/kzatest/audio_health.json','w'))
"
HOME_ASSISTANT_TOKEN=dummy .venv/bin/python tools/audio_watchdog_alert.py \
  --health-path /tmp/kzatest/audio_health.json --once ; echo "EXIT=$?"
```

Esperado: imprime `Sin audio de: cocina` y `EXIT=1`. (La notificación a HA fallará con el
token dummy — es lo correcto en un test local; lo que se valida acá es la detección.)

- [ ] **Step 8: Commit**

```bash
git add src/monitoring/audio_health.py tools/audio_watchdog_alert.py \
        tests/unit/monitoring/test_audio_health.py \
        src/pipeline/multi_room_audio_loop.py src/main.py config/settings.yaml
git commit -m "feat(obs): alerta de sordera — el heartbeat de audio ahora sale del proceso"
```

⚠️ **El deploy no es parte de este plan.** Instalar el timer de systemd que corre el poller
en el server es un paso de producción y se coordina con el usuario aparte.

---

### Task 4: El sync de Chroma aborta si va a indexar entidades muertas

Cuando una entidad está `unavailable`, HA reduce sus atributos a un stub.
`light.grupo_cuarto` hoy reporta `supported_color_modes: ['onoff']` mientras que
`light.grupo_living` (vivo) reporta `color_temp`/`xy`, `effect_list` y rango kelvin.

El sync corrió el 2026-07-27, **~15 h después** de que los grupos murieran, y
`discover_capabilities` derivó las capacidades de esos atributos amputados. Resultado
medido en el sqlite de producción: los grupos muertos quedaron con **8 frases** (solo
on/off) contra **92** de los sanos. *"Poné la luz del cuarto al 50%"* ya no tiene frase
indexada.

Lo importante: **ese índice empobrecido sobrevive a que el dispositivo se recupere**,
porque la cache key del sync no incluye el estado. Con la migración Hue→Z2M en curso, esto
va a volver a pasar habitación por habitación.

**Files:**
- Modify: `scripts/sync_ha_to_chroma.py` (~línea 123 `cache_key`, y tras construir `selected`)
- Test: `tests/unit/vectordb/test_sync_unavailable_guard.py` (crear)

**Interfaces:**
- Produces: `select_syncable(entities) -> tuple[list, list]` → `(sanas, muertas)`.

- [ ] **Step 1: Escribir el test que falla**

```python
"""Syncing a dead entity writes a permanently impoverished index."""
import pytest

from scripts.sync_ha_to_chroma import select_syncable


def test_dead_entities_are_separated_from_live_ones():
    entities = [
        {"entity_id": "light.grupo_living", "state": "off"},
        {"entity_id": "light.grupo_cuarto", "state": "unavailable"},
        {"entity_id": "light.grupo_bano", "state": "unknown"},
    ]
    live, dead = select_syncable(entities)
    assert [e["entity_id"] for e in live] == ["light.grupo_living"]
    assert sorted(e["entity_id"] for e in dead) == [
        "light.grupo_bano", "light.grupo_cuarto",
    ]


def test_all_live_returns_no_dead():
    entities = [{"entity_id": "light.grupo_living", "state": "on"}]
    live, dead = select_syncable(entities)
    assert len(live) == 1 and dead == []
```

- [ ] **Step 2: Correr para verificar que falla**

```bash
.venv/bin/python -m pytest tests/unit/vectordb/test_sync_unavailable_guard.py -v ; echo "EXIT=$?"
```

Esperado: FAIL con `ImportError: cannot import name 'select_syncable'`.

- [ ] **Step 3: Implementar y cablear**

En `scripts/sync_ha_to_chroma.py`:

```python
def select_syncable(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separar entidades sanas de las que HA dejó sin atributos.

    Una entidad `unavailable` llega con los atributos amputados (HA la reduce a
    un stub), así que discover_capabilities le descubre menos capacidades de las
    que realmente tiene y la indexa con menos frases. Ese índice empobrecido
    sobrevive a la recuperación del dispositivo, porque la cache key del sync no
    mira el estado.
    """
    live, dead = [], []
    for e in entities:
        (dead if e.get("state") in ("unavailable", "unknown") else live).append(e)
    return live, dead
```

Tras construir `selected`, agregar el guard:

```python
    selected, dead = select_syncable(selected)
    if dead and not args.allow_unavailable:
        print(
            f"ABORTO: {len(dead)} entidades están unavailable/unknown y se "
            f"indexarían con capacidades amputadas:",
            file=sys.stderr,
        )
        for e in dead:
            print(f"  - {e['entity_id']} ({e.get('state')})", file=sys.stderr)
        print(
            "Corregí el dispositivo y reintentá, o pasá --allow-unavailable "
            "si de verdad querés indexarlas así.",
            file=sys.stderr,
        )
        sys.exit(2)
```

Agregar el flag: `ap.add_argument("--allow-unavailable", action="store_true")`.

Y meter el estado en la cache key (~línea 123) para que la recuperación fuerce
reindexación: agregar `e.get("state")` a la tupla que la compone.

- [ ] **Step 4: Verificar y commitear**

```bash
.venv/bin/python -m pytest tests/unit/vectordb/test_sync_unavailable_guard.py -v ; echo "EXIT=$?"
git add scripts/sync_ha_to_chroma.py tests/unit/vectordb/test_sync_unavailable_guard.py
git commit -m "fix(sync): no indexar entidades caídas con capacidades amputadas"
```

⚠️ Recordá que el sync manual **exige** `source /home/kza/secrets/llama-api-key.env` y que
el model id sea el path GGUF completo. Sin la key, `--wipe` borra igual y deja solo escenas.

---

### Task 5: El smoke test deriva su cobertura del índice real

`tools/smoke_test.py` evalúa dos conjuntos: 5 frases hardcodeadas (que resuelven solo a
living/cocina/escritorio, las tres que funcionan) y los `default_light` de las rooms del
config. Por eso reportó "2 problemas" con 5 grupos de luz caídos: `cuarto`, `balcón` y
`escalera` **no son `default_light` de ninguna room** y son invisibles para él — pese a que
resuelven por voz con similitud 0.92-1.00.

La cobertura la tiene que definir el sistema, no la memoria de quien escribió la lista.

**Files:**
- Modify: `src/monitoring/smoke_check.py`
- Modify: `tools/smoke_test.py:152-166`
- Test: `tests/unit/monitoring/test_smoke_check.py` (extender el existente)

- [ ] **Step 1: Escribir el test que falla**

```python
def test_indexed_entities_are_derived_from_chroma_not_a_list():
    """Coverage must come from what is addressable, not from a hardcoded list."""
    from src.monitoring.smoke_check import indexed_entity_ids

    fake_collection = type("C", (), {
        "get": staticmethod(lambda **kw: {"metadatas": [
            {"entity_id": "light.grupo_living"},
            {"entity_id": "light.grupo_cuarto"},
            {"entity_id": "light.grupo_living"},  # duplicado
        ]})
    })()
    assert indexed_entity_ids(fake_collection) == [
        "light.grupo_cuarto", "light.grupo_living",
    ]
```

- [ ] **Step 2: Correr para verificar que falla**

```bash
.venv/bin/python -m pytest tests/unit/monitoring/test_smoke_check.py -v ; echo "EXIT=$?"
```

- [ ] **Step 3: Implementar**

En `src/monitoring/smoke_check.py`:

```python
def indexed_entity_ids(collection) -> list[str]:
    """Los entity_id realmente direccionables por voz, según el índice.

    La lista hardcodeada de rooms deja fuera entidades vivas del índice
    (cuarto/balcón/escalera no son default_light de ninguna room), así que el
    smoke test salía verde para ellas.
    """
    got = collection.get(include=["metadatas"])
    ids = {
        m.get("entity_id")
        for m in (got.get("metadatas") or [])
        if m and m.get("entity_id")
    }
    return sorted(ids)
```

En `tools/smoke_test.py`, reemplazar el bloque que itera `default_light` por una iteración
sobre `indexed_entity_ids(...)`, corriendo `entity_problem()` sobre cada uno.

- [ ] **Step 4: Verificar y commitear**

```bash
.venv/bin/python -m pytest tests/unit/monitoring/test_smoke_check.py -v ; echo "EXIT=$?"
git add src/monitoring/smoke_check.py tools/smoke_test.py tests/unit/monitoring/test_smoke_check.py
git commit -m "fix(diag): el smoke test cubre lo que el índice expone, no una lista a mano"
```

Tras el deploy, el smoke test debería reportar **8 problemas** (5 luces + climate + 2 TV) en
vez de 2 — todos de la migración Hue pendiente, o sea esperados y fuera de alcance. El valor
es que ahora **se ven**.

---

## Self-review

**Cobertura.** Los cinco tasks cubren: colección de tests (A-6), pre-chequeo de
disponibilidad (A-4 parcial), alerta de sordera (A-5 + A-27), guard del sync (A-14) y
cobertura del smoke test (A-16). Quedan fuera, a propósito, los hallazgos que forman
proyectos independientes — listados abajo.

**Consistencia de tipos.** `is_entity_available` devuelve `bool | None` en el task 2 y el
task 3 no lo consume. `evaluate_health` y `write_audio_health` comparten la forma del
snapshot (`{"wall": float, "rooms": {id: {"age_s": float, "ever": bool}}}`) y están
definidas en el mismo task. `select_syncable` e `indexed_entity_ids` no se cruzan.

**Riesgo conocido.** El task 1 puede destapar tests rojos que hoy nadie ve, porque hoy no
corre ninguno. Eso es el punto: el baseline del step 5 existe para no confundir "lo rompí
yo" con "ya estaba roto".

---

## Planes siguientes (no incluidos acá)

Cada uno es un proyecto con su propio ciclo de test y merece su plan:

1. **Confirmación de efecto real por `context.id`.** HA propaga el context de la
   `ServiceCall` al `state_changed` de la entidad (ventana de 5 s,
   `homeassistant/helpers/entity.py:86`), y el frame del WS ya lo devuelve
   (`websocket_api/commands.py:292`). KZA lo tira. Cerrar ese lazo elimina la clase entera
   de "HA ackeó y no pasó nada" — y cuesta 0 ms porque `_fire_and_reconcile_ha` ya corre en
   background. Esfuerzo alto: toca el ciclo de vida del WS y la escritura de `events.db`.
2. **Instrumentación honesta de latencia.** `action_ms` (wake→HA) además del tramo interno;
   `LatencyMonitor.record()` recibiendo el wall-clock en vez de sumar el diccionario;
   resolver la colisión de la clave `timings["total"]`. Hoy el objetivo <300 ms se juzga
   contra un número que excluye ~1100 ms por delante y toda la llamada a HA por detrás.
3. **Resolver determinístico de entidades.** El vector search cuesta 158-208 ms —el
   componente más caro del fast path— para elegir entre **13 entidades indexadas**, con
   exactamente una por par `(area, service)`. Con `prefer_area` y `service_filter` ya
   resueltos antes de la consulta, no le queda nada que desambiguar. Es probablemente la
   mayor reducción de latencia disponible en todo el sistema.
4. **Higiene de config.** Anotar el bloque `rooms.wake_word.*` como inerte con
   `engine: openwakeword` (hoy carga los comentarios de calibración más detallados del
   archivo y **nada de eso se lee**); exponer `shadow_veto_timeout_s` (corre con 0.8 s
   hardcodeado = 2.6× el presupuesto del fast path); borrar o marcar
   `device_aliases`/`room_aliases`/`action_aliases`, que contradicen al mapa vivo; corregir
   el docstring de `multi_room_audio_loop.py:59` (dice que siempre se consume `indata[:, 0]`
   y es falso: cocina y escritorio están en `capture_channel: 1`).
5. **Bug silencioso de Kokoro.** `split_pattern` por defecto es `r'\n+'` y KZA nunca manda
   `\n`: **911 chars producen 24.77 s de audio cuando corresponden 56.40 s — se pierde el
   56% del texto**, sin error ni log. Solo afecta idiomas no ingleses.
6. **Experimentos de STT** (shadow, no destructivos): veto por backchannel inglés en modo
   log-only 48 h; A/B offline de `canary-180m-flash` con `language='es'` sobre los 2001
   clips ya capturados.

