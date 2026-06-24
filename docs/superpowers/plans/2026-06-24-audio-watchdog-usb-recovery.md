# Audio Stream USB Auto-Recovery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el pipeline detecte cuando un micrófono USB deja de entregar audio (re-enumeración USB) y reabra el stream sobre el device actual, sin reiniciar el servicio.

**Architecture:** Un watchdog async (`_stream_watchdog`) en `MultiRoomAudioLoop` vigila `RoomStream.last_frame_ts` (estampado en el callback C de PortAudio). Si un stream supera el timeout sin frames, `_recover_streams` cierra **todos** los streams, reinicia el backend de PortAudio (`sd._terminate()/_initialize()` — global, invalida todo), espera a que el device reaparezca en sysfs y reabre re-resolviendo el índice por `mic_usb_port`. systemd `Restart=on-failure` (ya en el unit) es la red de seguridad última.

**Tech Stack:** Python 3.13, asyncio, sounddevice (PortAudio), pytest. Spec: `docs/superpowers/specs/2026-06-24-audio-watchdog-usb-recovery-design.md`.

## Global Constraints

- Python 3.13; tests con el venv del repo: `/Users/yo/Documents/kza/.venv/bin/python` (el `python3` del sistema es 3.9 y rompe en `dataclass slots=True`).
- async/await para todo I/O; nunca bloquear el event loop (usar `asyncio.to_thread` para llamadas sync largas).
- Inyección por constructor; `@dataclass` para DTOs; logging con `logger = logging.getLogger(__name__)`.
- Mensajes de voz/UI en español, código/logs en inglés.
- ⚠️ Los tests de `multi_room_audio_loop` mockean `sounddevice` vía `sys.modules` — **correr cada archivo de test aislado** (`pytest tests/unit/pipeline/test_multi_room_audio_loop.py`), no la suite completa (los mocks de sys.modules interfieren con otros módulos).
- Config nueva va bajo `rooms.stream_watchdog` en `config/settings.yaml` (el spec la llamó `audio.stream_watchdog`; se usa `rooms.*` por consistencia con el parseo existente, que lee todo el wiring del loop desde `config["rooms"]`).
- Nombres canónicos (idénticos en todas las tasks): `detect_stale_streams`, `_open_stream`, `_reinit_portaudio`, `_reopen_room`, `_recover_streams`, `_stream_watchdog`, `self._streams`, `self._watchdog_task`, `self._watchdog_enabled`, `self._watchdog_timeout_s`, `self._watchdog_check_interval_s`, `self._watchdog_backoff_min_s`, `self._watchdog_backoff_max_s`, `RoomStream.last_frame_ts`, `RoomStream.mic_usb_port`.

---

## File Structure

- `src/pipeline/multi_room_audio_loop.py` — toda la lógica del watchdog (función pura + métodos en `MultiRoomAudioLoop`), el campo nuevo en `RoomStream`, el estampado en el callback, y el refactor `streams`→`self._streams` / `_open_stream`.
- `src/main.py` — pasar `mic_usb_port` al construir cada `RoomStream` (línea ~774) y los kwargs del watchdog al construir `MultiRoomAudioLoop` (línea ~913).
- `config/settings.yaml` — sección `rooms.stream_watchdog`.
- `tests/unit/pipeline/test_multi_room_audio_loop.py` — tests nuevos (función pura, estampado, recovery, watchdog).

`src/rooms/room_context.py` NO se modifica: `resolve_mic_usb_port(usb_port, usb_root=..., devices=None)` y `usb_port_to_alsa_card(usb_port, usb_root=...)` se reusan tal cual.

---

### Task 1: Función pura `detect_stale_streams`

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (agregar función a nivel módulo, junto a `_resolve_capture_channels` ~línea 67)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Interfaces:**
- Produces: `detect_stale_streams(states: list[tuple[str, float]], now: float, timeout_s: float) -> list[str]` — recibe `[(room_id, last_frame_ts_monotonic), ...]`; devuelve los `room_id` cuyo `last_frame_ts > 0` y `now - last_frame_ts > timeout_s`. Streams con `last_frame_ts == 0.0` (nunca abiertos) se ignoran.

- [ ] **Step 1: Write the failing test**

Agregar al final de `tests/unit/pipeline/test_multi_room_audio_loop.py`:

```python
from src.pipeline.multi_room_audio_loop import detect_stale_streams


class TestDetectStaleStreams:
    def test_marks_stream_past_timeout(self):
        # last_frame_ts=100.0, now=109.0 → 9s sin frames > 8s
        assert detect_stale_streams([("escritorio", 100.0)], now=109.0, timeout_s=8.0) == ["escritorio"]

    def test_ignores_fresh_stream(self):
        # 2s sin frames < 8s
        assert detect_stale_streams([("escritorio", 100.0)], now=102.0, timeout_s=8.0) == []

    def test_ignores_never_opened_stream(self):
        # last_frame_ts=0.0 → nunca recibió/abrió, no se marca
        assert detect_stale_streams([("escritorio", 0.0)], now=999.0, timeout_s=8.0) == []

    def test_multiple_streams_only_stale_returned(self):
        states = [("a", 100.0), ("b", 108.5), ("c", 0.0)]
        # now=110: a=10s stale, b=1.5s fresh, c=never
        assert detect_stale_streams(states, now=110.0, timeout_s=8.0) == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestDetectStaleStreams -v`
Expected: FAIL with `ImportError: cannot import name 'detect_stale_streams'`

- [ ] **Step 3: Write minimal implementation**

En `src/pipeline/multi_room_audio_loop.py`, después de `_resolve_capture_channels` (~línea 67):

```python
def detect_stale_streams(
    states: list[tuple[str, float]], now: float, timeout_s: float
) -> list[str]:
    """Return room_ids whose audio stream stopped delivering frames.

    A stream is stale when it has produced at least one frame (last_frame_ts > 0)
    and more than `timeout_s` seconds elapsed since the last one. Streams that
    never opened (last_frame_ts == 0.0) are ignored — there is nothing to recover
    until `run()` opens them.

    Args:
        states: list of (room_id, last_frame_ts) with monotonic timestamps.
        now: current monotonic time.
        timeout_s: seconds without frames before a stream is considered dead.
    """
    return [
        room_id
        for room_id, last_frame_ts in states
        if last_frame_ts > 0.0 and (now - last_frame_ts) > timeout_s
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestDetectStaleStreams -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(audio): detect_stale_streams pure helper for stream watchdog"
```

---

### Task 2: `RoomStream` gana `last_frame_ts` y `mic_usb_port`; callback estampa el timestamp

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py:69-106` (dataclass `RoomStream`) y `:570` (primera línea de `audio_callback`)
- Modify: `src/main.py:774-783` (pasar `mic_usb_port=rc.mic_usb_port`)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Interfaces:**
- Consumes: nada de tasks previas.
- Produces: `RoomStream.last_frame_ts: float` (monotonic del último frame, `0.0` si ninguno) y `RoomStream.mic_usb_port: str | None` (puerto USB físico estable para re-resolver el índice). El callback de cada room actualiza `rs.last_frame_ts` en cada invocación.

- [ ] **Step 1: Write the failing test**

Agregar a `tests/unit/pipeline/test_multi_room_audio_loop.py`:

```python
import numpy as np
from src.pipeline.multi_room_audio_loop import RoomStream


class TestCallbackStampsFrameTimestamp:
    def test_fields_default(self):
        rs = _make_room_stream("escritorio", device_index=4)
        assert rs.last_frame_ts == 0.0
        assert rs.mic_usb_port is None

    def test_callback_updates_last_frame_ts(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        callback = loop._make_audio_callback(rs)
        indata = np.zeros((160, 2), dtype="float32")
        assert rs.last_frame_ts == 0.0
        callback(indata, 160, None, None)
        assert rs.last_frame_ts > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestCallbackStampsFrameTimestamp -v`
Expected: FAIL (`AttributeError: 'RoomStream' object has no attribute 'last_frame_ts'`)

- [ ] **Step 3: Write minimal implementation**

3a. En la dataclass `RoomStream`, después de `wake_score: float = 1.0` (~línea 106):

```python
    # Puerto USB físico estable (ej "3-1.4") para re-resolver el índice de
    # PortAudio si el device se re-enumera. None = no re-resolver por puerto.
    mic_usb_port: Optional[str] = None
    # Timestamp monotónico del último frame recibido por el callback. 0.0 hasta
    # que el stream se abre/recibe el primer frame. Lo vigila _stream_watchdog
    # para detectar un mic muerto por re-enumeración USB.
    last_frame_ts: float = 0.0
```

3b. En `audio_callback` (~línea 570), como **primera** sentencia dentro de la función, antes del tee al ambient path:

```python
        def audio_callback(indata, frames, time_info, status):
            # Watchdog heartbeat: marca que el stream entregó un frame. Primera
            # línea, O(1), nunca lanza — si esto deja de actualizarse, el mic
            # murió (re-enumeración USB) y _stream_watchdog dispara recovery.
            rs.last_frame_ts = time.monotonic()
```

3c. En `src/main.py`, en la construcción de `RoomStream` (~línea 774), agregar el kwarg:

```python
    room_streams[room_key] = RoomStream(
        room_id=room_key,
        device_index=rc.mic_device_index,
        wake_detector=wake_detector,
        echo_suppressor=room_echo,
        capture_channel=room_dict.get("capture_channel", 0),
        mic_usb_port=rc.mic_usb_port,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestCallbackStampsFrameTimestamp -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py src/main.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(audio): RoomStream.last_frame_ts heartbeat + mic_usb_port for re-resolution"
```

---

### Task 3: Extraer `_open_stream` y mover `streams` local → `self._streams`

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py:443-467` (apertura en `run()`), `:545-547` (finally), `__init__` (~línea 350: inicializar `self._streams`)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Interfaces:**
- Consumes: nada.
- Produces: `self._streams: dict[str, sd.InputStream]` (streams abiertos por room) y `MultiRoomAudioLoop._open_stream(rs: RoomStream) -> sd.InputStream | None` (abre y arranca un stream; `None` si `sd.PortAudioError`). Sin cambio de comportamiento observable: `run()` sigue abriendo un stream por room.

- [ ] **Step 1: Write the failing test**

Agregar a `tests/unit/pipeline/test_multi_room_audio_loop.py`:

```python
from unittest.mock import MagicMock, patch


class TestOpenStream:
    def test_open_stream_returns_started_stream(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        fake_stream = MagicMock()
        mock_sd.InputStream.return_value = fake_stream
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd):
            result = loop._open_stream(rs)
        assert result is fake_stream
        fake_stream.start.assert_called_once()

    def test_open_stream_returns_none_on_portaudio_error(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.side_effect = mock_sd.PortAudioError("no device")
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd):
            result = loop._open_stream(rs)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestOpenStream -v`
Expected: FAIL (`AttributeError: ... has no attribute '_open_stream'`)

- [ ] **Step 3: Write minimal implementation**

3a. En `__init__`, junto a las demás inicializaciones de estado de runtime (buscar `self._running` / `self._loop`; agregar cerca, ~línea 350):

```python
        self._streams: dict = {}
```

3b. Agregar el método `_open_stream` (por ejemplo justo antes de `run()`, ~línea 430):

```python
    def _open_stream(self, rs: "RoomStream"):
        """Open and start an InputStream for one room. None on PortAudioError."""
        callback = self._make_audio_callback(rs)
        try:
            dev_info = sd.query_devices(rs.device_index)
            capture_channels = _resolve_capture_channels(
                int(dev_info.get("max_input_channels", 0))
            )
            stream = sd.InputStream(
                device=rs.device_index,
                samplerate=self.sample_rate,
                channels=capture_channels,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=callback,
            )
            stream.start()
            logger.info(
                f"Room {rs.room_id}: audio stream started "
                f"(device={rs.device_index}, channels={capture_channels})"
            )
            return stream
        except sd.PortAudioError as e:
            logger.error(
                f"Room {rs.room_id}: failed to open device {rs.device_index}: {e}"
            )
            return None
```

3c. Reemplazar el bloque de apertura en `run()` (líneas 443-469) por:

```python
        self._streams = {}
        for room_id, rs in self.room_streams.items():
            stream = self._open_stream(rs)
            if stream is not None:
                self._streams[room_id] = stream
                rs.last_frame_ts = time.monotonic()

        logger.info(
            f"MultiRoomAudioLoop ready "
            f"({len(self._streams)}/{len(self.room_streams)} streams)"
        )
```

3d. Reemplazar el `finally` (líneas 544-547) por:

```python
        finally:
            for stream in self._streams.values():
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
```

- [ ] **Step 4: Run tests to verify they pass (incl. regression del run() existente)**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py -v`
Expected: PASS — los tests nuevos de `TestOpenStream` y **todos** los tests existentes de `run()`/`stop()` siguen verdes (refactor sin cambio de comportamiento).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "refactor(audio): extract _open_stream and track open streams in self._streams"
```

---

### Task 4: Recovery — `_reinit_portaudio`, `_reopen_room`, `_recover_streams`

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (agregar 3 métodos tras `_open_stream`) y `__init__` (params de backoff/timeout)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Interfaces:**
- Consumes: `_open_stream` (Task 3), `self._streams` (Task 3), `RoomStream.mic_usb_port` y `last_frame_ts` (Task 2), `resolve_mic_usb_port` (de `src.rooms.room_context`).
- Produces:
  - `_reinit_portaudio() -> None` (sync; `sd._terminate()` + `sd._initialize()`, fail-open).
  - `async _reopen_room(rs) -> None` (re-resuelve índice por `mic_usb_port` con `resolve_mic_usb_port`, espera con backoff si el device está ausente, reabre vía `_open_stream`, re-estampa `last_frame_ts`).
  - `async _recover_streams(trigger_room_ids: list[str]) -> None` (cierra **todos** los streams, `_reinit_portaudio` en `to_thread`, reabre **todos** los rooms).
- Constructor gana: `self._watchdog_timeout_s`, `self._watchdog_backoff_min_s`, `self._watchdog_backoff_max_s` (usados aquí y en Task 5).

- [ ] **Step 1: Write the failing test**

Agregar a `tests/unit/pipeline/test_multi_room_audio_loop.py`:

```python
import asyncio
import pytest


class TestRecoverStreams:
    @pytest.mark.asyncio
    async def test_recover_reinits_portaudio_and_reopens(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        old_stream = MagicMock()
        loop._streams = {"escritorio": old_stream}

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        new_stream = MagicMock()
        mock_sd.InputStream.return_value = new_stream
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            return_value=7,
        ):
            await loop._recover_streams(["escritorio"])

        old_stream.close.assert_called_once()          # cerró el muerto
        assert mock_sd._terminate.called and mock_sd._initialize.called  # reinit
        assert rs.device_index == 7                    # re-resolvió por puerto
        assert loop._streams["escritorio"] is new_stream  # reabrió
        assert rs.last_frame_ts > 0.0                  # re-estampó

    @pytest.mark.asyncio
    async def test_reopen_waits_with_backoff_when_device_absent(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_backoff_min_s = 0.001
        loop._watchdog_backoff_max_s = 0.004

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        mock_sd.InputStream.return_value = MagicMock()
        # 1ra resolución None (ausente), 2da devuelve índice → 1 reintento
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            side_effect=[None, 7],
        ):
            await loop._reopen_room(rs)

        assert rs.device_index == 7
        assert "escritorio" in loop._streams
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestRecoverStreams -v`
Expected: FAIL (`AttributeError: ... '_recover_streams'` y `_watchdog_backoff_min_s`)

- [ ] **Step 3: Write minimal implementation**

3a. Asegurar el import de `resolve_mic_usb_port` arriba en `multi_room_audio_loop.py` (junto a los otros `from src.rooms...` si existen; si no, agregar):

```python
from src.rooms.room_context import resolve_mic_usb_port
```

3b. En `__init__`, agregar a la **firma** (después de `wake_clip_writer=None,`, línea 148):

```python
        stream_watchdog_enabled: bool = False,
        stream_watchdog_no_frames_timeout_s: float = 8.0,
        stream_watchdog_check_interval_s: float = 2.0,
        stream_watchdog_reopen_backoff_min_s: float = 1.0,
        stream_watchdog_reopen_backoff_max_s: float = 10.0,
```

y al **cuerpo** del `__init__` (junto a `self._streams`, ~línea 350):

```python
        self._watchdog_enabled = stream_watchdog_enabled
        self._watchdog_timeout_s = stream_watchdog_no_frames_timeout_s
        self._watchdog_check_interval_s = stream_watchdog_check_interval_s
        self._watchdog_backoff_min_s = stream_watchdog_reopen_backoff_min_s
        self._watchdog_backoff_max_s = stream_watchdog_reopen_backoff_max_s
        self._watchdog_task = None
```

3c. Agregar los 3 métodos tras `_open_stream`:

```python
    def _reinit_portaudio(self) -> None:
        """Force PortAudio to re-scan the device list (sync; fail-open).

        PortAudio snapshots devices at Pa_Initialize and never re-scans. After a
        USB re-enumeration the cached indices point to dead devices, so we must
        terminate+initialize to see the new ones. This is GLOBAL: it invalidates
        every open stream, which is why _recover_streams reopens all of them.
        """
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio reinit failed: {e}")

    async def _reopen_room(self, rs: "RoomStream") -> None:
        """Re-resolve the device by USB port and reopen its stream, with backoff.

        Waits indefinitely (while self._running) for the device to reappear in
        sysfs — the service stays alive; only this mic waits. Never raises.
        """
        backoff = self._watchdog_backoff_min_s
        while self._running:
            new_index = rs.device_index
            if rs.mic_usb_port:
                resolved = resolve_mic_usb_port(rs.mic_usb_port)
                if resolved is None:
                    logger.warning(
                        f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} "
                        f"absent, retry in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._watchdog_backoff_max_s)
                    continue
                new_index = resolved
            rs.device_index = new_index
            stream = self._open_stream(rs)
            if stream is not None:
                self._streams[rs.room_id] = stream
                rs.last_frame_ts = time.monotonic()
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: recovered (device={new_index})"
                )
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._watchdog_backoff_max_s)

    async def _recover_streams(self, trigger_room_ids: list) -> None:
        """Close all streams, reinit PortAudio, reopen all rooms.

        sd._terminate() invalidates every stream, so recovery is all-or-nothing
        even if only one room went stale.
        """
        logger.error(
            f"[audio-watchdog] streams {trigger_room_ids} stopped delivering "
            f"audio → recovering all streams"
        )
        for room_id, stream in list(self._streams.items()):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()
        await asyncio.to_thread(self._reinit_portaudio)
        for room_id, rs in self.room_streams.items():
            await self._reopen_room(rs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestRecoverStreams -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(audio): _recover_streams reopens mics after USB re-enumeration"
```

---

### Task 5: `_stream_watchdog` task + wiring en `run()`

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (método `_stream_watchdog`; lanzar/cancelar en `run()`)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Interfaces:**
- Consumes: `detect_stale_streams` (Task 1), `_recover_streams` (Task 4), `self._watchdog_*` (Task 4), `self._running` (existente).
- Produces: `async _stream_watchdog() -> None` — loop que cada `check_interval_s` evalúa staleness y llama `_recover_streams`. Arranca en `run()` si `self._watchdog_enabled`; se cancela en el `finally`.

- [ ] **Step 1: Write the failing test**

```python
class TestStreamWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_recovers_when_stream_stale(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 0.05
        # frame "viejo": monotonic muy atrás → stale
        rs.last_frame_ts = time.monotonic() - 10.0

        called = {}
        async def fake_recover(ids):
            called["ids"] = ids
            loop._running = False  # corta el loop tras una recuperación
        loop._recover_streams = fake_recover

        await asyncio.wait_for(loop._stream_watchdog(), timeout=1.0)
        assert called.get("ids") == ["escritorio"]

    @pytest.mark.asyncio
    async def test_watchdog_noop_when_fresh(self):
        rs = _make_room_stream("escritorio", device_index=4)
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 5.0
        rs.last_frame_ts = time.monotonic()  # fresco

        called = {"n": 0}
        async def fake_recover(ids):
            called["n"] += 1
        loop._recover_streams = fake_recover

        async def stop_soon():
            await asyncio.sleep(0.05)
            loop._running = False
        await asyncio.gather(loop._stream_watchdog(), stop_soon())
        assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestStreamWatchdog -v`
Expected: FAIL (`AttributeError: ... '_stream_watchdog'`)

- [ ] **Step 3: Write minimal implementation**

3a. Agregar el método (tras `_recover_streams`):

```python
    async def _stream_watchdog(self) -> None:
        """Periodically detect mics that stopped delivering frames and recover.

        Reuses detect_stale_streams (pure) for the decision. Runs only while
        self._running; recovery is awaited so we never overlap two recoveries.
        """
        while self._running:
            await asyncio.sleep(self._watchdog_check_interval_s)
            if not self._running:
                break
            now = time.monotonic()
            states = [
                (room_id, rs.last_frame_ts)
                for room_id, rs in self.room_streams.items()
            ]
            stale = detect_stale_streams(states, now, self._watchdog_timeout_s)
            if stale:
                await self._recover_streams(stale)
```

3b. En `run()`, tras abrir los streams y loguear "ready" (después del bloque del Step 3c de Task 3), lanzar el task:

```python
        if self._watchdog_enabled:
            self._watchdog_task = asyncio.create_task(self._stream_watchdog())
            logger.info(
                f"[audio-watchdog] ACTIVO (timeout={self._watchdog_timeout_s}s, "
                f"check={self._watchdog_check_interval_s}s)"
            )
```

3c. En el `finally` de `run()` (el del Step 3d de Task 3), cancelar el task antes de cerrar los streams:

```python
        finally:
            if self._watchdog_task is not None:
                self._watchdog_task.cancel()
                self._watchdog_task = None
            for stream in self._streams.values():
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py -v`
Expected: PASS — `TestStreamWatchdog` verde y el resto de la suite del archivo sin regresiones.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(audio): _stream_watchdog task wired into run() lifecycle"
```

---

### Task 6: Config en `settings.yaml` + wiring en `main.py`

**Files:**
- Modify: `config/settings.yaml` (sección `rooms.stream_watchdog`)
- Modify: `src/main.py:~835` (parseo) y `:~913` (kwargs al constructor)
- Test: `tests/unit/pipeline/test_multi_room_audio_loop.py` (default-off sanity)

**Interfaces:**
- Consumes: los kwargs `stream_watchdog_*` del constructor (Task 4).
- Produces: el watchdog queda configurable y **activado** (`enabled: true`) en producción.

- [ ] **Step 1: Write the failing test (constructor default-off contract)**

```python
class TestWatchdogConfigContract:
    def test_disabled_by_default(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        assert loop._watchdog_enabled is False

    def test_enabled_via_kwarg(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)},
            stream_watchdog_enabled=True,
            stream_watchdog_no_frames_timeout_s=8.0,
        )
        assert loop._watchdog_enabled is True
        assert loop._watchdog_timeout_s == 8.0
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestWatchdogConfigContract -v`
Expected: PASS si Task 4 ya agregó los kwargs (este test fija el contrato; si falla, revisar Task 4 Step 3b).

- [ ] **Step 3: Add config + main.py wiring**

3a. En `config/settings.yaml`, bajo la clave `rooms:` (hermano de `wake_word:`, `endpointing:`, `barge_in:`), agregar:

```yaml
  # Watchdog del stream de audio: detecta cuando un mic USB deja de entregar
  # frames (re-enumeración USB / pérdida de corriente del bus) y reabre el
  # stream sobre el device actual sin reiniciar el servicio. Ver
  # docs/superpowers/specs/2026-06-24-audio-watchdog-usb-recovery-design.md
  stream_watchdog:
    enabled: true
    no_frames_timeout_s: 8.0     # sin frames del callback por más de esto → recuperar
    check_interval_s: 2.0        # cada cuánto evalúa el watchdog
    reopen_backoff_min_s: 1.0    # espera inicial entre reintentos de reabrir
    reopen_backoff_max_s: 10.0   # tope del backoff
```

3b. En `src/main.py`, junto a `early_cfg = rooms_config.get("wake_word", {})` (~línea 835):

```python
    watchdog_cfg = rooms_config.get("stream_watchdog", {}) or {}
```

3c. En el constructor `MultiRoomAudioLoop(...)` (~línea 913), agregar los kwargs:

```python
        stream_watchdog_enabled=watchdog_cfg.get("enabled", False),
        stream_watchdog_no_frames_timeout_s=watchdog_cfg.get("no_frames_timeout_s", 8.0),
        stream_watchdog_check_interval_s=watchdog_cfg.get("check_interval_s", 2.0),
        stream_watchdog_reopen_backoff_min_s=watchdog_cfg.get("reopen_backoff_min_s", 1.0),
        stream_watchdog_reopen_backoff_max_s=watchdog_cfg.get("reopen_backoff_max_s", 10.0),
```

- [ ] **Step 4: Run test + verify config parses**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_multi_room_audio_loop.py::TestWatchdogConfigContract -v`
Expected: PASS (2 passed)

Run (sanity de YAML): `.venv/bin/python -c "import yaml; c=yaml.safe_load(open('config/settings.yaml')); print(c['rooms']['stream_watchdog'])"`
Expected: imprime `{'enabled': True, 'no_frames_timeout_s': 8.0, 'check_interval_s': 2.0, 'reopen_backoff_min_s': 1.0, 'reopen_backoff_max_s': 10.0}`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py config/settings.yaml src/main.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat(audio): enable stream watchdog via rooms.stream_watchdog config"
```

---

### Task 7: Validación en hardware (manual, coordinada con el usuario — NO automatizada)

**No es un test de CI.** Es el procedimiento de validación en el server, en producción, junto al usuario. Documentarlo aquí cierra el plan; ejecutarlo requiere el XVF3800 real.

- [ ] **Step 1: Deploy de la rama al server** (coordinado; seguir el flujo `kza-push` / deploy in-place habitual; el server corre 10 commits atrás de main — alinear o cherry-pickear según se decida).

- [ ] **Step 2: Confirmar watchdog activo en el arranque**

Run (server): `journalctl --user -u kza-voice --no-pager | grep "audio-watchdog"`
Expected: `[audio-watchdog] ACTIVO (timeout=8.0s, check=2.0s)`

- [ ] **Step 3: Forzar la re-enumeración USB sin tocar el cable**

⚠️ Producción — ejecutar junto al usuario. Reproduce el incidente del 21/6:

```bash
echo 0 | sudo tee /sys/bus/usb/devices/3-1.4/authorized
sleep 3
echo 1 | sudo tee /sys/bus/usb/devices/3-1.4/authorized
```

- [ ] **Step 4: Verificar detección + recuperación automática**

Run (server): `journalctl --user -u kza-voice --no-pager --since "2 min ago" | grep "audio-watchdog"`
Expected: `... stopped delivering audio → recovering all streams` seguido (dentro de ~10-15s) de `... escritorio: recovered (device=N)`.

- [ ] **Step 5: Prueba de voz end-to-end**

Decir "Nexa, prendé la luz del escritorio". Expected: la luz responde (el wake vuelve a dispararse sobre el stream recuperado). Confirmar en logs `[oww-dbg]` con `rms` fluctuante.

---

## Self-Review

**Spec coverage** (contra `docs/superpowers/specs/2026-06-24-...md`):
- Detección por `last_frame_ts` + watchdog → Tasks 1, 2, 5. ✓
- `_recover_streams` (stop → terminate/initialize → esperar device → re-resolver por puerto → reabrir) → Task 4. ✓
- Refactor `_open_stream` → Task 3. ✓
- Reinit global de PortAudio + reapertura coordinada multi-room → Task 4 (`_recover_streams` reabre todos). ✓
- Device ausente: backoff indefinido sin crashear → Task 4 (`_reopen_room`). ✓
- Red de seguridad systemd `Restart=on-failure` → ya en el unit (no requiere código); excepción no prevista se propaga. ✓
- Config `stream_watchdog` (timeout 8s, backoff) → Task 6 (bajo `rooms.*`, ajuste anotado vs `audio.*` del spec). ✓
- Testing unit sin hardware + integración por `authorized` → Tasks 1-6 (unit) + Task 7 (hardware). ✓
- Falso positivo / idempotencia: el umbral 8s + `_reopen_room` que reabre el mismo índice si está sano. ✓

**Placeholder scan:** sin TBD/TODO; todo step de código incluye el código literal. ✓

**Type consistency:** `detect_stale_streams(states, now, timeout_s)`, `_open_stream(rs)→stream|None`, `_recover_streams(trigger_room_ids: list)`, `_reopen_room(rs)`, `_reinit_portaudio()`, `_stream_watchdog()`, atributos `self._watchdog_*`/`self._streams`/`self._watchdog_task`, campos `RoomStream.last_frame_ts`/`mic_usb_port` — nombres idénticos en todas las tasks. ✓

**Nota de riesgo (validar en Task 7):** el comportamiento exacto de `sd._terminate()/_initialize()` para re-enumerar dentro de un proceso vivo es el supuesto central; si en hardware no refresca la lista, el fallback es propagar la excepción → systemd reinicia (nunca peor que hoy).
