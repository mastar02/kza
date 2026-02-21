# Q1: Architecture Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the 1,870-line VoicePipeline god object into 5 focused components, fix DI violations, convert ha_client to fully async, and clean up imports.

**Architecture:** Extract responsibilities from VoicePipeline into AudioLoop, RequestRouter, and FeatureManager. CommandProcessor and ResponseHandler already exist. VoicePipeline becomes a thin orchestrator (~200 lines) that wires the 5 components. All services are constructed in main.py and injected.

**Tech Stack:** Python 3.10+, asyncio, aiohttp, pytest, pytest-asyncio

---

## Task 1: Convert HomeAssistantClient REST Methods to Async

The most impactful single change: `ha_client.py` uses synchronous `requests` library for REST calls, blocking the event loop when called from async code. The WebSocket methods already use `aiohttp`.

**Files:**
- Modify: `src/home_assistant/ha_client.py` (all REST methods)
- Modify: `tests/unit/test_ha_client.py` (update to async tests)
- Modify: `tests/mocks/mock_ha_client.py` (update to async mock)
- Modify: `src/main.py:96` (test_connection becomes async)

**Step 1: Write failing tests for async ha_client**

```python
# tests/unit/test_ha_client_async.py
import pytest
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from src.home_assistant.ha_client import HomeAssistantClient


@pytest.mark.asyncio
async def test_get_all_entities_async():
    """get_all_entities should be async and use aiohttp"""
    client = HomeAssistantClient(url="http://localhost:8123", token="test")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[
        {"entity_id": "light.living", "state": "off"}
    ])
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)

    client._session = mock_session

    result = await client.get_all_entities()
    assert len(result) == 1
    assert result[0]["entity_id"] == "light.living"


@pytest.mark.asyncio
async def test_call_service_async():
    """call_service should be async and use aiohttp"""
    client = HomeAssistantClient(url="http://localhost:8123", token="test")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    client._session = mock_session

    result = await client.call_service("light", "turn_on", "light.living")
    assert result is True


@pytest.mark.asyncio
async def test_test_connection_async():
    """test_connection should be async"""
    client = HomeAssistantClient(url="http://localhost:8123", token="test")

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)

    client._session = mock_session

    result = await client.test_connection()
    assert result is True


@pytest.mark.asyncio
async def test_call_service_falls_back_on_error():
    """call_service should return False on error, not crash"""
    client = HomeAssistantClient(url="http://localhost:8123", token="test")

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

    client._session = mock_session

    result = await client.call_service("light", "turn_on", "light.living")
    assert result is False


@pytest.mark.asyncio
async def test_bare_except_replaced():
    """_reload_automations should use except Exception, not bare except"""
    import inspect
    source = inspect.getsource(HomeAssistantClient._reload_automations)
    assert "except:" not in source or "except Exception" in source
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ha_client_async.py -v`
Expected: FAIL — methods are sync, not async

**Step 3: Convert ha_client to fully async**

Replace the entire `ha_client.py` with async implementation:

```python
"""
Home Assistant Client
Comunicación con Home Assistant via REST API (aiohttp) y WebSocket
"""

import asyncio
import json
import logging
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Cliente async para comunicación con Home Assistant"""

    def __init__(self, url: str, token: str, timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection = None
        self._ws_connected = False
        self._ws_msg_id = 1
        self._ws_reconnect_attempts = 0
        self._ws_max_reconnect_attempts = 3

    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    # ==================== REST API (async) ====================

    async def get_all_entities(self) -> list[dict]:
        """Obtener todas las entidades de Home Assistant"""
        try:
            await self._ensure_session()
            async with self._session.get(
                f"{self.url}/api/states",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo entidades: {e}")
            return []

    async def get_all_services(self) -> list[dict]:
        """Obtener todos los servicios disponibles"""
        try:
            await self._ensure_session()
            async with self._session.get(
                f"{self.url}/api/services",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo servicios: {e}")
            return []

    async def get_entity_state(self, entity_id: str) -> Optional[dict]:
        """Obtener estado de una entidad específica"""
        try:
            await self._ensure_session()
            async with self._session.get(
                f"{self.url}/api/states/{entity_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo estado de {entity_id}: {e}")
            return None

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[dict] = None
    ) -> bool:
        """Ejecutar un servicio de Home Assistant (async)"""
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        try:
            await self._ensure_session()
            async with self._session.post(
                f"{self.url}/api/services/{domain}/{service}",
                json=payload
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Ejecutado: {domain}.{service} en {entity_id}")
                else:
                    logger.warning(f"Error {response.status}: {domain}.{service}")
                return success
        except Exception as e:
            logger.error(f"Error llamando servicio: {e}")
            return False

    # ==================== Automatizaciones ====================

    async def create_automation(self, automation_id: str, config: dict) -> tuple[bool, str]:
        """Crear una nueva automatización"""
        try:
            await self._ensure_session()
            async with self._session.post(
                f"{self.url}/api/config/automation/config/{automation_id}",
                json=config,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 201]:
                    await self._reload_automations()
                    logger.info(f"Automatización creada: {automation_id}")
                    return True, "OK"
                else:
                    error = await response.text()
                    logger.error(f"Error creando automatización: {error}")
                    return False, error
        except Exception as e:
            logger.error(f"Error creando automatización: {e}")
            return False, str(e)

    async def delete_automation(self, automation_id: str) -> bool:
        """Eliminar una automatización"""
        try:
            await self._ensure_session()
            async with self._session.delete(
                f"{self.url}/api/config/automation/config/{automation_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 204]:
                    await self._reload_automations()
                    logger.info(f"Automatización eliminada: {automation_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error eliminando automatización: {e}")
            return False

    async def get_automations(self) -> list[dict]:
        """Obtener lista de automatizaciones"""
        entities = await self.get_all_entities()
        return [e for e in entities if e["entity_id"].startswith("automation.")]

    async def _reload_automations(self):
        """Recargar automatizaciones en Home Assistant"""
        try:
            await self._ensure_session()
            async with self._session.post(
                f"{self.url}/api/services/automation/reload",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                pass
        except Exception as e:
            logger.error(f"Error reloading automations: {e}")

    # ==================== WebSocket ====================
    # (keep existing async WebSocket methods unchanged)

    async def connect_websocket(self) -> bool:
        """Conectar via WebSocket para menor latencia."""
        if self._ws_connected:
            return True

        ws_url = self.url.replace("http", "ws") + "/api/websocket"

        try:
            await self._ensure_session()

            self._ws_connection = await self._session.ws_connect(
                ws_url,
                heartbeat=30.0
            )

            msg = await asyncio.wait_for(
                self._ws_connection.receive_json(),
                timeout=5.0
            )

            if msg.get("type") == "auth_required":
                await self._ws_connection.send_json({
                    "type": "auth",
                    "access_token": self.token
                })

                auth_result = await asyncio.wait_for(
                    self._ws_connection.receive_json(),
                    timeout=5.0
                )

                if auth_result.get("type") == "auth_ok":
                    self._ws_connected = True
                    self._ws_reconnect_attempts = 0
                    logger.info("WebSocket HA conectado y autenticado")
                    return True
                else:
                    logger.error(f"Auth WebSocket falló: {auth_result}")

            return False

        except Exception as e:
            logger.error(f"Error conectando WebSocket HA: {e}")
            self._ws_connected = False
            return False

    async def ensure_websocket_connected(self) -> bool:
        """Asegurar que WebSocket está conectado."""
        if self._ws_connected and self._ws_connection:
            if self._ws_connection.closed:
                self._ws_connected = False

        if not self._ws_connected:
            if self._ws_reconnect_attempts < self._ws_max_reconnect_attempts:
                self._ws_reconnect_attempts += 1
                return await self.connect_websocket()

        return self._ws_connected

    async def call_service_ws(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[dict] = None
    ) -> bool:
        """Llamar servicio via WebSocket (más rápido que REST)."""
        if not await self.ensure_websocket_connected():
            logger.debug("WebSocket no disponible, usando REST")
            return await self.call_service(domain, service, entity_id, data)

        try:
            self._ws_msg_id += 1
            msg_id = self._ws_msg_id

            service_data = {"entity_id": entity_id}
            if data:
                service_data.update(data)

            await self._ws_connection.send_json({
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "service_data": service_data
            })

            response = await asyncio.wait_for(
                self._ws_connection.receive_json(),
                timeout=self.timeout
            )

            success = response.get("success", False)
            if success:
                logger.debug(f"WS: {domain}.{service} -> {entity_id}")
            return success

        except asyncio.TimeoutError:
            logger.warning("Timeout WebSocket, fallback a REST")
            return await self.call_service(domain, service, entity_id, data)

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False
            return await self.call_service(domain, service, entity_id, data)

    async def close(self):
        """Cerrar conexiones"""
        if self._ws_connection:
            await self._ws_connection.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ==================== Utilidades ====================

    async def get_domotics_entities(self) -> list[dict]:
        """Obtener solo entidades relevantes para domótica"""
        domotics_domains = [
            "light", "switch", "cover", "climate", "fan",
            "media_player", "vacuum", "lock", "scene",
            "script", "automation", "input_boolean"
        ]
        entities = await self.get_all_entities()
        return [
            e for e in entities
            if e["entity_id"].split(".")[0] in domotics_domains
        ]

    async def get_services_by_domain(self) -> dict[str, list[str]]:
        """Obtener servicios organizados por dominio"""
        services = await self.get_all_services()
        return {
            s["domain"]: list(s["services"].keys())
            for s in services
        }

    async def test_connection(self) -> bool:
        """Verificar conexión con Home Assistant (async)"""
        try:
            await self._ensure_session()
            async with self._session.get(
                f"{self.url}/api/",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
```

**Step 4: Update mock_ha_client to be async**

```python
# tests/mocks/mock_ha_client.py — update all methods to be async
# The mock should mirror the new async interface
```

**Step 5: Update main.py for async test_connection**

Change line 96 from `if not ha_client.test_connection():` to:
```python
if not await ha_client.test_connection():
```

Also add `await` to every `ha_client.call_service()` call throughout the codebase that was calling the sync version.

**Step 6: Update all callers of sync ha_client methods**

Search for all sync calls to ha_client methods and add `await`:
- `src/orchestrator/dispatcher.py` — `ha.call_service()` → `await ha.call_service()`
- `src/pipeline/voice_pipeline.py` — `ha.call_service()` → `await ha.call_service()`
- `src/vectordb/chroma_sync.py` — `ha.get_all_entities()` / `ha.get_domotics_entities()`
- `src/alerts/alert_scheduler.py` — `ha.get_entity_state()`
- `src/routines/routine_executor.py` — `ha.call_service()`

Run: `grep -rn "self\.ha\.\|ha_client\." src/ --include="*.py" | grep -v "async\|await\|__init__\|import"`

**Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 617+ tests pass. Fix any that break due to sync→async changes.

**Step 8: Commit**

```bash
git add src/home_assistant/ha_client.py tests/unit/test_ha_client_async.py tests/mocks/mock_ha_client.py src/main.py
git commit -m "refactor: convert HomeAssistantClient REST methods to async aiohttp

Eliminates synchronous requests library calls that were blocking
the event loop. All REST methods now use aiohttp.ClientSession.
Bare except in _reload_automations replaced with except Exception."
```

---

## Task 2: Extract FeatureManager from VoicePipeline

The simplest extraction: the "differentiating features" (timers, intercom, notifications, alerts, briefings) are already self-contained objects. VoicePipeline just holds references and delegates.

**Files:**
- Create: `src/pipeline/feature_manager.py`
- Create: `tests/unit/pipeline/test_feature_manager.py`
- Modify: `src/pipeline/voice_pipeline.py` (remove feature code)

**Step 1: Write failing test for FeatureManager**

```python
# tests/unit/pipeline/test_feature_manager.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.pipeline.feature_manager import FeatureManager


@pytest.fixture
def feature_deps():
    return {
        "timer_manager": MagicMock(start=AsyncMock(), stop=AsyncMock(), get_status=MagicMock(return_value={})),
        "intercom": MagicMock(start=AsyncMock(), stop=AsyncMock(), get_status=MagicMock(return_value={})),
        "notifications": MagicMock(start=AsyncMock(), stop=AsyncMock(), get_status=MagicMock(return_value={})),
        "alert_manager": MagicMock(),
        "alert_scheduler": MagicMock(start=AsyncMock(), stop=AsyncMock()),
        "ha_integration": MagicMock(start=AsyncMock(), stop=AsyncMock()),
    }


def test_feature_manager_init(feature_deps):
    fm = FeatureManager(**feature_deps)
    assert fm.timer_manager is feature_deps["timer_manager"]
    assert fm.intercom is feature_deps["intercom"]


@pytest.mark.asyncio
async def test_feature_manager_start(feature_deps):
    fm = FeatureManager(**feature_deps)
    await fm.start()
    feature_deps["timer_manager"].start.assert_called_once()
    feature_deps["intercom"].start.assert_called_once()
    feature_deps["notifications"].start.assert_called_once()
    feature_deps["alert_scheduler"].start.assert_called_once()
    feature_deps["ha_integration"].start.assert_called_once()


@pytest.mark.asyncio
async def test_feature_manager_stop(feature_deps):
    fm = FeatureManager(**feature_deps)
    await fm.stop()
    feature_deps["timer_manager"].stop.assert_called_once()
    feature_deps["intercom"].stop.assert_called_once()
    feature_deps["notifications"].stop.assert_called_once()
    feature_deps["alert_scheduler"].stop.assert_called_once()
    feature_deps["ha_integration"].stop.assert_called_once()


def test_feature_manager_get_status(feature_deps):
    fm = FeatureManager(**feature_deps)
    status = fm.get_status()
    assert "timers" in status
    assert "intercom" in status
    assert "notifications" in status
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_feature_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.feature_manager'`

**Step 3: Implement FeatureManager**

```python
# src/pipeline/feature_manager.py
"""
Feature Manager
Manages lifecycle and delegation for KZA differentiating features:
timers, intercom, notifications, alerts, HA integration.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FeatureManager:
    """Manages differentiating features lifecycle."""

    def __init__(
        self,
        timer_manager=None,
        intercom=None,
        notifications=None,
        alert_manager=None,
        alert_scheduler=None,
        ha_integration=None,
        briefing=None,
    ):
        self.timer_manager = timer_manager
        self.intercom = intercom
        self.notifications = notifications
        self.alert_manager = alert_manager
        self.alert_scheduler = alert_scheduler
        self.ha_integration = ha_integration
        self.briefing = briefing

    async def start(self):
        """Start all feature subsystems."""
        if self.timer_manager:
            await self.timer_manager.start()
            logger.info("Timer manager started")

        if self.intercom:
            await self.intercom.start()
            logger.info("Intercom system started")

        if self.notifications:
            await self.notifications.start()
            logger.info("Smart notifications started")

        if self.alert_scheduler:
            await self.alert_scheduler.start()
            logger.info("Alert scheduler started")

        if self.ha_integration:
            await self.ha_integration.start()
            logger.info("HA integration started")

    async def stop(self):
        """Stop all feature subsystems."""
        if self.timer_manager:
            await self.timer_manager.stop()
        if self.intercom:
            await self.intercom.stop()
        if self.notifications:
            await self.notifications.stop()
        if self.alert_scheduler:
            await self.alert_scheduler.stop()
        if self.ha_integration:
            await self.ha_integration.stop()

    def get_status(self) -> dict:
        """Get status of all features."""
        return {
            "timers": self.timer_manager.get_status() if self.timer_manager else None,
            "intercom": self.intercom.get_status() if self.intercom else None,
            "notifications": self.notifications.get_status() if self.notifications else None,
            "alerts": {
                "active": len(self.alert_manager.get_pending_alerts())
                if self.alert_manager and hasattr(self.alert_manager, "get_pending_alerts")
                else 0
            },
            "briefings": {
                "enabled": self.briefing is not None,
                "status": self.briefing.get_status() if self.briefing else None,
            },
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_feature_manager.py -v`
Expected: PASS

**Step 5: Remove feature code from VoicePipeline**

In `voice_pipeline.py`:
- Remove lines 319-359 (timer_manager, intercom, notifications, alert_manager, alert_scheduler, ha_integration init)
- Remove lines 1062-1080 (start calls for these in `run()`)
- Remove lines 1236-1240 (stop calls for these in `stop()`)
- Remove lines 1724-1869 (all passthrough methods: create_timer, announce, send_notification, add_alert_condition, etc.)
- Add `self.features = feature_manager` parameter and use it

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/pipeline/feature_manager.py tests/unit/pipeline/test_feature_manager.py src/pipeline/voice_pipeline.py
git commit -m "refactor: extract FeatureManager from VoicePipeline

Moves timer, intercom, notification, alert, and HA integration
lifecycle management into dedicated FeatureManager class.
Reduces VoicePipeline by ~200 lines."
```

---

## Task 3: Extract AudioLoop from VoicePipeline

The `run()` method's audio callback, wake word detection, VAD, echo suppression, and ambient detection loop become their own class.

**Files:**
- Create: `src/pipeline/audio_loop.py`
- Create: `tests/unit/pipeline/test_audio_loop.py`
- Modify: `src/pipeline/voice_pipeline.py` (remove run loop code)

**Step 1: Write failing test for AudioLoop**

```python
# tests/unit/pipeline/test_audio_loop.py
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from src.pipeline.audio_loop import AudioLoop


@pytest.fixture
def audio_deps():
    audio_mgr = MagicMock()
    audio_mgr.detect_wake_word = MagicMock(return_value=None)
    audio_mgr.wake_word_model = "hey_jarvis"
    audio_mgr.load_wake_word = MagicMock()
    audio_mgr.capture_command_with_vad = MagicMock(
        return_value=(True, 500.0, np.zeros(8000, dtype=np.float32), True)
    )

    echo_sup = MagicMock()
    echo_sup.is_safe_to_listen = True
    echo_sup.should_process_audio = MagicMock(return_value=(True, "ok"))
    echo_sup.config = MagicMock(post_speech_buffer_ms=400)
    echo_sup.state = MagicMock(value="idle")
    echo_sup.get_stats = MagicMock(return_value={})

    follow_up = MagicMock()
    follow_up.is_active = False
    follow_up.follow_up_window = 8.0

    return {
        "audio_manager": audio_mgr,
        "echo_suppressor": echo_sup,
        "follow_up": follow_up,
        "sample_rate": 16000,
    }


def test_audio_loop_init(audio_deps):
    loop = AudioLoop(**audio_deps)
    assert loop.sample_rate == 16000
    assert loop._running is False


def test_audio_loop_registers_command_callback(audio_deps):
    loop = AudioLoop(**audio_deps)
    callback = AsyncMock()
    loop.on_command(callback)
    assert loop._on_command_callback is callback


@pytest.mark.asyncio
async def test_audio_loop_stop(audio_deps):
    loop = AudioLoop(**audio_deps)
    loop._running = True
    await loop.stop()
    assert loop._running is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_audio_loop.py -v`
Expected: FAIL — module not found

**Step 3: Implement AudioLoop**

```python
# src/pipeline/audio_loop.py
"""
Audio Loop
Handles audio capture, wake word detection, VAD, echo suppression,
and ambient audio detection. Calls back when a command is captured.
"""

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioLoop:
    """Main audio capture and wake word detection loop."""

    def __init__(
        self,
        audio_manager,
        echo_suppressor,
        follow_up,
        sample_rate: int = 16000,
        ambient_detector=None,
    ):
        self.audio_manager = audio_manager
        self.echo_suppressor = echo_suppressor
        self.follow_up = follow_up
        self.sample_rate = sample_rate
        self.ambient_detector = ambient_detector
        self._running = False
        self._on_command_callback: Optional[Callable] = None

    def on_command(self, callback: Callable):
        """Register callback for when a command audio is captured."""
        self._on_command_callback = callback

    async def start(self):
        """Initialize audio subsystems."""
        self.audio_manager.load_wake_word()

        if self.ambient_detector:
            await self.ambient_detector.initialize()
            await self.ambient_detector.start()
            logger.info("Ambient detection active")

    async def run(self):
        """Main audio loop — listens for wake words and captures commands."""
        import sounddevice as sd

        CHUNK_SIZE = 1280
        audio_buffer = []
        listening_for_command = False
        command_start_time = None
        ambient_buffer = []

        self._running = True

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, listening_for_command, command_start_time, ambient_buffer

            audio_chunk = indata[:, 0].copy()

            if not self.echo_suppressor.is_safe_to_listen:
                return

            should_process, reason = self.echo_suppressor.should_process_audio(audio_chunk)
            if not should_process:
                return

            ambient_buffer.extend(audio_chunk)

            if not listening_for_command:
                detection = self.audio_manager.detect_wake_word(audio_chunk)

                if detection:
                    listening_for_command = True
                    command_start_time = time.time()
                    audio_buffer = []

                elif self.follow_up.is_active:
                    rms = np.sqrt(np.mean(audio_chunk ** 2))
                    if rms > 0.02:
                        if self.echo_suppressor.is_human_voice(audio_chunk):
                            listening_for_command = True
                            command_start_time = time.time()
                            audio_buffer = []
            else:
                audio_buffer.extend(audio_chunk)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        )

        with stream:
            while self._running:
                await asyncio.sleep(0.05)

                # Ambient detection
                if self.ambient_detector and len(ambient_buffer) >= self.sample_rate:
                    chunk = np.array(ambient_buffer[: self.sample_rate], dtype=np.float32)
                    ambient_buffer = ambient_buffer[self.sample_rate :]
                    task = asyncio.create_task(self._analyze_ambient(chunk))
                    task.add_done_callback(self._handle_task_error)

                # Command capture
                if listening_for_command:
                    is_complete, elapsed_ms, audio_data, early_exit = (
                        self.audio_manager.capture_command_with_vad(
                            audio_buffer,
                            command_start_time,
                            silence_threshold=0.015,
                            silence_duration_ms=300,
                            min_speech_ms=300,
                        )
                    )

                    if is_complete and audio_data is not None:
                        if self._on_command_callback:
                            await self._on_command_callback(audio_data)

                        listening_for_command = False
                        audio_buffer = []

    async def stop(self):
        """Stop the audio loop."""
        self._running = False
        if self.ambient_detector:
            await self.ambient_detector.stop()

    async def _analyze_ambient(self, chunk: np.ndarray):
        """Analyze ambient audio chunk."""
        if self.ambient_detector:
            await self.ambient_detector.analyze(chunk)

    @staticmethod
    def _handle_task_error(task: asyncio.Task):
        """Log errors from background tasks."""
        if not task.cancelled() and task.exception():
            logger.error(f"Background audio task failed: {task.exception()}")
```

**Step 4: Run tests**

Run: `pytest tests/unit/pipeline/test_audio_loop.py -v`
Expected: PASS

**Step 5: Update VoicePipeline to use AudioLoop**

Remove `run()` method body, replace with:
```python
async def run(self):
    """Start pipeline."""
    # Init models
    self.command_processor.load_models()
    self.chroma.initialize()
    if self.memory:
        self.memory.initialize()
    await self._connect_websocket()
    if self._orchestrator:
        await self._orchestrator.start()

    # Start features
    await self.features.start()

    # Start audio loop
    self.audio_loop.on_command(self.process_command)
    await self.audio_loop.start()
    await self.audio_loop.run()
```

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 7: Commit**

```bash
git add src/pipeline/audio_loop.py tests/unit/pipeline/test_audio_loop.py src/pipeline/voice_pipeline.py
git commit -m "refactor: extract AudioLoop from VoicePipeline

Moves audio capture, wake word detection, VAD, echo suppression,
and ambient detection into AudioLoop class. Adds error handling
for fire-and-forget asyncio tasks."
```

---

## Task 4: Extract RequestRouter from VoicePipeline

The command routing logic (`_process_command_orchestrated`, `_process_command_legacy`, cache, feedback detection, suggestion handling) moves out.

**Files:**
- Create: `src/pipeline/request_router.py`
- Create: `tests/unit/pipeline/test_request_router.py`
- Modify: `src/pipeline/voice_pipeline.py` (remove routing code)

**Step 1: Write failing test**

```python
# tests/unit/pipeline/test_request_router.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.pipeline.request_router import RequestRouter


@pytest.fixture
def router_deps():
    orchestrator = MagicMock()
    orchestrator.process = AsyncMock(return_value=MagicMock(
        path=MagicMock(value="fast_domotics"),
        priority=MagicMock(name="HIGH"),
        success=True,
        response="Luz encendida",
        intent="domotics",
        action={"domain": "light", "service": "turn_on", "entity_id": "light.living"},
        timings={"dispatch_ms": 50},
        was_queued=False,
        queue_position=None,
        to_dict=MagicMock(return_value={})
    ))

    cmd_processor = MagicMock()
    cmd_processor.process_command = AsyncMock(return_value={
        "text": "prende la luz",
        "user": None,
        "emotion": None,
        "timings": {"stt_ms": 80}
    })

    response_handler = MagicMock()
    response_handler.speak = AsyncMock()

    audio_manager = MagicMock()
    audio_manager.detect_source_zone = MagicMock(return_value="living")

    return {
        "orchestrator": orchestrator,
        "command_processor": cmd_processor,
        "response_handler": response_handler,
        "audio_manager": audio_manager,
    }


def test_request_router_init(router_deps):
    router = RequestRouter(**router_deps)
    assert router._orchestrator is router_deps["orchestrator"]


@pytest.mark.asyncio
async def test_request_router_process_command(router_deps):
    import numpy as np
    router = RequestRouter(**router_deps)
    audio = np.zeros(16000, dtype=np.float32)

    result = await router.process_command(audio)
    assert result["text"] == "prende la luz"
    router_deps["orchestrator"].process.assert_called_once()
```

**Step 2-7:** Same TDD pattern — implement, test, integrate, commit.

The `RequestRouter` class absorbs:
- `process_command()`, `_process_command_orchestrated()`, `_process_command_legacy()` from VoicePipeline
- `_query_cache`, `_cache_max_size`, `_add_to_cache()` (upgraded to TTLCache)
- `_detect_feedback()`, `_handle_feedback()`, `_check_suggestion()`

**Step 8: Commit**

```bash
git commit -m "refactor: extract RequestRouter from VoicePipeline

Moves command routing, cache (with TTL), feedback detection,
and suggestion handling into RequestRouter class."
```

---

## Task 5: Slim Down VoicePipeline to Thin Orchestrator

After Tasks 2-4, VoicePipeline should only hold the 5 components and wire them together.

**Files:**
- Modify: `src/pipeline/voice_pipeline.py` (reduce to ~200 lines)
- Modify: `tests/integration/test_voice_pipeline.py` (update)

**Step 1: Verify VoicePipeline is slim**

The `__init__` should accept only:
```python
def __init__(
    self,
    audio_loop: AudioLoop,
    command_processor: CommandProcessor,
    request_router: RequestRouter,
    response_handler: ResponseHandler,
    feature_manager: FeatureManager,
    chroma_sync=None,
    memory_manager=None,
    orchestrator=None,
):
```

**Step 2: Write test for slim pipeline**

```python
# tests/unit/pipeline/test_slim_pipeline.py
def test_voice_pipeline_has_few_params():
    import inspect
    from src.pipeline.voice_pipeline import VoicePipeline
    sig = inspect.signature(VoicePipeline.__init__)
    # self + max 10 params (down from 39)
    assert len(sig.parameters) <= 11, f"VoicePipeline has {len(sig.parameters)} params, expected <= 11"
```

**Step 3: Run test, verify it passes**

Run: `pytest tests/unit/pipeline/test_slim_pipeline.py -v`

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git commit -m "refactor: slim VoicePipeline to thin orchestrator (~200 lines)

VoicePipeline now accepts 5-8 components instead of 39 params.
All responsibilities delegated to AudioLoop, CommandProcessor,
RequestRouter, ResponseHandler, and FeatureManager."
```

---

## Task 6: Fix DI in main.py

Now that VoicePipeline is slim, main.py builds all components explicitly.

**Files:**
- Modify: `src/main.py` (rewrite DI chain)

**Step 1: Fix imports to canonical style**

Replace:
```python
sys.path.insert(0, str(Path(__file__).parent))
from stt.whisper_fast import create_stt
from tts.piper_tts import create_tts
```

With:
```python
from src.stt.whisper_fast import create_stt
from src.tts.piper_tts import create_tts
from src.vectordb.chroma_sync import ChromaSync
from src.home_assistant.ha_client import HomeAssistantClient
from src.llm.reasoner import LLMReasoner, FastRouter
from src.routines.routine_manager import RoutineManager
from src.pipeline.voice_pipeline import VoicePipeline
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.response_handler import ResponseHandler
from src.pipeline.feature_manager import FeatureManager
from src.pipeline.request_router import RequestRouter
# ... etc
```

**Step 2: Rewrite DI chain**

Build all components top-down, then inject:
```python
# Build infrastructure
ha_client = HomeAssistantClient(...)
chroma = ChromaSync(...)
llm = LLMReasoner(...)
router = FastRouter(...)

# Build audio
audio_manager = AudioManager(...)
echo_suppressor = EchoSuppressor(...)
follow_up = FollowUpMode(...)
audio_loop = AudioLoop(audio_manager, echo_suppressor, follow_up, ...)

# Build processing
command_processor = CommandProcessor(stt, speaker_id, user_manager, emotion_detector)
response_handler = ResponseHandler(tts, zone_manager, llm, ...)

# Build orchestration
context_manager = ContextManager(...)
queue = PriorityRequestQueue(...)
dispatcher = RequestDispatcher(chroma, ha_client, routine_manager, router, llm, ...)
orchestrator = MultiUserOrchestrator(dispatcher, context_manager, queue, ...)

# Build routing
request_router = RequestRouter(orchestrator, command_processor, response_handler, audio_manager)

# Build features
feature_manager = FeatureManager(timer_manager, intercom, notifications, alert_manager, alert_scheduler, ha_integration)

# Build pipeline
pipeline = VoicePipeline(audio_loop, command_processor, request_router, response_handler, feature_manager, chroma, memory_manager, orchestrator)
```

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git commit -m "refactor: rewrite main.py DI chain for new architecture

All components constructed explicitly in main.py and injected.
Eliminates sys.path hack, uses canonical imports.
VoicePipeline no longer creates MultiUserOrchestrator internally."
```

---

## Task 7: Remove Dead Code

**Files:**
- Move: `src/alerts/example_usage.py` → `docs/examples/alerts/example_usage.py`
- Move: `src/alerts/integration_example.py` → `docs/examples/alerts/integration_example.py`
- Move: `src/alerts/complete_integration_demo.py` → `docs/examples/alerts/complete_integration_demo.py`
- Modify: `src/pipeline/command_processor.py` (remove `classify_intent`)

**Step 1: Move example files**

```bash
mkdir -p docs/examples/alerts
mv src/alerts/example_usage.py docs/examples/alerts/
mv src/alerts/integration_example.py docs/examples/alerts/
mv src/alerts/complete_integration_demo.py docs/examples/alerts/
```

**Step 2: Remove dead classify_intent method**

Delete `CommandProcessor.classify_intent` static method (line ~287 in command_processor.py) — it's never called externally and has a dead `pass` branch.

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git commit -m "cleanup: remove dead code and move examples out of src/

Move 3 alert example files to docs/examples/alerts/.
Remove unused classify_intent static method from CommandProcessor."
```

---

## Verification Checklist

After completing all 7 tasks, verify:

- [ ] `pytest tests/ -v` — all 617+ tests pass
- [ ] `VoicePipeline.__init__` has ≤ 11 parameters (down from 39)
- [ ] `voice_pipeline.py` is ≤ 300 lines (down from 1,870)
- [ ] No `import requests` in `ha_client.py` (fully async)
- [ ] No bare `except:` in `ha_client.py` (replaced with `except Exception`)
- [ ] No `sys.path.insert` in `main.py`
- [ ] No example/demo files in `src/alerts/`
- [ ] `asyncio.create_task` calls have `.add_done_callback` error handlers
- [ ] All `ha_client` method calls use `await`

Run: `python -c "from src.pipeline.voice_pipeline import VoicePipeline; import inspect; print(len(inspect.signature(VoicePipeline.__init__).parameters))"`
Expected: ≤ 11

Run: `wc -l src/pipeline/voice_pipeline.py`
Expected: ≤ 300
