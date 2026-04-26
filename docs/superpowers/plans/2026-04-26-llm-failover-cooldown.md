# LLM Failover + Cooldown + Idle Timeout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cuando un endpoint LLM (vLLM 7B `:8100` o llama-cpp 72B `:8200`) falla o se cuelga, el dispatcher rota automáticamente al siguiente candidato sin que el usuario sienta latencia adicional, con cooldown exponencial para no martillar endpoints caídos.

**Architecture:** Una clase nueva `LLMRouter` envuelve los endpoints existentes (`FastRouter`, `HttpReasoner`) sin modificarlos estructuralmente. El router itera una candidate chain ordenada por prioridad, salta los que están en cooldown, y ante fallo clasificado como "failover-worthy" marca el endpoint con backoff exponencial (1m→5m→25m→1h) persistido en disco. Un idle watchdog aborta requests al 72B en CPU si no llegan chunks por X segundos. Patrón basado en `docs/concepts/model-failover.md` y `docs/concepts/agent-loop.md` de OpenClaw.

**Tech Stack:** Python 3.13, asyncio, dataclasses, Enum, pytest + fixtures existentes en `tests/conftest.py` y mocks en `tests/mocks/mock_llm.py`. Persistencia: JSON file en `./data/llm_cooldowns.json`. Sin dependencias nuevas.

---

## File Structure

### Crear

| Path | Responsabilidad |
|------|-----------------|
| `src/llm/types.py` | Enums (`EndpointKind`, `ErrorKind`) y dataclasses (`LLMEndpoint`, `CooldownState`, `RouterResult`). |
| `src/llm/error_classifier.py` | Mapper puro: `Exception → ErrorKind` (rate-limit, timeout, billing, format, idle, permanent). |
| `src/llm/cooldown.py` | `CooldownManager` con backoff exponencial in-memory + persistence JSON. |
| `src/llm/router.py` | `LLMRouter` con candidate chain, cooldown skip, retry attempts, `FallbackSummaryError`. |
| `src/llm/idle_watchdog.py` | Helper async que envuelve un stream y aborta si no recibe chunks en N segundos. |
| `src/llm/router_factory.py` | Construye `LLMRouter` desde dict de config (`config["llm"]["failover"]`). |
| `tests/unit/llm/__init__.py` | Vacío. |
| `tests/unit/llm/test_types.py` | Tests de Enums + dataclasses (smoke: instantiation, comparison, defaults). |
| `tests/unit/llm/test_error_classifier.py` | Tests de clasificación con exceptions reales (httpx.TimeoutException, openai.RateLimitError, etc.). |
| `tests/unit/llm/test_cooldown.py` | Backoff sequence, `is_available()`, persistence round-trip, success-clears. |
| `tests/unit/llm/test_router.py` | Candidate chain skip, all-failed → `FallbackSummaryError`, retry attempts, success records. |
| `tests/unit/llm/test_idle_watchdog.py` | Aborta cuando no llega chunk en N segundos; pasa cuando llegan chunks. |
| `tests/integration/test_llm_failover_e2e.py` | E2E: 7B mock falla → router cae a 72B mock → 7B vuelve después del cooldown. |

### Modificar

| Path | Cambio |
|------|--------|
| `src/llm/__init__.py` | Exportar `LLMRouter`, `LLMEndpoint`, `EndpointKind`, `ErrorKind`, `FallbackSummaryError`. |
| `src/llm/reasoner.py` | Agregar parámetro opcional `idle_timeout_s` a `HttpReasoner.__init__`. Wrap del stream interno con `idle_watchdog`. |
| `src/main.py` | Líneas 237-286: en lugar de pasar `fast_router` directo al `RequestDispatcher`, construir un `LLMRouter` con ambos endpoints y pasar el router. |
| `src/orchestrator/dispatcher.py` | Línea 565: cambiar `self.router.generate([text], max_tokens=128)[0]` por `await self.router.complete(text, max_tokens=128)` que retorna `RouterResult`. |
| `config/settings.yaml` | Después del bloque `reasoner:` (línea ~250), agregar bloque `llm.failover` con `endpoints`, `cooldowns`, `retry`, `idle_timeout_s`. |

---

## Conventions

- **Imports:** stdlib → third-party → `from src.modulo import Clase`. Nada relativo.
- **Logging:** `logger = logging.getLogger(__name__)` en cada módulo, prefijo descriptivo (`[LLMRouter]`, `[Cooldown]`).
- **Tests:** clases `TestX` con métodos `test_y(self, fixtures)`. Fixtures vía conftest.
- **Async:** todo I/O de red usa `await`. CooldownManager es sync (solo file I/O simple).
- **No bloquear event loop:** persistence file I/O usa `asyncio.to_thread(...)` o se hace en thread pool.

---

## Task 1: Tipos básicos (Enums + dataclasses)

**Files:**
- Create: `src/llm/types.py`
- Test: `tests/unit/llm/test_types.py`
- Create: `tests/unit/llm/__init__.py` (vacío)

- [ ] **Step 1.1: Crear directorio + __init__**

```bash
mkdir -p tests/unit/llm
touch tests/unit/llm/__init__.py
```

- [ ] **Step 1.2: Escribir test fallido `tests/unit/llm/test_types.py`**

```python
"""Tests for LLM router type primitives."""

import pytest
from src.llm.types import (
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    CooldownState,
    RouterResult,
)


class TestEndpointKind:
    def test_values(self):
        assert EndpointKind.FAST_ROUTER.value == "fast_router"
        assert EndpointKind.HTTP_REASONER.value == "http_reasoner"
        assert EndpointKind.LOCAL_REASONER.value == "local_reasoner"
        assert EndpointKind.CLOUD.value == "cloud"


class TestErrorKind:
    def test_values(self):
        assert ErrorKind.RATE_LIMIT.value == "rate_limit"
        assert ErrorKind.TIMEOUT.value == "timeout"
        assert ErrorKind.IDLE_TIMEOUT.value == "idle_timeout"
        assert ErrorKind.BILLING.value == "billing"
        assert ErrorKind.AUTH.value == "auth"
        assert ErrorKind.FORMAT.value == "format"
        assert ErrorKind.PERMANENT.value == "permanent"

    def test_is_failover_worthy(self):
        assert ErrorKind.RATE_LIMIT.is_failover_worthy() is True
        assert ErrorKind.TIMEOUT.is_failover_worthy() is True
        assert ErrorKind.IDLE_TIMEOUT.is_failover_worthy() is True
        assert ErrorKind.BILLING.is_failover_worthy() is True
        assert ErrorKind.FORMAT.is_failover_worthy() is True
        # Auth/permanent suben a la app — no rotación automática
        assert ErrorKind.AUTH.is_failover_worthy() is False
        assert ErrorKind.PERMANENT.is_failover_worthy() is False


class TestLLMEndpoint:
    def test_required_fields(self):
        ep = LLMEndpoint(
            id="primary",
            kind=EndpointKind.FAST_ROUTER,
            client=object(),
            priority=0,
        )
        assert ep.id == "primary"
        assert ep.kind == EndpointKind.FAST_ROUTER
        assert ep.priority == 0
        # defaults
        assert ep.timeout_s == 30.0
        assert ep.idle_timeout_s is None
        assert ep.max_tokens_default == 256

    def test_ordering_by_priority(self):
        a = LLMEndpoint(id="a", kind=EndpointKind.FAST_ROUTER, client=None, priority=0)
        b = LLMEndpoint(id="b", kind=EndpointKind.HTTP_REASONER, client=None, priority=1)
        # priority lower = tried first
        assert a.priority < b.priority


class TestCooldownState:
    def test_defaults(self):
        s = CooldownState(endpoint_id="primary")
        assert s.endpoint_id == "primary"
        assert s.error_count == 0
        assert s.cooldown_until == 0.0
        assert s.last_used == 0.0
        assert s.last_error_kind is None

    def test_to_dict_roundtrip(self):
        s = CooldownState(
            endpoint_id="primary",
            error_count=2,
            cooldown_until=1234567890.5,
            last_used=1234567880.0,
            last_error_kind=ErrorKind.RATE_LIMIT,
        )
        d = s.to_dict()
        s2 = CooldownState.from_dict(d)
        assert s2 == s


class TestRouterResult:
    def test_success(self):
        r = RouterResult(
            text="hola",
            endpoint_id="primary",
            attempts=1,
            elapsed_ms=42.0,
        )
        assert r.text == "hola"
        assert r.endpoint_id == "primary"
        assert r.attempts == 1
        assert r.elapsed_ms == 42.0
```

- [ ] **Step 1.3: Run test — must FAIL (module not found)**

```bash
pytest tests/unit/llm/test_types.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.types'`

- [ ] **Step 1.4: Implementar `src/llm/types.py`**

```python
"""
LLM Router type primitives.

Patrón inspirado en OpenClaw model-failover (docs/concepts/model-failover.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EndpointKind(Enum):
    """Tipo de endpoint LLM. Usado para logging y selección de cliente."""

    FAST_ROUTER = "fast_router"        # vLLM 7B :8100 (compartido infra)
    HTTP_REASONER = "http_reasoner"    # llama-cpp 72B :8200 (kza-72b.service)
    LOCAL_REASONER = "local_reasoner"  # llama-cpp en proceso (legacy)
    CLOUD = "cloud"                    # OpenAI/Anthropic/etc (futuro)


class ErrorKind(Enum):
    """Clasificación de errores LLM. Determina si rotamos o subimos a la app."""

    RATE_LIMIT = "rate_limit"      # 429, "throttled", "concurrency limit"
    TIMEOUT = "timeout"            # connect timeout, read timeout
    IDLE_TIMEOUT = "idle_timeout"  # stream sin chunks por N segundos
    BILLING = "billing"            # 402, "insufficient credits"
    FORMAT = "format"              # JSON inválido, schema mismatch
    AUTH = "auth"                  # 401, 403 (no rotamos: requiere acción humana)
    PERMANENT = "permanent"        # cualquier otro error definitivo

    def is_failover_worthy(self) -> bool:
        """¿Este error justifica rotar al siguiente candidato?"""
        return self in {
            ErrorKind.RATE_LIMIT,
            ErrorKind.TIMEOUT,
            ErrorKind.IDLE_TIMEOUT,
            ErrorKind.BILLING,
            ErrorKind.FORMAT,
        }


@dataclass
class LLMEndpoint:
    """Un endpoint LLM concreto en la candidate chain."""

    id: str
    kind: EndpointKind
    client: Any  # FastRouter | HttpReasoner | LLMReasoner | otro
    priority: int  # menor = primero
    timeout_s: float = 30.0
    idle_timeout_s: Optional[float] = None  # None = sin watchdog
    max_tokens_default: int = 256


@dataclass
class CooldownState:
    """Estado de cooldown persistido por endpoint."""

    endpoint_id: str
    error_count: int = 0
    cooldown_until: float = 0.0  # epoch seconds; 0 = no cooldown
    last_used: float = 0.0
    last_error_kind: Optional[ErrorKind] = None

    def to_dict(self) -> dict:
        return {
            "endpoint_id": self.endpoint_id,
            "error_count": self.error_count,
            "cooldown_until": self.cooldown_until,
            "last_used": self.last_used,
            "last_error_kind": self.last_error_kind.value if self.last_error_kind else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CooldownState:
        kind_raw = d.get("last_error_kind")
        return cls(
            endpoint_id=d["endpoint_id"],
            error_count=d.get("error_count", 0),
            cooldown_until=d.get("cooldown_until", 0.0),
            last_used=d.get("last_used", 0.0),
            last_error_kind=ErrorKind(kind_raw) if kind_raw else None,
        )


@dataclass
class RouterResult:
    """Resultado de una invocación exitosa del router."""

    text: str
    endpoint_id: str
    attempts: int
    elapsed_ms: float
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 1.5: Run test — must PASS**

```bash
pytest tests/unit/llm/test_types.py -v
```
Expected: 9 passed

- [ ] **Step 1.6: Commit**

```bash
git add src/llm/types.py tests/unit/llm/__init__.py tests/unit/llm/test_types.py
git commit -m "feat(llm): add router type primitives (EndpointKind, ErrorKind, LLMEndpoint, CooldownState)"
```

---

## Task 2: Error classifier

**Files:**
- Create: `src/llm/error_classifier.py`
- Test: `tests/unit/llm/test_error_classifier.py`

- [ ] **Step 2.1: Escribir test fallido**

```python
"""Tests for LLM error classifier."""

import asyncio
import pytest
from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind


class TestClassifyError:
    def test_asyncio_timeout(self):
        assert classify_error(asyncio.TimeoutError()) == ErrorKind.TIMEOUT

    def test_timeout_error_builtin(self):
        assert classify_error(TimeoutError("read")) == ErrorKind.TIMEOUT

    def test_connection_error(self):
        assert classify_error(ConnectionError("refused")) == ErrorKind.TIMEOUT

    def test_rate_limit_message(self):
        e = RuntimeError("429 Too Many Requests")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_throttling(self):
        e = RuntimeError("ThrottlingException: too many requests")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_concurrency(self):
        e = RuntimeError("concurrency limit reached")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_rate_limit_quota(self):
        e = RuntimeError("quota limit exceeded for this minute")
        assert classify_error(e) == ErrorKind.RATE_LIMIT

    def test_billing_credits(self):
        e = RuntimeError("insufficient credits in your account")
        assert classify_error(e) == ErrorKind.BILLING

    def test_billing_balance(self):
        e = RuntimeError("credit balance too low")
        assert classify_error(e) == ErrorKind.BILLING

    def test_auth_401(self):
        e = RuntimeError("401 Unauthorized: invalid api key")
        assert classify_error(e) == ErrorKind.AUTH

    def test_auth_403(self):
        e = RuntimeError("403 Forbidden")
        assert classify_error(e) == ErrorKind.AUTH

    def test_format_invalid_json(self):
        e = ValueError("Invalid JSON in response")
        assert classify_error(e) == ErrorKind.FORMAT

    def test_idle_timeout_explicit(self):
        # Custom exception class for idle timeout
        from src.llm.error_classifier import IdleTimeoutError
        assert classify_error(IdleTimeoutError(15.0)) == ErrorKind.IDLE_TIMEOUT

    def test_unknown_error_is_permanent(self):
        e = RuntimeError("something completely unexpected")
        assert classify_error(e) == ErrorKind.PERMANENT
```

- [ ] **Step 2.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_error_classifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.error_classifier'`

- [ ] **Step 2.3: Implementar `src/llm/error_classifier.py`**

```python
"""
Mapper Exception → ErrorKind.

Patrón inspirado en OpenClaw model-failover.md ("What lands in the rate-limit /
timeout bucket"). Conservador: errores desconocidos se clasifican como PERMANENT
para no rotar erróneamente.
"""

from __future__ import annotations

import asyncio
import logging

from src.llm.types import ErrorKind

logger = logging.getLogger(__name__)


class IdleTimeoutError(Exception):
    """Raised por idle_watchdog cuando el stream no emite chunks en N segundos."""

    def __init__(self, idle_seconds: float):
        self.idle_seconds = idle_seconds
        super().__init__(f"No chunks received for {idle_seconds:.1f}s")


# Patrones de texto que indican rate-limit / quota
_RATE_LIMIT_PATTERNS = (
    "429",
    "too many requests",
    "rate limit",
    "rate_limit",
    "throttl",  # Throttling, throttled
    "concurrency limit",
    "quota",
    "resource exhausted",
    "weekly limit",
    "monthly limit",
    "daily limit",
)

# Patrones que indican billing/credits agotados
_BILLING_PATTERNS = (
    "insufficient credits",
    "credit balance too low",
    "credits exhausted",
    "billing",
    "402",
)

# Patrones de auth (401/403)
_AUTH_PATTERNS = (
    "401",
    "unauthorized",
    "403",
    "forbidden",
    "invalid api key",
    "invalid token",
)

# Patrones de format error
_FORMAT_PATTERNS = (
    "invalid json",
    "json decode",
    "schema",
    "unexpected token",
)


def classify_error(exc: BaseException) -> ErrorKind:
    """Clasificar una exception en un ErrorKind."""

    # Idle timeout primero (es nuestro tipo, sabemos qué es)
    if isinstance(exc, IdleTimeoutError):
        return ErrorKind.IDLE_TIMEOUT

    # Timeouts a nivel asyncio o builtin
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return ErrorKind.TIMEOUT

    # Texto del error para matchear patrones
    msg = str(exc).lower()

    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return ErrorKind.RATE_LIMIT

    if any(p in msg for p in _BILLING_PATTERNS):
        return ErrorKind.BILLING

    if any(p in msg for p in _AUTH_PATTERNS):
        return ErrorKind.AUTH

    if any(p in msg for p in _FORMAT_PATTERNS):
        return ErrorKind.FORMAT

    if isinstance(exc, ValueError):
        # ValueError sin patrón conocido = format
        return ErrorKind.FORMAT

    return ErrorKind.PERMANENT
```

- [ ] **Step 2.4: Run test — must PASS**

```bash
pytest tests/unit/llm/test_error_classifier.py -v
```
Expected: 14 passed

- [ ] **Step 2.5: Commit**

```bash
git add src/llm/error_classifier.py tests/unit/llm/test_error_classifier.py
git commit -m "feat(llm): classify exceptions into ErrorKind for failover decisions"
```

---

## Task 3: CooldownManager (in-memory)

**Files:**
- Create: `src/llm/cooldown.py`
- Test: `tests/unit/llm/test_cooldown.py`

- [ ] **Step 3.1: Escribir test fallido**

```python
"""Tests for CooldownManager (in-memory behavior, sin persistence)."""

import time
import pytest
from src.llm.cooldown import CooldownManager, BACKOFF_SCHEDULE_S
from src.llm.types import ErrorKind


class TestCooldownManagerInMemory:
    def test_initially_available(self, tmp_path):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        assert mgr.is_available("primary") is True

    def test_record_success_clears_state(self, tmp_path):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.is_available("primary") is False
        mgr.record_success("primary")
        assert mgr.is_available("primary") is True
        assert mgr.get_state("primary").error_count == 0

    def test_backoff_schedule_progresses(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        # 1ra falla → 60s
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1060.0)

        # avanzar past cooldown
        now[0] = 1100.0
        assert mgr.is_available("primary") is True

        # 2da falla → 300s (5min)
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1400.0)

        # 3ra falla → 1500s (25min)
        now[0] = 1500.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1500.0 + 1500.0)

        # 4ta falla → 3600s (1h, cap)
        now[0] = 5000.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(5000.0 + 3600.0)

        # 5ta falla → sigue 3600s (cap)
        now[0] = 10000.0
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(10000.0 + 3600.0)

    def test_billing_uses_long_backoff(self, tmp_path, monkeypatch):
        """Billing → cooldown directo de 1h (no progresión gradual)."""
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        mgr.record_failure("primary", ErrorKind.BILLING)
        assert mgr.get_state("primary").cooldown_until == pytest.approx(1000.0 + 3600.0)

    def test_is_available_respects_now(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.is_available("primary") is False

        # apenas antes del expiry
        now[0] = 1059.9
        assert mgr.is_available("primary") is False

        # justo después
        now[0] = 1060.1
        assert mgr.is_available("primary") is True

    def test_next_attempt_at(self, tmp_path, monkeypatch):
        mgr = CooldownManager(persistence_path=tmp_path / "cd.json")
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        assert mgr.next_attempt_at("primary") == 0.0  # available now
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert mgr.next_attempt_at("primary") == pytest.approx(1060.0)

    def test_backoff_schedule_constants(self):
        # Verifica que los valores documentados no se alteren accidentalmente
        assert BACKOFF_SCHEDULE_S == (60, 300, 1500, 3600)
```

- [ ] **Step 3.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_cooldown.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.cooldown'`

- [ ] **Step 3.3: Implementar `src/llm/cooldown.py` (sin persistence todavía — solo in-memory + stub disco)**

```python
"""
CooldownManager: backoff exponencial 1m → 5m → 25m → 1h por endpoint.

Patrón en OpenClaw model-failover.md. Cooldown se persiste a disco para
sobrevivir reinicios del proceso (si el endpoint estaba caído al apagar,
queremos seguir respetando su cooldown al reiniciar).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock

from src.llm.types import CooldownState, ErrorKind

logger = logging.getLogger(__name__)


# Backoff: 1min, 5min, 25min, 1h (cap)
BACKOFF_SCHEDULE_S = (60, 300, 1500, 3600)

# Para BILLING saltamos directo al máximo (no es transitorio)
BILLING_COOLDOWN_S = 3600


class CooldownManager:
    """Maneja cooldowns por endpoint con backoff exponencial."""

    def __init__(self, persistence_path: Path | str):
        self.persistence_path = Path(persistence_path)
        self._states: dict[str, CooldownState] = {}
        self._lock = Lock()
        self._load_from_disk()

    def is_available(self, endpoint_id: str) -> bool:
        """¿Se puede intentar este endpoint ahora?"""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                return True
            return time.time() >= state.cooldown_until

    def next_attempt_at(self, endpoint_id: str) -> float:
        """Epoch seconds del próximo intento permitido. 0 si está disponible."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                return 0.0
            now = time.time()
            return state.cooldown_until if state.cooldown_until > now else 0.0

    def get_state(self, endpoint_id: str) -> CooldownState:
        """Estado actual (crea uno fresh si no existía)."""
        with self._lock:
            return self._states.get(endpoint_id, CooldownState(endpoint_id=endpoint_id))

    def record_failure(self, endpoint_id: str, kind: ErrorKind) -> None:
        """Registrar fallo y aplicar backoff según kind + error_count previo."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                state = CooldownState(endpoint_id=endpoint_id)
                self._states[endpoint_id] = state

            now = time.time()
            state.last_error_kind = kind

            if kind == ErrorKind.BILLING:
                cooldown_s = BILLING_COOLDOWN_S
            else:
                idx = min(state.error_count, len(BACKOFF_SCHEDULE_S) - 1)
                cooldown_s = BACKOFF_SCHEDULE_S[idx]

            state.error_count += 1
            state.cooldown_until = now + cooldown_s

            logger.warning(
                f"[Cooldown] {endpoint_id}: kind={kind.value} "
                f"count={state.error_count} cooldown={cooldown_s}s"
            )

            self._save_to_disk()

    def record_success(self, endpoint_id: str) -> None:
        """Resetear contador y limpiar cooldown."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                state = CooldownState(endpoint_id=endpoint_id)
                self._states[endpoint_id] = state

            state.error_count = 0
            state.cooldown_until = 0.0
            state.last_used = time.time()
            state.last_error_kind = None

            self._save_to_disk()

    # ---- Persistence (file-based, simple) ----

    def _load_from_disk(self) -> None:
        if not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            for entry in data.get("states", []):
                state = CooldownState.from_dict(entry)
                self._states[state.endpoint_id] = state
            logger.info(f"[Cooldown] Loaded {len(self._states)} states from disk")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[Cooldown] Failed to load {self.persistence_path}: {e}")

    def _save_to_disk(self) -> None:
        """Atomic write (write temp + rename)."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persistence_path.with_suffix(".tmp")
            payload = {"states": [s.to_dict() for s in self._states.values()]}
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.persistence_path)
        except OSError as e:
            logger.error(f"[Cooldown] Failed to save {self.persistence_path}: {e}")
```

- [ ] **Step 3.4: Run test — must PASS**

```bash
pytest tests/unit/llm/test_cooldown.py -v
```
Expected: 7 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/llm/cooldown.py tests/unit/llm/test_cooldown.py
git commit -m "feat(llm): cooldown manager with exponential backoff (1m/5m/25m/1h)"
```

---

## Task 4: CooldownManager — persistence round-trip test

**Files:**
- Modify: `tests/unit/llm/test_cooldown.py` (agregar clase nueva)

- [ ] **Step 4.1: Agregar test de persistence al final de `test_cooldown.py`**

```python
class TestCooldownPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        path = tmp_path / "cd.json"
        now = [1000.0]
        monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])

        mgr1 = CooldownManager(persistence_path=path)
        mgr1.record_failure("primary", ErrorKind.RATE_LIMIT)
        mgr1.record_failure("secondary", ErrorKind.BILLING)

        # Nuevo manager lee del mismo path
        mgr2 = CooldownManager(persistence_path=path)
        assert mgr2.get_state("primary").error_count == 1
        assert mgr2.get_state("primary").last_error_kind == ErrorKind.RATE_LIMIT
        assert mgr2.get_state("secondary").last_error_kind == ErrorKind.BILLING

    def test_corrupt_file_does_not_crash(self, tmp_path):
        path = tmp_path / "cd.json"
        path.write_text("not valid json {{{")
        # Constructor no debe explotar; solo log warning + estado vacío
        mgr = CooldownManager(persistence_path=path)
        assert mgr.is_available("anything") is True

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "cd.json"
        mgr = CooldownManager(persistence_path=path)
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert path.exists()

    def test_atomic_write_via_tmp(self, tmp_path):
        """Verifica que NO queda un .tmp después de operación exitosa."""
        path = tmp_path / "cd.json"
        mgr = CooldownManager(persistence_path=path)
        mgr.record_failure("primary", ErrorKind.RATE_LIMIT)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
```

- [ ] **Step 4.2: Run tests — must PASS (CooldownManager ya tiene save/load implementado en Task 3)**

```bash
pytest tests/unit/llm/test_cooldown.py -v
```
Expected: 11 passed

- [ ] **Step 4.3: Commit**

```bash
git add tests/unit/llm/test_cooldown.py
git commit -m "test(llm): cooldown persistence roundtrip + corruption resilience"
```

---

## Task 5: LLMRouter — candidate chain básica

**Files:**
- Create: `src/llm/router.py`
- Test: `tests/unit/llm/test_router.py`

- [ ] **Step 5.1: Escribir test fallido**

```python
"""Tests for LLMRouter candidate chain."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter, FallbackSummaryError
from src.llm.types import EndpointKind, ErrorKind, LLMEndpoint


class _FakeClient:
    """Cliente fake con interface mínima: async complete(prompt, **kw) -> str."""

    def __init__(self, response: str = "ok", raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.calls = 0

    async def complete(self, prompt: str, max_tokens: int = 256, **_kw) -> str:
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.response


@pytest.fixture
def cd_manager(tmp_path):
    return CooldownManager(persistence_path=tmp_path / "cd.json")


@pytest.fixture
def primary_ok():
    return LLMEndpoint(
        id="primary", kind=EndpointKind.FAST_ROUTER,
        client=_FakeClient(response="primary-ok"), priority=0,
    )


@pytest.fixture
def secondary_ok():
    return LLMEndpoint(
        id="secondary", kind=EndpointKind.HTTP_REASONER,
        client=_FakeClient(response="secondary-ok"), priority=1,
    )


class TestLLMRouterHappyPath:
    @pytest.mark.asyncio
    async def test_uses_primary_when_available(self, cd_manager, primary_ok, secondary_ok):
        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.text == "primary-ok"
        assert result.endpoint_id == "primary"
        assert result.attempts == 1
        assert primary_ok.client.calls == 1
        assert secondary_ok.client.calls == 0

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_fails(self, cd_manager, secondary_ok):
        primary = LLMEndpoint(
            id="primary", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("429 rate limit")),
            priority=0,
        )
        router = LLMRouter([primary, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.text == "secondary-ok"
        assert result.endpoint_id == "secondary"
        assert result.attempts == 2

        # primary debe quedar en cooldown
        assert cd_manager.is_available("primary") is False
        # secondary debe haber registrado success
        assert cd_manager.get_state("secondary").error_count == 0


class TestLLMRouterCooldownSkip:
    @pytest.mark.asyncio
    async def test_skips_endpoint_in_cooldown(self, cd_manager, primary_ok, secondary_ok):
        # Marcar primary en cooldown manualmente
        cd_manager.record_failure("primary", ErrorKind.RATE_LIMIT)

        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.endpoint_id == "secondary"
        # primary nunca se invocó (cooldown skip)
        assert primary_ok.client.calls == 0


class TestLLMRouterAllFailed:
    @pytest.mark.asyncio
    async def test_all_failed_raises_fallback_summary(self, cd_manager):
        a = LLMEndpoint(
            id="a", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("429")),
            priority=0,
        )
        b = LLMEndpoint(
            id="b", kind=EndpointKind.HTTP_REASONER,
            client=_FakeClient(raises=TimeoutError("read")),
            priority=1,
        )
        router = LLMRouter([a, b], cd_manager)

        with pytest.raises(FallbackSummaryError) as exc_info:
            await router.complete("hola", max_tokens=10)

        err = exc_info.value
        assert len(err.attempts) == 2
        assert err.attempts[0].endpoint_id == "a"
        assert err.attempts[0].error_kind == ErrorKind.RATE_LIMIT
        assert err.attempts[1].endpoint_id == "b"
        assert err.attempts[1].error_kind == ErrorKind.TIMEOUT

    @pytest.mark.asyncio
    async def test_auth_error_does_not_failover(self, cd_manager, secondary_ok):
        """ErrorKind.AUTH no es failover-worthy → debe propagar la exception original."""
        primary = LLMEndpoint(
            id="primary", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("401 Unauthorized")),
            priority=0,
        )
        router = LLMRouter([primary, secondary_ok], cd_manager)

        with pytest.raises(RuntimeError, match="401"):
            await router.complete("hola", max_tokens=10)

        # secondary nunca se invocó porque auth no triggered failover
        assert secondary_ok.client.calls == 0


class TestLLMRouterAllInCooldown:
    @pytest.mark.asyncio
    async def test_all_in_cooldown_raises_with_next_attempt(self, cd_manager, primary_ok, secondary_ok):
        cd_manager.record_failure("primary", ErrorKind.RATE_LIMIT)
        cd_manager.record_failure("secondary", ErrorKind.RATE_LIMIT)

        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        with pytest.raises(FallbackSummaryError) as exc_info:
            await router.complete("hola", max_tokens=10)

        err = exc_info.value
        # No se llamó a ningún cliente
        assert primary_ok.client.calls == 0
        assert secondary_ok.client.calls == 0
        # Pero el FallbackSummaryError reporta que todos están en cooldown
        assert err.soonest_retry_at is not None
        assert err.soonest_retry_at > 0
```

- [ ] **Step 5.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_router.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.router'`

- [ ] **Step 5.3: Implementar `src/llm/router.py`**

```python
"""
LLMRouter: candidate chain con cooldown skip y failover.

Patrón en OpenClaw model-failover.md "Runtime flow":
1. Resolve session state (skip — no session aquí)
2. Build candidate chain
3. Try current provider
4. Advance on failover-worthy errors
5. Persist fallback override (skip — sin sesión)
6. Roll back narrowly (skip)
7. Throw FallbackSummaryError if exhausted

Nota: el router NO maneja retry attempts dentro del mismo endpoint —
eso es responsabilidad del cliente (HttpReasoner / FastRouter ya tienen su
propio retry interno). Aquí solo rotamos entre endpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.llm.cooldown import CooldownManager
from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind, LLMEndpoint, RouterResult

logger = logging.getLogger(__name__)


@dataclass
class FailedAttempt:
    """Registro per-attempt para el FallbackSummaryError."""
    endpoint_id: str
    error_kind: ErrorKind
    error_message: str


class FallbackSummaryError(Exception):
    """Todos los candidatos fallaron o están en cooldown."""

    def __init__(
        self,
        attempts: list[FailedAttempt],
        soonest_retry_at: Optional[float] = None,
    ):
        self.attempts = attempts
        self.soonest_retry_at = soonest_retry_at
        retry_str = (
            f" (next attempt at epoch {soonest_retry_at:.0f})"
            if soonest_retry_at else ""
        )
        super().__init__(
            f"All {len(attempts)} LLM endpoints exhausted{retry_str}: "
            + "; ".join(f"{a.endpoint_id}={a.error_kind.value}" for a in attempts)
        )


class LLMRouter:
    """Router con candidate chain ordenada por priority."""

    def __init__(
        self,
        endpoints: list[LLMEndpoint],
        cooldown_manager: CooldownManager,
    ):
        if not endpoints:
            raise ValueError("LLMRouter requires at least one endpoint")
        # Orden estable por priority asc
        self._endpoints = sorted(endpoints, key=lambda e: e.priority)
        self._cd = cooldown_manager

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs,
    ) -> RouterResult:
        """
        Iterar candidatos en orden de priority. Saltar los en cooldown.
        Ante fallo failover-worthy → cooldown + next. Ante AUTH/PERMANENT → propagar.
        """
        start = time.perf_counter()
        attempts: list[FailedAttempt] = []
        soonest_retry: Optional[float] = None

        for ep in self._endpoints:
            if not self._cd.is_available(ep.id):
                next_at = self._cd.next_attempt_at(ep.id)
                if soonest_retry is None or next_at < soonest_retry:
                    soonest_retry = next_at
                logger.debug(f"[LLMRouter] skip {ep.id} (cooldown until {next_at:.0f})")
                continue

            try:
                logger.debug(f"[LLMRouter] try {ep.id} (kind={ep.kind.value})")
                text = await ep.client.complete(prompt, max_tokens=max_tokens, **kwargs)
            except Exception as exc:
                kind = classify_error(exc)
                logger.warning(
                    f"[LLMRouter] {ep.id} failed: kind={kind.value} "
                    f"err={type(exc).__name__}: {exc}"
                )

                if not kind.is_failover_worthy():
                    # AUTH/PERMANENT — no rotamos. Propagamos.
                    raise

                self._cd.record_failure(ep.id, kind)
                attempts.append(FailedAttempt(
                    endpoint_id=ep.id,
                    error_kind=kind,
                    error_message=str(exc),
                ))
                continue

            # Éxito
            self._cd.record_success(ep.id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RouterResult(
                text=text,
                endpoint_id=ep.id,
                attempts=len(attempts) + 1,
                elapsed_ms=elapsed_ms,
            )

        # Todos los endpoints fallaron o están en cooldown
        raise FallbackSummaryError(attempts=attempts, soonest_retry_at=soonest_retry)
```

- [ ] **Step 5.4: Asegurar que `pytest-asyncio` esté instalado**

```bash
pip show pytest-asyncio || pip install pytest-asyncio
```

Si no estaba instalado, agregarlo a `requirements-dev.txt` (o `requirements.txt` si KZA no separa). Verificar:

```bash
grep pytest-asyncio requirements*.txt 2>/dev/null
```

Si no aparece, agregar a `requirements.txt`:

```bash
echo "pytest-asyncio>=0.21" >> requirements.txt
```

- [ ] **Step 5.5: Configurar pytest-asyncio mode (en `pyproject.toml` o `pytest.ini`)**

Verificar config existente:

```bash
grep -E "asyncio_mode|asyncio" /Users/yo/Documents/kza/pyproject.toml /Users/yo/Documents/kza/pytest.ini /Users/yo/Documents/kza/setup.cfg 2>/dev/null
```

Si no hay config, agregar en `pyproject.toml` (o crear si no existe sección `[tool.pytest.ini_options]`):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 5.6: Run test — must PASS**

```bash
pytest tests/unit/llm/test_router.py -v
```
Expected: 6 passed

- [ ] **Step 5.7: Commit**

```bash
git add src/llm/router.py tests/unit/llm/test_router.py requirements.txt pyproject.toml
git commit -m "feat(llm): LLMRouter with candidate chain failover and cooldown skip"
```

---

## Task 6: Idle watchdog para streams

**Files:**
- Create: `src/llm/idle_watchdog.py`
- Test: `tests/unit/llm/test_idle_watchdog.py`

- [ ] **Step 6.1: Escribir test fallido**

```python
"""Tests for idle_watchdog wrapping async streams."""

import asyncio
import pytest
from src.llm.idle_watchdog import idle_watchdog
from src.llm.error_classifier import IdleTimeoutError


async def _fast_stream():
    """Emite 3 chunks en ~30ms total."""
    for i in range(3):
        await asyncio.sleep(0.01)
        yield f"chunk-{i}"


async def _hanging_stream():
    """Emite 1 chunk y luego se cuelga."""
    yield "first"
    await asyncio.sleep(10.0)  # mucho más que el watchdog


async def _slow_stream(gap: float):
    """Emite 2 chunks separados por `gap` segundos."""
    yield "a"
    await asyncio.sleep(gap)
    yield "b"


class TestIdleWatchdog:
    @pytest.mark.asyncio
    async def test_passes_fast_stream(self):
        chunks = []
        async for chunk in idle_watchdog(_fast_stream(), idle_seconds=1.0):
            chunks.append(chunk)
        assert chunks == ["chunk-0", "chunk-1", "chunk-2"]

    @pytest.mark.asyncio
    async def test_aborts_hanging_stream(self):
        chunks = []
        with pytest.raises(IdleTimeoutError) as exc_info:
            async for chunk in idle_watchdog(_hanging_stream(), idle_seconds=0.1):
                chunks.append(chunk)
        assert chunks == ["first"]  # alcanzó a emitir 1 antes de colgarse
        assert exc_info.value.idle_seconds == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_passes_slow_stream_within_budget(self):
        # gap=0.05s, watchdog=0.2s → pasa
        chunks = []
        async for chunk in idle_watchdog(_slow_stream(0.05), idle_seconds=0.2):
            chunks.append(chunk)
        assert chunks == ["a", "b"]

    @pytest.mark.asyncio
    async def test_aborts_slow_stream_exceeding_budget(self):
        # gap=0.3s, watchdog=0.1s → aborta
        chunks = []
        with pytest.raises(IdleTimeoutError):
            async for chunk in idle_watchdog(_slow_stream(0.3), idle_seconds=0.1):
                chunks.append(chunk)
        assert chunks == ["a"]
```

- [ ] **Step 6.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_idle_watchdog.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.idle_watchdog'`

- [ ] **Step 6.3: Implementar `src/llm/idle_watchdog.py`**

```python
"""
idle_watchdog: envuelve un async iterator y aborta si no llega chunk en N segundos.

Patrón en OpenClaw agent-loop.md (`agents.defaults.llm.idleTimeoutSeconds`).
Útil para 72B en CPU que puede colgarse silenciosamente sin cerrar el stream.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, TypeVar

from src.llm.error_classifier import IdleTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def idle_watchdog(
    source: AsyncIterator[T],
    idle_seconds: float,
) -> AsyncIterator[T]:
    """
    Re-yield cada chunk de `source`. Si no llega nuevo chunk en `idle_seconds`,
    raise IdleTimeoutError.

    Args:
        source: async iterator/generator
        idle_seconds: segundos máximos sin chunk

    Yields:
        Cada chunk de `source`

    Raises:
        IdleTimeoutError: si el gap entre chunks excede `idle_seconds`
    """
    iterator = source.__aiter__()

    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=idle_seconds)
        except asyncio.TimeoutError as e:
            logger.warning(f"[IdleWatchdog] no chunks for {idle_seconds:.1f}s — aborting")
            raise IdleTimeoutError(idle_seconds) from e
        except StopAsyncIteration:
            return
        yield chunk
```

- [ ] **Step 6.4: Run test — must PASS**

```bash
pytest tests/unit/llm/test_idle_watchdog.py -v
```
Expected: 4 passed

- [ ] **Step 6.5: Commit**

```bash
git add src/llm/idle_watchdog.py tests/unit/llm/test_idle_watchdog.py
git commit -m "feat(llm): idle_watchdog aborts streams that stall (for slow 72B)"
```

---

## Task 7: Adapter clients (`complete()` async wrappers)

**Files:**
- Create: `src/llm/adapters.py`
- Test: `tests/unit/llm/test_adapters.py`

**Contexto:** `FastRouter` y `HttpReasoner` actuales tienen `generate()` y `__call__` síncronos. El `LLMRouter.complete()` espera `async def complete(prompt, ...) -> str`. Creamos adapters thin que envuelven los clientes existentes en una interface uniforme `complete()` async (usando `asyncio.to_thread` para no bloquear el event loop).

- [ ] **Step 7.1: Escribir test fallido**

```python
"""Tests for sync→async adapter clients."""

import asyncio
import pytest
from unittest.mock import MagicMock

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter


class TestFastRouterAdapter:
    @pytest.mark.asyncio
    async def test_complete_calls_generate_in_thread(self):
        # FastRouter.generate retorna list[str]
        fr = MagicMock()
        fr.generate = MagicMock(return_value=["respuesta-fast"])
        adapter = FastRouterAdapter(fr)

        result = await adapter.complete("hola", max_tokens=10)

        assert result == "respuesta-fast"
        fr.generate.assert_called_once_with(["hola"], max_tokens=10, temperature=0.3)

    @pytest.mark.asyncio
    async def test_complete_temperature_passes_through(self):
        fr = MagicMock()
        fr.generate = MagicMock(return_value=["x"])
        adapter = FastRouterAdapter(fr)

        await adapter.complete("hola", max_tokens=20, temperature=0.7)
        fr.generate.assert_called_once_with(["hola"], max_tokens=20, temperature=0.7)

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        fr = MagicMock()
        fr.generate = MagicMock(side_effect=RuntimeError("boom"))
        adapter = FastRouterAdapter(fr)

        with pytest.raises(RuntimeError, match="boom"):
            await adapter.complete("hola")


class TestHttpReasonerAdapter:
    @pytest.mark.asyncio
    async def test_complete_extracts_text(self):
        # HttpReasoner.__call__ retorna {choices:[{text:...}], usage:{...}}
        hr = MagicMock()
        hr.return_value = {
            "choices": [{"text": "respuesta-deep"}],
            "usage": {"completion_tokens": 5},
        }
        adapter = HttpReasonerAdapter(hr)

        result = await adapter.complete("explica X", max_tokens=128)

        assert result == "respuesta-deep"
        hr.assert_called_once_with("explica X", max_tokens=128, temperature=0.7)
```

- [ ] **Step 7.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_adapters.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.adapters'`

- [ ] **Step 7.3: Implementar `src/llm/adapters.py`**

```python
"""
Adapters sync→async para los clientes existentes (FastRouter, HttpReasoner).

LLMRouter espera `await client.complete(prompt, **kw) -> str`. Los clientes
actuales tienen interfaces propias (sync). Estos adapters traducen sin tocar
los clientes (mínima invasión).
"""

from __future__ import annotations

import asyncio
from typing import Any


class FastRouterAdapter:
    """Adapter para FastRouter (vLLM compartido :8100)."""

    def __init__(self, fast_router: Any):
        self._client = fast_router

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **_kw,
    ) -> str:
        results = await asyncio.to_thread(
            self._client.generate, [prompt], max_tokens=max_tokens, temperature=temperature
        )
        return results[0]


class HttpReasonerAdapter:
    """Adapter para HttpReasoner (llama-cpp 72B :8200)."""

    def __init__(self, http_reasoner: Any):
        self._client = http_reasoner

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **_kw,
    ) -> str:
        result = await asyncio.to_thread(
            self._client, prompt, max_tokens=max_tokens, temperature=temperature
        )
        return result["choices"][0]["text"]
```

- [ ] **Step 7.4: Run test — must PASS**

```bash
pytest tests/unit/llm/test_adapters.py -v
```
Expected: 4 passed

- [ ] **Step 7.5: Commit**

```bash
git add src/llm/adapters.py tests/unit/llm/test_adapters.py
git commit -m "feat(llm): sync→async adapters for FastRouter and HttpReasoner"
```

---

## Task 8: Router factory desde config

**Files:**
- Create: `src/llm/router_factory.py`
- Test: `tests/unit/llm/test_router_factory.py`

- [ ] **Step 8.1: Escribir test fallido**

```python
"""Tests for build_llm_router_from_config."""

import pytest
from unittest.mock import MagicMock

from src.llm.router import LLMRouter
from src.llm.router_factory import build_llm_router_from_config
from src.llm.types import EndpointKind


def _config(persistence_path):
    return {
        "llm": {
            "failover": {
                "cooldown_persistence_path": str(persistence_path),
                "endpoints": [
                    {
                        "id": "fast",
                        "kind": "fast_router",
                        "priority": 0,
                        "max_tokens_default": 128,
                    },
                    {
                        "id": "deep",
                        "kind": "http_reasoner",
                        "priority": 1,
                        "max_tokens_default": 512,
                    },
                ],
            }
        }
    }


class TestBuildRouterFromConfig:
    def test_builds_router_with_endpoints_in_priority_order(self, tmp_path):
        fast = MagicMock(name="FastRouter")
        deep = MagicMock(name="HttpReasoner")
        clients = {"fast": fast, "deep": deep}

        router = build_llm_router_from_config(
            _config(tmp_path / "cd.json"), clients=clients
        )

        assert isinstance(router, LLMRouter)
        # primer endpoint = fast (priority 0)
        assert router._endpoints[0].id == "fast"
        assert router._endpoints[0].kind == EndpointKind.FAST_ROUTER
        assert router._endpoints[1].id == "deep"
        assert router._endpoints[1].kind == EndpointKind.HTTP_REASONER

    def test_skips_endpoint_when_client_missing(self, tmp_path):
        # solo "fast" tiene cliente; "deep" no
        clients = {"fast": MagicMock()}

        router = build_llm_router_from_config(
            _config(tmp_path / "cd.json"), clients=clients
        )

        assert len(router._endpoints) == 1
        assert router._endpoints[0].id == "fast"

    def test_no_endpoints_raises(self, tmp_path):
        cfg = _config(tmp_path / "cd.json")
        cfg["llm"]["failover"]["endpoints"] = []
        with pytest.raises(ValueError, match="at least one"):
            build_llm_router_from_config(cfg, clients={})

    def test_unknown_kind_raises(self, tmp_path):
        cfg = _config(tmp_path / "cd.json")
        cfg["llm"]["failover"]["endpoints"][0]["kind"] = "xyz"
        with pytest.raises(ValueError, match="unknown.*kind"):
            build_llm_router_from_config(cfg, clients={"fast": MagicMock(), "deep": MagicMock()})
```

- [ ] **Step 8.2: Run test — must FAIL**

```bash
pytest tests/unit/llm/test_router_factory.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.llm.router_factory'`

- [ ] **Step 8.3: Implementar `src/llm/router_factory.py`**

```python
"""
Factory que construye un LLMRouter desde un dict de config.

Espera formato:
    config["llm"]["failover"] = {
        "cooldown_persistence_path": "./data/llm_cooldowns.json",
        "endpoints": [
            {"id": "fast", "kind": "fast_router", "priority": 0,
             "max_tokens_default": 128, "idle_timeout_s": null},
            {"id": "deep", "kind": "http_reasoner", "priority": 1,
             "max_tokens_default": 512, "idle_timeout_s": 30.0},
        ],
    }

`clients` es un dict {endpoint_id: ClienteAdapter} que el caller pasa.
Ya armados (FastRouterAdapter / HttpReasonerAdapter envolviendo los clientes
nativos). Separamos esto del factory para que main.py mantenga la
responsabilidad de cargar/conectar a los servicios reales.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint

logger = logging.getLogger(__name__)


def build_llm_router_from_config(
    config: dict,
    clients: dict[str, Any],
) -> LLMRouter:
    """Construir LLMRouter desde config + dict de clientes ya creados."""
    failover_cfg = config.get("llm", {}).get("failover", {})

    persistence_path = Path(
        failover_cfg.get("cooldown_persistence_path", "./data/llm_cooldowns.json")
    )
    cd_manager = CooldownManager(persistence_path=persistence_path)

    endpoints: list[LLMEndpoint] = []
    for ep_cfg in failover_cfg.get("endpoints", []):
        ep_id = ep_cfg["id"]
        client = clients.get(ep_id)
        if client is None:
            logger.warning(
                f"[RouterFactory] endpoint '{ep_id}' configured but no client provided — skipping"
            )
            continue

        kind_raw = ep_cfg["kind"]
        try:
            kind = EndpointKind(kind_raw)
        except ValueError as e:
            raise ValueError(f"unknown endpoint kind '{kind_raw}' for id '{ep_id}'") from e

        endpoints.append(LLMEndpoint(
            id=ep_id,
            kind=kind,
            client=client,
            priority=ep_cfg.get("priority", 0),
            timeout_s=ep_cfg.get("timeout_s", 30.0),
            idle_timeout_s=ep_cfg.get("idle_timeout_s"),
            max_tokens_default=ep_cfg.get("max_tokens_default", 256),
        ))

    if not endpoints:
        raise ValueError("at least one endpoint must be configured with a valid client")

    return LLMRouter(endpoints, cd_manager)
```

- [ ] **Step 8.4: Run test — must PASS**

```bash
pytest tests/unit/llm/test_router_factory.py -v
```
Expected: 4 passed

- [ ] **Step 8.5: Commit**

```bash
git add src/llm/router_factory.py tests/unit/llm/test_router_factory.py
git commit -m "feat(llm): build_llm_router_from_config factory"
```

---

## Task 9: Config schema en `settings.yaml`

**Files:**
- Modify: `config/settings.yaml`

- [ ] **Step 9.1: Leer la sección actual del reasoner**

```bash
sed -n '200,260p' config/settings.yaml
```

- [ ] **Step 9.2: Insertar bloque `llm.failover` después de la línea ~260 (final del bloque `reasoner:`)**

Ubicar exactamente DÓNDE: después del cierre del bloque `reasoner:` (donde termina la config CPU/RAM/RoPE), ANTES del próximo `# ===` separator. Ejecutar:

```bash
grep -n "^# ===\|^[a-z_]*:$" config/settings.yaml | head -30
```

para ubicar el siguiente top-level key después de `reasoner:`.

Agregar este bloque (path exacto al insertar — entre `reasoner:` y la próxima top-level section):

```yaml
# ===========================================================================
# LLM Failover — candidate chain con cooldown exponencial (1m/5m/25m/1h)
# Patrón inspirado en OpenClaw model-failover.md.
# Si vLLM 7B :8100 cae, el router rota al 72B :8200 sin perder el turno.
# ===========================================================================
llm:
  failover:
    # Persistencia del estado de cooldowns (sobrevive reinicios del proceso).
    cooldown_persistence_path: "./data/llm_cooldowns.json"

    # Endpoints en orden de prioridad (priority menor = se intenta primero).
    # `kind` debe coincidir con EndpointKind (fast_router | http_reasoner | local_reasoner | cloud).
    # `idle_timeout_s`: si es null, sin watchdog. Recomendado para 72B en CPU (lento).
    endpoints:
      - id: "fast"
        kind: "fast_router"
        priority: 0
        timeout_s: 30.0
        idle_timeout_s: null    # vLLM responde rápido — no necesita watchdog
        max_tokens_default: 128

      - id: "deep"
        kind: "http_reasoner"
        priority: 1
        timeout_s: 120.0
        idle_timeout_s: 30.0    # 72B en CPU: si no llegan tokens en 30s, abort
        max_tokens_default: 512

    # Reservado para futuro (cloud fallback, retry policy per-endpoint, etc.)
    # cloud:
    #   id: "anthropic"
    #   kind: "cloud"
    #   priority: 2
```

Hacer la edición usando Edit tool (no escribir todo el archivo). Buscar la línea exacta de cierre del bloque `reasoner:` y el separator siguiente.

- [ ] **Step 9.3: Verificar que el YAML sigue siendo válido**

```bash
python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))" && echo "YAML OK"
```
Expected: `YAML OK`

- [ ] **Step 9.4: Commit**

```bash
git add config/settings.yaml
git commit -m "config: add llm.failover block (cooldown + candidate chain)"
```

---

## Task 10: Wire en `main.py` y refactor `dispatcher.py`

**Files:**
- Modify: `src/main.py` (líneas 237-286)
- Modify: `src/orchestrator/dispatcher.py` (línea 565 area)
- Modify: `src/llm/__init__.py`

- [ ] **Step 10.1: Update `src/llm/__init__.py` para exportar lo nuevo**

Leer primero:

```bash
cat src/llm/__init__.py
```

Reemplazar contenido con:

```python
"""LLM module — reasoners, router, failover."""

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter
from src.llm.cooldown import CooldownManager
from src.llm.error_classifier import IdleTimeoutError, classify_error
from src.llm.idle_watchdog import idle_watchdog
from src.llm.reasoner import FastRouter, HttpReasoner, LLMReasoner
from src.llm.router import FailedAttempt, FallbackSummaryError, LLMRouter
from src.llm.router_factory import build_llm_router_from_config
from src.llm.types import (
    CooldownState,
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    RouterResult,
)

__all__ = [
    # Reasoners (legacy)
    "LLMReasoner",
    "HttpReasoner",
    "FastRouter",
    # Router + types
    "LLMRouter",
    "LLMEndpoint",
    "EndpointKind",
    "ErrorKind",
    "CooldownState",
    "RouterResult",
    "FallbackSummaryError",
    "FailedAttempt",
    # Manager
    "CooldownManager",
    # Adapters
    "FastRouterAdapter",
    "HttpReasonerAdapter",
    # Helpers
    "build_llm_router_from_config",
    "idle_watchdog",
    "IdleTimeoutError",
    "classify_error",
]
```

- [ ] **Step 10.2: Modificar `src/main.py` líneas 277-286 para construir el router**

Leer la sección actual:

```bash
sed -n '275,295p' src/main.py
```

Insertar después de la línea donde termina el bloque `if router_config.get("enabled", True):` (después del log "Fast router (HTTP) → ..."), reemplazando esa zona con:

```python
    # Fast Router (HTTP) — vLLM compartido :8100
    router_config = config.get("router", {})
    fast_router = None
    if router_config.get("enabled", True):
        fast_router = FastRouter(
            base_url=router_config.get("base_url", "http://127.0.0.1:8100/v1"),
            model=router_config.get("model", "qwen2.5-7b-awq"),
            timeout=router_config.get("timeout", 30),
        )
        logger.info(f"Fast router (HTTP) → {router_config.get('base_url', 'http://127.0.0.1:8100/v1')}")

    # LLM Router — candidate chain con cooldown/failover
    # Si llm.failover.endpoints está definido en config, construye un LLMRouter
    # que envuelve fast_router (vLLM 7B) y llm (HttpReasoner 72B). Si no, queda
    # None y el dispatcher cae al path legacy (fast_router directo).
    from src.llm import (
        FastRouterAdapter,
        HttpReasonerAdapter,
        build_llm_router_from_config,
    )

    llm_router = None
    if config.get("llm", {}).get("failover", {}).get("endpoints"):
        clients = {}
        if fast_router is not None:
            clients["fast"] = FastRouterAdapter(fast_router)
        if llm is not None:
            clients["deep"] = HttpReasonerAdapter(llm)
        try:
            llm_router = build_llm_router_from_config(config, clients)
            logger.info(f"LLM router activo con {len(llm_router._endpoints)} endpoints")
        except ValueError as e:
            logger.warning(f"No pude construir LLM router: {e}. Fallback a path legacy.")
            llm_router = None
```

Luego, donde se construye el `RequestDispatcher` (buscar en main.py), pasarle `llm_router=llm_router` (cuando exista) además del `fast_router` actual.

Buscar la línea:

```bash
grep -n "RequestDispatcher\|Dispatcher(" src/main.py | head -5
```

Y agregar el kwarg `llm_router=llm_router` al constructor.

- [ ] **Step 10.3: Modificar `src/orchestrator/dispatcher.py` para usar `llm_router` cuando esté disponible**

Leer constructor del dispatcher:

```bash
grep -n "def __init__" src/orchestrator/dispatcher.py | head -5
```

Agregar parámetro `llm_router: "LLMRouter | None" = None` al `__init__` y guardarlo como `self.llm_router`.

Modificar la línea ~565 (`response = self.router.generate([text], max_tokens=128)[0]`) para usar el router cuando esté:

```python
        if path == PathType.FAST_ROUTER and self.router:
            # Usar router para respuesta rapida
            t0 = time.perf_counter()
            try:
                if self.llm_router is not None:
                    # Path nuevo: candidate chain con failover
                    result = await self.llm_router.complete(text, max_tokens=128)
                    response = result.text
                    timings["router"] = (time.perf_counter() - t0) * 1000
                    timings["router_endpoint"] = result.endpoint_id
                    timings["router_attempts"] = result.attempts
                else:
                    # Path legacy: FastRouter directo
                    response = self.router.generate([text], max_tokens=128)[0]
                    timings["router"] = (time.perf_counter() - t0) * 1000

                return DispatchResult(
                    path=path,
                    priority=Priority.MEDIUM,
                    success=True,
                    response=response.strip(),
                    intent="simple_query",
                    timings=timings
                )
            except Exception as e:
                logger.warning(f"Router fallo, pasando a slow path: {e}")
```

- [ ] **Step 10.4: Run test suite completo (regression check)**

```bash
pytest tests/ -x --ignore=tests/integration 2>&1 | tail -20
```

Expected: todos los tests pasan; ningún test pre-existente roto.

Si algún test del dispatcher rompe por cambio de signature, agregar `llm_router=None` al fixture/constructor en `tests/unit/orchestrator/test_dispatcher.py`. Buscar:

```bash
grep -rn "RequestDispatcher\|Dispatcher(" tests/ | head -10
```

- [ ] **Step 10.5: Commit**

```bash
git add src/llm/__init__.py src/main.py src/orchestrator/dispatcher.py tests/
git commit -m "feat(llm): wire LLMRouter into main.py + dispatcher (legacy path preserved)"
```

---

## Task 11: Integration test end-to-end

**Files:**
- Create: `tests/integration/test_llm_failover_e2e.py`

- [ ] **Step 11.1: Verificar que existe `tests/integration/__init__.py`**

```bash
ls tests/integration/
```

Si no existe `__init__.py`:

```bash
touch tests/integration/__init__.py
```

- [ ] **Step 11.2: Escribir test e2e**

```python
"""
End-to-end integration test: 7B falla → router cae a 72B → 7B vuelve.

Simula el escenario real:
1. Comando "prendé la luz" llega → router intenta 7B
2. 7B retorna 429 (mock) → cooldown 60s, rota a 72B
3. 72B responde OK → success persistido
4. Segundo comando: 7B sigue en cooldown → directo a 72B (sin latencia extra)
5. Después de avanzar el reloj +61s, 7B vuelve disponible
6. Tercer comando: 7B intenta primero → success
"""

import pytest
from unittest.mock import MagicMock

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter
from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint


@pytest.fixture
def fake_clock(monkeypatch):
    now = [10000.0]
    monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])
    return now


@pytest.fixture
def fast_router_mock():
    fr = MagicMock(name="FastRouter")
    fr.generate = MagicMock(return_value=["respuesta-fast"])
    return fr


@pytest.fixture
def http_reasoner_mock():
    hr = MagicMock(name="HttpReasoner")
    hr.return_value = {
        "choices": [{"text": "respuesta-deep"}],
        "usage": {"completion_tokens": 5},
    }
    return hr


@pytest.fixture
def router(tmp_path, fast_router_mock, http_reasoner_mock):
    cd_manager = CooldownManager(persistence_path=tmp_path / "cd.json")

    endpoints = [
        LLMEndpoint(
            id="fast", kind=EndpointKind.FAST_ROUTER,
            client=FastRouterAdapter(fast_router_mock),
            priority=0,
        ),
        LLMEndpoint(
            id="deep", kind=EndpointKind.HTTP_REASONER,
            client=HttpReasonerAdapter(http_reasoner_mock),
            priority=1,
        ),
    ]
    return LLMRouter(endpoints, cd_manager), cd_manager


class TestLLMFailoverE2E:
    @pytest.mark.asyncio
    async def test_full_failover_scenario(
        self, router, fake_clock, fast_router_mock, http_reasoner_mock
    ):
        rt, cd = router

        # ====== Turn 1: 7B funciona ======
        result = await rt.complete("prendé la luz", max_tokens=64)
        assert result.endpoint_id == "fast"
        assert result.text == "respuesta-fast"
        assert fast_router_mock.generate.call_count == 1
        assert http_reasoner_mock.call_count == 0

        # ====== Turn 2: 7B falla con 429 ======
        fast_router_mock.generate.side_effect = RuntimeError("429 Too Many Requests")
        result = await rt.complete("apagá el aire", max_tokens=64)
        assert result.endpoint_id == "deep"
        assert result.text == "respuesta-deep"
        assert fast_router_mock.generate.call_count == 2  # se intentó
        assert http_reasoner_mock.call_count == 1
        # 7B en cooldown
        assert cd.is_available("fast") is False

        # ====== Turn 3: 7B sigue en cooldown — directo a 72B ======
        fake_clock[0] = 10030.0  # +30s, todavía dentro del cooldown de 60s
        result = await rt.complete("subí persianas", max_tokens=64)
        assert result.endpoint_id == "deep"
        # 7B no se intentó (cooldown skip)
        assert fast_router_mock.generate.call_count == 2  # sin incremento
        assert http_reasoner_mock.call_count == 2

        # ====== Turn 4: cooldown expiró, 7B disponible y arreglado ======
        fake_clock[0] = 10100.0  # +100s, fuera del cooldown
        fast_router_mock.generate.side_effect = None  # se "arregla"
        fast_router_mock.generate.return_value = ["fast-recovered"]
        result = await rt.complete("temperatura ambiente", max_tokens=64)
        assert result.endpoint_id == "fast"
        assert result.text == "fast-recovered"
        # Después del éxito, error_count se resetea
        assert cd.get_state("fast").error_count == 0

    @pytest.mark.asyncio
    async def test_both_endpoints_down_raises(
        self, router, fake_clock, fast_router_mock, http_reasoner_mock
    ):
        from src.llm.router import FallbackSummaryError

        rt, _ = router

        fast_router_mock.generate.side_effect = RuntimeError("429")
        http_reasoner_mock.side_effect = TimeoutError("read")

        with pytest.raises(FallbackSummaryError) as exc_info:
            await rt.complete("hola", max_tokens=64)

        assert len(exc_info.value.attempts) == 2
        assert {a.endpoint_id for a in exc_info.value.attempts} == {"fast", "deep"}
```

- [ ] **Step 11.3: Run test — must PASS**

```bash
pytest tests/integration/test_llm_failover_e2e.py -v
```
Expected: 2 passed

- [ ] **Step 11.4: Run TODO el suite de tests (regression final)**

```bash
pytest tests/ -x 2>&1 | tail -20
```
Expected: todos los tests pasan, ningún regression.

- [ ] **Step 11.5: Commit final**

```bash
git add tests/integration/test_llm_failover_e2e.py tests/integration/__init__.py
git commit -m "test(llm): e2e failover scenario (7B→72B with cooldown recovery)"
```

---

## Task 12: Smoke test manual del sistema

**Sin tests automatizados — verificación humana de que arranca el sistema sin romperse.**

- [ ] **Step 12.1: Iniciar el sistema con failover habilitado**

```bash
cd /Users/yo/Documents/kza
python -m src.main 2>&1 | grep -E "LLM router|failover|endpoint" | head -10
```

Expected: log line `LLM router activo con N endpoints`.

- [ ] **Step 12.2: Verificar que el archivo de cooldown se crea cuando hay un fallo**

Si no podés probar con un endpoint real caído, simular vía un endpoint mal configurado en `settings.yaml` (`http_base_url: "http://127.0.0.1:9999/v1"` para `deep`). Disparar un comando complejo (que requiera deep reasoning) y verificar:

```bash
ls -la data/llm_cooldowns.json
cat data/llm_cooldowns.json
```

Expected: archivo existe con `cooldown_until` > 0 para el endpoint caído.

- [ ] **Step 12.3: Restaurar config + commit final si hubo cambios de prueba**

Si modificaste `settings.yaml` para test, revertir:

```bash
git checkout config/settings.yaml
```

---

## Verificación final (checklist humano)

- [ ] Todos los tests unitarios pasan (`pytest tests/unit/llm/ -v`)
- [ ] El test e2e pasa (`pytest tests/integration/test_llm_failover_e2e.py -v`)
- [ ] El test suite completo pasa (`pytest tests/ -x`)
- [ ] El sistema arranca sin errores con la config nueva
- [ ] El archivo `./data/llm_cooldowns.json` se crea cuando un endpoint falla
- [ ] El path legacy (sin `llm.failover` en config) sigue funcionando — backwards compatible
- [ ] No se introdujeron dependencias nuevas (solo `pytest-asyncio` si no estaba)
- [ ] El log muestra `endpoint_id` y `attempts` cuando se rota

## Limitaciones conocidas (futuro)

1. **Idle watchdog no aplicado a HttpReasoner aún.** El `idle_watchdog` existe como módulo, pero `HttpReasoner` no lo usa internamente porque no expone API streaming en este plan. Aplicar en plan futuro cuando agreguemos streaming end-to-end (relacionado con item #5 del roadmap, transcript JSONL).
2. **Sin model-scoped cooldown.** OpenClaw distingue cooldowns por (provider, model). Aquí solo por endpoint_id. Suficiente para 7B/72B/cloud distintos. Si en el futuro varios modelos comparten el mismo endpoint físico, agregar dimensión.
3. **Sin SDK retry-after cap.** Los SDKs internos (openai-python) pueden dormir según `Retry-After`. Si llega a ser problema, agregar `OPENCLAW_SDK_RETRY_MAX_WAIT_SECONDS` equivalente.
4. **Sin session stickiness.** OpenClaw "pinna" un endpoint por sesión para mantener cache caliente. Aquí cada turno arranca por priority. Si hay impacto en cache hit rate de vLLM, agregar luego.

---

## Plan complete

Plan completo y guardado en `docs/superpowers/plans/2026-04-26-llm-failover-cooldown.md`. Dos opciones de ejecución:

1. **Subagent-Driven (recomendado)** — Dispatch fresh subagent per task, review entre tasks, fast iteration.
2. **Inline Execution** — Execute tasks en esta sesión usando `superpowers:executing-plans`, batch con checkpoints para review.

¿Cuál preferís?
