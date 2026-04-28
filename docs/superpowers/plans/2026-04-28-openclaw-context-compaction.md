# Context Auto-Compaction + Identifier Policy Strict — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cuando el `conversation_history` de un usuario alcance el umbral configurado, un `Compactor` async resume los turnos viejos vía Qwen3-30B-A3B (`kza-llm-ik` :8200) sin tocar identificadores HA opacos. Los contextos inactivos se persisten como snapshot a `data/contexts/<user_id>.json` y se hidratan en la próxima sesión. Backward-compatible: ambos features OFF por default.

**Architecture:** Dos componentes nuevos (`Compactor`, `ContextPersister`) inyectados al `ContextManager` existente. Trigger in-memory: turn-count proactivo (turno N=6) que dispara `asyncio.create_task` fire-and-forget. Trigger persistencia: snapshot al expirar inactividad (`cleanup_inactive` reescrito de thread→asyncio task). Identifier policy por construcción: `preserved_ids` se extraen de `turn.entities` antes de mandar al modelo y nunca pasan por el LLM. Consumo en NLU: `request_router.py` los pasa como hint a `VectorSearch` y al regex extractor (no inyectados al prompt LLM).

**Tech Stack:** Python 3.13, asyncio, dataclasses, pytest + fixtures existentes en `tests/conftest.py` y mocks en `tests/mocks/mock_llm.py`. Persistencia: JSON en `./data/contexts/<user_id>.json` con write atómico (`os.replace`). Sin dependencias nuevas. Spec: `docs/superpowers/specs/2026-04-28-openclaw-context-compaction-design.md`.

---

## File Structure

### Crear

| Path | Responsabilidad |
|------|-----------------|
| `src/orchestrator/compactor.py` | `Compactor`, `CompactionResult`, `CompactionError`. Llama al `HttpReasoner` y parsea JSON. |
| `src/orchestrator/context_persister.py` | `ContextPersister` con save/load/exists; write atómico a `data/contexts/<id>.json`. |
| `tests/unit/orchestrator/__init__.py` | Vacío. |
| `tests/unit/orchestrator/test_compactor.py` | Tests unit del Compactor: happy path, JSON malformado, timeout, preserved_entities passthrough. |
| `tests/unit/orchestrator/test_context_persister.py` | Save/load round-trip, atomic rename, file no existe, JSON corrupto. |
| `tests/unit/orchestrator/test_context_manager_compaction.py` | Hooks en `add_turn`, snapshot al expirar, hidratación, failure modes, backward-compat. |
| `tests/integration/test_context_compaction_e2e.py` | E2E: mock HttpReasoner → trigger → snapshot → hidratación. |

### Modificar

| Path | Cambio |
|------|--------|
| `src/orchestrator/__init__.py` | Exportar `Compactor`, `CompactionResult`, `CompactionError`, `ContextPersister`. |
| `src/orchestrator/context_manager.py` | Campos nuevos en `UserContext`. `ContextManager.__init__` acepta `compactor` + `persister` opcionales. Hook proactivo en `add_turn`. Refactor `cleanup_inactive` thread→asyncio task. Hidratación en `get_or_create`. |
| `src/main.py` | Construir `HttpReasoner` adicional para `kza-llm-ik` (apunta al mismo :8200), instanciar `Compactor` y `ContextPersister`, inyectar al `ContextManager`. |
| `src/pipeline/request_router.py` | Cuando construye el contexto del turno, extraer `ctx.preserved_ids` y pasarlo a `VectorSearch.search()` como `hint_entities`. |
| `src/vectordb/...` (módulo VectorSearch) | Aceptar parámetro opcional `hint_entities: list[str] | None` en `search()`. Boost ligero a entidades cuyo id matchee algún hint (no necesariamente cambia ranking si no aparece). |
| `config/settings.yaml` | Sub-bloque nuevo `orchestrator.context.compaction` y `orchestrator.context.persistence` (anidados bajo el bloque existente línea 631-638). |

---

## Conventions

- **Imports:** stdlib → third-party → `from src.modulo import Clase`. Nada relativo.
- **Logging:** `logger = logging.getLogger(__name__)` en cada módulo, prefijo descriptivo (`[Compactor]`, `[ContextPersister]`, `[ContextManager]`).
- **Tests:** clases `TestX` con métodos `test_y(self, fixtures)`. Fixtures vía conftest.
- **Async:** todo I/O de red usa `await`. File I/O del Persister es sync (rápido). Background tasks lanzadas con `asyncio.create_task`.
- **No bloquear event loop:** la compactación llama al 30B vía `await reasoner.complete(...)` que internamente usa `asyncio.to_thread`. El Persister es sync pero las llamadas son cortas; donde sea async-context, envolver en `asyncio.to_thread`.
- **Backward-compat:** sin Compactor inyectado, `ContextManager` se comporta como hoy (truncate duro, cleanup sin persistir).

---

## Task 1: Compactor — tipos + happy path

**Files:**
- Create: `src/orchestrator/compactor.py`
- Create: `tests/unit/orchestrator/__init__.py` (vacío)
- Create: `tests/unit/orchestrator/test_compactor.py`

- [ ] **Step 1.1: Crear directorio + __init__**

```bash
mkdir -p tests/unit/orchestrator
touch tests/unit/orchestrator/__init__.py
```

- [ ] **Step 1.2: Escribir test fallido `tests/unit/orchestrator/test_compactor.py`**

```python
"""Tests for context Compactor."""

import json
import pytest
from unittest.mock import AsyncMock

from src.orchestrator.compactor import (
    Compactor,
    CompactionResult,
    CompactionError,
)
from src.orchestrator.context_manager import ConversationTurn


def _turn(role: str, content: str, entities: list[str] | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, entities=entities or [])


class TestCompactorHappyPath:
    @pytest.mark.asyncio
    async def test_returns_summary_from_json(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(
            return_value='{"summary": "El usuario controló la luz del escritorio."}'
        )
        compactor = Compactor(reasoner=reasoner)

        turns = [
            _turn("user", "Prendé la luz del escritorio", entities=["light.escritorio"]),
            _turn("assistant", "Listo"),
        ]
        result = await compactor.compact(turns, preserved_entities=["light.escritorio"])

        assert isinstance(result, CompactionResult)
        assert result.summary == "El usuario controló la luz del escritorio."
        assert result.preserved_ids == ["light.escritorio"]
        assert result.compacted_turns_count == 2
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_dedupes_preserved_entities(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value='{"summary": "ok"}')
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")],
            preserved_entities=["light.a", "light.a", "scene.b"],
        )
        assert sorted(result.preserved_ids) == ["light.a", "scene.b"]

    @pytest.mark.asyncio
    async def test_passes_max_tokens_to_reasoner(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value='{"summary": "ok"}')
        compactor = Compactor(reasoner=reasoner, max_summary_tokens=128)

        await compactor.compact(turns=[_turn("user", "x")], preserved_entities=[])

        kwargs = reasoner.complete.await_args.kwargs
        assert kwargs.get("max_tokens") == 128
```

- [ ] **Step 1.3: Run test to verify it fails**

```bash
pytest tests/unit/orchestrator/test_compactor.py -v
```

Expected: ImportError or ModuleNotFoundError on `src.orchestrator.compactor`.

- [ ] **Step 1.4: Implementar `src/orchestrator/compactor.py` mínimo (happy path)**

```python
"""Context Compactor — turns into a summary using a background LLM call."""

import json
import logging
import time
from dataclasses import dataclass, field

from src.orchestrator.context_manager import ConversationTurn

logger = logging.getLogger(__name__)


COMPACTOR_SYSTEM_PROMPT = (
    "Sos un compactador de contexto conversacional. Recibís N turnos de "
    "diálogo entre un usuario y un asistente de hogar. Tu tarea: producir "
    "un resumen en 2-4 oraciones en español que capture (a) preferencias "
    "estables del usuario, (b) decisiones tomadas, (c) entidades del hogar "
    "referenciadas en lenguaje natural (NO uses identificadores técnicos). "
    "Usá tercera persona. NO menciones IDs tipo light.X, scene.Y, area.Z — "
    "esos se preservan aparte.\n\n"
    'Devolvé JSON: {"summary": "..."}'
)


class CompactionError(Exception):
    """Raised when compaction cannot produce a usable summary."""


@dataclass
class CompactionResult:
    summary: str
    preserved_ids: list[str]
    compacted_turns_count: int
    model: str
    latency_ms: float


class Compactor:
    """Compacts a list of conversation turns into a short summary.

    The LLM call is async and may take seconds; callers should run this in a
    background task. Identifier policy strict: HA entity_ids are NEVER passed
    to the model — they are extracted from `preserved_entities` and surfaced
    verbatim in the result.
    """

    def __init__(
        self,
        reasoner,  # HttpReasoner-like with async complete(prompt, max_tokens, temperature)
        max_summary_tokens: int = 200,
        timeout_s: float = 30.0,
    ):
        self.reasoner = reasoner
        self.max_summary_tokens = max_summary_tokens
        self.timeout_s = timeout_s

    async def compact(
        self,
        turns: list[ConversationTurn],
        preserved_entities: list[str],
    ) -> CompactionResult:
        if not turns:
            raise CompactionError("No turns to compact")

        prompt = self._build_prompt(turns)
        start = time.perf_counter()
        text = await self.reasoner.complete(
            prompt=prompt,
            max_tokens=self.max_summary_tokens,
            temperature=0.3,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        summary = self._parse_summary(text)
        preserved_ids = sorted(set(preserved_entities))
        model = getattr(self.reasoner, "_resolved_model", None) or "unknown"

        logger.info(
            f"[Compactor] turns={len(turns)} summary_chars={len(summary)} "
            f"preserved_ids={len(preserved_ids)} latency={latency_ms:.0f}ms"
        )

        return CompactionResult(
            summary=summary,
            preserved_ids=preserved_ids,
            compacted_turns_count=len(turns),
            model=model,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, turns: list[ConversationTurn]) -> str:
        lines = [COMPACTOR_SYSTEM_PROMPT, "", "Turnos a compactar:"]
        for i, turn in enumerate(turns, start=1):
            lines.append(f"{i}. [{turn.role}] {turn.content}")
        lines.append("")
        lines.append("Resumen JSON:")
        return "\n".join(lines)

    def _parse_summary(self, text: str) -> str:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("summary"), str):
                return data["summary"].strip()
        except json.JSONDecodeError:
            pass
        # Fallback: try to extract first JSON object substring
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if isinstance(data, dict) and isinstance(data.get("summary"), str):
                return data["summary"].strip()
        except (ValueError, json.JSONDecodeError):
            pass
        # Final fallback: return text literal trimmed
        logger.warning(f"[Compactor] JSON parse failed, using literal text ({len(text)} chars)")
        return text
```

- [ ] **Step 1.5: Run test to verify happy path passes**

```bash
pytest tests/unit/orchestrator/test_compactor.py::TestCompactorHappyPath -v
```

Expected: 3 passed.

- [ ] **Step 1.6: Commit**

```bash
git add src/orchestrator/compactor.py tests/unit/orchestrator/__init__.py tests/unit/orchestrator/test_compactor.py
git commit -m "feat(orchestrator): Compactor happy path (plan #2 OpenClaw)"
```

---

## Task 2: Compactor — JSON malformed fallback + timeout

**Files:**
- Modify: `tests/unit/orchestrator/test_compactor.py` (append cases)
- Modify: `src/orchestrator/compactor.py` (add timeout wrap)

- [ ] **Step 2.1: Append failing cases a `test_compactor.py`**

```python
class TestCompactorErrorPaths:
    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_text(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(return_value="No JSON here, just text.")
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")], preserved_entities=[]
        )
        assert "No JSON here" in result.summary

    @pytest.mark.asyncio
    async def test_extra_text_around_json_recovered(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(
            return_value='Pensemos... {"summary": "Hola"} fin.'
        )
        compactor = Compactor(reasoner=reasoner)

        result = await compactor.compact(
            turns=[_turn("user", "x")], preserved_entities=[]
        )
        assert result.summary == "Hola"

    @pytest.mark.asyncio
    async def test_empty_turns_raises(self):
        reasoner = AsyncMock()
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError):
            await compactor.compact(turns=[], preserved_entities=[])

    @pytest.mark.asyncio
    async def test_timeout_wraps_into_compaction_error(self):
        import asyncio

        async def slow(*_, **__):
            await asyncio.sleep(10)
            return '{"summary": "no llega"}'

        reasoner = AsyncMock()
        reasoner.complete = slow
        compactor = Compactor(reasoner=reasoner, timeout_s=0.05)

        with pytest.raises(CompactionError) as exc_info:
            await compactor.compact(
                turns=[_turn("user", "x")], preserved_entities=[]
            )
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reasoner_exception_wraps_into_compaction_error(self):
        reasoner = AsyncMock()
        reasoner.complete = AsyncMock(side_effect=ConnectionError("boom"))
        compactor = Compactor(reasoner=reasoner)

        with pytest.raises(CompactionError):
            await compactor.compact(
                turns=[_turn("user", "x")], preserved_entities=[]
            )
```

- [ ] **Step 2.2: Run failing tests**

```bash
pytest tests/unit/orchestrator/test_compactor.py::TestCompactorErrorPaths -v
```

Expected: failures on `test_timeout_wraps_into_compaction_error` and `test_reasoner_exception_wraps_into_compaction_error` (timeout + bare exceptions not wrapped). The other 3 should already pass.

- [ ] **Step 2.3: Add timeout + exception wrapping in `compactor.py`**

Replace `compact()` in `src/orchestrator/compactor.py`:

```python
    async def compact(
        self,
        turns: list[ConversationTurn],
        preserved_entities: list[str],
    ) -> CompactionResult:
        import asyncio

        if not turns:
            raise CompactionError("No turns to compact")

        prompt = self._build_prompt(turns)
        start = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                self.reasoner.complete(
                    prompt=prompt,
                    max_tokens=self.max_summary_tokens,
                    temperature=0.3,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise CompactionError(f"Compactor timeout after {self.timeout_s}s") from e
        except Exception as e:
            raise CompactionError(f"Compactor reasoner error: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        summary = self._parse_summary(text)
        preserved_ids = sorted(set(preserved_entities))
        model = getattr(self.reasoner, "_resolved_model", None) or "unknown"

        logger.info(
            f"[Compactor] turns={len(turns)} summary_chars={len(summary)} "
            f"preserved_ids={len(preserved_ids)} latency={latency_ms:.0f}ms"
        )

        return CompactionResult(
            summary=summary,
            preserved_ids=preserved_ids,
            compacted_turns_count=len(turns),
            model=model,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 2.4: Run all Compactor tests**

```bash
pytest tests/unit/orchestrator/test_compactor.py -v
```

Expected: all pass (8+ tests).

- [ ] **Step 2.5: Commit**

```bash
git add tests/unit/orchestrator/test_compactor.py src/orchestrator/compactor.py
git commit -m "feat(orchestrator): Compactor timeout + JSON fallback (plan #2 OpenClaw)"
```

---

## Task 3: ContextPersister — save/load atomic

**Files:**
- Create: `src/orchestrator/context_persister.py`
- Create: `tests/unit/orchestrator/test_context_persister.py`

- [ ] **Step 3.1: Escribir test fallido `tests/unit/orchestrator/test_context_persister.py`**

```python
"""Tests for ContextPersister — atomic JSON snapshot per user."""

import json
from pathlib import Path

import pytest

from src.orchestrator.context_persister import ContextPersister, PERSISTED_VERSION
from src.orchestrator.context_manager import UserContext


@pytest.fixture
def tmp_persister(tmp_path: Path) -> ContextPersister:
    return ContextPersister(base_path=tmp_path / "contexts")


def _ctx(user_id: str = "u1", **overrides) -> UserContext:
    base = dict(
        user_id=user_id,
        user_name="Juan",
        compacted_summary="Resumen previo.",
        preserved_ids=["light.escritorio"],
        session_count=2,
    )
    base.update(overrides)
    return UserContext(**base)


class TestContextPersister:
    def test_save_creates_directory_and_file(self, tmp_path: Path):
        persister = ContextPersister(base_path=tmp_path / "contexts")
        persister.save(_ctx())
        assert (tmp_path / "contexts" / "u1.json").exists()

    def test_save_load_roundtrip(self, tmp_persister):
        ctx = _ctx(user_id="alice", compacted_summary="Hola.", preserved_ids=["a.b"])
        tmp_persister.save(ctx)

        data = tmp_persister.load("alice")
        assert data["user_id"] == "alice"
        assert data["compacted_summary"] == "Hola."
        assert data["preserved_ids"] == ["a.b"]
        assert data["version"] == PERSISTED_VERSION
        assert "last_seen" in data

    def test_exists(self, tmp_persister):
        assert tmp_persister.exists("noone") is False
        tmp_persister.save(_ctx(user_id="bob"))
        assert tmp_persister.exists("bob") is True

    def test_load_missing_returns_none(self, tmp_persister):
        assert tmp_persister.load("ghost") is None

    def test_load_corrupt_returns_none_and_logs(self, tmp_persister, caplog):
        path = tmp_persister.base_path / "broken.json"
        tmp_persister.base_path.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json")

        result = tmp_persister.load("broken")
        assert result is None
        assert any("corrupt" in rec.message.lower() or "json" in rec.message.lower()
                   for rec in caplog.records)

    def test_save_atomic_no_partial_file(self, tmp_persister, monkeypatch):
        ctx = _ctx(user_id="atomic", compacted_summary="v1")
        tmp_persister.save(ctx)
        original_path = tmp_persister.base_path / "atomic.json"
        original = original_path.read_text()

        # Provocar fallo durante write
        import os
        real_replace = os.replace

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)

        ctx2 = _ctx(user_id="atomic", compacted_summary="v2_FAILED")
        with pytest.raises(OSError):
            tmp_persister.save(ctx2)

        # El archivo original no debe haber sido pisado
        assert original_path.read_text() == original
        # Y no debe quedar .tmp huérfano viable como contexto
        assert not any(
            p.name == "atomic.json"
            for p in tmp_persister.base_path.iterdir()
            if p.read_text() != original
        )

    def test_user_id_with_path_separator_rejected(self, tmp_persister):
        ctx = _ctx(user_id="../etc/passwd")
        with pytest.raises(ValueError):
            tmp_persister.save(ctx)
```

- [ ] **Step 3.2: Run failing tests**

```bash
pytest tests/unit/orchestrator/test_context_persister.py -v
```

Expected: ImportError on `context_persister`.

- [ ] **Step 3.3: Implementar `src/orchestrator/context_persister.py`**

```python
"""ContextPersister — atomic JSON snapshot of UserContext per user_id."""

import json
import logging
import os
import re
import time
from pathlib import Path

from src.orchestrator.context_manager import UserContext

logger = logging.getLogger(__name__)

PERSISTED_VERSION = 1
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class ContextPersister:
    """Saves and loads UserContext snapshots to disk.

    Format: one JSON file per user at base_path/<user_id>.json. Writes are
    atomic via .tmp + os.replace. Reads return None for missing/corrupt
    files (logged as warning) — callers treat that as "no prior context".
    """

    def __init__(self, base_path: Path | str = Path("data/contexts")):
        self.base_path = Path(base_path)

    def _validate_user_id(self, user_id: str) -> None:
        if not _SAFE_USER_ID.match(user_id):
            raise ValueError(
                f"Unsafe user_id for filesystem path: {user_id!r}. "
                "Allowed: alphanumeric, underscore, hyphen."
            )

    def _path(self, user_id: str) -> Path:
        return self.base_path / f"{user_id}.json"

    def exists(self, user_id: str) -> bool:
        try:
            self._validate_user_id(user_id)
        except ValueError:
            return False
        return self._path(user_id).is_file()

    def save(self, ctx: UserContext) -> None:
        self._validate_user_id(ctx.user_id)
        self.base_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": PERSISTED_VERSION,
            "user_id": ctx.user_id,
            "user_name": ctx.user_name,
            "last_seen": time.time(),
            "session_count": ctx.session_count,
            "compacted_summary": ctx.compacted_summary,
            "preserved_ids": list(ctx.preserved_ids),
        }

        target = self._path(ctx.user_id)
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            os.replace(tmp, target)
            logger.info(
                f"[ContextPersister] saved user={ctx.user_id} "
                f"summary_chars={len(ctx.compacted_summary or '')} "
                f"preserved_ids={len(ctx.preserved_ids)}"
            )
        except Exception:
            # cleanup partial tmp; do NOT touch target
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise

    def load(self, user_id: str) -> dict | None:
        try:
            self._validate_user_id(user_id)
        except ValueError as e:
            logger.warning(f"[ContextPersister] invalid user_id on load: {e}")
            return None

        path = self._path(user_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ContextPersister] corrupt or unreadable JSON for {user_id}: {e}")
            return None
```

> ⚠️ Nota para el implementador: `ContextPersister` importa `UserContext` desde `context_manager`, pero `UserContext` se modificará en Task 4 con campos nuevos (`compacted_summary`, `preserved_ids`, `session_count`). Para que este task pase, hay que **agregar primero los 4 campos a `UserContext`** antes de implementar este módulo. Hacerlo como Step 3.4 de este task (preview de Task 4).

- [ ] **Step 3.4: Agregar campos nuevos a `UserContext` (preview de Task 4)**

Modificar `src/orchestrator/context_manager.py`. Localizar el `@dataclass class UserContext` (~línea 80) y agregar AL FINAL de los campos (antes de cualquier método):

```python
    # Compaction state (plan #2 OpenClaw)
    compacted_summary: str | None = None
    preserved_ids: list[str] = field(default_factory=list)
    compaction_inflight: bool = False  # transient, no se serializa
    session_count: int = 1
```

- [ ] **Step 3.5: Run persister tests**

```bash
pytest tests/unit/orchestrator/test_context_persister.py -v
```

Expected: 7 passed.

- [ ] **Step 3.6: Commit**

```bash
git add src/orchestrator/context_persister.py src/orchestrator/context_manager.py tests/unit/orchestrator/test_context_persister.py
git commit -m "feat(orchestrator): ContextPersister atomic snapshot + UserContext fields (plan #2 OpenClaw)"
```

---

## Task 4: ContextManager — compaction trigger en add_turn

**Files:**
- Modify: `src/orchestrator/context_manager.py`
- Create: `tests/unit/orchestrator/test_context_manager_compaction.py`

> Pre-condición: Task 3 ya añadió los 4 campos nuevos a `UserContext`.

- [ ] **Step 4.1: Escribir test fallido `tests/unit/orchestrator/test_context_manager_compaction.py`**

```python
"""Tests for ContextManager compaction integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.context_manager import ContextManager, ConversationTurn
from src.orchestrator.compactor import CompactionResult, CompactionError


def _result(summary: str = "ok", ids: list[str] | None = None, count: int = 3) -> CompactionResult:
    return CompactionResult(
        summary=summary,
        preserved_ids=ids or [],
        compacted_turns_count=count,
        model="test-30b",
        latency_ms=10.0,
    )


@pytest.fixture
def manager_with_compactor():
    compactor = AsyncMock()
    compactor.compact = AsyncMock(return_value=_result(summary="compacted!", ids=["light.x"]))
    mgr = ContextManager(
        max_history=20,  # más que threshold para probar trigger sin trunc
        compactor=compactor,
        compaction_threshold=6,
        keep_recent_turns=3,
    )
    return mgr, compactor


class TestCompactionTrigger:
    @pytest.mark.asyncio
    async def test_trigger_fires_at_threshold(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        mgr.get_or_create("u1", "Juan")

        # Llenar hasta el turno 6 (threshold)
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}", entities=[f"light.{i}"])

        # Esperar que la task background termine
        await asyncio.sleep(0.05)

        compactor.compact.assert_awaited_once()
        ctx = mgr.get("u1")
        # Tras compactación: summary set + history reducido a keep_recent
        assert ctx.compacted_summary == "compacted!"
        assert len(ctx.conversation_history) == 3
        assert "light.x" in ctx.preserved_ids

    @pytest.mark.asyncio
    async def test_trigger_only_once_while_inflight(self, manager_with_compactor):
        """Si la compactación está corriendo, turnos extra no disparan otra."""
        mgr, compactor = manager_with_compactor

        # Hacer la compactación lenta
        gate = asyncio.Event()
        async def slow(*a, **kw):
            await gate.wait()
            return _result(summary="slow")
        compactor.compact = slow

        mgr.get_or_create("u1", "Juan")
        for i in range(7):  # 7 turnos: trigger en 6, turno 7 no re-dispara
            mgr.add_turn("u1", "user", f"msg {i}")

        # Soltar el gate y esperar
        gate.set()
        await asyncio.sleep(0.05)
        # exact-once via inflight flag (compactor.compact ya no es AsyncMock; chequear estado)
        ctx = mgr.get("u1")
        assert ctx.compaction_inflight is False

    @pytest.mark.asyncio
    async def test_trigger_failure_leaves_history_intact(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor
        compactor.compact = AsyncMock(side_effect=CompactionError("boom"))

        mgr.get_or_create("u1", "Juan")
        for i in range(6):
            mgr.add_turn("u1", "user", f"msg {i}")

        await asyncio.sleep(0.05)

        ctx = mgr.get("u1")
        assert ctx.compacted_summary is None
        assert len(ctx.conversation_history) == 6
        assert ctx.compaction_inflight is False

    @pytest.mark.asyncio
    async def test_no_compactor_no_trigger(self):
        """Sin compactor inyectado: comportamiento baseline (truncate duro)."""
        mgr = ContextManager(max_history=4, compactor=None)
        mgr.get_or_create("u1", "Juan")
        for i in range(10):
            mgr.add_turn("u1", "user", f"msg {i}")

        ctx = mgr.get("u1")
        assert ctx.compacted_summary is None
        assert len(ctx.conversation_history) == 4  # truncate baseline

    @pytest.mark.asyncio
    async def test_concatenates_summary_on_second_compaction(self, manager_with_compactor):
        mgr, compactor = manager_with_compactor

        compactor.compact = AsyncMock(side_effect=[
            _result(summary="A.", ids=["light.a"]),
            _result(summary="B.", ids=["light.b"]),
        ])

        mgr.get_or_create("u1", "Juan")
        # Primera ronda: 6 turnos → trigger
        for i in range(6):
            mgr.add_turn("u1", "user", f"r1-{i}")
        await asyncio.sleep(0.05)
        # Segunda ronda: agregar otros 3 + 3 más = 9 total visible (6 nuevos)
        # Tras primera compaction quedaron 3 en history. Para llegar a threshold de
        # nuevo, el manager debe re-disparar al alcanzar 6 nuevos.
        for i in range(6):
            mgr.add_turn("u1", "user", f"r2-{i}")
        await asyncio.sleep(0.05)

        ctx = mgr.get("u1")
        assert "A." in ctx.compacted_summary
        assert "B." in ctx.compacted_summary
        assert sorted(ctx.preserved_ids) == ["light.a", "light.b"]
```

- [ ] **Step 4.2: Run failing tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestCompactionTrigger -v
```

Expected: failures — `ContextManager` doesn't accept `compactor` kwarg yet.

- [ ] **Step 4.3: Modificar `ContextManager.__init__` y `add_turn`**

En `src/orchestrator/context_manager.py`, modificar `__init__` para aceptar nuevos kwargs (mantener firma existente compatible):

```python
    def __init__(
        self,
        max_history: int = 10,
        inactive_timeout: float = 300,
        cleanup_interval: float = 60,
        system_prompt: str = None,
        compactor=None,  # Compactor | None — plan #2 OpenClaw
        persister=None,  # ContextPersister | None — plan #2 OpenClaw
        compaction_threshold: int = 6,
        keep_recent_turns: int = 3,
    ):
        self.max_history = max_history
        self.inactive_timeout = inactive_timeout
        self.cleanup_interval = cleanup_interval
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Plan #2 OpenClaw — compaction + persistence
        self.compactor = compactor
        self.persister = persister
        self.compaction_threshold = compaction_threshold
        self.keep_recent_turns = keep_recent_turns

        self._contexts: dict[str, UserContext] = {}
        self._lock = threading.RLock()

        self._cleanup_running = False
        self._cleanup_thread: threading.Thread | None = None

        self._total_contexts_created = 0
        self._total_contexts_cleaned = 0
```

Modificar `add_turn` para disparar compactación al alcanzar el umbral:

```python
    def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        intent: str = None,
        entities: list = None,
    ):
        """Agregar turno al historial. Si hay compactor y se alcanza el
        umbral, lanza compactación en background fire-and-forget."""
        should_compact = False
        with self._lock:
            ctx = self._contexts.get(user_id)
            if not ctx:
                logger.warning(f"Contexto no encontrado: {user_id}")
                return
            ctx.add_turn(role, content, intent, entities)

            if (
                self.compactor is not None
                and not ctx.compaction_inflight
                and len(ctx.conversation_history) >= self.compaction_threshold
            ):
                ctx.compaction_inflight = True
                should_compact = True

        if should_compact:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._compact_background(user_id))
            except RuntimeError:
                # No event loop (uso desde código sync) — schedule via threadsafe is overkill;
                # mejor desactivar trigger silenciosamente
                logger.warning(
                    f"[ContextManager] no event loop running, skipping compaction for {user_id}"
                )
                with self._lock:
                    ctx = self._contexts.get(user_id)
                    if ctx:
                        ctx.compaction_inflight = False
```

Agregar el método `_compact_background`:

```python
    async def _compact_background(self, user_id: str) -> None:
        """Compactación fire-and-forget: copia turnos viejos, llama Compactor,
        muta el contexto bajo lock. Errores se loguean pero no propagan."""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return
            n_compact = max(0, len(ctx.conversation_history) - self.keep_recent_turns)
            if n_compact <= 0:
                ctx.compaction_inflight = False
                return
            turns_snapshot = list(ctx.conversation_history[:n_compact])
            preserved = []
            for t in turns_snapshot:
                preserved.extend(t.entities or [])
            preserved.extend(ctx.preserved_ids)

        try:
            result = await self.compactor.compact(turns_snapshot, preserved_entities=preserved)
        except Exception as e:
            logger.warning(f"[ContextManager] compaction failed for {user_id}: {e}")
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx:
                    ctx.compaction_inflight = False
            return

        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return
            existing = (ctx.compacted_summary + " ") if ctx.compacted_summary else ""
            ctx.compacted_summary = (existing + result.summary).strip()
            ctx.preserved_ids = sorted(set(ctx.preserved_ids) | set(result.preserved_ids))
            # Drop the compacted prefix; keep tail
            ctx.conversation_history = ctx.conversation_history[n_compact:]
            ctx.compaction_inflight = False
            logger.info(
                f"[ContextManager] user={user_id} compacted_turns={n_compact} "
                f"summary_chars={len(ctx.compacted_summary)} preserved={len(ctx.preserved_ids)}"
            )
```

Agregar import al tope del archivo si no existe:

```python
import asyncio
```

- [ ] **Step 4.4: Run all compaction-trigger tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestCompactionTrigger -v
```

Expected: 5 passed.

- [ ] **Step 4.5: Verificar que tests existentes de ContextManager siguen pasando**

```bash
pytest tests/unit/orchestrator/ -v
```

Expected: all green (no regression).

- [ ] **Step 4.6: Commit**

```bash
git add src/orchestrator/context_manager.py tests/unit/orchestrator/test_context_manager_compaction.py
git commit -m "feat(orchestrator): ContextManager compaction trigger in add_turn (plan #2 OpenClaw)"
```

---

## Task 5: ContextManager — refactor cleanup_inactive thread→asyncio + snapshot

**Files:**
- Modify: `src/orchestrator/context_manager.py`
- Modify: `tests/unit/orchestrator/test_context_manager_compaction.py` (append)

- [ ] **Step 5.1: Append failing tests para snapshot al expirar**

```python
class TestCleanupSnapshot:
    @pytest.mark.asyncio
    async def test_cleanup_persists_expired_context(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister

        persister = ContextPersister(base_path=tmp_path / "contexts")
        compactor = AsyncMock()
        compactor.compact = AsyncMock(return_value=_result(summary="snap"))

        mgr = ContextManager(
            inactive_timeout=0.01,  # casi instantáneo
            cleanup_interval=0.05,
            compactor=compactor,
            persister=persister,
        )
        mgr.get_or_create("snap_user", "Ana")
        mgr.add_turn("snap_user", "user", "hola", entities=["light.a"])

        # Iniciar el cleanup loop como asyncio task
        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())

        # Esperar más que inactive_timeout + cleanup_interval
        await asyncio.sleep(0.2)

        mgr.stop_cleanup_loop_async()
        await cleanup_task

        # Verificar persistencia
        assert persister.exists("snap_user")
        data = persister.load("snap_user")
        assert "snap" in (data["compacted_summary"] or "")

        # Y el contexto fue removido de memoria
        assert mgr.get("snap_user") is None

    @pytest.mark.asyncio
    async def test_cleanup_skips_active_contexts(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister

        persister = ContextPersister(base_path=tmp_path / "contexts")
        mgr = ContextManager(
            inactive_timeout=10.0,  # nadie expira en este test
            cleanup_interval=0.02,
            persister=persister,
        )
        mgr.get_or_create("active_user", "Bob")

        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
        await asyncio.sleep(0.1)
        mgr.stop_cleanup_loop_async()
        await cleanup_task

        assert mgr.get("active_user") is not None
        assert not persister.exists("active_user")

    @pytest.mark.asyncio
    async def test_cleanup_without_persister_just_deletes(self):
        mgr = ContextManager(
            inactive_timeout=0.01,
            cleanup_interval=0.02,
            persister=None,
        )
        mgr.get_or_create("ghost", "x")

        cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
        await asyncio.sleep(0.1)
        mgr.stop_cleanup_loop_async()
        await cleanup_task

        assert mgr.get("ghost") is None  # baseline cleanup behavior
```

- [ ] **Step 5.2: Run failing tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestCleanupSnapshot -v
```

Expected: AttributeError — `start_cleanup_loop_async` doesn't exist.

- [ ] **Step 5.3: Agregar el cleanup loop async + snapshot**

En `src/orchestrator/context_manager.py`, agregar métodos nuevos (no remover los thread-based aún; coexisten):

```python
    async def start_cleanup_loop_async(self) -> None:
        """Loop async de cleanup. Reemplaza al thread daemon cuando hay event loop.

        Llamar desde main.py: asyncio.create_task(mgr.start_cleanup_loop_async()).
        Detener con stop_cleanup_loop_async().
        """
        self._cleanup_running = True
        logger.info("[ContextManager] async cleanup loop started")
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive_async()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ContextManager] cleanup loop error: {e}")
        logger.info("[ContextManager] async cleanup loop stopped")

    def stop_cleanup_loop_async(self) -> None:
        self._cleanup_running = False

    async def _cleanup_inactive_async(self) -> int:
        """Async equivalente de cleanup_inactive: si hay persister, snapshot
        antes de eliminar el contexto."""
        now = time.time()
        with self._lock:
            inactive_ids = [
                uid for uid, ctx in self._contexts.items()
                if (now - ctx.last_active) > self.inactive_timeout
            ]

        cleaned = 0
        for uid in inactive_ids:
            await self._snapshot_and_remove(uid)
            cleaned += 1

        if cleaned > 0:
            logger.info(f"[ContextManager] async cleanup removed {cleaned} contexts")
        return cleaned

    async def _snapshot_and_remove(self, user_id: str) -> None:
        """Si hay persister: compactar pendiente + persistir. Luego remover."""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return

        # Compactar resto si hay turnos pendientes y compactor disponible
        if self.compactor is not None and self.persister is not None:
            with self._lock:
                pending = list(ctx.conversation_history) if ctx else []
                preserved_seed = list(ctx.preserved_ids) if ctx else []
            if pending:
                try:
                    extra = []
                    for t in pending:
                        extra.extend(t.entities or [])
                    result = await self.compactor.compact(
                        pending, preserved_entities=preserved_seed + extra
                    )
                    with self._lock:
                        ctx = self._contexts.get(user_id)
                        if ctx:
                            existing = (ctx.compacted_summary + " ") if ctx.compacted_summary else ""
                            ctx.compacted_summary = (existing + result.summary).strip()
                            ctx.preserved_ids = sorted(
                                set(ctx.preserved_ids) | set(result.preserved_ids)
                            )
                            ctx.conversation_history = []
                except Exception as e:
                    logger.warning(
                        f"[ContextManager] final compaction failed for {user_id}: {e}"
                    )

        # Persistir si hay persister
        if self.persister is not None:
            with self._lock:
                ctx = self._contexts.get(user_id)
            if ctx is not None and (ctx.compacted_summary or ctx.conversation_history):
                try:
                    await asyncio.to_thread(self.persister.save, ctx)
                except Exception as e:
                    logger.warning(f"[ContextManager] snapshot save failed for {user_id}: {e}")

        # Eliminar de memoria
        with self._lock:
            if user_id in self._contexts:
                del self._contexts[user_id]
                self._total_contexts_cleaned += 1
```

- [ ] **Step 5.4: Run snapshot tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestCleanupSnapshot -v
```

Expected: 3 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/orchestrator/context_manager.py tests/unit/orchestrator/test_context_manager_compaction.py
git commit -m "feat(orchestrator): async cleanup loop + snapshot persistence (plan #2 OpenClaw)"
```

---

## Task 6: ContextManager — hidratación en get_or_create

**Files:**
- Modify: `src/orchestrator/context_manager.py`
- Modify: `tests/unit/orchestrator/test_context_manager_compaction.py` (append)

- [ ] **Step 6.1: Append failing tests**

```python
class TestHydration:
    def test_hydrates_summary_and_preserved_ids(self, tmp_path):
        from src.orchestrator.context_persister import ContextPersister
        from src.orchestrator.context_manager import UserContext

        persister = ContextPersister(base_path=tmp_path / "contexts")
        prior = UserContext(
            user_id="returning",
            user_name="Carla",
            compacted_summary="Resumen viejo.",
            preserved_ids=["light.cocina"],
            session_count=3,
        )
        persister.save(prior)

        mgr = ContextManager(persister=persister)
        ctx = mgr.get_or_create("returning", "Carla")

        assert ctx.compacted_summary == "Resumen viejo."
        assert ctx.preserved_ids == ["light.cocina"]
        assert ctx.session_count == 4  # incrementa
        assert ctx.conversation_history == []  # turnos no se restauran

    def test_no_persister_no_hydration(self, tmp_path):
        # Aunque exista archivo en disk, sin persister no hidrata
        mgr = ContextManager(persister=None)
        ctx = mgr.get_or_create("anyone", "x")
        assert ctx.compacted_summary is None
        assert ctx.session_count == 1

    def test_corrupt_file_creates_fresh_context(self, tmp_path, caplog):
        from src.orchestrator.context_persister import ContextPersister
        base = tmp_path / "contexts"
        base.mkdir(parents=True, exist_ok=True)
        (base / "broken.json").write_text("{ not json")
        persister = ContextPersister(base_path=base)

        mgr = ContextManager(persister=persister)
        ctx = mgr.get_or_create("broken", "x")

        assert ctx.compacted_summary is None
        assert ctx.session_count == 1
```

- [ ] **Step 6.2: Run failing tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestHydration -v
```

Expected: failures — hydration logic not yet present.

- [ ] **Step 6.3: Modificar `get_or_create` para hidratar**

En `src/orchestrator/context_manager.py`, dentro de `get_or_create`, agregar el path de hidratación entre el chequeo "ya existe" y "crear nuevo":

```python
    def get_or_create(
        self,
        user_id: str,
        user_name: str = None,
        zone_id: str = None,
        preferences: dict = None,
        permission_level: int = 0,
    ) -> UserContext:
        with self._lock:
            if user_id in self._contexts:
                ctx = self._contexts[user_id]
                if zone_id:
                    ctx.zone_id = zone_id
                ctx.last_active = time.time()
                return ctx

            # Plan #2 OpenClaw — hidratar desde disco si hay persister
            hydrated = None
            if self.persister is not None:
                data = self.persister.load(user_id)
                if data is not None:
                    hydrated = data

            ctx = UserContext(
                user_id=user_id,
                user_name=user_name or (hydrated and hydrated.get("user_name")) or f"Usuario_{user_id[:8]}",
                zone_id=zone_id,
                max_history=self.max_history,
                preferences=preferences or {},
                permission_level=permission_level,
            )
            if hydrated:
                ctx.compacted_summary = hydrated.get("compacted_summary")
                ctx.preserved_ids = list(hydrated.get("preserved_ids") or [])
                ctx.session_count = (hydrated.get("session_count") or 1) + 1
                logger.info(
                    f"[ContextManager] hydrated user={user_id} session_count={ctx.session_count}"
                )

            self._contexts[user_id] = ctx
            self._total_contexts_created += 1

            logger.debug(f"Contexto creado: {user_id} ({ctx.user_name})")
            return ctx
```

- [ ] **Step 6.4: Run hydration tests**

```bash
pytest tests/unit/orchestrator/test_context_manager_compaction.py::TestHydration -v
```

Expected: 3 passed.

- [ ] **Step 6.5: Run full ContextManager + Compactor + Persister suite**

```bash
pytest tests/unit/orchestrator/ -v
```

Expected: all green.

- [ ] **Step 6.6: Commit**

```bash
git add src/orchestrator/context_manager.py tests/unit/orchestrator/test_context_manager_compaction.py
git commit -m "feat(orchestrator): hydrate context from persister in get_or_create (plan #2 OpenClaw)"
```

---

## Task 7: Settings + main.py wiring

**Files:**
- Modify: `config/settings.yaml`
- Modify: `src/main.py`
- Modify: `src/orchestrator/__init__.py`

- [ ] **Step 7.1: Agregar bloque a `config/settings.yaml`**

Localizar el bloque existente `orchestrator.context` (~línea 631-638) y EXTENDERLO (sin tocar las claves existentes):

```yaml
  # Contexto por usuario
  context:
    max_history: 10              # Turnos de conversacion por usuario
    inactive_timeout: 300        # Segundos antes de limpiar contexto inactivo
    cleanup_interval: 60         # Intervalo de limpieza (segundos)

    # Plan #2 OpenClaw — auto-compaction de contexto
    compaction:
      enabled: false             # OFF hasta validar en prod
      threshold_turns: 6         # disparar al alcanzar este turno
      keep_recent_turns: 3       # turnos literal preservados al final
      max_summary_tokens: 200    # cap del resumen del LLM
      timeout_s: 30.0            # timeout de la llamada al 30B

    # Plan #2 OpenClaw — persistencia cross-sesión
    persistence:
      enabled: false             # OFF hasta validar. Requiere compaction.enabled=true.
      base_path: "data/contexts"
```

- [ ] **Step 7.2: Exportar nuevos símbolos en `src/orchestrator/__init__.py`**

Localizar el archivo y agregar (manteniendo exports existentes):

```python
from src.orchestrator.compactor import (
    Compactor,
    CompactionResult,
    CompactionError,
)
from src.orchestrator.context_persister import ContextPersister, PERSISTED_VERSION

__all__ = [
    # ... lo que ya estaba
    "Compactor",
    "CompactionResult",
    "CompactionError",
    "ContextPersister",
    "PERSISTED_VERSION",
]
```

> Si `__all__` no existe en el archivo, no hace falta agregarlo — los imports plain bastan.

- [ ] **Step 7.3: Wire-up en `src/main.py`**

Localizar la sección donde se construye `ContextManager` (buscar `ContextManager(` con grep). Reemplazar el bloque por algo equivalente a:

```python
# === Plan #2 OpenClaw — Compactor + Persister (opcionales) ===
context_cfg = (config.get("orchestrator", {}) or {}).get("context", {}) or {}
compaction_cfg = context_cfg.get("compaction", {}) or {}
persistence_cfg = context_cfg.get("persistence", {}) or {}

compactor = None
persister = None

if compaction_cfg.get("enabled", False):
    # HttpReasoner dedicado para compaction (apunta al mismo kza-llm-ik :8200)
    from src.llm.reasoner import HttpReasoner
    from src.orchestrator import Compactor

    compaction_reasoner = HttpReasoner(
        base_url="http://127.0.0.1:8200/v1",
        timeout=compaction_cfg.get("timeout_s", 30.0),
    )
    try:
        compaction_reasoner.load()
        compactor = Compactor(
            reasoner=compaction_reasoner,
            max_summary_tokens=compaction_cfg.get("max_summary_tokens", 200),
            timeout_s=compaction_cfg.get("timeout_s", 30.0),
        )
        logger.info(
            f"[main] Compactor habilitado (threshold={compaction_cfg.get('threshold_turns', 6)}, "
            f"keep_recent={compaction_cfg.get('keep_recent_turns', 3)})"
        )
    except Exception as e:
        logger.warning(f"[main] Compactor disabled — could not load reasoner: {e}")
        compactor = None

if persistence_cfg.get("enabled", False):
    if compactor is None:
        logger.error(
            "[main] context.persistence.enabled=true requires compaction.enabled=true. "
            "Disabling persistence."
        )
    else:
        from src.orchestrator import ContextPersister
        persister = ContextPersister(
            base_path=persistence_cfg.get("base_path", "data/contexts"),
        )
        logger.info(f"[main] Context persister habilitado en {persister.base_path}")

context_manager = ContextManager(
    max_history=context_cfg.get("max_history", 10),
    inactive_timeout=context_cfg.get("inactive_timeout", 300),
    cleanup_interval=context_cfg.get("cleanup_interval", 60),
    compactor=compactor,
    persister=persister,
    compaction_threshold=compaction_cfg.get("threshold_turns", 6),
    keep_recent_turns=compaction_cfg.get("keep_recent_turns", 3),
)

# Si hay persister, usar el cleanup loop async (no el thread)
if persister is not None:
    asyncio.create_task(context_manager.start_cleanup_loop_async())
else:
    context_manager.start_cleanup_thread()
```

> Atención al implementador: el bloque de DI exacto depende de cómo está estructurado `main.py`. Mantener el resto del wiring (alerts, vector, etc) intacto. La regla: si `compactor`/`persister` son None, el `ContextManager` se comporta IDÉNTICO a hoy. Lo nuevo solo aplica con flags `enabled: true`.

- [ ] **Step 7.4: Smoke test arranque**

```bash
python -c "from src.orchestrator import Compactor, CompactionResult, CompactionError, ContextPersister; print('imports ok')"
```

Expected: `imports ok`.

- [ ] **Step 7.5: Verificar suite completa**

```bash
pytest tests/ -x --timeout=60
```

Expected: all green (o el delta esperado: solo tests nuevos sumados).

- [ ] **Step 7.6: Commit**

```bash
git add config/settings.yaml src/main.py src/orchestrator/__init__.py
git commit -m "feat(main): DI Compactor + ContextPersister opcionales (plan #2 OpenClaw)"
```

---

## Task 8: request_router — consumo de preserved_ids

**Files:**
- Modify: `src/pipeline/request_router.py`
- Modify: `src/vectordb/...` (módulo VectorSearch — ubicar exact path con grep)
- Create/Modify: tests del request_router que correspondan

- [ ] **Step 8.1: Localizar VectorSearch.search signature**

```bash
grep -rn "def search" src/vectordb/ src/pipeline/ | head
grep -n "vector_search.search\|vector_search\.search" src/pipeline/request_router.py
```

Anotar la firma actual y el callsite exacto en `request_router.py:783`.

- [ ] **Step 8.2: Escribir test fallido para VectorSearch hint_entities**

En `tests/unit/vectordb/` (crear si no existe), agregar test que verifica que `search()` acepta `hint_entities=[...]` sin error y, si una de las hint_entities matchea exactamente algún resultado, ese resultado sube en el ranking. Este test es light: lo importante es que la firma se extienda sin romper callers.

```python
# tests/unit/vectordb/test_vector_search_hints.py
import pytest
from src.vectordb.vector_search import VectorSearch  # ajustar path real


class TestHintEntities:
    def test_signature_accepts_hint_entities(self, vector_search_fixture):
        # Solo verificamos que la firma acepta el kwarg
        result = vector_search_fixture.search(
            "prendé la luz",
            top_k=3,
            hint_entities=["light.escritorio_principal"],
        )
        assert result is not None  # cualquier shape razonable
```

> Si no existe `vector_search_fixture`, agregarlo a `tests/conftest.py` con un mock simple devolviendo lista vacía. La idea: probar la firma, no el ranking real (eso vendría en plan #2.5).

- [ ] **Step 8.3: Modificar VectorSearch.search para aceptar el kwarg**

```python
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        hint_entities: list[str] | None = None,  # plan #2 OpenClaw
    ):
        # ... lógica existente ...
        results = ...  # ranking actual
        if hint_entities:
            hint_set = set(hint_entities)
            # Boost ligero: si entity_id matchea alguna hint, +0.05 al score (clamp 1.0)
            for r in results:
                eid = r.get("entity_id") or r.get("id")
                if eid and eid in hint_set:
                    r["score"] = min(1.0, r.get("score", 0.0) + 0.05)
            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results
```

> Si la estructura interna usa otro nombre que `entity_id` / `score`, ajustar al esquema real del módulo. La operación es: identificar los results, comparar con hint_entities, boost si matchea, re-ordenar. **El boost es opcional**; el requisito mínimo es que `hint_entities` no rompa nada y que se loguee el conteo:

```python
        if hint_entities:
            logger.debug(f"[VectorSearch] hint_entities count={len(hint_entities)}")
```

- [ ] **Step 8.4: Modificar `request_router.py` para pasar preserved_ids**

Localizar el callsite ~`request_router.py:783`:

```python
        results = await asyncio.to_thread(
            self.vector_search.search,
            text,
            top_k,
            self.vector_search_threshold,
        )
```

Reemplazar por (asumiendo que ya hay un `ctx` accesible en el scope; si no, obtenerlo del context_manager):

```python
        # plan #2 OpenClaw — pasar preserved_ids como hint a VectorSearch
        ctx = self.context_manager.get(user_id) if hasattr(self, "context_manager") else None
        hint_entities = list(ctx.preserved_ids) if ctx and ctx.preserved_ids else None

        results = await asyncio.to_thread(
            self.vector_search.search,
            text,
            top_k,
            self.vector_search_threshold,
            hint_entities,
        )
```

> Si `request_router.py` no tiene referencia a `context_manager` directamente, agregarla por DI en su `__init__` (parámetro opcional, default None — backward-compat).

- [ ] **Step 8.5: Run tests del request_router + vectordb**

```bash
pytest tests/unit/vectordb/ tests/unit/pipeline/ -v
```

Expected: green.

- [ ] **Step 8.6: Commit**

```bash
git add src/vectordb/ src/pipeline/request_router.py tests/unit/vectordb/
git commit -m "feat(vectordb,pipeline): preserved_ids hint en VectorSearch (plan #2 OpenClaw)"
```

---

## Task 9: E2E integration test

**Files:**
- Create: `tests/integration/test_context_compaction_e2e.py`

- [ ] **Step 9.1: Escribir test E2E**

```python
"""E2E: trigger → compaction → snapshot → hydration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.orchestrator.context_manager import ContextManager
from src.orchestrator.compactor import Compactor, CompactionResult
from src.orchestrator.context_persister import ContextPersister


@pytest.mark.asyncio
async def test_full_lifecycle(tmp_path: Path):
    # Mock reasoner que retorna JSON canónico
    reasoner = AsyncMock()
    reasoner.complete = AsyncMock(
        return_value='{"summary": "El usuario controló iluminación en la oficina."}'
    )
    reasoner._resolved_model = "qwen3-30b-a3b"

    compactor = Compactor(reasoner=reasoner, max_summary_tokens=200, timeout_s=30.0)
    persister = ContextPersister(base_path=tmp_path / "contexts")

    mgr = ContextManager(
        max_history=20,
        inactive_timeout=0.05,  # expira rápido para snapshot
        cleanup_interval=0.05,
        compactor=compactor,
        persister=persister,
        compaction_threshold=6,
        keep_recent_turns=3,
    )

    # === Sesión 1 ===
    mgr.get_or_create("alice", "Alice")
    for i in range(7):
        mgr.add_turn("alice", "user", f"comando {i}", entities=[f"light.{i}"])

    # Esperar compaction in-memory
    await asyncio.sleep(0.1)

    ctx_pre_snapshot = mgr.get("alice")
    assert ctx_pre_snapshot.compacted_summary is not None
    assert "iluminación" in ctx_pre_snapshot.compacted_summary
    assert len(ctx_pre_snapshot.conversation_history) == 4  # 7 - keep_recent(3) compactados, +1 nuevo después del trigger en t=6

    # === Cleanup loop async — provoca snapshot ===
    cleanup_task = asyncio.create_task(mgr.start_cleanup_loop_async())
    await asyncio.sleep(0.3)  # > inactive_timeout + cleanup_interval
    mgr.stop_cleanup_loop_async()
    await cleanup_task

    assert persister.exists("alice")
    assert mgr.get("alice") is None  # purgado de memoria

    # === Sesión 2 — hidratación ===
    ctx_hydrated = mgr.get_or_create("alice", "Alice")
    assert ctx_hydrated.compacted_summary is not None
    assert "iluminación" in ctx_hydrated.compacted_summary
    assert ctx_hydrated.session_count == 2
    assert ctx_hydrated.conversation_history == []
    # preserved_ids preservados literal
    assert any(eid.startswith("light.") for eid in ctx_hydrated.preserved_ids)
```

- [ ] **Step 9.2: Run E2E**

```bash
pytest tests/integration/test_context_compaction_e2e.py -v
```

Expected: passed.

- [ ] **Step 9.3: Verificar suite completa final**

```bash
pytest tests/ --timeout=60
```

Expected: all green; nuevos tests sumados ≥ 25.

- [ ] **Step 9.4: Commit**

```bash
git add tests/integration/test_context_compaction_e2e.py
git commit -m "test(integration): E2E context compaction lifecycle (plan #2 OpenClaw)"
```

---

## Task 10: Docs + activar feature en server (manual)

> Este task NO es código. Es la activación controlada del feature en producción tras merge a main.

- [ ] **Step 10.1: Push del branch a origin + merge via PR (o merge directo si trabajamos en main)**

```bash
git push origin HEAD
```

- [ ] **Step 10.2: Deploy a server**

```bash
ssh kza 'cd ~/kza && git pull --ff-only origin main'
```

- [ ] **Step 10.3: Activar compaction (no persistencia aún) y reiniciar**

En el server, editar `~/kza/config/settings.yaml`:

```yaml
  context:
    compaction:
      enabled: true   # ← activar
```

```bash
ssh kza 'systemctl --user restart kza-voice'
```

- [ ] **Step 10.4: Validar logs en runtime**

```bash
ssh kza 'journalctl --user -u kza-voice -n 200 --no-pager | grep -E "Compactor|ContextManager"'
```

Esperado tras 6+ turnos en una sesión real:
- `[Compactor] turns=3 summary_chars=... preserved_ids=... latency=...ms`
- `[ContextManager] user=X compacted_turns=3 summary_chars=... preserved=...`

- [ ] **Step 10.5: Activar persistencia tras 24h estable**

Editar settings.yaml en server:

```yaml
    persistence:
      enabled: true
```

Reiniciar y validar:

```bash
ssh kza 'ls -la ~/kza/data/contexts/'
```

Tras una sesión y 5+ minutos de inactividad: archivos `<user_id>.json` deben aparecer.

- [ ] **Step 10.6: Actualizar memoria del proyecto**

Crear/actualizar memoria en `~/.claude/projects/-Users-yo-Documents-kza/memory/project_openclaw_plan2_done.md` siguiendo el formato de `project_openclaw_plan1_done.md`.

---

## Outcome verificable

Tras completar todos los tasks:

1. `pytest tests/` pasa con ≥ 25 nuevos tests verdes.
2. Con `compaction.enabled: false` → comportamiento idéntico al baseline (regression-safe).
3. Con `compaction.enabled: true` y sesiones >6 turnos: logs `[Compactor]` aparecen, summary se acumula, history se reduce.
4. Con `persistence.enabled: true`: tras inactividad, archivos `data/contexts/<user_id>.json` se crean. Próxima sesión del mismo usuario: log `hydrated user=X session_count=N`.
5. Identifier policy verificable: ningún log del Compactor menciona substring matching `[a-z_]+\.[a-z_0-9]+` en el campo `summary` (los IDs viven en `preserved_ids`).
