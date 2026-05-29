# Fast path + Cloud Reasoner (MiniMax-M2.7-highspeed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reducir latencia percibida del fast path (<300ms) y reemplazar el reasoner local del slow path por MiniMax-M2.7-highspeed servido en el cloud de MiniMax (OpenAI-compatible).

**Architecture:** FASE A toca el camino crítico de audio (búsqueda vectorial async, warmup de embeddings, speaker ID diferido) sin cambiar GPU ni la premisa local. FASE B extiende `HttpReasoner`/`LLMRouter` para hablar `chat/completions` contra `https://api.minimax.io/v1`, con auth por env var explícita, timeouts de red respetados, clasificación de errores de red, y gates de privacidad/log.

**Tech Stack:** Python 3.13, async/await, pytest, ChromaDB + SentenceTransformer (BGE-M3), `openai` SDK (OpenAI-compat), llama.cpp/ik_llama (local), MiniMax cloud API.

**Spec:** `docs/superpowers/specs/2026-05-29-fastpath-and-cloud-reasoner-design.md`

**Decisiones del dueño fijadas:**
- MiniMax = **reemplazo total** del reasoner slow path (sin fallback GLM-Air; GLM-Air queda comentado para rollback). Model id confirmado: `MiniMax-M2.7-highspeed`.
- A4 (BGE-M3 → GPU) **fuera de alcance**.
- A3 (speaker ID): **diferir siempre** en domótica; `voice_auth` espera el speaker aparte.

**Convenciones del repo (respetar):** DI por constructor, composición sobre herencia, `@dataclass` para DTOs, `from src.modulo import X` (no relativos), `logging.getLogger(__name__)`, mensajes de usuario en español / código y logs en inglés, tests con pytest + fixtures en `conftest.py` / mocks en `tests/mocks/`, **no** crear archivos de config nuevos (todo en `config/settings.yaml`).

---

## FASE A — Fast path

### Task A1: Búsqueda vectorial async (no bloquear el event loop)

**Contexto:** `ChromaSync.search_command()` es síncrono; `embedder.encode()` (BGE-M3 en CPU, ~48ms) corre en el hilo del event loop y serializa el pipeline async. 5 call sites; solo el hot path (`dispatcher.py:590`) está en contexto async crítico. Estrategia DRY/bajo-riesgo: **mantener `search_command` síncrono** (lo usan callers sync) y agregar un wrapper async `asearch_command` que delega vía `asyncio.to_thread`; migrar solo el dispatcher.

**Files:**
- Modify: `src/vectordb/chroma_sync.py` (agregar `asearch_command`, ~después de línea 312 donde termina la firma de `search_command`)
- Modify: `src/orchestrator/dispatcher.py:590`
- Test: `tests/unit/vectordb/test_chroma_search_async.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/vectordb/test_chroma_search_async.py`:

```python
import asyncio
import time

import pytest

from src.vectordb.chroma_sync import ChromaSync


class _SlowSyncChroma(ChromaSync):
    """Override search_command con un sleep síncrono para probar el offload."""

    def __init__(self):
        # No llamamos super().__init__ — no necesitamos chroma real para este test.
        self._search_calls = []

    def search_command(self, query, threshold=0.65, **kwargs):
        time.sleep(0.05)  # simula el encode CPU bloqueante (~48ms)
        self._search_calls.append((query, threshold, kwargs))
        return {"entity_id": "light.living", "similarity": 0.9}


@pytest.mark.asyncio
async def test_asearch_command_runs_off_event_loop():
    chroma = _SlowSyncChroma()

    # Mientras asearch_command corre en un thread, el event loop debe seguir
    # libre para ejecutar otra corutina concurrentemente.
    ticks = 0

    async def _ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1

    result, _ = await asyncio.gather(
        chroma.asearch_command("prendé la luz", 0.65, service_filter="turn_on"),
        _ticker(),
    )

    assert result == {"entity_id": "light.living", "similarity": 0.9}
    assert ticks == 5, "el event loop quedó bloqueado durante el encode"
    # Verifica passthrough fiel de args/kwargs al sync subyacente.
    assert chroma._search_calls == [("prendé la luz", 0.65, {"service_filter": "turn_on"})]
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/vectordb/test_chroma_search_async.py -v`
Expected: FAIL con `AttributeError: 'ChromaSync' object has no attribute 'asearch_command'`

- [ ] **Step 3: Implementar el wrapper async**

En `src/vectordb/chroma_sync.py`, inmediatamente después del cierre del método `search_command` (tras su `return` final, antes del siguiente método), agregar:

```python
    async def asearch_command(
        self,
        query: str,
        threshold: float = 0.65,
        service_filter: str | None = None,
        query_slots: dict | None = None,
        hint_entities: list[str] | None = None,
        prefer_area: str | None = None,
    ) -> dict | None:
        """Variante async de search_command para el fast path.

        Delega el trabajo síncrono (encode BGE-M3 en CPU + query a Chroma) a un
        thread vía asyncio.to_thread para no bloquear el event loop. Misma
        firma y retorno que search_command. Usar en contextos async calientes
        (dispatcher fast path); los callers sync siguen usando search_command.
        """
        import asyncio

        return await asyncio.to_thread(
            self.search_command,
            query,
            threshold,
            service_filter=service_filter,
            query_slots=query_slots,
            hint_entities=hint_entities,
            prefer_area=prefer_area,
        )
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/unit/vectordb/test_chroma_search_async.py -v`
Expected: PASS

- [ ] **Step 5: Migrar el hot path del dispatcher a la versión async**

En `src/orchestrator/dispatcher.py:590`, reemplazar:

```python
            command = self.chroma.search_command(
                text,
                self.vector_threshold,
                service_filter=service_filter,
                query_slots=query_slots,
                prefer_area=prefer_area,
            )
```

por:

```python
            command = await self.chroma.asearch_command(
                text,
                self.vector_threshold,
                service_filter=service_filter,
                query_slots=query_slots,
                prefer_area=prefer_area,
            )
```

(El método que contiene la línea 590 ya es `async` — el bloque `if path == PathType.FAST_DOMOTICS:` corre dentro de la corutina del dispatcher.)

- [ ] **Step 6: Verificar que no rompimos los otros call sites**

Run: `grep -rn "search_command" src/ | grep -v "def .*search_command"`
Expected: los call sites en `voice_pipeline.py:213`, `request_router.py:949/982`, `routine_manager.py:320` siguen usando `search_command` (sync, sin cambios); solo `dispatcher.py` usa `asearch_command`.

- [ ] **Step 7: Correr la suite del dispatcher y vectordb**

Run: `pytest tests/unit/vectordb/ tests/unit/orchestrator/ -v`
Expected: PASS (sin regresiones)

- [ ] **Step 8: Commit**

```bash
git add src/vectordb/chroma_sync.py src/orchestrator/dispatcher.py tests/unit/vectordb/test_chroma_search_async.py
git commit -m "perf(fastpath): async vector search wrapper, unblock event loop in dispatcher"
```

---

### Task A2: Arreglar el warmup de embeddings (cold-start del primer comando)

**Contexto:** En `main.py:155` el guard `getattr(chroma, "_embedder", None) is not None` da `False` porque el embedder se crea lazy (`_embedder` arranca en `None`, se materializa en `initialize()`). El warmup se saltea en silencio → el primer comando paga ~48ms de cold start. Fix: exponer `ChromaSync.warmup_embedder()` que fuerza la materialización vía la property `embedder` y hace un encode dummy; llamarlo desde el warmup sin el guard frágil.

**Files:**
- Modify: `src/vectordb/chroma_sync.py` (agregar `warmup_embedder`)
- Modify: `src/main.py:154-161` (bloque "BGE-M3 warmup")
- Test: `tests/unit/vectordb/test_chroma_warmup.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/vectordb/test_chroma_warmup.py`:

```python
from unittest.mock import MagicMock

from src.vectordb.chroma_sync import ChromaSync


def test_warmup_embedder_forces_lazy_init_and_encodes(monkeypatch):
    chroma = ChromaSync.__new__(ChromaSync)  # sin __init__ (evita chroma real)
    chroma._embedder = None

    fake_embedder = MagicMock()
    init_called = {"n": 0}

    def _fake_initialize():
        init_called["n"] += 1
        chroma._embedder = fake_embedder

    # La property `embedder` llama initialize() si _embedder es None.
    monkeypatch.setattr(chroma, "initialize", _fake_initialize)

    chroma.warmup_embedder()

    assert init_called["n"] == 1, "warmup no materializó el embedder lazy"
    fake_embedder.encode.assert_called_once()  # encode dummy ejecutado
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/vectordb/test_chroma_warmup.py -v`
Expected: FAIL con `AttributeError: 'ChromaSync' object has no attribute 'warmup_embedder'`

- [ ] **Step 3: Implementar `warmup_embedder`**

En `src/vectordb/chroma_sync.py`, después de la property `embedder` (línea ~94), agregar:

```python
    def warmup_embedder(self) -> float:
        """Forzar la inicialización lazy del embedder y compilar kernels.

        Usa la property `embedder` (que llama initialize() si hace falta) en
        vez de leer `_embedder` directo — así el warmup NO se saltea cuando el
        embedder todavía no fue materializado (bug 2026-05-29: el guard lazy en
        main.py daba False y el primer comando pagaba el cold start de ~48ms).

        Returns:
            Latencia del encode dummy en ms (para logging del warmup).
        """
        import time

        start = time.perf_counter()
        self.embedder.encode(["warmup"])
        return (time.perf_counter() - start) * 1000
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/unit/vectordb/test_chroma_warmup.py -v`
Expected: PASS

- [ ] **Step 5: Usar `warmup_embedder` en `main.py` (sin el guard frágil)**

En `src/main.py`, reemplazar el bloque (líneas 154-161):

```python
    # BGE-M3 warmup — compila kernels del embedder usado por ChromaDB
    if chroma is not None and getattr(chroma, "_embedder", None) is not None:
        t0 = time.perf_counter()
        try:
            chroma._embedder.encode(["warmup"])
            timings["bge_m3"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning(f"Warmup BGE-M3 skipped: {e}")
```

por:

```python
    # BGE-M3 warmup — compila kernels del embedder usado por ChromaDB.
    # Usa warmup_embedder() (vía property) en vez del guard lazy `_embedder`,
    # que daba False y salteaba el warmup → el primer comando pagaba ~48ms cold.
    if chroma is not None:
        try:
            timings["bge_m3"] = chroma.warmup_embedder()
        except Exception as e:
            logger.warning(f"Warmup BGE-M3 skipped: {e}")
```

- [ ] **Step 6: Verificar arranque y log de warmup**

Run: `pytest tests/unit/vectordb/ -v`
Expected: PASS. El log de arranque ahora muestra `bge_m3=<N>ms` en la línea `Warmup: ...` (verificable en runtime, no en este test unitario).

- [ ] **Step 7: Commit**

```bash
git add src/vectordb/chroma_sync.py src/main.py tests/unit/vectordb/test_chroma_warmup.py
git commit -m "fix(fastpath): warmup BGE-M3 via property, kill silent cold-start on first command"
```

---

### Task A3: Diferir Speaker ID en el fast path de domótica

**Contexto:** `process_command()` corre STT + speaker ID + emotion en paralelo (`_process_parallel`, `asyncio.gather`). El total = max(stt, speaker, emotion); cuando STT es rápido (<100ms) ECAPA (speaker) es el limitante (−50/−80ms). El intent NO se conoce antes del STT, así que no se puede "saltear por intent" pre-STT. **Decisión:** agregar `await_speaker_id: bool` a `process_command`. Cuando `False` (fast path domótica), el speaker ID corre **detached** y `process_command` retorna apenas termina el STT con `user=None`; el resultado del speaker popula `self._current_user` cuando llega. Las acciones `voice_auth` que requieren identidad esperan el speaker aparte (ver sub-task A3b).

**Files:**
- Modify: `src/pipeline/command_processor.py` (`process_command`, `_process_parallel`)
- Test: `tests/unit/pipeline/test_command_processor_defer_speaker.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/pipeline/test_command_processor_defer_speaker.py`:

```python
import asyncio
import time

import numpy as np
import pytest

from src.pipeline.command_processor import CommandProcessor


class _FastSTT:
    def transcribe(self, audio, sample_rate):
        return "prendé la luz", 10.0  # STT rápido (10ms)


class _SlowSpeakerID:
    """ECAPA lento (60ms) — debe NO bloquear el retorno cuando se difiere."""

    def identify(self, audio, embeddings):
        time.sleep(0.06)
        match = type("M", (), {"is_known": True, "user_id": "u1", "confidence": 0.9})()
        return match


class _UserMgr:
    _version = 0

    def get_all_embeddings(self):
        return {"u1": np.zeros(192, dtype=np.float32)}

    def get_user(self, uid):
        return type("U", (), {"name": "Gabriel"})()

    def update_last_seen(self, uid):
        pass


@pytest.mark.asyncio
async def test_defer_speaker_id_returns_before_speaker_resolves():
    cp = CommandProcessor(
        stt=_FastSTT(),
        speaker_identifier=_SlowSpeakerID(),
        user_manager=_UserMgr(),
        emotion_detector=None,
    )

    t0 = time.perf_counter()
    result = await cp.process_command(
        np.zeros(16000, dtype=np.float32),
        await_speaker_id=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert result.text == "prendé la luz"
    assert result.user is None, "no debe esperar el speaker cuando await_speaker_id=False"
    assert elapsed_ms < 50, f"retornó en {elapsed_ms:.0f}ms — bloqueó por el speaker (60ms)"

    # El speaker resuelve en background y popula _current_user a posteriori.
    await asyncio.sleep(0.1)
    assert cp.get_current_user() is not None
    assert cp.get_current_user().name == "Gabriel"


@pytest.mark.asyncio
async def test_await_speaker_id_true_keeps_blocking_behavior():
    cp = CommandProcessor(
        stt=_FastSTT(),
        speaker_identifier=_SlowSpeakerID(),
        user_manager=_UserMgr(),
        emotion_detector=None,
    )
    result = await cp.process_command(
        np.zeros(16000, dtype=np.float32),
        await_speaker_id=True,
    )
    assert result.user is not None  # comportamiento actual: espera al speaker
    assert result.user.name == "Gabriel"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/pipeline/test_command_processor_defer_speaker.py -v`
Expected: FAIL con `TypeError: process_command() got an unexpected keyword argument 'await_speaker_id'`

- [ ] **Step 3: Implementar el deferral**

En `src/pipeline/command_processor.py`, cambiar la firma de `process_command` (línea 92):

```python
    async def process_command(
        self,
        audio: np.ndarray,
        use_parallel: bool = True,
        pretranscribed_text: str | None = None,
        await_speaker_id: bool = True,
    ) -> ProcessedCommand:
```

Y actualizar el docstring `Args:` agregando:

```python
            await_speaker_id: Si False (fast path domótica), NO espera el
                speaker ID — transcribe, retorna con user=None, y corre el
                speaker en background populando _current_user al terminar.
                La acción HA no necesita identidad. voice_auth lo espera aparte.
```

En la rama paralela (línea 134), pasar el flag a `_process_parallel`:

```python
        elif use_parallel and (self.speaker_id or self.emotion_detector):
            text, stt_ms, speaker_result, emotion_result = await self._process_parallel(
                audio, await_speaker_id=await_speaker_id
            )
```

Modificar `_process_parallel` (línea 175) para aceptar el flag y, cuando `await_speaker_id=False`, NO incluir el speaker en el `gather` — lanzarlo detached:

```python
    async def _process_parallel(
        self, audio: np.ndarray, await_speaker_id: bool = True
    ) -> tuple[str, float, tuple | None, object | None]:
        """
        Procesar STT, Speaker ID y Emotion en paralelo REAL con asyncio.gather().

        Si await_speaker_id=False, el speaker ID se lanza detached (no entra al
        gather): el retorno no lo espera y _current_user se popula en background.

        Returns:
            Tuple[text, stt_ms, speaker_result, emotion_result]
            (speaker_result es None cuando se difiere)
        """
        loop = asyncio.get_running_loop()
        t_parallel = time.perf_counter()

        stt_task = loop.run_in_executor(None, self.stt.transcribe, audio, self.sample_rate)

        defer_speaker = (
            not await_speaker_id and self.speaker_id is not None and self.user_manager is not None
        )

        speaker_task = (
            loop.run_in_executor(None, self._identify_speaker, audio)
            if self.speaker_id and self.user_manager and not defer_speaker
            else asyncio.sleep(0)
        )

        emotion_task = (
            loop.run_in_executor(None, self.emotion_detector.detect, audio)
            if self.emotion_detector
            else asyncio.sleep(0)
        )

        results = await asyncio.gather(
            stt_task, speaker_task, emotion_task,
            return_exceptions=True
        )

        parallel_ms = (time.perf_counter() - t_parallel) * 1000

        stt_result = results[0] if not isinstance(results[0], Exception) else ("", 0)
        text, stt_ms = stt_result if isinstance(stt_result, tuple) else ("", 0)

        if defer_speaker:
            # Lanzar el speaker en background; popula _current_user al terminar.
            self._spawn_deferred_speaker_id(audio)
            speaker_result = None
        elif self.speaker_id and self.user_manager and not isinstance(results[1], Exception):
            speaker_result = results[1]
        else:
            speaker_result = None

        emotion_result = (
            results[2] if self.emotion_detector
            and not isinstance(results[2], Exception) and results[2] is not None
            else None
        )

        logger.debug(
            f"[Parallel GATHER {parallel_ms:.0f}ms] STT + "
            f"{'Speaker(deferred) + ' if defer_speaker else 'Speaker ID + '}Emotion"
        )

        return text, stt_ms, speaker_result, emotion_result

    def _spawn_deferred_speaker_id(self, audio: np.ndarray) -> None:
        """Resolver speaker ID en background (fast path domótica).

        No bloquea el retorno de process_command. Cuando termina, popula
        self._current_user para que el contexto multi-usuario quede actualizado
        para el siguiente turno. Excepciones se loguean, no se propagan.
        """
        loop = asyncio.get_running_loop()

        async def _runner():
            try:
                user, confidence, _ = await loop.run_in_executor(
                    None, self._identify_speaker, audio
                )
                if user is not None:
                    self._current_user = user
            except Exception as e:
                logger.debug(f"Deferred speaker ID skipped: {e}")

        # Guardar referencia para que el GC no cancele la task en vuelo.
        task = loop.create_task(_runner())
        self._deferred_speaker_tasks = getattr(self, "_deferred_speaker_tasks", set())
        self._deferred_speaker_tasks.add(task)
        task.add_done_callback(self._deferred_speaker_tasks.discard)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/unit/pipeline/test_command_processor_defer_speaker.py -v`
Expected: PASS (ambos tests)

- [ ] **Step 5: Correr la suite del pipeline (no regresiones en el path default)**

Run: `pytest tests/unit/pipeline/ -v`
Expected: PASS. `await_speaker_id` default `True` preserva el comportamiento actual de todos los callers existentes.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/command_processor.py tests/unit/pipeline/test_command_processor_defer_speaker.py
git commit -m "perf(fastpath): defer speaker ID off the critical path for domotics (await_speaker_id flag)"
```

---

### Task A3b: Wirear los callers para diferir speaker; el slow path lo espera antes de razonar

**Contexto (corregido con la realidad del código):** No existe ningún feature `voice_auth` hoy (`grep -rn voice_auth src/` → vacío). Los únicos callers de `process_command` son `request_router.py:422` y `:703`, y **ambos procesan el comando antes de conocer el path** (fast/slow se decide post-STT, sobre el texto). Diseño: pasar `await_speaker_id=False` en ambos call sites (todo difiere el speaker al arrancar). El **fast path de domótica** procede sin identidad (la acción HA no la necesita; `_current_user` se popula en background para el próximo turno). El **slow path** (que sí usa identidad para el contexto del reasoner) llama `ensure_speaker_resolved()` antes de armar el prompt — ahí esperar ~60ms es ruido frente a los 5-30s del LLM. `ensure_speaker_resolved` queda además como hook listo para un futuro `voice_auth`.

**Files:**
- Modify: `src/pipeline/command_processor.py` (agregar `ensure_speaker_resolved`)
- Modify: `src/pipeline/request_router.py:422` y `:703` (pasar `await_speaker_id=False`); rama slow path (await `ensure_speaker_resolved` antes de razonar)
- Test: `tests/unit/pipeline/test_ensure_speaker_resolved.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/pipeline/test_ensure_speaker_resolved.py`:

```python
import asyncio
import time

import numpy as np
import pytest

from src.pipeline.command_processor import CommandProcessor


class _FastSTT:
    def transcribe(self, audio, sample_rate):
        return "contame un chiste", 10.0


class _SlowSpeakerID:
    def identify(self, audio, embeddings):
        time.sleep(0.06)
        return type("M", (), {"is_known": True, "user_id": "u1", "confidence": 0.9})()


class _UserMgr:
    _version = 0
    def get_all_embeddings(self): return {"u1": np.zeros(192, dtype=np.float32)}
    def get_user(self, uid): return type("U", (), {"name": "Gabriel"})()
    def update_last_seen(self, uid): pass


@pytest.mark.asyncio
async def test_ensure_speaker_resolved_waits_for_deferred_task():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=_SlowSpeakerID(),
                          user_manager=_UserMgr(), emotion_detector=None)
    result = await cp.process_command(np.zeros(16000, dtype=np.float32), await_speaker_id=False)
    assert result.user is None  # difirió

    user = await cp.ensure_speaker_resolved(timeout_s=1.0)  # slow path espera
    assert user is not None and user.name == "Gabriel"


@pytest.mark.asyncio
async def test_ensure_speaker_resolved_returns_none_on_timeout():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=None,
                          user_manager=None, emotion_detector=None)
    # Sin tasks diferidas, devuelve el _current_user actual (None) sin colgar.
    user = await cp.ensure_speaker_resolved(timeout_s=0.1)
    assert user is None
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/pipeline/test_ensure_speaker_resolved.py -v`
Expected: FAIL (`AttributeError: 'CommandProcessor' object has no attribute 'ensure_speaker_resolved'`)

- [ ] **Step 3: Implementar `ensure_speaker_resolved`**

En `src/pipeline/command_processor.py`, agregar tras `_spawn_deferred_speaker_id` (de A3):

```python
    async def ensure_speaker_resolved(self, timeout_s: float = 1.0) -> object | None:
        """Esperar a que las tasks de speaker ID diferidas terminen.

        Lo usa el slow path (y un futuro voice_auth) cuando necesita identidad
        confirmada tras un process_command(await_speaker_id=False). Devuelve el
        usuario actual (o None si no se identificó o hubo timeout).
        """
        import asyncio

        tasks = getattr(self, "_deferred_speaker_tasks", set())
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                logger.warning("ensure_speaker_resolved: timeout esperando speaker ID")
        return self._current_user
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/unit/pipeline/test_ensure_speaker_resolved.py -v`
Expected: PASS

- [ ] **Step 5: Wirear los dos call sites para diferir**

En `src/pipeline/request_router.py`, en `:422-424` y `:703-705`, agregar `await_speaker_id=False`:

```python
        cmd = await self.command_processor.process_command(
            audio, use_parallel=True, pretranscribed_text=pretranscribed_text,
            await_speaker_id=False,
        )
```

- [ ] **Step 6: Que el slow path espere el speaker antes de razonar**

Localizar la rama del slow path en `request_router.py` (donde se decide reasoning / se llama al `llm_router`/reasoner — buscar con `grep -n "slow\|reason\|llm_router\|DEEP" src/pipeline/request_router.py`). Justo antes de armar el contexto/prompt del reasoner, agregar:

```python
        # Slow path: necesitamos identidad para el contexto del reasoner. El
        # speaker se difirió en process_command (fast path no la necesita);
        # acá la esperamos — ~60ms es ruido frente a los 5-30s del LLM.
        if cmd.user is None:
            resolved_user = await self.command_processor.ensure_speaker_resolved(timeout_s=1.0)
            if resolved_user is not None:
                cmd.user = resolved_user
                result["user"] = resolved_user
```

- [ ] **Step 7: Suite completa del pipeline (sin regresiones)**

Run: `pytest tests/unit/pipeline/ -v`
Expected: PASS. Verificar que los tests existentes de `request_router` que asumen `cmd.user` poblado en fast path siguen pasando; si alguno asumía identidad inmediata en fast path, ajustarlo para reflejar el defer (es el comportamiento deseado).

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/command_processor.py src/pipeline/request_router.py tests/unit/pipeline/test_ensure_speaker_resolved.py
git commit -m "perf(fastpath): defer speaker at call sites, slow path awaits identity before reasoning"
```

- [ ] **Step 8: Validar latencia end-to-end (FASE A completa)**

Run: `python tools/benchmark_latency.py --iterations 20`
Expected: latencia p95 del fast path menor que el baseline (objetivo: recuperar el cold-start de A2 + el no-bloqueo de A1 + el defer de A3). Anotar números antes/después.

---

## FASE B — Reasoner cloud (MiniMax-M2.7-highspeed)

### Task B2: API key por env var explícita (`api_key_env`)

**Contexto:** `_resolve_api_key(base_url)` infiere la env var por puerto. `https://api.minimax.io/v1` tiene `port=None` → cae al fallback y usa la key local equivocada o `"not-used"`. Fix: parámetro `api_key_env` explícito; si está seteado, usar `os.getenv(api_key_env)` y **fallar ruidoso** si falta (no sentinel silencioso para cloud). (Hacemos B2 antes que B1 porque B3/main.py los necesita juntos; el orden de tasks B2→B1→B5→B3→B6 minimiza rework.)

**Files:**
- Modify: `src/llm/reasoner.py` (`_resolve_api_key`, `HttpReasoner.__init__`/`_try_connect`, `FastRouter.__init__`/`load`)
- Test: `tests/unit/llm/test_resolve_api_key.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/llm/test_resolve_api_key.py`:

```python
import pytest

from src.llm.reasoner import _resolve_api_key


def test_explicit_api_key_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret-123")
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    key = _resolve_api_key("https://api.minimax.io/v1", api_key_env="MINIMAX_API_KEY")
    assert key == "mm-secret-123"


def test_explicit_api_key_env_missing_raises(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        _resolve_api_key("https://api.minimax.io/v1", api_key_env="MINIMAX_API_KEY")


def test_legacy_port_heuristic_still_works(monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "llama-key")
    key = _resolve_api_key("http://127.0.0.1:8200/v1")  # sin api_key_env
    assert key == "llama-key"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/llm/test_resolve_api_key.py -v`
Expected: FAIL (`test_explicit_api_key_env_takes_precedence` falla con `TypeError: _resolve_api_key() got an unexpected keyword argument 'api_key_env'`)

- [ ] **Step 3: Implementar `api_key_env`**

En `src/llm/reasoner.py`, reemplazar la firma y el inicio de `_resolve_api_key` (línea 17):

```python
def _resolve_api_key(base_url: str, api_key_env: str | None = None) -> str:
    """Pick the bearer token for an OpenAI-compat endpoint.

    Si `api_key_env` está seteado (endpoints cloud / explícitos), usa esa env
    var directamente y FALLA RUIDOSO si no existe — un endpoint cloud sin key
    es un error de deploy, no algo para silenciar con el sentinel "not-used".

    Si `api_key_env` es None, cae a la heurística legacy por puerto:
    :8100 → VLLM_API_KEY, :8101/:8200 → LLAMA_API_KEY. Unknown port → intenta
    ambos; si no hay ninguna, devuelve "not-used" con warning (local-dev contra
    endpoints sin auth).
    """
    if api_key_env is not None:
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(
                f"API key env var {api_key_env!r} no está seteada para {base_url} — "
                f"endpoint cloud requiere auth. Configurar en /home/kza/secrets/.env "
                f"(chmod 600) y cargar vía systemd EnvironmentFile."
            )
        return key

    port = urlparse(base_url).port
    # ... (resto del cuerpo legacy sin cambios)
```

(El cuerpo desde `port = urlparse(base_url).port` hacia abajo queda idéntico al actual.)

- [ ] **Step 4: Propagar `api_key_env` a `HttpReasoner`**

En `HttpReasoner.__init__` (línea 404), agregar el parámetro y guardarlo:

```python
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8200/v1",
        model: str | None = None,
        timeout: float = 120.0,
        idle_timeout_s: float | None = None,
        api_key_env: str | None = None,
        api_style: str = "completions",
        verify_ssl: bool = True,
        **_ignored_legacy,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.idle_timeout_s = idle_timeout_s
        self.api_key_env = api_key_env
        self.api_style = api_style
        self.verify_ssl = verify_ssl
        self._client = None
        self._resolved_model = None
        self._resolved_base_url = None
        self._last_metrics: dict | None = None
        self._metrics_tracker = None
        self._endpoint_id: str | None = None
```

(`api_style` y `verify_ssl` se usan en B1/B5 — se agregan acá para no tocar la firma dos veces.)

En `_try_connect` (línea 433), pasar `api_key_env`:

```python
        client = OpenAI(
            base_url=base_url,
            api_key=_resolve_api_key(base_url, self.api_key_env),
            timeout=self.timeout,
        )
```

En `FastRouter.__init__` (línea 593) agregar `api_key_env: str | None = None` y guardarlo (`self.api_key_env = api_key_env`); en `FastRouter.load` (línea 618) cambiar a `api_key=_resolve_api_key(self.base_url, self.api_key_env)`.

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `pytest tests/unit/llm/test_resolve_api_key.py -v`
Expected: PASS

- [ ] **Step 6: Regresión de la suite llm**

Run: `pytest tests/unit/llm/ -v`
Expected: PASS (defaults preservan comportamiento)

- [ ] **Step 7: Commit**

```bash
git add src/llm/reasoner.py tests/unit/llm/test_resolve_api_key.py
git commit -m "feat(llm): explicit api_key_env for cloud endpoints, fail loud when missing"
```

---

### Task B1: Path `chat/completions` para endpoints cloud (`api_style`)

**Contexto (blocker):** `HttpReasoner` usa `client.completions.create(prompt=...)` → `/v1/completions`. MiniMax solo soporta `/v1/chat/completions` (`messages=...`). Fix: cuando `api_style == "chat"`, construir `messages=[{"role":"user","content":prompt}]`, llamar `chat.completions.create`, y parsear `message.content` / `delta.content`. Aislar el parseo en un helper para no duplicar en los 3 paths de completin (`__call__`, `complete` no-stream, `complete` stream).

**Files:**
- Modify: `src/llm/reasoner.py` (`HttpReasoner.__call__`, `complete`, + helpers `_create_completion`, `_extract_text`, `_extract_stream_delta`)
- Test: `tests/unit/llm/test_httpreasoner_chat_style.py` (crear)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/llm/test_httpreasoner_chat_style.py`:

```python
from unittest.mock import MagicMock

import pytest

from src.llm.reasoner import HttpReasoner


def _make_chat_client(reply_text):
    """Mock de cliente OpenAI que solo soporta chat.completions (como MiniMax)."""
    client = MagicMock()
    msg = MagicMock()
    msg.message.content = reply_text
    resp = MagicMock()
    resp.choices = [msg]
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    client.chat.completions.create.return_value = resp
    # completions.create (legacy) NO existe → AttributeError si se usa
    client.completions.create.side_effect = AttributeError("no legacy completions")
    return client


@pytest.mark.asyncio
async def test_complete_uses_chat_when_api_style_chat():
    r = HttpReasoner(
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M2.7-highspeed",
        api_style="chat",
    )
    r._client = _make_chat_client("la luz está encendida")
    r._resolved_model = "MiniMax-M2.7-highspeed"

    out = await r.complete("¿está la luz?", max_tokens=64)

    assert out == "la luz está encendida"
    # Verifica que mandó messages, no prompt.
    _, kwargs = r._client.chat.completions.create.call_args
    assert kwargs["messages"] == [{"role": "user", "content": "¿está la luz?"}]
    assert kwargs["model"] == "MiniMax-M2.7-highspeed"


def test_call_chat_style_returns_completions_shaped_dict():
    r = HttpReasoner(base_url="https://api.minimax.io/v1", api_style="chat")
    r._client = _make_chat_client("hola")
    r._resolved_model = "MiniMax-M2.7-highspeed"

    result = r("dummy prompt")
    assert result["choices"][0]["text"] == "hola"  # backward-compat shape
    assert result["usage"]["completion_tokens"] == 7
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/unit/llm/test_httpreasoner_chat_style.py -v`
Expected: FAIL (usa `completions.create` → `AttributeError` del mock)

- [ ] **Step 3: Implementar los helpers + ramas chat**

En `src/llm/reasoner.py`, dentro de `HttpReasoner`, agregar helpers (después de `_try_connect`, ~línea 444):

```python
    def _create_completion(self, prompt: str, *, max_tokens: int, temperature: float,
                           top_p: float = 0.9, stop=None, stream: bool = False):
        """Llamar al endpoint según api_style. Devuelve el objeto resp del SDK."""
        if self.api_style == "chat":
            return self._client.chat.completions.create(
                model=self._resolved_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or None,
                stream=stream,
            )
        return self._client.completions.create(
            model=self._resolved_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
            stream=stream,
        )

    def _extract_text(self, resp) -> str:
        """Texto de una respuesta no-streaming, según api_style."""
        if self.api_style == "chat":
            return resp.choices[0].message.content or ""
        return resp.choices[0].text or ""

    def _extract_stream_delta(self, chunk) -> str:
        """Delta de texto de un chunk streaming, según api_style."""
        try:
            if self.api_style == "chat":
                return chunk.choices[0].delta.content or ""
            return chunk.choices[0].text or ""
        except (AttributeError, IndexError):
            return ""
```

Reescribir `__call__` (líneas 457-487) para usar los helpers:

```python
    def __call__(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9,
                 top_k=40, repeat_penalty=1.1, stop=None) -> dict:
        """Completions-style: devuelve {choices: [{text: ...}], usage: {...}}."""
        if self._client is None:
            self.load()
        start = time.perf_counter()
        resp = self._create_completion(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop,
        )
        text = self._extract_text(resp)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"HttpReasoner ({elapsed_ms:.0f}ms) {len(prompt)}chars → {len(text)}chars")
        return {
            "choices": [{"text": text}],
            "usage": {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            },
        }
```

En `complete` (líneas 489-556) reemplazar las llamadas directas a `self._client.completions.create(...)` por `self._create_completion(...)` y los `resp.choices[0].text` por `self._extract_text(resp)`; en el path stream, `_open_stream` usa `self._create_completion(prompt, ..., stream=True)` y el loop usa `self._extract_stream_delta(chunk)` en vez del bloque `try: delta = chunk.choices[0].text`.

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/unit/llm/test_httpreasoner_chat_style.py -v`
Expected: PASS

- [ ] **Step 5: Regresión — el path completions legacy sigue intacto**

Run: `pytest tests/unit/llm/ -v`
Expected: PASS (tests existentes del HttpReasoner usan `api_style="completions"` default → mismas llamadas que antes)

- [ ] **Step 6: Commit**

```bash
git add src/llm/reasoner.py tests/unit/llm/test_httpreasoner_chat_style.py
git commit -m "feat(llm): chat/completions api_style for cloud reasoners (MiniMax compat)"
```

---

### Task B5: Respetar `ep.timeout_s` y clasificar errores de red para failover

**Contexto:** `LLMRouter.complete()` no envuelve `ep.client.complete()` en timeout → usa solo el 5.0s del dispatcher, frágil con red. Y `classify_error()` no mapea errores de red del SDK OpenAI (`APITimeoutError`, `APIConnectionError`, 5xx), `ssl.SSLError` ni `socket.gaierror` → caen como PERMANENT y **bloquean el failover**.

**Files:**
- Modify: `src/llm/router.py` (`complete`)
- Modify: `src/llm/error_classifier.py`
- Test: `tests/unit/llm/test_router_timeout.py`, `tests/unit/llm/test_error_classifier_network.py` (crear)

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/llm/test_error_classifier_network.py`:

```python
import socket
import ssl

from src.llm.error_classifier import classify_error
from src.llm.types import ErrorKind


def test_ssl_error_is_failover_worthy():
    kind = classify_error(ssl.SSLError("handshake failed"))
    assert kind == ErrorKind.TIMEOUT
    assert kind.is_failover_worthy()


def test_dns_error_is_failover_worthy():
    kind = classify_error(socket.gaierror("Name or service not known"))
    assert kind == ErrorKind.TIMEOUT
    assert kind.is_failover_worthy()


def test_openai_timeout_message_is_failover_worthy():
    kind = classify_error(Exception("Request timed out."))
    assert kind == ErrorKind.TIMEOUT


def test_5xx_is_failover_worthy():
    assert classify_error(Exception("503 Service Unavailable")).is_failover_worthy()
```

Crear `tests/unit/llm/test_router_timeout.py`:

```python
import asyncio

import pytest

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint


class _SlowClient:
    async def complete(self, prompt, max_tokens=256, **kwargs):
        await asyncio.sleep(2.0)  # más que el timeout del endpoint
        return "tarde"


class _FastClient:
    async def complete(self, prompt, max_tokens=256, **kwargs):
        return "rápido"


@pytest.mark.asyncio
async def test_router_times_out_slow_endpoint_and_fails_over(tmp_path):
    cd = CooldownManager(persistence_path=tmp_path / "cd.json")
    eps = [
        LLMEndpoint(id="slow", kind=EndpointKind.CLOUD, client=_SlowClient(),
                    priority=1, timeout_s=0.1),
        LLMEndpoint(id="fast", kind=EndpointKind.HTTP_REASONER, client=_FastClient(),
                    priority=2, timeout_s=5.0),
    ]
    router = LLMRouter(endpoints=eps, cooldown_manager=cd)
    result = await router.complete("hola", max_tokens=16)
    assert result.text == "rápido"
    assert result.endpoint_id == "fast"  # el slow hizo timeout y rotó
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `pytest tests/unit/llm/test_error_classifier_network.py tests/unit/llm/test_router_timeout.py -v`
Expected: FAIL (`ssl`/`gaierror` → PERMANENT; el router no aplica timeout)

- [ ] **Step 3: Agregar el timeout en el router**

En `src/llm/router.py`, agregar `import asyncio` al tope (tras `import logging`), y reemplazar la línea 100:

```python
                text = await ep.client.complete(prompt, max_tokens=max_tokens, **kwargs)
```

por:

```python
                if ep.timeout_s and ep.timeout_s > 0:
                    text = await asyncio.wait_for(
                        ep.client.complete(prompt, max_tokens=max_tokens, **kwargs),
                        timeout=ep.timeout_s,
                    )
                else:
                    text = await ep.client.complete(prompt, max_tokens=max_tokens, **kwargs)
```

(`asyncio.TimeoutError` ya se clasifica como `ErrorKind.TIMEOUT` en `classify_error`, que es failover-worthy → el router rota.)

- [ ] **Step 4: Agregar clasificación de errores de red**

En `src/llm/error_classifier.py`, agregar imports al tope:

```python
import socket
import ssl
```

Agregar un bucket de patrones transitorios (tras `_FORMAT_PATTERNS`):

```python
# Patrones de red transitorios (failover-worthy): timeouts del SDK, 5xx, DNS, TLS.
_TRANSIENT_PATTERNS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "temporarily unavailable",
    "name resolution",
    "getaddrinfo",
    "ssl",
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)
```

En `classify_error`, tras el bloque de `isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))` (línea 75), agregar:

```python
    # Errores de red de stdlib (TLS handshake, DNS) → transitorios, rotar.
    if isinstance(exc, (ssl.SSLError, socket.gaierror)):
        return ErrorKind.TIMEOUT
```

Y dentro del matching por texto, **antes** del check de AUTH (para que "503" no matchee mal), agregar:

```python
    if any(p in msg for p in _TRANSIENT_PATTERNS):
        return ErrorKind.TIMEOUT
```

Colocarlo justo después del check de `_RATE_LIMIT_PATTERNS` y `_BILLING_PATTERNS`, antes de `_AUTH_PATTERNS`.

- [ ] **Step 5: Correr los tests y verificar que pasan**

Run: `pytest tests/unit/llm/test_error_classifier_network.py tests/unit/llm/test_router_timeout.py -v`
Expected: PASS

- [ ] **Step 6: Regresión — clasificación existente intacta**

Run: `pytest tests/unit/llm/ -v`
Expected: PASS (los patrones AUTH 401/403 y rate-limit siguen ganando para sus casos)

- [ ] **Step 7: Commit**

```bash
git add src/llm/router.py src/llm/error_classifier.py tests/unit/llm/test_router_timeout.py tests/unit/llm/test_error_classifier_network.py
git commit -m "feat(llm): enforce ep.timeout_s in router + classify network/TLS/5xx errors as failover-worthy"
```

---

### Task B3: Configurar el endpoint cloud y wirearlo en main.py

**Contexto:** Reemplazo total del reasoner slow path. El bloque `reasoner` apunta a MiniMax; el endpoint `reasoner_72b` se renombra `reasoner_cloud` en `failover.endpoints` (priority 1) y en el dict `clients` de `main.py:322`. GLM-Air local queda comentado para rollback. `idle_timeout_s` del cloud (~25s) fluye vía la instancia `HttpReasoner` (no via `ep.idle_timeout_s`, que sigue muerto — aceptable: la instancia lo maneja).

**Files:**
- Modify: `config/settings.yaml` (bloques `reasoner` líneas 247-256 y `llm.failover.endpoints` líneas 273-284)
- Modify: `src/main.py:260-265` (construcción de `HttpReasoner`) y `:320-322` (dict clients)
- Modify: `.env.example` (documentar `MINIMAX_API_KEY`)
- Test: `tests/unit/llm/test_main_cloud_client_wiring.py` (crear, test de construcción del cliente)

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/unit/llm/test_main_cloud_client_wiring.py`:

```python
from src.llm.reasoner import HttpReasoner


def test_httpreasoner_cloud_construction_from_config():
    """El reasoner cloud se construye con api_style=chat y api_key_env."""
    reasoner_cfg = {
        "mode": "http",
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "http_timeout": 60,
        "idle_timeout_s": 25.0,
        "api_style": "chat",
        "api_key_env": "MINIMAX_API_KEY",
    }
    r = HttpReasoner(
        base_url=reasoner_cfg["http_base_url"],
        model=reasoner_cfg["http_model"],
        timeout=reasoner_cfg["http_timeout"],
        idle_timeout_s=reasoner_cfg["idle_timeout_s"],
        api_style=reasoner_cfg.get("api_style", "completions"),
        api_key_env=reasoner_cfg.get("api_key_env"),
    )
    assert r.base_url == "https://api.minimax.io/v1"
    assert r.model == "MiniMax-M2.7-highspeed"
    assert r.api_style == "chat"
    assert r.api_key_env == "MINIMAX_API_KEY"
    assert r.idle_timeout_s == 25.0
```

- [ ] **Step 2: Correr el test y verificar que pasa o falla según corresponda**

Run: `pytest tests/unit/llm/test_main_cloud_client_wiring.py -v`
Expected: PASS si B2/B1 ya están mergeados (el `HttpReasoner` ya acepta esos params). Sirve como contrato de la config que B3 debe cumplir.

- [ ] **Step 3: Actualizar `config/settings.yaml` — bloque `reasoner`**

Reemplazar el bloque `reasoner:` (líneas 247-256) por:

```yaml
# LLM para Razonamiento — REEMPLAZO TOTAL 2026-05-29: GLM-4.5-Air local (:8200)
# → MiniMax-M2.7-highspeed en el cloud de MiniMax (OpenAI-compat).
# Riesgo aceptado: si cae internet/MiniMax, el slow path queda sin reasoner
# (el fast path de domótica sigue 100% local). Rollback: descomentar el bloque
# GLM-Air local de abajo y revertir failover.endpoints.reasoner_cloud.
reasoner:
  mode: "http"
  http_base_url: "https://api.minimax.io/v1"
  http_timeout: 60
  http_model: "MiniMax-M2.7-highspeed"
  api_style: "chat"            # MiniMax solo soporta /v1/chat/completions
  api_key_env: "MINIMAX_API_KEY"
  # Cloud cold puede tardar 10-20s al primer token → watchdog generoso.
  idle_timeout_s: 25.0
  # --- Rollback de emergencia (GLM-Air local :8200) ---
  # http_base_url: "http://127.0.0.1:8200/v1"
  # http_model: null
  # api_style: "completions"
  # api_key_env: null
  # idle_timeout_s: 8.0
```

- [ ] **Step 4: Actualizar `failover.endpoints` (reasoner_72b → reasoner_cloud)**

En `config/settings.yaml`, reemplazar el endpoint `reasoner_72b` (líneas 279-284) por:

```yaml
      - id: reasoner_cloud
        kind: http_reasoner
        priority: 1            # reemplazo total: primary del slow path
        timeout_s: 60.0        # B5: ahora SÍ se respeta (TLS + TTFT cloud)
        idle_timeout_s: 25.0
        max_tokens_default: 512
```

(El `fast_router_7b` queda igual, priority 1, kind `fast_router` — es otro rol/path.)

- [ ] **Step 5: Actualizar `main.py` — construcción del HttpReasoner y dict clients**

En `src/main.py`, en la construcción de `HttpReasoner` (líneas 260-265), agregar los params nuevos:

```python
        llm = HttpReasoner(
            base_url=reasoner_config.get("http_base_url", "http://127.0.0.1:8200/v1"),
            model=reasoner_config.get("http_model"),
            timeout=reasoner_config.get("http_timeout", 120),
            idle_timeout_s=reasoner_config.get("idle_timeout_s"),
            api_style=reasoner_config.get("api_style", "completions"),
            api_key_env=reasoner_config.get("api_key_env"),
            verify_ssl=reasoner_config.get("verify_ssl", True),
        )
```

En el dict `clients` (líneas 320-322), renombrar la key del reasoner:

```python
        clients = {"fast_router_7b": fast_router}
        if llm is not None:
            clients["reasoner_cloud"] = llm
```

(Esto matchea el nuevo `id: reasoner_cloud` de la config. La factory `build_llm_router` mapea por id.)

- [ ] **Step 6: Documentar la env var en `.env.example`**

Agregar a `.env.example` (sin valores reales):

```bash
# Reasoner cloud (MiniMax) — slow path. Cargar en /home/kza/secrets/.env (chmod 600).
MINIMAX_API_KEY=
```

- [ ] **Step 7: Correr tests y verificar arranque lógico**

Run: `pytest tests/unit/llm/ -v`
Expected: PASS. Verificar manualmente que `grep -n "reasoner_72b" config/settings.yaml src/main.py` no devuelve nada (rename completo).

- [ ] **Step 8: Commit**

```bash
git add config/settings.yaml src/main.py .env.example tests/unit/llm/test_main_cloud_client_wiring.py
git commit -m "feat(llm): point reasoner at MiniMax-M2.7-highspeed cloud (full replacement, rollback commented)"
```

---

### Task B6: Privacidad — consent gate, log de salida de datos, sanitización de secrets

**Contexto:** El reasoner cloud manda al tercero texto verbatim + historial + nombre + zona + estado del hogar. Rompe la premisa 100%-local. Mitigaciones: (1) consent gate (`reasoner.cloud.consent`) — sin él, NO se instancia el cliente cloud; (2) log prominente por request de slow path; (3) sanitizar query params de auth en logs.

**Files:**
- Modify: `src/main.py` (consent gate al construir el reasoner cloud)
- Modify: `src/llm/reasoner.py` (`complete` — log de salida de datos cuando `api_style="chat"` y base_url no es localhost)
- Modify: `src/core/logging.py` (sanitizar query params de auth)
- Modify: `config/settings.yaml` (bloque `reasoner.cloud.consent`)
- Test: `tests/unit/llm/test_cloud_consent.py`, `tests/safety/test_api_no_secret_exposure.py` (crear)

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/safety/test_api_no_secret_exposure.py`:

```python
from src.core.logging import StructuredFormatter


def test_sanitizes_bearer_token():
    out = StructuredFormatter._sanitize_message("auth Bearer sk-abcdef123456 done")
    assert "sk-abcdef123456" not in out
    assert "Bearer sk-a***" in out


def test_sanitizes_query_param_api_key():
    out = StructuredFormatter._sanitize_message(
        "GET https://api.minimax.io/v1/chat?api_key=mm-secret-999&x=1"
    )
    assert "mm-secret-999" not in out
    assert "api_key=" in out  # la clave del param se conserva, el valor no
```

Crear `tests/unit/llm/test_cloud_consent.py`:

```python
from src.llm.cloud_consent import cloud_reasoner_allowed


def test_cloud_blocked_without_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is False


def test_cloud_allowed_with_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": True}}
    assert cloud_reasoner_allowed(cfg) is True


def test_localhost_always_allowed():
    cfg = {"http_base_url": "http://127.0.0.1:8200/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is True  # local no requiere consent
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `pytest tests/safety/test_api_no_secret_exposure.py tests/unit/llm/test_cloud_consent.py -v`
Expected: FAIL (`test_sanitizes_query_param_api_key` falla; `src.llm.cloud_consent` no existe)

- [ ] **Step 3: Implementar el helper de consent**

Crear `src/llm/cloud_consent.py`:

```python
"""Gate de consentimiento para reasoners cloud (privacidad).

El reasoner cloud manda datos del usuario (transcripción, historial, estado del
hogar) a un tercero — rompe la premisa 100%-local. Requiere consent explícito
en config para activarse. Endpoints localhost no requieren consent.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def is_cloud_endpoint(base_url: str) -> bool:
    """True si base_url no es localhost (sale de la máquina)."""
    host = urlparse(base_url).hostname or ""
    return host not in _LOCAL_HOSTS


def cloud_reasoner_allowed(reasoner_config: dict) -> bool:
    """¿Está permitido instanciar este reasoner?

    - Endpoint local → siempre permitido.
    - Endpoint cloud → solo si reasoner.cloud.consent es True.
    """
    base_url = reasoner_config.get("http_base_url", "")
    if not is_cloud_endpoint(base_url):
        return True
    consent = bool(reasoner_config.get("cloud", {}).get("consent", False))
    if not consent:
        logger.warning(
            "Reasoner cloud %s NO instanciado: reasoner.cloud.consent=false. "
            "Activar consent para enviar datos del usuario al tercero.",
            base_url,
        )
    return consent
```

- [ ] **Step 4: Sanitizar query params de auth en logs**

En `src/core/logging.py`, agregar tras `_BEARER_RE` (línea 100):

```python
    # Mask de valores de query params de auth (?api_key=..., &token=..., &key=...)
    _QUERY_SECRET_RE = re.compile(
        r"((?:api_key|apikey|token|key|access_token)=)([^&\s]+)", re.IGNORECASE
    )
```

Y en `_sanitize_message` (línea 107), encadenar la segunda sustitución:

```python
    @classmethod
    def _sanitize_message(cls, message: str) -> str:
        """Remove secrets that may have leaked into a log message."""
        message = cls._BEARER_RE.sub(lambda m: f"Bearer {m.group(1)[:4]}***", message)
        message = cls._QUERY_SECRET_RE.sub(lambda m: f"{m.group(1)}***", message)
        return message
```

- [ ] **Step 5: Aplicar el consent gate en main.py**

En `src/main.py`, justo antes de construir el `HttpReasoner` (línea 259, dentro de `if reasoner_mode == "http":`), agregar:

```python
        from src.llm.cloud_consent import cloud_reasoner_allowed
        if not cloud_reasoner_allowed(reasoner_config):
            logger.warning(
                "Reasoner cloud bloqueado por falta de consent — slow path sin reasoner. "
                "Setear reasoner.cloud.consent=true en settings.yaml para activarlo."
            )
            llm = None
        else:
            llm = HttpReasoner(...)  # (la construcción del Step B3.5)
            try:
                llm.load()
                ...
```

(Envolver la construcción + `load()` existente en el `else`.)

- [ ] **Step 6: Log de salida de datos en `complete` (cloud)**

En `src/llm/reasoner.py`, al inicio de `HttpReasoner.complete` (tras `if self._client is None: self.load()`), agregar:

```python
        from src.llm.cloud_consent import is_cloud_endpoint
        if self.api_style == "chat" and is_cloud_endpoint(self.base_url):
            logger.info(
                "Cloud reasoning: enviando ~%d chars a %s (modelo=%s)",
                len(prompt), urlparse(self.base_url).hostname, self._resolved_model,
            )
```

(Importar `urlparse` ya está al tope del módulo.)

- [ ] **Step 7: Agregar el bloque consent a la config**

En `config/settings.yaml`, dentro del bloque `reasoner:` (B3), agregar:

```yaml
  # Privacidad: el reasoner cloud manda datos del usuario a MiniMax (tercero).
  # consent=false bloquea el cliente cloud (slow path sin reasoner). Cambiar a
  # true es una decisión informada de enviar transcripciones/contexto afuera.
  cloud:
    consent: true
    strip_home_state: false   # true = no enviar estado del hogar al cloud
```

- [ ] **Step 8: Correr los tests y verificar que pasan**

Run: `pytest tests/safety/test_api_no_secret_exposure.py tests/unit/llm/test_cloud_consent.py -v`
Expected: PASS

- [ ] **Step 9: Regresión completa de llm + safety + core**

Run: `pytest tests/unit/llm/ tests/safety/ tests/unit/core/ -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/llm/cloud_consent.py src/llm/reasoner.py src/core/logging.py src/main.py config/settings.yaml tests/unit/llm/test_cloud_consent.py tests/safety/test_api_no_secret_exposure.py
git commit -m "feat(privacy): cloud consent gate + data-egress log + query-param secret sanitization"
```

---

## Cierre

- [ ] **Suite completa**

Run: `pytest tests/`
Expected: los 617+ tests existentes + los nuevos pasan (sin regresiones).

- [ ] **Validación contra MiniMax real (pre-deploy)**

Con `MINIMAX_API_KEY` seteada, un script/REPL mínimo: instanciar `HttpReasoner(base_url="https://api.minimax.io/v1", model="MiniMax-M2.7-highspeed", api_style="chat", api_key_env="MINIMAX_API_KEY")`, `.load()`, `await .complete("Hola, ¿quién sos?")`. Verificar respuesta no vacía y que el log muestra el aviso de egress.

- [ ] **Deploy al server** (`kza@192.168.1.2`)

Seguir el patrón anti-drift: comparar `git hash-object` por archivo antes de pull, setear `MINIMAX_API_KEY` en `/home/kza/secrets/.env` (chmod 600) + `EnvironmentFile`, reiniciar `kza-voice.service`, verificar service activo y un comando de slow path real post-deploy. Confirmar que el fast path de domótica sigue respondiendo aunque MiniMax esté lento (no debe bloquear).

---

## Notas de scope / YAGNI

- **B4 (idle_timeout por endpoint):** NO se hace refactor de construcción dinámica iterando `failover.endpoints`. El `idle_timeout_s` del cloud fluye vía la instancia `HttpReasoner` (config `reasoner.idle_timeout_s` → `main.py:264` → atributo de instancia, ya usado en `complete`). El campo `LLMEndpoint.idle_timeout_s` queda muerto (no rompe nada; la instancia lo maneja). Si en el futuro hay >1 reasoner con idle_timeouts distintos en la misma chain, ahí sí conviene el refactor.
- **Cifrado at-rest de `./data/contexts/`:** diferido (follow-up). No bloquea esta entrega.
- **`strip_home_state`:** el flag se agrega a config en B6 pero su consumo (no enviar `home_state` al cloud en `context_manager.build_prompt`) queda como follow-up si el dueño lo activa; hoy default `false`.
