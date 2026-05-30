# Command Acceptance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidar la aceptación de comandos en un gate dedicado que sume la confianza del STT (`no_speech_prob`/`avg_logprob`, hoy descartada) y reemplace el grab-bag de `_is_noise_text`, con rollout en shadow mode.

**Architecture:** El STT surfacea la confianza vía un nuevo `STTResult` + `transcribe_with_confidence()` (sin tocar `transcribe()`). `command_processor` la carga en `ProcessedCommand.stt_confidence`. Un nuevo `CommandAcceptanceGate` (en `src/nlu/command_gate.py`) evalúa hard rules (migradas de `_is_noise_text`, enforce) + confidence rules (shadow por default) y decide accept/reject. `request_router` lo invoca en el chokepoint; reject = descarte silencioso.

**Tech Stack:** Python 3.13, faster-whisper, pytest, dataclasses, DI por constructor.

**Entorno de tests (CRÍTICO):** usar SIEMPRE `venv/bin/python -m pytest` (el `python` del sistema es 3.9 y saltea tests falsamente). Correr por archivo/dir, no la suite completa.

---

### Task 1: STT surfacea la confianza (`STTResult` + `transcribe_with_confidence`)

**Files:**
- Modify: `src/stt/whisper_fast.py` (clase `FastWhisperSTT`, método `transcribe` ~líneas 71-135; clase `MoonshineSTT`)
- Test: `tests/unit/stt/test_whisper_fast_confidence.py` (nuevo)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/unit/stt/test_whisper_fast_confidence.py
"""Tests: surfacing de confianza del STT (no_speech_prob/avg_logprob)."""
import sys
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.stt.whisper_fast import FastWhisperSTT, STTResult


def _seg(text, no_speech_prob, avg_logprob):
    s = MagicMock()
    s.text = text
    s.no_speech_prob = no_speech_prob
    s.avg_logprob = avg_logprob
    return s


def _stt_with_segments(segments):
    stt = FastWhisperSTT(model="x", device="cpu")
    model = MagicMock()
    model.transcribe.return_value = (iter(segments), MagicMock())
    stt._model = model
    return stt


def test_with_confidence_aggregates_segments():
    stt = _stt_with_segments([_seg("hola ", 0.1, -0.3), _seg("mundo", 0.3, -0.5)])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert isinstance(r, STTResult)
    assert r.text == "hola mundo"
    assert r.no_speech_prob == pytest.approx(0.2)      # media
    assert r.avg_logprob == pytest.approx(-0.4)        # media


def test_with_confidence_empty_segments_returns_none():
    stt = _stt_with_segments([])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert r.text == ""
    assert r.no_speech_prob is None
    assert r.avg_logprob is None


def test_plain_transcribe_still_returns_2_tuple():
    stt = _stt_with_segments([_seg("hola", 0.1, -0.3)])
    out = stt.transcribe(np.zeros(16000, dtype="float32"))
    assert isinstance(out, tuple) and len(out) == 2
    text, ms = out
    assert text == "hola"
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv/bin/python -m pytest tests/unit/stt/test_whisper_fast_confidence.py -q`
Expected: FAIL con `ImportError: cannot import name 'STTResult'`.

- [ ] **Step 3: Implementar STTResult + refactor de transcribe**

En `src/stt/whisper_fast.py`, agregar el dataclass cerca del tope (después de los imports):

```python
from dataclasses import dataclass


@dataclass
class STTResult:
    """Resultado de transcripción con confianza del STT.

    no_speech_prob / avg_logprob son None cuando no hay segmentos (audio
    vacío) o el motor no los expone (Moonshine). El gate trata None como
    'sin penalizar'.
    """
    text: str
    elapsed_ms: float
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
```

Reemplazar el cuerpo de `transcribe` (desde el `segments, info = ...` hasta el `return text, elapsed_ms`) por un helper interno y dos métodos públicos. El cuerpo actual de `transcribe` (líneas ~110-135) pasa a `_transcribe_impl`:

```python
    def _transcribe_impl(self, audio_input) -> STTResult:
        """Transcribe y agrega la confianza de los segmentos."""
        start = time.time()
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 100,
                "threshold": 0.5,
            },
        )
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list)
        if seg_list:
            no_speech = sum(s.no_speech_prob for s in seg_list) / len(seg_list)
            avg_lp = sum(s.avg_logprob for s in seg_list) / len(seg_list)
        else:
            no_speech = None
            avg_lp = None
        elapsed_ms = (time.time() - start) * 1000
        logger.debug(f"STT ({elapsed_ms:.0f}ms): {text[:50]}...")
        return STTResult(text=text, elapsed_ms=elapsed_ms,
                         no_speech_prob=no_speech, avg_logprob=avg_lp)
```

Y reescribir `transcribe` para que prepare el audio (igual que hoy) y delegue:

```python
    def transcribe(self, audio, sample_rate: int = 16000) -> tuple[str, float]:
        """Transcribir audio a texto. Firma compat (text, elapsed_ms)."""
        r = self.transcribe_with_confidence(audio, sample_rate)
        return r.text, r.elapsed_ms

    def transcribe_with_confidence(self, audio, sample_rate: int = 16000) -> STTResult:
        """Transcribir devolviendo también la confianza del STT."""
        if self._model is None:
            self.load()
        if isinstance(audio, np.ndarray):
            audio_input = self._prepare_audio(audio)
        else:
            audio_input = str(audio)
        return self._transcribe_impl(audio_input)
```

(Mantener el resto del cuerpo actual de `transcribe` —la preparación de audio— dentro de `transcribe_with_confidence`; no duplicar la lógica de `_prepare_audio`.)

En `MoonshineSTT`, agregar para cumplir la interfaz:

```python
    def transcribe_with_confidence(self, audio, sample_rate: int = 16000) -> STTResult:
        text, ms = self.transcribe(audio, sample_rate)
        return STTResult(text=text, elapsed_ms=ms,
                         no_speech_prob=None, avg_logprob=None)
```

(Importar `STTResult` no hace falta si `MoonshineSTT` está en el mismo módulo; si está en otro, importarlo.)

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv/bin/python -m pytest tests/unit/stt/test_whisper_fast_confidence.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Regresión de los callers existentes de transcribe**

Run: `venv/bin/python -m pytest tests/unit/stt/ tests/unit/pipeline/ tests/unit/providers/ -q`
Expected: sin fallas nuevas (las pre-existentes conocidas siguen igual).

- [ ] **Step 6: Commit**

```bash
git add src/stt/whisper_fast.py tests/unit/stt/test_whisper_fast_confidence.py
git commit -m "feat(stt): surface STT confidence via STTResult + transcribe_with_confidence"
```

---

### Task 2: `ProcessedCommand.stt_confidence` + wiring en command_processor

**Files:**
- Modify: `src/pipeline/command_processor.py` (`ProcessedCommand` líneas 15-23; `process_command` ~líneas 146-159; `_process_parallel` ~líneas 175-224)
- Test: `tests/unit/pipeline/test_command_processor_confidence.py` (nuevo)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/unit/pipeline/test_command_processor_confidence.py
"""Tests: ProcessedCommand carga la confianza del STT."""
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult


class _FakeSTT:
    def transcribe(self, audio, sr=16000):
        return "prendé la luz", 5.0
    def transcribe_with_confidence(self, audio, sr=16000):
        return STTResult("prendé la luz", 5.0, no_speech_prob=0.15, avg_logprob=-0.4)


@pytest.mark.asyncio
async def test_process_command_populates_confidence_sequential():
    cp = CommandProcessor(stt=_FakeSTT(), speaker_identifier=None, user_manager=None)
    result = await cp.process_command(np.zeros(16000, dtype="float32"), use_parallel=False)
    assert result.stt_confidence is not None
    assert result.stt_confidence.no_speech_prob == pytest.approx(0.15)
    assert result.stt_confidence.avg_logprob == pytest.approx(-0.4)


@pytest.mark.asyncio
async def test_pretranscribed_leaves_confidence_none():
    cp = CommandProcessor(stt=_FakeSTT(), speaker_identifier=None, user_manager=None)
    result = await cp.process_command(
        np.zeros(16000, dtype="float32"), pretranscribed_text="prendé la luz",
    )
    assert result.stt_confidence is None
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv/bin/python -m pytest tests/unit/pipeline/test_command_processor_confidence.py -q`
Expected: FAIL con `AttributeError: 'ProcessedCommand' object has no attribute 'stt_confidence'`.

- [ ] **Step 3: Agregar el campo y poblarlo**

En `ProcessedCommand` (después de `speaker_confidence`):

```python
    stt_confidence: "STTResult | None" = None  # confianza del STT (None = desconocida)
```

Agregar el import arriba del archivo:

```python
from src.stt.whisper_fast import STTResult
```

En `process_command`, rama secuencial (línea ~147), reemplazar:

```python
            text, stt_ms = self.stt.transcribe(audio, self.sample_rate)
            result.timings["stt"] = stt_ms
```

por:

```python
            stt_res = self.stt.transcribe_with_confidence(audio, self.sample_rate)
            text = stt_res.text
            result.stt_confidence = stt_res
            result.timings["stt"] = stt_res.elapsed_ms
```

En `_process_parallel` (línea ~186), cambiar la task a:

```python
        stt_task = loop.run_in_executor(
            None, self.stt.transcribe_with_confidence, audio, self.sample_rate
        )
```

y el desempaquetado (líneas ~209-210):

```python
        stt_res = results[0] if not isinstance(results[0], Exception) else None
        if stt_res is not None:
            text, stt_ms = stt_res.text, stt_res.elapsed_ms
        else:
            text, stt_ms = "", 0
```

`_process_parallel` devuelve `(text, stt_ms, speaker_result, emotion_result)`; agregar `stt_res` al return y al unpack en `process_command` (la rama `if use_parallel:`), seteando `result.stt_confidence = stt_res`. Buscar el `return text, stt_ms, speaker_result, emotion_result` y el call site correspondiente; agregar el 5º valor.

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv/bin/python -m pytest tests/unit/pipeline/test_command_processor_confidence.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regresión command_processor + pipeline**

Run: `venv/bin/python -m pytest tests/unit/pipeline/ -q`
Expected: sin fallas nuevas.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/command_processor.py tests/unit/pipeline/test_command_processor_confidence.py
git commit -m "feat(pipeline): carry STT confidence in ProcessedCommand"
```

---

### Task 3: `CommandAcceptanceGate` — hard rules (migración de `_is_noise_text`)

**Files:**
- Create: `src/nlu/command_gate.py`
- Test: `tests/unit/nlu/test_command_gate.py` (nuevo)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/unit/nlu/test_command_gate.py
"""Tests: CommandAcceptanceGate."""
from src.nlu.command_gate import CommandAcceptanceGate, AcceptanceDecision


def _gate(**kw):
    return CommandAcceptanceGate(wake_words=("nexa", "alexa"), **kw)


def test_accepts_real_command():
    d = _gate().evaluate("nexa prendé la luz del escritorio")
    assert d.accept is True
    assert d.reason == "ok"


def test_rejects_empty():
    assert _gate().evaluate("").accept is False


def test_rejects_noise_phrase():
    d = _gate().evaluate("nexa suscribite al canal de youtube")
    assert d.accept is False
    assert "noise_phrase" in d.reason


def test_rejects_filler_word():
    assert _gate().evaluate("gracias").accept is False


def test_rejects_word_repetition():
    assert _gate().evaluate("nexa nexa nexa nexa").accept is False


def test_rejects_missing_wake():
    d = _gate().evaluate("prendé la luz del escritorio")
    assert d.accept is False
    assert "missing_wake" in d.reason
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.nlu.command_gate'`.

- [ ] **Step 3: Crear el gate con las hard rules**

```python
# src/nlu/command_gate.py
"""Command Acceptance Gate — decide si una captura post-wake es un comando real.

Consolida las heurísticas de noise/eco (antes en request_router._is_noise_text)
y suma la confianza del STT. Reject = descarte silencioso aguas arriba.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Frases que NO son comandos (TV/YouTube + eco típico del TTS). Match por
# substring sobre texto normalizado (lowercase, sin acentos).
_NOISE_PHRASES = (
    "suscribe", "suscrib", "campanita", "gracias por ver",
    "dale like", "dale lie", "dale mega like",
    "canal de youtube", "activa la",
    "luz encendida", "luz apagada", "luces encendidas", "luces apagadas",
    "hecho", "perfecto", "listo",
)
_FILLER_WORDS = {"gracias", "si", "no", "ok", "bueno", "dale"}


def _normalize(text: str) -> str:
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


@dataclass(frozen=True)
class AcceptanceDecision:
    accept: bool
    reason: str
    signals: dict = field(default_factory=dict)


class CommandAcceptanceGate:
    """Gate de aceptación de comandos. Hard rules (enforce) + confidence (shadow)."""

    def __init__(
        self,
        wake_words=(),
        enforce_confidence: bool = False,
        max_no_speech_prob: float = 0.60,
        min_avg_logprob: float = -1.20,
    ):
        self._wake_words = tuple(
            w.lower().strip() for w in wake_words if w and w.strip()
        )
        self._enforce_confidence = enforce_confidence
        self._max_no_speech_prob = max_no_speech_prob
        self._min_avg_logprob = min_avg_logprob

    def evaluate(self, text: str, stt_confidence=None) -> AcceptanceDecision:
        try:
            return self._evaluate(text, stt_confidence)
        except Exception as e:  # fail-open: nunca tumbar el control de voz
            logger.error(f"CommandAcceptanceGate error (fail-open accept): {e}")
            return AcceptanceDecision(True, "gate_error", {"error": str(e)})

    def _hard_reason(self, text: str) -> str | None:
        """Devuelve la regla hard que matchea, o None. Migrado de _is_noise_text."""
        if not text:
            return "empty"
        norm = _normalize(text)
        if not norm:
            return "empty_after_norm"
        for phrase in _NOISE_PHRASES:
            if phrase in norm:
                return f"noise_phrase:{phrase!r}"
        if norm in _FILLER_WORDS:
            return f"filler_word:{norm!r}"
        words = norm.split()
        if len(words) >= 4 and len(set(words)) == 1:
            return f"word_repetition:{words[0]!r}"
        if self._wake_words and not any(w in norm for w in self._wake_words):
            return f"missing_wake:{self._wake_words[0]!r}"
        return None

    def _evaluate(self, text: str, stt_confidence) -> AcceptanceDecision:
        hard = self._hard_reason(text)
        if hard is not None:
            return AcceptanceDecision(False, hard, {})
        return AcceptanceDecision(True, "ok", {})
```

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nlu/command_gate.py tests/unit/nlu/test_command_gate.py
git commit -m "feat(nlu): CommandAcceptanceGate with hard rules (migrated from _is_noise_text)"
```

---

### Task 4: Confidence rules + shadow mode + logging de calibración

**Files:**
- Modify: `src/nlu/command_gate.py` (`_evaluate`)
- Test: `tests/unit/nlu/test_command_gate.py` (agregar casos)

- [ ] **Step 1: Escribir el test que falla**

```python
# añadir a tests/unit/nlu/test_command_gate.py
from src.stt.whisper_fast import STTResult


def _conf(no_speech, logprob):
    return STTResult("x", 1.0, no_speech_prob=no_speech, avg_logprob=logprob)


def test_low_confidence_rejected_when_enforcing():
    g = _gate(enforce_confidence=True, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.9, -2.0))
    assert d.accept is False
    assert "low_confidence" in d.reason


def test_low_confidence_shadow_accepts_but_flags():
    g = _gate(enforce_confidence=False, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.9, -2.0))
    assert d.accept is True                       # shadow → pasa
    assert d.signals.get("would_reject")          # pero lo marca


def test_high_confidence_accepts():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("nexa prendé la luz", _conf(0.1, -0.3))
    assert d.accept is True


def test_none_confidence_not_penalized():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("nexa prendé la luz", None)
    assert d.accept is True


def test_hard_rule_wins_over_confidence():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("", _conf(0.9, -2.0))
    assert d.accept is False
    assert d.reason == "empty"


def test_fail_open_on_exception():
    g = _gate(enforce_confidence=True)
    # stt_confidence con atributos que rompen → fail-open accept
    class Boom:
        @property
        def no_speech_prob(self): raise ValueError("boom")
        avg_logprob = -0.1
    d = g.evaluate("nexa prendé la luz", Boom())
    assert d.accept is True
    assert d.reason == "gate_error"
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py -q`
Expected: FAIL (los nuevos casos de confidence fallan; `would_reject`/`low_confidence` no existen).

- [ ] **Step 3: Implementar confidence + shadow + logging**

Reemplazar `_evaluate` en `command_gate.py`:

```python
    def _confidence_reason(self, stt_confidence) -> tuple[str | None, dict]:
        """Evalúa las confidence rules. Devuelve (reason|None, signals)."""
        if stt_confidence is None:
            return None, {"no_speech_prob": None, "avg_logprob": None}
        nsp = stt_confidence.no_speech_prob
        alp = stt_confidence.avg_logprob
        signals = {"no_speech_prob": nsp, "avg_logprob": alp}
        bad = []
        if nsp is not None and nsp > self._max_no_speech_prob:
            bad.append(f"no_speech>{self._max_no_speech_prob}")
        if alp is not None and alp < self._min_avg_logprob:
            bad.append(f"avg_logprob<{self._min_avg_logprob}")
        reason = f"low_confidence:{','.join(bad)}" if bad else None
        return reason, signals

    def _evaluate(self, text: str, stt_confidence) -> AcceptanceDecision:
        hard = self._hard_reason(text)
        conf_reason, signals = self._confidence_reason(stt_confidence)

        if hard is not None:
            decision = AcceptanceDecision(False, hard, signals)
        elif conf_reason is not None and self._enforce_confidence:
            decision = AcceptanceDecision(False, conf_reason, signals)
        elif conf_reason is not None:
            # shadow: aceptamos pero marcamos qué rechazaríamos
            decision = AcceptanceDecision(
                True, "ok", {**signals, "would_reject": conf_reason}
            )
        else:
            decision = AcceptanceDecision(True, "ok", signals)

        logger.info(
            f"[CommandGate] accept={decision.accept} reason={decision.reason} "
            f"no_speech={signals.get('no_speech_prob')} "
            f"avg_logprob={signals.get('avg_logprob')} "
            f"would_reject={decision.signals.get('would_reject')} "
            f"text={text[:60]!r}"
        )
        return decision
```

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add src/nlu/command_gate.py tests/unit/nlu/test_command_gate.py
git commit -m "feat(nlu): confidence rules + shadow mode + calibration logging in gate"
```

---

### Task 5: Cablear el gate en `request_router` (reemplaza `_is_noise_text`)

**Files:**
- Modify: `src/pipeline/request_router.py` (`__init__` ~líneas 186-316; call sites `_is_noise_text` líneas 436 y 714; remover `_is_noise_text` + `_NOISE_PHRASES` líneas 55-113)
- Test: `tests/unit/pipeline/test_request_router_gate.py` (nuevo)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/unit/pipeline/test_request_router_gate.py
"""Integración: request_router usa CommandAcceptanceGate (accept/reject)."""
from unittest.mock import MagicMock, AsyncMock
import numpy as np
import pytest
from src.nlu.command_gate import CommandAcceptanceGate


def _router(gate):
    from src.pipeline.request_router import RequestRouter
    cmd = MagicMock()
    cmd.text = "nexa prendé la luz"
    cmd.user = None
    cmd.emotion = None
    cmd.timings = {}
    cmd.stt_confidence = None
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=cmd)
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None,
        path=None, timings={}, was_queued=False, queue_position=None,
    ))
    r = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=MagicMock(),
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        command_gate=gate,
    )
    return r, orch


@pytest.mark.asyncio
async def test_accepted_command_reaches_orchestrator():
    gate = CommandAcceptanceGate(wake_words=("nexa",))
    r, orch = _router(gate)
    await r.process_command(np.zeros(16000, dtype="float32"))
    assert orch.process.called


@pytest.mark.asyncio
async def test_rejected_command_does_not_reach_orchestrator():
    # Forzamos reject: gate que rechaza todo.
    gate = CommandAcceptanceGate(wake_words=("zzzz",))  # missing_wake siempre
    r, orch = _router(gate)
    result = await r.process_command(np.zeros(16000, dtype="float32"))
    assert not orch.process.called
    assert result["success"] is False
```

(Ajustar los kwargs de `RequestRouter(...)` a la firma real del `__init__`; los mostrados arriba son los mínimos del path orquestado — ver líneas 186-230 del archivo y completar los que sean obligatorios con `MagicMock()`.)

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `venv/bin/python -m pytest tests/unit/pipeline/test_request_router_gate.py -q`
Expected: FAIL (`RequestRouter` no acepta `command_gate`).

- [ ] **Step 3: Cablear el gate**

En `RequestRouter.__init__`, agregar el parámetro `command_gate=None` (junto a `llm_gate=None`, línea ~219). Después del bloque que setea `self._wake_words` (líneas ~297-302), agregar:

```python
        # Command acceptance gate (consolidación de _is_noise_text + confianza STT).
        # Si no se inyecta, default con hard rules sobre las wake words configuradas.
        if command_gate is not None:
            self.command_gate = command_gate
        else:
            from src.nlu.command_gate import CommandAcceptanceGate
            self.command_gate = CommandAcceptanceGate(wake_words=self._wake_words)
```

Reemplazar el call site de la línea ~436:

```python
        noise_reason = _is_noise_text(text, wake_words=self._wake_words)
        if noise_reason:
            logger.info(f"Noise discard ({noise_reason}): {text!r}")
            result["intent"] = "noise_discarded"
```

por:

```python
        gate_decision = self.command_gate.evaluate(
            text, stt_confidence=getattr(cmd, "stt_confidence", None)
        )
        if not gate_decision.accept:
            logger.info(f"Gate reject ({gate_decision.reason}): {text!r}")
            result["intent"] = "gate_rejected"
```

(Dejar el resto del bloque de early-return igual: `result["success"] = False`, `result["response"] = ""`, `result["latency_ms"] = ...`, `return result`.)

Hacer el mismo reemplazo en el segundo call site (línea ~714, path legacy): usar `self.command_gate.evaluate(text, stt_confidence=getattr(cmd, "stt_confidence", None))`.

Finalmente, **borrar** la función `_is_noise_text` (líneas ~73-113) y la constante `_NOISE_PHRASES` (líneas ~60-70) de `request_router.py` — ya viven en `command_gate.py`. Dejar `_texts_diverge` (se sigue usando).

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `venv/bin/python -m pytest tests/unit/pipeline/test_request_router_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regresión del router (los tests viejos que usaban _is_noise_text)**

Run: `venv/bin/python -m pytest tests/unit/pipeline/ -q`
Expected: sin fallas nuevas. Si algún test importaba `_is_noise_text` directamente, migrarlo a `from src.nlu.command_gate import CommandAcceptanceGate` o `_normalize` según corresponda.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/request_router.py tests/unit/pipeline/test_request_router_gate.py
git commit -m "feat(pipeline): wire CommandAcceptanceGate into request_router (replace _is_noise_text)"
```

---

### Task 6: Config + DI en main.py

**Files:**
- Modify: `config/settings.yaml` (bloque nuevo `command_gate`)
- Modify: `src/main.py` (construir el gate ~línea 1063, `RequestRouter(...)`)
- Test: `tests/unit/test_command_gate_config.py` (nuevo, smoke de construcción desde config)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/unit/test_command_gate_config.py
"""Smoke: el gate se construye desde un dict de config."""
from src.nlu.command_gate import CommandAcceptanceGate


def test_gate_from_config_dict():
    cfg = {"enforce_confidence": True, "max_no_speech_prob": 0.5, "min_avg_logprob": -1.0}
    g = CommandAcceptanceGate(wake_words=("nexa",), **cfg)
    assert g._enforce_confidence is True
    assert g._max_no_speech_prob == 0.5
    assert g._min_avg_logprob == -1.0
```

- [ ] **Step 2: Correr el test para verificar que falla/pasa**

Run: `venv/bin/python -m pytest tests/unit/test_command_gate_config.py -q`
Expected: PASS (el gate ya acepta esos kwargs desde Task 3-4). Si pasa, es la verificación de que la firma es config-friendly; seguir.

- [ ] **Step 3: Agregar el bloque de config**

En `config/settings.yaml`, agregar (cerca del bloque `nlu` o `stt`):

```yaml
command_gate:
  # Hard rules (noise phrases, filler, repetición, wake ausente) siempre enforce.
  # Confidence rules: shadow hasta calibrar con los logs [CommandGate].
  enforce_confidence: false
  max_no_speech_prob: 0.60   # > esto = sospechoso (conservador, a calibrar)
  min_avg_logprob: -1.20     # < esto = sospechoso (conservador, a calibrar)
```

- [ ] **Step 4: Construir e inyectar en main.py**

Antes de `request_router = RequestRouter(` (línea ~1063), agregar:

```python
    _gate_cfg = config.get("command_gate", {})
    command_gate = CommandAcceptanceGate(
        wake_words=_wake_words_tuple,  # las mismas wake words que usa el router
        enforce_confidence=_gate_cfg.get("enforce_confidence", False),
        max_no_speech_prob=_gate_cfg.get("max_no_speech_prob", 0.60),
        min_avg_logprob=_gate_cfg.get("min_avg_logprob", -1.20),
    )
```

Agregar el import arriba: `from src.nlu.command_gate import CommandAcceptanceGate`.
Pasar `command_gate=command_gate` en la llamada `RequestRouter(...)`.
(`_wake_words_tuple`: usar la variable de wake words que ya se le pasa a `RequestRouter` como `wake_words=`; si no existe una variable previa, derivarla de `config["wake_word"]` igual que el router. Verificar líneas 1063-1090.)

- [ ] **Step 5: Correr el smoke + arranque dry**

Run: `venv/bin/python -m pytest tests/unit/test_command_gate_config.py -q`
Expected: PASS.
Run: `venv/bin/python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"`
Expected: sin error (YAML válido).

- [ ] **Step 6: Commit**

```bash
git add config/settings.yaml src/main.py tests/unit/test_command_gate_config.py
git commit -m "feat(config): command_gate config block + DI in main.py"
```

---

### Task 7: Verificación end-to-end + deploy

**Files:** ninguno nuevo (validación).

- [ ] **Step 1: Suite completa de los módulos tocados**

Run: `venv/bin/python -m pytest tests/unit/stt/ tests/unit/nlu/ tests/unit/pipeline/ -q`
Expected: todo verde salvo fallas pre-existentes conocidas (`test_dispatcher.py` x3, `test_endpointing` x1, `test_search_command_prefer_area` x5). Confirmar que NO hay fallas nuevas.

- [ ] **Step 2: Merge a main + push**

```bash
git checkout main && git merge --ff-only feat/command-acceptance-gate && git push origin main
```

- [ ] **Step 3: Deploy al server (con cuidado de VRAM — ver project_stt_double_load_oom)**

```bash
ssh kza 'cd /home/kza/app && git pull --ff-only origin main && nvidia-smi --query-gpu=index,memory.free --format=csv,noheader && systemctl --user restart kza-voice.service && sleep 35 && systemctl --user show kza-voice.service -p ActiveState -p NRestarts'
```
Expected: `ActiveState=active`, `NRestarts=0`, preflight VRAM OK.

- [ ] **Step 4: Verificar el gate en producción (shadow)**

```bash
ssh kza 'journalctl --user -u kza-voice.service --no-pager -S "2 minutes ago" | grep "\[CommandGate\]" | tail -10'
```
Expected: líneas `[CommandGate] accept=... no_speech=... avg_logprob=... would_reject=...` apareciendo en comandos reales. Recolectar unos días para calibrar `max_no_speech_prob`/`min_avg_logprob` antes de flippear `enforce_confidence: true`.

---

## Notas de calibración (post-deploy)

Con varios días de logs `[CommandGate]`:
1. `grep "\[CommandGate\]" | grep "accept=True reason=ok"` → distribución de `no_speech`/`avg_logprob` de comandos REALES (no deben cruzar los umbrales).
2. `grep "would_reject"` → qué se habría rechazado en shadow; revisar manualmente si eran fantasmas o comandos reales.
3. Ajustar umbrales para que los reales queden holgados y los fantasmas caigan. Flippear `enforce_confidence: true`.
