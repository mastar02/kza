"""Unified STT protocol and reusable HybridVetoSTT strategy.

Provides a minimal, engine-agnostic contract shared by the A/B comparison tool
and the runtime pipelines (command and ambient/continuous).

``HybridVetoSTT`` implements the empirically validated anti-hallucination
strategy: a frame-synchronous gate engine (Parakeet-TDT CPU) suppresses
non-speech hallucinations, while a forced-language primary engine (Whisper with
``language="es"``) transcribes confirmed speech with higher Rioplatense Spanish
fidelity.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


@runtime_checkable
class STTEngine(Protocol):
    """Minimal contract shared across STT backends (Whisper, Parakeet, Hybrid)."""

    def transcribe_with_confidence(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> STTResult:
        """Transcribe mono audio and return an ``STTResult``."""
        ...


@dataclass(frozen=True)
class HybridTranscriptionResult:
    """Detailed outcome of a hybrid gate + primary transcription."""

    text: str
    elapsed_ms: float
    primary_text: str
    gate_text: str
    vetoed: bool
    primary_ms: float
    gate_ms: float
    primary_result: STTResult | None = None
    gate_result: STTResult | None = None

    def to_stt_result(self) -> STTResult:
        """Project to the standard ``STTResult`` consumed by runtime gates."""
        if self.vetoed or self.primary_result is None:
            return STTResult(
                text=self.text,
                elapsed_ms=self.elapsed_ms,
                no_speech_prob=None,
                avg_logprob=None,
                compression_ratio=None,
            )
        return STTResult(
            text=self.text,
            elapsed_ms=self.elapsed_ms,
            no_speech_prob=self.primary_result.no_speech_prob,
            avg_logprob=self.primary_result.avg_logprob,
            compression_ratio=self.primary_result.compression_ratio,
        )


def apply_hybrid_veto(
    primary_result: STTResult,
    gate_result: STTResult | None,
) -> HybridTranscriptionResult:
    """Combine precomputed primary and gate outputs using the veto rule.

    When ``gate_result`` is available and its stripped text is empty, any
    non-empty ``primary_result`` is treated as a non-speech hallucination and
    vetoed to ``""``. When ``gate_result`` is ``None`` (fail-open), the primary
    result is kept as-is.

    Args:
        primary_result: Output of the forced-language primary engine (Whisper).
        gate_result: Output of the anti-hallucination gate engine (Parakeet),
            or ``None`` if the gate was unavailable.

    Returns:
        ``HybridTranscriptionResult`` with full provenance of both engines.
    """
    primary_text = (primary_result.text or "").strip()
    gate_text = "" if gate_result is None else (gate_result.text or "").strip()
    gate_ms = 0.0 if gate_result is None else float(gate_result.elapsed_ms)
    primary_ms = float(primary_result.elapsed_ms)
    vetoed = gate_result is not None and not gate_text and bool(primary_text)
    final_text = "" if vetoed else primary_text
    return HybridTranscriptionResult(
        text=final_text,
        elapsed_ms=primary_ms + gate_ms,
        primary_text=primary_text,
        gate_text=gate_text,
        vetoed=vetoed,
        primary_ms=primary_ms,
        gate_ms=gate_ms,
        primary_result=primary_result,
        gate_result=gate_result,
    )


class HybridVetoSTT:
    """Gate + primary STT strategy (e.g. Parakeet anti-hallucination + Whisper es).

    When ``gate_first=True`` (default), the lightweight CPU gate runs first:
    if it returns empty text, the primary GPU engine is skipped completely
    (zero hallucination and zero GPU time on silence/noise). If the gate
    detects speech (non-empty text), the primary engine transcribes the clip
    in forced Spanish.

    When ``gate_first=False``, both engines run unconditionally and
    :func:`apply_hybrid_veto` resolves the output.
    """

    def __init__(
        self,
        primary: STTEngine,
        gate: STTEngine,
        *,
        gate_first: bool = True,
        fail_open: bool = True,
        owns_engines: bool = False,
    ):
        self._primary = primary
        self._gate = gate
        self.gate_first = gate_first
        self.fail_open = fail_open
        self._owns_engines = owns_engines

    def load(self) -> None:
        """Warm/load underlying engines if they expose ``load()``."""
        for engine in (self._gate, self._primary):
            load_fn = getattr(engine, "load", None)
            if callable(load_fn):
                load_fn()

    def request_close(self) -> None:
        """Seal owned underlying engines without blocking."""
        if not self._owns_engines:
            return
        for engine in (self._gate, self._primary):
            fn = getattr(engine, "request_close", None)
            if callable(fn):
                fn()

    def begin_close(self):
        """Async retirement hook compatible with ``AmbientSTT`` and runtime closure."""
        import asyncio
        from contextvars import Context

        self.request_close()

        async def _finish() -> None:
            if not self._owns_engines:
                return
            pending = []
            for engine in (self._gate, self._primary):
                begin = getattr(engine, "begin_close", None)
                if callable(begin):
                    pending.append(begin())
                else:
                    close = getattr(engine, "close", None)
                    if callable(close):
                        pending.append(asyncio.to_thread(close))
            if pending:
                await asyncio.gather(*pending)

        return asyncio.get_running_loop().create_task(_finish(), context=Context())

    def close(self) -> None:
        """Close owned underlying engines cleanly."""
        if not self._owns_engines:
            return
        errors: list[Exception] = []
        for engine in (self._gate, self._primary):
            close_fn = getattr(engine, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]

    def _run_gate(self, audio: np.ndarray, sample_rate: int) -> STTResult | None:
        try:
            return self._gate.transcribe_with_confidence(audio, sample_rate)
        except Exception as error:
            if not self.fail_open:
                raise
            logger.warning("HybridVetoSTT gate failed (fail-open): %s", type(error).__name__)
            return None

    def transcribe_detailed(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> HybridTranscriptionResult:
        """Run the hybrid strategy and return both engine outputs and veto status."""
        t0 = time.perf_counter()
        if self.gate_first:
            gate_res = self._run_gate(audio, sample_rate)
            if gate_res is not None and not (gate_res.text or "").strip():
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return HybridTranscriptionResult(
                    text="",
                    elapsed_ms=elapsed_ms,
                    primary_text="",
                    gate_text="",
                    vetoed=True,
                    primary_ms=0.0,
                    gate_ms=float(gate_res.elapsed_ms),
                    primary_result=None,
                    gate_result=gate_res,
                )
            primary_res = self._primary.transcribe_with_confidence(audio, sample_rate)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            primary_text = (primary_res.text or "").strip()
            gate_text = "" if gate_res is None else (gate_res.text or "").strip()
            return HybridTranscriptionResult(
                text=primary_text,
                elapsed_ms=elapsed_ms,
                primary_text=primary_text,
                gate_text=gate_text,
                vetoed=False,
                primary_ms=float(primary_res.elapsed_ms),
                gate_ms=0.0 if gate_res is None else float(gate_res.elapsed_ms),
                primary_result=primary_res,
                gate_result=gate_res,
            )

        primary_res = self._primary.transcribe_with_confidence(audio, sample_rate)
        gate_res = self._run_gate(audio, sample_rate)
        combined = apply_hybrid_veto(primary_res, gate_res)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return HybridTranscriptionResult(
            text=combined.text,
            elapsed_ms=elapsed_ms,
            primary_text=combined.primary_text,
            gate_text=combined.gate_text,
            vetoed=combined.vetoed,
            primary_ms=combined.primary_ms,
            gate_ms=combined.gate_ms,
            primary_result=combined.primary_result,
            gate_result=combined.gate_result,
        )

    def transcribe_with_confidence(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> STTResult:
        """Standard ``STTEngine`` entrypoint returning ``STTResult``."""
        detailed = self.transcribe_detailed(audio, sample_rate)
        from contextvars import Context
        Context().run(logger.debug, "Hybrid STT completed in %.0fms", detailed.elapsed_ms)
        return detailed.to_stt_result()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[str, float]:
        """Compatibility tuple API ``(text, elapsed_ms)``."""
        result = self.transcribe_with_confidence(audio, sample_rate)
        return result.text, result.elapsed_ms
