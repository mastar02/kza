"""Unit tests for src/stt/hybrid.py (STTEngine protocol + HybridVetoSTT)."""
from __future__ import annotations

import numpy as np
import pytest

from src.stt.hybrid import (
    HybridVetoSTT,
    STTEngine,
    apply_hybrid_veto,
    compute_transcript_agreement,
)
from src.stt.whisper_fast import STTResult


class _FakeSTT:
    def __init__(self, text: str = "", elapsed_ms: float = 12.0, *, error: Exception | None = None,
                 compression_ratio: float | None = 0.9):
        self.text = text
        self.elapsed_ms = elapsed_ms
        self.error = error
        self.compression_ratio = compression_ratio
        self.calls = 0
        self.loaded = False
        self.closed = False

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def transcribe_with_confidence(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return STTResult(
            text=self.text,
            elapsed_ms=self.elapsed_ms,
            no_speech_prob=0.01,
            avg_logprob=-0.3,
            compression_ratio=self.compression_ratio,
        )


def _audio() -> np.ndarray:
    return np.zeros(1600, dtype=np.float32)


def test_apply_hybrid_veto_drops_hallucination_when_gate_empty():
    primary = STTResult(text="¡Gracias por ver el video!", elapsed_ms=150.0, compression_ratio=1.2)
    gate = STTResult(text="   ", elapsed_ms=80.0)
    res = apply_hybrid_veto(primary, gate)
    assert res.vetoed is True
    assert res.text == ""
    assert res.primary_text == "¡Gracias por ver el video!"
    assert res.to_stt_result().text == ""
    assert res.to_stt_result().compression_ratio is None


def test_apply_hybrid_veto_keeps_primary_when_gate_detects_speech():
    primary = STTResult(text="Nexa prendé la luz del living", elapsed_ms=140.0, compression_ratio=0.85)
    gate = STTResult(text="Next up prende la luz", elapsed_ms=90.0)
    res = apply_hybrid_veto(primary, gate)
    assert res.vetoed is False
    assert res.text == "Nexa prendé la luz del living"
    projected = res.to_stt_result()
    assert projected.text == "Nexa prendé la luz del living"
    assert projected.compression_ratio == 0.85


def test_hybrid_veto_stt_gate_first_skips_primary_on_silence():
    primary = _FakeSTT("¡Gracias!")
    gate = _FakeSTT("")
    hybrid = HybridVetoSTT(primary=primary, gate=gate, gate_first=True)
    assert isinstance(hybrid, STTEngine)

    detailed = hybrid.transcribe_detailed(_audio())
    assert detailed.vetoed is True
    assert detailed.text == ""
    assert gate.calls == 1
    assert primary.calls == 0


def test_hybrid_veto_stt_gate_first_runs_primary_on_speech():
    primary = _FakeSTT("che apagá la luz de la cocina")
    gate = _FakeSTT("che apaga la luz")
    hybrid = HybridVetoSTT(primary=primary, gate=gate, gate_first=True)

    stt_result = hybrid.transcribe_with_confidence(_audio())
    assert stt_result.text == "che apagá la luz de la cocina"
    assert gate.calls == 1
    assert primary.calls == 1


def test_hybrid_veto_stt_gate_second_runs_both_and_vetoes():
    primary = _FakeSTT("Suscríbete")
    gate = _FakeSTT("")
    hybrid = HybridVetoSTT(primary=primary, gate=gate, gate_first=False)

    detailed = hybrid.transcribe_detailed(_audio())
    assert detailed.vetoed is True
    assert detailed.text == ""
    assert primary.calls == 1
    assert gate.calls == 1


def test_hybrid_veto_stt_fail_open_and_fail_closed():
    primary = _FakeSTT("prendé el aire")
    broken_gate = _FakeSTT(error=RuntimeError("onnx crash"))

    open_hybrid = HybridVetoSTT(primary=primary, gate=broken_gate, fail_open=True)
    text, _ms = open_hybrid.transcribe(_audio())
    assert text == "prendé el aire"

    closed_hybrid = HybridVetoSTT(primary=primary, gate=broken_gate, fail_open=False)
    with pytest.raises(RuntimeError, match="onnx crash"):
        closed_hybrid.transcribe(_audio())


def test_hybrid_veto_stt_lifecycle_ownership():
    primary = _FakeSTT("ok")
    gate = _FakeSTT("ok")

    borrowed = HybridVetoSTT(primary=primary, gate=gate, owns_engines=False)
    borrowed.load()
    borrowed.close()
    assert primary.loaded and gate.loaded
    assert not primary.closed and not gate.closed

    owned = HybridVetoSTT(primary=primary, gate=gate, owns_engines=True)
    owned.close()
    assert primary.closed and gate.closed


def test_hybrid_agreement_and_alt_text_flow_through_ambient_stt():
    import asyncio
    from src.ambient.ambient_stt import AmbientSTT

    assert compute_transcript_agreement("¿Cómo estás, che?", "como estas che") == 1.0
    assert compute_transcript_agreement("Hoy es maravilloso", "Hoy es para morirse") < 0.9

    primary = _FakeSTT("Nexa, prendé la luz. Hoy es un día maravilloso, ¿no?")
    gate = _FakeSTT("Nexa, aprender la luz. Hoy es un día maravilloso para morirse, ¿no?")
    hybrid = HybridVetoSTT(primary=primary, gate=gate, gate_first=True)

    stt_res = hybrid.transcribe_with_confidence(_audio())
    assert stt_res.alt_text == gate.text
    assert stt_res.agreement_ratio is not None and 0.5 < stt_res.agreement_ratio < 0.95

    ambient_stt = AmbientSTT(stt=hybrid, asr_col=0)
    ambient_res = asyncio.run(ambient_stt.transcribe(np.zeros((1600, 2), dtype=np.float32)))
    assert ambient_res.selected_channel == 0
    assert ambient_res.text == primary.text
    assert ambient_res.alt_text == gate.text
    assert ambient_res.agreement_ratio == stt_res.agreement_ratio


def test_stt_executor_owner_and_runtime_closure_support_hybrid_and_pool():
    import asyncio
    from src.application import STTExecutorOwner
    from src.memory.producer_erasure import _model_unknown
    from src.memory.runtime_closure import _native_closed, _supported_native
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT
    from src.stt.worker_pool import STTWorkerPool

    async def _run():
        pool = STTWorkerPool(
            [FastWhisperSTT("tiny", device="cpu"), FastWhisperSTT("tiny", device="cpu")],
            owns_workers=True,
        )
        gate = ParakeetSTT(model_name="nemo-parakeet-tdt-0.6b-v3", language="es")
        hybrid = HybridVetoSTT(primary=pool, gate=gate, owns_engines=True)

        owner = STTExecutorOwner()
        owner.add_model(hybrid)
        assert owner._capture_unreviewed is False
        assert _model_unknown(pool) is False
        assert _model_unknown(hybrid) is False
        assert _supported_native(pool) is True
        assert _supported_native(hybrid) is True

        await owner.aclose()
        assert _native_closed(pool) is True
        assert _native_closed(hybrid) is True
        assert _model_unknown(pool) is False
        assert _model_unknown(hybrid) is False

    asyncio.run(_run())

