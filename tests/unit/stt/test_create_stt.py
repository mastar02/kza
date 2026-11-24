"""Tests: create_stt — selector de engine del fast path.

Cubre la rama Parakeet (agregada 2026-07) sin descargar el modelo real:
onnx_asr se stubbea. La construcción de FastWhisperSTT es lazy (no carga el
modelo en el constructor), así que no toca GPU.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

from src.stt.hybrid import HybridVetoSTT
from src.stt.whisper_fast import FastWhisperSTT, create_stt


@pytest.fixture
def stub_onnx_asr(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "onnx_asr",
        types.SimpleNamespace(load_model=lambda name, **kw: MagicMock()),
    )


def test_create_stt_parakeet_by_engine(stub_onnx_asr):
    from src.stt.parakeet_stt import ParakeetSTT

    stt = create_stt({"engine": "parakeet", "language": "es"})
    assert isinstance(stt, ParakeetSTT)
    assert stt.language == "es"


def test_create_stt_parakeet_by_model_name(stub_onnx_asr):
    from src.stt.parakeet_stt import ParakeetSTT

    stt = create_stt({"model": "nemo-parakeet-tdt-0.6b-v3"})
    assert isinstance(stt, ParakeetSTT)


def test_create_stt_parakeet_custom_model(stub_onnx_asr):
    stt = create_stt({"engine": "parakeet", "parakeet_model": "otro-modelo"})
    assert stt.model_name == "otro-modelo"


def test_create_stt_default_is_whisper():
    stt = create_stt({"model": "./models/whisper-v3-turbo", "device": "cuda:1"})
    assert isinstance(stt, FastWhisperSTT)


def test_create_stt_hybrid(stub_onnx_asr):
    from src.stt.parakeet_stt import ParakeetSTT

    stt = create_stt({
        "engine": "hybrid",
        "model": "./models/whisper-v3-turbo",
        "device": "cuda:0",
        "language": "es",
    })
    assert isinstance(stt, HybridVetoSTT)
    assert isinstance(stt._primary, FastWhisperSTT)
    assert isinstance(stt._gate, ParakeetSTT)
    assert stt._primary.device == "cuda:0"


def test_build_ambient_stt_engine_supports_hybrid(stub_onnx_asr):
    from src.ambient.transcriber import _build_ambient_stt_engine

    stt = _build_ambient_stt_engine(
        stt_cfg={
            "engine": "hybrid",
            "model": "large-v3",
            "device": "cuda:0",
            "compute_type": "float16",
            "beam_size": 5,
            "vad_filter": True,
            "hotwords": "Nexa prendé escritorio",
        },
        stt_base_cfg={"language": "es", "compute_type": "int8_float16"},
    )
    assert isinstance(stt, HybridVetoSTT)
    assert stt._primary.device == "cuda:0"
    assert stt._primary.model_name == "large-v3"
    assert stt._primary.compute_type == "float16"
    assert stt._primary.beam_size == 5
    assert stt._primary.vad_filter is True
    assert stt._primary.hotwords == "Nexa prendé escritorio"


def test_fast_whisper_passes_hotwords_only_when_configured():
    import numpy as np

    captured = []

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.append(kwargs)
            return [], None

    stt_no_hw = FastWhisperSTT(model="large-v3", device="cpu")
    stt_no_hw._model = _FakeModel()
    stt_no_hw._transcribe_impl(np.zeros(1600, dtype=np.float32))
    assert "hotwords" not in captured[-1]

    stt_hw = FastWhisperSTT(model="large-v3", device="cpu", hotwords="Nexa prendé")
    stt_hw._model = _FakeModel()
    stt_hw._transcribe_impl(np.zeros(1600, dtype=np.float32))
    assert captured[-1]["hotwords"] == "Nexa prendé"
