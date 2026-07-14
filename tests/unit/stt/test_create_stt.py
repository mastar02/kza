"""Tests: create_stt — selector de engine del fast path.

Cubre la rama Parakeet (agregada 2026-07) sin descargar el modelo real:
onnx_asr se stubbea. La construcción de FastWhisperSTT es lazy (no carga el
modelo en el constructor), así que no toca GPU.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

from src.stt.whisper_fast import FastWhisperSTT, MoonshineSTT, create_stt


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


def test_create_stt_moonshine():
    stt = create_stt({"model": "moonshine/base"})
    assert isinstance(stt, MoonshineSTT)
