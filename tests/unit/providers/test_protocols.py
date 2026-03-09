"""
Tests — Verify existing classes satisfy the provider protocols.

These tests ensure that the local in-process implementations (WhisperFast,
LLMReasoner, DualTTS, FastRouter) are structurally compatible with the
Protocol definitions. This guarantees the provider abstraction doesn't
break existing code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.providers.protocols import (
    LLMProvider,
    RouterProvider,
    STTProvider,
    TTSProvider,
)


# ============================================================================
# STT
# ============================================================================


class TestSTTProtocol:
    """Verify STT classes implement STTProvider."""

    def test_fast_whisper_satisfies_protocol(self):
        """FastWhisperSTT has load() and transcribe(audio, sample_rate)."""
        from src.stt.whisper_fast import FastWhisperSTT

        stt = FastWhisperSTT.__new__(FastWhisperSTT)  # no __init__ side-effects
        assert isinstance(stt, STTProvider)

    def test_moonshine_satisfies_protocol(self):
        """MoonshineSTT has load() and transcribe(audio, sample_rate)."""
        from src.stt.whisper_fast import MoonshineSTT

        stt = MoonshineSTT.__new__(MoonshineSTT)
        assert isinstance(stt, STTProvider)

    def test_protocol_method_signatures(self):
        """Protocol declares expected methods."""
        assert hasattr(STTProvider, "load")
        assert hasattr(STTProvider, "transcribe")


# ============================================================================
# LLM
# ============================================================================


class TestLLMProtocol:
    """Verify LLMReasoner implements LLMProvider."""

    def test_reasoner_satisfies_protocol(self):
        """LLMReasoner has load(), generate(), chat()."""
        from src.llm.reasoner import LLMReasoner

        reasoner = LLMReasoner.__new__(LLMReasoner)
        assert isinstance(reasoner, LLMProvider)

    def test_protocol_method_signatures(self):
        """Protocol declares expected methods."""
        assert hasattr(LLMProvider, "load")
        assert hasattr(LLMProvider, "generate")
        assert hasattr(LLMProvider, "chat")


# ============================================================================
# TTS
# ============================================================================


class TestTTSProtocol:
    """Verify TTS classes implement TTSProvider."""

    def test_piper_satisfies_protocol(self):
        """PiperTTS has sample_rate, load(), synthesize()."""
        from src.tts.piper_tts import PiperTTS

        tts = PiperTTS.__new__(PiperTTS)
        tts.sample_rate = 22050  # Set attribute for protocol check
        assert isinstance(tts, TTSProvider)

    def test_kokoro_satisfies_protocol(self):
        """KokoroTTS has sample_rate, load(), synthesize()."""
        from src.tts.piper_tts import KokoroTTS

        tts = KokoroTTS.__new__(KokoroTTS)
        tts.sample_rate = 24000
        assert isinstance(tts, TTSProvider)

    def test_dual_tts_satisfies_protocol(self):
        """DualTTS has sample_rate, load(), synthesize()."""
        from src.tts.piper_tts import DualTTS

        tts = DualTTS.__new__(DualTTS)
        tts.sample_rate = 24000
        assert isinstance(tts, TTSProvider)

    def test_protocol_method_signatures(self):
        """Protocol declares expected methods."""
        assert hasattr(TTSProvider, "load")
        assert hasattr(TTSProvider, "synthesize")


# ============================================================================
# Router
# ============================================================================


class TestRouterProtocol:
    """Verify FastRouter implements RouterProvider."""

    def test_fast_router_satisfies_protocol(self):
        """FastRouter has load(), classify(), classify_and_respond()."""
        from src.llm.reasoner import FastRouter

        router = FastRouter.__new__(FastRouter)
        assert isinstance(router, RouterProvider)

    def test_protocol_method_signatures(self):
        """Protocol declares expected methods."""
        assert hasattr(RouterProvider, "load")
        assert hasattr(RouterProvider, "classify")
        assert hasattr(RouterProvider, "classify_and_respond")


# ============================================================================
# DTO tests
# ============================================================================


class TestDTOs:
    """Verify dataclass DTOs."""

    def test_transcription_result(self):
        from src.providers.protocols import TranscriptionResult

        r = TranscriptionResult(text="hola", elapsed_ms=42.5)
        assert r.text == "hola"
        assert r.elapsed_ms == 42.5

    def test_generation_result(self):
        from src.providers.protocols import GenerationResult

        r = GenerationResult(text="respuesta")
        assert r.text == "respuesta"

    def test_synthesis_result(self):
        from src.providers.protocols import SynthesisResult

        audio = np.zeros(100, dtype=np.float32)
        r = SynthesisResult(audio=audio, elapsed_ms=30.0, engine="kokoro")
        assert r.engine == "kokoro"
        assert len(r.audio) == 100

    def test_classification_result(self):
        from src.providers.protocols import ClassificationResult

        r = ClassificationResult(label="domotics", needs_deep=False, response="ok")
        assert r.label == "domotics"
        assert not r.needs_deep
