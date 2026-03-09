"""
Tests — Remote providers and factory.

Tests HTTP worker clients with mocked aiohttp responses, plus the
factory logic that selects local vs. remote based on config.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.providers.remote import (
    RemoteLLMProvider,
    RemoteProviderConfig,
    RemoteRouterProvider,
    RemoteSTTProvider,
    RemoteTTSProvider,
    _base64_to_ndarray,
    _ndarray_to_base64,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_config(url: str = "http://localhost:5001", **kwargs) -> RemoteProviderConfig:
    return RemoteProviderConfig(url=url, **kwargs)


def _make_audio_b64(length: int = 100) -> str:
    """Create a base64-encoded float32 ndarray."""
    audio = np.random.randn(length).astype(np.float32)
    return _ndarray_to_base64(audio)


class FakeResponse:
    """Minimal aiohttp response mock."""

    def __init__(self, status: int, body: dict | str):
        self.status = status
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        if isinstance(self._body, dict):
            return json.dumps(self._body)
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    """Minimal aiohttp.ClientSession mock."""

    def __init__(self, responses: list[FakeResponse] | None = None):
        self._responses = responses or []
        self._call_index = 0
        self.closed = False
        self._post_calls = []

    def post(self, url, *, json=None, data=None, timeout=None):
        self._post_calls.append({"url": url, "json": json, "data": data})
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return FakeResponse(200, {})

    async def close(self):
        self.closed = True


# ============================================================================
# Base64 encoding helpers
# ============================================================================


class TestBase64Helpers:
    """Test ndarray <-> base64 roundtrip."""

    def test_roundtrip(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = _ndarray_to_base64(original)
        decoded = _base64_to_ndarray(encoded)
        np.testing.assert_array_almost_equal(decoded, original)

    def test_large_array(self):
        original = np.random.randn(16000).astype(np.float32)
        encoded = _ndarray_to_base64(original)
        decoded = _base64_to_ndarray(encoded)
        np.testing.assert_array_almost_equal(decoded, original)


# ============================================================================
# Remote STT
# ============================================================================


class TestRemoteSTT:
    """Test RemoteSTTProvider with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        provider = RemoteSTTProvider(_make_config("http://stt-worker:5001"))
        session = FakeSession([
            FakeResponse(200, {"text": "enciende la luz", "elapsed_ms": 42.0})
        ])
        provider._session = session

        text, ms = await provider.transcribe_async(
            np.zeros(16000, dtype=np.float32)
        )
        assert text == "enciende la luz"
        assert ms == 42.0
        assert "transcribe" in session._post_calls[0]["url"]

    @pytest.mark.asyncio
    async def test_transcribe_sends_audio_b64(self):
        provider = RemoteSTTProvider(_make_config())
        session = FakeSession([
            FakeResponse(200, {"text": "ok", "elapsed_ms": 10.0})
        ])
        provider._session = session

        audio = np.ones(100, dtype=np.float32)
        await provider.transcribe_async(audio, sample_rate=8000)

        payload = session._post_calls[0]["json"]
        assert "audio_b64" in payload
        assert payload["sample_rate"] == 8000

    @pytest.mark.asyncio
    async def test_load_is_noop(self):
        provider = RemoteSTTProvider(_make_config())
        provider.load()  # Should not raise


# ============================================================================
# Remote LLM
# ============================================================================


class TestRemoteLLM:
    """Test RemoteLLMProvider with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        provider = RemoteLLMProvider(_make_config("http://llm-worker:5002"))
        session = FakeSession([
            FakeResponse(200, {"text": "La temperatura es 22 grados"})
        ])
        provider._session = session

        result = await provider.generate_async("¿Qué temperatura hace?")
        assert result == "La temperatura es 22 grados"

    @pytest.mark.asyncio
    async def test_chat_success(self):
        provider = RemoteLLMProvider(_make_config("http://llm-worker:5002"))
        session = FakeSession([
            FakeResponse(200, {"text": "Hola, soy KZA"})
        ])
        provider._session = session

        messages = [{"role": "user", "content": "Hola"}]
        result = await provider.chat_async(messages, system_prompt="Eres KZA")
        assert result == "Hola, soy KZA"

        payload = session._post_calls[0]["json"]
        assert payload["system_prompt"] == "Eres KZA"
        assert payload["messages"] == messages

    @pytest.mark.asyncio
    async def test_generate_sends_params(self):
        provider = RemoteLLMProvider(_make_config())
        session = FakeSession([
            FakeResponse(200, {"text": "result"})
        ])
        provider._session = session

        await provider.generate_async("test", max_tokens=256, temperature=0.1)
        payload = session._post_calls[0]["json"]
        assert payload["max_tokens"] == 256
        assert payload["temperature"] == 0.1


# ============================================================================
# Remote TTS
# ============================================================================


class TestRemoteTTS:
    """Test RemoteTTSProvider with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        provider = RemoteTTSProvider(_make_config("http://tts-worker:5003"))
        audio_b64 = _make_audio_b64(200)
        session = FakeSession([
            FakeResponse(200, {
                "audio_b64": audio_b64,
                "elapsed_ms": 35.0,
                "engine": "kokoro",
                "sample_rate": 24000,
            })
        ])
        provider._session = session

        audio, ms, engine = await provider.synthesize_async("Luz encendida")
        assert len(audio) == 200
        assert ms == 35.0
        assert engine == "kokoro"
        assert provider.sample_rate == 24000

    @pytest.mark.asyncio
    async def test_synthesize_with_voice(self):
        provider = RemoteTTSProvider(_make_config())
        session = FakeSession([
            FakeResponse(200, {
                "audio_b64": _make_audio_b64(50),
                "elapsed_ms": 10.0,
            })
        ])
        provider._session = session

        await provider.synthesize_async("test", voice="af_heart")
        payload = session._post_calls[0]["json"]
        assert payload["voice"] == "af_heart"

    @pytest.mark.asyncio
    async def test_sample_rate_default(self):
        provider = RemoteTTSProvider(_make_config())
        assert provider.sample_rate == 24000


# ============================================================================
# Remote Router
# ============================================================================


class TestRemoteRouter:
    """Test RemoteRouterProvider with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_classify_success(self):
        provider = RemoteRouterProvider(_make_config("http://router-worker:5004"))
        session = FakeSession([
            FakeResponse(200, {"label": "domotics"})
        ])
        provider._session = session

        result = await provider.classify_async(
            "enciende la luz", ["domotics", "conversation"]
        )
        assert result == "domotics"

    @pytest.mark.asyncio
    async def test_classify_and_respond_simple(self):
        provider = RemoteRouterProvider(_make_config())
        session = FakeSession([
            FakeResponse(200, {"needs_deep": False, "response": "Luz encendida"})
        ])
        provider._session = session

        needs_deep, response = await provider.classify_and_respond_async(
            "prende la luz del living"
        )
        assert not needs_deep
        assert response == "Luz encendida"

    @pytest.mark.asyncio
    async def test_classify_and_respond_deep(self):
        provider = RemoteRouterProvider(_make_config())
        session = FakeSession([
            FakeResponse(200, {"needs_deep": True, "response": ""})
        ])
        provider._session = session

        needs_deep, response = await provider.classify_and_respond_async(
            "explícame la teoría de la relatividad"
        )
        assert needs_deep
        assert response == ""


# ============================================================================
# Timeout and retry
# ============================================================================


class TestRetryBehavior:
    """Test that remote providers handle retries and timeouts."""

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Should retry when worker returns 500, then succeed on next attempt."""
        provider = RemoteSTTProvider(
            _make_config(max_retries=1, retry_delay=0.01)
        )
        session = FakeSession([
            FakeResponse(500, "Internal Server Error"),
            FakeResponse(200, {"text": "ok", "elapsed_ms": 5.0}),
        ])
        provider._session = session

        text, ms = await provider.transcribe_async(
            np.zeros(100, dtype=np.float32)
        )
        assert text == "ok"
        assert len(session._post_calls) == 2  # One failure + one success

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Should raise RuntimeError when all retries fail."""
        provider = RemoteSTTProvider(
            _make_config(max_retries=1, retry_delay=0.01)
        )
        session = FakeSession([
            FakeResponse(500, "fail"),
            FakeResponse(500, "fail again"),
        ])
        provider._session = session

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            await provider.transcribe_async(np.zeros(100, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        """With max_retries=0, should fail immediately on error."""
        provider = RemoteLLMProvider(
            _make_config(max_retries=0, retry_delay=0.01)
        )
        session = FakeSession([
            FakeResponse(503, "unavailable"),
        ])
        provider._session = session

        with pytest.raises(RuntimeError, match="failed after 1 attempts"):
            await provider.generate_async("test")


# ============================================================================
# Factory
# ============================================================================


class TestFactory:
    """Test the provider factory creates the right implementation."""

    def test_stt_local_default(self):
        """Default config creates a local STT (FastWhisperSTT)."""
        from src.providers.factory import create_stt_provider
        from src.stt.whisper_fast import FastWhisperSTT

        config = {"stt": {"model": "distil-whisper/distil-small.en"}}
        provider = create_stt_provider(config)
        assert isinstance(provider, FastWhisperSTT)

    def test_stt_remote(self):
        """Remote mode creates a RemoteSTTProvider."""
        from src.providers.factory import create_stt_provider

        config = {
            "providers": {
                "stt": {
                    "mode": "remote",
                    "url": "http://stt-worker:5001",
                }
            }
        }
        provider = create_stt_provider(config)
        assert isinstance(provider, RemoteSTTProvider)

    def test_llm_local_default(self):
        """Default config creates a local LLM (LLMReasoner)."""
        from src.providers.factory import create_llm_provider
        from src.llm.reasoner import LLMReasoner

        config = {"reasoner": {"model_path": "/tmp/fake.gguf"}}
        provider = create_llm_provider(config)
        assert isinstance(provider, LLMReasoner)

    def test_llm_remote(self):
        """Remote mode creates a RemoteLLMProvider."""
        from src.providers.factory import create_llm_provider

        config = {
            "providers": {
                "llm": {
                    "mode": "remote",
                    "url": "http://llm-worker:5002",
                }
            }
        }
        provider = create_llm_provider(config)
        assert isinstance(provider, RemoteLLMProvider)

    def test_tts_local_default(self):
        """Default config creates a local TTS (PiperTTS by default)."""
        from src.providers.factory import create_tts_provider
        from src.tts.piper_tts import PiperTTS

        config = {"tts": {"engine": "piper"}}
        provider = create_tts_provider(config)
        assert isinstance(provider, PiperTTS)

    def test_tts_remote(self):
        """Remote mode creates a RemoteTTSProvider."""
        from src.providers.factory import create_tts_provider

        config = {
            "providers": {
                "tts": {
                    "mode": "remote",
                    "url": "http://tts-worker:5003",
                }
            }
        }
        provider = create_tts_provider(config)
        assert isinstance(provider, RemoteTTSProvider)

    def test_router_local_default(self):
        """Default config creates a local Router (FastRouter)."""
        from src.providers.factory import create_router_provider
        from src.llm.reasoner import FastRouter

        config = {"router": {"model": "Qwen/Qwen2.5-7B-Instruct"}}
        provider = create_router_provider(config)
        assert isinstance(provider, FastRouter)

    def test_router_remote(self):
        """Remote mode creates a RemoteRouterProvider."""
        from src.providers.factory import create_router_provider

        config = {
            "providers": {
                "router": {
                    "mode": "remote",
                    "url": "http://router-worker:5004",
                }
            }
        }
        provider = create_router_provider(config)
        assert isinstance(provider, RemoteRouterProvider)

    def test_factory_passes_config_to_remote(self):
        """Remote provider receives timeout and retry settings."""
        from src.providers.factory import create_stt_provider

        config = {
            "providers": {
                "stt": {
                    "mode": "remote",
                    "url": "http://stt:5001",
                    "timeout": 60.0,
                    "max_retries": 5,
                    "retry_delay": 1.0,
                }
            }
        }
        provider = create_stt_provider(config)
        assert isinstance(provider, RemoteSTTProvider)
        assert provider._config.timeout == 60.0
        assert provider._config.max_retries == 5
        assert provider._config.retry_delay == 1.0


# ============================================================================
# Close / cleanup
# ============================================================================


class TestCleanup:
    """Test that remote providers can close their sessions."""

    @pytest.mark.asyncio
    async def test_stt_close(self):
        provider = RemoteSTTProvider(_make_config())
        session = FakeSession()
        provider._session = session
        await provider.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_llm_close(self):
        provider = RemoteLLMProvider(_make_config())
        session = FakeSession()
        provider._session = session
        await provider.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_tts_close(self):
        provider = RemoteTTSProvider(_make_config())
        session = FakeSession()
        provider._session = session
        await provider.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_router_close(self):
        provider = RemoteRouterProvider(_make_config())
        session = FakeSession()
        provider._session = session
        await provider.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_close_when_no_session(self):
        """Close should be safe when session was never created."""
        provider = RemoteSTTProvider(_make_config())
        await provider.close()  # Should not raise
