"""
Provider Factory — Creates the right provider based on configuration.

Reads ``providers.<service>.mode`` from settings.yaml:
- ``"local"`` (default): returns the existing in-process implementation
- ``"remote"``: returns a ``Remote*Provider`` that calls an HTTP worker

Usage::

    from src.providers.factory import create_stt_provider

    stt = create_stt_provider(config)
    text, ms = stt.transcribe(audio)
"""

from __future__ import annotations

import logging

from src.providers.protocols import (
    LLMProvider,
    RouterProvider,
    STTProvider,
    TTSProvider,
)
from src.providers.remote import (
    RemoteLLMProvider,
    RemoteProviderConfig,
    RemoteRouterProvider,
    RemoteSTTProvider,
    RemoteTTSProvider,
)

logger = logging.getLogger(__name__)


def _remote_config(provider_cfg: dict) -> RemoteProviderConfig:
    """Build a ``RemoteProviderConfig`` from a provider config dict."""
    return RemoteProviderConfig(
        url=provider_cfg["url"],
        timeout=provider_cfg.get("timeout", 30.0),
        max_retries=provider_cfg.get("max_retries", 2),
        retry_delay=provider_cfg.get("retry_delay", 0.5),
    )


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def create_stt_provider(config: dict) -> STTProvider:
    """Create an STT provider based on configuration.

    Args:
        config: Full application config dict (parsed settings.yaml).

    Returns:
        An object satisfying ``STTProvider``.
    """
    provider_cfg = config.get("providers", {}).get("stt", {})
    mode = provider_cfg.get("mode", "local")

    if mode == "remote":
        logger.info("STT provider: remote (%s)", provider_cfg.get("url"))
        return RemoteSTTProvider(_remote_config(provider_cfg))

    # Default: local — use existing FastWhisperSTT / create_stt factory
    logger.info("STT provider: local (in-process)")
    from src.stt.whisper_fast import create_stt as _create_local_stt

    stt_cfg = config.get("stt", {})
    return _create_local_stt(stt_cfg)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def create_llm_provider(config: dict) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: Full application config dict.

    Returns:
        An object satisfying ``LLMProvider``.
    """
    provider_cfg = config.get("providers", {}).get("llm", {})
    mode = provider_cfg.get("mode", "local")

    if mode == "remote":
        logger.info("LLM provider: remote (%s)", provider_cfg.get("url"))
        return RemoteLLMProvider(_remote_config(provider_cfg))

    logger.info("LLM provider: local (in-process)")
    from src.llm.reasoner import LLMReasoner

    reasoner_cfg = config.get("reasoner", {})
    return LLMReasoner(
        model_path=reasoner_cfg.get("model_path", ""),
        lora_path=reasoner_cfg.get("lora_path"),
        lora_scale=reasoner_cfg.get("lora_scale", 1.0),
        n_ctx=reasoner_cfg.get("n_ctx", 32768),
        n_threads=reasoner_cfg.get("n_threads", 24),
        n_batch=reasoner_cfg.get("n_batch", 512),
        n_gpu_layers=reasoner_cfg.get("n_gpu_layers", 0),
        chat_format=reasoner_cfg.get("chat_format", "llama-3"),
        rope_freq_base=reasoner_cfg.get("rope_freq_base", 500000.0),
        rope_freq_scale=reasoner_cfg.get("rope_freq_scale", 1.0),
    )


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def create_tts_provider(config: dict) -> TTSProvider:
    """Create a TTS provider based on configuration.

    Args:
        config: Full application config dict.

    Returns:
        An object satisfying ``TTSProvider``.
    """
    provider_cfg = config.get("providers", {}).get("tts", {})
    mode = provider_cfg.get("mode", "local")

    if mode == "remote":
        logger.info("TTS provider: remote (%s)", provider_cfg.get("url"))
        return RemoteTTSProvider(_remote_config(provider_cfg))

    logger.info("TTS provider: local (in-process)")
    from src.tts.piper_tts import create_tts as _create_local_tts

    tts_cfg = config.get("tts", {})
    return _create_local_tts(tts_cfg)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def create_router_provider(config: dict) -> RouterProvider:
    """Create a Router provider based on configuration.

    Args:
        config: Full application config dict.

    Returns:
        An object satisfying ``RouterProvider``.
    """
    provider_cfg = config.get("providers", {}).get("router", {})
    mode = provider_cfg.get("mode", "local")

    if mode == "remote":
        logger.info("Router provider: remote (%s)", provider_cfg.get("url"))
        return RemoteRouterProvider(_remote_config(provider_cfg))

    logger.info("Router provider: HTTP (vLLM compartido)")
    from src.llm.reasoner import FastRouter

    router_cfg = config.get("router", {})
    return FastRouter(
        base_url=router_cfg.get("base_url", "http://127.0.0.1:8100/v1"),
        model=router_cfg.get("model", "qwen2.5-7b-awq"),
        timeout=router_cfg.get("timeout", 30),
    )
