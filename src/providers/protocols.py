"""
Provider Protocols — Interfaces for STT, LLM, TTS, and Router.

These protocols define the contract that both local (in-process) and
remote (HTTP worker) implementations must satisfy.  They match the
method signatures of the existing concrete classes so that the current
in-process objects already satisfy the protocol at runtime.

Note: The existing local classes use synchronous methods.  These
protocols therefore declare synchronous signatures to guarantee
structural compatibility with ``isinstance()`` / ``runtime_checkable``.
Remote implementations wrap their async HTTP calls with the same
sync-looking signature (returning coroutines where needed in future
async wrappers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# DTOs — shared return types
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """Result from STT transcription."""

    text: str
    elapsed_ms: float


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str


@dataclass
class SynthesisResult:
    """Result from TTS synthesis."""

    audio: np.ndarray
    elapsed_ms: float
    engine: str = ""


@dataclass
class ClassificationResult:
    """Result from Router classification."""

    label: str
    needs_deep: bool = False
    response: str = ""


# ---------------------------------------------------------------------------
# STT Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-Text provider interface.

    Matches ``FastWhisperSTT.transcribe`` / ``MoonshineSTT.transcribe``.
    """

    def load(self) -> None:
        """Load the model into memory / GPU."""
        ...

    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Transcribe audio to text.

        Args:
            audio: Audio data (float32 ndarray) or file path.
            sample_rate: Sample rate of the audio.

        Returns:
            ``(text, elapsed_ms)``
        """
        ...


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Large-language-model provider interface.

    Matches ``LLMReasoner.generate`` and ``LLMReasoner.chat``.
    """

    def load(self) -> None:
        """Load the model into memory."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a raw prompt.

        Returns:
            Generated text string.
        """
        ...

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Chat-style generation with message list.

        Returns:
            Assistant response text.
        """
        ...


# ---------------------------------------------------------------------------
# TTS Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class TTSProvider(Protocol):
    """Text-to-Speech provider interface.

    Matches ``PiperTTS.synthesize``, ``KokoroTTS.synthesize``.
    The ``DualTTS.synthesize`` returns an extra *engine* string, but
    ``tuple[ndarray, float, str]`` is a superset of
    ``tuple[ndarray, float]`` for duck-typing purposes, so DualTTS also
    satisfies this protocol structurally.
    """

    sample_rate: int

    def load(self) -> None:
        """Load the TTS model."""
        ...

    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, str]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            **kwargs: Engine-specific options (``voice``, ``force_quality``, etc.).

        Returns:
            ``(audio_ndarray, elapsed_ms)`` or ``(audio_ndarray, elapsed_ms, engine)``.
        """
        ...


# ---------------------------------------------------------------------------
# Router Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class RouterProvider(Protocol):
    """Fast Router provider interface.

    Matches ``FastRouter.classify`` and ``FastRouter.classify_and_respond``.
    """

    def load(self) -> None:
        """Load the router model."""
        ...

    def classify(self, text: str, options: list[str]) -> str:
        """Classify *text* into one of *options*.

        Returns:
            The chosen label string.
        """
        ...

    def classify_and_respond(
        self,
        text: str,
        context: str = "",
        max_tokens: int = 256,
    ) -> tuple[bool, str]:
        """Classify and optionally respond in a single inference.

        Returns:
            ``(needs_deep_reasoning, response)``
        """
        ...
