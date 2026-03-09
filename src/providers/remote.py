"""
Remote Provider Implementations — HTTP worker clients.

Each class calls a remote HTTP worker that runs the heavy model
on a separate process or machine.  All use aiohttp with configurable
base_url, timeout, and retry logic.

These are used only when ``providers.<service>.mode`` is set to
``"remote"`` in ``config/settings.yaml``.  The default is always
``"local"`` (current in-process behavior).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Default retry / timeout settings
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_DELAY_S = 0.5


@dataclass
class RemoteProviderConfig:
    """Shared configuration for all remote providers."""

    url: str
    timeout: float = _DEFAULT_TIMEOUT_S
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_delay: float = _DEFAULT_RETRY_DELAY_S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _post_with_retry(
    session,
    url: str,
    *,
    json: dict | None = None,
    data=None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY_S,
) -> dict:
    """POST with exponential-backoff retry.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    import aiohttp

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 2):  # +2 because range is exclusive and attempt 1 is the first try
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.post(
                url, json=json, data=data, timeout=client_timeout
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                body = await resp.text()
                raise RuntimeError(
                    f"Remote worker returned {resp.status}: {body}"
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            last_exc = exc
            if attempt <= max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Remote call to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    url, attempt, max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                break

    raise RuntimeError(
        f"Remote call to {url} failed after {max_retries + 1} attempts"
    ) from last_exc


def _ndarray_to_base64(audio: np.ndarray) -> str:
    """Encode a float32 ndarray as base64 for JSON transport."""
    buf = io.BytesIO()
    np.save(buf, audio.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _base64_to_ndarray(b64: str) -> np.ndarray:
    """Decode a base64-encoded ndarray."""
    buf = io.BytesIO(base64.b64decode(b64))
    return np.load(buf)


# ---------------------------------------------------------------------------
# Remote STT
# ---------------------------------------------------------------------------

class RemoteSTTProvider:
    """STT provider that delegates to a remote HTTP worker.

    Expected worker endpoint: ``POST /transcribe``
    Request body (JSON)::

        {
            "audio_b64": "<base64 encoded float32 ndarray>",
            "sample_rate": 16000
        }

    Response (JSON)::

        {"text": "...", "elapsed_ms": 123.4}
    """

    def __init__(self, config: RemoteProviderConfig):
        self._config = config
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """No-op for remote provider — model lives on the worker."""
        logger.info("RemoteSTTProvider ready (worker at %s)", self._config.url)

    def transcribe(
        self,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Synchronous facade — delegates to async implementation.

        When called from an async context, prefer ``transcribe_async``.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync transcribe from running event loop. "
                "Use transcribe_async instead."
            )
        return loop.run_until_complete(
            self.transcribe_async(audio, sample_rate)
        )

    async def transcribe_async(
        self,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """Transcribe audio via the remote worker."""
        if isinstance(audio, np.ndarray):
            audio_b64 = _ndarray_to_base64(audio)
        else:
            # Read file and encode
            audio_data = np.fromfile(str(audio), dtype=np.float32)
            audio_b64 = _ndarray_to_base64(audio_data)

        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/transcribe"
        result = await _post_with_retry(
            session,
            url,
            json={"audio_b64": audio_b64, "sample_rate": sample_rate},
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )
        return result["text"], result["elapsed_ms"]

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# Remote LLM
# ---------------------------------------------------------------------------

class RemoteLLMProvider:
    """LLM provider that delegates to a remote HTTP worker.

    Expected worker endpoints:

    ``POST /generate``::

        Request:  {"prompt": "...", "max_tokens": 1024, "temperature": 0.7}
        Response: {"text": "..."}

    ``POST /chat``::

        Request:  {"messages": [...], "max_tokens": 1024, "temperature": 0.7,
                   "system_prompt": null}
        Response: {"text": "..."}
    """

    def __init__(self, config: RemoteProviderConfig):
        self._config = config
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """No-op for remote provider."""
        logger.info("RemoteLLMProvider ready (worker at %s)", self._config.url)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous facade — delegates to async implementation."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync generate from running event loop. "
                "Use generate_async instead."
            )
        return loop.run_until_complete(
            self.generate_async(prompt, max_tokens, temperature)
        )

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text via the remote worker."""
        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/generate"
        result = await _post_with_retry(
            session,
            url,
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )
        return result["text"]

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Synchronous facade — delegates to async implementation."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync chat from running event loop. "
                "Use chat_async instead."
            )
        return loop.run_until_complete(
            self.chat_async(messages, max_tokens, temperature, system_prompt)
        )

    async def chat_async(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Chat via the remote worker."""
        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/chat"
        result = await _post_with_retry(
            session,
            url,
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
            },
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )
        return result["text"]

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# Remote TTS
# ---------------------------------------------------------------------------

class RemoteTTSProvider:
    """TTS provider that delegates to a remote HTTP worker.

    Expected worker endpoint: ``POST /synthesize``::

        Request:  {"text": "...", "voice": null}
        Response: {"audio_b64": "<base64 ndarray>", "elapsed_ms": 45.0,
                   "sample_rate": 24000, "engine": "kokoro"}
    """

    def __init__(self, config: RemoteProviderConfig):
        self._config = config
        self._session = None
        self.sample_rate: int = 24000  # Updated from worker response

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """No-op for remote provider."""
        logger.info("RemoteTTSProvider ready (worker at %s)", self._config.url)

    def synthesize(
        self, text: str, **kwargs
    ) -> tuple[np.ndarray, float, str]:
        """Synchronous facade — delegates to async implementation."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync synthesize from running event loop. "
                "Use synthesize_async instead."
            )
        return loop.run_until_complete(self.synthesize_async(text, **kwargs))

    async def synthesize_async(
        self, text: str, **kwargs
    ) -> tuple[np.ndarray, float, str]:
        """Synthesize text via the remote worker."""
        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/synthesize"
        payload: dict = {"text": text}
        if "voice" in kwargs:
            payload["voice"] = kwargs["voice"]
        if "force_quality" in kwargs:
            payload["force_quality"] = kwargs["force_quality"]

        result = await _post_with_retry(
            session,
            url,
            json=payload,
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )

        audio = _base64_to_ndarray(result["audio_b64"])
        elapsed_ms = result.get("elapsed_ms", 0.0)
        engine = result.get("engine", "remote")

        # Keep sample_rate in sync with worker
        if "sample_rate" in result:
            self.sample_rate = result["sample_rate"]

        return audio, elapsed_ms, engine

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# Remote Router
# ---------------------------------------------------------------------------

class RemoteRouterProvider:
    """Router provider that delegates to a remote HTTP worker.

    Expected worker endpoints:

    ``POST /classify``::

        Request:  {"text": "...", "options": ["opt1", "opt2"]}
        Response: {"label": "opt1"}

    ``POST /classify_and_respond``::

        Request:  {"text": "...", "context": "", "max_tokens": 256}
        Response: {"needs_deep": false, "response": "Luz encendida"}
    """

    def __init__(self, config: RemoteProviderConfig):
        self._config = config
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """No-op for remote provider."""
        logger.info("RemoteRouterProvider ready (worker at %s)", self._config.url)

    def classify(self, text: str, options: list[str]) -> str:
        """Synchronous facade."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync classify from running event loop. "
                "Use classify_async instead."
            )
        return loop.run_until_complete(self.classify_async(text, options))

    async def classify_async(self, text: str, options: list[str]) -> str:
        """Classify via the remote worker."""
        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/classify"
        result = await _post_with_retry(
            session,
            url,
            json={"text": text, "options": options},
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )
        return result["label"]

    def classify_and_respond(
        self,
        text: str,
        context: str = "",
        max_tokens: int = 256,
    ) -> tuple[bool, str]:
        """Synchronous facade."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot call sync classify_and_respond from running event loop. "
                "Use classify_and_respond_async instead."
            )
        return loop.run_until_complete(
            self.classify_and_respond_async(text, context, max_tokens)
        )

    async def classify_and_respond_async(
        self,
        text: str,
        context: str = "",
        max_tokens: int = 256,
    ) -> tuple[bool, str]:
        """Classify and respond via the remote worker."""
        session = await self._get_session()
        url = f"{self._config.url.rstrip('/')}/classify_and_respond"
        result = await _post_with_retry(
            session,
            url,
            json={
                "text": text,
                "context": context,
                "max_tokens": max_tokens,
            },
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            retry_delay=self._config.retry_delay,
        )
        return result["needs_deep"], result["response"]

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
