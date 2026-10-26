"""Bounded native HTTP/SSE generation without SDK type initialization."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json

import httpx


async def finish_owned_close(closing: asyncio.Task) -> None:
    """Retain actual resource close through repeated caller cancellation."""
    cancelled = False
    while not closing.done():
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            cancelled = True
    closing.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def owned_resource(resource):
    """Close response/client resources without detaching their cleanup."""
    try:
        yield resource
    finally:
        await finish_owned_close(asyncio.create_task(resource.close()))


class OwnedHTTPBody(httpx.AsyncByteStream):
    """Protect the real byte-stream close behind HTTPX's early closed flag."""

    def __init__(self, stream: httpx.AsyncByteStream):
        self._stream = stream
        self._closing: asyncio.Task | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._stream.aclose())
        await finish_owned_close(self._closing)


class SSEDecoder:
    """Bound each event and accept LF, CRLF and CR across transport chunks."""

    def __init__(self, limit: int):
        self.limit = limit
        self._line = bytearray()
        self._data: list[bytes] = []
        self._size = 0
        self._cr = False
        self._first = True

    def feed(self, chunk: bytes) -> list[bytes]:
        packets = []
        for byte in chunk:
            if self._cr and byte == 10:
                self._cr = False
                continue
            self._cr = False
            self._size += 1
            if self._size > self.limit:
                raise ValueError("generation_event_limit")
            if byte not in (10, 13):
                self._line.append(byte)
                continue
            self._cr = byte == 13
            line = bytes(self._line)
            self._line.clear()
            if self._first:
                line = line.removeprefix(b"\xef\xbb\xbf")
                self._first = False
            if not line:
                if self._data:
                    packets.append(b"\n".join(self._data))
                    self._data.clear()
                self._size = 0
            elif not line.startswith(b":"):
                field, separator, value = line.partition(b":")
                if field == b"data":
                    self._data.append(value.removeprefix(b" ") if separator else b"")
        return packets

    def finish(self) -> None:
        """Reject a partial data event; a trailing comment contains no event."""
        if self._data or (self._line and not self._line.startswith(b":")):
            raise ValueError("generation_sse_incomplete")


@dataclass(frozen=True, slots=True)
class GenerationDelta:
    """Only fields needed by the application, without retaining provider objects."""

    text: str = ""
    finish_reason: str | None = None
    empty: bool = False


def decode_delta(packet: bytes, style: str) -> GenerationDelta:
    """Validate JSON choices before any response text reaches the consumer."""
    payload = json.loads(packet.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_generation_event")
    if "error" in payload:
        raise RuntimeError("generation_provider_error")
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise ValueError("invalid_generation_choices")
    if not choices:
        return GenerationDelta(empty=True)
    if len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("invalid_generation_choices")
    choice = choices[0]
    if type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0:
        raise ValueError("invalid_generation_choice_index")
    finish = choice.get("finish_reason")
    if finish is not None and type(finish) is not str:
        raise ValueError("invalid_generation_finish")
    if style == "chat":
        delta = choice.get("delta")
        if delta is None:
            delta = {}
        if not isinstance(delta, dict):
            raise ValueError("invalid_generation_delta")
        text = delta.get("content")
    else:
        text = choice.get("text")
    if text is None:
        text = ""
    if type(text) is not str:
        raise ValueError("invalid_stream_text")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("invalid_stream_text") from None
    return GenerationDelta(text, finish)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Validated complete text and optional provider-reported token usage."""

    text: str
    tokens: int | None = None


class HTTPGenerationStream:
    """Own one response with bounded parsing and cooperative buffered reads."""

    def __init__(self, response: httpx.Response, style: str, limit: int):
        self.response, self.style, self.limit = response, style, limit

    async def close(self) -> None:
        """Release the actual HTTP response stream."""
        await self.response.aclose()

    async def __aiter__(self) -> AsyncIterator[GenerationDelta]:
        decoder = SSEDecoder(self.limit)
        async for chunk in self.response.aiter_raw():
            # A buffered transport may yield a large burst. Parsing slices
            # preserves capture scheduling without buffering an entire reply.
            for offset in range(0, len(chunk), 8192):
                for packet in decoder.feed(chunk[offset:offset+8192]):
                    if packet == b"[DONE]":
                        return
                    yield decode_delta(packet, self.style)
                await asyncio.sleep(0)
        decoder.finish()


class HTTPGenerationClient:
    """One-request HTTPX client with explicit resource ownership and no retries."""

    def __init__(self, base_url: str, *, api_key: str, timeout: float = 120,
                 verify: bool = True, http_client: httpx.AsyncClient | None = None):
        self._client = http_client or httpx.AsyncClient(verify=verify, timeout=timeout, follow_redirects=False)
        self.base_url = base_url.rstrip("/") + "/"
        self._headers = {"Authorization": "Bearer " + api_key, "Accept-Encoding": "identity"}
        self._models_request: httpx.Request | None = None
        self._completion_path: str | None = None

    def prepare(self, style: str) -> None:
        """Build transport request metadata off-loop, including URL validation."""
        self._models_request = self._client.build_request("GET", self.base_url+"models", headers=self._headers)
        self._completion_path = "chat/completions" if style == "chat" else "completions"

    async def close(self) -> None:
        """Wait for HTTPX transport cleanup."""
        await self._client.aclose()

    def is_closed(self) -> bool:
        """Expose client admission state; owned close tasks track actual cleanup."""
        return self._client.is_closed

    async def _send(self, request: httpx.Request) -> httpx.Response:
        response = await self._client.send(request, stream=True, follow_redirects=False)
        response.stream = OwnedHTTPBody(response.stream)
        try:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("unsupported_generation_encoding")
        except BaseException:
            await finish_owned_close(asyncio.create_task(response.aclose()))
            raise
        return response

    async def _json(self, request: httpx.Request, limit: int) -> dict:
        response = await self._send(request)
        try:
            data = bytearray()
            if response.is_stream_consumed:
                if len(response.content) > limit:
                    raise ValueError("generation_metadata_limit")
                data.extend(response.content)
            else:
                async for chunk in response.aiter_raw():
                    if len(data)+len(chunk) > limit:
                        raise ValueError("generation_metadata_limit")
                    data.extend(chunk)
                    await asyncio.sleep(0)
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("invalid_generation_payload")
            if "error" in payload:
                raise RuntimeError("generation_provider_error")
            return payload
        finally:
            await finish_owned_close(asyncio.create_task(response.aclose()))

    async def models(self) -> list[str]:
        """Read a bounded discovery payload without sending a prompt."""
        payload = await self._json(self._models_request, 262144)
        if not isinstance(payload.get("data"), list):
            raise ValueError("invalid_generation_models")
        ids = []
        for item in payload["data"]:
            if not isinstance(item, dict) or type(item.get("id")) is not str or not item["id"]:
                raise ValueError("invalid_generation_model_id")
            ids.append(item["id"])
        return ids

    async def complete(self, payload: dict, *, style: str, max_chars: int) -> GenerationResult:
        """Read a bounded nonstream response and require a complete text choice."""
        request = self._client.build_request("POST", self.base_url+self._completion_path,
            headers=self._headers | {"Accept": "application/json"}, json=payload)
        result = await self._json(request, min(1048576, max(65536, max_chars*12+4096)))
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("invalid_generation_choices")
        choice = choices[0]
        if type(choice.get("index", 0)) is not int or choice.get("index", 0) != 0:
            raise ValueError("invalid_generation_choice_index")
        finish = choice.get("finish_reason")
        if finish != "stop":
            raise RuntimeError("generation_truncated" if finish == "length" else "generation_incomplete")
        if style == "chat":
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("invalid_generation_message")
            text = message.get("content")
        else:
            text = choice.get("text")
        if type(text) is not str:
            raise ValueError("invalid_generation_text")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("invalid_generation_text") from None
        if len(text) > max_chars:
            raise ValueError("generation_text_limit")
        usage = result.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0:
            tokens = None
        return GenerationResult(text, tokens)

    async def stream(self, payload: dict, *, style: str, max_chars: int) -> HTTPGenerationStream:
        """Open the pinned completion endpoint exactly once."""
        request = self._client.build_request("POST", self.base_url+self._completion_path,
            headers=self._headers | {"Accept": "text/event-stream"}, json=payload)
        response = await self._send(request)
        if response.headers.get("content-type", "").partition(";")[0].strip().lower() != "text/event-stream":
            await finish_owned_close(asyncio.create_task(response.aclose()))
            raise ValueError("invalid_generation_content_type")
        # JSON may escape a non-BMP codepoint as two six-byte surrogates.
        return HTTPGenerationStream(response, style, min(1048576, max(65536, max_chars*12+4096)))
