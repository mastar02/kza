"""``nexa.v1.MlService`` over a mode-0600 Unix domain socket (``grpc.aio``).

Status codes are fixed and never carry audio or text: ``INVALID_ARGUMENT`` for
bad input/voice/slot, ``UNAVAILABLE`` for a slot that is down or reloading,
``RESOURCE_EXHAUSTED`` for full queues or OOM, ``CANCELLED`` when the client
cancels (Kokoro stops at the next Result) and ``INTERNAL`` for anything else.
"""

import asyncio
import contextlib
import logging
import os
import pathlib
import signal
import stat
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NoReturn

import grpc

from nexa_ml.audio import WHISPER_SAMPLE_RATE, decode_pcm_f32le
from nexa_ml.notify import SdNotifier
from nexa_ml.pb import common_pb2, ml_pb2, ml_pb2_grpc
from nexa_ml.runtime import HealthLevel, MlRuntime
from nexa_ml.slot import (
    InvalidInputError,
    ResourceExhaustedError,
    SlotSnapshot,
    SlotState,
    SlotUnavailableError,
)
from nexa_ml.speaker_slot import MIN_SPEAKER_SECONDS

logger = logging.getLogger(__name__)

SERVICE_NAME = "nexa-ml"
MIN_MESSAGE_BYTES = 16 * 1024 * 1024
MESSAGE_OVERHEAD_BYTES = 1024 * 1024
STOP_GRACE_S = 5.0
# sockaddr_un.sun_path: 104 bytes on Darwin/BSD, 108 on Linux (incl. NUL).
MAX_SOCKET_PATH = 103 if sys.platform == "darwin" else 107

_LEVEL_TO_PB = {
    HealthLevel.OK: common_pb2.HEALTH_STATE_OK,
    HealthLevel.DEGRADED: common_pb2.HEALTH_STATE_DEGRADED,
    HealthLevel.UNAVAILABLE: common_pb2.HEALTH_STATE_UNAVAILABLE,
}


def _slot_health(snap: SlotSnapshot) -> ml_pb2.ModelSlotHealth:
    if snap.state is SlotState.READY:
        state = common_pb2.HEALTH_STATE_DEGRADED if snap.consecutive_failures else common_pb2.HEALTH_STATE_OK
    else:
        state = common_pb2.HEALTH_STATE_UNAVAILABLE
    return ml_pb2.ModelSlotHealth(
        slot_id=snap.slot_id,
        model_name=snap.model_name,
        device=snap.device,
        state=state,
        vram_mib=max(0, snap.vram_mib),
        consecutive_failures=snap.consecutive_failures,
    )


async def _abort_for(context: grpc.aio.ServicerContext, rpc: str, exc: BaseException) -> NoReturn:
    """Translate a slot exception into a fixed status (details never echo payloads)."""
    if isinstance(exc, InvalidInputError):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    if isinstance(exc, SlotUnavailableError):
        await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
    if isinstance(exc, ResourceExhaustedError):
        await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
    logger.error("%s failed: %s", rpc, type(exc).__name__)
    await context.abort(grpc.StatusCode.INTERNAL, "internal error")
    raise AssertionError("unreachable")  # pragma: no cover


async def _guarded(context: grpc.aio.ServicerContext, rpc: str, call: Callable[[], Awaitable[Any]]) -> Any:
    try:
        return await call()
    except asyncio.CancelledError:
        raise
    except grpc.aio.AbortError:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to fixed status codes
        await _abort_for(context, rpc, exc)


class MlServicer(ml_pb2_grpc.MlServiceServicer):
    """gRPC adapter over :class:`MlRuntime` (stateless; no business logic).

    Args:
        runtime: Slot composition root.
    """

    def __init__(self, runtime: MlRuntime) -> None:
        self._rt = runtime

    async def GetHealth(
        self, request: common_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> ml_pb2.MlHealthResponse:  # noqa: N802
        """Report aggregate and per-slot health without touching model threads."""
        level, detail = self._rt.aggregate()
        base = common_pb2.HealthResponse(
            service=SERVICE_NAME,
            version=self._rt.version,
            state=_LEVEL_TO_PB[level],
            uptime_ms=self._rt.uptime_ms(),
            detail=detail,
        )
        return ml_pb2.MlHealthResponse(base=base, models=[_slot_health(s) for s in self._rt.snapshots()])

    async def Transcribe(
        self, request: ml_pb2.TranscribeRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.TranscribeResponse:  # noqa: N802
        """Route COMMAND to the turbo slot and AMBIENT to the large-v3 pool."""

        async def call() -> ml_pb2.TranscribeResponse:
            if request.mode == ml_pb2.STT_MODE_COMMAND:
                target = self._rt.command
                template = target
            elif request.mode == ml_pb2.STT_MODE_AMBIENT:
                target = self._rt.ambient
                template = target.template
            else:
                raise InvalidInputError("mode must be COMMAND or AMBIENT")
            audio = decode_pcm_f32le(
                request.pcm_f32le, request.sample_rate, max_seconds=self._rt.config.max_audio_seconds
            )
            opts = template.options(
                language=request.language,
                beam_size=request.beam_size,
                initial_prompt=request.initial_prompt,
                hotwords=request.hotwords,
            )
            result = await target.transcribe(audio, opts)
            return ml_pb2.TranscribeResponse(
                text=result.text,
                compression_ratio=result.compression_ratio,
                avg_logprob=result.avg_logprob,
                no_speech_prob=result.no_speech_prob,
                latency_ms=result.latency_ms,
                model_id=result.model_id,
            )

        return await _guarded(context, "Transcribe", call)

    async def SynthesizeStream(
        self, request: ml_pb2.SynthesizeRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[ml_pb2.TtsAudioChunk]:  # noqa: N802
        """Stream one chunk per Kokoro Result, then an empty ``is_final`` chunk."""
        tts = self._rt.tts
        sequence = 0
        try:
            opts = tts.options(
                text=request.text, voice=request.voice, lang_code=request.lang_code, speed=request.speed
            )
            # aclosing: on client cancel the inner generator's finally (stop flag) runs now,
            # not whenever the abandoned async generator is garbage-collected.
            async with contextlib.aclosing(tts.stream(opts)) as chunks:
                async for pcm in chunks:
                    yield ml_pb2.TtsAudioChunk(
                        sequence=sequence,
                        sample_rate=tts.sample_rate,
                        pcm_f32le=pcm.astype("<f4", copy=False).tobytes(),
                    )
                    sequence += 1
        except asyncio.CancelledError:
            logger.info("SynthesizeStream cancelled by client after %d chunks", sequence)
            raise
        except grpc.aio.AbortError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to fixed status codes
            await _abort_for(context, "SynthesizeStream", exc)
        yield ml_pb2.TtsAudioChunk(sequence=sequence, sample_rate=tts.sample_rate, is_final=True)

    async def EmbedSpeaker(
        self, request: ml_pb2.EmbedSpeakerRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.SpeakerEmbeddingResponse:  # noqa: N802
        """Return the raw 192-d ECAPA vector (voice is never authority)."""

        async def call() -> ml_pb2.SpeakerEmbeddingResponse:
            audio = decode_pcm_f32le(
                request.pcm_f32le,
                request.sample_rate,
                min_seconds=MIN_SPEAKER_SECONDS,
                max_seconds=self._rt.config.max_audio_seconds,
            )
            emb = await self._rt.speaker.embed(audio)
            return ml_pb2.SpeakerEmbeddingResponse(
                vector=emb.vector.tolist(), encoder_model_key=emb.encoder_model_key, latency_ms=emb.latency_ms
            )

        return await _guarded(context, "EmbedSpeaker", call)

    async def EmbedText(
        self, request: ml_pb2.EmbedTextRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.BatchEmbeddingResponse:  # noqa: N802
        """Return one 1024-d BGE-M3 vector per input text."""

        async def call() -> ml_pb2.BatchEmbeddingResponse:
            out = await self._rt.embed.embed(list(request.texts))
            return ml_pb2.BatchEmbeddingResponse(
                vectors=[ml_pb2.EmbeddingVector(values=row.tolist()) for row in out.vectors],
                model_id=out.model_id,
                latency_ms=out.latency_ms,
            )

        return await _guarded(context, "EmbedText", call)

    async def ReloadModel(
        self, request: ml_pb2.ReloadModelRequest, context: grpc.aio.ServicerContext
    ) -> common_pb2.HealthResponse:  # noqa: N802
        """Drain, unload and recreate one slot; failure leaves it ``UNAVAILABLE``.

        Authorization is enforced upstream (control plane over the 0600 socket);
        the actor is only logged by role for audit, never by subject.
        """

        async def call() -> common_pb2.HealthResponse:
            logger.info(
                "ReloadModel %s requested (role=%s)",
                request.slot_id,
                common_pb2.Role.Name(request.actor.role),
            )
            snap = await self._rt.reload(request.slot_id)
            return common_pb2.HealthResponse(
                service=f"{SERVICE_NAME}/{snap.slot_id}",
                version=self._rt.version,
                state=_slot_health(snap).state,
                uptime_ms=self._rt.uptime_ms(),
                detail="reloaded",
            )

        return await _guarded(context, "ReloadModel", call)


def prepare_socket_path(path: pathlib.Path) -> None:
    """Validate length, create the 0700 parent and unlink a stale socket.

    Raises:
        ValueError: Path too long for ``sun_path`` or an existing non-socket file.
    """
    if len(os.fsencode(path)) > MAX_SOCKET_PATH:
        raise ValueError(f"socket path longer than {MAX_SOCKET_PATH} bytes")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        st = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        raise ValueError("socket path exists and is not a socket")
    path.unlink()


async def start_server(runtime: MlRuntime, socket_path: pathlib.Path) -> grpc.aio.Server:
    """Bind ``MlService`` on ``unix:<socket_path>`` with mode 0600 and start it.

    Raises:
        ValueError: Invalid socket path.
        RuntimeError: Bind failed.
    """
    prepare_socket_path(socket_path)
    # Largest accepted PCM (max_audio_seconds of 16 kHz float32) plus protobuf overhead.
    pcm_bytes = int(runtime.config.max_audio_seconds * WHISPER_SAMPLE_RATE * 4)
    max_bytes = max(MIN_MESSAGE_BYTES, pcm_bytes + MESSAGE_OVERHEAD_BYTES)
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", max_bytes),
            ("grpc.max_send_message_length", max_bytes),
        ]
    )
    ml_pb2_grpc.add_MlServiceServicer_to_server(MlServicer(runtime), server)
    previous = os.umask(0o077)
    try:
        if server.add_insecure_port(f"unix:{socket_path}") == 0:
            raise RuntimeError("failed to bind ml socket")
        await server.start()
    finally:
        os.umask(previous)
    os.chmod(socket_path, 0o600)
    return server


async def serve(
    runtime: MlRuntime,
    socket_path: pathlib.Path,
    notifier: SdNotifier,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Full worker lifecycle: bind, load slots, ``READY=1``, watchdog, ``STOPPING=1``.

    Args:
        runtime: Slot composition root (slots not yet loaded).
        socket_path: UDS path.
        notifier: systemd notifier (no-op off systemd).
        stop_event: Optional external stop trigger; SIGTERM/SIGINT also set it.
    """
    stop = stop_event or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, stop.set)
    server = await start_server(runtime, socket_path)
    watchdog = asyncio.create_task(notifier.watchdog_loop())
    loading = asyncio.create_task(runtime.load_all())
    try:
        notifier.notify("STATUS=loading models")
        try:
            # Shield: a slow load keeps running and flips its slot READY later.
            await asyncio.wait_for(asyncio.shield(loading), timeout=runtime.config.startup_timeout_s)
        except TimeoutError:
            logger.error("Model loading exceeded %.0f s; serving degraded", runtime.config.startup_timeout_s)
        level, detail = runtime.aggregate()
        notifier.notify("READY=1", f"STATUS={level}: {detail}")
        logger.info("nexa-ml ready (%s: %s)", level, detail)
        await stop.wait()
    finally:
        notifier.notify("STOPPING=1")
        for task in (watchdog, loading):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await server.stop(STOP_GRACE_S)
        runtime.close()
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
