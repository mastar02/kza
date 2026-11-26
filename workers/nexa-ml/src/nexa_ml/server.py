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
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import grpc

from nexa_ml.audio import WHISPER_SAMPLE_RATE, decode_pcm_f32le
from nexa_ml.embed_slot import EMBED_SLOT_ID
from nexa_ml.notify import SdNotifier
from nexa_ml.observability import InferenceTiming
from nexa_ml.pb import common_pb2, ml_pb2, ml_pb2_grpc
from nexa_ml.runtime import HealthLevel, MlRuntime
from nexa_ml.slot import (
    InvalidInputError,
    ResourceExhaustedError,
    SlotSnapshot,
    SlotState,
    SlotUnavailableError,
)
from nexa_ml.speaker_slot import MIN_SPEAKER_SECONDS, SPEAKER_SLOT_ID
from nexa_ml.tts_slot import TTS_SLOT_ID
from nexa_ml.whisper_slot import COMMAND_SLOT_ID

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


def _role_name(value: int) -> str:
    """Enum name for a role, tolerating unknown proto3 numeric values (never raises)."""
    known = common_pb2.Role.DESCRIPTOR.values_by_number.get(value)
    return known.name if known is not None else f"ROLE_UNKNOWN_{value}"


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


def _error_status(exc: Exception) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, InvalidInputError):
        return grpc.StatusCode.INVALID_ARGUMENT, str(exc)
    if isinstance(exc, SlotUnavailableError):
        return grpc.StatusCode.UNAVAILABLE, str(exc)
    if isinstance(exc, ResourceExhaustedError):
        return grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc)
    return grpc.StatusCode.INTERNAL, "internal error"


@dataclass
class RpcObservation:
    timing: InferenceTiming
    started_ns: int = field(default_factory=time.perf_counter_ns)
    status: grpc.StatusCode = grpc.StatusCode.OK
    mode: str | None = None
    audio_ms: int | None = None
    input_count: int | None = None
    ttfa_ms: int | None = None
    chunks: int | None = None
    error_kind: str | None = None

    def elapsed_ms(self) -> int:
        return (time.perf_counter_ns() - self.started_ns) // 1_000_000

    def log(self, rpc: str) -> None:
        infer_ms, inference_finished = self.timing.snapshot()
        fields = {
            "rpc": rpc,
            "slot": self.timing.slot,
            "mode": self.mode,
            "voice": self.timing.voice,
            "audio_ms": self.audio_ms,
            "input_count": self.input_count,
            "rpc_ms": self.elapsed_ms(),
            "infer_ms": infer_ms,
            "inference_finished": inference_finished,
            "ttfa_ms": self.ttfa_ms,
            "chunks": self.chunks,
            "status": self.status.name,
            "error_kind": self.error_kind,
        }
        if self.status not in (grpc.StatusCode.OK, grpc.StatusCode.CANCELLED):
            level = logging.WARNING
        else:
            level = logging.DEBUG if rpc == "GetHealth" else logging.INFO
        # Emit fields in the message for the worker's default journal formatter.
        message = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(level, "ml_rpc %s", message, extra=fields)


class MlServicer(ml_pb2_grpc.MlServiceServicer):
    """gRPC adapter over the model runtime with process-local RPC counters.

    Args:
        runtime: Slot composition root.
    """

    def __init__(self, runtime: MlRuntime) -> None:
        self._rt = runtime
        self._rpc_totals: Counter[str] = Counter()

    @contextlib.asynccontextmanager
    async def _observe(
        self, context: grpc.aio.ServicerContext, rpc: str, slot: str = "unresolved"
    ) -> AsyncIterator[RpcObservation]:
        obs = RpcObservation(InferenceTiming(slot))
        try:
            yield obs
        except (asyncio.CancelledError, GeneratorExit):
            obs.status = grpc.StatusCode.CANCELLED
            obs.error_kind = "client_cancelled"
            raise
        except grpc.aio.AbortError:
            # Authorization branches set a fixed status before aborting.
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to fixed status codes
            obs.status, detail = _error_status(exc)
            obs.error_kind = type(exc).__name__
            if isinstance(exc, ResourceExhaustedError):
                # ModelSlot chains OOMs from the native error. Capacity limits have no cause.
                obs.error_kind = "model_oom" if exc.__cause__ is not None else "capacity_exhausted"
            await context.abort(obs.status, detail)
        finally:
            if rpc != "GetHealth":
                if obs.status == grpc.StatusCode.OK:
                    self._rpc_totals["rpc_ok"] += 1
                elif obs.status == grpc.StatusCode.CANCELLED:
                    self._rpc_totals["rpc_cancelled"] += 1
                else:
                    self._rpc_totals["rpc_errors"] += 1
            obs.log(rpc)

    async def GetHealth(
        self, request: common_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> ml_pb2.MlHealthResponse:  # noqa: N802
        """Report aggregate and per-slot health without touching model threads."""
        async with self._observe(context, "GetHealth", "worker"):
            level, detail = self._rt.aggregate()
            totals = " ".join(
                f"{key}={self._rpc_totals[key]}" for key in ("rpc_ok", "rpc_errors", "rpc_cancelled")
            )
            base = common_pb2.HealthResponse(
                service=SERVICE_NAME,
                version=self._rt.version,
                state=_LEVEL_TO_PB[level],
                uptime_ms=self._rt.uptime_ms(),
                detail=f"{detail}; {totals}",
            )
            return ml_pb2.MlHealthResponse(base=base, models=[_slot_health(s) for s in self._rt.snapshots()])

    async def Transcribe(
        self, request: ml_pb2.TranscribeRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.TranscribeResponse:  # noqa: N802
        """Route COMMAND to the turbo slot and AMBIENT to the large-v3 pool."""

        async with self._observe(context, "Transcribe") as obs:
            if request.mode == ml_pb2.STT_MODE_COMMAND:
                obs.mode = "command"
                obs.timing.slot = COMMAND_SLOT_ID
                target = self._rt.command
                template = target
            elif request.mode == ml_pb2.STT_MODE_AMBIENT:
                obs.mode = "ambient"
                obs.timing.slot = "stt-ambient-pool"
                target = self._rt.ambient
                template = target.template
            else:
                raise InvalidInputError("mode must be COMMAND or AMBIENT")
            audio = decode_pcm_f32le(
                request.pcm_f32le, request.sample_rate, max_seconds=self._rt.config.max_audio_seconds
            )
            obs.audio_ms = len(audio) * 1000 // WHISPER_SAMPLE_RATE
            opts = template.options(
                language=request.language,
                beam_size=request.beam_size,
                initial_prompt=request.initial_prompt,
                hotwords=request.hotwords,
            )
            result = await target.transcribe(audio, opts, timing=obs.timing)
            return ml_pb2.TranscribeResponse(
                text=result.text,
                compression_ratio=result.compression_ratio,
                avg_logprob=result.avg_logprob,
                no_speech_prob=result.no_speech_prob,
                latency_ms=result.latency_ms,
                model_id=result.model_id,
            )

    async def SynthesizeStream(
        self, request: ml_pb2.SynthesizeRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[ml_pb2.TtsAudioChunk]:  # noqa: N802
        """Stream one chunk per Kokoro Result, then an empty ``is_final`` chunk."""
        tts = self._rt.tts
        async with self._observe(context, "SynthesizeStream", TTS_SLOT_ID) as obs:
            obs.input_count = len(request.text)
            obs.chunks = 0
            output_samples = 0
            opts = tts.options(
                text=request.text, voice=request.voice, lang_code=request.lang_code, speed=request.speed
            )
            # aclosing: on client cancel the inner generator's finally (stop flag) runs now,
            # not whenever the abandoned async generator is garbage-collected.
            async with contextlib.aclosing(tts.stream(opts, timing=obs.timing)) as chunks:
                async for pcm in chunks:
                    if obs.ttfa_ms is None:
                        obs.ttfa_ms = obs.elapsed_ms()
                    sequence = obs.chunks
                    obs.chunks += 1
                    output_samples += len(pcm)
                    obs.audio_ms = output_samples * 1000 // tts.sample_rate
                    yield ml_pb2.TtsAudioChunk(
                        sequence=sequence,
                        sample_rate=tts.sample_rate,
                        pcm_f32le=pcm.astype("<f4", copy=False).tobytes(),
                    )
            yield ml_pb2.TtsAudioChunk(sequence=obs.chunks, sample_rate=tts.sample_rate, is_final=True)

    async def EmbedSpeaker(
        self, request: ml_pb2.EmbedSpeakerRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.SpeakerEmbeddingResponse:  # noqa: N802
        """Return the raw 192-d ECAPA vector (voice is never authority)."""

        async with self._observe(context, "EmbedSpeaker", SPEAKER_SLOT_ID) as obs:
            audio = decode_pcm_f32le(
                request.pcm_f32le,
                request.sample_rate,
                min_seconds=MIN_SPEAKER_SECONDS,
                max_seconds=self._rt.config.max_audio_seconds,
            )
            obs.audio_ms = len(audio) * 1000 // WHISPER_SAMPLE_RATE
            emb = await self._rt.speaker.embed(audio, timing=obs.timing)
            return ml_pb2.SpeakerEmbeddingResponse(
                vector=emb.vector.tolist(), encoder_model_key=emb.encoder_model_key, latency_ms=emb.latency_ms
            )

    async def EmbedText(
        self, request: ml_pb2.EmbedTextRequest, context: grpc.aio.ServicerContext
    ) -> ml_pb2.BatchEmbeddingResponse:  # noqa: N802
        """Return one 1024-d BGE-M3 vector per input text."""

        async with self._observe(context, "EmbedText", EMBED_SLOT_ID) as obs:
            obs.input_count = len(request.texts)
            out = await self._rt.embed.embed(list(request.texts), timing=obs.timing)
            return ml_pb2.BatchEmbeddingResponse(
                vectors=[ml_pb2.EmbeddingVector(values=row.tolist()) for row in out.vectors],
                model_id=out.model_id,
                latency_ms=out.latency_ms,
            )

    async def ReloadModel(
        self, request: ml_pb2.ReloadModelRequest, context: grpc.aio.ServicerContext
    ) -> common_pb2.HealthResponse:  # noqa: N802
        """Drain, unload and recreate one slot; failure leaves it ``UNAVAILABLE``.

        Administrative: the verified ``ActorContext`` from the control plane is
        mandatory (``UNAUTHENTICATED`` if absent or without subject) and must be an
        ``ROLE_OPERATOR`` holding ``PERMISSION_ADMIN_MANAGE`` (``PERMISSION_DENIED``
        otherwise). Both checks run before the slot id is resolved or logged; the
        audit log only records the canonical slot id and role, never the subject.
        """

        async with self._observe(context, "ReloadModel") as obs:
            actor = request.actor if request.HasField("actor") else None
            if actor is None or not actor.subject:
                obs.status = grpc.StatusCode.UNAUTHENTICATED
                obs.error_kind = "missing_actor"
                logger.warning("ReloadModel rejected: missing actor context")
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "actor context required")
            if (
                actor.role != common_pb2.ROLE_OPERATOR
                or common_pb2.PERMISSION_ADMIN_MANAGE not in actor.permissions
            ):
                obs.status = grpc.StatusCode.PERMISSION_DENIED
                obs.error_kind = "insufficient_permissions"
                logger.warning("ReloadModel denied for role %s", _role_name(actor.role))
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "operator with admin.manage required")
            if request.slot_id not in self._rt.slot_ids:
                raise InvalidInputError("unknown slot_id")
            obs.timing.slot = request.slot_id
            logger.info("ReloadModel %s authorized (role=ROLE_OPERATOR)", request.slot_id)
            snap = await self._rt.reload(request.slot_id)
            return common_pb2.HealthResponse(
                service=f"{SERVICE_NAME}/{snap.slot_id}",
                version=self._rt.version,
                state=_slot_health(snap).state,
                uptime_ms=self._rt.uptime_ms(),
                detail="reloaded",
            )


def socket_identity(path: pathlib.Path) -> tuple[int, int]:
    """``(st_dev, st_ino)`` of the socket this process just bound."""
    st = path.lstat()
    return st.st_dev, st.st_ino


def remove_owned_socket(path: pathlib.Path, identity: tuple[int, int] | None) -> bool:
    """Unlink ``path`` only if it is still the very socket this process created.

    Returns:
        ``True`` if removed; ``False`` if missing, replaced or not a socket.
    """
    try:
        st = path.lstat()
    except FileNotFoundError:
        return False
    if identity is None or not stat.S_ISSOCK(st.st_mode) or (st.st_dev, st.st_ino) != identity:
        logger.warning("Socket path no longer holds this worker's socket; leaving it in place")
        return False
    path.unlink()
    return True


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
    identity = socket_identity(socket_path)
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
        remove_owned_socket(socket_path, identity)
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
