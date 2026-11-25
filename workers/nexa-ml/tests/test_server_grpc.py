"""L2: real grpc.aio server on a tmp UDS with torch-free model fakes."""

import asyncio
import logging
import os
import stat
import threading

import grpc
import numpy as np
import pytest
from conftest import GATE_TIMEOUT_S, Harness, pcm

from nexa_ml.pb import common_pb2, ml_pb2
from nexa_ml.slot import SlotState

EXPECTED_SLOTS = ["stt-command", "stt-ambient-0", "stt-ambient-1", "tts", "speaker", "embed"]


def _tx(mode: int, payload: bytes | None = None, **kw: object) -> ml_pb2.TranscribeRequest:
    return ml_pb2.TranscribeRequest(
        pcm_f32le=pcm() if payload is None else payload,
        sample_rate=kw.pop("sample_rate", 16_000),
        mode=mode,
        **kw,
    )


async def _wait_until(predicate, timeout: float = GATE_TIMEOUT_S) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------- socket / health
async def test_socket_is_mode_0600(harness: Harness) -> None:
    st = os.stat(harness.socket_path)
    assert stat.S_ISSOCK(st.st_mode)
    assert stat.S_IMODE(st.st_mode) == 0o600


async def test_health_reports_every_slot_ok(harness: Harness) -> None:
    resp = await harness.stub.GetHealth(common_pb2.Empty(), timeout=2)
    assert resp.base.service == "nexa-ml"
    assert resp.base.state == common_pb2.HEALTH_STATE_OK
    assert [m.slot_id for m in resp.models] == EXPECTED_SLOTS
    assert all(m.state == common_pb2.HEALTH_STATE_OK for m in resp.models)
    devices = {m.slot_id: m.device for m in resp.models}
    assert devices["stt-command"] == "cuda:1"
    assert devices["stt-ambient-0"] == "cuda:0"
    assert devices["speaker"] == "cuda:0"  # speaker_encoder_device default from config
    assert devices["embed"] == "cpu"


# ---------------------------------------------------------------- STT routing
async def test_command_routes_to_turbo_without_vad(harness: Harness) -> None:
    resp = await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert resp.model_id == "whisper-v3-turbo"
    assert resp.text == "cmd hola mundo"
    assert resp.compression_ratio == pytest.approx(1.8)
    assert resp.avg_logprob == pytest.approx(-0.2)
    call = harness.fakes.command.models[0].calls[-1]
    assert call["vad_filter"] is False
    assert call["beam_size"] == 1
    assert call["language"] == "es"
    assert call["condition_on_previous_text"] is False
    assert "hotwords" not in call
    assert all(f.models[0].calls == [] for f in harness.fakes.ambient)


async def test_ambient_routes_to_pool_with_vad(harness: Harness) -> None:
    resp = await harness.stub.Transcribe(
        _tx(ml_pb2.STT_MODE_AMBIENT, language="en", beam_size=3, hotwords="nexa"), timeout=30
    )
    assert resp.model_id == "whisper-large-v3"
    assert resp.text.startswith("amb")
    calls = [c for f in harness.fakes.ambient for c in f.models[0].calls]
    assert len(calls) == 1
    assert calls[0]["vad_filter"] is True
    assert calls[0]["beam_size"] == 3
    assert calls[0]["language"] == "en"
    assert calls[0]["hotwords"] == "nexa"
    assert harness.fakes.command.models[0].calls == []


async def test_each_whisper_runs_on_its_own_single_thread(harness: Harness) -> None:
    for _ in range(3):
        await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    names = set(harness.fakes.command.models[0].thread_names)
    assert len(names) == 1
    assert next(iter(names)).startswith("nexa-ml-stt-command")


@pytest.mark.parametrize(
    ("request_kwargs", "fragment"),
    [
        ({"mode": ml_pb2.STT_MODE_UNSPECIFIED}, "mode"),
        ({"payload": b""}, "empty"),
        ({"payload": b"\x00\x00\x00"}, "multiple of 4"),
        ({"sample_rate": 48_000}, "sample_rate"),
        ({"payload": np.array([np.nan] * 400, dtype="<f4").tobytes()}, "non-finite"),
        ({"beam_size": 11}, "beam_size"),
        ({"language": "Spanish!"}, "language"),
        ({"payload": pcm(seconds=121)}, "longer"),
    ],
)
async def test_transcribe_invalid_argument(harness: Harness, request_kwargs: dict, fragment: str) -> None:
    kw = dict(request_kwargs)
    mode = kw.pop("mode", ml_pb2.STT_MODE_COMMAND)
    payload = kw.pop("payload", None)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.Transcribe(_tx(mode, payload, **kw), timeout=10)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert fragment in err.value.details()


async def test_model_exception_is_internal_without_payload(harness: Harness) -> None:
    def boom() -> None:
        raise ValueError("decoder exploded on: hola secreta")

    harness.fakes.command.hook = boom
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND, initial_prompt="secreto"), timeout=10)
    assert err.value.code() == grpc.StatusCode.INTERNAL
    assert err.value.details() == "internal error"
    health = await harness.stub.GetHealth(common_pb2.Empty(), timeout=2)
    cmd = next(m for m in health.models if m.slot_id == "stt-command")
    assert cmd.consecutive_failures == 1
    assert cmd.state == common_pb2.HEALTH_STATE_DEGRADED


async def test_cuda_oom_maps_to_resource_exhausted(harness: Harness) -> None:
    def oom() -> None:
        raise RuntimeError("CUDA failed with error out of memory")

    harness.fakes.command.hook = oom
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert err.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


# ---------------------------------------------------------------- AMBIENT pool x2
async def test_ambient_pool_runs_two_requests_concurrently(harness: Harness) -> None:
    barrier = threading.Barrier(2, timeout=GATE_TIMEOUT_S)
    for factory in harness.fakes.ambient:
        factory.hook = barrier.wait  # deadlocks (BrokenBarrierError) if the pool serialises
    responses = await asyncio.gather(
        harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_AMBIENT), timeout=30),
        harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_AMBIENT), timeout=30),
    )
    assert sorted(r.text.split()[0] for r in responses) == ["amb0", "amb1"]


async def test_ambient_third_request_waits_for_a_free_member(harness: Harness) -> None:
    release = threading.Event()
    started = threading.Semaphore(0)

    def hold() -> None:
        started.release()
        release.wait(GATE_TIMEOUT_S)

    for factory in harness.fakes.ambient:
        factory.hook = hold
    calls = [
        asyncio.ensure_future(harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_AMBIENT), timeout=30))
        for _ in range(3)
    ]
    await _wait_until(lambda: sum(m.slot.inflight for m in harness.runtime.ambient.members) == 2)
    await asyncio.sleep(0.1)
    assert not any(c.done() for c in calls)
    assert sum(len(f.models[0].calls) for f in harness.fakes.ambient) == 2
    release.set()
    responses = await asyncio.gather(*calls)
    assert len(responses) == 3
    assert sum(len(f.models[0].calls) for f in harness.fakes.ambient) == 3


async def test_command_queue_full_is_resource_exhausted(harness: Harness) -> None:
    release = threading.Event()
    harness.fakes.command.hook = lambda: release.wait(GATE_TIMEOUT_S)
    calls = [
        asyncio.ensure_future(harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10))
        for _ in range(4)
    ]  # max_pending=4
    await _wait_until(lambda: harness.runtime.command.slot.inflight == 4)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert err.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    release.set()
    await asyncio.gather(*calls)


# ---------------------------------------------------------------- TTS streaming
async def test_first_tts_chunk_arrives_before_synthesis_completes(harness: Harness) -> None:
    """Anti-regression for piper_tts.py:833 (full buffering before chunking)."""
    gates = [threading.Event(), threading.Event()]
    harness.fakes.tts_gates.extend(gates)
    engine = harness.fakes.tts_engines[0]
    call = harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="Hola, buenas tardes."), timeout=10)
    stream = call.__aiter__()

    first = await asyncio.wait_for(stream.__anext__(), timeout=2)
    # Result 2 cannot exist yet: the fake blocks on gates[0] until we open it.
    assert engine.pulled == 1
    assert first.sequence == 0 and not first.is_final
    assert first.sample_rate == 24_000
    assert np.frombuffer(first.pcm_f32le, dtype="<f4")[0] == 1.0

    gates[0].set()
    second = await asyncio.wait_for(stream.__anext__(), timeout=2)
    assert second.sequence == 1 and engine.pulled == 2
    gates[1].set()
    rest = [chunk async for chunk in stream]
    assert [c.sequence for c in rest] == [2, 3]
    assert rest[-1].is_final and rest[-1].pcm_f32le == b""
    assert np.frombuffer(rest[0].pcm_f32le, dtype="<f4")[0] == 3.0
    assert engine.langs == ["e"] and engine.requests == [("ef_dora", 1.0)]


async def test_tts_cancel_stops_pulling_and_frees_the_slot(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="nexa_ml.server")
    gate = threading.Event()
    harness.fakes.tts_results = 5
    engine = harness.fakes.tts_engines[0]
    engine.results = 5
    harness.fakes.tts_gates.append(gate)
    call = harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="Texto largo."), timeout=10)
    stream = call.__aiter__()
    await asyncio.wait_for(stream.__anext__(), timeout=2)
    call.cancel()
    # Server-side cancellation observed (logged after the stream's stop flag is set);
    # the producer thread is still blocked inside Result 2, so the slot is still held.
    await _wait_until(lambda: "cancelled by client" in caplog.text)
    assert harness.runtime.tts.slot.inflight == 1
    gate.set()  # producer now sees the stop flag at the next Result boundary
    assert await asyncio.to_thread(engine.closed.wait, GATE_TIMEOUT_S)
    await _wait_until(lambda: harness.runtime.tts.slot.inflight == 0)
    assert engine.pulled <= 2 < engine.results
    assert await call.code() == grpc.StatusCode.CANCELLED

    engine.gates.clear()
    engine.results = 2
    chunks = [
        c
        async for c in harness.stub.SynthesizeStream(
            ml_pb2.SynthesizeRequest(text="Otra frase.", voice="em_alex", lang_code="e", speed=1.2),
            timeout=10,
        )
    ]
    assert [c.is_final for c in chunks] == [False, False, True]
    assert engine.requests[-1] == ("em_alex", pytest.approx(1.2))


@pytest.mark.parametrize(
    ("request_kwargs", "fragment"),
    [
        ({"text": "   "}, "text is empty"),
        ({"text": "x" * 2001}, "longer"),
        ({"text": "hola", "voice": "../../etc/passwd"}, "voice"),
        ({"text": "hola", "lang_code": "q"}, "lang_code"),
        ({"text": "hola", "speed": 3.0}, "speed"),
    ],
)
async def test_tts_invalid_argument(harness: Harness, request_kwargs: dict, fragment: str) -> None:
    call = harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(**request_kwargs), timeout=10)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        async for _ in call:
            pass
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert fragment in err.value.details()
    assert harness.fakes.tts_engines[0].requests == []


# ---------------------------------------------------------------- embeddings
async def test_speaker_embedding_is_192d_raw_vector(harness: Harness) -> None:
    resp = await harness.stub.EmbedSpeaker(
        ml_pb2.EmbedSpeakerRequest(pcm_f32le=pcm(amplitude=2.0), sample_rate=16_000), timeout=10
    )
    assert len(resp.vector) == 192
    assert resp.encoder_model_key == "speechbrain/spkrec-ecapa-voxceleb"
    assert harness.fakes.ecapa.peaks[-1] == pytest.approx(1.0)  # legacy peak normalisation
    assert set(resp.DESCRIPTOR.fields_by_name) == {"vector", "encoder_model_key", "latency_ms"}


async def test_speaker_rejects_short_or_wrong_rate(harness: Harness) -> None:
    for kwargs in (
        {"pcm_f32le": pcm(seconds=0.2), "sample_rate": 16_000},
        {"pcm_f32le": pcm(), "sample_rate": 8_000},
    ):
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await harness.stub.EmbedSpeaker(ml_pb2.EmbedSpeakerRequest(**kwargs), timeout=10)
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_speaker_wrong_dimension_is_internal(harness: Harness) -> None:
    harness.fakes.ecapa.dim = 256
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.EmbedSpeaker(
            ml_pb2.EmbedSpeakerRequest(pcm_f32le=pcm(), sample_rate=16_000), timeout=10
        )
    assert err.value.code() == grpc.StatusCode.INTERNAL


async def test_text_embeddings_are_1024d_via_to_thread(harness: Harness) -> None:
    resp = await harness.stub.EmbedText(ml_pb2.EmbedTextRequest(texts=["luz cocina", "persiana"]), timeout=10)
    assert len(resp.vectors) == 2
    assert all(len(v.values) == 1024 for v in resp.vectors)
    assert resp.model_id == "BAAI/bge-m3"
    assert not harness.fakes.embedder.thread_names[-1].startswith("nexa-ml-")


async def test_text_embeddings_validation(harness: Harness) -> None:
    for texts in ([], [""], ["ok"] * 65):
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await harness.stub.EmbedText(ml_pb2.EmbedTextRequest(texts=texts), timeout=10)
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    harness.fakes.embedder.dim = 768
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.EmbedText(ml_pb2.EmbedTextRequest(texts=["x"]), timeout=10)
    assert err.value.code() == grpc.StatusCode.INTERNAL


# ---------------------------------------------------------------- ReloadModel
def _reload(slot_id: str) -> ml_pb2.ReloadModelRequest:
    actor = common_pb2.ActorContext(
        subject="op", role=common_pb2.ROLE_OPERATOR, permissions=[common_pb2.PERMISSION_ADMIN_MANAGE]
    )
    return ml_pb2.ReloadModelRequest(slot_id=slot_id, actor=actor)


async def test_reload_unloads_old_model_and_recreates(harness: Harness) -> None:
    old = harness.fakes.command.models[0]
    resp = await harness.stub.ReloadModel(_reload("stt-command"), timeout=120)
    assert resp.state == common_pb2.HEALTH_STATE_OK
    assert resp.service == "nexa-ml/stt-command"
    assert harness.fakes.command.loads == 2
    assert old.model.unloaded == 1
    await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert harness.fakes.command.models[1].calls and len(old.calls) == 0


async def test_reload_failure_leaves_slot_unavailable_without_lazy_reload(harness: Harness) -> None:
    harness.fakes.command.fail = True
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.ReloadModel(_reload("stt-command"), timeout=120)
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE
    loads_after_failure = harness.fakes.command.loads

    health = await harness.stub.GetHealth(common_pb2.Empty(), timeout=2)
    cmd = next(m for m in health.models if m.slot_id == "stt-command")
    assert cmd.state == common_pb2.HEALTH_STATE_UNAVAILABLE
    assert health.base.state == common_pb2.HEALTH_STATE_DEGRADED

    for _ in range(3):
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
        assert err.value.code() == grpc.StatusCode.UNAVAILABLE
    assert harness.fakes.command.loads == loads_after_failure  # no lazy reload on the request path

    # AMBIENT keeps serving; an explicit reload is the only way back.
    await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_AMBIENT), timeout=30)
    harness.fakes.command.fail = False
    await harness.stub.ReloadModel(_reload("stt-command"), timeout=120)
    resp = await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert resp.model_id == "whisper-v3-turbo"


async def test_reload_drains_in_flight_and_rejects_new_requests(harness: Harness) -> None:
    release = threading.Event()
    harness.fakes.command.hook = lambda: release.wait(GATE_TIMEOUT_S)
    slot = harness.runtime.command.slot
    inflight = asyncio.ensure_future(harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10))
    await _wait_until(lambda: slot.inflight == 1)
    reload_call = asyncio.ensure_future(harness.stub.ReloadModel(_reload("stt-command"), timeout=120))
    await _wait_until(lambda: slot.state is SlotState.RELOADING)

    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10)
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE
    await asyncio.sleep(0.1)
    assert not reload_call.done() and harness.fakes.command.loads == 1

    release.set()
    assert (await inflight).model_id == "whisper-v3-turbo"  # drained, not killed
    assert (await reload_call).state == common_pb2.HEALTH_STATE_OK
    assert harness.fakes.command.loads == 2


async def test_reload_one_ambient_member_keeps_pool_serving(harness: Harness) -> None:
    harness.fakes.ambient[1].fail = True
    with pytest.raises(grpc.aio.AioRpcError):
        await harness.stub.ReloadModel(_reload("stt-ambient-1"), timeout=120)
    for _ in range(3):
        resp = await harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_AMBIENT), timeout=30)
        assert resp.text.startswith("amb0")


async def test_reload_unknown_slot_is_invalid_argument(harness: Harness) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.ReloadModel(_reload("gpu-everything"), timeout=120)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_cancelled_reload_still_completes_and_slot_recovers(harness: Harness) -> None:
    """A ReloadModel cancelled mid-drain must not strand the slot in RELOADING."""
    release = threading.Event()
    harness.fakes.command.hook = lambda: release.wait(GATE_TIMEOUT_S)
    slot = harness.runtime.command.slot
    inflight = asyncio.ensure_future(harness.stub.Transcribe(_tx(ml_pb2.STT_MODE_COMMAND), timeout=10))
    await _wait_until(lambda: slot.inflight == 1)
    reload_call = harness.stub.ReloadModel(_reload("stt-command"), timeout=120)
    reload_future = asyncio.ensure_future(reload_call)
    await _wait_until(lambda: slot.state is SlotState.RELOADING)
    reload_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reload_future
    await asyncio.sleep(0.05)
    harness.fakes.command.hook = None
    release.set()
    await inflight
    await _wait_until(lambda: slot.state is SlotState.READY)
    assert harness.fakes.command.loads == 2
    assert (await harness.stub.ReloadModel(_reload("stt-command"), timeout=120)).state == (
        common_pb2.HEALTH_STATE_OK
    )
    assert harness.fakes.command.loads == 3


async def test_tts_unknown_voice_is_invalid_argument_not_failure(harness: Harness) -> None:
    call = harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="hola", voice="zz_nadie"), timeout=10)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        async for _ in call:
            pass
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    health = await harness.stub.GetHealth(common_pb2.Empty(), timeout=2)
    tts = next(m for m in health.models if m.slot_id == "tts")
    assert tts.state == common_pb2.HEALTH_STATE_OK and tts.consecutive_failures == 0


# ---------------------------------------------------------------- ReloadModel authz
@pytest.mark.parametrize(
    ("actor", "code"),
    [
        (None, grpc.StatusCode.UNAUTHENTICATED),
        (
            common_pb2.ActorContext(
                role=common_pb2.ROLE_OPERATOR, permissions=[common_pb2.PERMISSION_ADMIN_MANAGE]
            ),
            grpc.StatusCode.UNAUTHENTICATED,
        ),
        (
            common_pb2.ActorContext(
                subject="casa", role=common_pb2.ROLE_HOME, permissions=[common_pb2.PERMISSION_ADMIN_MANAGE]
            ),
            grpc.StatusCode.PERMISSION_DENIED,
        ),
        (
            common_pb2.ActorContext(
                subject="op",
                role=common_pb2.ROLE_OPERATOR,
                permissions=[common_pb2.PERMISSION_DIAGNOSTICS_READ],
            ),
            grpc.StatusCode.PERMISSION_DENIED,
        ),
    ],
    ids=["actor-absent", "no-subject", "role-home", "operator-without-admin-manage"],
)
async def test_reload_requires_authorized_operator(
    harness: Harness,
    caplog: pytest.LogCaptureFixture,
    actor: common_pb2.ActorContext | None,
    code: grpc.StatusCode,
) -> None:
    caplog.set_level(logging.INFO, logger="nexa_ml.server")
    request = ml_pb2.ReloadModelRequest(slot_id="stt-command")
    if actor is not None:
        request.actor.CopyFrom(actor)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.ReloadModel(request, timeout=10)
    assert err.value.code() == code
    assert harness.fakes.command.loads == 1  # slot never touched
    assert harness.runtime.command.slot.state is SlotState.READY
    assert "stt-command" not in caplog.text


async def test_reload_authz_runs_before_slot_resolution(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="nexa_ml.server")
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.ReloadModel(ml_pb2.ReloadModelRequest(slot_id="evil\nslot"), timeout=10)
    assert err.value.code() == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.aio.AioRpcError) as err:
        await harness.stub.ReloadModel(_reload("evil\nslot"), timeout=10)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "evil" not in caplog.text  # unvalidated ids never reach the audit log


async def test_reload_authorized_operator_logs_canonical_id_only(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="nexa_ml.server")
    resp = await harness.stub.ReloadModel(_reload("tts"), timeout=120)
    assert resp.state == common_pb2.HEALTH_STATE_OK
    assert "ReloadModel tts authorized (role=ROLE_OPERATOR)" in caplog.text
    assert all("op" not in str(r.args) for r in caplog.records)  # subject never logged


# ---------------------------------------------------------------- TTS producer end guarantee
async def test_tts_close_failure_still_ends_the_rpc(harness: Harness) -> None:
    engine = harness.fakes.tts_engines[0]
    engine.close_raises = True
    call = harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="hola"), timeout=5)
    with pytest.raises(grpc.aio.AioRpcError) as err:
        async for _ in call:
            pass
    assert err.value.code() == grpc.StatusCode.INTERNAL  # not DEADLINE_EXCEEDED (hang)
    await _wait_until(lambda: harness.runtime.tts.slot.inflight == 0)
    engine.close_raises = False
    chunks = [
        c async for c in harness.stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="otra"), timeout=5)
    ]
    assert chunks[-1].is_final
