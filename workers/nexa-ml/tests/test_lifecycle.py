"""``serve()`` lifecycle (bind -> load -> READY -> STOPPING) and pure helpers."""

import asyncio
import pathlib
import socket

import grpc
import numpy as np
import pytest
from conftest import Fakes

from nexa_ml.audio import decode_pcm_f32le, parse_device
from nexa_ml.config import MlConfig
from nexa_ml.notify import SdNotifier
from nexa_ml.pb import common_pb2, ml_pb2_grpc
from nexa_ml.runtime import MlRuntime
from nexa_ml.server import prepare_socket_path, serve
from nexa_ml.slot import InvalidInputError, is_out_of_memory
from nexa_ml.whisper_slot import resolve_model_source


async def test_serve_notifies_ready_then_stopping_and_removes_socket(short_dir: pathlib.Path) -> None:
    notify_path = short_dir / "n.sock"
    sock_path = short_dir / "run" / "ml.sock"
    cfg = MlConfig()
    fakes = Fakes()
    runtime = MlRuntime(cfg, fakes.loaders(cfg.ambient_stt_workers))
    stop = asyncio.Event()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(notify_path))
        receiver.setblocking(False)
        task = asyncio.create_task(
            serve(runtime, sock_path, SdNotifier(env={"NOTIFY_SOCKET": str(notify_path)}), stop)
        )
        messages: list[bytes] = []

        async def wait_for(prefix: bytes) -> None:
            for _ in range(500):
                try:
                    messages.append(receiver.recv(256))
                except BlockingIOError:
                    await asyncio.sleep(0.01)
                if any(m.startswith(prefix) for m in messages):
                    return
            raise AssertionError(f"{prefix!r} never sent: {messages}")

        await wait_for(b"READY=1")
        assert (sock_path.parent.stat().st_mode & 0o777) == 0o700
        async with grpc.aio.insecure_channel(f"unix:{sock_path}") as channel:
            health = await ml_pb2_grpc.MlServiceStub(channel).GetHealth(common_pb2.Empty(), timeout=2)
        assert health.base.state == common_pb2.HEALTH_STATE_OK
        ready = next(m for m in messages if m.startswith(b"READY=1"))
        assert b"STATUS=ok: 6/6 slots ready" in ready
        stop.set()
        await asyncio.wait_for(task, timeout=10)
        await wait_for(b"STOPPING=1")
    assert not sock_path.exists()


async def test_serve_degraded_when_a_slot_fails_to_load(short_dir: pathlib.Path) -> None:
    cfg = MlConfig()
    fakes = Fakes()
    fakes.command.fail = True
    runtime = MlRuntime(cfg, fakes.loaders(cfg.ambient_stt_workers))
    await runtime.load_all()
    assert runtime.aggregate()[0] == "degraded"
    assert fakes.command.loads == 1
    runtime.close()


def test_prepare_socket_path_rejects_regular_file_and_long_paths(short_dir: pathlib.Path) -> None:
    regular = short_dir / "file"
    regular.write_text("x")
    with pytest.raises(ValueError):
        prepare_socket_path(regular)
    with pytest.raises(ValueError):
        prepare_socket_path(short_dir / ("x" * 120))


def test_decode_pcm_is_writable_float32() -> None:
    payload = np.array([0.5, -0.5], dtype="<f4").tobytes()
    audio = decode_pcm_f32le(payload, 16_000, max_seconds=1)
    assert audio.dtype == np.float32 and audio.flags.writeable
    with pytest.raises(InvalidInputError):
        decode_pcm_f32le(payload, 16_000, min_seconds=0.5, max_seconds=1)


def test_parse_device_and_model_ids(tmp_path: pathlib.Path) -> None:
    assert parse_device("cuda:1") == ("cuda", 1)
    assert parse_device("cpu") == ("cpu", 0)
    assert resolve_model_source("whisper-v3-turbo", None) == "large-v3-turbo"
    assert resolve_model_source("whisper-large-v3", None) == "large-v3"
    (tmp_path / "whisper-v3-turbo").mkdir()
    assert resolve_model_source("whisper-v3-turbo", str(tmp_path)) == str(tmp_path / "whisper-v3-turbo")
    assert resolve_model_source("/abs/model", None) == "/abs/model"


def test_oom_detection() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_out_of_memory(OutOfMemoryError("x"))
    assert is_out_of_memory(RuntimeError("CUDA failed with error out of memory"))
    assert not is_out_of_memory(RuntimeError("decoder failed"))
