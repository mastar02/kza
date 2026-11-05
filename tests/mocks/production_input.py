"""Synthetic PortAudio semantics with real actor and callable receipts."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


_CHILD_BOOTSTRAP = r'''
import os, sys
from pathlib import Path
from types import ModuleType
from importlib.machinery import ModuleSpec

ROOT = Path.cwd()
assert (ROOT / "src/audio/owned_input.py").is_file(), "child_checkout_mismatch"
_denied = []
_real_exit = os._exit

def _checked_exit(code):
    if code == 0 and _denied:
        os.write(2, b"synthetic child isolation guard was triggered\n")
        _real_exit(86)
    _real_exit(code)

os._exit = _checked_exit

def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        name = Path(path).name
        if name == ".env" or name.startswith(".env.") or path.startswith((
                "/dev/snd", "/dev/audio", "/dev/sound", "/dev/cu.", "/dev/tty.",
                "/sys/bus/usb", "/sys/class")):
            _denied.append("private_or_hardware_open")
            raise AssertionError("private_or_audio_device_forbidden")
    if event.startswith("socket."):
        # asyncio's selector needs only an unbound local AF_UNIX socketpair.
        if event == "socket.__new__" and len(args) > 1 and args[1] == 1:
            return
        _denied.append("network")
        raise AssertionError("network_forbidden")
    if event == "subprocess.Popen":
        _denied.append("process")
        raise AssertionError("service_or_process_forbidden")
    if event == "ctypes.dlopen" and args and args[0] is not None:
        library = str(args[0]).lower()
        if any(word in library for word in ("portaudio", "coreaudio", "audiounit", "libusb")):
            _denied.append("native_audio_library")
            raise AssertionError("native_audio_library_forbidden")

class _ModelImports:
    prefixes = ("torch", "tensorflow", "tensorflow_hub", "faster_whisper",
        "onnxruntime", "openwakeword", "transformers", "sentence_transformers",
        "chromadb", "kokoro", "ctranslate2", "cupy", "jax", "jaxlib", "triton")

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in self.prefixes):
            _denied.append("model_import")
            raise AssertionError("real_model_sdk_forbidden")

sys.addaudithook(_audit)
sys.meta_path.insert(0, _ModelImports())

def _before_fixture(*args, **kwargs):
    _denied.append("audio_before_fixture")
    raise AssertionError("sounddevice_acquisition_before_fixture")

sd = ModuleType("sounddevice")
sd.PortAudioError = RuntimeError
for name in ("InputStream", "RawInputStream", "Stream", "RawStream", "OutputStream",
             "rec", "wait", "stop", "play", "query_devices"):
    setattr(sd, name, _before_fixture)
sys.modules["sounddevice"] = sd

# Bypass eager package barrels, but import every nominal production class from
# its actual source module. No replacement loop, controller or owner type.
for name in ("src", "src.audio", "src.pipeline", "src.conversation", "src.memory",
             "src.ambient", "src.stt", "src.wakeword", "src.nlu", "src.rooms",
             "src.users", "src.training", "src.dashboard", "src.evaluation", "src.core",
             "tests", "tests.integration", "tests.mocks"):
    package = ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, package)

from src.audio.echo_suppressor import EchoSuppressor
from src.conversation.follow_up_mode import FollowUpMode
sys.modules["src.audio"].EchoSuppressor = EchoSuppressor
sys.modules["src.conversation"].FollowUpMode = FollowUpMode
'''


def _child_environment():
    environment = {key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE")
        if key in os.environ}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    return environment


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


class NativeInputs:
    def __init__(self, mode="success", *, hold=None):
        self.mode = mode
        self.hold = set(hold or ())
        self.streams = []
        self.entered = {}
        self.releases = {}
        self.open_returns = threading.Event()

    def event(self, name):
        return self.entered.setdefault(name, threading.Event())

    def release_event(self, name):
        return self.releases.setdefault(name, threading.Event())

    def checkpoint(self, name):
        self.event(name).set()
        if name in self.hold:
            assert self.release_event(name).wait(15), f"unreleased SDK fixture gate: {name}"

    def factory(self, **options):
        stream = NativeStream(self, options)
        self.streams.append(stream)
        if self.mode == "unreturned-constructor" and len(self.streams) == 1:
            stream.spawn(emit_callback=False)
            raise RuntimeError("synthetic constructor did not return its native handle")
        self.checkpoint("constructor")
        self.open_returns.set()
        return stream

    def query_devices(self, *args, **kwargs):
        return {"max_input_channels": 1, "name": "synthetic-input", "default_samplerate": 16000}

    def release(self):
        self.hold.clear()
        for event in self.releases.values():
            event.set()
        for stream in self.streams:
            stream.actual_join()


class NativeStream:
    def __init__(self, backend, options):
        self.backend, self.options = backend, options
        self.samplerate = options.get("samplerate", 16000)
        self.channels = options.get("channels", 1)
        self.device, self.dtype = options.get("device", 0), "float32"
        self.pointer = object()
        self.calls = []
        self.worker = None
        self.native_stop = threading.Event()
        self.native_entered = threading.Event()
        self.native_returned = threading.Event()
        self.callback_returned = threading.Event()
        self.worker_failures = []

    def spawn(self, *, emit_callback=True):
        if self.worker is not None:
            return

        def native():
            self.native_entered.set()
            audio = None
            try:
                if emit_callback:
                    audio = np.full((1280, self.channels), .25, np.float32)
                    self.options["callback"](audio, len(audio), None, None)
                    audio = None
                    self.callback_returned.set()
                self.native_stop.wait()
            except BaseException as error:
                self.worker_failures.append(type(error).__name__)
            finally:
                audio = None
                self.native_returned.set()

        self.worker = threading.Thread(target=native, name="synthetic-production-input", daemon=True)
        self.worker.start()
        assert self.native_entered.wait(2)

    def start(self):
        self.spawn()
        self.backend.checkpoint("start")
        if self.backend.mode == "delegated-start-error":
            raise RuntimeError("synthetic start failed after native admission")

    def stop(self, ignore_errors=True):
        self.calls.append(("stop", ignore_errors))
        self.backend.checkpoint("stop")
        if self.backend.mode in {"stop-error", "close-error"}:
            if not ignore_errors:
                raise RuntimeError("synthetic strict stop failure")
            return
        if self.backend.mode == "stop-control":
            raise KeyboardInterrupt("synthetic stop control")
        self.native_stop.set()

    def close(self, ignore_errors=True):
        self.calls.append(("close", ignore_errors))
        # SoundDevice nulls its pointer before reporting a checked error. A
        # later idempotent call is deliberately no independent native receipt.
        if self.pointer is None:
            return
        self.pointer = None
        self.backend.checkpoint("close")
        if self.backend.mode == "close-error":
            if not ignore_errors:
                raise RuntimeError("synthetic strict close failure after pointer clear")
            return
        self.actual_join()
        self.backend.event("close-returned").set()

    def actual_join(self):
        self.native_stop.set()
        if self.worker is not None:
            self.worker.join(3)
            assert self.native_returned.is_set() and not self.worker.is_alive()
            assert self.worker._handle.is_done() and not self.worker_failures


class Wake:
    def load(self):
        pass

    def get_active_models(self):
        return ["synthetic-wake"]

    def detect(self, audio):
        return None

    def reset(self):
        pass


@asynccontextmanager
async def production_graph(tmp_path, monkeypatch, transport, backend):
    from src.audio.echo_suppressor import EchoSuppressor
    from src.conversation.follow_up_mode import FollowUpMode
    from src.memory.capture_control import build_capture_control
    from src.memory.privacy_services import PrivacyServices
    from src.pipeline.audio_loop import AudioLoop
    from src.pipeline.audio_manager import AudioManager
    from src.pipeline.command_processor import CommandProcessor
    from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
    from tests.integration.test_global_capture_control import actor, capture_phase

    with monkeypatch.context() as patch:
        sd = SimpleNamespace(InputStream=backend.factory, query_devices=backend.query_devices,
            PortAudioError=type("PortAudioError", (Exception,), {}))
        patch.setattr("src.pipeline.audio_loop.sd", sd)
        patch.setattr("src.pipeline.multi_room_audio_loop.sd", sd)
        privacy = PrivacyServices(str(tmp_path / "privacy.db"), close_timeout_s=.03)
        await privacy.acall(privacy.initialize)
        follow_up, echo = FollowUpMode(), EchoSuppressor()
        if transport == "single":
            loop = AudioLoop(AudioManager(wake_factory=lambda **kwargs: Wake()), echo,
                follow_up, privacy=privacy, close_timeout_s=.03)
        else:
            room = RoomStream("synthetic-room", 0, Wake(), echo)
            loop = MultiRoomAudioLoop({room.room_id: room}, follow_up, privacy=privacy,
                close_timeout_s=.03, audio_health_path=str(tmp_path / "audio-health.json"))
        processor = CommandProcessor(None, privacy=privacy, close_timeout_s=.03)
        async with AsyncExitStack() as shutdown:
            control = await build_capture_control(privacy=privacy, audio_loop=loop,
                command_processor=processor, shutdown=shutdown)
            await capture_phase(control, "running")
            if transport == "single":
                await loop.start()

            def owner():
                return loop._input if transport == "single" else loop._inputs._owners.get("synthetic-room")

            async def open_input():
                if transport == "single":
                    await loop._input.open()
                    return loop._input.stream
                return await loop._inputs.open("synthetic-room")

            state = SimpleNamespace(loop=loop, privacy=privacy, control=control, processor=processor,
                backend=backend, owner=owner, open=open_input, transport=transport)
            try:
                yield state
            finally:
                # Only fixture-owned gates and native actors are released here.
                errors = []
                try:
                    snapshot = control.snapshot()
                    if not snapshot["target_muted"]:
                        control.request(True, snapshot["revision"], actor())
                except BaseException as error:
                    errors.append(error)
                finally:
                    try:
                        await asyncio.to_thread(backend.release)
                    except BaseException as error:
                        errors.append(error)
                closings = []
                for begin in (control.begin_close, loop.begin_close, processor.begin_close):
                    try:
                        closings.append(begin())
                    except BaseException as error:
                        errors.append(error)
                for closing in closings:
                    try:
                        await asyncio.shield(closing)
                    except BaseException as error:
                        errors.append(error)
                try:
                    await privacy.begin_close()
                except BaseException as error:
                    errors.append(error)
                if errors:
                    raise BaseExceptionGroup("synthetic graph cleanup failed", errors)


def run_isolated(transport, mode, tmp_path):
    command = [sys.executable, "-c",
        _CHILD_BOOTSTRAP + "\nimport asyncio,sys; from pathlib import Path; "
        "from tests.integration.test_production_input_receipts import _isolated_body; "
        "asyncio.run(_isolated_body(sys.argv[1],sys.argv[2],Path(sys.argv[3])))",
        transport, mode, str(tmp_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_child_environment())
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"unknown-owner child did not produce bounded receipt: {stdout}\n{stderr}")
    assert process.wait() == 0, f"unknown-owner child failed: {stdout}\n{stderr}"
    assert "Synthetic private close failure" not in stdout + stderr
    receipts = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"production_input_receipt":')]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["transport"] == transport and receipt["mode"] == mode
    assert receipt["valid"]
    if mode == "legacy-close":
        assert receipt["ack"] is None and receipt["ack_scope"] == "not_measured_legacy_loop_only"
    else:
        assert receipt["ack"] is False
    return receipt
