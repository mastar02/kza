#!/usr/bin/env python3
"""Run the isolated P6/T061 window and preregistered fault campaign."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, closing, contextmanager
from datetime import UTC, datetime
import fcntl
import getpass
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import pwd
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

LOG = logging.getLogger(__name__)
ROOMS = ("living", "hall", "escritorio", "cocina")
SERIES = ("kza", *ROOMS)
WINDOW_S = 259200
THERMOUY = tuple(
    f"thermouy-{job}.{kind}"
    for job in ("tier2-nightly", "ci-prune") for kind in ("timer", "service")
)
PROTECTED = (Path("/home/kza"), Path("/opt/homeassistant"))
FAULTS = {"ha_ws": ("kza",), "microphone": ("cocina",), "global_mute_restore": SERIES}
HA_TOKEN = "t061-synthetic-ha"
CAPTURE = "/api/capture/control"
CANARY = "light.t061_erased"


def guard(*paths: Path) -> None:
    """Reject production accounts and lexical or resolved production paths."""
    uids = {os.getuid(), os.geteuid()}
    if uids & {0, 1000} or getpass.getuser() == "kza" or any(
        pwd.getpwuid(uid).pw_name == "kza" for uid in uids
    ):
        raise ValueError("T061 requires an unprivileged development account other than UID 1000")
    forbidden = (*PROTECTED, *(root.resolve() for root in PROTECTED))
    for path in (Path.cwd(), *paths):
        for form in (Path(os.path.abspath(path.expanduser())), path.expanduser().resolve()):
            if any(form.is_relative_to(root) for root in forbidden):
                raise ValueError(f"protected path: {form}")


def linux_isolation() -> dict:
    """Verify the kernel separation used for an evidentiary run."""
    if sys.platform != "linux" or pwd.getpwuid(os.getuid()).pw_name != "yo":
        raise ValueError("the 72-hour stability window must run on Linux as yo")
    interfaces = [name for _, name in socket.if_nameindex()]
    cpus = sorted(os.sched_getaffinity(0))
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    proc_status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    no_new_privileges = proc_status["NoNewPrivs"].strip() == "1"
    if (interfaces != ["lo"] or cpus != list(range(24, 48)) or nice != 5 or not no_new_privileges
            or any(Path("/dev").glob("nvidia*")) or Path("/dev/dri").exists() or Path("/dev/kfd").exists()):
        raise ValueError("launch the 72-hour window in the isolated T061 user unit")
    return {"interfaces": interfaces, "cpus": cpus, "nice": nice, "no_new_privileges": no_new_privileges}


def thermouy_status() -> dict:
    """Read the state of competing units without modifying systemd."""
    if sys.platform != "linux":
        return {"status": "not_linux", "interference": None}
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", *THERMOUY,
             "--property=Id,LoadState,ActiveState,SubState,NextElapseUSecRealtime"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "interference": None, "error": type(error).__name__}
    units = {}
    for block in result.stdout.strip().split("\n\n"):
        properties = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if properties.get("Id") in THERMOUY:
            units[properties["Id"]] = properties
    running = [name for name, props in units.items() if name.endswith(".service")
               and props.get("ActiveState") not in {"inactive", "failed", None}]
    scheduled = [name for name, props in units.items() if name.endswith(".timer")
                 and props.get("ActiveState") == "active"]
    known = set(units) == set(THERMOUY) and all(
        unit.get("LoadState") in {"loaded", "not-found", "masked"} and unit.get("ActiveState")
        for unit in units.values()
    )
    return {"status": "observed" if known else "incomplete", "returncode": result.returncode,
            "units": units, "interference": bool(running or scheduled) if known else None,
            "running_services": running, "scheduled_timers": scheduled}


def utc(second: int | float) -> str:
    """Format a UTC timestamp for evidence."""
    return datetime.fromtimestamp(second, UTC).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path: Path, data: dict) -> None:
    """Publish derived JSON atomically."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        try:
            json.dump(data, target, allow_nan=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def records(path: Path):
    """Read complete evidence rows, ignoring an interrupted trailing write."""
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.endswith("\n"):
                    yield json.loads(line)


@contextmanager
def locked(state: Path):
    """Exclude concurrent phases or summaries over in-flight logs."""
    guard(state)
    if (not state.is_dir() or state != state.resolve() or state.stat().st_uid != os.getuid()
            or stat.S_IMODE(state.stat().st_mode) != 0o700):
        raise ValueError("state must be the private directory created by preflight")
    fd = os.open(state / ".lock", os.O_CREAT | os.O_RDONLY | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("soak lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another soak command owns this state") from error
        if any(path.is_symlink() or not (path.is_dir() or path.is_file()) for path in state.rglob("*")):
            raise ValueError("symlinks and special files are forbidden in soak state")
        yield fd
    finally:
        os.close(fd)


def preflight(archive: Path, review: Path, state_root: Path) -> dict:
    """Verify a snapshot with the release reviewer and extract read-only runtime."""
    guard(archive, review, state_root, Path(sys.executable))
    from scripts.stage_runtime import make_runtime_read_only
    from tools.release_manifest_review import _archive_snapshot, _read_bundle, _unique_object

    if not review.is_file():
        raise ValueError("release-review must be a regular file")
    raw_review = review.read_bytes()
    seal = json.loads(raw_review, object_pairs_hook=_unique_object)
    commit, artifact = seal["git_commit"], seal["artifact"]
    expected = artifact["sha256"]
    if (seal["manifest_version"] != 1 or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise ValueError("invalid release-review identity")
    state = state_root / commit
    guard(state)
    if state != state.resolve():
        raise ValueError("state must be a canonical path")
    state.mkdir(parents=True, mode=0o700)
    try:
        snapshot = state / "runtime.tar.gz"
        size = _archive_snapshot(archive, snapshot, expected)
        manifest, files = _read_bundle(snapshot, commit)
        expected_artifact = {
            "sha256": expected, "bytes": size, "format": manifest["format"],
            "entrypoint": manifest["entrypoint"], "python_abi": "cp313",
            "cache_tag": "cpython-313", "python_magic": manifest["python_magic"],
            "python_sources": 0, "runtime_files": len(files),
        }
        if artifact != expected_artifact:
            raise ValueError("release-review artifact metadata differs from the verified archive")
        runtime = state / "runtime"
        runtime.mkdir()
        for name, data in files.items():
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        release = runtime / "release.json"
        save_json(release, manifest)
        check = {
            "commit": commit, "artifact": artifact, "runtime_manifest_sha256": digest(release),
            "review_sha256": hashlib.sha256(raw_review).hexdigest(), "state": str(state),
            "checked_at_utc": utc(time.time()), "bind_host": "127.0.0.1", "bind_port": 0,
            "gpu": False, "network": "loopback", "thermouy": thermouy_status(),
            "fault_protocol": {name: list(series) for name, series in FAULTS.items()},
        }
        (state / "work").mkdir(mode=0o700)
        snapshot.unlink()
        make_runtime_read_only(runtime)
        save_json(state / "preflight.json", check)
        return check
    except BaseException:
        for directory, _, _ in os.walk(state / "runtime"):
            Path(directory).chmod(0o700)
        shutil.rmtree(state)
        raise


def verify_runtime(state: Path, receipt: dict) -> Path:
    """Check extracted bytes and file modes again before any candidate import."""
    runtime = state / "runtime"
    release = runtime / "release.json"
    if digest(release) != receipt["runtime_manifest_sha256"]:
        raise ValueError("runtime manifest changed since preflight")
    files = json.loads(release.read_text())["files"]
    actual = {path.relative_to(runtime).as_posix(): digest(path)
              for path in runtime.rglob("*") if path.is_file() and path != release}
    if actual != files:
        raise ValueError("runtime bytes changed since preflight")
    if any(stat.S_IMODE(path.stat().st_mode) != (0o555 if path.is_dir() else 0o444)
           for path in (runtime, *runtime.rglob("*"))):
        raise ValueError("runtime must remain read-only")
    return runtime


def systemd_quote(value: Path | str) -> str:
    return json.dumps(str(value).replace("%", "%%").replace("$", "$$"), ensure_ascii=False)


def render_unit(state: Path, duration: int, faults_first: bool) -> str:
    """Render both phases in one user cgroup and summarize even on exit failure."""
    commands = ["run-soak", "inject-faults"]
    if faults_first:
        commands.reverse()
    prefix = " ".join(map(systemd_quote, (Path(sys.executable), "-B", "-m", "tools.p6_local_soak")))
    lines = [
        "[Unit]", f"Description=KZA P6 local soak {state.name}", "", "[Service]",
        "Type=oneshot", f"WorkingDirectory={systemd_quote(Path(__file__).resolve().parents[1])}",
        "UMask=0077", "CPUAffinity=24-47", "Nice=5", "MemoryHigh=24G", "NoNewPrivileges=yes",
        "PrivateDevices=yes", "PrivateNetwork=yes", "IPAddressDeny=any", "IPAddressAllow=localhost",
        f"InaccessiblePaths={' '.join(f'-{path}' for path in PROTECTED)}", "TimeoutStartSec=infinity",
        "TimeoutStopSec=30s", "Restart=no",
    ]
    for command in commands:
        args = f" --duration-s {duration}" if command == "run-soak" else ""
        lines.append(f"ExecStart={prefix} {command} --state {systemd_quote(state)}{args}")
    lines.append(f"ExecStopPost={prefix} summarize --state {systemd_quote(state)}")
    return "\n".join(lines) + "\n"


def offline_io(event: str, args: tuple) -> None:
    """Reject Python network operations outside numeric loopback and child processes."""
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
                 "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}:
        if event in {"socket.connect", "socket.bind", "socket.sendto"}:
            if args[0].family not in {socket.AF_INET, socket.AF_INET6}:
                raise PermissionError("only loopback TCP/IP is allowed")
            host = args[1][0]
        else:
            host = args[0][0] if event == "socket.getnameinfo" else args[0]
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise PermissionError("external network and DNS are disabled")
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty"}:
        raise PermissionError("child processes are disabled in the candidate")


def worker_env(state: Path) -> dict[str, str]:
    """Pass an explicit environment without house credentials or GPU selectors."""
    return {
        "PATH": os.defpath, "HOME": str(state / "work"), "TMPDIR": str(state / "work"),
        "XDG_CACHE_HOME": str(state / "work" / "cache"),
        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "ANONYMIZED_TELEMETRY": "False",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "KZA_DASHBOARD_TOKEN": secrets.token_hex(32),
    }


def isolate_process(state: Path, runtime: Path) -> None:
    """Load the app only from read-only bytecode and enforce offline I/O."""
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("candidate modules already imported")
    sys.path.insert(0, str(runtime))
    os.chdir(state / "work")
    sys.addaudithook(offline_io)


class HomeSimulator:
    """Serve the synthetic HA API and authenticated entity events on loopback."""

    def __init__(self) -> None:
        self.online = True
        self.sockets = set()
        self.subscribers = {}
        self.rest_fault_reads = 0
        self.ws_fault_attempts = 0
        self.initial_frames = 0

    async def rest(self, request):
        from aiohttp import web

        if request.headers.get("Authorization") != f"Bearer {HA_TOKEN}":
            return web.Response(status=401)
        if not self.online:
            self.rest_fault_reads += 1
            await asyncio.sleep(0.2)
            return web.Response(status=503)
        if request.path == "/api/":
            return web.json_response({"message": "API running"})
        return web.json_response([{"entity_id": "light.synthetic", "state": "off", "attributes": {}}])

    async def websocket(self, request):
        from aiohttp import web

        if not self.online:
            self.ws_fault_attempts += 1
            return web.Response(status=503)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.add(ws)
        try:
            await ws.send_json({"type": "auth_required"})
            auth = await ws.receive_json()
            if auth != {"type": "auth", "access_token": HA_TOKEN}:
                await ws.send_json({"type": "auth_invalid"})
                return ws
            await ws.send_json({"type": "auth_ok"})
            async for message in ws:
                body = message.json()
                identifier = body["id"]
                if body["type"] == "subscribe_entities":
                    await ws.send_json({"id": identifier, "type": "result", "success": True})
                    await self.event(ws, identifier, {"a": {
                        "light.synthetic": {"s": "off", "a": {}, "c": "t061", "lc": 1, "lu": 1},
                    }})
                    self.subscribers[ws] = identifier
                    self.initial_frames += 1
                else:
                    await ws.send_json({"id": identifier, "type": "result", "success": False,
                                        "error": {"code": "not_supported", "message": "synthetic read only"}})
        finally:
            self.sockets.discard(ws)
            self.subscribers.pop(ws, None)
            await ws.close()
        return ws

    @staticmethod
    async def event(ws, identifier: int, event: dict) -> None:
        await ws.send_json({"id": identifier, "type": "event", "event": event})

    async def tick(self) -> None:
        while True:
            await asyncio.sleep(1)
            for ws, identifier in tuple(self.subscribers.items()):
                try:
                    await self.event(ws, identifier, {"c": {}})
                except ConnectionError:
                    pass

    async def cut(self) -> None:
        self.online = False
        await asyncio.gather(*(ws.close() for ws in tuple(self.sockets)))

    @asynccontextmanager
    async def serve(self):
        from aiohttp import web

        server = web.Application()
        server.add_routes([web.get("/api/", self.rest), web.get("/api/states", self.rest),
                           web.get("/api/websocket", self.websocket)])
        runner = web.AppRunner(server, access_log=None)
        await runner.setup()
        tick = asyncio.create_task(self.tick())
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            yield f"http://127.0.0.1:{runner.addresses[0][1]}"
        finally:
            await cancel(tick)
            await self.cut()
            await runner.cleanup()


class Microphones:
    """Supply four PortAudio handles with independently controllable audio and catalog."""

    class PortAudioError(Exception):
        pass

    class Stream:
        latency = 0.0

        def __init__(self, device, callback, samplerate, channels, dtype) -> None:
            self.device, self.callback = device, callback
            self.samplerate, self.channels, self.dtype = samplerate, channels, dtype
            self.active = False

        def start(self) -> None:
            self.active = True

        def stop(self, *, ignore_errors: bool = True) -> None:
            self.active = False

        abort = close = stop

    def __init__(self) -> None:
        self.connected = True
        self.enumerated_at = 0.0
        self.late_catalog_misses = 0
        self.reopen_counts = dict.fromkeys(ROOMS, 0)
        self.streams = {}

    def query_devices(self, index: int) -> dict:
        if ROOMS[index] == "cocina" and (not self.connected or time.monotonic() < self.enumerated_at):
            if self.connected:
                self.late_catalog_misses += 1
            raise self.PortAudioError("synthetic device absent from catalog")
        return {"max_input_channels": 1, "name": f"T061 {ROOMS[index]}"}

    def InputStream(self, *, device: int, callback, samplerate: int, channels: int, dtype: str, blocksize: int):
        self.query_devices(device)
        stream = self.streams[device] = self.Stream(device, callback, samplerate, channels, dtype)
        self.reopen_counts[ROOMS[device]] += 1
        return stream

    @staticmethod
    def _terminate() -> None:
        pass

    @staticmethod
    def _initialize() -> None:
        pass

    async def feed(self) -> None:
        import numpy as np

        pcm = np.full((1280, 1), 0.025, dtype=np.float32)
        while True:
            for index in range(len(ROOMS)):
                stream = self.streams.get(index)
                if stream and stream.active and (ROOMS[index] != "cocina" or self.connected):
                    stream.callback(pcm, len(pcm), None, None)
            await asyncio.sleep(0.08)


def configuration(url: str) -> dict:
    """Use explicit fixture settings, including storage inside private work."""
    return {
        "paths": {"models_root": str(Path("models").absolute())},
        "home_assistant": {"url": url, "token": HA_TOKEN, "timeout": 1.0,
                           "state_prefetch": {"enabled": True}},
        "embeddings": {"model": "synthetic", "device": "cpu"},
        "reasoner": {"mode": "http", "cloud": {"consent": False}},
        "router": {"enabled": False}, "speaker_id": {"enabled": False},
        "emotion": {"enabled": False}, "monitoring": {"enabled": False},
        "analytics": {"enabled": True, "events_db": "data/events.db"},
        "observability": {"metrics_path": "data/metrics.jsonl"},
        "memory": {"enabled": False}, "orchestrator": {"enabled": True},
        "rooms": {"enabled": True, "stream_watchdog": {"enabled": True},
                  **{room: {"name": room, "mic_device_index": index} for index, room in enumerate(ROOMS)}},
        "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 0},
        "zones": {"enabled": False}, "ambient": {"enabled": False}, "alerts": {"enabled": False},
        "warmup": {"enabled": False}, "tts": {"streaming": {"enabled": False}},
        "training": {"nightly": {"enabled": False}},
    }


def candidate_factories():
    """Replace native model leaves while retaining production resource owners."""
    from types import SimpleNamespace

    import numpy as np
    from src.ambient import segmenter
    from src.application import ApplicationFactories
    from src.stt.parakeet_stt import ParakeetSTT
    from src.tts.piper_tts import KokoroTTS
    from src.vectordb import chroma_sync
    from src.wakeword.detector import WakeWordDetector

    class Encoder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, text, **kwargs):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    class Wake(WakeWordDetector):
        last_pcm = 0.0

        def load(self) -> None:
            pass

        def detect(self, audio):
            if np.any(audio):
                self.last_pcm = time.monotonic()
            return None

    class VAD(segmenter._SileroPredictor):
        def __init__(self) -> None:
            super().__init__(None)

        def __call__(self, audio) -> float:
            with self._silero_admit():
                return 0.9 if np.any(audio) else 0.0

        def reset(self) -> None:
            with self._silero_admit():
                pass

    def stt(_):
        model = ParakeetSTT()
        model._model = SimpleNamespace(recognize=lambda *_args, **_kw: "reactivá el micrófono")
        return model

    def tts(_):
        model = KokoroTTS(device="cpu")
        model._model = lambda *_args, **_kw: iter([SimpleNamespace(audio=np.zeros(240, dtype=np.float32))])
        return model

    chroma_sync.SentenceTransformer = Encoder
    segmenter.make_silero_predictor = VAD
    return ApplicationFactories(stt=stt, tts=tts, wake_detector=Wake, preflight=lambda: None)


async def cancel(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def wait_for(check):
    """Poll for a real observation inside the caller's deadline."""
    while not (value := await check()):
        await asyncio.sleep(0.05)
    return value


def room_useful(room: dict) -> bool:
    age = room.get("age_s")
    return (room.get("ever") is True and type(age) in (int, float) and 0 <= age <= 1)


def ha_healthy(channels: dict) -> bool:
    return all(channels.get(name, {}).get("state") == "connected" and channels[name].get("current") is True
               for name in ("rest", "events"))


class Candidate:
    """Own the real application lifecycle and query its operator HTTP endpoints."""

    def __init__(self, state: Path, home: HomeSimulator, url: str) -> None:
        self.state, self.home, self.url = state, home, url
        self.graph = self.task = self.http = None
        self.port = None
        self.mics = Microphones()
        self.authenticated_mute = False

    @asynccontextmanager
    async def open(self, *, muted: bool = False):
        from src.audio.microphone_status import UsbInventory
        from src.main import build_application
        from src.pipeline import multi_room_audio_loop

        self.mics = Microphones()
        multi_room_audio_loop.sd = self.mics
        async with AsyncExitStack() as stack:
            feed = asyncio.create_task(self.mics.feed())
            stack.push_async_callback(cancel, feed)
            self.graph = await build_application(
                configuration(self.url), shutdown=stack, factories=candidate_factories(), verify_only=False,
            )
            self.graph.pipeline.audio_loop._microphone_reader = lambda _: UsbInventory(
                observed_at=time.time(), observed_monotonic=time.monotonic(), success=True,
            )
            task = self.task = asyncio.create_task(self.graph.run())
            stack.push_async_callback(self.stop, task)
            self.authenticated_mute = muted
            try:
                async with asyncio.timeout(45):
                    await wait_for(self.connect)
                    stack.push_async_callback(self.http.aclose)
                    await wait_for(self.ready)
                yield self
            finally:
                self.http = self.task = self.port = None

    @staticmethod
    async def stop(task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def connect(self) -> bool:
        import httpx

        if self.task.done():
            self.task.result()
            raise RuntimeError("candidate stopped before dashboard readiness")
        server = getattr(self.graph.dashboard, "_server", None)
        if not getattr(server, "started", False):
            return False
        self.port = server.servers[0].sockets[0].getsockname()[1]
        logging.getLogger("uvicorn.access").disabled = True
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.http = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{self.port}", trust_env=False, timeout=0.6,
            headers={"Authorization": f"Bearer {os.environ['KZA_DASHBOARD_TOKEN']}"},
        )
        response = await self.http.get("/api/health")
        if response.status_code != 200:
            await self.http.aclose()
            self.http = None
            raise RuntimeError(f"dashboard health returned {response.status_code}")
        return True

    async def ready(self) -> bool:
        await self.graph.ha_client.get_state_snapshot()
        sample = await self.observe()
        return sample["kza_useful"] and (
            self.authenticated_mute or all(room_useful(sample["rooms"][room]) for room in ROOMS)
        )

    async def refresh_rest(self) -> None:
        """Keep REST channel evidence fresh under the steady synthetic workload."""
        while True:
            await self.graph.ha_client.get_state_snapshot()
            await asyncio.sleep(0.5)

    async def observe(self) -> dict:
        import httpx

        sample = {"http_status": None, "health": "unreachable", "ha": {}, "kza_useful": False,
                  "subsystems": {},
                  "rooms": {room: {"ever": False, "age_s": None} for room in ROOMS}, "pause": None}
        client, task, graph = self.http, self.task, self.graph
        if client is None or task.done():
            return sample
        try:
            response = await client.get("/api/health")
        except httpx.TransportError:
            return sample
        sample["http_status"] = response.status_code
        if response.status_code != 200:
            return sample
        health = response.json()
        subsystems = {value["name"]: value for value in health["subsystems"]}
        capture = graph.capture_control.snapshot()
        if self.authenticated_mute and all(capture[key] is True for key in (
            "persisted_muted", "target_muted", "total_mute_confirmed", "ingress_sealed",
        )):
            sample["pause"] = "authenticated_global_mute"
        audio = subsystems.get("audio_capture", {}).get("extra", {}).get("rooms", {})
        sample["rooms"] = {room: audio.get(room, default) for room, default in sample["rooms"].items()}
        channels = subsystems.get("home_assistant", {}).get("extra", {}).get("channels", {})
        sample["ha"] = {key: channels[key] for key in ("rest", "events") if key in channels}
        sample["health"] = health["status"]
        statuses = sample["subsystems"] = {name: value["status"] for name, value in subsystems.items()}
        core = statuses.keys() - {"audio_capture", "latency_monitor"}
        required = {"home_assistant", "priority_queue", "reminders", "command_index", "voice_pipeline"}
        sample["kza_useful"] = (ha_healthy(sample["ha"]) and required <= core
                                and all(statuses[name] == "healthy" for name in core)
                                and statuses.get("latency_monitor") != "unhealthy")
        return sample

    async def capture_state(self) -> dict:
        response = await self.http.get(CAPTURE)
        response.raise_for_status()
        return response.json()

    async def set_muted(self, muted: bool) -> bool:
        """Mutate through the production operator cookie session only."""
        import httpx

        async with httpx.AsyncClient(base_url=self.http.base_url, trust_env=False, timeout=1) as client:
            login = await client.post("/api/admin/auth/login", json={"token": os.environ["KZA_DASHBOARD_TOKEN"]})
            if login.status_code != 200 or not client.cookies:
                return False
            before = await self.capture_state()
            reply = await client.post(CAPTURE, json={"muted": muted, "expected_revision": before["revision"]})
            if reply.status_code != 202:
                return False

            async def settled():
                state = await self.capture_state()
                return (state["persisted_muted"] is muted and state["target_muted"] is muted
                        and state["phase"] == ("muted" if muted else "running")
                        and state["ingress_sealed"] is muted and state["total_mute_confirmed"] is muted)

            async with asyncio.timeout(10):
                await wait_for(settled)
            self.authenticated_mute = muted
            return True


def journal(state: Path, kind: str, **values) -> None:
    now = time.time()
    append(state / "faults.jsonl", {"kind": kind, "at": now, "utc": utc(now), **values})


async def sample_window(candidate: Candidate, phase: str, start: int, end: int | None) -> None:
    """Schedule UTC slots independently of probe duration and never backfill."""
    loop = asyncio.get_running_loop()
    wall, mono = time.time(), loop.time()
    slot = start
    stop = None if end is None else mono + end - wall
    while end is None or (slot < end and loop.time() < stop):
        await asyncio.sleep(max(0, mono + slot - wall - loop.time()))
        current = int(time.time())
        if current < slot:
            await asyncio.sleep(0.05)
            continue
        if end is not None and current >= end:
            break
        slot = current
        sample = await candidate.observe()
        observed = time.time()
        if int(observed) == slot:
            append(candidate.state / "samples.jsonl", {
                "phase": phase, "second": slot, "observed_at": observed, "utc": utc(observed), **sample,
            })
        slot += 1
    if stop is not None:
        await asyncio.sleep(max(0, stop - loop.time()))


async def stability(candidate: Candidate, duration: int) -> None:
    async with candidate.open():
        start = math.ceil(time.time())
        journal(candidate.state, "window", start=start, end=start + duration, duration_s=duration,
                dashboard_port=candidate.port)
        refresh = asyncio.create_task(candidate.refresh_rest())
        try:
            await sample_window(candidate, "stability", start, start + duration)
        finally:
            await cancel(refresh)
    journal(candidate.state, "window_finished")


async def lag_probe(lags: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        deadline = loop.time() + 0.005
        await asyncio.sleep(0.005)
        lags.append(max(0.0, loop.time() - deadline) * 1000)


async def fault_ha(candidate: Candidate) -> dict:
    """Cut live WS and delay/return 503 on ten real REST observations."""
    home = candidate.home
    lags, observations = [], []
    old_frames = home.initial_frames
    beat = asyncio.create_task(lag_probe(lags))
    try:
        await home.cut()
        for _ in range(10):
            snapshot = await candidate.graph.ha_client.get_state_snapshot()
            observations.append(not snapshot.available)
        sample = await candidate.observe()

        async def rejected_ws():
            return home.ws_fault_attempts > 0

        async with asyncio.timeout(15):
            await wait_for(rejected_ws)
    finally:
        await cancel(beat)
        home.online = True
    restored = time.monotonic()
    ordered = sorted(lags)
    evidence = {
        "fault_reads": home.rest_fault_reads, "ws_rejected_connections": home.ws_fault_attempts,
        "http_status": sample["http_status"], "health": sample["health"], "ha_channels": sample["ha"],
        "degraded": (all(observations) and sample["ha"].get("rest", {}).get("state") != "connected"
                     and sample["ha"].get("events", {}).get("state") != "connected"
                     and (sample["http_status"] == 503 or sample["health"] in {"degraded", "unhealthy"})),
        "h2": {"interval_ms": 5, "stall_ms": 200, "repetitions": len(observations),
               "samples": len(ordered),
               "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
               "max_ms": max(ordered, default=None), "delays_ms": lags},
        "recovery_s": None,
    }

    async def recovered():
        if not (await candidate.graph.ha_client.get_state_snapshot()).available:
            return False
        return home.initial_frames > old_frames and ha_healthy((await candidate.observe())["ha"])

    try:
        async with asyncio.timeout(60):
            await wait_for(recovered)
            evidence["recovery_s"] = time.monotonic() - restored
    except TimeoutError:
        pass
    return evidence


async def fault_microphone(candidate: Candidate) -> dict:
    """Keep healthy handles intact while the watchdog reopens only the kitchen."""
    mics = candidate.mics
    loop = candidate.graph.pipeline.audio_loop
    kitchen = ROOMS[-1]
    counts = dict(mics.reopen_counts)
    misses = mics.late_catalog_misses
    others = ROOMS[:-1]
    max_ages = dict.fromkeys(others, 0.0)
    mics.connected = False

    async def kitchen_retired():
        health = loop.get_audio_health()["rooms"]
        return not room_useful(health[kitchen]) and not mics.streams[3].active

    async def monitor():
        while True:
            health = loop.get_audio_health()["rooms"]
            for room in others:
                max_ages[room] = max(max_ages[room], health[room]["age_s"])
            await asyncio.sleep(0.05)

    watching = asyncio.create_task(monitor())
    recovered_at, enumerated_at = None, None
    try:
        async with asyncio.timeout(15):
            await wait_for(kitchen_retired)
        mics.enumerated_at = enumerated_at = time.monotonic() + 2.0
        mics.connected = True

        async def captured():
            stamp = loop.room_streams[kitchen].wake_detector.last_pcm
            return stamp if stamp >= enumerated_at and mics.reopen_counts[kitchen] > counts[kitchen] else None

        async with asyncio.timeout_at(enumerated_at + 180):
            recovered_at = await wait_for(captured)
    except TimeoutError:
        pass
    finally:
        if watching.done():
            watching.result()
        await cancel(watching)
        mics.connected = True
        mics.enumerated_at = 0
    return {"room": kitchen, "enumeration_mono_s": enumerated_at,
            "recovery_s": recovered_at - enumerated_at if recovered_at is not None else None,
            "late_catalog_misses": mics.late_catalog_misses - misses, "healthy_max_age_s": max_ages,
            "healthy_handles_unchanged": all(mics.reopen_counts[room] == counts[room] for room in others)}


def canary_rows(path: Path) -> int:
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT COUNT(*) FROM events_policy_v1 WHERE entity_id=?", (CANARY,)).fetchone()[0]


def sqlite_copy(source: Path, destination: Path) -> None:
    with (closing(sqlite3.connect(source)) as reader, closing(sqlite3.connect(destination)) as writer):
        reader.backup(writer)


async def fault_mute(candidate: Candidate, original: AsyncExitStack) -> dict:
    """Mute with an operator cookie, rebuild the graph and restore an older producer copy."""
    import httpx
    import numpy as np
    from src.memory.fact_ledger import MemoryReviewer
    from src.pipeline.command_event import CommandEvent

    database = Path("data/events.db")
    backup = Path("pre-erasure-events.db")
    graph = candidate.graph
    source = graph.privacy.source_for_capture(str(uuid4()), time.time(), ROOMS[0])
    await graph.dashboard.event_logger.alog(CANARY, "synthetic", sources=(source,))
    await asyncio.to_thread(sqlite_copy, database, backup)
    evidence = {"rows_in_older_copy": await asyncio.to_thread(canary_rows, backup)}
    await graph.erasure_services.request(source.conversation_id, MemoryReviewer("t061"))
    async with asyncio.timeout(10):
        while await asyncio.to_thread(canary_rows, database):
            await graph.erasure_services.reconcile()
            await asyncio.sleep(0.05)
    evidence["erased_before_restore"] = not graph.privacy.sources_available((source,))
    evidence["operator_mute_confirmed"] = await candidate.set_muted(True)
    if not evidence["operator_mute_confirmed"]:
        return evidence
    await original.aclose()
    await asyncio.to_thread(sqlite_copy, backup, database)
    evidence["rows_copied_back"] = await asyncio.to_thread(canary_rows, database)
    async with candidate.open(muted=True):
        graph = candidate.graph
        evidence["source_denied_on_restart"] = not graph.privacy.sources_available((source,))

        async def muted():
            state = await candidate.capture_state()
            return state["total_mute_confirmed"] and state["persisted_muted"] and state["ingress_sealed"]

        async with asyncio.timeout(10):
            evidence["mute_survived_restart"] = await wait_for(muted)
        # The synthetic STT leaf would ask to reactivate; the actual capture gate must reject first.
        command = CommandEvent(np.full(16000, 0.025, dtype=np.float32), ROOMS[0],
                               capture_complete=True, capture_trigger="wake")
        reply = await graph.pipeline.process_command(command)
        evidence["voice_rejected"] = (reply.get("intent") == "cancelled"
                                      and reply.get("success") is False and await muted())
        current = await candidate.capture_state()
        async with httpx.AsyncClient(base_url=candidate.http.base_url, trust_env=False, timeout=1) as anonymous:
            reply = await anonymous.post(CAPTURE, json={"muted": False, "expected_revision": current["revision"]})
        evidence["unauthenticated_rejected"] = reply.status_code in {401, 403} and await muted()
        evidence["operator_resume_confirmed"] = await candidate.set_muted(False)
        evidence["source_still_denied"] = not graph.privacy.sources_available((source,))
        async with asyncio.timeout(10):
            while await asyncio.to_thread(canary_rows, database):
                await graph.erasure_services.reconcile()
                await asyncio.sleep(0.05)
        evidence["resurrected_rows"] = await asyncio.to_thread(canary_rows, database)
        async with asyncio.timeout(10):
            await wait_for(candidate.ready)
    backup.unlink()
    return evidence


async def exercise(candidate: Candidate, name: str, scenario, *args) -> None:
    state = candidate.state
    journal(state, "fault_started", name=name)
    try:
        async with asyncio.timeout(230):
            result = await scenario(candidate, *args)
    except Exception as error:
        LOG.exception("T061 fault failed: %s", name)
        result = {"error": type(error).__name__}
    journal(state, "fault_finished", name=name, evidence=result)


async def inject(candidate: Candidate) -> None:
    state = candidate.state
    async with AsyncExitStack() as original:
        await original.enter_async_context(candidate.open())
        start = math.ceil(time.time())
        journal(state, "fault_plan", start=start,
                scenarios={name: list(series) for name, series in FAULTS.items()},
                dashboard_port=candidate.port)
        await asyncio.sleep(max(0, start - time.time()))
        sampling = asyncio.create_task(sample_window(candidate, "faults", start, None))
        try:
            await exercise(candidate, "ha_ws", fault_ha)
            await exercise(candidate, "microphone", fault_microphone)
            await exercise(candidate, "global_mute_restore", fault_mute, original)
            end = math.ceil(time.time())
            await asyncio.sleep(max(0, end - time.time()))
            if sampling.done():
                sampling.result()
        finally:
            await cancel(sampling)
    journal(state, "fault_plan_finished", end=end)


def availability(start: int, end: int, samples: dict, planned: list) -> dict:
    """Count each fixed UTC slot for each series, with no retrospective exclusions."""
    if end <= start:
        raise ValueError("invalid availability interval")
    fields = ("useful_seconds", "authenticated_pause_seconds",
              "planned_unavailable_seconds", "spontaneous_unavailable_seconds",
              "unclassified_gap_seconds")
    counts = {key: dict.fromkeys(fields, 0) for key in SERIES}
    gaps = 0
    for second in range(start, end):
        row = samples.get(second)
        if row is None:
            gaps += 1
        paused = row is not None and row.get("pause") == "authenticated_global_mute"
        observed = row["observed_at"] if row else second
        affected = {
            key for left, right, keys in planned
            if (left <= observed < right if row else left < second + 1 and second < right)
            for key in keys
        }
        for key in SERIES:
            series = counts[key]
            if row is None:
                field = "unclassified_gap_seconds"
            elif key != "kza" and paused:
                field = "authenticated_pause_seconds"
            elif (row.get("kza_useful") is True if key == "kza" else room_useful(row["rooms"].get(key, {}))):
                field = "useful_seconds"
            elif key in affected:
                field = "planned_unavailable_seconds"
            else:
                field = "spontaneous_unavailable_seconds"
            series[field] += 1
    seconds = end - start
    for series in counts.values():
        series["availability"] = series["useful_seconds"] / seconds
        series["threshold_met"] = series["useful_seconds"] * 1000 >= seconds * 999
    return {"start_utc": utc(start), "end_utc_exclusive": utc(end), "window_seconds": seconds,
            "samples": seconds - gaps, "gap_seconds": gaps, "series": counts}


def at_most(value, limit: float) -> bool:
    return type(value) in (float, int) and 0 <= value <= limit


def verdicts(block: dict | None, finished: bool, faults: dict, faults_finished: bool) -> dict:
    """Require positive observations for each local contract."""
    full = block is not None and finished and block["window_seconds"] >= WINDOW_S
    c19 = full and all(series["threshold_met"] for series in block["series"].values())
    ha = faults.get("ha_ws", {})
    mic = faults.get("microphone", {})
    h2 = ha.get("h2", {})
    mute = faults.get("global_mute_restore", {})
    healthy_ages = mic.get("healthy_max_age_s", {})
    c20 = (faults_finished and ha.get("degraded") is True
           and at_most(ha.get("recovery_s"), 60) and mic.get("room") == "cocina"
           and at_most(mic.get("recovery_s"), 180) and mic.get("healthy_handles_unchanged") is True
           and mic.get("late_catalog_misses", 0) > 0 and set(healthy_ages) == set(ROOMS[:-1])
           and all(at_most(age, 1) for age in healthy_ages.values()))
    c03 = (faults_finished and ha.get("degraded") is True and ha.get("fault_reads", 0) >= 10
           and ha.get("ws_rejected_connections", 0) > 0 and h2.get("repetitions") == 10
           and h2.get("interval_ms") == 5 and h2.get("stall_ms") == 200 and h2.get("samples", 0) >= 200
           and at_most(h2.get("p95_ms"), 25) and at_most(h2.get("max_ms"), 75))
    c04 = (faults_finished and all(mute.get(key) is True for key in (
        "erased_before_restore", "operator_mute_confirmed", "mute_survived_restart", "voice_rejected",
        "unauthenticated_rejected", "operator_resume_confirmed", "source_denied_on_restart", "source_still_denied",
    )) and mute.get("rows_in_older_copy") == 1 and mute.get("rows_copied_back") == 1
           and mute.get("resurrected_rows") == 0)
    return {"c19_local_window_met": full, "c19_72h_passed": c19, "c20_faults_passed": c20,
            "c03_h2_passed": c03, "c04_mute_passed": c04,
            "overall_t061_local_passed": full and c19 and c20 and c03 and c04}


def summarize(state: Path, receipt: dict) -> dict:
    """Derive block and total availability from the two append-only journals."""
    events = list(records(state / "faults.jsonl"))
    windows = {event["kind"]: event for event in events}
    window = windows.get("window")
    plan = windows.get("fault_plan")
    done = windows.get("fault_plan_finished")
    starts = {event["name"]: event["at"] for event in events if event["kind"] == "fault_started"}
    ends = {event["name"]: event for event in events if event["kind"] == "fault_finished"}
    outcomes = {name: event["evidence"] for name, event in ends.items()}
    rows = {"stability": {}, "faults": {}}
    duplicate_seconds = {phase: set() for phase in rows}
    for row in records(state / "samples.jsonl"):
        phase, second, observed = row["phase"], row["second"], row["observed_at"]
        if type(second) is not int or int(observed) != second:
            raise ValueError("sample does not belong to its UTC slot")
        bucket = rows[phase]
        if second in bucket or second in duplicate_seconds[phase]:
            duplicate_seconds[phase].add(second)
            bucket.pop(second, None)
        else:
            bucket[second] = row

    block = availability(window["start"], window["end"], rows["stability"], []) if window else None
    fault_end, planned, fault_counts = None, [], None
    if plan:
        fault_end = (done["end"] if done else max(plan["start"] + 1,
            math.ceil(max(event["at"] for event in events)), max(rows["faults"], default=0) + 1))
        planned = [(start, ends.get(name, {"at": fault_end})["at"], plan["scenarios"][name])
                   for name, start in starts.items()]
        fault_counts = availability(plan["start"], fault_end, rows["faults"], planned)
    windows_disjoint = not (window and plan and plan["start"] < window["end"] and window["start"] < fault_end)
    all_scans = [part for part in (block, fault_counts) if part is not None]
    total = None
    if all_scans and windows_disjoint:
        start = min(value["start"] for value in (window, plan) if value)
        end = max(value for value in (window["end"] if window else None, fault_end) if value is not None)
        admitted = {}
        for phase, left, right in (("stability", window["start"] if window else 0, window["end"] if window else 0),
                                  ("faults", plan["start"] if plan else 0, fault_end or 0)):
            admitted.update((second, row) for second, row in rows[phase].items() if left <= second < right)
        total = availability(start, end, admitted, planned)

    passed_plan = (windows_disjoint and done is not None and set(starts) == set(ends) == set(FAULTS)
                   and plan["scenarios"] == {name: list(series) for name, series in FAULTS.items()}
                   and all(plan["start"] <= starts[name] <= ends[name]["at"] <= fault_end for name in starts))
    journal_hashes = {name: digest(state / name) for name in ("samples.jsonl", "faults.jsonl")
                      if (state / name).exists()}
    result = {"candidate": receipt, "stability": block, "fault_phase": fault_counts, "total": total,
              "journal_sha256": journal_hashes, "linux_at_start": windows.get("linux_at_start"),
              "faults_outside_stability": windows_disjoint, "planned_faults": plan,
              "fault_evidence": outcomes, "duplicate_sample_seconds": {
                  phase: sorted(seconds) for phase, seconds in duplicate_seconds.items()
              },
              "fault_timeline": [event for event in events if event["kind"] in {"fault_started", "fault_finished"}],
              **verdicts(block, windows_disjoint and "window_finished" in windows, outcomes, passed_plan)}
    save_json(state / "soak-summary.json", result)
    return result


async def worker(state: Path, mode: str, duration: int) -> None:
    """Execute one campaign phase with signal-driven cleanup in the isolated process."""
    runtime = state / "runtime"
    isolate_process(state, runtime)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        home = HomeSimulator()
        async with home.serve() as url:
            candidate = Candidate(state, home, url)
            if mode == "run-soak":
                await stability(candidate, duration)
            else:
                await inject(candidate)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def positive_seconds(value: str) -> int:
    duration = int(value)
    if duration < 1:
        raise argparse.ArgumentTypeError("duration must be at least one second")
    return duration


def main(argv: list[str] | None = None) -> int:
    """Run a safety-checked phase or render its operator evidence."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("preflight")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--review", type=Path, required=True)
    prep.add_argument("--state-root", type=Path, default=Path.home() / ".local/state/kza-soak")
    for name in ("run-soak", "inject-faults", "summarize", "render-systemd-unit"):
        command = commands.add_parser(name)
        command.add_argument("--state", type=Path, required=True)
        if name in ("run-soak", "render-systemd-unit"):
            command.add_argument("--duration-s", type=positive_seconds, default=WINDOW_S)
        if name == "render-systemd-unit":
            command.add_argument("--faults-first", action="store_true")
    args = parser.parse_args(argv)
    old_umask = os.umask(0o077)
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(args.archive, args.review, args.state_root.expanduser().absolute()), indent=2))
            return 0
        state = args.state.expanduser().absolute()
        with locked(state) as lock:
            receipt = json.loads((state / "preflight.json").read_text())
            if receipt["state"] != str(state) or receipt["commit"] != state.name:
                raise ValueError("state does not match its preflight receipt")
            if args.command == "render-systemd-unit":
                print(render_unit(state, args.duration_s, args.faults_first), end="")
                return 0
            if args.command == "summarize":
                summary = summarize(state, receipt)
                print(json.dumps({key: value for key, value in summary.items() if type(value) is bool}, indent=2))
                return 0
            verify_runtime(state, receipt)
            duration = getattr(args, "duration_s", WINDOW_S)
            if args.command == "run-soak" and duration >= WINDOW_S:
                isolation = linux_isolation()
                status = thermouy_status()
                journal(state, "linux_at_start", isolation=isolation, thermouy=status)
            events = list(records(state / "faults.jsonl"))
            if any(event["kind"] == ("window" if args.command == "run-soak" else "fault_plan") for event in events):
                raise ValueError("this phase already started and cannot replace its evidence")
            if any(event["kind"] == "window" and time.time() < event["end"] for event in events):
                raise ValueError("fault injection cannot overlap the fixed stability window")
            script = Path(__file__).resolve()
            guard(script, Path(sys.executable))
            entry = (
                "import asyncio, runpy, sys; from pathlib import Path; "
                "scope = runpy.run_path(sys.argv[1]); "
                "asyncio.run(scope['worker'](Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])))"
            )
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", entry, str(script), str(state), args.command, str(duration)],
                env=worker_env(state), pass_fds=(lock,), check=False,
            )
            if args.command == "inject-faults" and result.returncode == 0:
                summarize(state, receipt)
            return result.returncode
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        LOG.error("%s", error)
        return 1
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
