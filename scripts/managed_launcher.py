"""Managed launcher (U3b-0d): systemd starts kza-voice through the admission controller.

Stdlib only; loaded by ``scripts/stage_runtime`` (by path, beside it) for the
``serve``, ``adopt``, ``install-controller``, ``install-launcher`` and
``uninstall-launcher`` commands. Design: managed-launcher-design-v3.json with
v2/v1 for sections v3 does not restate, the v3 review amendments D1-D6, the
U3b-0d3 review prerequisites K2 and K7, and user decisions L1-L6, N1 = B, N3.

Process tree: systemd (kza --user, Type=notify, NotifyAccess=main,
KillMode=mixed) -> this controller (main PID, holds the admission lease,
the only writer of the head, relays notifications) -> the child ``src.main``
(cwd = the head slot's runtime; lease, notify and control descriptors
inherited).

Rules this module keeps:

* SIGTERM/SIGINT handlers only write one byte to a self-pipe. Nothing is
  masked, so the child never inherits a blocked mask (v3 N1). The loop reads
  the pipe after the run record exists and after any seal completes.
* The child never sees NOTIFY_SOCKET or WATCHDOG_*. It writes lines to a
  private pipe; READY=1 is relayed once, WATCHDOG=1 only after READY,
  STATUS/STOPPING/EXTEND_TIMEOUT_USEC as validated lines (v2 notify_relay).
* One seal rule for the immediate seal and for R-a: an exact receipt, a
  reaped child, the receipt-state record when the slot declares it (K2), a
  manifest recomputed under the root lock before any image (K7, N2) and a
  matching closed-state digest. The exit code does not decide.
* Exit codes: 0 clean stop, 75 the child ended by itself and was sealed,
  68 open head eligible for one automatic recovery, 69 open head not
  eligible, 78 refused before CAS running, 70 transient error before CAS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import IntEnum
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

logger = logging.getLogger("kza.launcher")

stage = None  # scripts/stage_runtime, bound by bind_stage()


def bind_stage(module) -> None:
    """Use ``module`` as the admission controller (stage_runtime) implementation."""
    global stage
    stage = module


if __package__ == "scripts":  # in-process users (tests); the CLI binds by path
    from scripts import stage_runtime as _stage_module  # noqa: E402

    bind_stage(_stage_module)


class ExitCode(IntEnum):
    """Controller exit codes (design v2 exit_codes_H1)."""

    CLEAN = 0
    RECOVERABLE = 68
    UNRECOVERABLE = 69
    TRANSIENT = 70
    CHILD_ENDED = 75
    REFUSED = 78


RECOVER_MODES = ("bounded", "operator", "off")
ARM_WAIT_SECONDS = 10.0  # longest wait for ARMED=1 before a stop is forwarded anyway
# Start-time extension asked of systemd before recovery and between R-b steps (review P6).
RECOVERY_EXTEND_SECONDS = 120
# install-launcher's TimeoutStartSec default (U3b-0e4, F1=C). systemd 255 drops any
# EXTEND_TIMEOUT_USEC on a `systemctl --user daemon-reload` and re-arms the start
# timer from TimeoutStartSec, so the value must cover a whole start on its own.
# Measured on the server (U3b-0e4 T-A8, 161 MB adopted state, 500k events): first
# managed start 12.2 s, crash + automatic R-b to READY 20.1 s, activation CPU-bound
# and linear in rows. 240 s = the path probe's own budget (--probe-timeout 60) plus
# 180 s, almost nine times the measured R-b start: room for a state several times
# larger and a loaded host. EXTEND stays as the extra margin.
DEFAULT_TIMEOUT_START = 240.0
# --timeout-start must be whole seconds in this range (review U3b-0e4 F3): 0 would disable the
# start timeout and an exponent from :g would be rejected by systemd, silently keeping the base value.
TIMEOUT_START_RANGE = (30, 3600)
CRASH_BUDGET_SECONDS = 24 * 3600
STATUS_LIMIT = 200
NOTIFY_FD_ENV = "KZA_NOTIFY_FD"
WATCHDOG_INTERVAL_ENV = "KZA_WATCHDOG_INTERVAL_USEC"
CONTROL_FD_ENV = "KZA_CONTROL_FD"
TICKET_ENV = "KZA_RUNTIME_ADMISSION"
# Removed from the controller environment before the child gets it; the
# KZA_RUNTIME_* prefix is removed as well (v1 child_environment, v2 relay).
_CHILD_ENV_DROPPED = frozenset({"PYTHONPATH", "CONFIG_PATH", "PYTHONHOME", "NOTIFY_SOCKET",
                                "WATCHDOG_USEC", "WATCHDOG_PID", NOTIFY_FD_ENV,
                                WATCHDOG_INTERVAL_ENV, CONTROL_FD_ENV})
_PROBE_ENVIRONMENT = ("LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SQLITE_TMPDIR")
UNINSTALL_PHRASE = "revivir datos borrados"
LEGACY_COMMITS = ("b0866e8", "63ea8d3")
# Q5 retirement of the frozen b0866e8 state (U3b-0e5, P7=B): admission/legacy-retirement/
# intent-<plan>.json before the deletion, retired-<plan>.json after it. Both name the slots
# that leave L4 eligibility; uninstall-launcher and adoption-abort refuse them.
LEGACY_RETIREMENT_DIRECTORY = "legacy-retirement"
LEGACY_RETIRE_INTENT_PREFIX = "intent-"
LEGACY_RETIRED_PREFIX = "retired-"
_LEGACY_RETIREMENT_NAME = re.compile(r"(intent|retired)-([0-9a-f]{64})\.json")
_LEGACY_RETIREMENT_KINDS = {"intent": "legacy_state_retire_intent", "retired": "legacy_state_retired"}
UNIT = "kza-voice.service"
DROP_IN_NAME = "zz-kza-stop-timeout.conf"
LOGROTATE_NAME = "console-logrotate.conf"  # <root>/console-logrotate.conf, run hourly by kza-console-rotate.timer
# Pre-managed properties recorded at install (v3 N8) and verified at uninstall.
RECORDED_PROPERTIES = ("Type", "NotifyAccess", "WatchdogUSec", "RestartPreventExitStatus",
                       "TimeoutStartUSec", "TimeoutStopUSec", "StandardOutput", "StandardError",
                       "WorkingDirectory", "ExecStart", "Environment")
# Immutable append-only families whose bound rows must survive (v2 R-b forward evolution).
_APPEND_ONLY_FAMILIES = ("conversation_tombstones", "conversation_erasures", "fact_owner_revisions",
                         "retired_users")
# Resurrection families and the ledger kinds that record them (v3 N2 R-b).
_RESURRECTION_KINDS = {"conversation": ("conversation_erased",),
                       "fact": ("fact_deleted", "fact_pruned_retired_owner"),
                       "voice_profile": ("voice_profile_deleted",),
                       "voice_consent": ("voice_consent_withdrawn",)}
_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_DIGEST_MISMATCH = ("runtime_admission_state_manifest_mismatch", "runtime_admission_closed_state_mismatch")
_RECEIPT_PROBLEMS = ("runtime_admission_close_receipt_mismatch", "runtime_admission_receipt_state")
# Serving-store keys the child's own _check_serving_paths pins under the state.
_SERVING_KEYS = ("db_path", "privacy_db_path", "chroma_path", "preferences_path", "data_path",
                 "data_dir", "conversations_dir", "exports_dir", "metrics_path", "events_db",
                 "suggestions_path", "base_path", "control_selection_path", "output_dir",
                 "manifest_path", "ledger_path", "storage_path")

# Runs in a fresh "-I -B" interpreter with an allow-listed environment: the
# controller's own read-only path preflight (S3). It needs yaml, which the
# stdlib controller does not import. It never imports slot code.
_PREFLIGHT_SOURCE = "\n".join([
    "import json, re, sys",
    "from pathlib import Path",
    "import yaml",
    "runtime, state = Path(sys.argv[1]), Path(sys.argv[2]).resolve()",
    "serving = set(json.loads(sys.argv[3]))",
    "special = {('vectordb', 'path'), ('ambient', 'keep_audio', 'dir'), ('rooms', 'wake_word', 'capture_clips', 'dir')}",
    "pathlike = re.compile(r'(^|_)(path|dir|file|model|root|savedir)$')",
    # A1 (U3b-0e): Hugging Face hub ids at these exact keys are not paths. One
    # slash, no leading '.' or '/', and not a local tree name; anything else
    # with '/' is still a relative path.
    "hub_keys = {('tts', 'kokoro', 'model'), ('embeddings', 'model'), ('speaker_id', 'model'), ('emotion', 'model'),",
    "            ('training', 'lora', 'base_model'), ('training', 'nightly', 'base_model'),",
    "            ('training', 'nightly', 'whisper_finetune', 'base_model'), ('code_index', 'embedder', 'model')}",
    "hub_id = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+')",
    "local_heads = {'models', 'logs', 'data', 'config', 'src', 'scripts', 'tools'}",
    "config = yaml.safe_load((state / 'config' / 'settings.yaml').read_text())",
    "problems = []",
    "def resolved(value):",
    "    return (runtime / value).resolve(strict=False)",
    "def visit(value, prefix):",
    "    if isinstance(value, dict):",
    "        for key, child in value.items():",
    "            visit(child, prefix + (str(key),))",
    "    elif isinstance(value, list):",
    "        for number, child in enumerate(value):",
    "            visit(child, prefix + (str(number),))",
    "    elif isinstance(value, str) and value and prefix and '${' not in value and '://' not in value:",
    "        name, dotted = prefix[-1], '.'.join(prefix)",
    "        if prefix == ('logging', 'file'):",
    "            target = resolved(value)",
    "            if target == state or target.is_relative_to(state):",
    "                problems.append('log_under_state:' + dotted)",
    "            elif not value.startswith('/'):",
    "                problems.append('relative_path:' + dotted)",
    "        elif name in serving or prefix in special:",
    "            if not resolved(value).is_relative_to(state):",
    "                problems.append('foreign_serving_store:' + dotted)",
    "        elif pathlike.search(name) and '/' in value and not value.startswith('/'):",
    "            if prefix in hub_keys and hub_id.fullmatch(value) and value.split('/')[0] not in local_heads:",
    "                pass",
    "            elif not resolved(value).is_relative_to(state):",
    "                problems.append('relative_path:' + dotted)",
    "visit(config if isinstance(config, dict) else {}, ())",
    "sys.stdout.write(json.dumps({'problems': sorted(problems)}))",
])


class LauncherRefusal(ValueError):
    """A coded, content-free refusal that maps to one exit code.

    ``problems`` lists every coded problem found when one refusal covers
    several checks (the install-launcher preflight).
    """

    def __init__(self, code: str, exit_code: ExitCode = ExitCode.REFUSED,
                 problems: list[str] | None = None) -> None:
        super().__init__(code if not problems else code + ": " + ", ".join(problems))
        self.code = code
        self.exit_code = exit_code
        self.problems = list(problems or [])


# --------------------------------------------------------------------------
# systemd notifications


class SystemdNotifier:
    """Send sd_notify datagrams to NOTIFY_SOCKET as the main PID (no-op without it)."""

    def __init__(self, path: str | None) -> None:
        self._path = path or None
        self._socket: socket.socket | None = None
        self.sent: list[str] = []
        self._failures = 0

    @classmethod
    def from_environment(cls, environ: dict) -> SystemdNotifier:
        """Keep NOTIFY_SOCKET for the controller only."""
        return cls(environ.get("NOTIFY_SOCKET"))

    def send(self, line: str) -> None:
        """Send one notification line; a failure is logged once and never raises."""
        self.sent.append(line)
        if self._path is None:
            return
        address = self._path
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            if self._socket is None:
                self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._socket.sendto(line.encode("ascii"), address)
        except OSError as error:
            self._failures += 1
            if self._failures == 1:
                logger.warning("[Launcher] notification to the service manager failed: %s",
                               type(error).__name__)

    def status(self, text: str) -> None:
        """Send a short ASCII STATUS= line (truncated, never raises)."""
        clean = "".join(char if char.isascii() and char.isprintable() else "?" for char in text)
        self.send("STATUS=" + clean[:STATUS_LIMIT])

    def close(self) -> None:
        """Close the datagram socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class NotifyRelay:
    """Validate the child's notification lines and relay them to systemd (NotifyAccess=main).

    READY=1 is relayed once; WATCHDOG=1 only after READY; STATUS=<ascii,
    at most 200>, STOPPING=1 and EXTEND_TIMEOUT_USEC=<int> as given. ARMED=1
    (the child's stop handler is installed) is consumed, never relayed. Any
    other line, or a malformed one, is dropped and counted.
    """

    def __init__(self, notifier: SystemdNotifier) -> None:
        self._notifier = notifier
        self._buffer = b""
        self.ready = False
        self.armed = False
        self.dropped = 0
        self.watchdog_relayed = 0

    def feed(self, data: bytes) -> None:
        """Consume raw bytes from the child's pipe."""
        self._buffer += data
        if len(self._buffer) > 65536:  # a flood without newlines: drop it
            self._buffer = b""
            self.dropped += 1
            return
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            self._line(raw)

    def _line(self, raw: bytes) -> None:
        try:
            line = raw.decode("ascii")
        except UnicodeDecodeError:
            self.dropped += 1
            return
        if line == "ARMED=1":  # the child's stop handler exists; private, never relayed
            self.armed = True
            return
        if line == "READY=1":
            if not self.ready:
                self.ready = True
                self._notifier.send(line)
            return
        if line == "WATCHDOG=1":
            if self.ready:
                self.watchdog_relayed += 1
                self._notifier.send(line)
            else:
                self.dropped += 1
            return
        if line == "STOPPING=1":
            self._notifier.send(line)
            return
        if line.startswith("STATUS="):
            text = line[len("STATUS="):]
            if 0 < len(text) <= STATUS_LIMIT and text.isprintable():
                self._notifier.send(line)
                return
        elif line.startswith("EXTEND_TIMEOUT_USEC="):
            value = line[len("EXTEND_TIMEOUT_USEC="):]
            if value.isdigit() and len(value) <= 18:
                self._notifier.send(line)
                return
        self.dropped += 1


# --------------------------------------------------------------------------
# stop requests


class StopPipe:
    """Self-pipe stop requests: the handler writes one byte and never raises (v2 M1, v3 N1)."""

    def __init__(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        self._previous: dict[int, object] = {}
        self.requested = False
        self.signals = 0

    def install(self) -> StopPipe:
        """Route SIGTERM and SIGINT to the pipe (main thread only)."""
        for number in (signal.SIGTERM, signal.SIGINT):
            self._previous[number] = signal.signal(number, self._handler)
        return self

    def _handler(self, signum, frame) -> None:
        try:
            os.write(self.write_fd, b"T" if signum == signal.SIGTERM else b"I")
        except OSError:
            pass

    def drain(self) -> bool:
        """Read pending stop bytes; return whether a stop has ever been requested."""
        while True:
            try:
                data = os.read(self.read_fd, 64)
            except (BlockingIOError, InterruptedError):
                break
            if not data:
                break
            self.signals += len(data)
            self.requested = True
        return self.requested

    def close(self) -> None:
        """Restore the previous handlers and close the pipe."""
        for number, previous in self._previous.items():
            signal.signal(number, previous)
        self._previous.clear()
        for descriptor in (self.read_fd, self.write_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


# --------------------------------------------------------------------------
# service manager queries (unprivileged systemctl; injectable for tests)


def parse_timespan(text: str | None) -> float | None:
    """Seconds of a systemd timespan as ``systemctl show`` prints it; None if unknown."""
    if text is None:
        return None
    text = text.strip()
    if not text or text in {"infinity", "[not set]"}:
        return None
    units = {"us": 1e-6, "usec": 1e-6, "ms": 1e-3, "msec": 1e-3, "s": 1.0, "sec": 1.0, "min": 60.0,
             "h": 3600.0, "hr": 3600.0, "d": 86400.0, "": 1.0}
    total = 0.0
    position = 0
    for match in re.finditer(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", text):
        if match.start() != position or match.group(2) not in units:
            return None
        total += float(match.group(1)) * units[match.group(2)]
        position = match.end()
    if position != len(text):
        return None
    return total


class ServiceManager:
    """Read-only systemctl queries plus ``--user daemon-reload`` (never sudo)."""

    def __init__(self, executable: str = "systemctl", timeout: float = 2.0) -> None:
        self.executable = executable
        self.timeout = timeout
        self.calls: list[tuple[str, ...]] = []

    def _run(self, *arguments: str, timeout: float | None = None, strict: bool = False) -> str | None:
        """stdout, or None on failure; ``strict`` accepts exit 0 only (state changes, U3b-0e3 F3)."""
        command = (self.executable, *arguments)
        self.calls.append(tuple(arguments))
        environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "XDG_RUNTIME_DIR",
                                                        "DBUS_SESSION_BUS_ADDRESS", "HOME")
                       if key in os.environ}
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False,
                                       stdin=subprocess.DEVNULL, env=environment,
                                       timeout=self.timeout if timeout is None else timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode not in ((0,) if strict else (0, 1)):  # is-system-running exits 1 for 'stopping'
            return None
        return completed.stdout

    def stop_budget(self, uid: int) -> float | None:
        """The effective TimeoutStopUSec of user@<uid>.service (system manager), in seconds."""
        output = self._run("show", f"user@{uid}.service", "-p", "TimeoutStopUSec", "--value")
        return None if output is None else parse_timespan(output.strip())

    def system_state(self) -> str | None:
        """``systemctl is-system-running`` against the system manager (0.5 s bound)."""
        output = self._run("is-system-running", timeout=0.5)
        return None if output is None else output.strip() or None

    def user_show(self, unit: str, properties: tuple[str, ...]) -> dict[str, str] | None:
        """``systemctl --user show <unit> -p P ...`` as a dict (None if unavailable)."""
        arguments = ["--user", "show", unit]
        for name in properties:
            arguments += ["-p", name]
        output = self._run(*arguments)
        if output is None:
            return None
        values = {}
        for line in output.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        return {name: values.get(name, "") for name in properties}

    def user_show_all(self, unit: str, name: str) -> list[str] | None:
        """Every ``<name>=`` line of ``systemctl --user show <unit> -p <name>`` (ExecStart, EnvironmentFiles)."""
        output = self._run("--user", "show", unit, "-p", name)
        if output is None:
            return None
        return [line.split("=", 1)[1] for line in output.splitlines() if line.startswith(name + "=")]

    def user_cat(self, unit: str) -> str | None:
        """``systemctl --user cat <unit>``: the unit file and drop-ins as written (exit 0 only), or None."""
        return self._run("--user", "cat", unit, strict=True)

    def daemon_reload(self, *, strict: bool = False) -> bool:
        """``systemctl --user daemon-reload`` (``strict``: exit 0 only)."""
        return self._run("--user", "daemon-reload", timeout=30.0, strict=strict) is not None

    def user_unit_command(self, verb: str, unit: str) -> bool:
        """``systemctl --user enable|disable|stop <unit>``; success is exit 0 only (systemctl exits 1 on failure)."""
        if verb not in ("enable", "disable", "stop"):
            raise ValueError("runtime_admission_unit_verb")
        return self._run("--user", verb, unit, timeout=90.0, strict=True) is not None

    def user_is_enabled(self, unit: str) -> str | None:
        """``systemctl --user is-enabled <unit>``: the state word when it exits 0, else None."""
        output = self._run("--user", "is-enabled", unit, strict=True)
        return None if output is None else (output.strip().splitlines() or [""])[-1].strip()

    def user_enable(self, unit: str) -> bool:
        """Enable ``unit`` (adoption abort, U3b-0e3) and read it back: True only if is-enabled says 'enabled'.

        Never starts the unit.
        """
        return self.user_unit_command("enable", unit) and self.user_is_enabled(unit) == "enabled"

    def user_units(self) -> list[str] | None:
        """Every user unit name systemd knows (list-unit-files plus loaded units), or None on failure."""
        names = set()
        for arguments in (("--user", "list-unit-files", "--no-legend", "--plain", "--no-pager"),
                          ("--user", "list-units", "--all", "--no-legend", "--plain", "--no-pager")):
            output = self._run(*arguments, strict=True)
            if output is None:
                return None
            for line in output.splitlines():  # list-units marks failed units with a leading bullet
                words = [word for word in line.split() if word not in ("\u25cf", "*")]
                if words:
                    names.add(words[0])
        return sorted(names)


@dataclass
class RebootBudget:
    """The effective user-manager stop budget and the reboot mode it allows (v3 N4)."""

    budget: float | None
    unit_timeout_stop: float
    mode: str = field(init=False)

    def __post_init__(self) -> None:
        clean = self.budget is not None and self.budget >= self.unit_timeout_stop + 10
        self.mode = "clean" if clean else "fallback"

    def status(self) -> str | None:
        """STATUS text for the fallback mode."""
        if self.mode == "clean":
            return None
        if self.budget is None:
            return "reboot budget unknown"
        return f"reboot budget insufficient ({self.budget:g}s)"

    def deadline_seconds(self, stop_grace: float, elapsed: float) -> float:
        """Seconds left for the child once the manager is stopping (v3 N9)."""
        if self.budget is None:
            return stop_grace
        return max(0.0, min(stop_grace, self.budget - elapsed - 2.0))


def render_stop_timeout_dropin(uid: int, unit_timeout_stop: float, chain_extra: float = 0.0) -> dict:
    """The sudo drop-in the user applies (N1 = B; v3 N4, review D4). Never written here.

    B = the longest After=-ordered stop chain from kza-voice (its own
    TimeoutStopSec plus the units that stop after it) plus 15 s. The file
    sorts after the vendor timeout.conf and lives in the instance directory,
    so the vendor 5 s stays in force for every other user manager.
    """
    value = int(math.ceil(unit_timeout_stop + chain_extra + 15))
    directory = f"/etc/systemd/system/user@{uid}.service.d"
    return {"path": f"{directory}/{DROP_IN_NAME}", "directory": directory, "value_seconds": value,
            "text": f"[Service]\nTimeoutStopSec={value}s\n", "mode": "0644 root:root",
            "apply": [f"sudo install -d -m 0755 {directory}",
                      f"sudo install -m 0644 /dev/stdin {directory}/{DROP_IN_NAME}",
                      "sudo systemctl daemon-reload"],
            "verify": [f"systemctl show user@{uid}.service -p TimeoutStopUSec,DropInPaths",
                       f"systemctl cat user@{uid}.service",
                       "ls /etc/systemd/system.control /run/systemd/system.control"],
            "never": f"do not restart user@{uid}.service (it would stop every kza unit)"}


# --------------------------------------------------------------------------
# process identity


def read_boot_id(path: str | None = None) -> str:
    """The current boot id (Linux /proc; macOS kern.boottime; a test file when given)."""
    candidates = [path] if path else ["/proc/sys/kernel/random/boot_id"]
    for candidate in candidates:
        try:
            value = Path(candidate).read_text().strip()
        except OSError:
            continue
        if value:
            return value
    if path is None:
        try:
            completed = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True,
                                       timeout=2, check=False)
            if completed.returncode == 0 and completed.stdout.strip():
                return "boottime-" + hashlib.sha256(completed.stdout.strip().encode()).hexdigest()[:32]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "unknown"


def process_start_time(pid: int) -> str | None:
    """An opaque start-time token for a live pid, or None when it does not exist."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.path.isdir("/proc/self"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        fields = raw.rsplit(")", 1)[-1].split()
        if len(fields) > 19 and fields[0] != "Z":
            return "ticks-" + fields[19]
        return None
    try:
        completed = subprocess.run(["ps", "-o", "stat=,lstart=", "-p", str(pid)], capture_output=True,
                                   text=True, timeout=2, check=False, env={"LC_ALL": "C", "PATH": "/bin:/usr/bin"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    if completed.returncode != 0 or not output or output.startswith("Z"):
        return None
    return "lstart-" + output.split(None, 1)[-1]


def child_alive(pid: int, start_time: str | None) -> bool:
    """Whether the recorded child (pid plus start time) is still alive."""
    current = process_start_time(pid)
    return current is not None and (start_time is None or current == start_time)


def _cgroup_procs_path() -> Path | None:
    """This process's unit cgroup.procs, only when systemd runs it as a service."""
    if "INVOCATION_ID" not in os.environ:
        return None
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            relative = line[3:].strip()
            if relative.endswith(".service"):
                return Path("/sys/fs/cgroup") / relative.lstrip("/") / "cgroup.procs"
    return None


def kill_stragglers(procs_path: Path | None, *, wait: float = 2.0) -> int | None:
    """SIGKILL every process left in the unit cgroup except this one (review D1).

    Returns the number killed, or None when not checked (no unit cgroup).
    """
    if procs_path is None:
        return None
    own = os.getpid()

    def others() -> list[int]:
        try:
            return [int(item) for item in procs_path.read_text().split() if int(item) != own]
        except (OSError, ValueError):
            return []

    found = others()
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.monotonic() + wait
    while others() and time.monotonic() < deadline:
        time.sleep(0.05)
    return len(found)


# --------------------------------------------------------------------------
# records


def _records(root: Path, *parts: str) -> Path:
    return Path(root) / "admission" / Path(*parts)


def run_record_path(root: Path, run_id: str) -> Path:
    """admission/runs/<run_id>.json."""
    return _records(root, "runs", run_id + ".json")


def exit_record_path(root: Path, run_id: str) -> Path:
    """admission/runs/<run_id>.exit.json."""
    return _records(root, "runs", run_id + ".exit.json")


def _read_optional(path: Path) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    return stage._read_json(path)


def read_run_record(root: Path, run_id: str) -> dict | None:
    """The immutable record of one started run, or None."""
    record = _read_optional(run_record_path(root, run_id))
    if record is not None and (record.get("run_id") != run_id or type(record.get("child_pid")) is not int):
        raise LauncherRefusal("runtime_admission_run_record_schema", ExitCode.UNRECOVERABLE)
    return record


def read_exit_record(root: Path, run_id: str) -> dict | None:
    """The immutable exit record of one run, or None (power loss or controller SIGKILL)."""
    record = _read_optional(exit_record_path(root, run_id))
    if record is not None and record.get("run_id") != run_id:
        raise LauncherRefusal("runtime_admission_run_record_schema", ExitCode.UNRECOVERABLE)
    return record


def _now_stamp(wall: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(wall)) + f"{int((wall % 1) * 1e6):06d}"


# --------------------------------------------------------------------------
# recovery (R-a with a receipt, R-b without one)


@dataclass
class RecoveryDecision:
    """Outcome of recovering an open head at serve start or by the operator."""

    closed_head: dict | None = None
    exit_code: ExitCode | None = None
    code: str = ""
    path: str = ""
    report: dict = field(default_factory=dict)


def _origin_kind(root: Path, record: dict, limit: int = 4096) -> str:
    """The first non-selection kind behind a manual head record (selection chains are copies)."""
    current = record
    for _ in range(limit):
        if current["kind"] != "selection" or current["previous"] is None:
            return current["kind"]
        path = root / "admission" / "manual-components" / "heads" / (current["previous"]["head_sha256"] + ".json")
        current = stage._read_json(path)
    raise LauncherRefusal("runtime_admission_manual_chain_too_long", ExitCode.UNRECOVERABLE)


def _recoveries(root: Path) -> list[dict]:
    directory = _records(root, "recoveries")
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(stage._read_json(path))
    return records


def classify_run(run: dict | None, exit_record: dict | None, boot_id: str) -> str:
    """'boot' when the run ended with the previous boot (v2 boot_class_H4), else 'crash'."""
    if run is None or run.get("boot_id") in (None, "unknown") or run["boot_id"] == boot_id:
        return "crash"
    if exit_record is None or exit_record.get("manager_stopping") is True:
        return "boot"
    return "crash"


def automatic_budget(root: Path, head: dict, klass: str, boot_id: str, wall: float) -> str | None:
    """Why automatic R-b is not allowed now, or None (L1 = A; v2 budget_L1).

    Crash class: at most once per 24 h, and never twice in a row without a
    clean close in between (the run must not have started from a recovered
    head). Boot class: once per boot id, without consuming the crash budget.
    """
    records = _recoveries(root)
    if klass == "boot":
        if any(item.get("mode") == "automatic" and item.get("class") == "boot"
               and item.get("boot_id") == boot_id for item in records):
            return "runtime_admission_recovery_boot_budget_spent"
        return None
    if any(item.get("mode") == "automatic" and item.get("class") == "crash"
           and type(item.get("at")) is float and item["at"] > wall - CRASH_BUDGET_SECONDS for item in records):
        return "runtime_admission_recovery_crash_budget_spent"
    start = stage.read_manual_head(root, stage.previous_closed_head(head))
    if _origin_kind(root, start) == "recovery":
        return "runtime_admission_recovery_second_in_a_row"
    return None


def _sqlite_files(state: Path) -> list[Path]:
    return sorted(path for path in state.rglob("*")
                  if path.is_file() and not path.is_symlink() and path.suffix in _SQLITE_SUFFIXES)


def _quick_check(path: Path) -> bool:
    """Open read-write under the lease (WAL recovery), run quick_check, close (checkpoint)."""
    try:
        connection = sqlite3.connect(path, timeout=1.0, isolation_level=None)
        try:
            rows = connection.execute("PRAGMA quick_check").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return rows == [("ok",)]


def _json_files(state: Path) -> list[Path]:
    return sorted(path for path in state.rglob("*.json")
                  if path.is_file() and not path.is_symlink() and path.name not in {stage._MARKER})


def _chroma_problems(state: Path) -> list[str]:
    """Structural segment check without the SDK (review D2: no PersistentClient open)."""
    problems = []
    for database in state.rglob("chroma.sqlite3"):
        for segment in sorted(item for item in database.parent.iterdir() if item.is_dir()):
            names = {item.name for item in segment.iterdir()}
            if not names & {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin"}:
                continue
            header = segment / "header.bin"
            if "header.bin" not in names or "data_level0.bin" not in names or header.stat().st_size == 0:
                problems.append("chroma_segment:" + segment.relative_to(state).as_posix())
    return problems


def _assert_writers_closed_read_only(state: Path) -> None:
    """The writers check without opening anything for writing (adopt --dry-run, review P3).

    flock works on a read-only descriptor, so a held writer lease is still seen.
    """
    for path in sorted(state.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name.endswith((".lock", ".lease")):
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("runtime_admission_writer_active") from None
            finally:
                os.close(descriptor)


def _shadow_wanted(relative: Path) -> bool:
    name = relative.name
    if relative.suffix in _SQLITE_SUFFIXES or relative.suffix == ".json" or name == ".env":
        return True
    if _file_source(relative) is not None:  # collector .jsonl exports the resurrection scan reads
        return True
    if relative.parts[0] == "config":
        return True
    for suffix in ("-wal", "-journal"):  # never -shm: SQLite rebuilds it from the -wal
        if name.endswith(suffix) and Path(name[:-len(suffix)]).suffix in _SQLITE_SUFFIXES:
            return True
    return False


def _shadow_state(state: Path, destination: Path) -> Path:
    """A private copy of what the R-b checks read: SQLite files with their -wal, JSON, config, .env.

    The originals are only read (O_RDONLY): no SQLite connection, no lock,
    no -shm created, no WAL checkpoint. The checks then run on the copy, so
    a dry run cannot change the state manifest a pending R-a depends on.
    """
    shadow = Path(destination).resolve()
    for path in sorted(state.rglob("*")):
        if path.is_symlink():
            raise ValueError("runtime_admission_state_symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(state)
        if not _shadow_wanted(relative):
            continue
        target = shadow / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source, open(target, "xb") as copy:
            while chunk := source.read(1 << 20):
                copy.write(chunk)
        os.chmod(target, 0o600)
    return shadow


def _chroma_ids(db: sqlite3.Connection, names: set[str]) -> set[str]:
    """Embedding ids a Chroma store holds, read from its SQLite tables (no SDK; review P1).

    src/memory/long_term.py stores each fact as ``ids=[fact.id]`` with
    metadata {fact_policy, revision}: the fact id lives only in
    ``embeddings.embedding_id`` (per collection, through the metadata
    segment). Every collection is read, the policy index included. An id
    whose newest ``embeddings_queue`` operation is not a delete also counts,
    in case the log was not yet applied to ``embeddings``.
    """
    found: set[str] = set()
    if {"embeddings", "segments", "collections"} <= names:
        found.update(row[0] for row in db.execute(
            "SELECT e.embedding_id FROM embeddings e JOIN segments s ON e.segment_id=s.id"
            " JOIN collections c ON s.collection=c.id") if isinstance(row[0], str) and row[0])
    if "embeddings_queue" in names:
        latest: dict[str, int] = {}
        for identifier, operation in db.execute("SELECT id, operation FROM embeddings_queue ORDER BY seq_id"):
            if isinstance(identifier, str) and identifier:
                latest[identifier] = operation
        found.update(identifier for identifier, operation in latest.items() if operation != 3)  # 3 = DELETE
    return found


def _walk_conversation_ids(value, found: set) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "conversation_id" and isinstance(child, str) and child:
                found.add(child)
            else:
                _walk_conversation_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk_conversation_ids(child, found)


_UNREADABLE = "runtime_admission_recovery_resurrection_unreadable"


def _json_conversation_ids(encoded) -> set[str]:
    """conversation_id values anywhere in one JSON document; any parse failure refuses (review F5).

    Writers produce bounded, valid JSON, so a document that does not parse
    (invalid, wrong type, or nested past the recursion limit) is a corrupt
    state that the scan cannot vouch for.
    """
    found: set[str] = set()
    try:
        _walk_conversation_ids(json.loads(encoded), found)
    except (TypeError, ValueError, RecursionError):
        raise ValueError(_UNREADABLE) from None
    return found


def subject_digest(salt: str, subject: str) -> str:
    """HMAC-SHA256 of a subject id keyed by a ledger salt (parity with src/memory/deletion_ledger)."""
    return hmac.new(bytes.fromhex(salt), subject.encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------
# Conversation resurrection per store (U3b-0d4a; u3b-0e-adoption-design-v3
# section U3b-0d4a, review findings F5 and N6).
#
# An erased conversation's payload occurrence is exempt only while the erasure
# step on the target that governs that store is not complete. Steps on other
# targets (the cleaner-less unresolved_* literals included) never exempt it.

_JOURNAL_TABLES = frozenset({"erasure_targets", "conversation_erasures", "erasure_steps"})


@dataclass(frozen=True)
class _Target:
    id: str
    participant: str
    version: int
    descriptor: dict


@dataclass
class _ErasureJournal:
    """What the scan reads from the single erasure journal: targets and unfinished steps."""

    targets: tuple[_Target, ...] = ()
    unfinished: frozenset = frozenset()  # (conversation_id, target_id) with a step state != 'complete'
    roots: tuple[Path, ...] = ()  # descriptor_root, and its resolved form when it differs

    def registered(self, participant: str, version: int = 1):
        return (target for target in self.targets
                if target.participant == participant and target.version == version)

    def relative(self, value) -> Path | None:
        """A descriptor path relativized against descriptor_root; None when it is not under it."""
        if type(value) is not str or not os.path.isabs(value):
            return None
        normal = Path(os.path.normpath(value))
        for root in self.roots:
            try:
                return normal.relative_to(root)
            except ValueError:
                continue
        return None


def _scan_utterances(db, names, relative):
    if "utterances" in names:
        for (value,) in db.execute("SELECT DISTINCT conversation_id FROM utterances"
                                   " WHERE conversation_id IS NOT NULL"):
            yield value, relative


def _scan_capture_conversations(db, names, relative):
    if "capture_conversations" in names:
        for (value,) in db.execute("SELECT id FROM capture_conversations WHERE id IS NOT NULL"):
            yield value, relative


def _scan_events(db, names, relative):
    if "events_policy_v1" in names:
        for (encoded,) in db.execute("SELECT sources_json FROM events_policy_v1 WHERE sources_json IS NOT NULL"):
            for value in _json_conversation_ids(encoded):
                yield value, relative


def _scan_wake_clips(db, names, relative):
    if "wake_clips_v1" in names:
        for target_id, encoded in db.execute("SELECT target_id, sources_json FROM wake_clips_v1"
                                             " WHERE sources_json IS NOT NULL"):
            for value in _json_conversation_ids(encoded):
                yield value, target_id


def _scan_conversation_turns(db, names, relative):
    """Each turn with its ON DELETE CASCADE parent's target (None for an orphan turn)."""
    if "conversation_turns_v1" not in names:
        return
    query = ("SELECT t.input_sources_json, w.target_id FROM conversation_turns_v1 t"
             " LEFT JOIN wake_clips_v1 w ON w.id=t.id WHERE t.input_sources_json IS NOT NULL"
             if "wake_clips_v1" in names else
             "SELECT input_sources_json, NULL FROM conversation_turns_v1 WHERE input_sources_json IS NOT NULL")
    for encoded, parent_target in db.execute(query):
        for value in _json_conversation_ids(encoded):
            yield value, parent_target


def _scan_chroma_metadata(db, names, relative):
    """conversation_id metadata with the owning collection's name (None when unresolvable)."""
    if "embedding_metadata" not in names:
        return
    query = ("SELECT m.string_value, c.name FROM embedding_metadata m LEFT JOIN embeddings e ON e.id=m.id"
             " LEFT JOIN segments s ON s.id=e.segment_id LEFT JOIN collections c ON c.id=s.collection"
             " WHERE m.key='conversation_id'"
             if {"embeddings", "segments", "collections"} <= names else
             "SELECT string_value, NULL FROM embedding_metadata WHERE key='conversation_id'")
    for value, collection in db.execute(query):
        yield value, (relative.parent, collection)


def _scan_json_file(path: Path, relative: Path):
    """One JSON document (.json) or one per non-empty line (.jsonl); unreadable refuses."""
    try:
        raw = path.read_bytes()
    except OSError:
        raise ValueError(_UNREADABLE) from None
    documents = [line for line in raw.splitlines() if line.strip()] if path.suffix == ".jsonl" else [raw]
    for document in documents:
        for value in _json_conversation_ids(document):
            yield value, relative


def _file_source(relative: Path) -> str | None:
    """The CONVERSATION_SOURCES key of a JSON file the scan reads, or None."""
    if relative.suffix == ".json" and "contexts" in relative.parts:
        return "contexts:*.json"
    if relative.parent.name == "policy_v1" and relative.suffix in (".json", ".jsonl"):
        return "collectors:policy_v1/*" + relative.suffix
    return None


def _governing_captures(journal: _ErasureJournal, relative: Path) -> frozenset:
    return frozenset(target.id for target in journal.registered("captures")
                     if journal.relative(target.descriptor.get("db_path")) == relative)


def _governing_events(journal: _ErasureJournal, relative: Path) -> frozenset:
    return frozenset(target.id for target in journal.registered("events")
                     if target.descriptor.get("table") == "events_policy_v1"
                     and journal.relative(target.descriptor.get("path")) == relative)


def _governing_clip(journal: _ErasureJournal, target_id) -> frozenset:
    """wake_clips_v1 rows name their target; conversation_turns_v1 rows arrive with their parent's."""
    return frozenset(target.id for target in journal.registered("collectors") if target.id == target_id)


def _governing_chroma(journal: _ErasureJournal, locator) -> frozenset:
    directory, collection = locator
    if type(collection) is not str:
        return frozenset()
    return frozenset(target.id for target in journal.registered("semantic_index")
                     if target.descriptor.get("collection") == collection
                     and journal.relative(target.descriptor.get("path")) == directory)


def _longest_directory(journal: _ErasureJournal, participant: str, key: str, relative: Path) -> frozenset:
    """The registered directory (descriptor[key]) that is the longest prefix of the file (ties: all of them)."""
    matches = []
    for target in journal.registered(participant):
        if key not in target.descriptor:
            continue
        directory = journal.relative(target.descriptor[key])
        if directory is not None and (directory == relative or directory in relative.parents):
            matches.append((len(directory.parts), target.id))
    if not matches:
        return frozenset()
    longest = max(length for length, _ in matches)
    return frozenset(identifier for length, identifier in matches if length == longest)


def _governing_contexts(journal: _ErasureJournal, relative: Path) -> frozenset:
    """{'kind': 'runtime', 'name': 'conversation_buffers'} has no 'path' and never governs a file."""
    return _longest_directory(journal, "contexts", "path", relative)


def _governing_collector_file(journal: _ErasureJournal, relative: Path) -> frozenset:
    """The collectors target whose 'root' holds the file: its cleaner unlinks <root>/policy_v1/<id>.<ext>,
    and its audit keeps the step pending while an unattributed file remains (collector_erasure.py)."""
    return _longest_directory(journal, "collectors", "root", relative)


@dataclass(frozen=True)
class ConversationSource:
    """One place a conversation id is read from, and how its governing erasure target is found.

    ``scan`` yields (conversation_id, locator) per occurrence; ``resolver``
    maps a locator to the ids of the governing ``participant``/``version``
    targets (empty: the occurrence has no governing target and counts).
    """

    source: str
    participant: str
    version: int
    resolver: object
    scan: object
    kind: str = "sqlite"  # "sqlite" (per database image) or "file" (per JSON file, see _file_source)


CONVERSATION_SOURCES: tuple[ConversationSource, ...] = (
    ConversationSource("utterances.conversation_id", "captures", 1, _governing_captures, _scan_utterances),
    ConversationSource("capture_conversations.id", "captures", 1, _governing_captures,
                       _scan_capture_conversations),
    ConversationSource("events_policy_v1.sources_json", "events", 1, _governing_events, _scan_events),
    ConversationSource("wake_clips_v1.sources_json", "collectors", 1, _governing_clip, _scan_wake_clips),
    ConversationSource("conversation_turns_v1.input_sources_json", "collectors", 1, _governing_clip,
                       _scan_conversation_turns),
    # Defensive: no writer puts conversation_id into Chroma metadata today. Only
    # embedding_metadata is read, not the embeddings_queue write log. If a writer
    # ever adds conversation_id metadata, apply _chroma_ids' newest-queue-operation
    # rule here too: a queue-only upsert would otherwise be missed, and a queued but
    # unapplied delete would count while the step is complete (review F6).
    ConversationSource("chroma:embedding_metadata.conversation_id", "semantic_index", 1, _governing_chroma,
                       _scan_chroma_metadata),
    ConversationSource("contexts:*.json", "contexts", 1, _governing_contexts, _scan_json_file, kind="file"),
    # Collector payload files (review F1): turn/record JSON and export JSONL carry their sources.
    ConversationSource("collectors:policy_v1/*.json", "collectors", 1, _governing_collector_file,
                       _scan_json_file, kind="file"),
    ConversationSource("collectors:policy_v1/*.jsonl", "collectors", 1, _governing_collector_file,
                       _scan_json_file, kind="file"),
)

# Conversation-bearing columns deliberately not scanned, with the reason
# (u3b-0e-adoption-design-v3 0d4a not_scanned_with_reason). A build-time test
# pins that every conversation_id / *sources_json column of
# stage_runtime._TABLE_COLUMNS is either here or in CONVERSATION_SOURCES.
CONVERSATION_NOT_SCANNED: dict[str, str] = {
    "conversation_erasures.conversation_id": "erasure record",
    "conversation_tombstones.conversation_id": "erasure record",
    "audio_write_intents.conversation_id": "cleanup obligation",
    "pending_audio_deletions.conversation_id": "cleanup obligation",
    "fact_sources.conversation_id": "fact provenance (producers participant); 0d4b",
    "voice_fingerprint_sources.conversation_id": "voice provenance (voices participant); 0d4b",
    "voice_profile_sources.conversation_id": "voice provenance (voices participant); 0d4b",
    "voice_fingerprints.sources_json": "voice provenance (voices participant); 0d4b",
    "voice_profiles.sources_json": "voice provenance (voices participant); 0d4b",
    "voice_groups.sources_json": "voice provenance (voices participant); 0d4b",
    "voice_group_requests.sources_json": "voice provenance (voices participant); 0d4b",
    "lineage.conversation_id": "diagnostics v2 catalog; 0d4b",
    "live_delivery_sources.conversation_id": "streams; 0d4b",
    "suggestions_policy_v1.sources_json": "suggestions participant; 0d4b",
    # Payload outside the name predicate (review F1). Columns:
    "stt_corrections_v1.original_text": "child of wake_clips_v1 (ON DELETE CASCADE) without its own sources;"
                                        " its parent clip's sources_json is scanned",
    "stt_corrections_v1.corrected_text": "same as stt_corrections_v1.original_text",
    "conversation_export_parents_v1.parent_id": "child of wake_clips_v1 (ON DELETE CASCADE); the export clip's"
                                                " sources_json is scanned",
    "voice_capture_reviews.transcript": "voice provenance keyed by fingerprint_id, no conversation id"
                                       " (voices participant, v2 unresolved); 0d4b",
    "voice_capture_reviews.audio_wav": "same as voice_capture_reviews.transcript",
    "voice_capture_reviews.audio_metadata_json": "same as voice_capture_reviews.transcript",
    "voice_capture_reviews.transcript_metadata_json": "same as voice_capture_reviews.transcript",
    "legacy_quarantine_records_v1.document": "quarantined legacy records: the collectors legacy_quarantine"
                                             " target counts them by batch_id, not by conversation; 0d4b",
    "legacy_quarantine_records_v1.metadata_json": "same as legacy_quarantine_records_v1.document",
    # Files (review F1). conversation_file_kind() names every file kind of the state.
    "file:policy_v1/*.wav": "wake-clip and STT-correction audio: no conversation id in the bytes; attributed only"
                            " through its wake_clips_v1 row, which is scanned; 0d4b",
    "file:policy_v1/*.flac": "ambient keep_audio archive: no conversation id in the bytes; attributed only through"
                             " utterances.audio_path, which is scanned; 0d4b",
    "file:policy_v1/*.part": "partial collector write: the audit keeps the collectors step pending while it exists",
    "file:diagnostics *.jsonl[.N]": "diagnostics v2 metric log: numeric records, lineage lives in the catalog; 0d4b",
    "file:chroma segment": "HNSW segment files (*.bin, index_metadata.pickle): ids and metadata live in"
                            " chroma.sqlite3, which is scanned",
    "file:*.lock / *.lease": "writer locks, no payload",
}


def conversation_file_kind(relative: Path) -> str:
    """The classification key of one state file (build-time test, review F1).

    'sqlite' files are covered by the column classification; every other key
    is in CONVERSATION_SOURCES or CONVERSATION_NOT_SCANNED, or starts with
    'unclassified:' (which the test refuses).
    """
    relative = Path(relative)
    name = relative.name
    if relative.suffix in _SQLITE_SUFFIXES or any(
            name.endswith(suffix) and Path(name[:-len(suffix)]).suffix in _SQLITE_SUFFIXES
            for suffix in ("-wal", "-journal", "-shm")):
        return "sqlite"
    source = _file_source(relative)
    if source is not None:
        return source
    if relative.suffix in (".lock", ".lease"):
        return "file:*.lock / *.lease"
    if relative.parent.name == "policy_v1" and relative.suffix in (".wav", ".flac", ".part"):
        return "file:policy_v1/*" + relative.suffix
    if re.fullmatch(r"metrics_policy_v1\.jsonl(\.\d+)?", name):
        return "file:diagnostics *.jsonl[.N]"
    if name in {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"} \
            and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", relative.parent.name):
        return "file:chroma segment"
    return "unclassified:" + (relative.suffix or name)


def _read_journal(db: sqlite3.Connection, names: set[str]) -> _ErasureJournal:
    """Targets and unfinished (conversation, target) steps; a missing table reads as empty."""
    targets = []
    if "erasure_targets" in names:
        for identifier, participant, version, encoded in db.execute(
                "SELECT id, participant, version, descriptor FROM erasure_targets"):
            try:
                descriptor = json.loads(encoded)
            except (TypeError, ValueError, RecursionError):
                continue  # an unreadable target governs nothing: its occurrences count
            if type(identifier) is str and type(participant) is str and type(version) is int \
                    and type(descriptor) is dict:
                targets.append(_Target(identifier, participant, version, descriptor))
    unfinished: frozenset = frozenset()
    if {"conversation_erasures", "erasure_steps"} <= names:
        unfinished = frozenset((conversation, target) for conversation, target in db.execute(
            "SELECT e.conversation_id, s.target_id FROM conversation_erasures e"
            " JOIN erasure_steps s ON s.operation_id=e.id WHERE s.state!='complete'"))
    return _ErasureJournal(tuple(targets), unfinished)


def _resurrection_scan(state: Path, descriptor_root: Path | None, residue: dict | None = None) -> tuple[dict, dict]:
    """One read of the state: (family counts, content-free counts per family and participant/source).

    ``residue``, when given, receives the counted ids themselves (in memory
    only, for the adoption's re-queue; see resurrection_residue).
    """
    state = Path(state)
    root = Path(descriptor_root) if descriptor_root is not None else state
    ledgers: list[tuple[str, dict[str, set]]] = []
    sources = {"fact": {"facts": set(), "semantic_index": set()},
               "voice_profile": {"voice_profiles": set(), "voice_fingerprints": set()}}
    consents: set[tuple[str, int]] = set()
    pending_index: set[str] = set()
    # Deduplicated (conversation_id, entry index, locator): locators are per file,
    # per (directory, collection) or per clip target, so few distinct ones (review F3).
    occurrences: set[tuple[str, int, object]] = set()
    journals: list[_ErasureJournal] = []
    for path in _sqlite_files(state):
        relative = path.relative_to(state)
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"deletion_ledger_v1", "deletion_ledger_salt_v1"} <= names:
                salt = db.execute("SELECT salt FROM deletion_ledger_salt_v1 WHERE id=1").fetchone()
                if salt is not None and type(salt[0]) is str and re.fullmatch("[0-9a-f]{64}", salt[0]):
                    subjects: dict[str, set] = {}
                    for kind, subject in db.execute("SELECT kind, subject_sha256 FROM deletion_ledger_v1"):
                        subjects.setdefault(kind, set()).add(subject)
                    ledgers.append((salt[0], subjects))
            if names & _JOURNAL_TABLES:
                journals.append(_read_journal(db, names))
            for index, entry in enumerate(CONVERSATION_SOURCES):
                if entry.kind == "sqlite":
                    occurrences.update((value, index, locator) for value, locator in entry.scan(db, names, relative)
                                       if type(value) is str and value)
            if "facts" in names:
                sources["fact"]["facts"].update(row[0] for row in db.execute(
                    "SELECT id FROM facts WHERE state!='deleted'"))
            if "voice_profiles" in names:
                sources["voice_profile"]["voice_profiles"].update(row[0] for row in db.execute(
                    "SELECT id FROM voice_profiles WHERE deleted_at IS NULL"))
            if "voice_fingerprints" in names:
                sources["voice_profile"]["voice_fingerprints"].update(row[0] for row in db.execute(
                    "SELECT suggested_profile FROM voice_fingerprints WHERE suggested_profile IS NOT NULL"))
            if "voice_attribution_consents" in names:
                consents.update((row[0], row[1]) for row in db.execute(
                    "SELECT person_id, revision FROM voice_attribution_consents")
                    if type(row[0]) is str and type(row[1]) is int)
            # A deletion still queued for the index (a crash before reconciliation)
            # leaves the fact's embedding in place legitimately: not a resurrection.
            if "index_jobs" in names:
                pending_index.update(row[0] for row in db.execute("SELECT fact_id FROM index_jobs"))
            if "policy_index_deletions" in names:
                pending_index.update(row[0] for row in db.execute(
                    "SELECT fact_id FROM policy_index_deletions WHERE state!='complete'"))
            sources["fact"]["semantic_index"].update(_chroma_ids(db, names))
    if len(journals) > 1:
        raise ValueError("runtime_admission_recovery_erasure_journal_ambiguous")
    journal = journals[0] if journals else _ErasureJournal()
    journal.roots = tuple(dict.fromkeys((root, root.resolve())))
    files = {entry.source: index for index, entry in enumerate(CONVERSATION_SOURCES) if entry.kind == "file"}
    for path in sorted(state.rglob("*.json*")):
        relative = path.relative_to(state)
        index = files.get(_file_source(relative))
        if index is not None and path.is_file() and not path.is_symlink():
            occurrences.update((value, index, locator)
                               for value, locator in CONVERSATION_SOURCES[index].scan(path, relative)
                               if type(value) is str and value)

    def recorded(candidate, kinds) -> bool:
        return type(candidate) is str and bool(candidate) and any(
            subject_digest(salt, candidate) in subjects.get(kind, set())
            for salt, subjects in ledgers for kind in kinds)

    erased: dict[str, bool] = {}
    governing_of: dict[tuple[int, object], frozenset] = {}
    counted = {entry.participant: set() for entry in CONVERSATION_SOURCES}
    exempt = {entry.participant: set() for entry in CONVERSATION_SOURCES}
    for conversation, index, locator in occurrences:
        if conversation not in erased:
            erased[conversation] = recorded(conversation, _RESURRECTION_KINDS["conversation"])
        if not erased[conversation]:
            continue
        entry = CONVERSATION_SOURCES[index]
        key = (index, locator)
        if key not in governing_of:
            governing_of[key] = entry.resolver(journal, locator)
        governing = governing_of[key]
        if governing and all((conversation, target) in journal.unfinished for target in governing):
            exempt[entry.participant].add(conversation)  # its own store's step is pending: the next run cleans it
            continue
        counted[entry.participant].add(conversation)
        if residue is not None:
            residue.setdefault("conversation", {}).setdefault(conversation, set()).add(
                (entry.participant, governing))
    counts = {"conversation": len(set().union(*counted.values()))}
    # 'conversation_exempt' (review F7): a store exempt run after run shows a step that never completes.
    detail = {"conversation": {participant: len(found) for participant, found in sorted(counted.items())},
              "conversation_exempt": {participant: len(found) for participant, found in sorted(exempt.items())}}
    for family in ("fact", "voice_profile"):
        # Only the embedding may outlive a deletion whose index job is pending: the
        # facts row turns 'deleted' in the deletion's own transaction (review P1).
        found = {name: {candidate for candidate in values - (pending_index if name == "semantic_index" else set())
                        if recorded(candidate, _RESURRECTION_KINDS[family])}
                 for name, values in sources[family].items()}
        counts[family] = len(set().union(*found.values()))
        detail[family] = {name: len(values) for name, values in found.items()}
        if residue is not None:
            residue[family] = set().union(*found.values())
    count = 0
    for salt, subjects in ledgers:
        withdrawn = subjects.get("voice_consent_withdrawn", set())
        window = len(withdrawn)
        count += sum(1 for person, revision in consents if any(
            subject_digest(salt, f"{person}#{later}") in withdrawn
            for later in range(revision + 1, revision + window + 1)))
    counts["voice_consent"] = count
    detail["voice_consent"] = {"voice_attribution_consents": count}
    return {family: counts[family] for family in _RESURRECTION_KINDS}, detail


def resurrection_counts(state: Path, *, descriptor_root: Path | None = None) -> dict[str, int]:
    """Content-free counts of payload ids whose deletion the ledger records (v3 N2 R-b, U3b-0d4a).

    Ids are hashed in memory with each ledger's own salt. A conversation
    counts when at least one of its occurrences (CONVERSATION_SOURCES) is
    not exempt: an occurrence is exempt only while some erasure operation
    for that conversation has a non-complete step on the target governing
    its store. Exactly one SQLite file may hold the erasure journal; more
    than one refuses (runtime_admission_recovery_erasure_journal_ambiguous).
    ``descriptor_root`` is the binding's state_root when ``state`` is a
    shadow copy: descriptor paths are relativized against it, candidate
    files against ``state``. Chroma is read only through a read-only image
    of chroma.sqlite3 (review D2); its fact candidates are the embedding ids
    (review P1), minus facts whose index deletion is still pending.
    """
    return _resurrection_scan(state, descriptor_root)[0]


def resurrection_detail(state: Path, *, descriptor_root: Path | None = None) -> dict[str, dict[str, int]]:
    """Content-free counts per family and participant (conversation) or source table (other families).

    For the R-b report and the adoption plan. A fact or voice profile found in
    two sources is counted under each; the family totals are those of
    resurrection_counts. 'conversation_exempt' counts, per participant, the
    erased conversations with an occurrence exempted by a pending step.
    """
    return _resurrection_scan(state, descriptor_root)[1]


def resurrection_residue(state: Path, *, descriptor_root: Path | None = None) -> dict:
    """The counted residue ids per family, for the adoption's re-queue (U3b-0e2, P3=A).

    Same semantics as resurrection_counts. 'conversation' maps each counted
    conversation to its (participant, governing target ids) occurrences; an
    empty governing set means no target governs that store. 'fact' and
    'voice_profile' are id sets. The ids stay in memory: callers report counts.
    """
    residue: dict = {"conversation": {}, "fact": set(), "voice_profile": set()}
    counts, _detail = _resurrection_scan(state, descriptor_root, residue)
    residue["voice_consent"] = counts["voice_consent"]
    return residue


def receipt_eligibility(root: Path, head: dict) -> dict:
    """Read-only R-a outlook: receipt present, record valid, manifest still matching (review P3).

    Only lstat and JSON reads: nothing under the state is opened, so the
    answer cannot change what it reports.
    """
    receipt_path = _records(root, "receipts", head["run_id"] + ".json")
    outlook = {"receipt": False, "receipt_state": None, "manifest_matches": None}
    if head["status"] != "running" or not receipt_path.exists():
        return outlook
    outlook["receipt"] = True
    binding = stage.read_binding(root, head["binding_id"])
    try:
        record = stage.read_receipt_state(root, stage._read_json(receipt_path))
    except (ValueError, OSError, KeyError):
        outlook["receipt_state"] = "invalid"
        return outlook
    if record is None:
        outlook["receipt_state"] = "missing"
        return outlook
    outlook["receipt_state"] = "present"
    try:
        outlook["manifest_matches"] = (stage.state_manifest_sha256(Path(binding["state_root"]))
                                       == record["state_manifest_sha256"])
    except (ValueError, OSError):
        outlook["manifest_matches"] = False
    return outlook


def recovery_report(root: Path, head: dict, *, read_only: bool = False, progress=None) -> dict:
    """The R-b checks on the head's state: failures refuse; regressions need the operator.

    Order: writers, SQLite WAL recovery plus quick_check, JSON stores,
    Chroma structure, deletion-ledger forward evolution, append-only
    families, resurrection. Ownership and manual-capture equivalence are
    enforced by stage_runtime.adopt_recovered_state itself.

    ``read_only`` (adopt --dry-run, review P3) opens nothing under the state
    for writing: the writers check uses read-only descriptors, and the
    database checks run on a private copy (_shadow_state) outside the state.
    ``progress(step)`` is called between steps (the serve start-time
    extension, review P6).
    """
    binding = stage.read_binding(root, head["binding_id"])
    state = Path(binding["state_root"])
    report = {"failures": [], "regressions": [], "resurrected": {}, "ledger_bound": False, "databases": 0}

    def step(name: str) -> None:
        if progress is not None:
            progress(name)
    step("writers")
    try:
        (_assert_writers_closed_read_only if read_only else stage._assert_writers_closed)(state)
    except ValueError as error:
        report["failures"].append(str(error))
        return report
    if not read_only:
        return _recovery_checks(state, state, binding, report, step)
    with tempfile.TemporaryDirectory(prefix="kza-adopt-dry-run-") as temporary:
        try:
            shadow = _shadow_state(state, Path(temporary) / "state")
        except (ValueError, OSError) as error:
            report["failures"].append(str(error) if isinstance(error, ValueError) else "shadow_copy_failed")
            return report
        return _recovery_checks(state, shadow, binding, report, step)


def _recovery_checks(state: Path, checked: Path, binding: dict, report: dict, step) -> dict:
    """The R-b checks after the writers check; database reads go to ``checked``."""
    databases = _sqlite_files(checked)
    report["databases"] = len(databases)
    for path in databases:
        step("quick_check")
        if not _quick_check(path):
            report["failures"].append("quick_check:" + path.relative_to(checked).as_posix())
    step("json")
    for path in _json_files(checked):
        try:
            json.loads(path.read_bytes())
        except (OSError, ValueError):
            report["failures"].append("json:" + path.relative_to(checked).as_posix())
    step("chroma")
    report["failures"].extend(_chroma_problems(state))  # stat and listing only, on the state itself
    if report["failures"]:
        return report
    tables = binding["authority"]["projection"]["tables"]
    report["ledger_bound"] = any("deletion_ledger_v1" in item for item in tables.values())
    # Nothing protected is bound (a fresh state): there is no deletion to regress.
    report["ledger_required"] = bool(tables)
    step("ledger")
    report["regressions"].extend("deletion_ledger_v1:" + item
                                 for item in stage.deletion_ledger_regressions(binding, checked))
    step("authority")
    try:
        current = stage.capture_state_authority(checked)["projection"]["tables"]
    except ValueError as error:
        report["failures"].append(str(error))
        return report
    except RecursionError:  # a scanned JSON column nested past the limit (review F5)
        report["failures"].append(_UNREADABLE)
        return report
    for relative, bound in sorted(tables.items()):
        for family in _APPEND_ONLY_FAMILIES:
            if family not in bound:
                continue
            now = current.get(relative, {}).get(family)
            if now is None or not set(bound[family]["row_hashes"]) <= set(now["row_hashes"]):
                report["regressions"].append(family + ":" + relative)
    step("resurrection")
    try:
        # Descriptors name paths under the binding's state_root; ``checked`` may be a shadow copy.
        report["resurrected"], report["resurrection_detail"] = _resurrection_scan(checked, state)
    except ValueError as error:
        report["failures"].append(str(error))
    except (sqlite3.Error, RecursionError):
        report["failures"].append(_UNREADABLE)
    return report


def _try_receipt_seal(root: Path, head: dict, run: dict | None) -> RecoveryDecision | None:
    """R-a: resume the seal from an exact receipt; None when there is no receipt."""
    receipt_path = _records(root, "receipts", head["run_id"] + ".json")
    if head["status"] != "running" or not receipt_path.exists():
        return None
    receipt = stage._read_json(receipt_path)
    child_pid = run["child_pid"] if run is not None else receipt.get("child_pid")
    if type(child_pid) is not int:
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code="runtime_admission_close_receipt_mismatch",
                                path="R-a")
    previous = stage.read_manual_head(root, stage.previous_closed_head(head))
    try:
        closed = stage.seal_admission_run(root, head, child_pid, previous)
    except ValueError as error:
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code=str(error), path="R-a")
    return RecoveryDecision(closed_head=closed, path="R-a")


def recover_open_head(root: Path, head: dict, *, mode: str, boot_id: str, wall: float,
                      operator: dict | None = None, progress=None) -> RecoveryDecision:
    """Recover a running or unknown head (R-a, then R-b per mode and budget).

    ``operator`` is set only by the ``adopt`` command: {"head_sha256",
    "regressions"} confirmed by the operator. R-a runs in every mode and
    consumes no budget; a digest mismatch bars R-a and automatic R-b.
    ``progress(step)`` is called before R-a, between the R-b steps and
    before adoption (serve extends its start timeout there, review P6).
    """
    root = Path(root)

    def step(name: str) -> None:
        if progress is not None:
            progress(name)
    step("R-a")
    run = read_run_record(root, head["run_id"])
    if run is not None and child_alive(run["child_pid"], run.get("child_start_time")):
        return RecoveryDecision(exit_code=ExitCode.REFUSED, code="runtime_admission_recovery_child_alive",
                                path="R-c")
    exit_record = read_exit_record(root, head["run_id"])
    sealed = _try_receipt_seal(root, head, run)
    if sealed is not None and sealed.closed_head is not None:
        logger.info("[Launcher] recovery R-a sealed run from its receipt")
        return sealed
    state_changed = ((exit_record or {}).get("state_changed_after_close") is True
                     or (sealed is not None and sealed.code in _DIGEST_MISMATCH))
    if operator is None:
        if mode == "off":
            return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code="runtime_admission_recovery_off",
                                    path="R-b")
        if mode != "bounded":
            return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code="runtime_admission_recovery_operator",
                                    path="R-b")
        if state_changed:
            return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE,
                                    code="runtime_admission_recovery_state_changed_after_close", path="R-b")
        klass = classify_run(run, exit_record, boot_id)
        refused = automatic_budget(root, head, klass, boot_id, wall)
        if refused is not None:
            return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code=refused, path="R-b")
    else:
        klass = "operator"
    report = recovery_report(root, head, progress=progress)
    report["state_changed_after_close"] = state_changed
    if report["failures"]:
        code = next((failure for failure in report["failures"]
                     if failure in ("runtime_admission_recovery_erasure_journal_ambiguous", _UNREADABLE)),
                    "runtime_admission_recovery_check_failed")
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code=code, path="R-b", report=report)
    regressed = sorted({item.split(":", 1)[0] for item in report["regressions"]}
                       | {"resurrected_" + name for name, count in report["resurrected"].items() if count}
                       | ({"state_changed_after_close"} if state_changed else set())
                       | ({"no_deletion_ledger"} if report.get("ledger_required") and not report["ledger_bound"]
                          else set()))
    if operator is None and regressed:
        code = ("runtime_admission_recovery_resurrection" if any(item.startswith("resurrected_") for item in regressed)
                else "runtime_admission_recovery_regression")
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code=code, path="R-b", report=report)
    if operator is not None and sorted(operator.get("regressions", [])) != regressed:
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code="runtime_admission_recovery_confirmation",
                                path="R-b", report=report)
    checks = {"databases": report["databases"], "regressed": regressed,
              "resurrected": report["resurrected"]}
    step("adopt")
    try:
        closed = stage.adopt_recovered_state(root, head, {
            "mode": "automatic" if operator is None else "operator", "class": klass, "boot_id": boot_id,
            "at": float(wall), "checks": checks,
            "confirmation": None if operator is None else {"head_sha256": operator["head_sha256"],
                                                          "regressions": regressed}})
    except ValueError as error:
        return RecoveryDecision(exit_code=ExitCode.UNRECOVERABLE, code=str(error), path="R-b", report=report)
    logger.info("[Launcher] recovery R-b adopted the state (%s, class=%s)",
                "automatic" if operator is None else "operator", klass)
    return RecoveryDecision(closed_head=closed, path="R-b", report=report)


def eligible_exit(root: Path, running: dict, *, mode: str, boot_id: str, wall: float, manager_stopping: bool,
                  state_changed: bool) -> ExitCode:
    """68 when the next start may recover automatically, else 69 (v2 exit_codes_H1)."""
    if mode != "bounded" or state_changed:
        return ExitCode.UNRECOVERABLE
    if manager_stopping:
        return ExitCode.RECOVERABLE
    try:
        refused = automatic_budget(root, running, "crash", boot_id, wall)
    except (ValueError, OSError):
        return ExitCode.UNRECOVERABLE
    return ExitCode.RECOVERABLE if refused is None else ExitCode.UNRECOVERABLE


# --------------------------------------------------------------------------
# serve (S0-S7)


@dataclass
class ServeOptions:
    """Inputs of one ``serve`` invocation."""

    root: Path
    stop_grace: float
    recover: str = "bounded"
    unit_timeout_stop: float = 0.0
    python: str = sys.executable
    child_command: list[str] | None = None
    boot_id_path: str | None = None
    probe_timeout: float = 60.0
    uid: int = field(default_factory=os.getuid)


def child_environment(environ: dict, runtime: Path, ticket: dict, notify_fd: int, control_fd: int,
                      watchdog_interval_usec: int | None) -> dict:
    """The child's environment: the controller's minus manager and loader variables.

    Credentials from the unit's EnvironmentFiles are kept. PYTHONPATH names
    the slot runtime; CONFIG_PATH is relative to that runtime (its config
    link resolves to the slot state).
    """
    child = {key: value for key, value in environ.items()
             if key not in _CHILD_ENV_DROPPED and not key.startswith("KZA_RUNTIME_")}
    child["PYTHONPATH"] = str(runtime)
    child["CONFIG_PATH"] = "config/settings.yaml"
    child[TICKET_ENV] = stage._canonical_json(ticket).decode()
    child[NOTIFY_FD_ENV] = str(notify_fd)
    child[CONTROL_FD_ENV] = str(control_fd)
    if watchdog_interval_usec:
        child[WATCHDOG_INTERVAL_ENV] = str(watchdog_interval_usec)
    return child


def watchdog_interval(environ: dict) -> int | None:
    """Half of WATCHDOG_USEC when systemd armed it for this process."""
    raw = environ.get("WATCHDOG_USEC")
    owner = environ.get("WATCHDOG_PID")
    if not raw or not raw.isdigit() or (owner is not None and owner != str(os.getpid())):
        return None
    value = int(raw) // 2
    return value if value > 0 else None


def path_preflight(binding: dict, python: str, timeout: float = 60.0) -> list[str]:
    """Relative model/log paths, logs under the state and foreign serving stores (S3, v2 M8)."""
    runtime, state = binding["runtime_root"], binding["state_root"]
    environment = {key: os.environ[key] for key in _PROBE_ENVIRONMENT if key in os.environ}
    try:
        completed = subprocess.run([python, "-I", "-B", "-c", _PREFLIGHT_SOURCE, runtime, state,
                                    json.dumps(sorted(_SERVING_KEYS))], cwd=runtime, env=environment,
                                   stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return ["preflight_timeout"]
    try:
        problems = json.loads(completed.stdout)["problems"]
    except (ValueError, KeyError, TypeError):
        return ["preflight_failed"]
    return [str(item)[:200] for item in problems] if completed.returncode == 0 else ["preflight_failed"]


def check_controller_epoch(root: Path) -> None:
    """Refuse a controller older than the installed one (L3 = A, K13), or a damaged install (P4).

    The files of controller/current are re-hashed against the manifest the
    install wrote beside them: a power loss after a non-durable install
    must refuse (78) instead of crash-looping on a partial .pyc.
    """
    record = _read_optional(_records(root, "controller.json"))
    if record is None:
        raise LauncherRefusal("runtime_admission_controller_not_installed")
    if type(record.get("controller_epoch")) is not int or record["controller_epoch"] > stage.CONTROLLER_EPOCH:
        raise LauncherRefusal("runtime_admission_controller_downgrade")
    current = Path(root) / "controller" / "current"
    try:
        manifest = json.loads((current / "release.json").read_bytes())
        intact = isinstance(manifest.get("files"), dict) and bool(manifest["files"]) and all(
            not (current / name).is_symlink() and (current / name).is_file()
            and hashlib.sha256((current / name).read_bytes()).hexdigest() == digest
            for name, digest in manifest["files"].items())
    except (OSError, ValueError, AttributeError):
        intact = False
    if not intact:
        raise LauncherRefusal("runtime_admission_controller_tampered")


def check_slot(root: Path, binding: dict) -> dict:
    """S3: selection, launcher contract, format epoch and companions (78 on refusal)."""
    current = root / "current"
    expected = "slots/" + binding["slot"] + "/runtime"
    if not current.is_symlink() or os.readlink(current) != expected:
        raise LauncherRefusal("runtime_admission_launcher_selection_mismatch")
    try:
        stage._require_launcher_slot(binding)  # the same integers operator activate applies (P10)
    except ValueError as error:
        raise LauncherRefusal(str(error)) from None
    values = stage.release_integers(binding)
    runtime = Path(binding["runtime_root"])
    for companion in stage._PROBE_COMPANIONS:
        path = runtime / companion
        if path.is_symlink() or not path.is_file():
            raise LauncherRefusal("runtime_admission_candidate_unmanaged")
    return values


class Serve:
    """One ``serve`` run: recovery, checks, the child and its seal (design v2 S0-S7)."""

    def __init__(self, options: ServeOptions, *, manager: ServiceManager | None = None,
                 notifier: SystemdNotifier | None = None, environ: dict | None = None,
                 clock=time.monotonic, wall=time.time, straggler_procs: Path | None = None) -> None:
        self.options = options
        self.root = Path(options.root).resolve()
        self.manager = manager if manager is not None else ServiceManager()
        self.environ = dict(os.environ if environ is None else environ)
        self.notifier = notifier if notifier is not None else SystemdNotifier.from_environment(self.environ)
        self.clock = clock
        self.wall = wall
        self.straggler_procs = straggler_procs if straggler_procs is not None else _cgroup_procs_path()
        self.stop = StopPipe()
        self.boot_id = read_boot_id(options.boot_id_path)
        self.events: list[str] = []  # content-free order log (tests, evidence)

    def _event(self, name: str) -> None:
        self.events.append(name)
        logger.info("[Launcher] %s", name)

    def run(self) -> int:
        """Serve until the child ends or a stop arrives; returns the exit code."""
        self.stop.install()
        try:
            code = self._serve()
        except LauncherRefusal as refusal:
            self._event("refused " + refusal.code)
            self.notifier.status("refused: " + refusal.code)
            code = refusal.exit_code
        except Exception as error:  # noqa: BLE001 - a coded transient refusal
            self._event("controller error " + type(error).__name__ + " " + str(error)[:120])
            self.notifier.status("controller error")
            code = ExitCode.TRANSIENT
        finally:
            self.stop.close()
            self.notifier.close()
        self._event(f"exit {int(code)}")
        return int(code)

    def _extend_start(self, step: str) -> None:
        """EXTEND_TIMEOUT_USEC for one more recovery step (systemd honours it while activating)."""
        self.notifier.send(f"EXTEND_TIMEOUT_USEC={RECOVERY_EXTEND_SECONDS * 1_000_000}")
        self.events.append("extend " + step)

    def _stopped_before_start(self) -> bool:
        if self.stop.drain():
            self._event("stop before CAS running; head unchanged")
            return True
        return False

    def _serve(self) -> int:
        options = self.options
        if options.recover not in RECOVER_MODES:
            raise LauncherRefusal("runtime_admission_recover_mode")
        if not (type(options.stop_grace) in (int, float) and math.isfinite(options.stop_grace)
                and options.stop_grace > 0):
            raise LauncherRefusal("runtime_admission_stop_grace")
        self.notifier.status("starting")
        budget = RebootBudget(self.manager.stop_budget(options.uid), float(options.unit_timeout_stop))
        self.budget = budget
        self._event(f"reboot budget mode={budget.mode} budget={budget.budget}")
        try:
            controller = stage.AdmissionController(self.root, probe_timeout=options.probe_timeout,
                                                   allow_open_head=True)
            controller.__enter__()
        except ValueError as error:
            # runtime_admission_busy: a live child of a dead controller still holds
            # the lease (R-c), or an operator command runs.
            raise LauncherRefusal(str(error)) from None
        try:
            return self._serve_leased(controller, budget)
        finally:
            controller.__exit__(None, None, None)

    def _serve_leased(self, controller, budget: RebootBudget) -> int:
        root = self.root
        if self._stopped_before_start():
            return ExitCode.CLEAN
        check_controller_epoch(root)
        head = stage.read_admission_head(root)
        if head is None:
            raise LauncherRefusal("runtime_admission_launcher_no_managed_head")
        if head["status"] != "closed":
            self.notifier.status("recovering")
            # R-b runs quick_check on every database, a full authority capture and
            # a whole-state scan: ask for start time before it and between its
            # steps, so TimeoutStartSec never lands inside an adoption (P6).
            self._extend_start("recovery")
            decision = recover_open_head(root, head, mode=self.options.recover, boot_id=self.boot_id,
                                         wall=self.wall(), progress=self._extend_start)
            self._event(f"recovery {decision.path} {'closed' if decision.closed_head else decision.code}")
            if decision.closed_head is None:
                self.notifier.status("recovery refused: " + decision.code)
                return decision.exit_code
            head = decision.closed_head
        if self._stopped_before_start():
            return ExitCode.CLEAN
        binding = stage.read_binding(root, head["binding_id"])
        try:
            check_slot(root, binding)
        except LauncherRefusal:
            raise
        problems = path_preflight(binding, self.options.python, timeout=self.options.probe_timeout)
        if problems:
            raise LauncherRefusal("runtime_admission_launcher_path:" + problems[0])
        if budget.status() is not None:
            self.notifier.status(budget.status())
        if self._stopped_before_start():
            return ExitCode.CLEAN
        self.notifier.send(f"EXTEND_TIMEOUT_USEC={int((self.options.probe_timeout + 30) * 1_000_000)}")
        try:
            controller.activate(binding["slot"], managed=True)
        except ValueError as error:
            raise LauncherRefusal(str(error)[:200]) from None
        if self._stopped_before_start():
            return ExitCode.CLEAN
        return self._run_child(controller, budget)

    def _spawn(self, controller, binding: dict, ticket: dict):
        runtime = Path(binding["runtime_root"])
        notify_read, notify_write = os.pipe()
        control_read, control_write = os.pipe()
        os.set_blocking(notify_read, False)
        os.set_blocking(control_write, False)
        environment = child_environment(self.environ, runtime, ticket, notify_write, control_read,
                                        watchdog_interval(self.environ))
        command = (list(self.options.child_command) if self.options.child_command is not None
                   else [self.options.python, "-B", "-P", "-m", "src.main"])
        try:
            process = subprocess.Popen(command, cwd=runtime, env=environment, stdin=subprocess.DEVNULL,
                                       pass_fds=(controller.lease_descriptor, notify_write, control_read))
        finally:
            os.close(notify_write)
            os.close(control_read)
        return process, notify_read, control_write

    def _run_child(self, controller, budget: RebootBudget) -> int:
        root = self.root
        binding, running, previous_manual, ticket = controller.begin_run()
        self._event("head running")
        process = None
        notify_read = control_write = None
        try:
            process, notify_read, control_write = self._spawn(controller, binding, ticket)
            stage._atomic_record(run_record_path(root, running["run_id"]), {
                "schema_version": 1, "run_id": running["run_id"], "generation": running["generation"],
                "binding_id": running["binding_id"], "child_pid": process.pid,
                "child_start_time": process_start_time(process.pid), "controller_pid": os.getpid(),
                "boot_id": self.boot_id, "started_at": float(self.wall())}, immutable=True)
            self._event("child started")
            outcome = self._wait(process, notify_read, control_write, budget)
        except Exception as error:  # noqa: BLE001 - a controller bug after CAS: stop the child, UNKNOWN
            self._event("controller error after CAS " + type(error).__name__)
            outcome = {"reason": "controller_error", "manager_stopping": False}
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=self.options.stop_grace)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            for descriptor in (notify_read, control_write):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        returncode = process.returncode if process is not None else None
        stragglers = kill_stragglers(self.straggler_procs)
        sealed, seal_error, transient = None, None, False
        if process is not None:
            try:
                sealed = stage.seal_admission_run(root, running, process.pid, previous_manual)
            except ValueError as error:
                seal_error = str(error)
            except OSError as error:
                seal_error, transient = "seal_os_error:" + type(error).__name__, True
        state_changed = seal_error in _DIGEST_MISMATCH
        stop_requested = self.stop.drain()
        stage._atomic_record(exit_record_path(root, running["run_id"]), {
            "schema_version": 1, "run_id": running["run_id"], "returncode": returncode,
            "reason": outcome["reason"], "manager_stopping": outcome["manager_stopping"],
            "boot_id": self.boot_id, "stragglers": stragglers, "sealed": sealed is not None,
            "seal_error": seal_error, "state_changed_after_close": state_changed,
            "stop_requested": stop_requested, "ended_at": float(self.wall())}, immutable=True)
        if sealed is not None:
            self._event("sealed closed")
            return ExitCode.CLEAN if stop_requested else ExitCode.CHILD_ENDED
        if transient:
            # The receipt may still prove the close: leave the head running for R-a.
            self._event("seal deferred " + seal_error)
            return ExitCode.TRANSIENT
        with stage._locked_root(root):
            if stage.read_admission_head(root) == running:
                stage._cas_head(root, running, running["binding_id"], "unknown", running["run_id"])
        self._event("head unknown " + (seal_error or ""))
        code = eligible_exit(root, running, mode=self.options.recover, boot_id=self.boot_id,
                             wall=self.wall(), manager_stopping=outcome["manager_stopping"],
                             state_changed=state_changed)
        self.notifier.status("head unknown: " + (seal_error or "no receipt"))
        return code

    def _wait(self, process, notify_read: int, control_write: int, budget: RebootBudget) -> dict:
        """The controller loop (v2 M1): self-pipe, notify pipe, 0.5 s poll; never an unbounded wait."""
        relay = NotifyRelay(self.notifier)
        self.relay = relay
        selector = selectors.DefaultSelector()
        selector.register(self.stop.read_fd, selectors.EVENT_READ, "stop")
        selector.register(notify_read, selectors.EVENT_READ, "notify")
        deadline = None
        stop_at = None
        forwarded = False
        manager_stopping = False
        reason = "child_exit"
        try:
            while process.poll() is None:
                timeout = 0.5 if deadline is None else max(0.0, min(0.5, deadline - self.clock()))
                for key, _ in selector.select(timeout):
                    if key.data == "notify":
                        self._read_notify(selector, notify_read, relay)
                    elif self.stop.drain() and stop_at is None:
                        stop_at = self.clock()
                        reason = "stop"
                        # (1) forward at once (as soon as the child's handler exists),
                        # (2) only then ask the system manager (v3 N9).
                        forwarded = self._forward(process, relay, stop_at)
                        state = self.manager.system_state()
                        self._event(f"system state {state}")
                        grace = self.options.stop_grace
                        if state == "stopping":
                            manager_stopping = True
                            reason = "manager_stopping"
                            try:
                                os.write(control_write, b"fast\n")
                            except OSError:
                                pass
                            grace = budget.deadline_seconds(grace, self.clock() - stop_at)
                            self._event(f"shutdown mode grace={grace:.1f}")
                        deadline = stop_at + grace
                        self.notifier.status("stopping")
                if stop_at is not None and not forwarded:
                    forwarded = self._forward(process, relay, stop_at)
                if deadline is not None and self.clock() >= deadline and process.poll() is None:
                    self._event("stop grace exceeded; SIGKILL")
                    process.kill()
                    process.wait()
                    reason = "stop_grace_exceeded" if not manager_stopping else "manager_stopping"
            self._read_notify(selector, notify_read, relay)
        finally:
            selector.close()
        if relay.dropped:
            self._event(f"notify lines dropped {relay.dropped}")
        return {"reason": reason, "manager_stopping": manager_stopping}

    def _forward(self, process, relay: NotifyRelay, stop_at: float) -> bool:
        """SIGTERM the child once its stop handler is armed (ARMED=1), or after ARM_WAIT.

        A SIGTERM that reached the child before arm_process_entry ran would
        meet the default disposition and kill it without a receipt.
        """
        if not relay.armed and self.clock() - stop_at < ARM_WAIT_SECONDS:
            return False
        try:
            process.send_signal(signal.SIGTERM)
        except OSError:
            pass
        self._event("stop forwarded" if relay.armed else "stop forwarded unarmed")
        return True

    @staticmethod
    def _read_notify(selector, descriptor: int, relay: NotifyRelay) -> None:
        while True:
            try:
                data = os.read(descriptor, 4096)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                data = b""
            if not data:
                try:
                    selector.unregister(descriptor)
                except (KeyError, ValueError):
                    pass
                return
            relay.feed(data)


# --------------------------------------------------------------------------
# operator adoption


def adopt(root: Path, *, confirm: str, regressions: list[str], boot_id_path: str | None = None,
          wall=time.time, dry_run: bool = False) -> dict:
    """Operator R-b (or R-a) for an open head; prints counts only (L1, v1 modes.operator).

    ``confirm`` must be the head's sha256 (the digest of head.json's
    content); ``regressions`` must list exactly the regressed families the
    checks report (forward evolution, resurrection, state changed after
    close, no ledger). With ``dry_run`` only the report is returned, and
    nothing under the state is opened for writing.
    """
    root = Path(root).resolve()
    with stage.AdmissionController(root, allow_open_head=True):
        head = stage.read_admission_head(root)
        if head is None or head["status"] == "closed":
            raise LauncherRefusal("runtime_admission_adopt_head_closed")
        head_sha = stage._digest(head)
        if dry_run:
            # Read-only (review P3): the R-a outlook first (lstat and JSON only),
            # then the R-b checks on a private copy. A dry run must never
            # checkpoint a -wal the receipt's manifest still names.
            outlook = receipt_eligibility(root, head)
            report = recovery_report(root, head, read_only=True)
            return {"head_sha256": head_sha, "status": head["status"], "r_a": outlook, "report": report}
        if confirm != head_sha:
            raise LauncherRefusal("runtime_admission_adopt_confirmation")
        decision = recover_open_head(root, head, mode="operator", boot_id=read_boot_id(boot_id_path),
                                     wall=wall(), operator={"head_sha256": head_sha,
                                                            "regressions": sorted(regressions)})
        if decision.closed_head is None:
            raise LauncherRefusal(decision.code, decision.exit_code or ExitCode.UNRECOVERABLE)
        return {"head": decision.closed_head, "path": decision.path,
                "checks": {"regressed": decision.report.get("regressions", []),
                           "resurrected": decision.report.get("resurrected", {})}}


def completion_status(root: Path) -> dict:
    """Content-free counts of privacy work still open in the head's state (review U3b-0e4 F5).

    For the runbook's check after W9: the adoption re-queues residue and
    relocates legacy_training steps, and the first managed runs must finish
    them. A pending step exempts its occurrences from the resurrection check,
    so they would otherwise stay unnoticed. Read only: every database is read
    through a private image (``mode=ro`` then backup), so it can run with the
    service up. ``complete`` is True only when every count is 0. Steps on the
    cleaner-less literal targets (``unresolved_*_inventory``, U3b-0d4a) never
    complete by design: they are counted apart, as ``honest_blocker_steps``,
    and do not count against ``complete``.
    """
    root = Path(root).resolve()
    head = stage.read_admission_head(root)
    if head is None:
        raise LauncherRefusal("runtime_admission_launcher_no_managed_head")
    state = Path(stage.read_binding(root, head["binding_id"])["state_root"])
    steps: dict[str, int] = {}
    blockers: dict[str, int] = {}
    training = index_deletions = index_jobs = 0
    for path in _sqlite_files(state):
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"erasure_steps", "erasure_targets"} <= names:
                for participant, descriptor in db.execute(
                        "SELECT s.participant, t.descriptor FROM erasure_steps s LEFT JOIN erasure_targets t"
                        " ON t.id = s.target_id WHERE s.state != 'complete'"):
                    try:
                        kind = json.loads(descriptor or "{}").get("kind")
                    except (ValueError, AttributeError):
                        kind = None
                    honest = type(kind) is str and re.fullmatch(r"unresolved_[a-z]+_inventory", kind) is not None
                    counted = blockers if honest else steps
                    counted[participant] = counted.get(participant, 0) + 1
                    training += kind == "legacy_training"
            if "policy_index_deletions" in names:
                index_deletions += db.execute(
                    "SELECT COUNT(*) FROM policy_index_deletions WHERE state != 'complete'").fetchone()[0]
            if "index_jobs" in names:
                index_jobs += db.execute("SELECT COUNT(*) FROM index_jobs WHERE operation='delete'").fetchone()[0]
    detail = resurrection_detail(state)
    exempt = sum(detail["conversation_exempt"].values())
    resurrected = {family: sum(values.values()) for family, values in detail.items() if family != "conversation_exempt"}
    complete = (not steps and training == 0 and index_deletions == 0 and index_jobs == 0 and exempt == 0
                and not any(resurrected.values()))
    return {"head_status": head["status"], "erasure_steps_not_complete": dict(sorted(steps.items())),
            "honest_blocker_steps": dict(sorted(blockers.items())),
            "legacy_training_steps_not_complete": training,
            "policy_index_deletions_not_complete": index_deletions, "index_delete_jobs": index_jobs,
            "conversation_exempt": detail["conversation_exempt"], "resurrected": resurrected,
            "complete": complete}


# --------------------------------------------------------------------------
# install-controller (L3 = A)


def _fsync_path(path: Path) -> None:
    """fsync one file or directory (opened read-only; directories with O_DIRECTORY)."""
    flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(directory: Path) -> None:
    """fsync every regular file under ``directory``, then each directory deepest first."""
    paths = sorted(Path(directory).rglob("*"), key=lambda item: (-len(item.parts), item.as_posix()))
    for path in paths:
        if path.is_file() and not path.is_symlink():
            _fsync_path(path)
    for path in [*paths, Path(directory)]:
        if path.is_dir() and not path.is_symlink():
            _fsync_path(path)


def install_controller(root: Path, archive: Path, *, wall=time.time) -> dict:
    """Install a controller under <root>/controller/<commit>; monotonic epoch, atomic switch.

    The archive is verified (manifest hashes, CPython 3.13) and must declare
    launcher_contract >= 3 and a controller_epoch at least the installed one.
    Returns which installed slots the new controller closes for rollback
    (format_epoch below its minimum) -- the rollback window.
    """
    root = Path(root).resolve()
    manifest, files = stage._read_archive(Path(archive))
    epoch = manifest.get("controller_epoch", 0)
    if type(epoch) is not int or epoch < 1 or manifest.get("launcher_contract", 0) < stage.LAUNCHER_CONTRACT:
        raise LauncherRefusal("runtime_admission_controller_unmanaged")
    if epoch < stage.CONTROLLER_EPOCH:
        # U3b-0e3: an older controller cannot read the adoption head and seal.
        raise LauncherRefusal("runtime_admission_controller_epoch_too_old")
    if "scripts/managed_launcher.pyc" not in files or "scripts/stage_runtime.pyc" not in files:
        raise LauncherRefusal("runtime_admission_controller_unmanaged")
    commit = manifest["git_commit"]
    with stage.AdmissionController(root, allow_open_head=True):
        record_path = _records(root, "controller.json")
        installed = _read_optional(record_path)
        if installed is not None and installed.get("controller_epoch", 0) > epoch:
            raise LauncherRefusal("runtime_admission_controller_downgrade")
        controllers = root / "controller"
        stage.private_directories(controllers)
        target = controllers / commit
        if target.exists():
            for name, digest in manifest["files"].items():
                path = target / name
                if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise LauncherRefusal("runtime_admission_controller_tampered")
        else:
            with tempfile.TemporaryDirectory(prefix=".install-", dir=controllers) as temporary:
                staging = Path(temporary) / commit
                # 0700/0600 whatever the umask: only the controller's user reads it.
                for name, data in files.items():
                    destination = staging / name
                    stage.private_directories(destination.parent)
                    stage.write_private_file(destination, data)
                stage.write_private_file(staging / "release.json", (json.dumps(manifest, indent=2) + "\n").encode())
                # Durable before it is reachable (P4): every file, then every directory.
                _fsync_tree(staging)
                os.rename(staging, target)
                _fsync_path(Path(temporary))
                _fsync_path(controllers)
        with tempfile.TemporaryDirectory(prefix=".current-", dir=controllers) as temporary:
            link = Path(temporary) / "current"
            link.symlink_to(commit)
            os.replace(link, controllers / "current")
            _fsync_path(controllers)  # the switch itself
        closed = []
        for release in sorted((root / "slots").glob("*/runtime/release.json")):
            try:
                value = json.loads(release.read_text()).get("format_epoch", 0)
            except (OSError, ValueError):
                value = 0
            if type(value) is not int or value < stage.MIN_FORMAT_EPOCH:
                closed.append(release.parent.parent.name)
        stamp = _now_stamp(wall())
        record = {"schema_version": 1, "controller_epoch": epoch, "commit": commit,
                  "min_format_epoch": stage.MIN_FORMAT_EPOCH, "installed_at": stamp}
        stage._atomic_record(record_path, record)
        stage._atomic_record(_records(root, "launcher", f"controller-{stamp}.json"),
                             dict(record, closed_for_rollback=closed, previous=installed), immutable=True)
    return {"commit": commit, "controller_epoch": epoch, "closed_for_rollback": closed,
            "path": str(controllers / "current" / "scripts" / "stage_runtime.pyc")}


# --------------------------------------------------------------------------
# install-launcher / uninstall-launcher (v1 I4, v2 install_M9, v3 N8)


@dataclass
class LauncherSettings:
    """Values rendered into managed.conf (measured on the server, L2/L6)."""

    python: str
    stop_grace: float
    timeout_start: float
    timeout_stop: float
    watchdog_sec: float
    console_log: str
    recover: str = "bounded"
    logrotate_conf: str | None = None  # default <root>/console-logrotate.conf


def render_managed_conf(template: str, root: Path, settings: LauncherSettings) -> str:
    """Render the tracked managed.conf template with absolute values."""
    values = {"python": settings.python, "root": str(root),
              "controller": str(Path(root) / "controller" / "current" / "scripts" / "stage_runtime.pyc"),
              "stop_grace": f"{settings.stop_grace:g}", "recover": settings.recover,
              "timeout_start": f"{settings.timeout_start:g}", "timeout_stop": f"{settings.timeout_stop:g}",
              "watchdog_sec": f"{settings.watchdog_sec:g}", "console_log": settings.console_log}
    for key, value in values.items():
        if any(char in str(value) for char in "\n\r%$"):
            raise LauncherRefusal("runtime_admission_launcher_value:" + key)
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    if re.search(r"\{[a-z_]+\}", text):
        raise LauncherRefusal("runtime_admission_launcher_template")
    return text


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _write_text_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".launcher-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _environment_digest(environment: str) -> str:
    entries = sorted(item for item in shlex.split(environment or "") if not item.startswith("PYTHONPATH="))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _content_free_properties(properties: dict) -> dict:
    """Recorded pre-managed properties; Environment only as a digest (it may hold values)."""
    recorded = {key: value for key, value in properties.items() if key != "Environment"}
    recorded["Environment_sha256"] = _environment_digest(properties.get("Environment", ""))
    return recorded


def exec_start_argv(value: str) -> list[str]:
    """argv of a ``systemctl show -p ExecStart`` value (``{ path=... ; argv[]=... ; ... }``)."""
    match = re.search(r"argv\[\]=(.*?) ;", value or "")
    if match is None:
        raise LauncherRefusal("runtime_admission_launcher_exec_start")
    return shlex.split(match.group(1))


def render_console_logrotate(console_log: str) -> str:
    """The logrotate stanza for the managed console log (review P5); kza applies it, never root.

    Mirrors the existing console rotation (hourly timer, 24 rotations,
    maxage 1, compression). copytruncate is required: systemd holds the
    StandardOutput=append: descriptor for the life of the unit.
    """
    path = str(console_log)
    if not Path(path).is_absolute() or any(char.isspace() or char in "{}\"'#" for char in path):
        raise LauncherRefusal("runtime_admission_launcher_value:console_log")
    return (f"{path} {{\n    hourly\n    rotate 24\n    maxage 1\n    compress\n    delaycompress\n"
            "    copytruncate\n    missingok\n    notifempty\n}\n")


def console_rotated(conf: Path, console_log: str) -> bool:
    """Whether a logrotate configuration rotates ``console_log`` with copytruncate."""
    try:
        text = Path(conf).read_text()
    except (OSError, UnicodeDecodeError):
        return False
    target = str(console_log)
    names: list[str] = []
    body: list[str] = []
    inside = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not inside and "{" in line:
            header, _, rest = line.partition("{")
            try:
                names = shlex.split(header)
            except ValueError:
                names = []
            body, inside = [rest.strip()], True
            line = rest
        elif inside:
            body.append(line)
        if inside and "}" in line:
            inside = False
            if target in names and "copytruncate" in " ".join(body).split():
                return True
    return False


def install_preflight(root: Path, binding: dict, settings: LauncherSettings, *, timeout: float = 60.0) -> list[str]:
    """The serve start gates, run before install-launcher touches the unit directory (review P2, P5).

    Exactly what serve checks at S3 on the head binding (selection, launcher
    contract, format epoch, companions, the path preflight with the
    rendered interpreter), plus what systemd needs before serve even runs:
    the console directory must exist and be writable (StandardOutput=append:
    creates no directories; status 209/STDOUT otherwise), and the console
    must be rotated. Relocating production paths is U3b-0e's job: this
    only detects and refuses.
    """
    problems = []
    try:
        check_slot(root, binding)
    except LauncherRefusal as refusal:
        problems.append(refusal.code)
    problems.extend("path:" + item for item in path_preflight(binding, settings.python, timeout=timeout))
    console = Path(settings.console_log)
    parent = console.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        problems.append("console_directory_missing_or_not_writable")
    elif console.exists() and (console.is_symlink() or not console.is_file() or not os.access(console, os.W_OK)):
        problems.append("console_log_not_writable")
    conf = Path(settings.logrotate_conf) if settings.logrotate_conf else Path(root) / LOGROTATE_NAME
    if not console_rotated(conf, settings.console_log):
        problems.append("console_not_rotated")
    return problems


def _rendered_exec_start(text: str) -> list[str]:
    lines = [line[len("ExecStart="):] for line in text.splitlines() if line.startswith("ExecStart=")]
    return shlex.split(lines[-1]) if lines and lines[-1] else []


def verify_managed_properties(shown: dict | None, text: str, root: Path) -> list[str]:
    """Mismatches between the effective unit and the rendered managed.conf (review P7, P9)."""
    if shown is None:
        return ["properties_unavailable"]
    expected = {"Type": "notify", "NotifyAccess": "main", "KillMode": "mixed"}
    mismatched = sorted(key for key, value in expected.items() if shown.get(key) != value)
    for key, directive in (("TimeoutStartUSec", "TimeoutStartSec"), ("TimeoutStopUSec", "TimeoutStopSec")):
        rendered = re.search(r"^" + directive + r"=(\S+)\s*$", text, re.M)  # read back (review U3b-0e4 F3)
        if rendered is None or parse_timespan(shown.get(key)) != parse_timespan(rendered.group(1)):
            mismatched.append(key)
    if set((shown.get("RestartPreventExitStatus") or "").split()) != {"69", "78"}:
        mismatched.append("RestartPreventExitStatus")
    try:
        if exec_start_argv(shown.get("ExecStart", "")) != _rendered_exec_start(text):
            mismatched.append("ExecStart")
    except LauncherRefusal:
        mismatched.append("ExecStart")
    if shown.get("WorkingDirectory") != str(root):
        mismatched.append("WorkingDirectory")
    return mismatched


def _require_inactive(manager: ServiceManager) -> None:
    state = manager.user_show(UNIT, ("ActiveState",))
    if state is None:  # systemctl failed or timed out: unknown, never reported as active (0e5 review F6)
        raise LauncherRefusal("runtime_admission_launcher_unit_state_unavailable")
    if state["ActiveState"] not in {"inactive", "failed"}:
        raise LauncherRefusal("runtime_admission_launcher_unit_active")


def install_launcher(root: Path, *, unit_dir: Path, template: str, settings: LauncherSettings,
                     manager: ServiceManager, uid: int | None = None,
                     after_chain_units: tuple[str, ...] = ("kza-llm-ik.service",), wall=time.time) -> dict:
    """Render managed.conf, set release.conf aside and record the pre-managed properties.

    Runs only with the unit stopped and a managed head present; refuses a
    console path under <root>/slots/, and refuses before touching the unit
    directory whatever serve would refuse at start (install_preflight).
    Never edits credential drop-ins and never writes the sudo drop-in: its
    text is returned (and recorded) for the user to apply in an authorized
    window (N1 = B). The effective unit is verified after the reload; any
    mismatch is returned and the CLI exits 78.
    """
    root = Path(root).resolve()
    unit_dir = Path(unit_dir)
    uid = os.getuid() if uid is None else uid
    if settings.recover not in RECOVER_MODES:
        raise LauncherRefusal("runtime_admission_recover_mode")
    console = Path(settings.console_log)
    if not console.is_absolute() or console.resolve(strict=False).is_relative_to(root / "slots"):
        raise LauncherRefusal("runtime_admission_launcher_console_under_slots")
    if settings.stop_grace + 30 > settings.timeout_stop:
        raise LauncherRefusal("runtime_admission_launcher_stop_budget")
    start = settings.timeout_start
    if not (type(start) in (int, float) and math.isfinite(start) and float(start).is_integer()
            and TIMEOUT_START_RANGE[0] <= start <= TIMEOUT_START_RANGE[1]):
        raise LauncherRefusal("runtime_admission_launcher_timeout_start")
    _require_inactive(manager)
    with stage.AdmissionController(root, allow_open_head=True):
        head = stage.read_admission_head(root)
        if head is None:
            raise LauncherRefusal("runtime_admission_launcher_no_managed_head")
        check_controller_epoch(root)
        # Never install a unit that serve would refuse with 78 (no restart).
        problems = install_preflight(root, stage.read_binding(root, head["binding_id"]), settings)
        if problems:
            raise LauncherRefusal("runtime_admission_launcher_preflight", problems=problems)
        properties = manager.user_show(UNIT, RECORDED_PROPERTIES)
        if properties is None:
            raise LauncherRefusal("runtime_admission_launcher_properties_unavailable")
        text = render_managed_conf(template, root, settings)
        release_conf = unit_dir / "release.conf"
        aside = unit_dir / "release.conf.unmanaged"
        release_sha = _sha256(release_conf)
        if release_conf.exists():
            if aside.exists() and _sha256(aside) != release_sha:
                raise LauncherRefusal("runtime_admission_launcher_release_conf_conflict")
            os.replace(release_conf, aside)
        managed = unit_dir / "managed.conf"
        _write_text_atomic(managed, text)
        chain = 0.0
        for unit in after_chain_units:
            shown = manager.user_show(unit, ("TimeoutStopUSec",))
            chain += parse_timespan((shown or {}).get("TimeoutStopUSec")) or 0.0
        dropin = render_stop_timeout_dropin(uid, settings.timeout_stop, chain)
        stamp = _now_stamp(wall())
        record = {"schema_version": 1, "launcher_contract": stage.LAUNCHER_CONTRACT,
                  "managed_conf_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "release_conf_sha256": release_sha, "properties": _content_free_properties(properties),
                  "legacy_exec_start": exec_start_argv(properties["ExecStart"]) if properties["ExecStart"] else None,
                  "settings": {"stop_grace": settings.stop_grace, "timeout_start": settings.timeout_start,
                               "timeout_stop": settings.timeout_stop, "watchdog_sec": settings.watchdog_sec,
                               "recover": settings.recover, "console_log": settings.console_log},
                  "stop_timeout_dropin": {key: dropin[key] for key in ("path", "value_seconds", "text")},
                  "logrotate_conf": str(Path(settings.logrotate_conf) if settings.logrotate_conf
                                        else root / LOGROTATE_NAME),
                  "retired_tools": ["2026-10-29 publication tool --rollback (restores release.conf)"],
                  "installed_at": stamp}
        stage._atomic_record(_records(root, "launcher", f"install-{stamp}.json"), record, immutable=True)
    reloaded = manager.daemon_reload()
    shown = manager.user_show(UNIT, ("Type", "NotifyAccess", "KillMode", "RestartPreventExitStatus", "ExecStart",
                                     "WorkingDirectory", "TimeoutStartUSec", "TimeoutStopUSec", "DropInPaths"))
    # Any mismatch fails the command (78) and leaves managed.conf in place for the operator.
    mismatched = verify_managed_properties(shown, text, root)
    return {"managed_conf": str(managed), "record": f"install-{stamp}.json", "reloaded": reloaded,
            "verification_mismatches": mismatched, "drop_in_paths": (shown or {}).get("DropInPaths", "").split(),
            "stop_timeout_dropin": dropin}


def _latest_install(root: Path) -> dict:
    records = sorted(_records(root, "launcher").glob("install-*.json"))
    if not records:
        raise LauncherRefusal("runtime_admission_launcher_not_installed")
    return stage._read_json(records[-1])


def legacy_retirement_directory(root: Path) -> Path:
    """admission/legacy-retirement: the Q5 intents and records of the b0866e8-state retirement (U3b-0e5)."""
    return _records(root, LEGACY_RETIREMENT_DIRECTORY)


def legacy_retirements(root: Path) -> list[tuple[str, dict]]:
    """Every Q5 intent and record as (name, value), validated; anything else in the directory refuses.

    An intent alone already makes its slots ineligible for L4: the deletion
    may have started (design v3 7, N4b).
    """
    directory = legacy_retirement_directory(root)
    if not os.path.lexists(directory):
        return []
    if os.path.islink(directory) or not directory.is_dir():
        raise LauncherRefusal("runtime_admission_uninstall_legacy_retirement_invalid")
    found = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):  # an interrupted atomic write's temporary file
            continue
        match = _LEGACY_RETIREMENT_NAME.fullmatch(name)
        try:
            value = stage._read_json(directory / name) if match else None
        except (OSError, ValueError):
            value = None
        if (match is None or type(value) is not dict or value.get("schema_version") != 1
                or value.get("kind") != _LEGACY_RETIREMENT_KINDS[match.group(1)]
                or value.get("plan_sha256") != match.group(2)
                or type(value.get("ineligible_slots")) is not list or not value["ineligible_slots"]
                or not all(type(slot) is str for slot in value["ineligible_slots"])):
            raise LauncherRefusal("runtime_admission_uninstall_legacy_retirement_invalid")
        found.append((name, value))
    return found


def legacy_retirement_naming(root: Path, slot: str) -> str | None:
    """The Q5 record (or, while it is pending, the intent) that makes ``slot`` ineligible for L4."""
    named = [name for name, value in legacy_retirements(root) if slot in value["ineligible_slots"]]
    if not named:
        return None
    retired = [name for name in named if name.startswith(LEGACY_RETIRED_PREFIX)]
    return LEGACY_RETIREMENT_DIRECTORY + "/" + (retired or named)[0]


def verify_legacy_links(runtime: Path, state: Path) -> None:
    """The rollback-target link check (design v3 7, review N4a).

    Every link directly inside the legacy slot's runtime and state (the
    63ea8d3 container is five of them) must resolve, through every hop, to
    an existing target; config and data must resolve to directories. A
    dangling, looping or missing target refuses with
    runtime_admission_uninstall_legacy_state, so L4 never renders a
    legacy.conf whose WorkingDirectory cannot serve.
    """
    for directory in (runtime, state):
        try:
            entries = list(os.scandir(directory))
        except OSError:
            raise LauncherRefusal("runtime_admission_uninstall_legacy_state") from None
        for entry in entries:
            if entry.is_symlink():
                try:
                    os.path.realpath(entry.path, strict=True)
                except OSError:
                    raise LauncherRefusal("runtime_admission_uninstall_legacy_state") from None
    for name in ("config", "data"):
        try:
            target = os.path.realpath(state / name, strict=True)
        except OSError:
            raise LauncherRefusal("runtime_admission_uninstall_legacy_state") from None
        if not os.path.isdir(target):
            raise LauncherRefusal("runtime_admission_uninstall_legacy_state")


def _verify_legacy_slot(root: Path, legacy_slot: str, legacy_commits: tuple[str, ...]) -> tuple[Path, Path, str]:
    name = stage._slot_name(legacy_slot)
    retired = legacy_retirement_naming(root, name)
    if retired is not None:  # Q5 (P7=B, Q3=A): after the retirement there is no unmanaged rollback
        raise LauncherRefusal("runtime_admission_uninstall_legacy_retired", problems=[retired])
    runtime = root / "slots" / name / "runtime"
    state = root / "slots" / name / "state"
    if runtime.is_symlink() or not runtime.is_dir() or state.is_symlink() or not state.is_dir():
        raise LauncherRefusal("runtime_admission_uninstall_legacy_slot")
    manifest = json.loads((runtime / "release.json").read_text())
    commit = manifest.get("git_commit", "")
    if not any(commit.startswith(prefix) for prefix in legacy_commits):
        raise LauncherRefusal("runtime_admission_uninstall_legacy_commit")
    for relative, digest in manifest.get("files", {}).items():
        path = runtime / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise LauncherRefusal("runtime_admission_uninstall_legacy_integrity")
    verify_legacy_links(runtime, state)
    return runtime, state, commit


# [Service] directives legacy.conf sets itself; every other directive of
# release.conf.unmanaged is carried over verbatim (review P8).
_LEGACY_FIXED = frozenset({"Type", "NotifyAccess", "WatchdogSec", "RestartPreventExitStatus", "ExecStart",
                           "WorkingDirectory", "TimeoutStartSec", "TimeoutStopSec", "StandardOutput",
                           "StandardError", "Environment"})


def _legacy_carry(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Environment lines without their PYTHONPATH assignment, and every other directive by section."""
    environment: list[str] = []
    carried: dict[str, list[str]] = {}
    section = "[Service]"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        key = line.split("=", 1)[0].strip()
        if section == "[Service]" and key == "Environment":
            # Only the PYTHONPATH assignment goes: CONFIG_PATH or any other
            # variable on the same line stays (token by token).
            try:
                tokens = shlex.split(line.split("=", 1)[1])
            except ValueError:
                raise LauncherRefusal("runtime_admission_uninstall_release_conf") from None
            kept = [token for token in tokens if not token.startswith("PYTHONPATH=")]
            if kept:
                environment.append("Environment=" + " ".join(shlex.quote(token) for token in kept))
        elif section == "[Service]" and key in _LEGACY_FIXED:
            continue
        else:
            carried.setdefault(section, []).append(line)
    return environment, carried


def render_legacy_conf(record: dict, runtime: Path, state: Path, release_conf_unmanaged: Path) -> str:
    """legacy.conf: the pre-managed service with the legacy slot's explicit paths (v3 N8).

    Starts from release.conf.unmanaged (moved aside at install): only the
    PYTHONPATH assignment is replaced, by the legacy runtime named
    explicitly ('current' is never used), and WorkingDirectory by the
    legacy state. Directives outside the fixed list (UnsetEnvironment,
    Limit*, KillMode, ...) are carried over verbatim (review P8).
    """
    properties = record["properties"]
    argv = record.get("legacy_exec_start")
    if not argv:
        raise LauncherRefusal("runtime_admission_launcher_exec_start")
    environment: list[str] = []
    carried: dict[str, list[str]] = {}
    if release_conf_unmanaged.is_file():
        environment, carried = _legacy_carry(release_conf_unmanaged.read_text())
    lines = ["# Legacy (unmanaged) service rendered by stage_runtime uninstall-launcher (U3b-0d, L4).",
             "[Service]", "Type=simple", "NotifyAccess=none", "WatchdogSec=0", "RestartPreventExitStatus=",
             "ExecStart=", "ExecStart=" + shlex.join(argv), *environment,
             "Environment=PYTHONPATH=" + str(runtime), "WorkingDirectory=" + str(state)]
    for key, directive in (("TimeoutStartUSec", "TimeoutStartSec"), ("TimeoutStopUSec", "TimeoutStopSec")):
        seconds = parse_timespan(properties.get(key))
        if seconds is not None:
            lines.append(f"{directive}={seconds:g}s")
    for key in ("StandardOutput", "StandardError"):
        if properties.get(key):
            lines.append(f"{key}={properties[key]}")
    lines.extend(carried.pop("[Service]", []))
    for section, entries in carried.items():
        lines += [section, *entries]
    return "\n".join(lines) + "\n"


def carried_directives(release_conf_unmanaged: Path) -> list[str]:
    """Names of the release.conf.unmanaged directives legacy.conf carries over verbatim."""
    if not release_conf_unmanaged.is_file():
        return []
    _environment, carried = _legacy_carry(release_conf_unmanaged.read_text())
    return sorted({section + entry.split("=", 1)[0].strip() for section, entries in carried.items()
                   for entry in entries})


def uninstall_launcher(root: Path, *, unit_dir: Path, decision_id: str, phrase: str, legacy_slot: str,
                       manager: ServiceManager, legacy_commits: tuple[str, ...] = LEGACY_COMMITS,
                       wall=time.time) -> dict:
    """Leave management for a legacy slot (L4 = A): decision id, typed phrase, unit stopped.

    Moves managed.conf aside, renders legacy.conf with the legacy slot's
    explicit runtime and state, reloads and verifies. The only change under
    admission/ is the immutable uninstall record; 'current', the head, the
    bindings and the slots stay byte-identical. Erased data comes back with
    the legacy slot: the user decides, not the operator.
    """
    root = Path(root).resolve()
    unit_dir = Path(unit_dir)
    if phrase != UNINSTALL_PHRASE:
        raise LauncherRefusal("runtime_admission_uninstall_phrase")
    if type(decision_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,79}", decision_id) is None:
        raise LauncherRefusal("runtime_admission_uninstall_decision")
    _require_inactive(manager)
    with stage.AdmissionController(root, allow_open_head=True):
        record = _latest_install(root)
        runtime, state, commit = _verify_legacy_slot(root, legacy_slot, legacy_commits)
        managed = unit_dir / "managed.conf"
        managed_sha = _sha256(managed)
        if managed_sha is None:
            raise LauncherRefusal("runtime_admission_uninstall_not_managed")
        text = render_legacy_conf(record, runtime, state, unit_dir / "release.conf.unmanaged")
        carried = carried_directives(unit_dir / "release.conf.unmanaged")
        os.replace(managed, unit_dir / "managed.conf.disabled")
        legacy = unit_dir / "legacy.conf"
        _write_text_atomic(legacy, text)
        manager.daemon_reload()
        shown = manager.user_show(UNIT, RECORDED_PROPERTIES)
        problems = _legacy_mismatches(record, shown, runtime, state)
        stamp = _now_stamp(wall())
        stage._atomic_record(_records(root, "launcher", f"uninstall-{stamp}.json"), {
            "schema_version": 1, "decision_id": decision_id,
            "phrase_sha256": hashlib.sha256(phrase.encode()).hexdigest(),
            "head": stage.read_admission_head(root), "legacy_slot": legacy_slot, "legacy_commit": commit,
            "managed_conf_sha256": managed_sha, "legacy_conf_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "verified_properties_sha256": stage._digest(shown or {}), "mismatches": problems,
            "carried_directives": carried, "uninstalled_at": stamp}, immutable=True)
    return {"legacy_conf": str(legacy), "mismatches": problems, "record": f"uninstall-{stamp}.json",
            "carried_directives": carried}


def _legacy_mismatches(record: dict, shown: dict | None, runtime: Path, state: Path) -> list[str]:
    if shown is None:
        return ["properties_unavailable"]
    before = record["properties"]
    problems = []
    for key in ("Type", "NotifyAccess", "RestartPreventExitStatus", "StandardOutput", "StandardError"):
        if shown.get(key, "") != before.get(key, ""):
            problems.append(key)
    for key in ("WatchdogUSec", "TimeoutStartUSec", "TimeoutStopUSec"):
        if parse_timespan(shown.get(key)) != parse_timespan(before.get(key)):
            problems.append(key)
    if shown.get("WorkingDirectory") != str(state):
        problems.append("WorkingDirectory")
    try:
        if exec_start_argv(shown.get("ExecStart", "")) != record.get("legacy_exec_start"):
            problems.append("ExecStart")
    except LauncherRefusal:
        problems.append("ExecStart")
    entries = shlex.split(shown.get("Environment", ""))
    if "PYTHONPATH=" + str(runtime) not in entries:
        problems.append("PYTHONPATH")
    if _environment_digest(shown.get("Environment", "")) != before.get("Environment_sha256"):
        problems.append("Environment")
    return problems


# --------------------------------------------------------------------------
# CLI (dispatched from stage_runtime.main)


COMMANDS = ("serve", "adopt", "install-controller", "install-launcher", "uninstall-launcher", "stop-timeout-dropin",
            "console-logrotate", "completion-status")


def add_commands(commands) -> None:
    """Register the launcher subcommands on stage_runtime's parser."""
    serve = commands.add_parser("serve", help="Run the head slot under systemd (Type=notify, U3b-0d)")
    serve.add_argument("--stop-grace", type=float, required=True)
    serve.add_argument("--recover", choices=RECOVER_MODES, default="bounded")
    serve.add_argument("--unit-timeout-stop", type=float, default=0.0,
                       help="kza-voice TimeoutStopSec, for the reboot budget check")
    serve.add_argument("--probe-timeout", type=float, default=60.0)
    # Rehearsal-only (compiled launcher tests); production managed.conf never sets them.
    serve.add_argument("--child-command", help=argparse.SUPPRESS)
    serve.add_argument("--boot-id-file", help=argparse.SUPPRESS)
    adopt_parser = commands.add_parser("adopt", help="Operator recovery of an UNKNOWN or running head")
    adopt_parser.add_argument("--confirm", default="", help="sha256 of the head (printed by --dry-run)")
    adopt_parser.add_argument("--confirm-regressions", default="",
                              help="Comma-separated regressed families exactly as --dry-run reports them")
    adopt_parser.add_argument("--dry-run", action="store_true")
    adopt_parser.add_argument("--boot-id-file", help=argparse.SUPPRESS)
    controller = commands.add_parser("install-controller")
    controller.add_argument("archive", type=Path)
    launcher = commands.add_parser("install-launcher")
    launcher.add_argument("--unit-dir", type=Path, required=True)
    launcher.add_argument("--template", type=Path, required=True)
    launcher.add_argument("--python", required=True)
    launcher.add_argument("--stop-grace", type=float, required=True)
    launcher.add_argument("--timeout-start", type=float, default=DEFAULT_TIMEOUT_START,
                          help="TimeoutStartSec of the managed unit; it must cover a whole start including R-b "
                               f"without EXTEND (default {DEFAULT_TIMEOUT_START:g})")
    launcher.add_argument("--timeout-stop", type=float, required=True)
    launcher.add_argument("--watchdog-sec", type=float, default=0.0)
    launcher.add_argument("--console-log", required=True)
    launcher.add_argument("--recover", choices=RECOVER_MODES, default="bounded")
    launcher.add_argument("--logrotate-conf", default=None,
                          help="logrotate configuration that must rotate --console-log (default <root>/"
                               + LOGROTATE_NAME + ")")
    uninstall = commands.add_parser("uninstall-launcher")
    uninstall.add_argument("--unit-dir", type=Path, required=True)
    uninstall.add_argument("--user-decision", required=True)
    uninstall.add_argument("--phrase", required=True)
    uninstall.add_argument("--legacy-slot", required=True)
    dropin = commands.add_parser("stop-timeout-dropin", help="Print the sudo drop-in text (never writes it)")
    dropin.add_argument("--unit-timeout-stop", type=float, required=True)
    dropin.add_argument("--chain-extra", type=float, default=0.0)
    commands.add_parser("completion-status", help="Read-only counts of re-queued privacy work still open "
                                                  "(after W9: every count must be 0)")
    rotate = commands.add_parser("console-logrotate",
                                 help="Print the logrotate stanza for the console log (kza applies it)")
    rotate.add_argument("--console-log", required=True)


def run_command(args) -> int:
    """Execute one launcher subcommand; returns the process exit code."""
    if args.command == "serve":
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")
        options = ServeOptions(root=args.root, stop_grace=args.stop_grace, recover=args.recover,
                               unit_timeout_stop=args.unit_timeout_stop, probe_timeout=args.probe_timeout,
                               child_command=json.loads(args.child_command) if args.child_command else None,
                               boot_id_path=args.boot_id_file)
        return Serve(options).run()
    try:
        if args.command == "adopt":
            result = adopt(args.root, confirm=args.confirm, dry_run=args.dry_run, boot_id_path=args.boot_id_file,
                           regressions=[item for item in args.confirm_regressions.split(",") if item])
        elif args.command == "install-controller":
            result = install_controller(args.root, args.archive)
        elif args.command == "install-launcher":
            result = install_launcher(args.root, unit_dir=args.unit_dir, template=args.template.read_text(),
                                      manager=ServiceManager(), settings=LauncherSettings(
                                          python=args.python, stop_grace=args.stop_grace,
                                          timeout_start=args.timeout_start, timeout_stop=args.timeout_stop,
                                          watchdog_sec=args.watchdog_sec, console_log=args.console_log,
                                          recover=args.recover, logrotate_conf=args.logrotate_conf))
            if result["verification_mismatches"]:
                # managed.conf stays in place; the operator sees exactly what differs (P9).
                sys.stdout.write(json.dumps(result, sort_keys=True, default=str) + "\n")
                sys.stderr.write(json.dumps({"refused": "runtime_admission_launcher_verification",
                                             "problems": result["verification_mismatches"]}) + "\n")
                return int(ExitCode.REFUSED)
        elif args.command == "uninstall-launcher":
            result = uninstall_launcher(args.root, unit_dir=args.unit_dir, decision_id=args.user_decision,
                                        phrase=args.phrase, legacy_slot=args.legacy_slot, manager=ServiceManager())
        elif args.command == "console-logrotate":
            result = {"stanza": render_console_logrotate(args.console_log)}
        elif args.command == "completion-status":
            result = completion_status(args.root)
        else:
            result = render_stop_timeout_dropin(os.getuid(), args.unit_timeout_stop, args.chain_extra)
    except LauncherRefusal as refusal:
        refused = {"refused": refusal.code}
        if refusal.problems:
            refused["problems"] = refusal.problems
        sys.stderr.write(json.dumps(refused) + "\n")
        return int(refusal.exit_code)
    sys.stdout.write(json.dumps(result, sort_keys=True, default=str) + "\n")
    return 0
