"""Real user systemd for the U3b-0e4 Linux rehearsals (test-only units, guarded systemctl).

The rehearsals run under the developer account's own user manager
(``systemctl --user``), never another account's, never sudo. Every unit they
create is named ``kza-rehearsal-<token>-<role>.<type>`` and lives in the
runtime unit directory ``$XDG_RUNTIME_DIR/systemd/user`` (a search-path
directory of every user manager), so nothing is written outside
``/run/user/<uid>``; enablement is ``--runtime`` for the same reason.

The product code talks to systemd through ``systemctl`` as it does in
production (``ServiceManager``: an executable path in process, ``PATH`` for
the compiled CLIs). Both get the guard written by :meth:`UserSystemd.shim`:

- the production names (kza-voice, kza-llm-fast, kza-code-index, kza-llm-ik)
  are mapped to their rehearsal stand-ins on the way in and back on the way
  out, and the rehearsal boot target reads as ``default.target``;
- after mapping, any argument still naming a production unit is refused
  (exit 97) before systemctl runs, as is any state change on a unit outside
  the rehearsal prefix and any system-manager call other than a read of
  ``user@<uid>.service`` properties or ``is-system-running``;
- ``enable``/``disable`` get ``--runtime`` (the unit file state then reads
  ``enabled-runtime``, shown to the product as ``enabled``);
- ``list-unit-files``/``list-units`` hide another concurrent rehearsal's
  units (its stand-in llama-server would otherwise be an unlisted unit
  reaching the models for this one, which the product rightly refuses);
- every call is logged (argv before and after mapping, exit code).

:meth:`UserSystemd.cleanup` stops, disables and removes every rehearsal
unit and drop-in, runs ``daemon-reload`` and returns the ``list-unit-files``
listings taken before the first unit was written and after the cleanup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

PRODUCTION_NAMES = ("kza-voice", "kza-llm-fast", "kza-code-index", "kza-llm-ik")
ROLES = {"kza-voice.service": "voice", "kza-llm-fast.service": "llm-fast",
         "kza-code-index.service": "code-index", "kza-llm-ik.service": "llm-ik"}
REFUSED = 97


def unavailable() -> str | None:
    """Why the rehearsals cannot run here (not Linux with a running user manager), or None."""
    if not sys.platform.startswith("linux"):
        return "U3b-0e4 rehearsals need Linux with a user systemd manager"
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime or not os.path.isdir(runtime) or shutil.which("systemctl") is None:
        return "U3b-0e4 rehearsals need a user systemd manager (XDG_RUNTIME_DIR, systemctl)"
    try:
        state = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True,
                               timeout=10, stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        state = ""
    if state not in ("running", "degraded"):
        return "U3b-0e4 rehearsals need a running user systemd manager"
    return None


_SHIM = r'''#!{python} -I
import json, os, re, subprocess, sys
CONFIG = json.loads({config!r})
original = sys.argv[1:]
mapped = [CONFIG["to_rehearsal"].get(token, token) for token in original]
words, skip = [], False
for token in mapped:
    if skip:
        skip = False
    elif token in ("-p", "--property", "-t", "--type", "--state", "-o", "--output"):
        skip = True
    elif not token.startswith("-"):
        words.append(token)
verb = words[0] if words else ""


def log(code, why=None):
    entry = {{"argv": original, "mapped": mapped, "exit": code}}
    if why:
        entry["refused"] = why
    with open(CONFIG["log"], "a") as stream:
        stream.write(json.dumps(entry) + "\n")


def refuse(why):
    log(CONFIG["refused"], why)
    sys.stderr.write("rehearsal guard: " + why + "\n")
    raise SystemExit(CONFIG["refused"])


for token in mapped:
    if any(name in token for name in CONFIG["forbidden"]):
        refuse("production unit name")
if "--user" not in mapped:
    if verb == "is-system-running":
        pass
    elif verb == "show" and words[1:] and all(re.fullmatch(r"user@[0-9]+\.service", w) for w in words[1:]):
        pass
    else:
        refuse("system manager call")
elif verb in ("start", "stop", "restart", "enable", "disable", "reset-failed", "kill"):
    units = words[1:]
    if not units or any(not unit.startswith(CONFIG["prefix"]) for unit in units):
        refuse("state change outside the rehearsal prefix")
    if verb in ("enable", "disable") and "--runtime" not in mapped:
        mapped.insert(mapped.index(verb), "--runtime")
elif verb == "daemon-reload":
    pass
elif verb not in ("show", "list-unit-files", "list-units", "is-enabled", "is-active", "is-failed", "cat"):
    refuse("verb not allowed")
completed = subprocess.run([CONFIG["real"], *mapped], capture_output=True, text=True, stdin=subprocess.DEVNULL)
output = completed.stdout
if verb in ("list-unit-files", "list-units"):  # a concurrent rehearsal's units are not this host's units
    output = "".join(line for line in output.splitlines(keepends=True)
                     if "kza-rehearsal-" not in line or CONFIG["prefix"] in line)
for rehearsal, product in CONFIG["to_product"]:
    output = re.sub(r"(?<![\w.@/-])" + re.escape(rehearsal) + r"(?![\w.@/-])", product, output)
output = re.sub(r"(?<![\w-])enabled-runtime(?![\w-])", "enabled", output)
sys.stdout.write(output)
sys.stderr.write(completed.stderr)
log(completed.returncode)
raise SystemExit(completed.returncode)
'''


class UserSystemd:
    """Rehearsal units in this account's user manager, the guarded shim and the cleanup proof."""

    def __init__(self, workdir: Path) -> None:
        self.token = secrets.token_hex(3)
        self.prefix = f"kza-rehearsal-{self.token}"
        self.real = shutil.which("systemctl")
        self.runtime = Path(os.environ["XDG_RUNTIME_DIR"])
        self.unit_dir = self.runtime / "systemd" / "user"
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log = self.workdir / "systemctl-calls.jsonl"
        self.log.touch()
        self.shim_dir = self.workdir / "bin"
        self.before = self.list_unit_files()
        self.created_dir = not self.unit_dir.exists()
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        self.created_wants: list[Path] = []
        self.boot = self.name("boot", "target")
        self.write_shim()
        self.cleaned: dict | None = None

    # --- names and the guard --------------------------------------------------------------------

    def name(self, role: str, kind: str = "service") -> str:
        return f"{self.prefix}-{role}.{kind}"

    def product_map(self) -> dict[str, str]:
        """Production name -> rehearsal stand-in."""
        return {product: self.name(role) for product, role in ROLES.items()}

    def write_shim(self) -> Path:
        to_rehearsal = self.product_map()
        to_product = sorted(((rehearsal, product) for product, rehearsal in to_rehearsal.items()),
                            key=lambda item: -len(item[0]))
        to_product.append((self.boot, "default.target"))
        config = json.dumps({"real": self.real, "to_rehearsal": to_rehearsal, "to_product": to_product,
                             "prefix": self.prefix + "-", "log": os.fspath(self.log), "refused": REFUSED,
                             "forbidden": list(PRODUCTION_NAMES)})
        self.shim_dir.mkdir(exist_ok=True)
        shim = self.shim_dir / "systemctl"
        shim.write_text(_SHIM.format(python=sys.executable, config=config))
        shim.chmod(0o755)
        self.shim = shim
        return shim

    def environment(self, extra: dict | None = None) -> dict:
        """A CLI environment whose ``systemctl`` is the guard."""
        keep = ("HOME", "LANG", "LC_ALL", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
        environment = {key: os.environ[key] for key in keep if key in os.environ}
        environment["PATH"] = os.fspath(self.shim_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")
        environment.update(extra or {})
        return environment

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def guard_breaches(self) -> list[dict]:
        """Calls whose mapped argv named a production unit and still reached systemctl (must be empty)."""
        return [call for call in self.calls() if call.get("refused") is None
                and any(name in token for token in call["mapped"] for name in PRODUCTION_NAMES)]

    # --- direct systemctl (test side; through the guard) ------------------------------------------

    def systemctl(self, *arguments: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return subprocess.run([os.fspath(self.shim), "--user", *arguments], capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL, env=self.environment())

    def raw_show(self, unit: str, *properties: str) -> dict[str, str]:
        """Properties of a rehearsal unit by its own name (read-only, real systemctl)."""
        assert unit.startswith(self.prefix), unit
        arguments = [self.real, "--user", "show", unit]
        for name in properties:
            arguments += ["-p", name]
        output = subprocess.run(arguments, capture_output=True, text=True, timeout=30,
                                stdin=subprocess.DEVNULL).stdout
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
        return {name: values.get(name, "") for name in properties}

    def list_unit_files(self) -> list[str]:
        output = subprocess.run([shutil.which("systemctl"), "--user", "list-unit-files", "--no-legend", "--plain",
                                 "--no-pager"], capture_output=True, text=True, timeout=60, check=True,
                                stdin=subprocess.DEVNULL).stdout
        return sorted(" ".join(line.split()) for line in output.splitlines() if line.strip())

    # --- units ----------------------------------------------------------------------------------

    def write_unit(self, name: str, text: str) -> Path:
        assert name.startswith(self.prefix + "-"), name
        path = self.unit_dir / name
        path.write_text(text)
        return path

    def dropin_dir(self, name: str) -> Path:
        assert name.startswith(self.prefix + "-"), name
        directory = self.unit_dir / (name + ".d")
        directory.mkdir(exist_ok=True)
        return directory

    def write_dropin(self, name: str, conf: str, text: str) -> Path:
        path = self.dropin_dir(name) / conf
        path.write_text(text)
        return path

    def reload(self) -> None:
        completed = self.systemctl("daemon-reload")
        assert completed.returncode == 0, completed.stderr

    def wait_for(self, unit: str, predicate, timeout: float, *properties: str) -> dict:
        names = properties or ("ActiveState", "SubState", "Result", "ExecMainStatus", "StatusText")
        deadline = time.monotonic() + timeout
        while True:
            shown = self.raw_show(unit, *names)
            if predicate(shown) or time.monotonic() >= deadline:
                return shown
            time.sleep(0.1)

    def wait_inactive(self, unit: str, timeout: float) -> dict:
        return self.wait_for(unit, lambda shown: shown["ActiveState"] in ("inactive", "failed"), timeout)

    # --- cleanup --------------------------------------------------------------------------------

    def _ours(self) -> list[Path]:
        found = sorted(self.unit_dir.glob(self.prefix + "-*"))
        for wants in self.unit_dir.glob("*.wants"):
            found += sorted(wants.glob(self.prefix + "-*"))
        return found

    def cleanup(self) -> dict:
        """Stop, disable and remove every rehearsal unit; daemon-reload; the before/after listings."""
        if self.cleaned is not None:
            return self.cleaned
        real = self.real
        units = sorted({path.name for path in self.unit_dir.glob(self.prefix + "-*")
                        if not path.name.endswith((".d", ".wants"))})
        for unit in units:
            subprocess.run([real, "--user", "stop", unit], capture_output=True, timeout=180,
                           stdin=subprocess.DEVNULL)
            subprocess.run([real, "--user", "disable", "--runtime", unit], capture_output=True, timeout=60,
                           stdin=subprocess.DEVNULL)
        for path in self._ours():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        for unit in units:
            subprocess.run([real, "--user", "reset-failed", unit], capture_output=True, timeout=60,
                           stdin=subprocess.DEVNULL)
        reload = subprocess.run([real, "--user", "daemon-reload"], capture_output=True, text=True, timeout=120,
                                stdin=subprocess.DEVNULL)
        for wants in self.unit_dir.glob(self.prefix + "-*"):
            shutil.rmtree(wants, ignore_errors=True)
        if self.created_dir:
            for _attempt in range(20):
                try:
                    self.unit_dir.rmdir()
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.25)
        after = self.list_unit_files()
        leftovers = [path.name for path in self._ours()] if self.unit_dir.exists() else []
        self.cleaned = {"token": self.token, "units_created": units, "daemon_reload_exit": reload.returncode,
                        "before": self.before, "after": after, "after_equals_before": after == self.before,
                        "rehearsal_units_after": [line for line in after if self.prefix in line],
                        "leftover_paths": leftovers, "unit_dir_removed": not self.unit_dir.exists(),
                        "calls": len(self.calls()), "guard_refusals": [call for call in self.calls()
                                                                      if call.get("refused")],
                        "guard_breaches": self.guard_breaches()}
        return self.cleaned


def evidence_directory() -> Path | None:
    """KZA_0E4_EVIDENCE: where the rehearsals write their measurements (optional)."""
    value = os.environ.get("KZA_0E4_EVIDENCE")
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def record(name: str, value) -> None:
    """Write one JSON measurement to the evidence directory (no-op without KZA_0E4_EVIDENCE)."""
    directory = evidence_directory()
    if directory is not None:
        (directory / (name + ".json")).write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
