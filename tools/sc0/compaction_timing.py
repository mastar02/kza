"""SC-0 (3): time today's 0c copy (compact plan + start) on a production-size synthetic state.

Builds the rehearsal artifacts from this checkout (tests/mocks/launcher_rehearsal),
installs the compiled controller and launcher against the fake systemctl shim
(tests/integration/test_compiled_managed_launcher.Deployment: no real units,
nothing under the production account), and closes H with one supervised run
whose child grows the state inside its admission run:

* events.db events_policy_v1: ``--events`` rows (T-A8 shape, no text deadline);
* ambient.db utterances: ``--utterances`` rows (probe-harness shape);
* privacy.db with one registered collector root (REGISTER_TARGET).

Defaults are DEPLOYMENT's production-size figure (161 MB adopted state:
500k events, 250k utterances, T-A8). Then it times ``compact plan`` and
``compact start`` (snapshot, K materialized by the worker, authorize,
activate) as the installed compiled controller, sampling the root's disk use
(bytes of regular files under it, and filesystem free space) every
``--sample`` seconds to find the peak. Everything is synthetic.

Usage (server, as yo; CPU only):
    python tools/sc0/compaction_timing.py --work /home/yo/projects/runs/sc0/compaction --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts import stage_runtime as stage  # noqa: E402
from tests.integration import test_compiled_managed_launcher as compiled  # noqa: E402
from tests.mocks import launcher_rehearsal as rehearsal  # noqa: E402
from tests.mocks.probe_cost_harness import _literal  # noqa: E402


def grow_snippet(events: int, utterances: int) -> str:
    """Child-side code (``state`` is the slot state): the production-size payload."""
    ambient_schema = _literal("src/ambient/store.py", assignment="_SCHEMA")
    events_schema = _literal("src/analytics/event_logger.py", table="events_policy_v1")
    return "\n".join([
        "import json, sqlite3",
        "from contextlib import closing",
        "now = 1_800_000_000.0",
        "text = 'texto sintético de una frase ambiental sin datos reales del hogar ' * 2",
        "data = state / 'data'",
        f"with closing(sqlite3.connect(data / 'ambient.db')) as db, db:",
        f"    db.executescript({ambient_schema!r})",
        "    db.execute('ALTER TABLE pending_audio_deletions ADD COLUMN conversation_id TEXT')",
        "    db.executemany('INSERT INTO utterances(room_id,t0,t1,text,created_at,capture_id,conversation_id,'",
        "                   'expires_at) VALUES(?,?,?,?,?,?,?,?)', (('room' + str(i % 6), now - i, now - i + 2,",
        "                   text[: 60 + i % 60], now - i, 'capture' + str(i), 'conv' + str(i // 20), now + 172800 - i)",
        f"                   for i in range({utterances})))",
        f"with closing(sqlite3.connect(data / 'events.db')) as db, db:",
        f"    db.executescript({events_schema!r})",
        "    db.executemany('INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,sources_json,'",
        "                   'metrics_json,expires_at) VALUES (?,?,?,?,?,?,?)', ((now - i * 10, 'state_change',",
        "                   'light.room' + str(i % 40), 'on', json.dumps([{'capture_id': 'capture-sc0',",
        "                   'conversation_id': 'conv' + str(i // 20), 'captured_at': 1_799_999_000.0,",
        f"                   'expires_at': 9e9}}]), '{{}}', now + 86400 * 30) for i in range({events})))",
    ])


def _tree_bytes(path: Path) -> int:
    total = 0
    for directory, _subdirs, files in os.walk(path):
        for name in files:
            try:
                info = os.lstat(os.path.join(directory, name))
            except FileNotFoundError:
                continue
            if not os.path.islink(os.path.join(directory, name)):
                total += info.st_blocks * 512
    return total


class Sampler:
    """Peak allocated bytes under the root and minimum free bytes, sampled on a thread."""

    def __init__(self, root: Path, interval: float):
        self.root, self.interval = root, interval
        self.peak, self.min_free, self.samples = 0, None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = _tree_bytes(self.root)
            free = shutil.disk_usage(self.root).free
            self.peak = max(self.peak, used)
            self.min_free = free if self.min_free is None else min(self.min_free, free)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _compact(deployment, *arguments):
    began = time.monotonic()
    result = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(deployment.root),
                             "--offline", "compact", *arguments], capture_output=True, text=True, timeout=3600,
                            env=deployment.env)
    seconds = time.monotonic() - began
    if result.returncode != 0:
        raise RuntimeError(f"compact {arguments[0]} exit {result.returncode}: "
                           f"{result.stdout[-2000:]} {result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1]), round(seconds, 2)


def _launcher(deployment, *arguments, env=None):
    began = time.monotonic()
    result = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(deployment.root),
                             "--offline", *arguments, "--boot-id-file", str(deployment.boot_id)],
                            capture_output=True, text=True, timeout=3600, env=env or deployment.env)
    seconds = time.monotonic() - began
    if result.returncode != 0:
        raise RuntimeError(f"{arguments[0]} exit {result.returncode}: {result.stdout[-2000:]} {result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1]), round(seconds, 2)


def measure_adopt_dry_run(deployment, work: Path, interval: float) -> dict:
    """A crash leaves H open; time ``adopt --dry-run`` and the peak of its TMPDIR shadow, then adopt."""
    unit = deployment.supervise("crash")
    unit.start()
    unit.wait_inactive(300)
    head = deployment.head()
    if head["status"] not in ("unknown", "running"):
        return {"skipped": "crash left head " + head["status"]}
    scratch = work / "dry-run-tmp"
    scratch.mkdir()
    env = {**deployment.env, "TMPDIR": str(scratch)}
    with Sampler(scratch, interval) as sampler:
        dry, seconds = _launcher(deployment, "adopt", "--dry-run", env=env)
    adopted, adopt_seconds = _launcher(deployment, "adopt", "--confirm", dry["head_sha256"], "--confirm-regressions",
                                       ",".join(dry["report"]["regressions"]))
    return {"dry_run_seconds": seconds, "shadow_peak_bytes": sampler.peak, "samples": sampler.samples,
            "dry_run_failures": dry["report"]["failures"], "adopt_seconds": adopt_seconds,
            "report_keys": sorted(dry["report"]), "report_sizes": dry["report"].get("sizes"),
            "head_after_adopt": deployment.head()["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--events", type=int, default=500_000)
    parser.add_argument("--utterances", type=int, default=250_000)
    parser.add_argument("--sample", type=float, default=0.5)
    parser.add_argument("--no-dry-run", action="store_true")
    args = parser.parse_args()
    work = Path(args.work).resolve()
    if work.exists():
        stage.make_writable_then_remove(work)
    work.mkdir(parents=True)
    result: dict = {"events": args.events, "utterances": args.utterances, "host": os.uname().nodename,
                    "python": sys.version.split()[0], "cpus": os.cpu_count()}
    began = time.monotonic()
    artifacts = rehearsal.build_artifacts(work / "artifacts")
    result["artifacts_seconds"] = round(time.monotonic() - began, 1)
    result["x_commit"] = artifacts["commits"]["x"]
    deployment = compiled.Deployment(artifacts, work / "deployment")
    try:
        began = time.monotonic()
        unit = deployment.supervise("clean", execute=compiled.REGISTER_TARGET + "\n"
                                    + grow_snippet(args.events, args.utterances))
        unit.start()
        assert unit.wait_for(lambda u: bool(u.events("ready")), 1800), deployment.console_text()[-3000:]
        assert unit.stop() == 0, deployment.console_text()[-3000:]
        assert deployment.head()["status"] == "closed"
        result["grow_and_close_h_seconds"] = round(time.monotonic() - began, 1)
        state = deployment.state
        result["h_state_bytes"] = sum(p.lstat().st_size for p in state.rglob("*") if p.is_file()
                                      and not p.is_symlink())
        result["h_state_allocated_bytes"] = _tree_bytes(state)
        result["h_databases"] = {p.name: p.stat().st_size for p in sorted((state / "data").glob("*.db"))}
        if not args.no_dry_run:
            result["adopt_dry_run"] = measure_adopt_dry_run(deployment, work, args.sample)
            if deployment.head()["status"] != "closed":
                raise RuntimeError("head not closed after adopt")
        result["root_allocated_before"] = _tree_bytes(deployment.root)
        archive = str(artifacts["archives"]["x"])
        with Sampler(deployment.root, args.sample) as sampler:
            planned, result["plan_seconds"] = _compact(deployment, "plan", "k", "--archive", archive)
            started, result["start_seconds"] = _compact(
                deployment, "start", "k", "--archive", archive, "--plan-sha256", planned["plan_sha256"],
                "--acknowledge-untouched", planned["untouched_sha256"], "--destroy-rollback-states")
        result["copy_seconds"] = round(result["plan_seconds"] + result["start_seconds"], 2)
        result["plan_view"] = {key: planned.get(key) for key in ("owners", "members", "capacity", "coverage")}
        result["start"] = {key: started.get(key) for key in ("phase", "materialized")}
        result["root_allocated_peak"] = sampler.peak
        result["root_peak_delta_bytes"] = sampler.peak - result["root_allocated_before"]
        result["root_allocated_after"] = _tree_bytes(deployment.root)
        result["fs_min_free"] = sampler.min_free
        result["samples"] = sampler.samples
        result["peak_over_h_state"] = round(sampler.peak / max(result["h_state_allocated_bytes"], 1), 2)
        log = stage.read_compaction_log(deployment.root, started["compaction_id"])
        result["log_events"] = [(event["kind"], event.get("at")) for event in log["events"]]
    finally:
        deployment.close()
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
