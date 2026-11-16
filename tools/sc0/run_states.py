"""SC-0 (2): protected-projection diff of a no-traffic managed boot, per residue state.

For each state: prepare a fresh slot at one fixed staging path from a source
whose data/ is the initialized template (a sealed state of a first managed
boot on this archive) plus the state's residue rows; take the protected
projection (stage_runtime.capture_state_authority) with the head closed;
run the compiled child (tools/sc0/sc0_child.py) under the real
AdmissionController with no traffic; send SIGTERM ``--settle`` seconds after
the child reports steady; after the run, check whether a receipt was sealed
and take the projection again. The per-table diff is joined with the child's
SQL write trace (statement, phase, src frames). Skip list: none (normal
mode; SC-4a's verify mode does not exist yet). A residue injector may
return ``config`` (dotted settings overrides for NEW) and ``settle`` (a
longer settle, for loops whose first write comes after 15 s).

Usage:
    python tools/sc0/run_states.py --work DIR --out results.json [--states a,b] [--settle 15]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from scripts import stage_runtime as stage  # noqa: E402
from scripts.build_runtime import build_runtime  # noqa: E402
from tests.mocks import compiled_restore as compiled  # noqa: E402
from tests.mocks.managed_slot_graph import canonical_template  # noqa: E402
from tests.mocks.restore_application_graph import configuration as restore_configuration  # noqa: E402
from tools.sc0 import residue  # noqa: E402
from tools.sc0.write_inventory import projected_tables  # noqa: E402

CHILD = REPO / "tools/sc0/sc0_child.py"
BASE_REVISION = "0bd9f98"
AMBIENT = ["real"]
CHILD_FLAGS: list[str] = []


def bootstrap_populated_state_patch() -> None:
    """Harness only (controller process): let a fresh slot over a populated state bootstrap.

    The real activate refuses a populated state without a head
    (runtime_admission_manual_legacy_unsealed): production reaches such a
    state only through adoption or a compaction K slot. SC-0 needs many
    residue states under one real admission run each, so in this driver
    process (never in the child) the bootstrap guard is opened:
    _has_authority answers False and the manual-components capture carries
    no collector stores. Ticket, child, close path, receipt and seal run
    unchanged; the measured projection is the unpatched
    capture_state_authority.
    """
    real_capture = stage._manual_capture
    stage._has_authority = lambda binding: False

    def capture(root, binding):
        value = real_capture(root, binding)
        return {**value, "stores": {}}

    stage._manual_capture = capture


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_archive(work: Path) -> Path:
    archive = work / "runtime-0bd9f98.tar.gz"
    if archive.exists():
        return archive
    clone = work / "archive-clone"
    if clone.exists():
        shutil.rmtree(clone)
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    manifest = build_runtime(clone, archive, revision=BASE_REVISION)
    (work / "runtime-manifest.json").write_text(json.dumps(
        {"git_commit": manifest["git_commit"], "files": len(manifest["files"])}))
    return archive


class Run:
    """One slot at the fixed staging path; one managed run of the child."""

    def __init__(self, archive: Path, work: Path, data_source: Path | None, label: str,
                 overrides: dict | None = None):
        self.work, self.label = work, label
        self.port = _free_port()
        self.staging = work / "staging"
        if self.staging.exists():
            stage.make_writable_then_remove(self.staging)
        self.staging.mkdir()
        self.staging = self.staging.resolve()
        self.state = self.staging / "slots/s/state"
        self.root = self.state / "data/graph"
        source = work / "offline"
        if source.exists():
            shutil.rmtree(source)
        (source / "config").mkdir(parents=True)
        if data_source is None:
            (source / "data").mkdir()
        else:
            shutil.copytree(data_source, source / "data", symlinks=True)

        def configuration(root, home, model):
            config = restore_configuration(Path(root), home, model)
            config["dashboard"].update(enabled=True, host="127.0.0.1", port=self.port)
            return config

        raw = canonical_template(self.root, configuration)
        for dotted, value in (overrides or {}).items():
            # A residue state may run NEW with a different setting than the
            # one its rows were written under (R4: shorter retention).
            *parents, leaf = dotted.split(".")
            owner = raw
            for key in parents:
                owner = owner.setdefault(key, {})
            owner[leaf] = value
        (source / "config/settings.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        stage.prepare_slot(archive, self.staging, "s", source)
        assert stage.activate_slot(self.staging, "s") is None
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.logs = work / "logs" / label
        if self.logs.exists():
            shutil.rmtree(self.logs)
        self.logs.mkdir(parents=True)
        self.phase_file = self.logs / "phases.txt"
        self.phase_file.touch()
        self.trace = self.logs / "sql-trace.jsonl"
        self.fifo = self.logs / "notify.fifo"
        os.mkfifo(self.fifo, 0o600)
        self.notify: list[str] = []

    def phases(self) -> dict:
        found = {}
        for line in self.phase_file.read_text().splitlines():
            key, _, value = line.partition(" ")
            found.setdefault(key, value)
        return found

    def run(self, settle: float, timeout: float) -> dict:
        command = [sys.executable, "-I", "-B", str(CHILD), "--repo", str(REPO),
                   "--phase-file", str(self.phase_file), "--notify-fifo", str(self.fifo),
                   "--graph-root", str(self.root), "--dashboard-port", str(self.port),
                   "--trace-file", str(self.trace), "--ambient", AMBIENT[0], *CHILD_FLAGS]
        reader = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        outcome: dict = {"settle_seconds": settle, "timeout_seconds": timeout}
        stdout, stderr = self.logs / "child-stdout.log", self.logs / "child-stderr.log"

        def execute():
            try:
                with stdout.open("w") as out, stderr.open("w") as err:
                    completed = stage.run_slot(self.staging, "s", command, timeout=timeout,
                                               env=compiled._environment(), stdout=out, stderr=err)
                outcome["returncode"] = completed.returncode
            except BaseException as error:  # recorded, not raised: a missing receipt is a result
                outcome["error"] = f"{type(error).__name__}: {error}"[:500]
            outcome["ended"] = time.monotonic()

        started = time.monotonic()
        runner = threading.Thread(target=execute)
        runner.start()
        buffer, steady_at, signalled = b"", None, None
        try:
            while runner.is_alive():
                try:
                    buffer += os.read(reader, 4096)
                except BlockingIOError:
                    pass
                *lines, buffer = buffer.split(b"\n")
                self.notify.extend(line.decode() for line in lines)
                phases = self.phases()
                if steady_at is None and "steady" in phases and "READY=1" in self.notify:
                    steady_at = time.monotonic()
                if signalled is None and steady_at is not None and time.monotonic() - steady_at >= settle:
                    signalled = time.monotonic()
                    os.kill(int(phases["pid"]), signal.SIGTERM)
                time.sleep(0.02)
            runner.join()
            try:
                buffer += os.read(reader, 4096)
            except BlockingIOError:
                pass
            self.notify.extend(line.decode() for line in buffer.split(b"\n") if line)
        finally:
            os.close(reader)
        outcome["notify"] = self.notify
        outcome["phases"] = self.phases()
        outcome["wall_seconds"] = round(outcome["ended"] - started, 2)
        outcome["stop_seconds"] = None if signalled is None else round(outcome["ended"] - signalled, 2)
        head = stage.read_admission_head(self.staging)
        outcome["head_status"] = None if head is None else head["status"]
        receipts = self.staging / "admission/receipts"
        outcome["receipts"] = sorted(p.name for p in receipts.glob("*.json")) if receipts.exists() else []
        outcome["receipt_sealed"] = outcome["head_status"] == "closed" and len(outcome["receipts"]) == 1
        log = stdout.read_text() + stderr.read_text()
        outcome["log_tail"] = [line for line in log.splitlines()
                               if any(k in line for k in ("[Stop]", "Error", "error", "Traceback",
                                                          "Warning", "WARNING", "pending", "prune"))][-60:]
        return outcome


def _rows(entry) -> set | None:
    return set(entry["row_hashes"]) if isinstance(entry, dict) and "row_hashes" in entry else None


def diff_projection(before: dict, after: dict) -> list[dict]:
    changes = []
    tb, ta = before["projection"]["tables"], after["projection"]["tables"]
    for database in sorted(set(tb) | set(ta)):
        b, a = tb.get(database, {}), ta.get(database, {})
        for table in sorted(set(b) | set(a)):
            if b.get(table) == a.get(table):
                continue
            rb, ra = _rows(b.get(table)), _rows(a.get(table))
            if rb is not None and ra is not None:
                changes.append({"db": database, "table": table, "removed": len(rb - ra),
                                "added": len(ra - rb), "before": len(rb), "after": len(ra)})
            else:
                changes.append({"db": database, "table": table, "changed": True,
                                "before": b.get(table), "after": a.get(table)})
    for kind in ("accounts", "configuration"):
        xb, xa = before["projection"][kind], after["projection"][kind]
        for name in sorted(set(xb) | set(xa)):
            if xb.get(name) != xa.get(name):
                changes.append({"kind": kind, "file": name, "changed": True})
    return changes


def _writers(trace: Path) -> list[dict]:
    if not trace.exists():
        return []
    summary: dict = {}
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        site = record["frames"][0] if record["frames"] else "?"
        key = (record["db"], record["table"], record["verb"], record["phase"], site)
        item = summary.setdefault(key, {"db": key[0], "table": key[1], "verb": key[2], "phase": key[3],
                                        "site": site, "stack": record["frames"], "count": 0,
                                        "first_t": record["t"], "thread": record["thread"],
                                        "caller": record.get("caller")})
        item["count"] += 1
    return sorted(summary.values(), key=lambda item: item["first_t"])


def measure(archive: Path, work: Path, label: str, data_source: Path | None, *, settle: float,
            timeout: float, inject=None) -> dict:
    result: dict = {"state": label}
    CHILD_FLAGS[:] = []
    if inject is not None:
        injected = work / "injected-data"
        if injected.exists():
            shutil.rmtree(injected)
        shutil.copytree(data_source, injected, symlinks=True)
        result["residue"] = inject(injected, work.resolve() / "staging/slots/s/state/data/graph")
        flags = result["residue"].get("child_flags", [])
        if flags:
            CHILD_FLAGS[:] = flags
            settle = max(settle, 45.0)
        settle = max(settle, result["residue"].get("settle", 0.0))
        data_source = injected
    else:
        CHILD_FLAGS[:] = []
    overrides = (result.get("residue") or {}).get("config", {})
    run = Run(archive, work, data_source, label, overrides)
    before = stage.capture_state_authority(run.state)
    result["outcome"] = run.run(settle, timeout)
    try:
        after = stage.capture_state_authority(run.state)
        result["diff"] = diff_projection(before, after)
    except Exception as error:  # an unreadable state is a result too
        result["diff_error"] = f"{type(error).__name__}: {error}"
    projected = projected_tables(REPO)
    writers = _writers(run.trace)
    # Every table capture_state_authority can hash (full, partial, schema
    # marker), each with its frames; other_writes is the rest.
    result["projected_writes"] = [{**w, "projection": projected[w["table"]]}
                                  for w in writers if w["table"] in projected]
    result["other_writes"] = sorted({(w["db"], w["table"], w["verb"], w["phase"]) for w in writers
                                     if w["table"] not in projected})
    (run.logs / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--states", default="")
    parser.add_argument("--settle", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--ambient", choices=("idle", "real"), default="real")
    args = parser.parse_args()
    AMBIENT[0] = args.ambient
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    archive = build_archive(work)
    template = work / "template-data"
    bootstrap_populated_state_patch()
    results = []
    if not template.exists():
        first = measure(archive, work, "first_boot_empty", None, settle=args.settle, timeout=args.timeout)
        results.append(first)
        if not first["outcome"].get("receipt_sealed"):
            raise SystemExit("first boot did not seal; no template")
        shutil.copytree(work / "staging/slots/s/state/data", template, symlinks=True)
    wanted = [name for name in args.states.split(",") if name] or list(residue.STATES)
    for name in wanted:
        print(f"[sc0] state {name}", flush=True)
        try:
            results.append(measure(archive, work, name, template, settle=args.settle,
                                   timeout=args.timeout, inject=residue.STATES[name]))
        except Exception:
            results.append({"state": name, "harness_error": traceback.format_exc()[-3000:]})
        out = Path(args.out)
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        os.replace(temporary, out)


if __name__ == "__main__":
    main()
