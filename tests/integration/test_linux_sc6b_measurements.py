"""SC-6b: the final downtime and disk numbers, compiled and under this account's real user systemd.

sc-release-design v4 implementation_units SC-6b ("final downtime and disk
numbers, downtime checked against the SC-D2 A budget") and downtime_and_disk.
Not a functional rehearsal: it measures, so it runs **alone, in a solo
window** (no other pytest of any account, no other compiled rehearsal, the
server otherwise idle) and only when asked: it is skipped unless
``KZA_SC6B_MEASURE=1`` (and skipped like the rehearsals off Linux, without a
user manager, or as root or kza). Suggested command (server, as yo, own clone)::

    KZA_SC6B_MEASURE=1 KZA_0E4_EVIDENCE=<evidence dir> nohup taskset -c 24-47 \\
        .venv/bin/python -m pytest -p no:cacheprovider -v \\
        --basetemp=<scratch on the filesystem being measured> \\
        tests/integration/test_linux_sc6b_measurements.py > <evidence dir>/pytest.log 2>&1

One world, built like test_linux_sc6b_release_rehearsal's (R1's stand-in X4
root, the real stand-in unit with the production managed.conf), whose H is
grown to the SC-0 production-size synthetic state by its own first run
(tools/sc0/compaction_timing.grow_snippet: ``KZA_SC6B_EVENTS`` events,
default 500 000, and ``KZA_SC6B_UTTERANCES`` utterances, default 250 000,
plus R1's ledger and one tombstone). Then three paths, one after the other,
each starting with the unit serving (enabled, active) and paso 0 done with
the service up:

- A, the first transition with T0 (X4 -> X5): paso 1 stop and disable,
  extract and check NEW, paso 2's decision (T0), T0, H's boot and stop under
  the new controller, plan, start, verify, V, status, retire, K' to READY;
- B, a later transition without T0 (X5 -> X5, the equal triple), the same
  without T0 and its H check;
- C, the lossless abort: as B, but V removes H's tombstone, retire refuses
  (W loss-only) and ``abort --to-h`` returns to H, which is started to READY.

D is measured end to end, from the ``systemctl stop`` of the serving unit to
READY (ActiveState active) of the new head, or of H after the abort, and
checked against 600 s (SC-D2 A); every part's time is kept (the plan, start,
T0, boots, stops, V, retire, abort). The boots here are the rehearsal child's,
not the real application with its models: W8's real boot-to-READY B is the
operator's, so each path also records D with every boot replaced by
``KZA_SC6B_W8_B_SECONDS`` when it is given, and the default N per
downtime_and_disk.formula_check (N = min(60, 600 - T_copy - boots x B -
stops), refused below 10) from this run's T_copy. V's N is
``KZA_SC6B_SMOKE_SECONDS`` (default 10, the runbook's N with T0).

Disk: the bytes allocated under the root (tree walk every 0.5 s) and the
free space of the root's filesystem and of SQLite's temporary directory's
(statvfs every 10 ms: VACUUM's unlinked temporary database), from paso 1 to
READY; the retention after activation (after start) and after retire or
abort; each against plan's P9 need (``capacity``). SQLite's temporary
directory is ``KZA_SC6B_SQLITE_TMPDIR`` (default: a directory beside the root,
on its filesystem, as SC-0b measured) and is passed to every controller
command as TMPDIR and SQLITE_TMPDIR. Results: ``sc6b-measurements.json``
(rewritten after each path with ``complete`` false, so a later failure keeps
the finished paths, and once more at the end) and the usual per-step logs
(``sc6b-measure.json``, ``logs/measure/``) in KZA_0E4_EVIDENCE. Nothing
here touches the kza account, /home/kza or a production unit.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from tests.integration.test_linux_sc6b_release_rehearsal import (  # noqa: F401 (fixtures)
    DELETE_TOMBSTONES, H_TOMBSTONE, STATUS, World, builds, cleanup_evidence, record, systemd)
from tests.mocks import compaction_world as cw
from tests.mocks import real_systemd
from tools.sc0.compaction_timing import FreeSampler, Sampler, _tree_bytes, grow_snippet

pytestmark = [
    pytest.mark.skipif(real_systemd.unavailable() is not None, reason=real_systemd.unavailable() or "available"),
    pytest.mark.skipif(os.environ.get("KZA_SC6B_MEASURE") != "1",
                       reason="SC-6b final measurements: run alone in a solo window with KZA_SC6B_MEASURE=1"),
]

BUDGET_SECONDS = 600.0  # SC-D2 A: at most 10 minutes per transition, every path
EVENTS = int(os.environ.get("KZA_SC6B_EVENTS", "500000"))
UTTERANCES = int(os.environ.get("KZA_SC6B_UTTERANCES", "250000"))
SMOKE_SECONDS = int(os.environ.get("KZA_SC6B_SMOKE_SECONDS", "10"))
W8_B = float(os.environ["KZA_SC6B_W8_B_SECONDS"]) if os.environ.get("KZA_SC6B_W8_B_SECONDS") else None
TREE_SAMPLE, FREE_SAMPLE = 0.5, 0.01


class MeasuredWorld(World):
    """The rehearsal world whose controller commands name SQLite's temporary directory."""

    sqlite_tmp: Path | None = None

    def run(self, name, argv, *, expect=0, env=None, timeout=1200, stdin=""):
        if env is None and self.sqlite_tmp is not None:
            env = self.systemd.environment({"TMPDIR": str(self.sqlite_tmp), "SQLITE_TMPDIR": str(self.sqlite_tmp)})
        return super().run(name, argv, expect=expect, env=env, timeout=timeout, stdin=stdin)


class Disk:
    """Peak bytes under the root, minimum free space of the root's and SQLite's temp filesystems, marks."""

    def __init__(self, world: MeasuredWorld) -> None:
        self.root = world.root
        self.sqlite_tmp = world.sqlite_tmp
        self.separate = os.stat(self.sqlite_tmp).st_dev != os.stat(self.root).st_dev
        self.marks: dict[str, int] = {}

    def __enter__(self) -> "Disk":
        self.before = _tree_bytes(self.root)
        self.free_before = shutil.disk_usage(self.root).free
        self.tmp_free_before = shutil.disk_usage(self.sqlite_tmp).free
        self.tree = Sampler(self.root, TREE_SAMPLE).__enter__()
        self.free = FreeSampler(self.root, FREE_SAMPLE).__enter__()
        self.tmp = FreeSampler(self.sqlite_tmp, FREE_SAMPLE).__enter__() if self.separate else None
        return self

    def mark(self, name: str) -> None:
        self.marks[name] = _tree_bytes(self.root)

    def __exit__(self, *exc) -> None:
        for sampler in (self.tree, self.free, self.tmp):
            if sampler is not None:
                sampler.__exit__(None, None, None)

    def result(self, capacity: dict) -> dict:
        root_growth = self.free_before - min(self.free.min_free, self.tree.min_free)
        need, temp_need = capacity.get("need_bytes"), capacity.get("temp_need_bytes")
        found = {"root_allocated_before": self.before, "root_allocated_peak": self.tree.peak,
                 "root_tree_peak_growth": self.tree.peak - self.before, "root_fs_peak_growth": root_growth,
                 "root_fs_samples": self.free.samples, "tree_samples": self.tree.samples,
                 "retention": {name: value - self.before for name, value in self.marks.items()},
                 "sqlite_tmp_on_root_fs": not self.separate, "p9_capacity": capacity,
                 "root_fs_peak_within_p9_need": None if need is None else root_growth <= need}
        if self.tmp is not None:
            growth = self.tmp_free_before - self.tmp.min_free
            found["sqlite_tmp_fs_peak_growth"] = growth
            found["sqlite_tmp_fs_peak_within_p9_temp_need"] = None if temp_need is None else growth <= temp_need
        return found


def _serve_until_ready(world: MeasuredWorld, name: str) -> float:
    """Enable and start the unit (production's serving state); the monotonic time it reports READY."""
    world.child("clean")
    world.ctl(name + "-enable", "enable", world.unit)
    started = world.begin(name + "-start")
    shown = world.wait_active(600)
    ready = time.monotonic()
    assert shown["ActiveState"] == "active", (name, shown, world.console()[-4000:])
    world.rec.time(name + ".start_to_ready", ready - started)
    return ready


def _v(world: MeasuredWorld, name: str, *, execute: str | None = None) -> None:
    """Paso 6: one start of the disabled unit; serve runs V for N seconds and stops it itself."""
    world.child("verify", execute=execute)
    started = world.begin(name + "-start")
    shown = world.systemd.wait_for(world.d.unit, lambda value: value["ActiveState"] in ("inactive", "failed"),
                                   600, *STATUS)
    world.rec.time(name + ".start_to_inactive", time.monotonic() - started)
    assert shown["ActiveState"] == "inactive" and shown["Result"] == "success", (name, shown,
                                                                                world.console()[-4000:])


def _status(world: MeasuredWorld, compaction_id: str) -> dict:
    """Paso 7 (no witness here: reading the whole production-size root would add to D)."""
    [found] = world.st("paso7-status", "compact", "status", compaction_id)["compactions"]
    return found


def _path(world: MeasuredWorld, name: str, build: str, *, t0: bool = False, abort: bool = False) -> dict:
    """One path from the serving unit's stop to READY of the new head (or of H after the abort)."""
    world.k += 1
    k = f"k{world.k}"
    times_before = dict(world.rec.times)
    world.st(name + "-paso0-completion-status", "completion-status")  # the service is up: not in D
    with Disk(world) as disk:
        down = time.monotonic()
        world.stop_and_disable(name + "-paso1")
        world.rec.time(name + ".paso1_stop_and_disable", time.monotonic() - down)
        boots = 0
        if t0:
            new = world.extract(build)
            assert world.triple_check(new) == "T0"
            world.t0(build, new=new)
            world.serve_clean(name + "-t0-h-check", enable=True)
            boots += 1
        planned = world.plan(build, k)
        compaction_id = world.start(build, k, planned)
        disk.mark("after_start_activated")
        world.verify(compaction_id, SMOKE_SECONDS)
        _v(world, name + "-v", execute=DELETE_TOMBSTONES if abort else None)
        boots += 1
        status = _status(world, compaction_id)
        assert status["verified"] is True and status["armed_run"] == "consumed", status
        h_binding = world.log(compaction_id)["intent"]["source_binding_id"]
        if abort:
            parsed = world.st("paso8-retire-refused", "compact", "retire", compaction_id, expect=78)
            assert parsed["refused"] == compaction.RETIRE_PROTECTED_CHANGED, parsed
            assert parsed["report"]["w"]["class"] == compaction.W_LOSS_ONLY, parsed["report"]["w"]
            world.abort_to_h(compaction_id)
            disk.mark("after_abort")
            # The abort's outcome before H serves (serving opens the head again): H's head, closed, the log
            # terminal and aborted, K gone.
            head = world.head()
            assert head["status"] == "closed" and head["binding_id"] == h_binding, head
            assert world.log(compaction_id)["terminal"] and world.kinds(compaction_id)[-1] == "aborted"
            assert not (world.root / "slots" / k).exists()
        else:
            world.retire(compaction_id)
            disk.mark("after_retire")
        ready = _serve_until_ready(world, name + ("-h-after-abort" if abort else "-k-prime"))
        boots += 1
        d_seconds = ready - down
    parts = {key: value for key, value in world.rec.times.items() if key not in times_before}
    t_copy = parts[f"plan-{k}"] + parts[f"start-{k}"]
    to_ready = [value for key, value in parts.items() if key.endswith(".start_to_ready")]
    stops = [value for key, value in parts.items() if key.endswith((".stop", ".paso1_stop_and_disable"))]
    v_seconds = parts[name + "-v.start_to_inactive"]  # V's boot, its N seconds and its stop
    other = d_seconds - t_copy - sum(to_ready) - sum(stops) - v_seconds  # extract, T0, verify, status, ...
    if abort:
        assert world.head()["binding_id"] == h_binding  # H serves (its head is running while it does)
    else:
        assert world.slot_of_head() == k
    result = {"path": name, "k": k, "compaction_id": compaction_id, "t0": t0, "lossless_abort": abort,
              "d_seconds": round(d_seconds, 3), "budget_seconds": BUDGET_SECONDS,
              "within_budget": d_seconds <= BUDGET_SECONDS, "t_copy_seconds": round(t_copy, 3),
              "smoke_seconds": SMOKE_SECONDS, "boots": boots, "boot_to_ready_seconds": to_ready,
              "stop_seconds": stops, "v_seconds": v_seconds, "other_seconds": round(other, 3), "parts": parts,
              "disk": disk.result(planned.get("capacity") or {})}
    if W8_B is not None:
        # Conservative: every boot to READY becomes B, and V keeps its measured time plus B for its boot.
        projected = d_seconds - sum(to_ready) + boots * W8_B
        result["w8"] = {"b_seconds": W8_B, "d_seconds_projected": round(projected, 3),
                        "within_budget_projected": projected <= BUDGET_SECONDS}
    # downtime_and_disk.formula_check: N = min(60, 600 - T_copy - boots x B - stops), refused below 10; here
    # also minus what the formula leaves out (V's measured run minus its N, and the commands in between).
    b = W8_B if W8_B is not None else max(to_ready)
    rest = sum(stops) + (v_seconds - SMOKE_SECONDS) + other
    n = min(60.0, BUDGET_SECONDS - t_copy - boots * b - rest)
    result["formula_check"] = {"b_seconds": round(b, 3), "b_source": "W8" if W8_B is not None else "measured_max",
                               "stops_and_rest_seconds": round(rest, 3), "default_n": round(n, 1),
                               "n_refused_below_10": n < 10}
    world.rec.result(**{name: result})
    return result


def test_sc6b_final_downtime_and_disk_numbers(builds, tmp_path, systemd, record):
    rec = record("measure", "SC-6b final downtime and disk numbers: first transition with T0, later transition "
                            "without T0, lossless abort; D end to end against 600 s; disk against P9")
    sqlite_tmp = Path(os.environ.get("KZA_SC6B_SQLITE_TMPDIR") or tmp_path / "sqlite-tmp")
    sqlite_tmp.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    world = MeasuredWorld(builds, tmp_path, systemd, rec,
                          first=cw.LEDGER + "\n" + H_TOMBSTONE + "\n" + grow_snippet(EVENTS, UTTERANCES))
    world.sqlite_tmp = sqlite_tmp
    state = Path(stage.read_binding(world.root, world.h_binding)["state_root"])
    h_state = {"bytes": sum(path.lstat().st_size for path in state.rglob("*")
                            if path.is_file() and not path.is_symlink()),
               "allocated_bytes": _tree_bytes(state),
               "databases": {path.name: path.stat().st_size for path in sorted((state / "data").glob("*.db"))},
               "grow_and_close_seconds": round(time.monotonic() - started, 1)}
    _serve_until_ready(world, "h-serving")  # production's state before the first transition
    paths: list[dict] = []

    def summarize(complete: bool) -> dict:
        return {"complete": complete, "events": EVENTS, "utterances": UTTERANCES, "h_state": h_state,
                "host": os.uname().nodename, "cpus": os.cpu_count(), "load_now": os.getloadavg(),
                "sqlite_tmp": str(sqlite_tmp), "w8_b_seconds": W8_B, "smoke_seconds": SMOKE_SECONDS,
                "budget_seconds": BUDGET_SECONDS,
                "paths": {item["path"]: {key: item[key] for key in ("d_seconds", "within_budget", "t_copy_seconds",
                                                                    "boots", "formula_check", "disk")}
                          | ({"w8": item["w8"]} if "w8" in item else {}) for item in paths}}

    # Each path's numbers reach the evidence as soon as it finishes, so a later failure keeps them.
    for name, t0, abort in (("a-first-with-t0", True, False), ("b-later-without-t0", False, False),
                            ("c-lossless-abort", False, True)):
        paths.append(_path(world, name, "x5", t0=t0, abort=abort))
        real_systemd.record("sc6b-measurements", summarize(False))
    world.ctl("final-stop", "stop", world.unit)
    world.d.wait_inactive(120)
    world.ctl("final-disable", "disable", world.unit)
    summary = summarize(True)
    rec.result(summary=summary)
    real_systemd.record("sc6b-measurements", summary)
    # SC-D2 A, re-checked by SC-6b: every path within 600 s (with W8's B when it is given).
    over = [item["path"] for item in paths
            if not (item["w8"]["within_budget_projected"] if "w8" in item else item["within_budget"])]
    assert over == [], summary
