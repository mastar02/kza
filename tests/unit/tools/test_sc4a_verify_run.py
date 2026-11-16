"""SC-4a controller side: the isolated verification run V of a release-mode log.

sc-release-design v4 mechanism.v_verification_run: the flag (serve adds
KZA_RUNTIME_RELEASE_VERIFY=1 after the KZA_RUNTIME_* filter, only for the
armed verify run of a release-mode log, CR2-8), the records (ready.json after
the child's VERIFY=<run_id> acknowledgement, a monotonic smoke interval and
serve's own stop, smoke-stop.json), the verified predicate (row V4; rows V3
for a short run and for READY without the acknowledgement, R5-PRIV-2) and
'verify branch reached' (R5V-5). The child is the U3b-0d rehearsal child in
NEW's compiled runtime: its verify scenarios read the flag through that
runtime's own take_release_verify_flag and acknowledge through
ProcessLifecycle.acknowledge_verify, as Application.run's verify branch does.

Serve runs with a scaled monotonic clock where the test needs N seconds to
pass; the wall clock is separate (and stepped backwards in one test).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from src.core import runtime_admission
from src.core.process_lifecycle import ProcessLifecycle, ReadinessNotifier, StopRequested
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.unit.tools.test_sc1b_release_plan import _complete_log, _post_r1_world, _refusal, clone, x4, x5  # noqa: F401

SMOKE = 10  # the smallest N compact verify accepts


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller.

    SC4A-N13: a release-mode log is armed only once SC-4b ships V's exits; these tests cover V
    itself, so they run as if that gate were lifted (its refusal is pinned in
    test_the_release_gate_refuses_to_arm_until_sc4b and in test_sc1b_release_plan).
    """
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    monkeypatch.setattr(compaction, "RELEASE_EXITS_AVAILABLE", True)


def _scaled(factor: float):
    origin = time.monotonic()
    return lambda: origin + (time.monotonic() - origin) * factor


def _armed_world(x4, x5, tmp_path: Path) -> dict:
    """The first post-R1 transition up to an armed release-mode verification run."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    world["h_head"] = stage.read_admission_head(root)
    started = cw.start(root, x5["archive"], planned=cw.plan(root, x5["archive"]))
    assert started["next"] == "compact verify " + started["compaction_id"]
    world["id"] = started["compaction_id"]
    armed = compaction.compact_verify(root, world["id"], smoke_seconds=SMOKE)
    assert armed == {"compaction_id": world["id"], "armed": True, "started": False, "smoke_seconds": SMOKE}
    return world


def _serve(root: Path, tmp_path: Path, name: str, scenario: str, *, factor: float = 20.0, stop_when=None,
           wall=time.time, environ: dict | None = None):
    phase_file = tmp_path / ("phases-" + name)
    return rehearsal.run_serve(root, rehearsal.child_command(phase_file, scenario), phase_file=phase_file,
                               stop_when=stop_when, stop_grace=200.0, timeout=60.0, clock=_scaled(factor),
                               wall=wall, environ=environ)


def _phase(result, name: str) -> str | None:
    return next((value for key, value in result.phases if key == name), None)


def _run_id(root: Path, log: dict) -> str:
    _armed, started = stage._armed_run(log)
    return compaction._started_run_id(root, started)


def _status(root: Path, compaction_id: str) -> dict:
    [status] = compaction.compact_status(root, compaction_id)["compactions"]
    return status


def _rewrite(path: Path, raw: bytes) -> None:
    path.unlink()
    path.write_bytes(raw)


# --- a correct NEW: verified on serve's records -----------------------------------------------------

def test_release_verify_run_is_the_acknowledged_isolated_boot_and_serve_stops_it(x4, x5, tmp_path):
    """Row V4: flag, VERIFY before READY, ready.json, N monotonic seconds, serve's stop, receipt, verified."""
    world = _armed_world(x4, x5, tmp_path)
    root, compaction_id = world["root"], world["id"]
    assert _status(root, compaction_id)["verify_branch_reached"] == "not run"
    result = _serve(root, tmp_path, "v", "verify")
    assert result.code == 0  # serve's own stop is a requested stop: no restart
    assert _phase(result, "release_verify") == "True" and _phase(result, "flag_left") == "False"
    assert _phase(result, "verify_sent") is not None and _phase(result, "receipt") is not None
    assert "release verify run armed" in result.serve.events and "smoke stop smoke_elapsed" in result.serve.events
    assert "VERIFY" not in " ".join(result.serve.notifier.sent)  # private, never relayed
    log = stage.read_compaction_log(root, compaction_id)
    run_id = _run_id(root, log)
    ready, smoke = stage.verify_run_records(root, run_id)
    assert ready["verify_acknowledged"] is True
    assert smoke["reason"] == "smoke_elapsed" and smoke["verify_unacknowledged"] is False
    assert smoke["smoke_seconds"] == SMOKE and smoke["measured_seconds"] >= SMOKE
    assert [event["kind"] for event in log["events"]][-3:] == ["verify_armed", "verify_started", "verified"]
    assert log["events"][-1]["data"]["run_id"] == run_id
    assert _status(root, compaction_id)["verify_branch_reached"] == "yes"
    # One armed run per log: verify again only reports; serve refuses the next start (retire pending).
    again = compaction.compact_verify(root, compaction_id, smoke_seconds=SMOKE)
    assert again["verified"] is True and again["started"] is True
    assert cw.serve(root, tmp_path, "after").code == 78
    # SC4A-N1: retire with the strict W equality is SC-4b's; the plain retire refuses a release log.
    _refusal(compaction.RELEASE_RETIRE_UNAVAILABLE, lambda: compaction.compact_retire(root, compaction_id))
    assert not any(event["kind"] == "retire_started"
                   for event in stage.read_compaction_log(root, compaction_id)["events"])


def test_a_wall_clock_step_backwards_during_v_keeps_a_good_run_verified(x4, x5, tmp_path):
    """CR2-8: the interval is monotonic; wall-clock fields are informational only."""
    world = _armed_world(x4, x5, tmp_path)
    calls = []

    def stepping_back():
        calls.append(1)
        return time.time() - 3600.0 * len(calls)

    result = _serve(world["root"], tmp_path, "v", "verify", wall=stepping_back)
    assert result.code == 0 and len(calls) > 3
    log = stage.read_compaction_log(world["root"], world["id"])
    ready, smoke = stage.verify_run_records(world["root"], _run_id(world["root"], log))
    assert smoke["stopped_at"] < ready["ready_at"]  # the wall clock went backwards in between
    assert smoke["measured_seconds"] >= SMOKE and log["events"][-1]["kind"] == "verified"
    assert smoke["measured_seconds"] < 3 * SMOKE  # the monotonic interval, not a wall-clock difference


# --- rows V3: a short run; READY without the acknowledgement (R5-PRIV-2) ---------------------------

def test_a_short_run_is_not_verified_and_serve_refuses_it_afterwards(x4, x5, tmp_path):
    world = _armed_world(x4, x5, tmp_path)
    root, compaction_id = world["root"], world["id"]
    result = _serve(root, tmp_path, "short", "verify", factor=1.0,
                    stop_when=lambda phases: any(key == "ready" for key, _ in phases))
    assert result.code == 0 and _phase(result, "receipt") is not None
    log = stage.read_compaction_log(root, compaction_id)
    ready, smoke = stage.verify_run_records(root, _run_id(root, log))
    assert ready is not None and smoke["reason"] == "stop_request" and smoke["measured_seconds"] < SMOKE
    assert stage.read_admission_head(root)["status"] == "closed"  # sealed with its receipt ...
    assert not compaction._has(log, "verified")  # ... and still not verified
    assert _status(root, compaction_id)["verify_branch_reached"] == "yes"
    # Review RB-3: an unreadable ready.json, or an unreadable run record of another run, leaves the
    # log's phase visible in status (V2/V3's read-only view) instead of 'invalid'.
    ready_path = stage.verify_ready_path(root, _run_id(root, log))
    saved = ready_path.read_bytes()
    other = root / "admission" / "runs" / ("0" * 32 + ".json")  # sorts before every real run record
    for damage, undo in ((lambda: _rewrite(ready_path, b"{}"), lambda: _rewrite(ready_path, saved)),
                         (lambda: other.write_bytes(b"{}"), other.unlink)):
        damage()
        status = _status(root, compaction_id)
        assert "invalid" not in status and status["verify_branch_reached"] == "unreadable"
        assert status["phase"] == log["events"][-1]["kind"] and status["mode"] == "release"
        undo()
    assert _status(root, compaction_id)["verify_branch_reached"] == "yes"
    assert stage.compaction_serve_decision(root, stage.read_admission_head(root), wall=time.time())["code"] \
        == "runtime_admission_compaction_verify_consumed"
    assert cw.serve(root, tmp_path, "again").code == 78
    _refusal("runtime_admission_compaction_not_verified", lambda: compaction.compact_retire(root, compaction_id))


@pytest.mark.parametrize("scenario", ["verify_ignored", "verify_wrong", "verify_after_ready"])
def test_ready_without_the_acknowledgement_is_stopped_at_once_and_never_verified(x4, x5, tmp_path, scenario):
    """R5-PRIV-2: a NEW that ignores the flag (no VERIFY), names another run or acknowledges too late."""
    world = _armed_world(x4, x5, tmp_path)
    root, compaction_id = world["root"], world["id"]
    result = _serve(root, tmp_path, scenario, scenario)
    assert result.code == 0 and _phase(result, "receipt") is not None
    assert _phase(result, "release_verify") == "True"
    assert "smoke stop verify_unacknowledged" in result.serve.events
    assert ("verify acknowledged; smoke interval started" in result.serve.events) is False
    if scenario != "verify_ignored":
        assert result.serve.relay.dropped >= 1
    log = stage.read_compaction_log(root, compaction_id)
    ready, smoke = stage.verify_run_records(root, _run_id(root, log))
    assert ready is None  # no ready.json
    assert smoke["verify_unacknowledged"] is True and smoke["reason"] == "verify_unacknowledged"
    assert smoke["measured_seconds"] == 0.0
    assert stage.read_admission_head(root)["status"] == "closed" and not compaction._has(log, "verified")
    assert _status(root, compaction_id)["verify_branch_reached"] == "no"
    assert cw.serve(root, tmp_path, "again").code == 78
    _refusal("runtime_admission_compaction_not_verified", lambda: compaction.compact_retire(root, compaction_id))


# --- N of a release-mode run (review RB-2, SC4A-N14) ----------------------------------------------

def test_a_release_log_needs_an_explicit_n_within_the_design_cap(x4, x5, tmp_path):
    """SC-D2 A: N is a term of the downtime budget (formula_check: N = min(60, ...), never below 10),
    so a release-mode log has no default N and an N above the cap is refused; nothing is armed."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    compaction_id = cw.start(root, x5["archive"], planned=cw.plan(root, x5["archive"]))["compaction_id"]
    for smoke in (None, compaction.RELEASE_SMOKE_SECONDS_MAX + 1, 120):
        _refusal(compaction.RELEASE_SMOKE_SECONDS,
                 lambda: compaction.compact_verify(root, compaction_id, smoke_seconds=smoke))
    _refusal("runtime_admission_compaction_smoke_seconds",
             lambda: compaction.compact_verify(root, compaction_id, smoke_seconds=9))
    assert not any(event["kind"] == "verify_armed"
                   for event in stage.read_compaction_log(root, compaction_id)["events"])
    armed = compaction.compact_verify(root, compaction_id, smoke_seconds=compaction.RELEASE_SMOKE_SECONDS_MAX)
    assert armed == {"compaction_id": compaction_id, "armed": True, "started": False, "smoke_seconds": 60}


# --- the flag: only serve, only the armed verify run of a release-mode log (CR2-8) -------------------

def test_the_unit_cannot_inject_the_flag_and_h_runs_never_carry_it(x4, x5, tmp_path):
    """A KZA_RUNTIME_RELEASE_VERIFY in the unit's environment is dropped; a VERIFY line in a run that
    is not a release-mode verify run is dropped; an aborted log's next run and a completed log's
    runs never carry the flag."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    environ = {**{key: value for key, value in os.environ.items() if key != "NOTIFY_SOCKET"},
               "KZA_RUNTIME_RELEASE_VERIFY": "1"}
    injected = _serve(root, tmp_path, "h", "verify_unflagged", factor=1.0, environ=environ,
                      stop_when=lambda phases: any(key == "ready" for key, _ in phases))
    assert injected.code == 0 and _phase(injected, "release_verify") == "False"
    assert "KZA_RUNTIME_RELEASE_VERIFY" not in _phase(injected, "env")
    assert _phase(injected, "verify_sent") is not None and injected.serve.relay.dropped >= 1
    assert injected.serve.release_verify is None
    assert not list((root / "admission" / "runs").glob("*" + stage.VERIFY_READY_SUFFIX))
    # An aborted release-mode log (row A0): the next run is H's, without the flag.
    started = cw.start(root, x5["archive"], planned=cw.plan(root, x5["archive"]))
    assert compaction.compact_abort(root, started["compaction_id"])["phase"] == "aborted"
    after_abort = _serve(root, tmp_path, "aborted", "verify", factor=1.0,
                         stop_when=lambda phases: any(key == "ready" for key, _ in phases))
    assert after_abort.code == 0 and _phase(after_abort, "release_verify") == "False"
    assert _phase(after_abort, "verify_sent") is None
    # A completed log (written by hand: no release-mode log completes before SC-4b) serves nothing.
    _complete_log(root, "0" * 64)
    completed = _serve(root, tmp_path, "completed", "verify", factor=1.0,
                       stop_when=lambda phases: any(key == "ready" for key, _ in phases))
    assert completed.code == 0 and _phase(completed, "release_verify") == "False"
    assert not list((root / "admission" / "runs").glob("*" + stage.VERIFY_SMOKE_STOP_SUFFIX))


def test_a_plain_log_s_verify_and_allow_run_runs_never_carry_the_flag(x4_archive, tmp_path):
    """Flag only for release: a plain log's armed verify run and its allow-run run are NEW's ordinary
    boots (U3b-0c), so serve arms nothing, the child reads no flag and its acknowledgement is not sent."""
    world = cw.world(x4_archive, tmp_path, hops=("b",))
    root = world["root"]
    started = cw.start(root, world["archive"])
    compaction_id = started["compaction_id"]
    assert compaction.compact_verify(root, compaction_id) == {  # plain mode keeps the 120 s default
        "compaction_id": compaction_id, "armed": True, "started": False, "smoke_seconds": 120}
    clock = [1_900_000_000.0]
    runs = [cw.serve(root, tmp_path, "plain-verify", scenario="verify")]
    assert "verified" in {event["kind"] for event in stage.read_compaction_log(root, compaction_id)["events"]}
    compaction.compact_allow_run(root, compaction_id, reason="synthetic", hours=1, wall=lambda: clock[0])
    runs.append(cw.serve(root, tmp_path, "plain-allowed", scenario="verify", wall=lambda: clock[0] + 10))
    for result in runs:
        assert result.code == 0, result.phases
        assert _phase(result, "release_verify") == "False" and _phase(result, "verify_sent") is None
        assert "KZA_RUNTIME_RELEASE_VERIFY" not in _phase(result, "env")
        assert result.serve.release_verify is None
    assert not list((root / "admission" / "runs").glob("*" + stage.VERIFY_READY_SUFFIX))
    assert not list((root / "admission" / "runs").glob("*" + stage.VERIFY_SMOKE_STOP_SUFFIX))


def test_child_environment_adds_the_flag_after_the_filter_only_when_asked():
    environ = {"KZA_RUNTIME_RELEASE_VERIFY": "1", "KZA_RUNTIME_OTHER": "x", "HOME": "/h"}
    plain = launcher.child_environment(environ, Path("/r"), {"run_id": "a" * 32}, 5, 6, None)
    assert "KZA_RUNTIME_RELEASE_VERIFY" not in plain and "KZA_RUNTIME_OTHER" not in plain
    flagged = launcher.child_environment({**environ, "KZA_RUNTIME_RELEASE_VERIFY": "0"}, Path("/r"),
                                         {"run_id": "a" * 32}, 5, 6, None, release_verify=True)
    assert flagged["KZA_RUNTIME_RELEASE_VERIFY"] == "1" and "KZA_RUNTIME_OTHER" not in flagged
    assert launcher.RELEASE_VERIFY_ENV == runtime_admission.RELEASE_VERIFY_ENV


def test_notify_relay_counts_verify_only_for_its_run_before_ready():
    run_id = "ab" * 16
    for verify_run_id, lines, acknowledged in (
            (run_id, [f"VERIFY={run_id}", "READY=1"], True),
            (None, [f"VERIFY={run_id}", "READY=1"], False),  # not a release-mode verify run
            (run_id, ["VERIFY=" + "cd" * 16, "READY=1"], False),  # another run
            (run_id, ["READY=1", f"VERIFY={run_id}"], False),  # after READY
            (run_id, [f"VERIFY={run_id.upper()}", "READY=1"], False),  # malformed
    ):
        notifier = launcher.SystemdNotifier(None)
        relay = launcher.NotifyRelay(notifier, verify_run_id=verify_run_id)
        relay.feed(("\n".join(lines) + "\n").encode())
        assert relay.verify_acknowledged is acknowledged and relay.ready
        assert notifier.sent == ["READY=1"]  # VERIFY is never relayed
        assert relay.dropped == (0 if acknowledged else 1)


def test_release_run_verified_needs_both_records_and_the_monotonic_interval(tmp_path):
    run_id = "ef" * 16
    (tmp_path / "admission" / "runs").mkdir(parents=True)

    def records(ready=True, measured=12.0, unacknowledged=False, smoke=SMOKE, extra=None):
        for path in (stage.verify_ready_path(tmp_path, run_id), stage.verify_smoke_stop_path(tmp_path, run_id)):
            if path.exists():
                path.unlink()
        if ready:
            path = stage.verify_ready_path(tmp_path, run_id)
            path.write_text(json.dumps({"schema_version": 1, "run_id": run_id, "verify_acknowledged": True,
                                        "ready_at": 1.0, **(extra or {})}))
        if measured is not None:
            stage.verify_smoke_stop_path(tmp_path, run_id).write_text(json.dumps({
                "schema_version": 1, "run_id": run_id, "smoke_seconds": smoke, "measured_seconds": measured,
                "reason": "smoke_elapsed", "verify_unacknowledged": unacknowledged, "stopped_at": 0.0}))
        return stage.release_run_verified(tmp_path, run_id, SMOKE)

    assert records() is True
    assert records(measured=float(SMOKE)) is True
    assert records(ready=False) is False
    assert records(measured=None) is False
    assert records(measured=SMOKE - 0.01) is False
    assert records(measured=float("nan")) is False
    assert records(unacknowledged=True) is False
    assert records(smoke=SMOKE + 1) is False  # another N than the armed one
    assert records(extra={"note": "x"}) is False  # a malformed record is not a verified run


# --- the child side of the flag and the acknowledgement --------------------------------------------

def test_take_release_verify_flag_reads_once_and_needs_a_ticket():
    assert runtime_admission.take_release_verify_flag({}) is False
    environ = {"KZA_RUNTIME_RELEASE_VERIFY": "1", "KZA_RUNTIME_ADMISSION": "{}"}
    assert runtime_admission.take_release_verify_flag(environ) is True
    assert "KZA_RUNTIME_RELEASE_VERIFY" not in environ
    assert runtime_admission.take_release_verify_flag(environ) is False  # read once
    for bad in ({"KZA_RUNTIME_RELEASE_VERIFY": "1"}, {"KZA_RUNTIME_RELEASE_VERIFY": "0", "KZA_RUNTIME_ADMISSION": "{}"},
                {"KZA_RUNTIME_RELEASE_VERIFY": "true", "KZA_RUNTIME_ADMISSION": "{}"}):
        with pytest.raises(RuntimeError, match="runtime_admission_release_verify_invalid"):
            runtime_admission.take_release_verify_flag(bad)


def test_acknowledge_verify_writes_one_private_line_and_never_after_a_stop():
    read_fd, write_fd = os.pipe()
    try:
        lifecycle = ProcessLifecycle(notifier=ReadinessNotifier(write_fd))
        with pytest.raises(ValueError, match="notify_verify_invalid"):
            lifecycle.acknowledge_verify("not-a-run")
        lifecycle.acknowledge_verify("12" * 16)
        lifecycle.acknowledge_verify("12" * 16)
        assert os.read(read_fd, 4096) == b"VERIFY=" + b"12" * 16 + b"\n"
        lifecycle.stop.request("test")
        with pytest.raises(StopRequested):
            lifecycle.acknowledge_verify("34" * 16)
    finally:
        os.close(read_fd)
        os.close(write_fd)
