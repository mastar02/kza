"""U3b-0c compiled rehearsals: the installed controller compacts under the fake supervisor (C1, C2, C3, C6, T-C5).

Artifacts as in test_compiled_managed_launcher (X = HEAD plus this working
tree). Every compact command runs as the installed, compiled controller
(controller/current/scripts/stage_runtime.pyc) in its own process, so P7
(controller location), the compiled compaction module and the compaction
worker (spawned from controller/current with its G1 capability) are the
production ones. P14 runs against the systemctl shim, whose listings are
empty (the units live outside the user unit directories); the real user
manager is rehearsed in test_linux_compaction_units. Every serve start,
including the verification run, goes through the rendered managed.conf
under tests/mocks/fake_systemd, in both reboot modes (v5: vendor 5 s budget
and budget >= unit stop + 10 s).
"""

import gzip
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import pytest

from scripts import stage_runtime as stage
from tests.integration import test_compiled_managed_launcher as compiled
from tests.mocks import launcher_rehearsal as rehearsal

REGISTER = compiled.REGISTER_TARGET.replace("'clips'", "('clips-' + state.parent.name)").replace(
    "INSERT INTO voice_group_schema", "INSERT OR IGNORE INTO voice_group_schema")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return rehearsal.build_artifacts(tmp_path_factory.mktemp("compiled-compaction"))


@pytest.fixture
def deployments():
    created = []
    yield created
    for item in created:
        item.close()


def _ready(unit):
    return bool(unit.events("ready"))


def _serve(deployment, scenario="clean", *, execute=None, seconds=0.0):
    """One supervised start of the current head, stopped at READY; returns the unit."""
    if deployment.phase_file.exists():
        deployment.phase_file.unlink()
    unit = deployment.supervise(scenario, execute=execute, seconds=seconds)
    unit.start()
    return unit


def _clean_run(deployment, *, execute=None):
    unit = _serve(deployment, execute=execute)
    assert unit.wait_for(_ready, 90), deployment.console_text()[-3000:]
    assert unit.stop() == 0, deployment.console_text()[-3000:]
    assert deployment.head()["status"] == "closed"


def _refused_start(deployment):
    unit = _serve(deployment)
    unit.wait_inactive(60)
    return unit.exits


def _refused_start_code(deployment):
    unit = _serve(deployment)
    unit.wait_inactive(60)
    return unit.exits[-1]


def _launcher_cli(deployment, *arguments):
    result = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(deployment.root),
                             "--offline", *arguments, "--boot-id-file", str(deployment.boot_id)],
                            capture_output=True, text=True, timeout=600, env=deployment.env)
    assert result.returncode == 0, (result.stdout, result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def _compact(deployment, *arguments, expect=0):
    result = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(deployment.root),
                             "--offline", "compact", *arguments], capture_output=True, text=True, timeout=600,
                            env=deployment.env)
    assert result.returncode == expect, (result.stdout, result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def _world(artifacts, tmp_path, deployments, **options):
    """s (A) -> b (H), each closed by a supervised run that registers its own collector root."""
    deployment = compiled.Deployment(artifacts, tmp_path, **options)
    deployments.append(deployment)
    _clean_run(deployment, execute=REGISTER)
    stage.prepare_slot(artifacts["archives"]["x"], deployment.root, "b", deployment.state)
    assert stage.delegate_slot(deployment.root, "b")["targets"]
    stage.activate_slot(deployment.root, "b")
    _clean_run(deployment, execute=REGISTER)
    return deployment


def _plan_and_start(deployment, archive, k="k", *extra):
    """``extra`` are start-only flags (plan takes none of them here)."""
    planned = _compact(deployment, "plan", k, "--archive", str(archive))
    started = _compact(deployment, "start", k, "--archive", str(archive), "--plan-sha256", planned["plan_sha256"],
                       "--acknowledge-untouched", planned["untouched_sha256"], "--destroy-rollback-states", *extra)
    assert started["phase"] == "activated"
    return planned, started["compaction_id"]


def _hits(root, needle):
    found = set()
    for directory, _names, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw += gzip.decompress(raw)
            if needle in raw:
                found.add(path.relative_to(root).as_posix())
    return found


# --- C1 + C3 + T-C5: the whole compaction, both reboot modes ------------------------------------------------

@pytest.mark.parametrize("budget", ["5s", "75s"])
def test_c1_compiled_compaction_with_canaries_and_a_reboot_during_allow_run(artifacts, tmp_path, deployments,
                                                                            budget):
    archive = artifacts["archives"]["x"]
    deployment = _world(artifacts, tmp_path, deployments, manager_timeout_stop=budget)
    root = deployment.root
    # C3 canaries: in an owner only (A's private file), in an untouched entry, in a member log.
    (deployment.state / "data" / "owner-only.txt").write_text("CANARY-OWNER-A")
    (root / "backups").mkdir()
    (root / "backups" / "copy.txt").write_text("CANARY-UNTOUCHED")
    (root / "slots" / "s" / "state" / "data" / "old.log.1.gz").write_bytes(gzip.compress(b"CANARY-GZ"))
    refused = _compact(deployment, "plan", "k", "--archive", str(archive))
    assert refused["coverage"]["unexplained"] == 2
    _compact(deployment, "start", "k", "--archive", str(archive), "--plan-sha256", refused["plan_sha256"],
             "--acknowledge-untouched", refused["untouched_sha256"], "--destroy-rollback-states", expect=78)
    _planned, compaction_id = _plan_and_start(deployment, archive, "k", "--accept-owner-only-loss")
    assert _refused_start(deployment) == [78]  # pending, not armed
    _compact(deployment, "verify", compaction_id)
    _clean_run(deployment)  # T-C5: the armed run, through managed.conf
    status = _compact(deployment, "status", compaction_id)["compactions"][0]
    assert "verified" in status["phases"]
    assert _refused_start(deployment) == [78]  # consumed: no restart loop (RestartPreventExitStatus=78)
    # allow-run, and a reboot while it serves.
    _compact(deployment, "allow-run", compaction_id, "--reason", "rehearsal", "--hours", "1")
    unit = _serve(deployment, "slow_stop", seconds=2.0 if budget == "75s" else 4.0)
    assert unit.wait_for(_ready, 90), deployment.console_text()[-3000:]
    unit.manager_stop(75.0 if budget == "75s" else 5.0)
    if budget == "75s":
        assert unit.exits[-1] == 0 and deployment.head()["status"] == "closed"
    else:
        # F1: while a compaction is pending an unclean stop exits 69 (no systemd restart), not 68.
        assert unit.exits[-1] == 69 and deployment.head()["status"] == "unknown"
        unit.reboot(deployment.boot_id)
        if (root / "admission" / "receipts" / (deployment.head()["run_id"] + ".json")).exists():
            _clean_run(deployment)  # F1 (a): a receipt exists, R-a seals it at the next start
        else:
            # F1 (b)/(c): no receipt; automatic R-b stays off while pending (69, no restart) ...
            assert _refused_start_code(deployment) == 69
            # ... and the operator's R-b (adopt) is allowed for an allow-run run, then serving resumes.
            dry = _launcher_cli(deployment, "adopt", "--dry-run")
            assert dry["report"]["failures"] == [], dry
            _launcher_cli(deployment, "adopt", "--confirm", dry["head_sha256"], "--confirm-regressions",
                          ",".join(dry["report"]["regressions"]))
            assert deployment.head()["status"] == "closed"
            _clean_run(deployment)
    retired = _compact(deployment, "retire", compaction_id)
    assert retired["phase"] == "complete"
    assert sorted(os.listdir(root / "slots")) == ["k"]
    for needle, expected in ((b"CANARY-OWNER-A", set()), (b"CANARY-GZ", set()),
                             (b"CANARY-UNTOUCHED", {"backups/copy.txt"})):
        assert _hits(root, needle) == expected, needle
    _clean_run(deployment)
    assert _compact(deployment, "resume", compaction_id)["noop"] is True


# --- C6: the verification run fails, abort --to-h through the compiled CLI ----------------------------------

def test_c6_failed_verification_returns_to_h(artifacts, tmp_path, deployments):
    archive = artifacts["archives"]["x"]
    deployment = _world(artifacts, tmp_path, deployments)
    h = deployment.head()
    _planned, compaction_id = _plan_and_start(deployment, archive)
    _compact(deployment, "verify", compaction_id)
    unit = _serve(deployment, "crash", seconds=0.3)
    unit.wait_inactive(90)
    assert unit.exits and unit.exits[-1] == 69, (unit.exits, deployment.console_text()[-3000:])
    assert deployment.head()["status"] == "unknown"
    _compact(deployment, "retire", compaction_id, expect=78)
    aborted = _compact(deployment, "abort", compaction_id, "--to-h", "--confirm-h", h["binding_id"][:16])
    assert aborted["phase"] == "aborted" and aborted["head"]["binding_id"] == h["binding_id"]
    assert sorted(os.listdir(deployment.root / "slots")) == ["b", "s"]
    _clean_run(deployment)


# --- C2: seeded SIGKILLs of the compiled start and retire, then resume ----------------------------------------

def _killed(deployment, arguments, delay):
    process = subprocess.Popen([sys.executable, "-I", "-B", str(deployment.controller), "--root",
                                str(deployment.root), "--offline", "compact", *arguments],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=deployment.env,
                               start_new_session=True)
    time.sleep(delay)
    if process.poll() is None:
        # The whole group, worker included (a power loss); the controller alone would leave the
        # worker holding the lease, and resume would rightly answer runtime_admission_busy.
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=120)
    return process.returncode


def _record(name, value):
    """KZA_0C_EVIDENCE=<dir>: content-free outcomes of the seeded kills (which phase each one cut)."""
    directory = os.environ.get("KZA_0C_EVIDENCE")
    if directory:
        with open(Path(directory) / "c2-kills.jsonl", "a") as stream:
            stream.write(json.dumps({"case": name, **value}) + "\n")


@pytest.mark.parametrize("seed, start_window, retire_window", [
    (3, (0.2, 0.8), (0.05, 0.15)), (11, (1.0, 2.5), (0.15, 0.3)), (19, (2.5, 5.0), (0.3, 0.6)),
    (27, (5.0, 9.0), (0.6, 1.2))])
def test_c2_sigkill_during_start_and_retire_then_resume(artifacts, tmp_path, deployments, seed, start_window,
                                                        retire_window):
    chooser = random.Random(seed)
    archive = artifacts["archives"]["x"]
    deployment = _world(artifacts, tmp_path, deployments)
    root = deployment.root
    planned = _compact(deployment, "plan", "k", "--archive", str(archive))
    code = _killed(deployment, ["start", "k", "--archive", str(archive), "--plan-sha256", planned["plan_sha256"],
                                "--acknowledge-untouched", planned["untouched_sha256"],
                                "--destroy-rollback-states"], chooser.uniform(*start_window))
    logs = stage.compaction_logs(root)
    if not logs:  # killed before the intent: nothing changed
        assert code == -signal.SIGKILL and sorted(os.listdir(root / "slots")) == ["b", "s"]
        _record("start", {"seed": seed, "cut": "before_intent"})
        return
    [log] = logs
    compaction_id = log["compaction_id"]
    _record("start", {"seed": seed, "cut": log["events"][-1]["kind"] if log["events"] else "intent"})
    resumed = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(root),
                              "--offline", "compact", "resume", compaction_id, "--archive", str(archive)],
                             capture_output=True, text=True, timeout=600, env=deployment.env)
    outcome = json.loads(resumed.stdout.strip().splitlines()[-1])
    if resumed.returncode == 78:  # a materialization cut mid-way is never redone (C3): abort, new K name
        assert outcome["refused"] == "runtime_admission_compaction_resume_abort_required", outcome
        assert _compact(deployment, "abort", compaction_id)["phase"] == "aborted"
        assert sorted(os.listdir(root / "slots")) == ["b", "s"]
        _clean_run(deployment)
        _planned, compaction_id = _plan_and_start(deployment, archive, "k2")
    else:
        assert outcome["phase"] in {"activated"}, outcome
    assert _compact(deployment, "resume", compaction_id)["noop"] is True
    _compact(deployment, "verify", compaction_id)
    _clean_run(deployment)
    _killed(deployment, ["retire", compaction_id], chooser.uniform(*retire_window))
    events = stage.read_compaction_log(root, compaction_id)["events"]
    _record("retire", {"seed": seed, "cut": events[-1]["kind"],
                       "destroyed": sum(1 for event in events if event["kind"] == "destroyed")})
    if any(event["kind"] == "retire_started" for event in events):
        assert _compact(deployment, "resume", compaction_id)["phase"] == "complete"
    else:  # cut before anything was destroyed: resume has nothing to finish; retire runs again
        assert _compact(deployment, "resume", compaction_id) == {"compaction_id": compaction_id,
                                                                 "phase": "activated", "noop": True}
        assert _compact(deployment, "retire", compaction_id)["phase"] == "complete"
    assert _compact(deployment, "resume", compaction_id)["noop"] is True
    assert sorted(name for name in os.listdir(root / "slots") if not name.startswith(".")) in (["k"], ["k2"])
    _clean_run(deployment)
