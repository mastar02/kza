"""U3b-0c under this account's real user systemd: the compaction gate, the verification run and P14.

The compiled controller (archive X = HEAD plus this working tree) is
installed with its own install-controller and install-launcher against the
real user manager (tests/integration/test_linux_managed_launcher.RealDeployment,
test-only units kza-rehearsal-*, guarded systemctl). Every compact command
runs as that installed controller, so P14 reads this account's real units
through ``systemctl --user`` (list-unit-files, list-units, show, cat) and
the verification run is a real ``systemctl --user start``/``stop`` of the
stand-in unit:

- a drop-in naming a member state refuses the plan (unit_path:<unit>);
- while the compaction is pending and not armed, a start exits 78 and
  systemd does not restart it (RestartPreventExitStatus);
- the armed run serves and closes; ``verified`` is recorded; the next start
  exits 78 again without a restart;
- retire completes (P14 again) and the unit then serves K'.

Skipped off Linux, without a running user manager, or as root or kza
(real_systemd.unavailable). Run alone, one pytest at a time (it
daemon-reloads this account's manager). KZA_0E4_EVIDENCE=<dir> records
content-free results.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.integration import test_linux_managed_launcher as real
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks import real_systemd
from tests.mocks.epoch4 import epoch4_commit
from tests.mocks.legacy_transitions import legacy_managed_transitions

pytestmark = pytest.mark.skipif(real_systemd.unavailable() is not None,
                                reason=real_systemd.unavailable() or "available")



@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """X at controller epoch 4 (SC-1a): an owner world is never served from epoch 5."""
    base = tmp_path_factory.mktemp("linux-compaction-epoch4")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(real.REPO), str(clone)],
                   check=True, timeout=120)
    revision = epoch4_commit(clone, rehearsal.overlay_commit(real.REPO, clone))
    path = base / "x4.tar.gz"
    manifest = build_runtime(clone, path, revision)
    assert manifest["git_commit"] == revision and manifest["controller_epoch"] == 4
    return path


@pytest.fixture(autouse=True)
def _epoch4(epoch4_controller):
    """The in-process controller as the same epoch-4 build."""

systemd = real.systemd
REGISTER = "\n".join([
    "import os, types",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
    "from src.memory.erasure_journal import ErasureParticipant",
    "ledger = fact_ledger.FactLedger(str(state / 'data' / 'privacy.db'))",
    "ledger.initialize()",
    "name = 'clips-' + state.parent.name",
    "os.makedirs(state / 'data' / name, exist_ok=True)",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1, {'kind': 'wake_clips',",
    "    'root': str(state / 'data' / name)})",
    "ledger._db.execute('INSERT OR IGNORE INTO voice_group_schema VALUES (1)')",
    "ledger.close()",
])


@pytest.fixture(scope="module", autouse=True)
def cleanup_evidence():
    real_systemd.record("compaction-stale-sweep", real_systemd.sweep_stale())
    yield
    real_systemd.record("compaction-unit-cleanup", real.CLEANUPS)


def _compact(deployment, *arguments, expect=0):
    result = subprocess.run([sys.executable, "-I", "-B", str(deployment.controller), "--root", str(deployment.root),
                             "--offline", "compact", *arguments], capture_output=True, text=True, timeout=900,
                            env=deployment.systemd.environment())
    assert result.returncode == expect, (result.stdout, result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def _clean_run(deployment, *, execute=None):
    if deployment.phase_file.exists():
        deployment.phase_file.unlink()
    deployment.launcher_child("clean", execute=execute)
    deployment.start()
    assert deployment.wait_phase("ready", 120), deployment.console_text()[-3000:]
    completed, seconds = deployment.stop()
    assert completed.returncode == 0, completed.stderr
    shown = deployment.wait_inactive(120)
    assert shown["Result"] == "success", (shown, deployment.console_text()[-3000:])
    assert deployment.head()["status"] == "closed"
    return seconds


def _refused_start(deployment):
    """A start the gate refuses: exit 78, inactive or failed, never restarted."""
    deployment.launcher_child("clean")
    deployment.start()
    shown = deployment.wait_inactive(120)
    time.sleep(1.0)
    shown = deployment.show()
    assert shown["ExecMainStatus"] == "78" and shown["NRestarts"] == "0", (shown, deployment.console_text()[-2000:])
    deployment.systemd.systemctl("reset-failed", "kza-voice.service")
    return shown


def test_compaction_gate_verification_run_and_p14_under_real_systemd(archive, tmp_path, systemd):
    deployment = real.RealDeployment(archive, tmp_path, systemd)
    root = deployment.root
    _clean_run(deployment, execute=REGISTER)
    stage.prepare_slot(archive, root, "b", deployment.state)
    # SC-2: the owner hop is a legacy world built in process under legacy_managed_transitions;
    # the compiled controller then serves and compacts it under real systemd unchanged.
    with legacy_managed_transitions():
        assert stage.delegate_slot(root, "b")["targets"]
        stage.activate_slot(root, "b")
    _clean_run(deployment, execute=REGISTER)
    # P14: a drop-in on this account's unit naming a member refuses the plan.
    systemd.write_dropin(deployment.unit, "zz-member.conf",
                         f"[Service]\nReadOnlyPaths={root / 'slots' / 's' / 'state' / 'data'}\n")
    systemd.reload()
    refused = _compact(deployment, "plan", "k", "--archive", str(archive), expect=78)
    assert refused["refused"].startswith("runtime_admission_compaction_unit_path:"), refused
    (deployment.unit_dir / "zz-member.conf").unlink()
    systemd.reload()
    planned = _compact(deployment, "plan", "k", "--archive", str(archive))
    assert planned["units"]["units_scanned"] >= 1
    started = _compact(deployment, "start", "k", "--archive", str(archive), "--plan-sha256", planned["plan_sha256"],
                       "--acknowledge-untouched", planned["untouched_sha256"], "--destroy-rollback-states")
    compaction_id = started["compaction_id"]
    pending = _refused_start(deployment)
    _compact(deployment, "verify", compaction_id)
    verify_seconds = _clean_run(deployment)
    log = stage.read_compaction_log(root, compaction_id)
    assert [event["kind"] for event in log["events"]][-3:] == ["verify_armed", "verify_started", "verified"]
    consumed = _refused_start(deployment)
    retired = _compact(deployment, "retire", compaction_id)
    assert retired["phase"] == "complete" and sorted(os.listdir(root / "slots")) == ["k"]
    _clean_run(deployment)
    real_systemd.record("compaction-real-systemd", {
        "p14_refusal": refused["refused"].split(":", 1)[0], "units_scanned": planned["units"]["units_scanned"],
        "pending_start": {key: pending[key] for key in ("ExecMainStatus", "NRestarts", "Result")},
        "consumed_start": {key: consumed[key] for key in ("ExecMainStatus", "NRestarts", "Result")},
        "verify_stop_seconds": round(verify_seconds, 2), "members_destroyed": retired["destroyed"]})
