"""Availability accounting and acceptance against a real sealed candidate."""

import json
from pathlib import Path
import socket
import stat
import subprocess
import sys

import pytest

from scripts.build_runtime import build_runtime
from tools import p6_local_soak as soak
from tools.release_manifest_review import ReviewInputs, review_manifest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def dev_user(monkeypatch):
    monkeypatch.setattr(soak.os, "getuid", lambda: 1500)
    monkeypatch.setattr(soak.os, "geteuid", lambda: 1500)
    monkeypatch.setattr(soak.pwd, "getpwuid", lambda uid: type("User", (), {"pw_name": "yo"})())
    monkeypatch.setattr(soak.getpass, "getuser", lambda: "yo")


@pytest.mark.parametrize("path", ["/home/kza", "/home/kza/release", "/opt/homeassistant/data"])
def test_protected_paths(dev_user, path):
    with pytest.raises(ValueError, match="protected path"):
        soak.guard(Path(path))


def test_guard_detects_symlink_into_protected_path(dev_user, tmp_path):
    redirect = tmp_path / "redirect"
    redirect.symlink_to("/home/kza", target_is_directory=True)
    with pytest.raises(ValueError, match="protected path"):
        soak.guard(redirect / "artifact")


def test_protected_accounts(dev_user, monkeypatch):
    monkeypatch.setattr(soak.getpass, "getuser", lambda: "kza")
    with pytest.raises(ValueError, match="unprivileged development"):
        soak.guard()
    monkeypatch.setattr(soak.getpass, "getuser", lambda: "yo")
    monkeypatch.setattr(soak.os, "geteuid", lambda: 1000)
    with pytest.raises(ValueError, match="UID 1000"):
        soak.guard()


def test_thermouy_queries_all_four_units_and_reports_interference(monkeypatch):
    monkeypatch.setattr(soak.sys, "platform", "linux")
    queried = []

    def show(cmd, **kwargs):
        queried.extend(cmd)
        blocks = [
            f"Id={name}\nLoadState=loaded\nActiveState={'active' if index < 2 else 'inactive'}\n"
            "SubState=waiting\nNextElapseUSecRealtime=Sat 2026-11-21 05:00:00 UTC"
            for index, name in enumerate(soak.THERMOUY)
        ]
        return subprocess.CompletedProcess(cmd, 0, "\n\n".join(blocks), "")

    monkeypatch.setattr(soak.subprocess, "run", show)
    result = soak.thermouy_status()
    assert queried[:3] == ["systemctl", "--user", "show"]
    assert tuple(queried[3:7]) == soak.THERMOUY
    assert result["running_services"] == ["thermouy-tier2-nightly.service"]
    assert result["scheduled_timers"] == ["thermouy-tier2-nightly.timer"]
    assert result["interference"] is True

    monkeypatch.setattr(soak.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", ""))
    assert soak.thermouy_status()["interference"] is None


def test_render_user_unit(tmp_path):
    unit = soak.render_unit(tmp_path / "sealed artifact", 259200, faults_first=True)
    assert "CPUAffinity=24-47\nNice=5\nMemoryHigh=24G\nNoNewPrivileges=yes\n" in unit
    assert "IPAddressDeny=any\nIPAddressAllow=localhost" in unit
    assert "PrivateDevices=yes" in unit
    assert "PrivateNetwork=yes" in unit
    assert unit.index("inject-faults") < unit.index("run-soak")
    assert "--duration-s 259200" in unit
    assert "ExecStopPost=" in unit and "summarize" in unit
    assert "kza-voice.service" not in unit


def test_socket_policy_accepts_loopback_and_refuses_external_connections():
    with socket.socket(socket.AF_INET) as sock:
        soak.offline_io("socket.bind", (sock, ("127.0.0.1", 0)))
        with pytest.raises(PermissionError, match="external network"):
            soak.offline_io("socket.connect", (sock, ("192.0.2.1", 443)))
    with pytest.raises(PermissionError, match="DNS"):
        soak.offline_io("socket.getaddrinfo", ("example.invalid", 443))


def sample(second, **kwargs):
    return dict(observed_at=second + 0.1, kza_useful=True,
                rooms={room: {"ever": True, "age_s": 0.02} for room in soak.ROOMS}, **kwargs)


def test_gaps_and_authenticated_mute_stay_in_denominator():
    block = soak.availability(0, 4, {
        0: sample(0), 2: sample(2, pause="authenticated_global_mute"), 3: sample(3),
    }, [])
    assert block["gap_seconds"] == 1 and block["window_seconds"] == 4
    assert block["series"]["kza"]["availability"] == 0.75
    for room in soak.ROOMS:
        assert block["series"][room]["availability"] == 0.50
        assert block["series"][room]["authenticated_pause_seconds"] == 1
        assert block["series"][room]["spontaneous_unavailable_seconds"] == 0
        assert block["series"][room]["unclassified_gap_seconds"] == 1


def test_one_room_below_threshold_fails_c19():
    late = sample(0)
    late["rooms"]["hall"]["age_s"] = 2
    normal = sample(0)
    samples = dict.fromkeys(range(270000), normal)
    for second in range(297):
        samples[second] = late
    block = soak.availability(0, 270000, samples, [])
    assert block["series"]["hall"]["availability"] == pytest.approx(0.9989)
    assert all(block["series"][key]["availability"] == 1 for key in ("kza", "living", "escritorio", "cocina"))
    assert not soak.verdicts(block, True, {}, False)["c19_72h_passed"]
    for second in range(270, 297):
        samples[second] = normal
    block = soak.availability(0, 270000, samples, [])
    assert soak.verdicts(block, True, {}, False)["c19_72h_passed"]
    assert not soak.verdicts(block, False, {}, False)["c19_local_window_met"]


def test_short_perfect_window_cannot_pass_c19():
    block = soak.availability(0, 3, {second: sample(second) for second in range(3)}, [])
    verdicts = soak.verdicts(block, True, {}, True)
    assert verdicts["c19_local_window_met"] is verdicts["c19_72h_passed"] is False
    assert verdicts["overall_t061_local_passed"] is False


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealed-t061").resolve()
    archive = root / "runtime.tar.gz"
    commit = build_runtime(ROOT, archive)["git_commit"]
    lock = root / "runtime.lock"
    lock.write_text("aiosqlite==0.22.1 --hash=sha256:" + "a" * 64 + "\n")
    review = root / "release-review.json"
    seal = review_manifest(ReviewInputs(ROOT, archive, commit, soak.digest(archive), lock))
    review.write_text(json.dumps(seal))
    return archive, review, commit


def cli(*args):
    result = subprocess.run([sys.executable, "-B", "-m", "tools.p6_local_soak", *map(str, args)],
                            capture_output=True, text=True, cwd=ROOT, timeout=280, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_rejects_archive_with_wrong_hash(candidate, tmp_path):
    archive, review, commit = candidate
    manifest = json.loads(review.read_text())
    manifest["artifact"]["sha256"] = "0" * 64
    wrong = tmp_path / "release-review.json"
    wrong.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sealed hash"):
        soak.preflight(archive, wrong, tmp_path.resolve())
    assert not (tmp_path / commit).exists()


def test_real_candidate_short_soak_and_faults(candidate, tmp_path):
    archive, review, commit = candidate
    receipt = json.loads(cli("preflight", "--archive", archive, "--review", review, "--state-root", tmp_path.resolve()))
    state = Path(receipt["state"])
    assert state.name == commit and receipt["bind_port"] == 0
    assert stat.S_IMODE((state / "runtime").stat().st_mode) == 0o555
    assert stat.S_IMODE((state / "runtime/src/main.pyc").stat().st_mode) == 0o444
    assert not list((state / "runtime").rglob("*.py"))

    cli("run-soak", "--state", state, "--duration-s", 3)
    cli("inject-faults", "--state", state)
    cli("summarize", "--state", state)
    result = json.loads((state / "soak-summary.json").read_text())
    assert result["stability"]["window_seconds"] == 3
    assert result["stability"]["samples"] == 3
    assert all(value["useful_seconds"] == 3 for value in result["stability"]["series"].values())
    assert result["c19_72h_passed"] is result["overall_t061_local_passed"] is False
    assert result["faults_outside_stability"] is True
    assert result["total"]["window_seconds"] >= 3 + result["fault_phase"]["window_seconds"]
    assert result["total"]["series"]["kza"]["planned_unavailable_seconds"] > 0
    evidence = result["fault_evidence"]
    assert evidence["ha_ws"]["recovery_s"] <= 60, evidence
    assert evidence["microphone"]["recovery_s"] <= 180, evidence
    assert evidence["ha_ws"]["h2"]["p95_ms"] <= 25
    assert evidence["ha_ws"]["h2"]["max_ms"] <= 75
    assert evidence["global_mute_restore"]["resurrected_rows"] == 0, evidence
    assert result["c20_faults_passed"] is result["c03_h2_passed"] is result["c04_mute_passed"] is True, evidence

    # pytest owns this private fixture and needs to remove its read-only tree later.
    for directory in (state / "runtime", *(path for path in (state / "runtime").rglob("*") if path.is_dir())):
        directory.chmod(0o700)
