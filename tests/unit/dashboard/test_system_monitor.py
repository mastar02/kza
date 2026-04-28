"""Tests para system_monitor: parsers de nvidia-smi y systemctl."""

import subprocess
from unittest.mock import MagicMock

import pytest

from src.dashboard import system_monitor


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


# ---------- gpu_snapshot ----------

def test_gpu_snapshot_returns_none_when_smi_missing(monkeypatch):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: None)
    assert system_monitor.gpu_snapshot() is None


def test_gpu_snapshot_parses_csv_sample(monkeypatch):
    monkeypatch.setattr(system_monitor, "_have_nvidia_smi", lambda: True)
    csv = ("0, NVIDIA GeForce RTX 3070, 47, 5102, 8192, 64, 142.5\n"
           "1, NVIDIA GeForce RTX 3070, 38, 6144, 8192, 71, 168\n")

    def fake_run(cmd, **_):
        if "--query-gpu=" in " ".join(cmd):
            return _FakeProc(stdout=csv)
        # _gpu_procs sub-call
        return _FakeProc(stdout="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    gpus = system_monitor.gpu_snapshot()
    assert len(gpus) == 2
    g0 = gpus[0]
    assert g0["id"] == 0
    assert g0["name"].endswith("(cuda:0)")
    assert g0["util"] == 47
    assert g0["vramUsed"] == round(5102 / 1024, 1)
    assert g0["vramTotal"] == round(8192 / 1024, 1)
    assert g0["temp"] == 64
    assert g0["power"] == 142
    assert g0["role"].startswith("STT")
    assert gpus[1]["role"].startswith("vLLM")


def test_gpu_snapshot_skips_malformed_line(monkeypatch):
    monkeypatch.setattr(system_monitor, "_have_nvidia_smi", lambda: True)
    csv = "0, GPU, 47, only-five-fields\n1, NVIDIA RTX, 50, 1024, 8192, 60, 100\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **_: _FakeProc(stdout=csv if "--query-gpu=" in " ".join(cmd) else ""))
    gpus = system_monitor.gpu_snapshot()
    assert len(gpus) == 1
    assert gpus[0]["id"] == 1


def test_gpu_snapshot_returns_none_on_timeout(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor, "_have_nvidia_smi", lambda: True)

    def boom(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.5)
    monkeypatch.setattr(subprocess, "run", boom)
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("driver wedged" in m for m in caplog.messages)


def test_gpu_snapshot_returns_none_on_called_process_error(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor, "_have_nvidia_smi", lambda: True)

    def boom(*_a, **_kw):
        raise subprocess.CalledProcessError(returncode=2, cmd="nvidia-smi", stderr="driver mismatch")
    monkeypatch.setattr(subprocess, "run", boom)
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("rc=2" in m for m in caplog.messages)


def test_gpu_snapshot_returns_none_on_parse_error(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor, "_have_nvidia_smi", lambda: True)
    csv = "not-an-int, GPU, 47, 1024, 8192, 60, 100\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=csv))
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("version mismatch" in m for m in caplog.messages)


# ---------- services_snapshot ----------

def test_services_snapshot_returns_none_when_systemctl_missing(monkeypatch):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: None)
    assert system_monitor.services_snapshot() is None


def test_services_snapshot_returns_none_when_user_units_fail(monkeypatch):
    monkeypatch.setattr(system_monitor, "_have_systemctl", lambda: True)
    monkeypatch.setattr(system_monitor, "_systemctl_user_works", lambda: False)
    assert system_monitor.services_snapshot() is None


def test_services_snapshot_individual_missing_does_not_abort(monkeypatch):
    monkeypatch.setattr(system_monitor, "_have_systemctl", lambda: True)
    monkeypatch.setattr(system_monitor, "_systemctl_user_works", lambda: True)

    def fake_show(svc):
        if svc == "kza-voice":
            return None  # missing
        return {"name": svc, "status": "active", "uptime": "—",
                "mem": "1.0 GB", "cpu": "—", "pid": 100}
    monkeypatch.setattr(system_monitor, "_systemctl_show", fake_show)

    snap = system_monitor.services_snapshot()
    assert len(snap) == len(system_monitor.KZA_SERVICES)
    by_name = {s["name"]: s for s in snap}
    assert by_name["kza-voice"]["status"] == "missing"
    assert by_name["kza-llm-ik"]["status"] == "active"


def test_systemctl_show_parses_key_value(monkeypatch):
    out = ("ActiveState=active\nMainPID=1234\n"
           "ExecMainStartTimestamp=Mon 2026-04-28 14:30:00\n"
           "MemoryCurrent=2147483648\n")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=out, returncode=0))
    info = system_monitor._systemctl_show("kza-voice")
    assert info["status"] == "active"
    assert info["pid"] == 1234
    assert info["mem"] == "2.0 GB"


def test_systemctl_show_handles_empty_memory(monkeypatch):
    out = "ActiveState=active\nMainPID=\nMemoryCurrent=\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=out, returncode=0))
    info = system_monitor._systemctl_show("kza-voice")
    assert info["mem"] == "—"
    assert info["pid"] == 0


def test_systemctl_show_returns_none_on_nonzero_rc(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout="", returncode=1))
    assert system_monitor._systemctl_show("nonexistent") is None


# ---------- _fmt_bytes ----------

@pytest.mark.parametrize("n,expected", [
    (0, "—"), (-1, "—"), (512, "512.0 B"),
    (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"), (2 * 1024 ** 4, "2.0 TB"),
])
def test_fmt_bytes(n, expected):
    assert system_monitor._fmt_bytes(n) == expected
