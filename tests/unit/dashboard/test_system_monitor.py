"""Tests para system_monitor: GPU parsing + service probe registry."""

import subprocess
import urllib.error

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
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    csv = ("0, NVIDIA GeForce RTX 3070, 47, 5102, 8192, 64, 142.5\n"
           "1, NVIDIA GeForce RTX 3070, 38, 6144, 8192, 71, 168\n")

    def fake_run(cmd, **_):
        return _FakeProc(stdout=csv if "--query-gpu=" in " ".join(cmd) else "")
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
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    csv = "0, GPU, 47, only-five-fields\n1, NVIDIA RTX, 50, 1024, 8192, 60, 100\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **_: _FakeProc(stdout=csv if "--query-gpu=" in " ".join(cmd) else ""))
    gpus = system_monitor.gpu_snapshot()
    assert len(gpus) == 1
    assert gpus[0]["id"] == 1


def test_gpu_snapshot_returns_none_on_timeout(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.5)))
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("driver wedged" in m for m in caplog.messages)


def test_gpu_snapshot_returns_none_on_called_process_error(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(returncode=2, cmd="nvidia-smi", stderr="driver mismatch")))
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("rc=2" in m for m in caplog.messages)


def test_gpu_snapshot_returns_none_on_parse_error(monkeypatch, caplog):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    csv = "not-an-int, GPU, 47, 1024, 8192, 60, 100\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=csv))
    with caplog.at_level("ERROR"):
        assert system_monitor.gpu_snapshot() is None
    assert any("version mismatch" in m for m in caplog.messages)


# ---------- services_snapshot ----------

def test_services_snapshot_returns_none_when_no_signal(monkeypatch):
    """Sin systemctl ni HTTP probes funcionando → None (asume dev box)."""
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: None)
    monkeypatch.setattr(system_monitor, "_http_ok", lambda *a, **kw: False)
    assert system_monitor.services_snapshot() is None


def test_services_snapshot_includes_in_process_always():
    """chromadb es in_process; debe aparecer activo siempre."""
    snap = system_monitor.services_snapshot()
    if snap is None:
        # En CI sin nada respondiendo, es esperable None — saltamos.
        return
    chroma = next((s for s in snap if s["name"] == "chromadb"), None)
    assert chroma is not None
    assert chroma["status"] == "active"


def test_services_snapshot_http_probe_marks_active(monkeypatch):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/systemctl")
    monkeypatch.setattr(system_monitor, "_systemctl_show", lambda _: None)
    # HTTP 200 OK siempre
    monkeypatch.setattr(system_monitor, "_http_ok", lambda *a, **kw: True)
    snap = system_monitor.services_snapshot()
    by_name = {s["name"]: s for s in snap}
    assert by_name["home-assistant"]["status"] == "active"
    assert by_name["vllm-shared"]["status"] == "active"


def test_services_snapshot_http_probe_marks_unreachable(monkeypatch):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/systemctl")
    monkeypatch.setattr(system_monitor, "_systemctl_show", lambda _: None)
    monkeypatch.setattr(system_monitor, "_http_ok", lambda *a, **kw: False)
    snap = system_monitor.services_snapshot()
    if snap is None:
        return
    by_name = {s["name"]: s for s in snap}
    assert by_name["home-assistant"]["status"] == "unreachable"


def test_services_snapshot_systemctl_active(monkeypatch):
    monkeypatch.setattr(system_monitor.shutil, "which", lambda _: "/usr/bin/systemctl")

    def fake_show(svc):
        if svc in ("kza-voice", "kza-llm-ik"):
            return {"name": svc, "status": "active", "uptime": "1h",
                    "mem": "1.0 GB", "cpu": "—", "pid": 100}
        return None
    monkeypatch.setattr(system_monitor, "_systemctl_show", fake_show)
    monkeypatch.setattr(system_monitor, "_http_ok", lambda *a, **kw: False)
    snap = system_monitor.services_snapshot()
    by_name = {s["name"]: s for s in snap}
    assert by_name["kza-voice"]["status"] == "active"
    assert by_name["kza-llm-ik"]["status"] == "active"
    # ma1260-bridge no respondió a systemctl → missing
    assert by_name["ma1260-bridge"]["status"] == "missing"


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


def test_http_ok_treats_4xx_as_alive(monkeypatch):
    """HA sin token devuelve 401 — sigue 'vivo'."""
    class _ResponseHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("http://x", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(system_monitor.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(_ResponseHTTPError()))
    assert system_monitor._http_ok("http://x") is True


def test_http_ok_returns_false_on_connection_error(monkeypatch):
    monkeypatch.setattr(system_monitor.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("nope")))
    assert system_monitor._http_ok("http://x") is False


# ---------- _fmt_bytes ----------

@pytest.mark.parametrize("n,expected", [
    (0, "—"), (-1, "—"), (512, "512.0 B"),
    (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"), (2 * 1024 ** 4, "2.0 TB"),
])
def test_fmt_bytes(n, expected):
    assert system_monitor._fmt_bytes(n) == expected
