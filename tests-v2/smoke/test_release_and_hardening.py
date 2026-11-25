"""L5 release checks. Linux additionally verifies the real systemd datagrams."""
import configparser
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from threading import Event
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/nexa"
BINARIES = ("nexa-audiod", "nexa-brain", "nexa-memoryd", "nexa-web")
SERVICES = {*BINARIES, "nexa-ml", "nexa-dashboard-proxy", "nexa-llm-fast"}


def run(*args, ok=True, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    assert (result.returncode == 0) == ok, (result.args, result.stdout, result.stderr)
    return result


def test_service_units():
    paths = list((DEPLOY / "systemd").iterdir())
    assert {p.name for p in paths} == {s + ".service" for s in SERVICES} | {"nexa.target"}
    for path in paths:
        text = path.read_text()
        assert "/home/kza" not in text
        unit = configparser.ConfigParser(interpolation=None)
        unit.read_string(text)
        if path.suffix == ".target":
            assert set(unit["Unit"]["Wants"].split()) == {s + ".service" for s in SERVICES}
            continue
        s = unit["Service"]
        assert unit["Unit"]["PartOf"] == "nexa.target"
        assert s["NoNewPrivileges"] == s["PrivateTmp"] == "yes"
        assert s["ProtectSystem"] == "strict"
        assert s["TimeoutStopSec"] == "10s"
        if path.stem in (*BINARIES, "nexa-ml"):
            assert s["Type"] == "notify"
            assert s["RuntimeDirectory"] == "nexa" and s["RuntimeDirectoryMode"] == "0700"
            assert "/etc/nexa" in s["ExecStart"]
        if path.stem in BINARIES:
            assert s["ExecStart"].split()[0] == "/opt/nexa/current/bin/" + path.stem
            assert s["ReadWritePaths"] == {
                "nexa-audiod": "/var/lib/nexa/clips",
                "nexa-brain": "/var/lib/nexa/household",
                "nexa-memoryd": "/var/lib/nexa/memory",
                "nexa-web": "/var/lib/nexa/auth",
            }[path.stem]
        if path.stem in {"nexa-llm-fast", "nexa-dashboard-proxy"}:
            assert s["Type"] == "simple"
            assert s["RuntimeDirectoryMode"] == "0700"
        if path.stem == "nexa-dashboard-proxy":
            assert s["RuntimeDirectory"] == "nexa-proxy"
            assert s["ReadWritePaths"] == "/var/lib/nexa/proxy"
        if path.stem == "nexa-llm-fast":
            assert s["RuntimeDirectory"] == "nexa-llm-fast"
            assert s["Environment"] == "CUDA_VISIBLE_DEVICES=0"
        if path.stem in {"nexa-memoryd", "nexa-ml"}:
            assert s["IPAddressDeny"] == "any" and "IPAddressAllow" not in s
        if path.stem == "nexa-ml":
            assert "PYTHONDONTWRITEBYTECODE=1" in s["Environment"].split()
            assert s["WorkingDirectory"] == "/opt/nexa/current/ml"
            assert "%t/nexa/ml.sock" in s["ExecStart"]


def test_ml_persistent_offline_cache():
    unit = configparser.ConfigParser(interpolation=None)
    unit.read(DEPLOY / "systemd/nexa-ml.service")
    service = unit["Service"]
    env = dict(item.split("=", 1) for item in service["Environment"].split())
    cache = Path(env["HOME"]) / ".cache"
    assert cache == Path("/var/lib/nexa/.cache")
    assert Path(env["XDG_CACHE_HOME"]) == cache
    assert Path(env["HF_HOME"]) == cache / "huggingface"
    assert service["ReadWritePaths"].split() == [str(cache)]
    provision = (DEPLOY / "provision-nexa-user.sh").read_text()
    assert 'NEXA_USER="nexa"' in provision.splitlines()
    commands = provision.replace(chr(92) + "\n", " ").splitlines()
    assert any(line.startswith('install -d -o "${NEXA_USER}" -g "${NEXA_USER}" -m 0700 ')
               and str(cache) in line.split() for line in commands)


def test_provisioning_and_udev():
    provision = (DEPLOY / "provision-nexa-user.sh").read_text()
    rules = (DEPLOY / "udev/99-xvf3800-nexa.rules").read_text()
    assert "/home/kza" not in provision + rules
    assert 'for forbidden in sudo adm wheel docker; do' in provision
    assert 'grep -qx "${forbidden}"' in provision
    assert re.search(r'grep -qx "\$\{forbidden\}".*?exit 1\s+fi', provision, re.S)
    assert 'for grp in audio dialout video render; do' in provision
    assert re.findall(r"^\s*usermod .*", provision, re.M) == ['    usermod -aG "${grp}" "${NEXA_USER}"']
    commands = provision.replace(chr(92) + "\n", " ").splitlines()
    assert any(line.startswith('install -d -o "${NEXA_USER}" -g "${NEXA_USER}" -m 0700 ')
               and "/var/lib/nexa/proxy" in line.split() for line in commands)
    assert "99-xvf3800-nexa.rules" in provision
    active = [line for line in rules.splitlines() if line and not line.startswith("#")]
    assert len(active) == 1
    assert all(field in active[0] for field in ('GROUP="nexa"', 'MODE="0660"', 'idVendor}=="2886"', 'idProduct}=="001a"'))
    assert "0666" not in rules


@pytest.fixture
def inputs(tmp_path):
    for folder in ("bin", "ml", "dist"):
        (tmp_path / folder).mkdir()
    for name in BINARIES:
        shutil.copy("/usr/bin/true", tmp_path / "bin" / name)
    (tmp_path / "ml/worker.py").write_text("pass\n")
    (tmp_path / "dist/index.html").touch()
    for folder in (".venv", "__pycache__", ".pytest_cache", "tests"):
        (tmp_path / "ml" / folder).mkdir()
        (tmp_path / "ml" / folder / "ignored").touch()
    (tmp_path / "ml/cache.pyc").touch()
    return tmp_path


def run_packager(inputs, deploy, version, out, *extra, ok=True):
    return run(sys.executable, deploy / "package-release.py", version, out,
        "--target-dir", inputs / "bin", "--ml-src", inputs / "ml",
        "--frontend-dist", inputs / "dist", *extra, ok=ok)


def package(inputs, version="v1", *extra, ok=True):
    out = inputs / version
    run_packager(inputs, DEPLOY, version, out, "--allow-dirty", *extra, ok=ok)
    assert out.exists() == ok
    return out


def test_release_provenance_and_reproducibility(inputs, monkeypatch):
    repo = inputs / "repo"
    deploy = repo / "deploy/nexa"
    shutil.copytree(DEPLOY, deploy, ignore=shutil.ignore_patterns("__pycache__"))
    run("git", "init", "-q", cwd=repo)
    run("git", "add", "deploy", cwd=repo)
    run("git", "-c", "user.name=Smoke", "-c", "user.email=smoke@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture", cwd=repo)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1704067200")
    first, second = inputs / "first", inputs / "second"
    for out in (first, second):
        run_packager(inputs, deploy, "v1", out)
    for name in ("release-manifest.json", "SHA256SUMS"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "release-manifest.json").read_text())
    assert manifest["git_dirty"] is False
    assert manifest["timestamp_utc"] == "2024-01-01T00:00:00+00:00"

    (repo / "untracked").touch()
    dirty = inputs / "dirty"
    rejected = run_packager(inputs, deploy, "v1", dirty, ok=False)
    assert "dirty checkout" in rejected.stderr
    assert not dirty.exists()
    run_packager(inputs, deploy, "v1", dirty, "--allow-dirty")
    assert json.loads((dirty / "release-manifest.json").read_text())["git_dirty"] is True


def test_package_and_switch(inputs):
    v1 = package(inputs)
    files = sorted(p.relative_to(v1).as_posix() for p in v1.rglob("*") if p.is_file())
    sums = (v1 / "SHA256SUMS").read_text().splitlines()
    assert [line[66:] for line in sums] == sorted(set(files) - {"SHA256SUMS"})
    manifest = json.loads((v1 / "release-manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["git_sha"] == run("git", "rev-parse", "HEAD", cwd=ROOT).stdout.strip()
    assert manifest["files"] == sorted(set(files) - {"SHA256SUMS", "release-manifest.json"})
    assert [p.name for p in (v1 / "ml").iterdir()] == ["worker.py"]
    for cmd in (["sha256sum"], ["shasum", "-a", "256"]):
        if shutil.which(cmd[0]):
            run(*cmd, "-c", "SHA256SUMS", cwd=v1)
    root = inputs / "sandbox"
    root.mkdir()
    env = dict(os.environ, NEXA_INSTALL_TEST_ROOT=str(root), NEXA_SKIP_SYSTEMCTL="1", NEXA_SKIP_CHOWN="1")

    def install(*args, ok=True):
        return run("bash", DEPLOY / "install-release.sh", *args, env=env, ok=ok)

    current = root / "opt/nexa/current"
    install("install", v1, "v1")
    install("install", v1, "v1", ok=False)
    install("install", v1, "wrong-version", ok=False)
    assert not (root / "opt/nexa/releases/wrong-version").exists()
    (inputs / "ml/worker.py").write_text("# second version\n")
    units = inputs / "units"
    shutil.copytree(DEPLOY / "systemd", units)
    with (units / "nexa.target").open("a") as target:
        target.write("# second version\n")
    v2 = package(inputs, "v2", "--systemd-dir", units)
    for version in ("../evil", "a..b", ".", "/absolute"):
        install("install", v2, version, ok=False)
        install("rollback", version, ok=False)
    content = (v2 / "ml/worker.py").read_bytes()
    (v2 / "ml/worker.py").write_bytes(content + b"!")
    install("install", v2, "v2", ok=False)
    (v2 / "ml/worker.py").write_bytes(content)
    (v2 / "ml/extra").touch()
    install("install", v2, "v2", ok=False)
    (v2 / "ml/extra").unlink()
    assert os.readlink(current) == "releases/v1"
    ready, done = Event(), Event()

    def observe_current():
        assert os.readlink(current) == "releases/v1"
        ready.set()
        while not done.is_set():
            try:
                observed = os.readlink(current)
            except OSError as error:
                if sys.platform == "darwin" and error.errno in (errno.EINVAL, errno.ENOENT):
                    continue
                raise
            assert observed in {"releases/v1", "releases/v2"}

    with ThreadPoolExecutor(max_workers=1) as pool:
        observer = pool.submit(observe_current)
        try:
            assert ready.wait(timeout=3)
            install("install", v2, "v2")
        finally:
            done.set()
        observer.result()
    assert os.readlink(current) == "releases/v2"
    target = root / "home/nexa/.config/systemd/user/nexa.target"
    assert target.read_bytes() == (units / "nexa.target").read_bytes()
    assert stat.S_IMODE((current / "bin/nexa-web").stat().st_mode) == 0o750
    assert (current / "ml/worker.py").read_bytes() == content
    assert stat.S_IMODE((current / "ml/worker.py").stat().st_mode) == 0o440
    install("rollback", "v1")
    assert os.readlink(current) == "releases/v1"
    assert target.read_bytes() == (v1 / "systemd/nexa.target").read_bytes()
    assert (current / "ml/worker.py").read_text() == "pass\n"


def test_partial_harness_environment(tmp_path):
    env = dict(os.environ, NEXA_INSTALL_TEST_ROOT=str(tmp_path), NEXA_SKIP_CHOWN="1", NEXA_SKIP_SYSTEMCTL="")
    rejected = run("bash", DEPLOY / "install-release.sh", "rollback", "v1", env=env, ok=False)
    assert "test mode requires" in rejected.stderr
    assert not (tmp_path / "opt").exists()


@pytest.mark.parametrize("name", [".env", ".env.local", "prod.env", "casa.glb", "memory.db", "x.sqlite-wal"])
def test_forbidden_files(inputs, name):
    (inputs / "dist" / name).touch()
    package(inputs, ok=False)


def regenerate_sums(release):
    sums = release / "SHA256SUMS"
    sums.chmod(0o600)
    sums.write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(release).as_posix()}\n"
        for p in sorted(release.rglob("*")) if p.is_file() and p != sums
    ))


@pytest.mark.parametrize("entry,content", [
    ("ml/.env", "secret"), ("ml/prod.env", "secret"), ("ml/.env.local", "secret"),
    ("ml/settings.toml", "/home/kza"), ("frontend/casa.glb", "model"),
])
def test_reject_handmade_release(inputs, entry, content):
    release = package(inputs)
    (release / entry).write_text(content)
    regenerate_sums(release)
    env = dict(os.environ, NEXA_INSTALL_TEST_ROOT=str(inputs), NEXA_SKIP_CHOWN="1", NEXA_SKIP_SYSTEMCTL="1")
    run("bash", DEPLOY / "install-release.sh", "install", release, "v1", env=env, ok=False)
    assert not (inputs / "opt/nexa/releases/v1").exists()
    assert not (inputs / "opt/nexa/current").is_symlink()


@pytest.mark.parametrize("missing", ["release-manifest.json", "frontend/index.html"])
def test_production_stage_requires_complete_release(inputs, missing):
    release = package(inputs)
    (release / missing).unlink()
    regenerate_sums(release)
    run(sys.executable, DEPLOY / "stage-release.py", release, inputs / "stage", "v1", "", ok=False)


def test_corrupt_rollback_and_empty_units(inputs):
    env = dict(os.environ, NEXA_INSTALL_TEST_ROOT=str(inputs), NEXA_SKIP_CHOWN="1", NEXA_SKIP_SYSTEMCTL="1")

    def install(*args, ok=True):
        run("bash", DEPLOY / "install-release.sh", *args, env=env, ok=ok)

    v1 = package(inputs)
    v2 = package(inputs, "v2")
    install("install", v1, "v1")
    install("install", v2, "v2")
    old = inputs / "opt/nexa/releases/v1"
    current = inputs / "opt/nexa/current"
    binary = old / "bin/nexa-web"
    binary.write_bytes(b"corrupt")
    install("rollback", "v1", ok=False)
    assert os.readlink(current) == "releases/v2"
    binary.unlink()
    regenerate_sums(old)
    install("rollback", "v1", ok=False)
    assert os.readlink(current) == "releases/v2"
    shutil.copy(v1 / "bin/nexa-web", binary)
    for release in (old, package(inputs, "v3")):
        for unit in (release / "systemd").iterdir():
            unit.unlink()
        regenerate_sums(release)
    install("rollback", "v1", ok=False)
    install("install", inputs / "v3", "v3", ok=False)
    assert os.readlink(current) == "releases/v2"
    assert not (inputs / "opt/nexa/releases/v3").exists()


def test_private_config_and_units(inputs):
    path = inputs / "ml/settings.toml"
    path.write_text('dir = "/home/kza"')
    package(inputs, ok=False)
    path.unlink()
    units = inputs / "units"
    shutil.copytree(DEPLOY / "systemd", units)
    (units / "bad.service").write_text("WorkingDirectory=/home/kza")
    package(inputs, "v1", "--systemd-dir", units, ok=False)


def test_missing_frontend_and_invalid_binary(inputs):
    shutil.rmtree(inputs / "dist")
    package(inputs, ok=False)
    package(inputs, "partial", "--allow-missing-frontend")
    path = inputs / "bin/nexa-web"
    path.chmod(0o644)
    package(inputs, "bad", "--allow-missing-frontend", ok=False)
    path.write_text("placeholder")
    path.chmod(0o755)
    package(inputs, "bad", "--allow-missing-frontend", ok=False)


def test_memoryd_shutdown():
    binary = os.environ.get("NEXA_MEMORYD_BIN")
    if not binary:
        build = run("cargo", "build", "--locked", "-p", "nexa-memoryd", "--message-format=json", cwd=ROOT)
        artifacts = (json.loads(line) for line in build.stdout.splitlines())
        binary = next(a["executable"] for a in artifacts if a.get("executable")
                      and a["target"]["name"] == "nexa-memoryd")
    # Short paths keep sockaddr_un within its macOS and Linux length limits.
    with tempfile.TemporaryDirectory(dir="/tmp") as temp, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
        directory = Path(temp)
        config = (ROOT / "config/nexa/nexa.toml").read_text().replace("/var/lib/nexa", temp + "/data")
        (directory / "nexa.toml").write_text(config)
        notify.bind(temp + "/notify")
        notify.settimeout(3)
        runtime = directory / "run"
        endpoint = runtime / "memory.sock"
        env = dict(os.environ, RUNTIME_DIRECTORY=str(runtime), NOTIFY_SOCKET=temp + "/notify")
        with subprocess.Popen([binary, "--config", temp], env=env, stdout=subprocess.PIPE) as daemon:
            try:
                if sys.platform.startswith("linux"):
                    assert b"READY=1" in notify.recv(4096).splitlines()
                else:
                    # sd_notify is compiled only on Linux. Wait for the real readiness log.
                    assert select.select([daemon.stdout], [], [], 5)[0]
                    assert b"private memory socket ready" in daemon.stdout.readline()
                mode = endpoint.stat().st_mode
                assert stat.S_ISSOCK(mode) and stat.S_IMODE(mode) == 0o600
                start = time.monotonic()
                daemon.terminate()
                assert daemon.wait(timeout=3) == 0
                assert time.monotonic() - start < 3
                assert not endpoint.exists()
                if sys.platform.startswith("linux"):
                    assert b"STOPPING=1" in notify.recv(4096).splitlines()
            finally:
                if daemon.poll() is None:
                    daemon.kill()
