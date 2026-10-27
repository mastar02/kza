"""Release boundaries: imports work without sources or production state."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_runtime.py"
SPEC = importlib.util.spec_from_file_location("kza_build_runtime", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def committed_app(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    files = {
        "src/__init__.py": "",
        "src/main.py": "from src.feature import VALUE\nprint(VALUE)\n",
        "src/feature.py": "VALUE = 'committed-runtime'\n",
        "src/dashboard/frontend/home/index.html": "<h1>Casa</h1>",
        "src/dashboard/frontend/obs/vendor/library.LICENSE.txt": "Synthetic distribution notice\n",
        "config/settings.yaml": "private: production\n",
        ".env": "SECRET=must-not-ship\n",
        "data/users.json": "[]",
        "models/private.bin": "model",
        "tests/test_private.py": "PRIVATE = True\n",
    }
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "fixture"], cwd=repo, check=True,
    )
    return repo


def test_release_runs_without_sources_and_excludes_private_state(committed_app, tmp_path):
    (committed_app / "src/feature.py").write_text("raise RuntimeError('uncommitted')\n")
    output = tmp_path / "runtime.tar.gz"
    manifest = builder.build_runtime(committed_app, output)
    runtime = tmp_path / "runtime"
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert not any(name.endswith(".py") for name in names)
        assert not any(name.startswith(("config/", "data/", "models/", "tests/", ".env"))
                       for name in names)
        archive.extractall(runtime, filter="data")
    assert (runtime / "src/dashboard/frontend/home/index.html").read_text() == "<h1>Casa</h1>"
    assert (runtime / "src/dashboard/frontend/obs/vendor/library.LICENSE.txt").read_text() == "Synthetic distribution notice\n"
    result = subprocess.run(
        [sys.executable, "-E", "-m", "src.main"], cwd=runtime,
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "committed-runtime"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest
    assert json.loads((runtime / "release.json").read_text())["git_commit"] == manifest["git_commit"]


def test_build_is_reproducible_and_refuses_existing_output(committed_app, tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    builder.build_runtime(committed_app, first)
    builder.build_runtime(committed_app, second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        builder.build_runtime(committed_app, first)


def test_runtime_symlink_is_not_followed(committed_app, tmp_path):
    target = committed_app / "src/feature.py"
    target.unlink()
    target.symlink_to(committed_app / ".env")
    subprocess.run(["git", "add", "src/feature.py"], cwd=committed_app, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "symlink"], cwd=committed_app, check=True,
    )
    output = tmp_path / "runtime.tar.gz"
    with pytest.raises(ValueError, match="symlinks"):
        builder.build_runtime(committed_app, output)
    assert not output.exists()
