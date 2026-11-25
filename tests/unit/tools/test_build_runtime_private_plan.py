"""Private plan inputs enter only the intended release archive and manifest."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tarfile

import pytest

from scripts.build_runtime import NEXO_PRIVATE_PLAN_DIR, NEXO_PLAN_LOADER, build_runtime
from tests.mocks.runtime_artifact_cache import CACHE_DIR_ENV, cached_build_runtime


def test_private_nexo_plan_is_optional_and_required_when_explicit(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init")
    sources = ("src/__init__.py", "src/main.py", NEXO_PLAN_LOADER, ".gitignore")
    for name in sources:
        source = repo / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    (repo / ".gitignore").write_text(f"/{NEXO_PRIVATE_PLAN_DIR}/\n")
    git("add", "--", *sources)
    git("-c", "user.name=Runtime Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--no-verify", "-m", "Fixture")

    private = repo / NEXO_PRIVATE_PLAN_DIR
    private.mkdir()
    model = private / "casa.glb"
    model.write_bytes(b"invalid")
    output = tmp_path / "required.tar.gz"
    with pytest.raises(FileNotFoundError, match="requires casa.glb and house-plan.js"):
        build_runtime(repo, output, private_plan=private)
    assert not output.exists()

    model_path = str(NEXO_PRIVATE_PLAN_DIR / model.name)
    plan_path = str(NEXO_PRIVATE_PLAN_DIR / "house-plan.js")
    public = tmp_path / "public.tar.gz"
    manifest = build_runtime(repo, public)
    assert model_path not in manifest["files"] and plan_path not in manifest["files"]
    with tarfile.open(public) as archive:
        assert model_path not in archive.getnames() and plan_path not in archive.getnames()

    plan = b"export const SCAN_ROOMS = {};\n"
    (private / "house-plan.js").write_bytes(plan)
    with pytest.raises(ValueError, match="Invalid glTF 2.0 header"):
        build_runtime(repo, output, private_plan=private)
    assert not output.exists()

    gltf = b'{"asset":{"version":"2.0"}} '
    glb = struct.pack("<4sIII4s", b"glTF", 2, 20 + len(gltf), len(gltf), b"JSON") + gltf
    model.write_bytes(glb)
    (private / "house-picking-fixture.js").write_text("// private QA inputs")
    git("check-ignore", str(model), str(private / "house-plan.js"))

    cache = tmp_path / "shared-cache"
    monkeypatch.setenv(CACHE_DIR_ENV, str(cache))
    local = tmp_path / "local.tar.gz"
    local_manifest = cached_build_runtime(repo, local)
    assert not cache.exists()
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / model.name).write_bytes(glb)
    (explicit / "house-plan.js").write_bytes(plan)
    manifest = build_runtime(repo, output, private_plan=explicit)
    assert manifest == local_manifest
    with tarfile.open(output) as archive:
        assert json.load(archive.extractfile("release.json")) == manifest
        for name, data in ((model_path, glb), (plan_path, plan)):
            assert archive.extractfile(name).read() == data
            assert manifest["files"][name] == hashlib.sha256(data).hexdigest()
        assert set(name for name in archive.getnames() if name.startswith(f"{NEXO_PRIVATE_PLAN_DIR}/")) == {
            model_path, plan_path,
        }
