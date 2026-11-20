"""Tests for content-addressed build_runtime artifact caching (T090-U2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tarfile

import pytest

from tests.mocks import runtime_artifact_cache as cache_mod


def _fake_build(calls: list[tuple[Path, Path, str]]):
    def fake_build_runtime(repo: Path, output: Path, revision: str = "HEAD") -> dict:
        if output.exists():
            raise FileExistsError(output)
        calls.append((Path(repo), Path(output), revision))
        commit = (
            revision
            if len(revision) == 40
            else ("a" * 40 if revision == "HEAD" or revision.startswith("a") else "b" * 40)
        )
        manifest = {
            "format": "kza-cpython-bytecode-v1",
            "git_commit": commit,
            "python": "3.13",
            "python_magic": cache_mod.importlib.util.MAGIC_NUMBER.hex(),
            "entrypoint": ["python", "-m", "src.main"],
            "external_paths": ["config", "data", "models", "logs"],
            "files": {"release.json": "0" * 64, "src/main.pyc": "1" * 64},
            "call_index": len(calls),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tf:
            import io
            data = json.dumps(manifest).encode()
            ti = tarfile.TarInfo("release.json")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        return manifest

    return fake_build_runtime


def test_bypasses_cache_when_env_unset(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.delenv(cache_mod.CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))

    out1 = tmp_path / "one.tar.gz"
    out2 = tmp_path / "two.tar.gz"
    cache_mod.cached_build_runtime(tmp_path, out1, "HEAD")
    cache_mod.cached_build_runtime(tmp_path, out2, "HEAD")

    assert len(calls) == 2


def test_caches_by_commit_and_reuses_bytes_and_manifest(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path, str]] = []
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))
    monkeypatch.setattr(
        cache_mod,
        "_git",
        lambda _cwd, *args, **_kw: ("a" * 40 if args[-1].startswith("HEAD") else "b" * 40),
    )

    out1 = tmp_path / "out1" / "r.tar.gz"
    out2 = tmp_path / "out2" / "r.tar.gz"
    out3 = tmp_path / "out3" / "r.tar.gz"

    m1 = cache_mod.cached_build_runtime(tmp_path / "repo", out1, "HEAD")
    m2 = cache_mod.cached_build_runtime(tmp_path / "repo", out2, "HEAD")
    m3 = cache_mod.cached_build_runtime(tmp_path / "repo", out3, "other-rev")

    assert len(calls) == 2
    assert m1["git_commit"] == m2["git_commit"] == "a" * 40
    assert m2 == m1 and m2 is not m1
    assert out1.read_bytes() == out2.read_bytes()
    assert m3["git_commit"] == "b" * 40


def test_refuses_existing_output_on_cache_hit(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))
    monkeypatch.setattr(cache_mod, "_git", lambda _cwd, *_args, **_kw: "a" * 40)

    existing = tmp_path / "already.tar.gz"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        cache_mod.cached_build_runtime(tmp_path / "repo", existing, "HEAD")
    assert existing.read_bytes() == b"keep"
    assert calls == []


def test_concurrent_workers_share_lock_and_build_once(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))
    monkeypatch.setattr(cache_mod, "_git", lambda _cwd, *_args, **_kw: "c" * 40)

    outputs = [tmp_path / f"w{i}" / "r.tar.gz" for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda p: cache_mod.cached_build_runtime(tmp_path / "repo", p, "HEAD"), outputs))

    assert len(calls) == 1
    payload = outputs[0].read_bytes()
    assert all(out.read_bytes() == payload for out in outputs)
    assert all(res["git_commit"] == "c" * 40 for res in results)
