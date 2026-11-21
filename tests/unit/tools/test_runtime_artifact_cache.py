"""Tests for content-addressed build_runtime artifact caching (T090-U2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import stat
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


def test_cached_output_is_immutable_0444_across_link_and_exdev(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))
    monkeypatch.setattr(cache_mod, "_git", lambda _cwd, *_args, **_kw: "d" * 40)

    linked = tmp_path / "linked.tar.gz"
    cache_mod.cached_build_runtime(tmp_path / "repo", linked, "HEAD")
    assert stat.S_IMODE(linked.stat().st_mode) == 0o444

    real_link = os.link
    exdev_calls = 0

    def fake_link(src, dst):
        nonlocal exdev_calls
        if exdev_calls == 0:
            exdev_calls += 1
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_link(src, dst)

    monkeypatch.setattr(cache_mod.os, "link", fake_link)
    copied = tmp_path / "copied.tar.gz"
    cache_mod.cached_build_runtime(tmp_path / "repo", copied, "HEAD")
    assert exdev_calls == 1
    assert stat.S_IMODE(copied.stat().st_mode) == 0o444
    assert copied.read_bytes() == linked.read_bytes()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "corrupt_sidecar",
    [
        "[]",
        "null",
        "42",
        json.dumps({"git_commit": "e" * 40, "python_magic": cache_mod.importlib.util.MAGIC_NUMBER.hex()}),
        "{broken json",
    ],
)
def test_corrupt_sidecar_or_tarball_rebuilds_cleanly(tmp_path, monkeypatch, corrupt_sidecar):
    calls: list[tuple[Path, Path, str]] = []
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(calls))
    monkeypatch.setattr(cache_mod, "_git", lambda _cwd, *_args, **_kw: "e" * 40)

    out1 = tmp_path / "one.tar.gz"
    m1 = cache_mod.cached_build_runtime(tmp_path / "repo", out1, "HEAD")
    assert len(calls) == 1

    key = f"{'e' * 40}-{cache_mod.importlib.util.MAGIC_NUMBER.hex()}"
    json_path = cache_dir / "by-commit" / f"{key}.manifest.json"
    os.chmod(json_path, 0o600)
    json_path.write_text(corrupt_sidecar, encoding="utf-8")

    out2 = tmp_path / "two.tar.gz"
    m2 = cache_mod.cached_build_runtime(tmp_path / "repo", out2, "HEAD")
    assert len(calls) == 2
    assert m2["call_index"] == 2 and m1["call_index"] == 1

    # Now corrupt the tarball bytes while leaving sidecars intact.
    tar_path = cache_dir / "by-commit" / f"{key}.tar.gz"
    meta_path = cache_dir / "by-commit" / f"{key}.meta.json"
    os.chmod(tar_path, 0o600)
    tar_path.write_bytes(b"corrupted tarball bytes")
    out3 = tmp_path / "three.tar.gz"
    m3 = cache_mod.cached_build_runtime(tmp_path / "repo", out3, "HEAD")
    assert len(calls) == 3
    assert m3["call_index"] == 3
    assert stat.S_IMODE(out3.stat().st_mode) == 0o444

    # Truncate a valid gzip tarball in half keeping the ORIGINAL meta -> rejected before tarfile.open.
    raw_tar = tar_path.read_bytes()
    truncated = raw_tar[: len(raw_tar) // 2]
    os.chmod(tar_path, 0o600)
    tar_path.write_bytes(truncated)

    real_tar_open = tarfile.open

    def _forbid_corrupt_cache_tar_read(name, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if Path(name) == tar_path and str(mode).startswith("r"):
            raise AssertionError("tarfile.open must not read cached tarball when digest/size already mismatches")
        return real_tar_open(name, *args, **kwargs)

    monkeypatch.setattr(cache_mod.tarfile, "open", _forbid_corrupt_cache_tar_read)
    out4_precheck = tmp_path / "four-precheck.tar.gz"
    m4_pre = cache_mod.cached_build_runtime(tmp_path / "repo", out4_precheck, "HEAD")
    assert len(calls) == 4
    assert m4_pre["call_index"] == 4
    monkeypatch.setattr(cache_mod.tarfile, "open", real_tar_open)

    # Now truncate AND adjust meta to reach gzip EOFError directly in tarfile.open and still recover.
    import hashlib
    raw_tar2 = tar_path.read_bytes()
    truncated2 = raw_tar2[: len(raw_tar2) // 2]
    os.chmod(tar_path, 0o600)
    tar_path.write_bytes(truncated2)
    os.chmod(meta_path, 0o600)
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_payload["tar_sha256"] = hashlib.sha256(truncated2).hexdigest()
    meta_payload["tar_size"] = len(truncated2)
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")

    out4 = tmp_path / "four.tar.gz"
    m4 = cache_mod.cached_build_runtime(tmp_path / "repo", out4, "HEAD")
    assert len(calls) == 5
    assert m4["call_index"] == 5
    assert stat.S_IMODE(out4.stat().st_mode) == 0o444


def test_prebuild_returns_only_cache_dir_without_launcher_bypass(tmp_path, monkeypatch):
    from tests.mocks import epoch4, launcher_rehearsal, slot_reconciliation

    prebuild_calls: list[str] = []
    built: list[tuple[Path, Path, str]] = []
    head_sha = "a" * 40
    x4_sha = "b" * 40
    pre_sha = "c" * 40
    y5_sha = "d" * 40
    y4_sha = "e" * 40
    compat_sha = "f" * 40

    monkeypatch.setattr(launcher_rehearsal, "overlay_commit", lambda *_a, **_kw: head_sha)
    monkeypatch.setattr(epoch4, "epoch4_commit", lambda _clone, rev: x4_sha if rev == head_sha else "9" * 40)
    monkeypatch.setattr(
        launcher_rehearsal,
        "pre_launcher_revision",
        lambda *_a, **_kw: prebuild_calls.append("pre_checked") or pre_sha,
    )
    monkeypatch.setattr(
        slot_reconciliation,
        "schema_compatible_revision",
        lambda repo, rev: prebuild_calls.append(f"schema_compat:{rev}") or compat_sha,
    )
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(built))

    def fake_git(_cwd, *args, **_kw):
        if "commit-tree" in args:
            return y5_sha if args[1].startswith(head_sha) else y4_sha
        target = args[-1].removesuffix("^{commit}")
        return head_sha if target == "HEAD" else target

    monkeypatch.setattr(cache_mod, "_git", fake_git)
    monkeypatch.setattr(
        cache_mod.subprocess,
        "run",
        lambda *_a, **_kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    env = cache_mod.prebuild_standard_artifacts(tmp_path / "repo", tmp_path / "cache")
    assert prebuild_calls == ["pre_checked", "schema_compat:HEAD"]
    assert {rev for _repo, _out, rev in built} == {head_sha, x4_sha, y5_sha, y4_sha, pre_sha, compat_sha}
    assert env == {cache_mod.CACHE_DIR_ENV: str((tmp_path / "cache").resolve())}
    assert "KZA_LAUNCHER_ARTIFACTS_CACHE" not in env
    assert "KZA_COMPACTION_ARTIFACT" not in env


def test_prebuild_warms_distinct_head_and_head_epoch4_when_overlay_differs(tmp_path, monkeypatch):
    from tests.mocks import epoch4, launcher_rehearsal, slot_reconciliation

    built: list[tuple[Path, Path, str]] = []
    epoch4_inputs: list[str] = []
    head_sha = "1" * 40
    x5_sha = "2" * 40
    x4_sha = "3" * 40
    head_epoch4_sha = "4" * 40
    y5_sha = "5" * 40
    y4_sha = "6" * 40
    pre_sha = "7" * 40
    compat_sha = "8" * 40

    monkeypatch.setattr(launcher_rehearsal, "overlay_commit", lambda *_a, **_kw: x5_sha)

    def fake_epoch4(_clone, rev):
        epoch4_inputs.append(rev)
        return x4_sha if rev == x5_sha else head_epoch4_sha

    monkeypatch.setattr(epoch4, "epoch4_commit", fake_epoch4)
    monkeypatch.setattr(launcher_rehearsal, "pre_launcher_revision", lambda *_a, **_kw: pre_sha)
    monkeypatch.setattr(slot_reconciliation, "schema_compatible_revision", lambda _repo, _rev: compat_sha)
    monkeypatch.setattr(cache_mod, "build_runtime", _fake_build(built))

    def fake_git(_cwd, *args, **_kw):
        if "commit-tree" in args:
            return y5_sha if args[1].startswith(x5_sha) else y4_sha
        target = args[-1].removesuffix("^{commit}")
        return head_sha if target == "HEAD" else target

    monkeypatch.setattr(cache_mod, "_git", fake_git)
    monkeypatch.setattr(
        cache_mod.subprocess,
        "run",
        lambda *_a, **_kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    cache_mod.prebuild_standard_artifacts(tmp_path / "repo", tmp_path / "cache")
    assert epoch4_inputs == [x5_sha, head_sha]
    assert {rev for _repo, _out, rev in built} == {
        head_sha,
        x5_sha,
        x4_sha,
        head_epoch4_sha,
        y5_sha,
        y4_sha,
        pre_sha,
        compat_sha,
    }


def test_slot_reconciliation_table_columns_memoizes_and_returns_copy(tmp_path, monkeypatch):
    from tests.mocks import slot_reconciliation as sr

    sr.clear_revision_caches()
    git_calls: list[tuple[Path, tuple[str, ...]]] = []
    commit_a = "1" * 40
    commit_b = "2" * 40
    head_target = [commit_a]

    def fake_check_output(cmd, *, cwd, text=True):
        git_calls.append((Path(cwd), tuple(cmd)))
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return head_target[0] + "\n"
        if cmd[:2] == ["git", "show"]:
            rev = cmd[2].split(":", 1)[0]
            if rev == commit_a:
                return '_TABLE_COLUMNS = {"privacy": "id:TEXT"}\n'
            return '_TABLE_COLUMNS = {"privacy": "id:TEXT owner:TEXT"}\n'
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(sr.subprocess, "check_output", fake_check_output)
    try:
        cols_first = sr._table_columns(tmp_path / "clone1", commit_a)
        cols_first["privacy"] = "mutated"
        cols_second = sr._table_columns(tmp_path / "clone2", commit_a)
        assert cols_second == {"privacy": "id:TEXT"}
        assert len(git_calls) == 1

        # Symbolic HEAD resolves dynamically and hits cache for commit_a, then misses once HEAD advances.
        assert sr._table_columns(tmp_path / "clone1", "HEAD") == {"privacy": "id:TEXT"}
        assert sum(1 for _cwd, cmd in git_calls if cmd[:2] == ("git", "show")) == 1

        head_target[0] = commit_b
        assert sr._table_columns(tmp_path / "clone1", "HEAD") == {"privacy": "id:TEXT owner:TEXT"}
        assert sum(1 for _cwd, cmd in git_calls if cmd[:2] == ("git", "show")) == 2
    finally:
        sr.clear_revision_caches()


def test_slot_reconciliation_schema_compatible_revision_memoizes_across_clones(tmp_path, monkeypatch):
    from tests.mocks import slot_reconciliation as sr
    from tests.mocks import slot_rollback as sb

    sr.clear_revision_caches()
    old_sha = "3" * 40
    mid_sha = "4" * 40
    new_sha = "5" * 40
    rev_list_calls = 0
    show_calls = 0
    managed_calls = 0

    def fake_check_output(cmd, *, cwd, text=True):
        nonlocal rev_list_calls, show_calls
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return new_sha + "\n"
        if cmd[:2] == ["git", "rev-list"]:
            rev_list_calls += 1
            return f"{old_sha}\n{mid_sha}\n{new_sha}\n"
        if cmd[:2] == ["git", "show"]:
            show_calls += 1
            rev = cmd[2].split(":", 1)[0]
            if rev == old_sha:
                return '_TABLE_COLUMNS = {"t": "v1"}\n'
            return '_TABLE_COLUMNS = {"t": "v2"}\n'
        raise AssertionError(f"unexpected git command: {cmd}")

    def fake_is_managed(_repo, rev):
        nonlocal managed_calls
        managed_calls += 1
        return rev in {old_sha, mid_sha, new_sha}

    monkeypatch.setattr(sr.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(sb, "is_managed_capable", fake_is_managed)
    try:
        assert sr.schema_compatible_revision(tmp_path / "clone1", new_sha) == mid_sha
        assert rev_list_calls == 1
        assert show_calls == 3  # new_sha, old_sha, mid_sha
        assert managed_calls == 2  # old_sha, mid_sha

        # Repeated calls across distinct shared clones or via HEAD hit the in-process cache.
        assert sr.schema_compatible_revision(tmp_path / "clone2", new_sha) == mid_sha
        assert sr.schema_compatible_revision(tmp_path / "clone3", "HEAD") == mid_sha
        assert rev_list_calls == 1
        assert show_calls == 3
        assert managed_calls == 2

        sr.clear_revision_caches()
        assert sr.schema_compatible_revision(tmp_path / "clone1", new_sha) == mid_sha
        assert rev_list_calls == 2
    finally:
        sr.clear_revision_caches()
