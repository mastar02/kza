"""Unified content-addressed cache for build_runtime test artifacts (T090-U2).

Reduces redundant CPython 3.13 bytecode compilations across rehearsals and tests
without changing build_runtime.py or any runtime security/reproducibility guarantee.

Keyed strictly by ``<40-char-commit-sha>-<python_magic>``, synchronized with
``fcntl.flock`` so parallel ``pytest-xdist`` workers never rebuild the same
immutable revision simultaneously.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zlib
from pathlib import Path

from scripts.build_runtime import build_runtime

CACHE_DIR_ENV = "KZA_RUNTIME_ARTIFACT_CACHE_DIR"
_EXPECTED_FORMAT = "kza-cpython-bytecode-v1"
_IMMUTABLE_TAR_MODE = 0o444

# Fixed historical revisions reused by rehearsal/upgrade/compaction fixtures.
PINNED_HISTORICAL_REVISIONS: tuple[str, ...] = (
    "HEAD",
    "40aa05b8d56ffb608e79f3c26b55a9386a9a0605",  # PRE_LAUNCHER_COMMIT
    "b0866e8",                                   # LEGACY_STATE_COMMIT
    "e9669ec",                                   # LEGACY_CONTAINER_COMMIT
    "26de262",                                   # PINNED_EPOCH3 / EPOCH3_R1
    "b0c322d~1",                                 # EARLY_COPY (AUTHZ3-1)
    "b0c322d",                                   # FIRST_LOCKED
    "357dcb5",                                   # R1_COMMIT
    "b15ce86a1ea7e1926e11c7fdb6c67f6831a9d3c1",  # PRE_SC1A
    "80b47f55716e980b93a7312b6732de0611e3e2e0",  # slot_rollback bridge
    "0fd3d166f1b17726f8ef998b320f37a42ff11d81",  # test_stage_runtime old archive
    "238b463",                                   # _PRE_0B
)


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, env=env, stderr=subprocess.PIPE
    ).strip()


def _materialize_archive(cached_tar: Path, output: Path) -> None:
    """Place ``cached_tar`` at ``output`` as an immutable read-only file without overwriting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if (cached_tar.stat().st_mode & 0o777) != _IMMUTABLE_TAR_MODE:
        os.chmod(cached_tar, _IMMUTABLE_TAR_MODE)
    try:
        os.link(cached_tar, output)
        return
    except FileExistsError:
        raise
    except OSError:
        # Cross-filesystem fallback (e.g. /dev/shm cache -> disk tmp_path).
        fd, tmp_name = tempfile.mkstemp(prefix=".kza-cache-copy-", dir=output.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as dst, cached_tar.open("rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.chmod(tmp_path, _IMMUTABLE_TAR_MODE)
            os.link(tmp_path, output)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _tar_sha256_and_verify_release(
    tar_path: Path,
    *,
    expected_manifest: dict,
    commit: str,
    magic: str,
    expected_sha: str | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    """Compute SHA-256 and size of ``tar_path`` before opening gzip/tar to verify ``release.json``."""
    hasher = hashlib.sha256()
    size = 0
    with tar_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    actual_sha = hasher.hexdigest()
    if expected_size is not None and size != expected_size:
        raise ValueError("cached tarball size mismatch")
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("cached tarball digest mismatch")
    with tarfile.open(tar_path, "r:gz") as archive:
        member = archive.getmember("release.json")
        if not member.isfile():
            raise ValueError("release.json is not a regular file inside cached tarball")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("unable to read release.json from cached tarball")
        embedded = json.loads(extracted.read().decode("utf-8"))
    if not isinstance(embedded, dict):
        raise ValueError("embedded release.json is not a JSON object")
    if embedded != expected_manifest:
        raise ValueError("embedded release.json differs from sidecar manifest")
    if embedded.get("git_commit") != commit or embedded.get("python_magic") != magic:
        raise ValueError("embedded release.json commit/magic mismatch")
    return actual_sha, size


def _is_valid_manifest(manifest: object, *, commit: str, magic: str) -> bool:
    return (
        isinstance(manifest, dict)
        and manifest.get("format") == _EXPECTED_FORMAT
        and manifest.get("git_commit") == commit
        and manifest.get("python_magic") == magic
        and isinstance(manifest.get("files"), dict)
    )


def _unlink_cache_entry(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _load_verified_cache_entry(
    cache_tar: Path,
    cache_json: Path,
    cache_meta: Path,
    *,
    commit: str,
    magic: str,
) -> dict | None:
    """Validate cached tarball, manifest JSON and digest metadata under lock."""
    paths = (cache_tar, cache_json, cache_meta)
    if any(p.is_symlink() or not p.is_file() for p in paths):
        return None
    try:
        loaded_manifest = json.loads(cache_json.read_text(encoding="utf-8"))
        loaded_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if not _is_valid_manifest(loaded_manifest, commit=commit, magic=magic):
            raise ValueError("invalid cached manifest schema")
        if not isinstance(loaded_meta, dict):
            raise ValueError("invalid cached meta schema")
        expected_sha = loaded_meta.get("tar_sha256")
        expected_size = loaded_meta.get("tar_size")
        if (
            loaded_meta.get("git_commit") != commit
            or loaded_meta.get("python_magic") != magic
            or not isinstance(expected_sha, str)
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
        ):
            raise ValueError("invalid cached meta identity")
        actual_sha, actual_size = _tar_sha256_and_verify_release(
            cache_tar,
            expected_manifest=loaded_manifest,
            expected_sha=expected_sha,
            expected_size=expected_size,
            commit=commit,
            magic=magic,
        )
        if actual_sha != expected_sha or actual_size != expected_size:
            raise ValueError("cached tarball digest/size mismatch")
        if (cache_tar.stat().st_mode & 0o777) != _IMMUTABLE_TAR_MODE:
            os.chmod(cache_tar, _IMMUTABLE_TAR_MODE)
        return loaded_manifest
    except (
        OSError,
        EOFError,
        zlib.error,
        json.JSONDecodeError,
        tarfile.TarError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ):
        _unlink_cache_entry(cache_tar, cache_json, cache_meta)
        return None


def cached_build_runtime(repo: Path, output: Path, revision: str = "HEAD") -> dict:
    """Build or reuse a content-addressed ``build_runtime`` archive when cache env is active.

    Falls back directly to :func:`scripts.build_runtime.build_runtime` whenever
    ``KZA_RUNTIME_ARTIFACT_CACHE_DIR`` is unset or empty.
    """
    cache_raw = os.environ.get(CACHE_DIR_ENV, "").strip()
    if not cache_raw:
        return build_runtime(repo, output, revision)

    repo_resolved = Path(repo).resolve()
    output_resolved = Path(output).resolve()
    if output_resolved == repo_resolved or repo_resolved.is_relative_to(output_resolved):
        raise ValueError("Output must be an archive file outside repository parents")
    if output_resolved.exists():
        raise FileExistsError(output_resolved)

    commit = _git(repo_resolved, "rev-parse", "--verify", revision + "^{commit}")
    magic = importlib.util.MAGIC_NUMBER.hex()
    key = f"{commit}-{magic}"

    cache_root = Path(cache_raw).resolve()
    by_commit = cache_root / "by-commit"
    by_commit.mkdir(parents=True, exist_ok=True)
    cache_tar = by_commit / f"{key}.tar.gz"
    cache_json = by_commit / f"{key}.manifest.json"
    cache_meta = by_commit / f"{key}.meta.json"
    lock_path = by_commit / f"{key}.lock"

    manifest: dict | None = None
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            manifest = _load_verified_cache_entry(
                cache_tar, cache_json, cache_meta, commit=commit, magic=magic
            )

            if manifest is None:
                with tempfile.TemporaryDirectory(prefix=f".build-{key[:12]}-", dir=by_commit) as tmp_dir:
                    tmp_dir_path = Path(tmp_dir)
                    tmp_tar = tmp_dir_path / "runtime.tar.gz"
                    manifest = build_runtime(repo_resolved, tmp_tar, commit)
                    if not _is_valid_manifest(manifest, commit=commit, magic=magic):
                        raise ValueError("build_runtime returned invalid manifest for cached key")
                    tar_sha256, tar_size = _tar_sha256_and_verify_release(
                        tmp_tar, expected_manifest=manifest, commit=commit, magic=magic
                    )
                    tmp_json = tmp_dir_path / "manifest.json"
                    tmp_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    tmp_meta = tmp_dir_path / "meta.json"
                    meta_payload = {
                        "git_commit": commit,
                        "python_magic": magic,
                        "tar_sha256": tar_sha256,
                        "tar_size": tar_size,
                    }
                    tmp_meta.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
                    os.chmod(tmp_tar, _IMMUTABLE_TAR_MODE)
                    os.chmod(tmp_json, _IMMUTABLE_TAR_MODE)
                    os.chmod(tmp_meta, _IMMUTABLE_TAR_MODE)
                    os.replace(tmp_json, cache_json)
                    os.replace(tmp_meta, cache_meta)
                    os.replace(tmp_tar, cache_tar)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    _materialize_archive(cache_tar, output_resolved)
    return copy.deepcopy(manifest)


def working_tree_digest(repo: Path) -> str:
    """Compute deterministic SHA-256 of HEAD plus uncommitted scripts/ and src/ changes."""
    repo = Path(repo).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "scripts", "src"], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "scripts", "src"],
        cwd=repo,
    ).splitlines()
    hasher = hashlib.sha256()
    hasher.update(head.encode())
    hasher.update(b"\0")
    hasher.update(diff)
    for rel_bytes in sorted(untracked):
        rel = rel_bytes.decode()
        path = repo / rel
        hasher.update(b"\0" + rel.encode() + b"\0")
        if path.is_file():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def prebuild_standard_artifacts(repo: Path, cache_dir: Path) -> dict[str, str]:
    """Pre-populate ``by-commit`` cache with current X5/X4/Y5 overlay and PRE_LAUNCHER_COMMIT.

    Unlike mutant-only bypass env vars, this returns only ``KZA_RUNTIME_ARTIFACT_CACHE_DIR``
    so test fixtures still execute their own Git ancestry, ``LAUNCHER_CONTRACT`` and
    commit-matching assertions while hitting immutable verified tarballs in ``by-commit/``.
    """
    from tests.mocks import epoch4, launcher_rehearsal

    repo = Path(repo).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    previous_env = os.environ.get(CACHE_DIR_ENV)
    os.environ[CACHE_DIR_ENV] = str(cache_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="kza-prebuild-clone-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            clone = tmp_path / "clone"
            subprocess.run(
                ["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)],
                check=True,
                timeout=120,
            )
            x5 = launcher_rehearsal.overlay_commit(repo, clone, message="u3b-0d rehearsal")
            x4 = epoch4.epoch4_commit(clone, x5)
            environment = dict(
                os.environ,
                GIT_AUTHOR_NAME="rehearsal",
                GIT_AUTHOR_EMAIL="rehearsal@invalid",
                GIT_COMMITTER_NAME="rehearsal",
                GIT_COMMITTER_EMAIL="rehearsal@invalid",
                GIT_AUTHOR_DATE="2026-11-12T00:00:01Z",
                GIT_COMMITTER_DATE="2026-11-12T00:00:01Z",
            )
            y5 = _git(clone, "commit-tree", x5 + "^{tree}", "-p", x5, "-m", "u3b-0d rehearsal Y", env=environment)
            pre_commit = launcher_rehearsal.pre_launcher_revision(clone, x5)

            revisions: list[tuple[str, str]] = [
                ("x5.tar.gz", x5),
                ("x4.tar.gz", x4),
                ("y5.tar.gz", y5),
                ("pre.tar.gz", pre_commit),
            ]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(cached_build_runtime, clone, tmp_path / filename, rev)
                    for filename, rev in revisions
                ]
                for fut in futures:
                    fut.result()
    finally:
        if previous_env is None:
            os.environ.pop(CACHE_DIR_ENV, None)
        else:
            os.environ[CACHE_DIR_ENV] = previous_env

    return {CACHE_DIR_ENV: str(cache_dir)}
