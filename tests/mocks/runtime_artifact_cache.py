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
import subprocess
import tempfile
from pathlib import Path

from scripts.build_runtime import build_runtime

CACHE_DIR_ENV = "KZA_RUNTIME_ARTIFACT_CACHE_DIR"

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
    """Place ``cached_tar`` at ``output`` without overwriting an existing file."""
    output.parent.mkdir(parents=True, exist_ok=True)
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
            os.link(tmp_path, output)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


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
    lock_path = by_commit / f"{key}.lock"

    manifest: dict | None = None
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if cache_tar.is_file() and cache_json.is_file():
                try:
                    loaded = json.loads(cache_json.read_text())
                    if loaded.get("git_commit") == commit and loaded.get("python_magic") == magic:
                        manifest = loaded
                except (OSError, json.JSONDecodeError):
                    manifest = None

            if manifest is None:
                with tempfile.TemporaryDirectory(prefix=f".build-{key[:12]}-", dir=by_commit) as tmp_dir:
                    tmp_dir_path = Path(tmp_dir)
                    tmp_tar = tmp_dir_path / "runtime.tar.gz"
                    manifest = build_runtime(repo_resolved, tmp_tar, commit)
                    tmp_json = tmp_dir_path / "manifest.json"
                    tmp_json.write_text(json.dumps(manifest, indent=2) + "\n")
                    os.replace(tmp_json, cache_json)
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
    """Pre-populate cache with current X5/X4/Y5 overlay and pinned historical revisions.

    Returns environment variables that wire existing compaction and launcher helpers
    to the prebuilt artifacts for the exact current working tree.
    """
    from tests.mocks import epoch4, launcher_rehearsal

    repo = Path(repo).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    wt_dir = cache_dir / f"wt-{working_tree_digest(repo)}"
    wt_dir.mkdir(parents=True, exist_ok=True)

    previous_env = os.environ.get(CACHE_DIR_ENV)
    os.environ[CACHE_DIR_ENV] = str(cache_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="kza-prebuild-clone-") as tmp_dir:
            clone = Path(tmp_dir) / "clone"
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

            x5_tar = wt_dir / "x5.tar.gz"
            x4_tar = wt_dir / "x4.tar.gz"
            if not x5_tar.exists():
                cached_build_runtime(clone, x5_tar, x5)
            if not x4_tar.exists():
                cached_build_runtime(clone, x4_tar, x4)

            launcher_cache = wt_dir / "launcher-cache"
            launcher_cache.mkdir(parents=True, exist_ok=True)
            pre_commit = launcher_rehearsal.PRE_LAUNCHER_COMMIT
            has_pre = subprocess.run(
                ["git", "cat-file", "-e", pre_commit + "^{commit}"],
                cwd=clone,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            launcher_commits = {"x": x5, "y": y5}
            if has_pre:
                launcher_commits["pre"] = pre_commit

            # Prebuild only the active tree artifacts + PRE_LAUNCHER_COMMIT in parallel;
            # rarer historical commits stay lazy via cached_build_runtime into by-commit/.
            build_jobs: list[tuple[Path, str]] = [
                (x5_tar, x5),
                (x4_tar, x4),
                (launcher_cache / "y.tar.gz", y5),
            ]
            if has_pre:
                build_jobs.append((launcher_cache / "pre.tar.gz", pre_commit))
            with ThreadPoolExecutor(max_workers=min(4, len(build_jobs))) as pool:
                futures = [
                    pool.submit(cached_build_runtime, clone, dest, rev)
                    for dest, rev in build_jobs
                    if not dest.exists()
                ]
                for fut in futures:
                    fut.result()

            x_launcher = launcher_cache / "x.tar.gz"
            if not x_launcher.exists():
                cached_build_runtime(clone, x_launcher, x5)
            commits_json = launcher_cache / "commits.json"
            if not commits_json.exists():
                commits_json.write_text(json.dumps(launcher_commits, indent=2) + "\n")
    finally:
        if previous_env is None:
            os.environ.pop(CACHE_DIR_ENV, None)
        else:
            os.environ[CACHE_DIR_ENV] = previous_env

    return {
        CACHE_DIR_ENV: str(cache_dir),
        "KZA_COMPACTION_ARTIFACT": str(x5_tar),
        "KZA_COMPACTION_ARTIFACT_EPOCH4": str(x4_tar),
        "KZA_LAUNCHER_ARTIFACTS_CACHE": str(launcher_cache),
    }
