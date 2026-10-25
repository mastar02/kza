#!/usr/bin/env python3
"""Prepare and select source-free runtime slots with independent state copies.

Offline tool: stop every writer before copying state or changing the pointer.
It never stops services, runs migrations or starts the application.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile


def _slot_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Invalid slot name")
    return name


@contextmanager
def _locked_root(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _read_archive(archive: Path) -> tuple[dict, dict[str, bytes]]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Runtime staging requires CPython 3.13")
    files = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if (
                not member.isfile() or path.is_absolute() or ".." in path.parts
                or str(path) != member.name or member.name in files
                or path.suffix == ".py"
                or (member.name != "release.json" and path.parts[0] not in {"src", "scripts", "tools"})
            ):
                raise ValueError("Unexpected or unsafe runtime archive member")
            files[member.name] = bundle.extractfile(member).read()
    manifest = json.loads(files.pop("release.json"))
    _validate_manifest(manifest, files)
    return manifest, files


def _validate_manifest(manifest: dict, files: dict[str, bytes]) -> None:
    if (
        manifest.get("format") != "kza-cpython-bytecode-v1"
        or manifest.get("python") != "3.13"
        or manifest.get("python_magic") != importlib.util.MAGIC_NUMBER.hex()
        or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_commit", ""))
        or manifest.get("external_paths") != ["config", "data", "models", "logs"]
        or not {"src/main.pyc", "src/__init__.pyc"} <= files.keys()
        or manifest.get("files") != {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        }
    ):
        raise ValueError("Runtime manifest or file integrity mismatch")


def _copy_private_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("State source must contain regular config and data directories")
    for entry in source.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError("State links and special files require separate review")
    shutil.copytree(source, destination)
    destination.chmod(0o700)


def prepare_slot(archive: Path, root: Path, name: str, state_source: Path) -> dict:
    """Verify an archive and copy offline config/data into a new private slot.

    Args:
        archive: Archive produced by build_runtime.py.
        root: Explicit staging root; no production path is assumed.
        name: New slot name, independent of the commit (allows repeated trials).
        state_source: Offline directory containing config/ and data/. Optional
            .env is copied privately. Models, logs and dependencies stay external.

    Returns:
        Runtime manifest, without the private state's contents.
    """
    name = _slot_name(name)
    root = root.resolve()
    state_source = state_source.resolve()
    # Do not recurse through an output nested in its own input state.
    if root.is_relative_to(state_source):
        raise ValueError("Staging root and state source must be separate trees")
    manifest, files = _read_archive(archive)
    with _locked_root(root):
        slots = root / "slots"
        slots.mkdir(exist_ok=True, mode=0o700)
        target = slots / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=slots) as temporary:
            staging = Path(temporary)
            runtime = staging / "runtime"
            runtime.mkdir()
            for filename, data in files.items():
                destination = runtime / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (runtime / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
            state = staging / "state"
            state.mkdir(mode=0o700)
            for directory in ("config", "data"):
                _copy_private_tree(state_source / directory, state / directory)
                (runtime / directory).symlink_to(Path("..") / "state" / directory)
            env_file = state_source / ".env"
            if env_file.exists() or env_file.is_symlink():
                if env_file.is_symlink() or not env_file.is_file():
                    raise ValueError("Environment file must be regular")
                shutil.copyfile(env_file, state / ".env")
                (state / ".env").chmod(0o600)
                (runtime / ".env").symlink_to(Path("..") / "state" / ".env")
            os.rename(staging, target)
    return manifest


def activate_slot(root: Path, name: str) -> str | None:
    """Atomically select a verified runtime and its associated state for restart.

    Args:
        root: Staging root whose service is stopped by the caller.
        name: Previously prepared slot. Selecting the old slot is rollback.

    Returns:
        Previous relative slot target, or None on first activation.
    """
    name = _slot_name(name)
    root = root.resolve()
    with _locked_root(root):
        target = root / "slots" / name
        if target.resolve() != target:
            raise ValueError("Slot must not redirect to another directory")
        runtime = target / "runtime"
        manifest = json.loads((runtime / "release.json").read_text())
        files = {}
        for filename in manifest["files"]:
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"src", "scripts", "tools"}:
                raise ValueError("Unsafe installed runtime path")
            installed = runtime / filename
            if installed.resolve() != installed or installed.suffix == ".py":
                raise ValueError("Installed runtime integrity mismatch")
            files[filename] = installed.read_bytes()
        _validate_manifest(manifest, files)
        expected_files = set(files) | {"release.json", "config", "data"}
        if (runtime / ".env").is_symlink():
            expected_files.add(".env")
            if (runtime / ".env").resolve() != target / "state" / ".env":
                raise ValueError("Environment does not belong to the same slot")
        actual_files = {
            str(path.relative_to(runtime)) for path in runtime.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_files != expected_files:
            raise ValueError("Unexpected installed runtime files")
        for directory in ("config", "data"):
            if (runtime / directory).resolve() != target / "state" / directory:
                raise ValueError("Runtime and state do not belong to the same slot")
            if not (runtime / directory).is_dir():
                raise ValueError("Missing slot state")
        current = root / "current"
        previous = os.readlink(current) if current.is_symlink() else None
        with tempfile.TemporaryDirectory(prefix=".activate-", dir=root) as temporary:
            link = Path(temporary) / "current"
            link.symlink_to(Path("slots") / name / "runtime")
            os.replace(link, current)
        return previous


def main() -> None:
    """Prepare or select a slot after the operator has stopped all state writers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", required=True,
                        help="Assert that all writers are stopped; this tool does not stop services")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("name")
    prepare.add_argument("--state-source", type=Path, required=True)
    activate = commands.add_parser("activate")
    activate.add_argument("name")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_slot(args.archive, args.root, args.name, args.state_source)
        print(json.dumps({"slot": args.name, "commit": manifest["git_commit"]}))
    else:
        previous = activate_slot(args.root, args.name)
        print(json.dumps({"slot": args.name, "previous": previous}))


if __name__ == "__main__":
    main()
