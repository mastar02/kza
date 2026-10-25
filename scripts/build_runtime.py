#!/usr/bin/env python3
"""Build a committed KZA revision into a source-free CPython 3.13 archive."""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import subprocess
import sys
import tarfile
import tempfile


ASSET_SUFFIXES = {".html", ".css", ".js", ".jsx", ".json", ".csv", ".svg",
                  ".png", ".jpg", ".ico", ".woff", ".woff2", ".ttf", ".wav"}


def build_runtime(repo: Path, output: Path, revision: str = "HEAD") -> dict:
    """Compile tracked runtime modules and bundle assets without runtime state.

    Only the selected Git revision is read. Configuration, environment files,
    dependencies, models and data remain external to the release archive.
    Existing output files are never replaced.
    """
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        raise ValueError("Build with CPython 3.13, matching the production runtime")
    repo = repo.resolve()
    output = output.resolve()
    if output == repo or repo.is_relative_to(output):
        raise ValueError("Output must be an archive file outside repository parents")
    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=repo, text=True,
    ).strip()
    timestamp = int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", commit], cwd=repo, text=True,
    ).strip())
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-rz", commit, "--", "src", "tools", "scripts"], cwd=repo,
    )
    files = {}
    with tempfile.TemporaryDirectory(prefix="kza-runtime-") as temporary:
        staging = Path(temporary)
        for entry in tracked.split(b"\0"):
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split()
            path = PurePosixPath(raw_name.decode())
            if kind != b"blob":
                continue
            if path == PurePosixPath("scripts/build_runtime.py"):
                continue
            python_module = path.suffix == ".py"
            frontend_asset = (
                path.is_relative_to("src/dashboard/frontend")
                and path.suffix in ASSET_SUFFIXES
            )
            package_asset = path.parts[0] == "src" and path.suffix == ".csv"
            if not (python_module or frontend_asset or package_asset):
                continue
            if mode == b"120000":
                raise ValueError(f"Runtime symlinks require explicit review: {path}")
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid runtime path: {path}")
            data = subprocess.check_output(["git", "cat-file", "blob", object_id], cwd=repo)
            destination = path.with_suffix(".pyc") if python_module else path
            if python_module:
                source = staging / str(path)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(data)
                compiled = source.with_suffix(".pyc")
                py_compile.compile(
                    str(source), cfile=str(compiled), dfile=str(path),
                    doraise=True, optimize=0,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
                data = compiled.read_bytes()
            files[str(destination)] = data

    if "src/main.pyc" not in files or "src/__init__.pyc" not in files:
        raise ValueError("Revision does not contain the KZA application")
    manifest = {
        "format": "kza-cpython-bytecode-v1", "git_commit": commit,
        "python": "3.13", "python_magic": importlib.util.MAGIC_NUMBER.hex(),
        "entrypoint": ["python", "-m", "src.main"],
        "external_paths": ["config", "data", "models", "logs"],
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(files.items())},
    }
    files["release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".kza-release-", dir=output.parent)
    temporary_output = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=timestamp) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for name, data in sorted(files.items()):
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = timestamp
                        archive.addfile(info, io.BytesIO(data))
        # Atomic publication that refuses a concurrent overwrite.
        os.link(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return manifest


def main() -> None:
    """Build an archive from the requested committed revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    manifest = build_runtime(Path(__file__).resolve().parents[1], args.output, args.revision)
    print(json.dumps({"archive": str(args.output), "commit": manifest["git_commit"],
                      "compiled_modules": sum(p.endswith(".pyc") for p in manifest["files"]),
                      "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
