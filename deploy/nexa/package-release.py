#!/usr/bin/env python3
"""Assemble a native Nexa release from prebuilt artifacts."""
import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

DEPLOY = Path(__file__).resolve().parent
ROOT = DEPLOY.parents[1]
BINARIES = ("nexa-audiod", "nexa-brain", "nexa-memoryd", "nexa-web")
ML_IGNORE = shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", ".pytest_cache", "tests")


def native_executable(path):
    """Check ELF type or Mach-O filetype, including universal Mach-O slices."""
    if not os.access(path, os.X_OK):
        return False
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 52:
            return False
        magic = header[:4].hex()
        if magic == "7f454c46":
            endian = {1: "little", 2: "big"}.get(header[5])
            return (endian is not None and header[4] in (1, 2)
                    and int.from_bytes(header[16:18], endian) in (2, 3)
                    and int.from_bytes(header[24:28 if header[4] == 1 else 32], endian) != 0)
        if magic in {"cafebabe", "cafebabf", "bebafeca", "bfbafeca"}:
            endian = "big" if magic.startswith("cafe") else "little"
            width = 4 if magic in {"cafebabe", "bebafeca"} else 8
            offset = int.from_bytes(header[16:16 + width], endian)
            if offset >= path.stat().st_size:
                return False
            stream.seek(offset)
            header = stream.read(32)
            magic = header[:4].hex()
        endian = {"feedface": "big", "feedfacf": "big", "cefaedfe": "little", "cffaedfe": "little"}.get(magic)
        return endian is not None and int.from_bytes(header[12:16], endian) == 2


def copy_file(source, dest):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"expected a regular file: {source}")
    shutil.copy(source, dest)


def copy_tree(source, dest, ignore=None):
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"expected a directory: {source}")
    dest.mkdir()
    excluded = ignore(source, os.listdir(source)) if ignore else set()
    for child in source.iterdir():
        if child.name in excluded:
            continue
        if child.is_dir():
            copy_tree(child, dest / child.name, ignore)
        else:
            copy_file(child, dest / child.name)


def digest(path, check_content):
    checksum = hashlib.sha256()
    needle = b"/home/kza"
    tail = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if check_content and needle in tail + chunk:
                raise ValueError(f"private host path in {path}")
            tail = chunk[-(len(needle) - 1):]
            checksum.update(chunk)
    return checksum.hexdigest()


def assemble(args, stage):
    (stage / "bin").mkdir()
    for name in BINARIES:
        source = args.target_dir / name
        dest = stage / "bin" / name
        copy_file(source, dest)
        if not native_executable(dest):
            raise ValueError(f"expected a native executable: {source}")
    copy_tree(args.ml_src, stage / "ml", ML_IGNORE)
    if args.frontend_dist.exists() or args.frontend_dist.is_symlink() or not args.allow_missing_frontend:
        copy_tree(args.frontend_dist, stage / "frontend")
    (stage / "systemd").mkdir()
    for suffix in (".service", ".target"):
        units = list(args.systemd_dir.glob("*" + suffix))
        if not units:
            raise ValueError(f"no {suffix} units in {args.systemd_dir}")
        for unit in units:
            copy_file(unit, stage / "systemd" / unit.name)

    sums = {}
    for path in stage.rglob("*"):
        rel = path.relative_to(stage).as_posix()
        if any(c in rel for c in "\r\n\\"):
            raise ValueError(f"unsupported checksum path: {rel!r}")
        if any(fnmatch.fnmatchcase(path.name.lower(), pat)
               for pat in (".env", "*.glb", "*.db", "*.sqlite*")):
            raise ValueError(f"forbidden release entry: {rel}")
        if path.is_file():
            sums[rel] = digest(path, not rel.startswith("bin/"))
    manifest = {
        "version": args.version,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": sorted(sums),
    }
    path = stage / "release-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums[path.name] = digest(path, False)
    (stage / "SHA256SUMS").write_text(
        "".join(f"{sums[name]}  {name}\n" for name in sorted(sums)), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("out", type=Path)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "target/release")
    parser.add_argument("--frontend-dist", type=Path, default=ROOT / "frontend/dist")
    parser.add_argument("--ml-src", type=Path, default=ROOT / "workers/nexa-ml")
    parser.add_argument("--systemd-dir", type=Path, default=DEPLOY / "systemd")
    parser.add_argument("--allow-missing-frontend", action="store_true", help="partial CI/dry run only")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.version) or ".." in args.version or args.version == ".":
            raise ValueError("invalid version")
        out = args.out.absolute()
        if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
            raise ValueError("output must be absent or an empty directory")
        for source in (args.frontend_dist, args.ml_src, args.systemd_dir, args.target_dir):
            if source.resolve() in (out.resolve(), *out.resolve().parents):
                raise ValueError("output directory overlaps an input")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            stage = Path(tmp) / "release"
            stage.mkdir()
            assemble(args, stage)
            stage.replace(out)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
