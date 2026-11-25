"""Copy a closed set of regular files to private staging before checksum verification."""
import json
import os
from pathlib import Path
import re
import shutil
import sys


def stage(src, dest, version, test_root):
    sums = src / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise ValueError("missing regular SHA256SUMS")
    dest.mkdir()
    shutil.copyfile(sums, dest / sums.name)
    listed = set()
    for line in (dest / sums.name).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"[0-9a-fA-F]{64} [ *](.+)", line)
        if not match:
            raise ValueError("invalid checksum entry")
        path = Path(match[1])
        if path.is_absolute() or ".." in path.parts or path in listed:
            raise ValueError(f"invalid checksum path: {path}")
        listed.add(path)

    copied = set()

    def fail(error):
        raise error

    for root, dirs, files in os.walk(src, onerror=fail):
        directory = Path(root)
        rel = directory.relative_to(src)
        for name in dirs:
            if (directory / name).is_symlink():
                raise ValueError(f"unexpected symlink: {rel / name}")
            (dest / rel / name).mkdir()
        for name in files:
            path = directory / name
            if path == sums:
                continue
            if path.is_symlink() or not path.is_file() or rel / name not in listed:
                raise ValueError(f"unexpected file: {rel / name}")
            shutil.copyfile(path, dest / rel / name)
            (dest / rel / name).chmod(0o750 if os.access(path, os.X_OK) else 0o640)
            copied.add(rel / name)
    if copied != listed:
        raise ValueError("checksum inventory does not match release")
    manifest = dest / "release-manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["version"] != version:
            raise ValueError("version differs from release manifest")
        if not test_root and not (dest / "frontend").is_dir():
            raise ValueError("partial release requires test mode")


if __name__ == "__main__":
    try:
        stage(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"error: {error}")
