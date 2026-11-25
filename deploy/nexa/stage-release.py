"""Copy into private staging and validate the inventory before switching releases."""
import json
import os
from pathlib import Path
import re
import shutil
import sys

from release_policy import check_entry, digest


def copy_release(source, dest):
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"expected a directory: {source}")
    dest.mkdir()
    for entry in source.iterdir():
        target = dest / entry.name
        if entry.is_symlink():
            raise ValueError(f"unexpected symlink: {entry}")
        if entry.is_dir():
            copy_release(entry, target)
        elif entry.is_file():
            shutil.copyfile(entry, target)
            target.chmod(0o750 if os.access(entry, os.X_OK) else 0o640)
        else:
            raise ValueError(f"expected a regular file: {entry}")


def validate(root, version, test_root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("expected a release directory")
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise ValueError("missing regular SHA256SUMS")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"[0-9a-fA-F]{64} [ *](.+)", line)
        if not match:
            raise ValueError("invalid checksum entry")
        path = Path(match[1])
        if path.is_absolute() or ".." in path.parts or path in listed:
            raise ValueError(f"invalid checksum path: {path}")
        listed.add(path)

    actual = set()

    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, onerror=fail):
        for name in dirs + files:
            path = Path(directory) / name
            rel = path.relative_to(root)
            check_entry(rel)
            if path.is_symlink():
                raise ValueError(f"unexpected symlink: {rel}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"expected regular file: {rel}")
            if rel.parts[0] != "bin":
                digest(path, True)
            if path != sums:
                actual.add(rel)
    if actual != listed:
        raise ValueError("checksum inventory does not match release")
    manifest = json.loads((root / "release-manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != version:
        raise ValueError("version differs from release manifest")
    if not test_root and manifest.get("git_dirty") is True:
        raise ValueError("dirty release requires test mode")
    if not test_root and not any(path.parts[0] == "frontend" and len(path.parts) > 1 for path in actual):
        raise ValueError("partial release requires test mode")
    units = (path for path in actual if path.parent == Path("systemd") and not path.name.startswith("."))
    if not any(path.suffix in (".service", ".target") for path in units):
        raise ValueError("missing systemd units")


if __name__ == "__main__":
    try:
        src, dest, version, test_root = sys.argv[1:]
        if src != "--verify-installed":
            if Path(src).resolve() in Path(dest).resolve().parents:
                raise ValueError("release source contains staging directory")
            copy_release(Path(src), Path(dest))
        validate(Path(dest), version, test_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"error: {error}")
