"""Test scaffolding for prepared runtimes, which prepare_slot makes read-only.

U3b-0e1: prepare_slot sets files 0444 and directories 0555. Fixtures that
add test content to a prepared runtime (or simulate an owner-level
tampering) first give the owner write access with ``writable_runtime``.
They call it before any observation, so the integrity checks under test see
an ordinary tree. tests/unit/tools/test_runtime_read_only.py covers the
read-only property itself.
"""
import os
from pathlib import Path
import stat


def writable_runtime(runtime: Path) -> Path:
    """Add owner write to every directory and regular file; links untouched."""
    runtime = Path(runtime)
    for _directory, _names, files, descriptor in os.fwalk(runtime, follow_symlinks=False):
        os.fchmod(descriptor, stat.S_IMODE(os.fstat(descriptor).st_mode) | stat.S_IRWXU)
        for name in files:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                os.chmod(name, stat.S_IMODE(info.st_mode) | stat.S_IWUSR, dir_fd=descriptor)
    return runtime
