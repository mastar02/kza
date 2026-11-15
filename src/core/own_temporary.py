"""Own mkstemp temporaries whose cleanup deletes only the own inode.

A cleanup that unlinks its temporary *name* in a ``finally`` would delete a
file someone substituted at that name. The allocation is therefore pinned: a
duplicate of the mkstemp descriptor stays open until cleanup decides, so the
own inode cannot be freed and its (st_dev, st_ino) cannot name any other file
(ext4 hands a freed number to the next allocation at once). Cleanup unlinks
the name only while its lstat is that pinned inode; anything else stays.

Stdlib only. ``scripts/stage_runtime.py`` and ``scripts/build_runtime.py``
keep their own copies of these few lines because they are loaded by path and
must not import the application package.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator

logger = logging.getLogger(__name__)


def remove_own_temporary(name: str | os.PathLike, pin: int | None) -> bool:
    """Unlink ``name`` only while it is the pinned own inode.

    The comparison is sound only while ``pin`` is still open, so the caller
    closes the pin after this returns, never before.

    Args:
        name: The own temporary's path.
        pin: An open descriptor of the own inode, or None when unpinned.

    Returns:
        True when the name is absent or was the own inode and is now
        unlinked; False when it was retained (foreign or unproven).
    """
    try:
        info = os.lstat(name)
    except FileNotFoundError:
        return True
    if pin is None:
        logger.warning("[OwnTemporary] Unpinned temporary retained: %s", os.fspath(name))
        return False
    held = os.fstat(pin)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (held.st_dev, held.st_ino):
        logger.warning("[OwnTemporary] Foreign file at the temporary name retained: %s", os.fspath(name))
        return False
    Path(name).unlink(missing_ok=True)
    return True


@contextmanager
def own_temporary(directory: str | os.PathLike, *, prefix: str,
                  suffix: str = "") -> Iterator[tuple[int, str]]:
    """Allocate a pinned mkstemp temporary and clean up only the own inode.

    Yields the raw mkstemp descriptor (the caller owns and closes it, e.g.
    through ``os.fdopen``) and the name. On exit the name is unlinked only if
    it is still the own inode; after a successful ``os.replace`` it is
    normally absent and nothing is unlinked. If the pin cannot be duplicated
    the empty temporary is removed through the mkstemp descriptor and the
    error is raised before the body runs.

    Args:
        directory: Directory for the temporary.
        prefix: mkstemp prefix.
        suffix: mkstemp suffix.

    Yields:
        ``(descriptor, name)`` of the new temporary.
    """
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    try:
        pin = os.dup(descriptor)
    except OSError:
        # The still-open mkstemp descriptor is itself the pin: remove the own
        # empty temporary through it, then close it once.
        try:
            remove_own_temporary(name, descriptor)
        finally:
            os.close(descriptor)
        raise
    try:
        yield descriptor, name
    finally:
        try:
            remove_own_temporary(name, pin)
        finally:
            os.close(pin)
