"""Simulate ext4 inode-number reuse on any platform.

ext4 hands a freed inode number to the next allocation at once: a file
unlinked and written again at the same name usually reports the old
(st_dev, st_ino). APFS never reuses one so soon. ``report_inode_reuse``
makes every stat call report the substitute inode with the original's
(st_dev, st_ino); all other fields (size, times) stay the substitute's own,
as they do on ext4.
"""
from __future__ import annotations

import os
from types import SimpleNamespace


def report_inode_reuse(monkeypatch, substitute: os.stat_result, original: os.stat_result) -> None:
    """Disguise ``substitute`` as ``original`` in os.stat, os.lstat and os.fstat."""
    real_stat, real_lstat, real_fstat = os.stat, os.lstat, os.fstat
    disguised = (substitute.st_dev, substitute.st_ino)

    def disguise(info):
        if (info.st_dev, info.st_ino) != disguised:
            return info
        fields = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        fields.update(st_dev=original.st_dev, st_ino=original.st_ino)
        return SimpleNamespace(**fields)

    monkeypatch.setattr(os, "stat", lambda path, *args, **kwargs: disguise(real_stat(path, *args, **kwargs)))
    monkeypatch.setattr(os, "lstat", lambda path, *args, **kwargs: disguise(real_lstat(path, *args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: disguise(real_fstat(descriptor)))


_REAL_LSTAT = os.lstat  # Captured at import, before any test patches os.lstat.


def substitute_same_stat(path, content: bytes | None = None) -> tuple[bytes, tuple, tuple]:
    """Unlink ``path`` and write another file at the name with the same size and mtime.

    ``content`` defaults to the original bytes inverted (same size, other
    content); pass the original bytes for a same-content rewrite. The mode,
    size and mtime_ns equal the original's (os.utime). The ctime cannot be
    set on Linux or macOS and stays the substitute's own: under simulated
    number reuse it is the only field that still differs, while on ext4 a
    held pin makes the real st_ino differ. ext4 timestamps are coarse (one
    clock tick), and the unlink itself sets the original inode's ctime, so
    the substitute's ctime could equal it. A probe file created after the
    substitute carries a ctime no earlier than that unlink; each os.utime
    sets the substitute's ctime to now, so it is re-applied until it is past
    the probe's. That wait is the one simulated part (on a real reuse the
    number would not differ either); the helper itself never holds the
    original open, so it never pins it.
    Returns the bytes written and the real (st_dev, st_ino) of the original
    and of the substitute.
    """
    original = _REAL_LSTAT(path)
    with open(path, "rb") as stream:
        raw = stream.read()
    data = bytes(byte ^ 0xFF for byte in raw) if content is None else content
    assert len(data) == original.st_size
    os.unlink(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, original.st_mode & 0o7777)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    probe = os.path.join(os.path.dirname(os.fspath(path)), ".kza-ctime-probe-%d" % os.getpid())
    os.close(os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
    try:
        unlinked_by = _REAL_LSTAT(probe).st_ctime_ns  # >= the original's ctime after its unlink.
        for _attempt in range(1_000_000):  # Until the coarse clock ticks: milliseconds, not a timeout.
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns), follow_symlinks=False)
            substitute = _REAL_LSTAT(path)
            if substitute.st_ctime_ns > unlinked_by:
                break
    finally:
        os.unlink(probe)
    assert substitute.st_ctime_ns > unlinked_by
    assert (substitute.st_size, substitute.st_mtime_ns) == (original.st_size, original.st_mtime_ns)
    return data, (original.st_dev, original.st_ino), (substitute.st_dev, substitute.st_ino)
