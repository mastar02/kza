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
