"""Physical snapshot locations and shared barriers for their actual writers."""

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import stat
import threading

from src.memory.erasure_journal import ErasureParticipant, ErasureResult
from src.memory.fact_ledger import MemorySource

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_USER = re.compile(r"^[A-Za-z0-9_-]+$")
_TEMP = re.compile(r"^\.snapshot-[a-z0-9_]{8}$")
MAX_SNAPSHOT_BYTES = 1024 * 1024


def snapshot_lock(path: str | Path):
    """Share a physical barrier across persister objects and cleanup workers."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.RLock())


def snapshot_targets(base_path: str | Path) -> tuple[dict, ...]:
    """Resolve each actual policy directory before an optional writer opens it."""
    root = Path(base_path)
    return tuple(
        {
            "kind": "snapshots",
            "path": str((root / f"policy_v{version}").resolve()),
            "format": version,
        }
        for version in (2, 3)
    )


def register_snapshots(ledger, base_path: str | Path) -> tuple[dict, ...]:
    """Persist historical policy directories independently of feature enablement."""
    targets = snapshot_targets(base_path)
    if ledger is not None:
        for descriptor in targets:
            ledger.erasures.register_target(ErasureParticipant.CONTEXTS, 1, descriptor)
    return targets


def snapshot_sources(payload: dict) -> tuple[MemorySource, ...]:
    """Reject unknown provenance instead of interpreting arbitrary JSON as empty."""
    values = payload.get("summary_sources", [])
    if type(values) is not list or len(values) > 256:
        raise ValueError("unknown_context_sources")
    sources = tuple(MemorySource.from_dict(value) for value in values)
    if not sources and (
        payload.get("compacted_summary") or payload.get("preserved_ids")
    ):
        raise ValueError("unattributed_context_summary")
    return sources


@dataclass(frozen=True)
class ContextSweep:
    """Content-free native progress and the original source deadline."""

    checked_at: float
    cursor: str | None = None
    next_at: float | None = None
    deleted: int = 0
    pending: int = 0


def validate_snapshot_descriptor(descriptor: dict) -> None:
    """Reject unknown protocols before inspecting a physical directory."""
    if (
        set(descriptor) != {"kind", "path", "format"}
        or descriptor["kind"] != "snapshots"
        or type(descriptor["format"]) is not int
        or descriptor["format"] not in (2, 3)
        or type(descriptor["path"]) is not str
        or not Path(descriptor["path"]).is_absolute()
    ):
        raise ValueError("unsupported_context_snapshot_target")


def sweep_snapshots(ledger, descriptor: dict, after: str | None) -> ContextSweep:
    """Expire at most 32 policy entries without initializing optional features."""
    validate_snapshot_descriptor(descriptor)
    if not ledger.authority_available:
        raise RuntimeError("context_retention_authority_unavailable")
    if after is not None and (
        len(after) != 64 or any(c not in "0123456789abcdef" for c in after)
    ):
        raise ValueError("invalid_context_cursor")
    root = Path(descriptor["path"])
    with snapshot_lock(root):
        now = ledger.clock()
        try:
            info = root.lstat()
        except FileNotFoundError:
            return ContextSweep(now)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("unknown_context_directory")

        def selected():
            with os.scandir(root) as entries:
                for entry in entries:
                    key = hashlib.sha256(os.fsencode(entry.name)).hexdigest()
                    if after is None or key > after:
                        yield key, entry.name

        batch = heapq.nsmallest(32, selected())
        deleted = pending = 0
        deadlines = []
        for _, name in batch:
            path = root / name
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("unknown_context_file_type")
                if _TEMP.fullmatch(name):
                    path.unlink()
                    deleted += 1
                    continue
                if (
                    not name.endswith(".json")
                    or not _USER.fullmatch(name[:-5])
                    or info.st_size > MAX_SNAPSHOT_BYTES
                ):
                    raise ValueError("unknown_context_file_protocol")
                payload = json.loads(path.read_text())
                if (
                    type(payload) is not dict
                    or payload.get("version") != descriptor["format"]
                    or payload.get("user_id") != name[:-5]
                ):
                    raise ValueError("unknown_context_snapshot")
                sources = snapshot_sources(payload)
                if not sources:
                    raise ValueError("unknown_context_deadline")
                deadline = min(source.expires_at for source in sources)
                if deadline <= now or ledger.sources_committed_unavailable(sources):
                    path.unlink()
                    deleted += 1
                else:
                    deadlines.append(deadline)
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                # An unattributable entry must not starve a later attributable
                # one. It remains present and explicitly pending for review.
                pending += 1
        return ContextSweep(
            now,
            batch[-1][0] if len(batch) == 32 else None,
            min(deadlines) if deadlines else None,
            deleted,
            pending,
        )


def cleanup_snapshots(
    ledger, descriptor: dict, conversation_id: str, after: str | None
) -> ErasureResult:
    """Erase attributable policy snapshots in batches under the writer barrier."""
    validate_snapshot_descriptor(descriptor)
    if not ledger.authority_available or not ledger.conversation_revoked(
        conversation_id
    ):
        raise RuntimeError("context_erasure_authority_unavailable")
    root = Path(descriptor["path"])
    if after is not None and (
        len(after) != 64 or any(c not in "0123456789abcdef" for c in after)
    ):
        raise ValueError("invalid_context_cursor")
    with snapshot_lock(root):
        try:
            info = root.lstat()
        except FileNotFoundError:
            return ErasureResult(True, pending_count=0)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("unknown_context_directory")

        def selected():
            with os.scandir(root) as entries:
                for entry in entries:
                    key = hashlib.sha256(os.fsencode(entry.name)).hexdigest()
                    if after is None or key > after:
                        yield key, entry.name

        batch = heapq.nsmallest(32, selected())
        now = ledger.clock()
        for _, name in batch:
            path = root / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError("unknown_context_file_type")
            if _TEMP.fullmatch(name):
                path.unlink()
                continue
            if not name.endswith(".json") or not _USER.fullmatch(name[:-5]):
                raise ValueError("unknown_context_file_protocol")
            if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                raise ValueError("context_snapshot_size_limit")
            payload = json.loads(path.read_text())
            if (
                type(payload) is not dict
                or payload.get("version") != descriptor["format"]
                or payload.get("user_id") != name[:-5]
            ):
                raise ValueError("unknown_context_snapshot")
            sources = snapshot_sources(payload)
            if any(
                s.conversation_id == conversation_id or s.expires_at <= now
                for s in sources
            ) or ledger.sources_committed_unavailable(sources):
                path.unlink()
        return ErasureResult(
            len(batch) < 32,
            batch[-1][0] if len(batch) == 32 else None,
            0 if len(batch) < 32 else None,
        )
