"""Private, sourced context snapshots with raw-capture expiry."""

import json
import logging
import os
import re
import time
import tempfile
import threading
from contextlib import nullcontext
from pathlib import Path

from src.orchestrator.context_manager import UserContext
from src.memory.fact_ledger import MemorySource, sources_current

logger = logging.getLogger(__name__)

PERSISTED_VERSION = 2

# Path-traversal guard: user_id is concatenated into base_path/<user_id>.json,
# so we reject anything that could escape the directory ('..', '/', NUL,
# whitespace, leading dots). Only [A-Za-z0-9_-] is allowed.
# DO NOT relax without auditing every callsite of self._path().
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class ContextPersister:
    """Saves and loads UserContext snapshots to disk.

    Plan #2 OpenClaw — see docs/superpowers/specs/2026-04-28-openclaw-context-compaction-design.md

    New snapshots live in base_path/policy_v2/<user_id>.json. Legacy files
    remain untouched and are never hydrated. Writes use private unique
    temporary files and an atomic replace, serialized against revocations.

    Persisted fields (subset of UserContext): version, user_id, user_name,
    last_seen, session_count, compacted_summary, preserved_ids, summary_sources.
    Sources determine the deadline; snapshot/access time never renews it.

    NOT persisted: conversation_history (literal turns die at session
    boundary — assumed already compacted into summary before snapshot),
    preferences (managed by UserManager elsewhere), compaction_inflight
    (transient mutex flag).

    Error contract:
    - save(): raises ValueError on unsafe user_id, propagates OSError on
      I/O failures (caller should log + continue, NOT crash).
    - load(): returns None for any failure mode (missing file, unsafe
      user_id, JSON decode error, version mismatch, OS error). Callers
      treat None as "no prior context" and create fresh.
    - Corrupt JSON files are quarantined to <user_id>.json.corrupt-<ts>
      to prevent next snapshot from overwriting them.
    """

    def __init__(self, base_path: Path | str = Path("data/contexts"), *, source_policy=None, clock=None):
        self.legacy_base_path = Path(base_path)
        self.base_path = self.legacy_base_path / "policy_v2"
        self.source_policy = source_policy
        self._clock = time.time if clock is None else clock
        self._lock = threading.RLock()

    def _sources(self, payload: dict) -> tuple[MemorySource, ...]:
        return tuple(MemorySource.from_dict(s) for s in payload.get("summary_sources", []))

    def _available(self, payload: dict) -> bool:
        if not payload.get("compacted_summary"):
            return not payload.get("preserved_ids")
        sources = self._sources(payload)
        return sources_current(sources, self._clock()) and (self.source_policy is None or self.source_policy.sources_available(sources))

    def _validate_user_id(self, user_id: str) -> None:
        if not _SAFE_USER_ID.match(user_id):
            raise ValueError(
                f"Unsafe user_id for filesystem path: {user_id!r}. "
                "Allowed: alphanumeric, underscore, hyphen."
            )

    def _path(self, user_id: str) -> Path:
        return self.base_path / f"{user_id}.json"

    def exists(self, user_id: str) -> bool:
        """Check if a snapshot exists for this user_id.

        Returns False (silently) for unsafe user_ids — does NOT log,
        unlike load(). This asymmetry is intentional: callers usually
        follow exists() with load() which logs the same condition.
        """
        try:
            self._validate_user_id(user_id)
        except ValueError:
            return False
        return self._path(user_id).is_file()

    def save(self, ctx: UserContext) -> None:
        """Atomic snapshot to base_path/<user_id>.json.

        Writes to a private temporary sibling and replaces it, so a crash
        mid-write cannot leave a half-written file. The .tmp is unlinked
        on failure (debug-logged if cleanup fails).

        Delegates to build_payload() + save_payload() — useful when the
        caller can build an immutable snapshot under lock and release
        before disk I/O.

        Args:
            ctx: UserContext to snapshot. Must have a safe user_id.

        Raises:
            ValueError: if ctx.user_id contains unsafe characters.
            OSError: filesystem errors (caller should log + continue).
        """
        payload = self.build_payload(ctx)
        self.save_payload(payload)

    def build_payload(self, ctx: UserContext) -> dict:
        """Convert UserContext → persisted JSON shape. Pure, no I/O.

        Public API for cross-module callers (e.g., ContextManager) that
        need to build a snapshot under lock and then call save_payload
        outside the lock. Keep in sync with the load() reader contract.
        """
        ctx.prune()
        return {
            "version": PERSISTED_VERSION,
            "user_id": ctx.user_id,
            "user_name": ctx.user_name,
            "last_seen": time.time(),
            "session_count": ctx.session_count,
            "compacted_summary": ctx.compacted_summary,
            "preserved_ids": list(ctx.preserved_ids),
            "summary_sources": [s.to_dict() for s in ctx.summary_sources],
        }

    def save_payload(self, payload: dict) -> None:
        """Atomic write of an already-built payload dict.

        Revalidates sources and expiry at commit. Callers building snapshots
        under a lock can release that in-memory lock before disk I/O.

        Args:
            payload: dict with at minimum 'user_id'. Must already include
                'version' and 'last_seen' (use build_payload to construct).

        Raises:
            ValueError: if payload['user_id'] is unsafe or missing.
            OSError: filesystem errors.
        """
        user_id = payload.get("user_id")
        if not isinstance(user_id, str):
            raise ValueError(f"payload missing string user_id: {user_id!r}")
        self._validate_user_id(user_id)
        if payload.get("version") != PERSISTED_VERSION or not self._available(payload):
            raise PermissionError("context_snapshot_sources_unavailable")
        sources = self._sources(payload)
        guard = self.source_policy.admit_source_write(sources) if self.source_policy is not None and sources else nullcontext()
        with self._lock, guard:
            self.base_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, filename = tempfile.mkstemp(prefix=".snapshot-", dir=self.base_path)
            tmp = Path(filename)
            try:
                with os.fdopen(descriptor, "w") as output:
                    output.write(json.dumps(payload, ensure_ascii=False, indent=2))
                if not self._available(payload):
                    raise PermissionError("context_snapshot_sources_unavailable")
                os.replace(tmp, self._path(user_id))
            finally:
                tmp.unlink(missing_ok=True)

    def load(self, user_id: str) -> dict | None:
        """Load a snapshot if it exists and is valid.

        Args:
            user_id: ID to look up.

        Returns:
            The persisted payload dict, OR None if any of:
            - file does not exist
            - user_id is unsafe (logged warning)
            - JSON is corrupt (file is quarantined, logged error)
            - version mismatch with PERSISTED_VERSION (logged warning)
            - other OS error (logged warning)
        """
        with self._lock:
            return self._load(user_id)

    def _load(self, user_id: str) -> dict | None:
        try:
            self._validate_user_id(user_id)
        except ValueError as e:
            logger.warning(f"[ContextPersister] invalid user_id on load: {e}")
            return None

        path = self._path(user_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            quarantine = path.with_suffix(f".json.corrupt-{int(time.time())}")
            try:
                os.replace(path, quarantine)
                logger.error(
                    f"[ContextPersister] CORRUPT JSON for user={user_id}; "
                    f"quarantined to {quarantine.name}: {e}"
                )
            except OSError as rename_err:
                logger.error(
                    f"[ContextPersister] CORRUPT JSON for user={user_id} AND quarantine "
                    f"failed ({rename_err}); leaving file in place: {e}"
                )
            return None
        except OSError as e:
            logger.warning(f"[ContextPersister] unreadable JSON for {user_id}: {e}")
            return None

        if not isinstance(data, dict):
            return None
        stored_version = data.get("version")
        if stored_version != PERSISTED_VERSION:
            logger.warning(
                f"[ContextPersister] version mismatch for {user_id}: "
                f"stored={stored_version} expected={PERSISTED_VERSION}. Treating as no prior context."
            )
            return None
        try:
            if data.get("user_id") != user_id or not self._available(data):
                return None
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        return data

    def purge_expired(self) -> int:
        """Remove only expired/revoked policy snapshots; preserve legacy files."""
        removed = 0
        with self._lock:
            # The single policy writer holds this same lock until its actual
            # write exits. Uncommitted leftovers are never recoverable context.
            for temporary in self.base_path.glob(".snapshot-*"):
                temporary.unlink()
                removed += 1
            for path in self.base_path.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    if isinstance(data, dict) and data.get("version") == PERSISTED_VERSION and not self._available(data):
                        path.unlink()
                        removed += 1
                except (ValueError, TypeError, KeyError, AttributeError):
                    # Unknown/corrupt payloads are not accepted as live context.
                    continue
        return removed
