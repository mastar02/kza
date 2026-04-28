"""ContextPersister — atomic JSON snapshot of UserContext per user_id."""

import json
import logging
import os
import re
import time
from pathlib import Path

from src.orchestrator.context_manager import UserContext

logger = logging.getLogger(__name__)

PERSISTED_VERSION = 1

# Path-traversal guard: user_id is concatenated into base_path/<user_id>.json,
# so we reject anything that could escape the directory ('..', '/', NUL,
# whitespace, leading dots). Only [A-Za-z0-9_-] is allowed.
# DO NOT relax without auditing every callsite of self._path().
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class ContextPersister:
    """Saves and loads UserContext snapshots to disk.

    Plan #2 OpenClaw — see docs/superpowers/specs/2026-04-28-openclaw-context-compaction-design.md

    Format: one JSON file per user at base_path/<user_id>.json. Writes are
    atomic via .tmp + os.replace.

    Persisted fields (subset of UserContext): version, user_id, user_name,
    last_seen, session_count, compacted_summary, preserved_ids.

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

    def __init__(self, base_path: Path | str = Path("data/contexts")):
        self.base_path = Path(base_path)

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

        Writes to a .tmp sibling and os.replace's into place, so a crash
        mid-write cannot leave a half-written file. The .tmp is unlinked
        on failure (debug-logged if cleanup fails).

        Args:
            ctx: UserContext to snapshot. Must have a safe user_id.

        Raises:
            ValueError: if ctx.user_id contains unsafe characters.
            OSError: filesystem errors (caller should log + continue).
        """
        self._validate_user_id(ctx.user_id)
        self.base_path.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": PERSISTED_VERSION,
            "user_id": ctx.user_id,
            "user_name": ctx.user_name,
            "last_seen": time.time(),
            "session_count": ctx.session_count,
            "compacted_summary": ctx.compacted_summary,
            "preserved_ids": list(ctx.preserved_ids),
        }

        target = self._path(ctx.user_id)
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            os.replace(tmp, target)
            logger.info(
                f"[ContextPersister] saved user={ctx.user_id} "
                f"summary_chars={len(ctx.compacted_summary or '')} "
                f"preserved_ids={len(ctx.preserved_ids)}"
            )
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError as cleanup_err:
                logger.debug(f"[ContextPersister] could not remove tmp {tmp}: {cleanup_err}")
            raise

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

        stored_version = data.get("version")
        if stored_version != PERSISTED_VERSION:
            logger.warning(
                f"[ContextPersister] version mismatch for {user_id}: "
                f"stored={stored_version} expected={PERSISTED_VERSION}. Treating as no prior context."
            )
            return None
        return data
