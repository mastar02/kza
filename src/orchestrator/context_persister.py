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
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class ContextPersister:
    """Saves and loads UserContext snapshots to disk.

    Format: one JSON file per user at base_path/<user_id>.json. Writes are
    atomic via .tmp + os.replace. Reads return None for missing/corrupt
    files (logged as warning) — callers treat that as "no prior context".
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
        try:
            self._validate_user_id(user_id)
        except ValueError:
            return False
        return self._path(user_id).is_file()

    def save(self, ctx: UserContext) -> None:
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
            except OSError:
                pass
            raise

    def load(self, user_id: str) -> dict | None:
        try:
            self._validate_user_id(user_id)
        except ValueError as e:
            logger.warning(f"[ContextPersister] invalid user_id on load: {e}")
            return None

        path = self._path(user_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ContextPersister] corrupt or unreadable JSON for {user_id}: {e}")
            return None
