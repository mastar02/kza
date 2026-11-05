"""Durable browser recorder reservations and explicit retirement receipts."""
from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import threading
from contextvars import Context
from enum import Enum
from typing import TYPE_CHECKING, Callable

from src.core.owned_work import OwnedBlockingWork

if TYPE_CHECKING:
    from src.memory.capture_control import CaptureController
    from src.memory.privacy_services import PrivacyServices


class RecorderState(str, Enum):
    """Permission already issued is never inferred retired from silence."""

    RESERVED = "reserved"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class CaptureRecorders:
    """Retain durable browser owners until an explicit stop acknowledgement.

    Args:
        privacy: Actual private SQLite service and its retained I/O owner.
        capture_allowed: Current global admission, independent of actor roles.
    """

    def __init__(self, privacy: PrivacyServices, *,
                 capture_allowed: Callable[[], bool]) -> None:
        self._privacy = privacy
        # CaptureControl retains this borrower through privacy shutdown. Keep
        # admitted SQL usable until its initialization and retirement finish;
        # closing public privacy admission does not retire this owner.
        self._work = OwnedBlockingWork(worker=privacy._offload)
        self._capture_allowed = capture_allowed
        self._lock = threading.RLock()
        self._records: dict[str, RecorderState] = {}
        self._kinds: dict[str, str] = {}
        self._actors: dict[str, CaptureController] = {}
        self._epochs: dict[str, int] = {}
        self._operations: set[asyncio.Task] = set()
        self._initializing: asyncio.Task | None = None
        self._ready = False
        self._closed = False
        self._sealed = False
        self._fault = False
        self._generation = 0

    async def _call(self, function: Callable, *args: object) -> object:
        task = asyncio.create_task(self._work.run(function, *args), context=Context())
        self._operations.add(task)

        def finished(done: asyncio.Task) -> None:
            self._operations.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    def _read_cache(self) -> None:
        with self._privacy.ledger._lock:
            rows = self._privacy.ledger._db.execute(
                "SELECT token_hash, state, kind FROM capture_browser_recorders").fetchall()
            records = {str(row[0]): RecorderState(row[1]) for row in rows}
            kinds = {str(row[0]): str(row[2]) for row in rows}
            if any(kind not in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics") for kind in kinds.values()):
                raise RuntimeError("capture_recorder_kind_unconfirmed")
            with self._lock:
                self._records = records
                self._kinds = kinds
                self._actors = {key: actor for key, actor in self._actors.items()
                                if key in self._records}
                self._epochs = {key: epoch for key, epoch in self._epochs.items()
                                if key in self._records}

    def _initialize(self) -> None:
        try:
            with self._privacy.ledger._transaction() as db:
                db.execute("CREATE TABLE IF NOT EXISTS capture_browser_recorders ("
                           "token_hash TEXT PRIMARY KEY, state TEXT NOT NULL, "
                           "kind TEXT NOT NULL DEFAULT 'browser')")
                columns = {row[1] for row in db.execute(
                    "PRAGMA table_info(capture_browser_recorders)").fetchall()}
                if "kind" not in columns:
                    db.execute("ALTER TABLE capture_browser_recorders "
                               "ADD COLUMN kind TEXT NOT NULL DEFAULT 'browser'")
                db.execute("UPDATE capture_browser_recorders SET state='unknown' WHERE state='active'")
            self._read_cache()
            self._ready = True
        except BaseException:
            self._fault = True
            raise

    async def initialize(self) -> None:
        """Load durable unknown owners without awaiting disconnected browsers."""
        if self._closed:
            raise RuntimeError("capture_recorders_unavailable")
        if self._initializing is None:
            self._initializing = asyncio.create_task(self._call(self._initialize), context=Context())
            self._initializing.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._initializing)

    @staticmethod
    def _key(token: str) -> str:
        if type(token) is not str or re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None:
            raise ValueError("invalid_capture_recorder_token")
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    @staticmethod
    def _actor_current(actor: CaptureController) -> bool:
        try:
            actor.require_current()
        except (PermissionError, ValueError, RuntimeError):
            return False
        return True

    @staticmethod
    def _validate_kind(kind: str) -> None:
        if kind not in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics"):
            raise ValueError("capture_recorder_kind_denied")

    def _require_kind(self, key: str, kind: str) -> None:
        self._validate_kind(kind)
        with self._lock:
            actual = self._kinds.get(key)
            if actual is not None and actual != kind:
                raise ValueError("capture_recorder_kind_denied")

    def _admission(self, actor: CaptureController, key: str | None = None) -> None:
        if not self._actor_current(actor):
            raise ValueError("capture_recorder_actor_retired")
        with self._lock:
            original = self._actors.get(key) if key is not None else None
            available = self._ready and not self._closed and not self._sealed and not self._fault
            if key is not None and self._epochs.get(key) != self._generation:
                available = False
        if key is not None and (original is None or not self._actor_current(original)):
            raise ValueError("capture_recorder_actor_retired")
        if not available or self._capture_allowed() is not True:
            raise ValueError("capture_muted")

    def _register(self, key: str, actor: CaptureController, kind: str) -> None:
        self._validate_kind(kind)
        self._admission(actor)
        with self._lock:
            epoch = self._generation
        try:
            with self._privacy.ledger._transaction(lambda: self._admission(actor)) as db:
                if db.execute("SELECT COUNT(*) FROM capture_browser_recorders").fetchone()[0] >= 256:
                    raise ValueError("capture_recorders_full")
                db.execute("INSERT INTO capture_browser_recorders(token_hash,state,kind) VALUES (?,?,?)",
                           (key, RecorderState.RESERVED.value, kind))
            with self._lock:
                self._actors[key] = actor
                self._epochs[key] = epoch
            self._read_cache()
        except ValueError:
            raise
        except BaseException:
            self._fault = True
            raise

    async def register(self, actor: CaptureController, *, kind: str = "browser") -> dict[str, object]:
        """Reserve an owner; this response grants no native capture permission."""
        self._validate_kind(kind)
        token = secrets.token_urlsafe(32)
        await self._call(self._register, self._key(token), actor, kind)
        result = {"token": token, "state": RecorderState.RESERVED.value,
                  "capture_allowed": False, "generation": self._generation}
        if kind != "browser":
            result["kind"] = kind
        return result

    def _activate(self, key: str, actor: CaptureController, kind: str) -> None:
        self._require_kind(key, kind)
        self._admission(actor, key)
        with self._lock:
            state = self._records.get(key)
        if state not in (RecorderState.RESERVED, RecorderState.ACTIVE):
            raise ValueError("capture_recorder_not_reserved")
        try:
            with self._privacy.ledger._transaction(lambda: self._admission(actor, key)) as db:
                db.execute("UPDATE capture_browser_recorders SET state='active' "
                           "WHERE token_hash=? AND kind=?", (key, kind))
            self._read_cache()
        except ValueError:
            raise
        except BaseException:
            self._fault = True
            raise

    async def activate(self, token: str, actor: CaptureController, *,
                       kind: str = "browser") -> dict[str, object]:
        """Commit ACTIVE before granting permission to acquire audio."""
        self._validate_kind(kind)
        key = self._key(token)
        await self._call(self._activate, key, actor, kind)
        return await self.poll(token, actor, kind=kind)

    async def poll(self, token: str, actor: CaptureController, *,
                   kind: str = "browser") -> dict[str, object]:
        """Read policy for the original actor; silence is never a stop receipt."""
        key = self._key(token)
        self._require_kind(key, kind)
        with self._lock:
            state = self._records.get(key)
        allowed = state is RecorderState.ACTIVE
        try:
            self._admission(actor, key)
        except ValueError:
            allowed = False
        result = {"state": state.value if state is not None else "retired",
                  "capture_allowed": allowed, "retirement_required": not allowed,
                  "generation": self._generation}
        if kind != "browser":
            result["kind"] = kind
        return result

    def _retire(self, key: str, kind: str) -> None:
        self._validate_kind(kind)
        try:
            with self._privacy.ledger._transaction() as db:
                row = db.execute("SELECT kind FROM capture_browser_recorders "
                                 "WHERE token_hash=?", (key,)).fetchone()
                if row is not None and row[0] != kind:
                    raise ValueError("capture_recorder_kind_denied")
                db.execute("DELETE FROM capture_browser_recorders WHERE token_hash=? AND kind=?",
                           (key, kind))
            self._read_cache()
        except ValueError:
            raise
        except BaseException:
            self._fault = True
            raise

    async def retire(self, token: str, *, kind: str = "browser") -> dict[str, bool]:
        """Accept only this token's retirement, including after actor logout."""
        self._validate_kind(kind)
        if self._closed:
            raise RuntimeError("capture_recorders_unavailable")
        await self._call(self._retire, self._key(token), kind)
        return {"retired": True}

    def seal(self) -> None:
        """Immediately deny activation and ask all issued owners to retire."""
        with self._lock:
            self._sealed = True
            self._generation += 1

    def resume(self) -> None:
        """Permit new reservations only; never revive an old unknown owner."""
        with self._lock:
            if self._closed or self._fault:
                raise RuntimeError("capture_recorders_unavailable")
            self._sealed = False
            self._generation += 1

    def pending(self, *, kind: str | None = None) -> int:
        """Count a kind's records, or all real records/work for the ACK barrier."""
        if kind is not None:
            self._validate_kind(kind)
        with self._lock:
            if kind is not None:
                return sum(value == kind for value in self._kinds.values())
            return (len(self._records) + len(self._operations)
                    + self._work.status()['operations'] + int(self._fault))

    def _retire_reserved(self) -> None:
        try:
            with self._privacy.ledger._transaction() as db:
                db.execute("DELETE FROM capture_browser_recorders WHERE state='reserved'")
            self._read_cache()
        except BaseException:
            self._fault = True
            raise

    async def drain(self) -> None:
        """Wait for issued owners' explicit ACKs; reserve-only owners may retire."""
        self.seal()
        while self._operations:
            await asyncio.shield(asyncio.gather(*tuple(self._operations), return_exceptions=True))
        await self._work.drain_admitted()
        if self._fault:
            raise RuntimeError("capture_recorder_state_unconfirmed")
        await self._call(self._retire_reserved)
        while self.pending():
            if self._fault:
                raise RuntimeError("capture_recorder_state_unconfirmed")
            await asyncio.sleep(0.025)

    async def aclose(self) -> None:
        """Drain local SQL only; persist remote owners without inventing an ACK."""
        self._closed = True
        self.seal()
        failures = []
        if self._initializing is not None:
            failures.extend(await asyncio.gather(asyncio.shield(self._initializing), return_exceptions=True))
        while self._operations:
            failures.extend(await asyncio.shield(asyncio.gather(*tuple(self._operations), return_exceptions=True)))
        await asyncio.shield(self._work.begin_close())
        if self._fault or any(isinstance(result, BaseException) for result in failures):
            raise RuntimeError("capture_recorder_state_unconfirmed")
