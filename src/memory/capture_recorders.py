"""Durable browser recorder reservations and explicit retirement receipts."""
from __future__ import annotations

import asyncio
import hashlib
import logging
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

logger = logging.getLogger(__name__)
# Owners issued by this process that may wait for an explicit ACK at once.
LIVE_OWNER_CAPACITY = 256


class RecorderState(str, Enum):
    """Permission already issued is never inferred retired from silence."""

    RESERVED = "reserved"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class CaptureRecorders:
    """Retain durable browser owners until an explicit stop acknowledgement.

    Owners already ``unknown`` when this process initialized are *carried*
    (R5V-1): their token lived only in a page or native client of an earlier
    process, and the server grants them nothing. The server cannot tell a lost
    token from a holder that outlived the restart, so drains do not wait for a
    carried owner. That relies on the client fail-closed monitors: a native
    client polls every 0.25 s with a 1 s timeout and a browser owner every
    0.5 s with a 5 s timeout, and each seals its capture when a poll fails or
    reports ``retirement_required``. A carried owner is never counted as
    retired: ``pending()`` keeps it until an explicit ``retire``.

    A holder that presents its token to ``poll`` or ``activate`` (matching
    kind, before ``aclose``) proves it is alive: its key leaves the carried
    set and drains wait for its ``retire`` again, which the client sends after
    seeing ``retirement_required``. Owners issued by this process keep the full
    explicit-ACK barrier.

    A carried owner leaves ``pending()`` only through its ``retire`` or
    ``forget_carried`` (R1-B2, user decision 2026-11-15): an administrator
    affirms that no tab or tool is still recording and confirms the carried
    count the page showed. Forgetting deletes only rows still carried; a
    holder that presents its token before the commit aborts it.

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
        # Set once per process in ``_initialize``; never grows afterwards and
        # shrinks only when a holder presents its token (``_reclaim``).
        self._carried: set[str] = set()
        # Rows a forget_carried transaction is deleting, and whether one of
        # their holders presented its token meanwhile (which aborts it).
        self._forgetting: frozenset[str] = frozenset()
        self._forget_contested = False
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
            if any(kind not in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics", "zone_setup", "manual_voice_command") for kind in kinds.values()):
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
                carried = frozenset(str(row[0]) for row in db.execute(
                    "SELECT token_hash FROM capture_browser_recorders WHERE state='unknown'"))
            with self._lock:
                self._carried = set(carried)
            self._read_cache()
            if carried:
                logger.warning("[CaptureRecorders] %d unacknowledged owner(s) carried from an "
                               "earlier process; never counted as retired", len(carried))
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
        if kind not in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics", "zone_setup", "manual_voice_command"):
            raise ValueError("capture_recorder_kind_denied")

    def _require_kind(self, key: str, kind: str) -> None:
        self._validate_kind(kind)
        with self._lock:
            actual = self._kinds.get(key)
            if actual is not None and actual != kind:
                raise ValueError("capture_recorder_kind_denied")

    def _reclaim(self, key: str) -> None:
        """A carried owner whose token was presented is alive: await it again."""
        with self._lock:
            if self._closed or key not in self._carried:
                return
            self._carried.discard(key)
            if key in self._forgetting:
                self._forget_contested = True
        logger.info("[CaptureRecorders] a carried owner contacted this process; "
                    "its explicit retire is awaited again")

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
                rows = db.execute("SELECT token_hash FROM capture_browser_recorders").fetchall()
                with self._lock:
                    carried = frozenset(self._carried)
                if sum(str(row[0]) not in carried for row in rows) >= LIVE_OWNER_CAPACITY:
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
        self._reclaim(key)
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
        self._reclaim(key)
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
        """Count a kind's records, or all records/work; carried owners included.

        This is the confirmation count: a carried owner never sent its ACK.
        """
        if kind is not None:
            self._validate_kind(kind)
        with self._lock:
            if kind is not None:
                return sum(value == kind for value in self._kinds.values())
            return (len(self._records) + len(self._operations)
                    + self._work.status()['operations'] + int(self._fault))

    def carried(self) -> int:
        """Count carried owners still unretired and never heard from here."""
        with self._lock:
            return sum(key in self._carried for key in self._records)

    def live_owners(self) -> int:
        """Count issued owners a drain will wait for: not reserved, not carried."""
        with self._lock:
            return sum(state is not RecorderState.RESERVED and key not in self._carried
                       for key, state in self._records.items())

    def pending_live(self) -> int:
        """Count what a drain waits for: everything except carried owners."""
        with self._lock:
            return (sum(key not in self._carried for key in self._records)
                    + len(self._operations) + self._work.status()['operations']
                    + int(self._fault))

    def _forget_carried(self, expected: int, authorize: Callable[[], None]) -> int:
        def guard() -> None:
            authorize()
            with self._lock:
                if self._forget_contested:
                    raise ValueError("capture_carried_count_conflict")

        # R1B2-2: the ledger lock (reentrant) spans the whole forget, its
        # cleanup included. A second forget on the other privacy worker only
        # starts once this one reset ``_forgetting``, so it can never clear
        # the set another forget installed and miss a returning holder.
        with self._privacy.ledger._lock:
            try:
                with self._privacy.ledger._transaction(guard) as db:
                    rows = db.execute("SELECT token_hash FROM capture_browser_recorders "
                                      "WHERE state='unknown'").fetchall()
                    with self._lock:
                        keys = frozenset(str(row[0]) for row in rows) & self._carried
                        if keys and len(keys) != expected:
                            raise ValueError("capture_carried_count_conflict")
                        self._forgetting, self._forget_contested = keys, False
                    db.executemany("DELETE FROM capture_browser_recorders "
                                   "WHERE token_hash=? AND state='unknown'", [(key,) for key in keys])
                with self._lock:
                    self._carried -= keys
                self._read_cache()
                return len(keys)
            except (ValueError, PermissionError):
                raise
            except BaseException:
                self._fault = True
                raise
            finally:
                with self._lock:
                    self._forgetting, self._forget_contested = frozenset(), False

    async def forget_carried(self, expected: int, *,
                             authorize: Callable[[], None]) -> int:
        """Delete the carried owners an administrator confirmed by their count.

        Args:
            expected: Carried count the administrator saw; any other current
                count is a conflict, so a stale page forgets nothing it did
                not show. With nothing carried this is a no-op returning 0.
            authorize: Raises ``PermissionError`` once the administrator's
                authority lapses; checked inside the transaction.

        Returns:
            How many carried owners were forgotten (0 when none remained).

        Raises:
            ValueError: ``capture_carried_count_conflict`` when the count
                differs or a holder presented its token before the commit.
            RuntimeError: The recorders are closed, faulted or not ready.
        """
        if type(expected) is not int or expected < 0:
            raise ValueError("capture_carried_count_conflict")
        if self._closed or self._fault or not self._ready:
            raise RuntimeError("capture_recorders_unavailable")
        forgotten = await self._call(self._forget_carried, expected, authorize)
        if forgotten:
            logger.warning("[CaptureRecorders] an administrator forgot %d carried owner(s), "
                           "affirming none is still recording", forgotten)
        return forgotten

    def _retire_reserved(self) -> None:
        try:
            with self._privacy.ledger._transaction() as db:
                db.execute("DELETE FROM capture_browser_recorders WHERE state='reserved'")
            self._read_cache()
        except BaseException:
            self._fault = True
            raise

    async def _drain_operations(self) -> list[object]:
        """Wait for every admitted operation, including ones admitted meanwhile.

        A finished task leaves ``_operations`` only through its queued
        done-callback, and ``gather`` over finished tasks completes without
        yielding. Discarding each awaited snapshot here keeps the loop from
        spinning on a done member whose callback the loop never gets to run.
        """
        results: list[object] = []
        while self._operations:
            observed = tuple(self._operations)
            results.extend(await asyncio.shield(asyncio.gather(*observed, return_exceptions=True)))
            self._operations.difference_update(observed)
        return results

    async def drain(self) -> None:
        """Wait for this process's issued owners' explicit ACKs.

        Reserve-only owners may retire. Carried owners are not awaited (a
        reclaimed one is) and stay pending: no drain turns their silence into
        an ACK.
        """
        self.seal()
        await self._drain_operations()
        await self._work.drain_admitted()
        if self._fault:
            raise RuntimeError("capture_recorder_state_unconfirmed")
        await self._call(self._retire_reserved)
        while self.pending_live():
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
        failures.extend(await self._drain_operations())
        await asyncio.shield(self._work.begin_close())
        if self._fault or any(isinstance(result, BaseException) for result in failures):
            raise RuntimeError("capture_recorder_state_unconfirmed")
