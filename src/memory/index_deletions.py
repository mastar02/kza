"""Durable, revisioned cleanup receipts for every registered policy index."""

from dataclasses import dataclass
from enum import StrEnum
import json
from uuid import uuid4
from weakref import WeakMethod


class IndexCleanupError(StrEnum):
    FAILED = 'index_cleanup_failed'
    UNSUPPORTED = 'unsupported_index_target'
    RECHECK = 'index_startup_recheck'


INDEX_DELETION_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_index_deletions (
  target_id TEXT NOT NULL REFERENCES erasure_targets(id), fact_id TEXT NOT NULL REFERENCES facts(id),
  revision INTEGER NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at REAL NOT NULL DEFAULT 0, claim_token TEXT, last_error TEXT,
  PRIMARY KEY(target_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_index_deletion_due ON policy_index_deletions(state, next_attempt_at, target_id);
CREATE INDEX IF NOT EXISTS idx_index_deletion_fact ON policy_index_deletions(fact_id, revision, state);
"""


@dataclass(frozen=True)
class IndexDeletionItem:
    """Exclusive receipt for one revision, without a fact's raw content."""

    fact_id: str
    revision: int
    token: str
    attempts: int


@dataclass(frozen=True)
class IndexDeletionBatch:
    """A bounded target-specific native cleanup claim."""

    target_id: str
    version: int
    descriptor: dict
    items: tuple[IndexDeletionItem, ...]


class IndexDeletionQueue:
    """Compose with the ledger transaction; never let current-index ack lose history."""

    def __init__(self, ledger):
        self.ledger = ledger
        self._listener: WeakMethod | None = None

    def set_listener(self, callback) -> None:
        """Wake a runtime owner without keeping its private lifecycle alive."""
        self._listener = WeakMethod(callback)

    def _notify(self) -> None:
        callback = self._listener() if self._listener is not None else None
        if callback is not None:
            try:
                callback()
            except RuntimeError:
                # Closing event-loop observation cannot roll back erasure.
                pass

    def changed(self) -> None:
        """Wake for a newly earlier fact/proposal deadline as well as deletions."""
        self._notify()

    def enqueue(self, db, fact_id: str, revision: int) -> None:
        """Persist every known destination inside the fact mutation transaction."""
        db.execute("""INSERT INTO policy_index_deletions(target_id,fact_id,revision,state)
            SELECT id, ?, ?, 'pending' FROM erasure_targets WHERE participant='semantic_index'
            ON CONFLICT(target_id,fact_id) DO UPDATE SET revision=excluded.revision,state='pending',
            attempts=0,next_attempt_at=0,claim_token=NULL,last_error=NULL
            WHERE policy_index_deletions.revision<excluded.revision""", (fact_id, revision))
        self._notify()

    def register_target(self, db, target_id: str) -> None:
        """Backfill existing tombstones without resetting completed live claims."""
        db.execute("""INSERT INTO policy_index_deletions(target_id,fact_id,revision,state)
            SELECT ?, id, revision, 'pending' FROM facts WHERE state IN ('deleted','quarantined')
            ON CONFLICT(target_id,fact_id) DO UPDATE SET revision=excluded.revision,state='pending',
            attempts=0,next_attempt_at=0,claim_token=NULL,last_error=NULL
            WHERE policy_index_deletions.revision<excluded.revision""", (target_id,))
        self._notify()

    def recover(self) -> None:
        """Recheck restored physical indexes only after acquiring the runtime lease."""
        with self.ledger._transaction() as db:
            for row in list(db.execute("SELECT id FROM erasure_targets WHERE participant='semantic_index'")):
                self.register_target(db, row[0])
            db.execute("UPDATE policy_index_deletions SET state='pending',claim_token=NULL,next_attempt_at=0,last_error=?",
                       (IndexCleanupError.RECHECK.value,))
            # Old receipts cannot prove a physical index was not restored.
            db.execute("""UPDATE erasure_steps SET state='pending',claim_token=NULL,cursor=NULL,
                next_attempt_at=0,last_error=? WHERE participant='semantic_index' AND state='complete'""",
                (IndexCleanupError.RECHECK.value,))

    def targets(self) -> list[dict]:
        """Read cleanup counts and due times, without documents or storage paths."""
        with self.ledger._transaction() as db:
            return [dict(row) for row in db.execute("""SELECT target_id,
                SUM(state!='complete') AS pending, SUM(state='running') AS running, MAX(last_error) AS error,
                MIN(CASE WHEN state='pending' THEN next_attempt_at END) AS due
                FROM policy_index_deletions GROUP BY target_id ORDER BY target_id""")]

    def pending_targets(self) -> list[dict]:
        """Schedule unfinished work without rescanning completed receipt history."""
        with self.ledger._transaction() as db:
            return [dict(row) for row in db.execute("""SELECT target_id,COUNT(*) AS pending,
                SUM(state='running') AS running,MAX(last_error) AS error,
                MIN(CASE WHEN state='pending' THEN next_attempt_at END) AS due
                FROM policy_index_deletions WHERE state IN ('pending','running')
                GROUP BY target_id ORDER BY target_id""")]

    def pending_count(self, target_id: str) -> int:
        """Do not let a conversation receipt hide unresolved target deletion work."""
        with self.ledger._transaction() as db:
            return db.execute("SELECT COUNT(*) FROM policy_index_deletions WHERE target_id=? AND state!='complete'", (target_id,)).fetchone()[0]

    def claim(self, target_id: str, limit: int = 32) -> IndexDeletionBatch | None:
        """Claim bounded pending revisions; live native claims have no timed expiry."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError('invalid_index_cleanup_batch')
        with self.ledger._transaction() as db:
            target = db.execute('SELECT * FROM erasure_targets WHERE id=? AND participant=?', (target_id, 'semantic_index')).fetchone()
            if target is None:
                raise ValueError('unknown_index_cleanup_target')
            rows = list(db.execute("""SELECT fact_id,revision,attempts FROM policy_index_deletions
                WHERE target_id=? AND state='pending' AND next_attempt_at<=?
                ORDER BY attempts,fact_id LIMIT ?""", (target_id, self.ledger.clock(), limit)))
            items = []
            for row in rows:
                token = str(uuid4())
                db.execute("""UPDATE policy_index_deletions SET state='running',claim_token=?,attempts=attempts+1,last_error=NULL
                    WHERE target_id=? AND fact_id=? AND revision=? AND state='pending'""",
                    (token, target_id, row['fact_id'], row['revision']))
                items.append(IndexDeletionItem(row['fact_id'], row['revision'], token, row['attempts']+1))
            return IndexDeletionBatch(target_id, target['version'], json.loads(target['descriptor']), tuple(items)) if items else None

    def eligible(self, batch: IndexDeletionBatch) -> tuple[IndexDeletionItem, ...]:
        """Revalidate the SOT revision under the physical writer's barrier."""
        with self.ledger._transaction() as db:
            self._validate_batch(db, batch)
            for item in batch.items:
                authority = db.execute('SELECT state FROM facts WHERE id=?', (item.fact_id,)).fetchone()
                if authority is None or authority['state'] not in {'automatic','proposed','confirmed','deleted','quarantined'}:
                    raise RuntimeError('unknown_index_fact_authority')
            return tuple(item for item in batch.items if db.execute("""SELECT 1 FROM facts f JOIN policy_index_deletions d
                ON d.fact_id=f.id WHERE f.id=? AND f.revision=? AND f.state IN ('deleted','quarantined')
                AND d.target_id=? AND d.revision=? AND d.state='running' AND d.claim_token=?""",
                (item.fact_id, item.revision, batch.target_id, item.revision, item.token)).fetchone() is not None)

    def _validate_batch(self, db, batch: IndexDeletionBatch) -> None:
        target = db.execute('SELECT version,descriptor FROM erasure_targets WHERE id=? AND participant=?',
                            (batch.target_id, 'semantic_index')).fetchone()
        encoded = json.dumps(batch.descriptor,sort_keys=True,separators=(',', ':'),allow_nan=False)
        if target is None or target['version'] != batch.version or target['descriptor'] != encoded:
            raise ValueError('invalid_index_cleanup_claim')

    def unfinished(self, target_id: str, rows: list[dict]) -> list[dict]:
        """Reuse a physical receipt only within the current leased generation."""
        with self.ledger._transaction() as db:
            return [row for row in rows if db.execute("""SELECT 1 FROM policy_index_deletions
                WHERE target_id=? AND fact_id=? AND revision=? AND state='complete'""",
                (target_id, row['id'], row['revision'])).fetchone() is None]

    def _ack_global(self, db, fact_id: str, revision: int) -> None:
        db.execute("""DELETE FROM index_jobs WHERE fact_id=? AND revision=? AND operation='delete'
            AND NOT EXISTS (SELECT 1 FROM policy_index_deletions
                WHERE fact_id=? AND revision>=? AND state!='complete')""", (fact_id, revision, fact_id, revision))

    def record_completed(self, target_id: str, rows: list[dict]) -> None:
        """Share conversation-cleanup proof while holding the same native barrier."""
        with self.ledger._transaction() as db:
            for row in rows:
                db.execute("""UPDATE policy_index_deletions SET state='complete',claim_token=NULL,
                    next_attempt_at=0,last_error=NULL WHERE target_id=? AND fact_id=? AND revision=?""",
                    (target_id, row['id'], row['revision']))
                self._ack_global(db, row['id'], row['revision'])

    def finish(self, batch: IndexDeletionBatch, error: IndexCleanupError | None = None) -> None:
        """Acknowledge exact claim tokens, preserving newer revisions and failures."""
        if error is not None and not isinstance(error, IndexCleanupError):
            raise ValueError('invalid_index_cleanup_error')
        with self.ledger._transaction() as db:
            self._validate_batch(db, batch)
            for item in batch.items:
                due = self.ledger.clock()+min(60, 2**min(item.attempts-1, 6)) if error is not None else 0
                db.execute("""UPDATE policy_index_deletions SET state=?,claim_token=NULL,next_attempt_at=?,last_error=?
                    WHERE target_id=? AND fact_id=? AND revision=? AND state='running' AND claim_token=?""",
                    ('pending' if error is not None else 'complete', due, error.value if error is not None else None,
                     batch.target_id, item.fact_id, item.revision, item.token))
                if error is None:
                    self._ack_global(db, item.fact_id, item.revision)
