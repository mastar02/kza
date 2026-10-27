"""Content-free durable tickets for browser-owned live frames."""

import asyncio
from contextvars import Context
from dataclasses import dataclass, field
from enum import StrEnum
import math
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary, ref

from src.core.owned_work import OwnedBlockingWork

LIVE_RECEIPT_SCHEMA = '''
CREATE TABLE IF NOT EXISTS live_deliveries (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, ticket TEXT NOT NULL UNIQUE,
 client_id TEXT NOT NULL, connection_id TEXT NOT NULL, event_id TEXT NOT NULL,
 nonce TEXT NOT NULL, issued_at REAL NOT NULL, expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS live_delivery_client ON live_deliveries(client_id,seq);
CREATE TABLE IF NOT EXISTS live_delivery_sources (
 ticket TEXT NOT NULL REFERENCES live_deliveries(ticket) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL, PRIMARY KEY(ticket,conversation_id)
);
CREATE INDEX IF NOT EXISTS live_delivery_conversation ON live_delivery_sources(conversation_id);
CREATE TABLE IF NOT EXISTS live_connections (
 connection_id TEXT PRIMARY KEY, client_id TEXT NOT NULL UNIQUE, runtime_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_resets (
 id TEXT PRIMARY KEY, client_id TEXT NOT NULL, connection_id TEXT NOT NULL UNIQUE,
 nonce TEXT NOT NULL, cutoff INTEGER NOT NULL
);
'''
_OWNERS = WeakKeyDictionary()


def canonical_id(value: object) -> str:
    """Reject opaque objects and noncanonical protocol identifiers."""
    if type(value) is not str or len(value) != 36 or str(UUID(value)) != value:
        raise ValueError('invalid_live_receipt_id')
    return value


class ReceiptFailure(StrEnum):
    FAILED = 'live_receipt_failed'


@dataclass(frozen=True)
class LiveTicket:
    """A capability names a single delivery; no transcript or identity label."""

    ticket: str
    nonce: str = field(repr=False)

    def to_dict(self) -> dict:
        """Serialize only the private channel's receipt capability."""
        return {'ticket':self.ticket,'nonce':self.nonce}


@dataclass(frozen=True)
class LiveReset:
    """One connection may acknowledge only the captured reset cutoff."""

    id: str
    nonce: str = field(repr=False)

    def to_frame(self) -> dict:
        """Ask the client to clear its cache and rendered ticker before ACK."""
        return {'type':'live_reset','payload':{'id':self.id,'nonce':self.nonce},'ts':''}


class LiveReceiptStore:
    """Own bounded native metadata transactions until their actual completion."""

    def __init__(self, ledger, *, capacity=1024, client_capacity=64):
        if (type(capacity) is not int or not 1 <= capacity <= 4096
                or type(client_capacity) is not int or not 1 <= client_capacity <= capacity):
            raise ValueError('invalid_live_receipt_capacity')
        prior = _OWNERS.get(ledger)
        prior = prior() if prior is not None else None
        if prior is not None and (prior._closing is None or not prior._closing.done()):
            raise RuntimeError('live_receipt_owner_active')
        self.ledger = ledger
        self.capacity, self.client_capacity = capacity, client_capacity
        self.runtime_id = str(uuid4())
        self._initialized = False
        self._departed = set()
        self._disk = OwnedBlockingWork(capacity=2)
        self._slots = asyncio.Semaphore(2)
        self._tasks = set()
        self._closed = False
        self._closing = None
        _OWNERS[ledger] = ref(self)

    def _recover(self, db):
        if not self._initialized:
            # The sole leased runtime has recovered. Retain every unacked ticket;
            # a disconnected or restored page is never inferred to be cleared.
            db.execute('DELETE FROM live_connections')
            db.execute('DELETE FROM live_resets')

    def _invoke(self, function, args):
        # Recovery and its published initialized flag share the authority lock.
        # Another worker must not recover over a hello that just committed.
        with self.ledger._lock:
            try:
                with self.ledger._transaction() as db:
                    self._recover(db)
                    result = function(db, *args)
                self._initialized = True
                if self._departed:
                    present = {row[0] for row in self.ledger._db.execute('SELECT connection_id FROM live_connections')}
                    self._departed.intersection_update(present)
                return result
            except (Exception, asyncio.CancelledError):
                return ReceiptFailure.FAILED

    async def _call(self, function, *args):
        if self._closed or len(self._tasks) >= 64:
            raise RuntimeError('live_receipt_unavailable')
        async def execute():
            async with self._slots:
                result = await self._disk.run(self._invoke, function, args)
                if isinstance(result, ReceiptFailure):
                    raise RuntimeError(result.value)
                return result
        task = asyncio.create_task(execute(), context=Context())
        self._tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    def _connection(self, db, connection):
        connection = canonical_id(connection)
        row = db.execute('SELECT client_id FROM live_connections WHERE connection_id=? AND runtime_id=?',
            (connection,self.runtime_id)).fetchone()
        if row is None:
            raise ValueError('unknown_live_connection')
        return row[0]

    def _hello(self, db, client, connection):
        client, connection = canonical_id(client), canonical_id(connection)
        existing = db.execute('SELECT connection_id FROM live_connections WHERE client_id=?',(client,)).fetchone()
        if existing is not None:
            if existing[0] not in self._departed:
                raise ValueError('live_client_already_connected')
            db.execute('DELETE FROM live_resets WHERE connection_id=?',(existing[0],))
            db.execute('DELETE FROM live_connections WHERE connection_id=?',(existing[0],))
        if db.execute('SELECT COUNT(*) FROM live_connections').fetchone()[0]>=self.capacity:
            raise ValueError('live_connection_capacity')
        db.execute('INSERT INTO live_connections VALUES (?,?,?)',(connection,client,self.runtime_id))
        return self._reset(db,connection)

    async def hello(self, client: str, connection: str) -> LiveReset:
        """Bind one page instance and request a clear before any private bytes."""
        return await self._call(self._hello,client,connection)

    def _reset(self, db, connection):
        client = self._connection(db,connection)
        cutoff = db.execute('SELECT COALESCE(MAX(seq),0) FROM live_deliveries WHERE client_id=?',(client,)).fetchone()[0]
        challenge = LiveReset(str(uuid4()),str(uuid4()))
        db.execute('DELETE FROM live_resets WHERE connection_id=?',(connection,))
        db.execute('INSERT INTO live_resets VALUES (?,?,?,?,?)',(challenge.id,client,connection,challenge.nonce,cutoff))
        return challenge

    async def reset(self, connection: str) -> LiveReset:
        """Freeze a cutoff; tickets issued later are never covered by its ACK."""
        return await self._call(self._reset,connection)

    def _reset_ack(self, db, connection, identity, nonce):
        client = self._connection(db,connection)
        identity, nonce = canonical_id(identity), canonical_id(nonce)
        row = db.execute('SELECT cutoff FROM live_resets WHERE id=? AND nonce=? AND connection_id=? AND client_id=?',
            (identity,nonce,connection,client)).fetchone()
        if row is None:
            return False
        db.execute('DELETE FROM live_deliveries WHERE client_id=? AND seq<=?',(client,row[0]))
        db.execute('DELETE FROM live_resets WHERE id=?',(identity,))
        return True

    async def reset_ack(self, connection: str, identity: str, nonce: str) -> bool:
        """Persist a page clear only for its exact authenticated connection."""
        return await self._call(self._reset_ack,connection,identity,nonce)

    def _issue(self, db, connection, event_id, expires_at, sources):
        client = self._connection(db,connection)
        event_id = canonical_id(event_id)
        from src.memory.fact_ledger import MemorySource, FactDependency
        if (type(sources) is not tuple or not sources or len(sources)>256
                or any(type(source) is not MemorySource or any(type(dep) is not FactDependency
                    for dep in source.fact_dependencies) for source in sources)
                or sum(len(source.fact_dependencies) for source in sources)>256
                or not self.ledger.sources_available(sources)
                or type(expires_at) not in (int,float) or not math.isfinite(expires_at)
                or expires_at > min(source.expires_at for source in sources)
                or expires_at <= self.ledger.clock()):
            return None
        total = db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0]
        count = db.execute('SELECT COUNT(*) FROM live_deliveries WHERE client_id=?',(client,)).fetchone()[0]
        if total>=self.capacity or count>=self.client_capacity:
            return None
        ticket = LiveTicket(str(uuid4()),str(uuid4()))
        db.execute('INSERT INTO live_deliveries(ticket,client_id,connection_id,event_id,nonce,issued_at,expires_at) VALUES (?,?,?,?,?,?,?)',
            (ticket.ticket,client,connection,event_id,ticket.nonce,self.ledger.clock(),expires_at))
        db.executemany('INSERT OR IGNORE INTO live_delivery_sources VALUES (?,?)',
            [(ticket.ticket,source.conversation_id) for source in sources])
        return ticket

    async def issue(self, connection: str, event_id: str, expires_at: float, sources: tuple) -> LiveTicket | None:
        """Commit a content-free ticket before handing bytes to the transport."""
        return await self._call(self._issue,connection,event_id,expires_at,sources)

    def _ack(self, db, connection, receipts):
        client = self._connection(db,connection)
        if type(receipts) is not tuple or not 1<=len(receipts)<=16:
            raise ValueError('invalid_live_ack')
        for ticket,nonce in receipts:
            canonical_id(ticket)
            canonical_id(nonce)
            db.execute('DELETE FROM live_deliveries WHERE ticket=? AND nonce=? AND client_id=? AND connection_id=?',
                (ticket,nonce,client,connection))

    async def ack(self, connection: str, receipts: tuple[tuple[str,str],...]) -> None:
        """Retire only matching capabilities; unknown or duplicate ACKs are inert."""
        await self._call(self._ack,connection,receipts)

    def _disconnect(self, db, connection):
        canonical_id(connection)
        self._departed.add(connection)
        db.execute('DELETE FROM live_resets WHERE connection_id=?',(connection,))
        db.execute('DELETE FROM live_connections WHERE connection_id=? AND runtime_id=?',(connection,self.runtime_id))

    async def disconnect(self, connection: str) -> None:
        """Release the connection lease, preserving all unacknowledged tickets."""
        await self._call(self._disconnect,connection)

    def _pending(self, db, conversation):
        canonical_id(conversation)
        attributed = db.execute('SELECT COUNT(*) FROM live_delivery_sources WHERE conversation_id=?',(conversation,)).fetchone()[0]
        unknown = db.execute('SELECT COUNT(*) FROM live_deliveries d WHERE NOT EXISTS (SELECT 1 FROM live_delivery_sources s WHERE s.ticket=d.ticket)').fetchone()[0]
        return attributed+unknown

    async def pending(self, conversation: str) -> int:
        """Count private-metadata obligations without trusting wall clock expiry."""
        return await self._call(self._pending,conversation)

    def status(self) -> dict:
        """Expose actual metadata work counts without tickets or capabilities."""
        return {'closed':self._closed,'operations':len(self._tasks),
            'native_operations':self._disk.status()['operations']}

    def begin_close(self) -> asyncio.Task:
        """Drain actual ACK/issue transactions before releasing shared authority."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*tuple(self._tasks),return_exceptions=True)
                await self._disk.begin_close()
            self._closing = asyncio.create_task(finish(),context=Context())
        return self._closing
