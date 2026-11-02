"""Content-free durable tickets for browser-owned live frames."""

import asyncio
from contextvars import Context
from dataclasses import dataclass, field
from enum import StrEnum
import math
import json
import hashlib
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary, ref

from src.core.owned_work import OwnedBlockingWork
from src.core.reader_binding import ReaderBinding

_CONNECTION_SCHEMA = '''CREATE TABLE IF NOT EXISTS live_connections (
 connection_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, runtime_id TEXT NOT NULL,
 owner_namespace TEXT NOT NULL, UNIQUE(owner_namespace,client_id))'''
_RESET_SCHEMA = '''CREATE TABLE IF NOT EXISTS live_resets (
 id TEXT PRIMARY KEY, client_id TEXT NOT NULL, connection_id TEXT NOT NULL UNIQUE,
 nonce TEXT NOT NULL, cutoff INTEGER NOT NULL, owner_namespace TEXT NOT NULL)'''

LIVE_RECEIPT_SCHEMA = '''
CREATE TABLE IF NOT EXISTS live_deliveries (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, ticket TEXT NOT NULL UNIQUE,
 client_id TEXT NOT NULL, connection_id TEXT NOT NULL, event_id TEXT NOT NULL,
 nonce TEXT NOT NULL, issued_at REAL NOT NULL, expires_at REAL NOT NULL,
 owner_namespace TEXT NOT NULL DEFAULT 'legacy-unbound'
);
CREATE INDEX IF NOT EXISTS live_delivery_client ON live_deliveries(client_id,seq);
CREATE TABLE IF NOT EXISTS live_delivery_sources (
 ticket TEXT NOT NULL REFERENCES live_deliveries(ticket) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL, PRIMARY KEY(ticket,conversation_id)
);
CREATE INDEX IF NOT EXISTS live_delivery_conversation ON live_delivery_sources(conversation_id);
CREATE TABLE IF NOT EXISTS live_delivery_lineage (
 ticket TEXT PRIMARY KEY REFERENCES live_deliveries(ticket) ON DELETE CASCADE,
 version INTEGER NOT NULL, source_digest TEXT NOT NULL
);
''' + _CONNECTION_SCHEMA + ';' + _RESET_SCHEMA + ';'
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
            # Legacy tickets remain unclaimable obligations. Connections and reset
            # challenges are runtime leases, so only those ephemeral tables reset.
            columns = {row[1] for row in db.execute('PRAGMA table_info(live_deliveries)')}
            if 'owner_namespace' not in columns:
                db.execute("ALTER TABLE live_deliveries ADD COLUMN owner_namespace TEXT NOT NULL DEFAULT 'legacy-unbound'")
            for table, schema in (('live_connections', _CONNECTION_SCHEMA), ('live_resets', _RESET_SCHEMA)):
                columns = {row[1] for row in db.execute('PRAGMA table_info(' + table + ')')}
                if 'owner_namespace' not in columns:
                    db.execute('DROP TABLE ' + table)
                    db.execute(schema)
                else:
                    db.execute('DELETE FROM ' + table)

    def _invoke(self, function, args, reader, release):
        # The guard runs after both native locks and again immediately before commit.
        with self.ledger._lock:
            try:
                if reader is not None and not release:
                    reader.require_current()
                with self.ledger._transaction(guard=reader.require_current if reader is not None and not release else None) as db:
                    if reader is not None and not release:
                        reader.require_current()
                    self._recover(db)
                    result = function(db, *(() if reader is None else (reader.namespace,)), *args)
                    if reader is not None and not release:
                        reader.require_current()
                self._initialized = True
                if self._departed:
                    present = {(row[0], row[1]) for row in self.ledger._db.execute(
                        'SELECT owner_namespace,connection_id FROM live_connections')}
                    self._departed.intersection_update(present)
                return result
            except (Exception, asyncio.CancelledError):
                return ReceiptFailure.FAILED
            finally:
                function = args = reader = None

    async def _call(self, function, *args, reader: ReaderBinding | None = None, release: bool = False):
        if reader is None and function != self._pending:
            raise ValueError('live_reader_required')
        if reader is not None:
            if type(reader) is not ReaderBinding:
                raise ValueError('invalid_live_reader')
            if not release:
                reader.require_current()
        if self._closed or len(self._tasks) >= 64:
            raise RuntimeError('live_receipt_unavailable')
        async def execute():
            async with self._slots:
                result = await self._disk.run(self._invoke, function, args, reader, release)
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

    def _connection(self, db, namespace, connection):
        connection = canonical_id(connection)
        row = db.execute('SELECT client_id FROM live_connections WHERE connection_id=? AND runtime_id=? AND owner_namespace=?',
            (connection,self.runtime_id,namespace)).fetchone()
        if row is None:
            raise ValueError('unknown_live_connection')
        return row[0]

    def _hello(self, db, namespace, client, connection):
        client, connection = canonical_id(client), canonical_id(connection)
        existing = db.execute('SELECT connection_id FROM live_connections WHERE owner_namespace=? AND client_id=?',(namespace,client)).fetchone()
        if existing is not None:
            if (namespace,existing[0]) not in self._departed:
                raise ValueError('live_client_already_connected')
            db.execute('DELETE FROM live_resets WHERE connection_id=?',(existing[0],))
            db.execute('DELETE FROM live_connections WHERE connection_id=?',(existing[0],))
        if db.execute('SELECT COUNT(*) FROM live_connections').fetchone()[0]>=self.capacity:
            raise ValueError('live_connection_capacity')
        db.execute('INSERT INTO live_connections VALUES (?,?,?,?)',(connection,client,self.runtime_id,namespace))
        return self._reset(db,namespace,connection)

    async def hello(self, client: str, connection: str, *, reader: ReaderBinding) -> LiveReset:
        """Bind one page instance and request a clear before any private bytes."""
        return await self._call(self._hello,client,connection,reader=reader)

    def _reset(self, db, namespace, connection):
        client = self._connection(db,namespace,connection)
        cutoff = db.execute('SELECT COALESCE(MAX(seq),0) FROM live_deliveries WHERE owner_namespace=? AND client_id=?',(namespace,client)).fetchone()[0]
        challenge = LiveReset(str(uuid4()),str(uuid4()))
        db.execute('DELETE FROM live_resets WHERE connection_id=?',(connection,))
        db.execute('INSERT INTO live_resets VALUES (?,?,?,?,?,?)',(challenge.id,client,connection,challenge.nonce,cutoff,namespace))
        return challenge

    async def reset(self, connection: str, *, reader: ReaderBinding) -> LiveReset:
        """Freeze a cutoff; tickets issued later are never covered by its ACK."""
        return await self._call(self._reset,connection,reader=reader)

    def _reset_ack(self, db, namespace, connection, identity, nonce):
        client = self._connection(db,namespace,connection)
        identity, nonce = canonical_id(identity), canonical_id(nonce)
        row = db.execute('SELECT cutoff FROM live_resets WHERE id=? AND nonce=? AND connection_id=? AND client_id=? AND owner_namespace=?',
            (identity,nonce,connection,client,namespace)).fetchone()
        if row is None:
            return False
        db.execute('DELETE FROM live_deliveries WHERE owner_namespace=? AND client_id=? AND seq<=?',(namespace,client,row[0]))
        db.execute('DELETE FROM live_resets WHERE id=? AND owner_namespace=?',(identity,namespace))
        return True

    async def reset_ack(self, connection: str, identity: str, nonce: str, *, reader: ReaderBinding) -> bool:
        """Persist a page clear only for its exact authenticated connection."""
        return await self._call(self._reset_ack,connection,identity,nonce,reader=reader)

    def _profile_sources(self, db, profiles):
        if type(profiles) is not tuple or not 1<=len(profiles)<=50:
            return None
        from src.users.voice_ledger import VoiceLedger
        voices = VoiceLedger(self.ledger)
        expected = set()
        for identity,revision in profiles:
            row = db.execute('SELECT id,revision,sources_json,confirmed_at,confirmed_by,deleted_at FROM voice_profiles WHERE id=?',(canonical_id(identity),)).fetchone()
            if row is None or type(revision) is not int or row['revision']!=revision or not voices._live(row,confirmed=True):
                return None
            expected.update(voices._sources(row['sources_json']))
        return expected

    def _lineage(self, db, sources):
        conversations = {source.conversation_id for source in sources}
        parents = {dep.fact_id for source in sources for dep in source.fact_dependencies}
        if parents:
            ancestors = [row[0] for row in db.execute('''WITH RECURSIVE ancestors(id) AS (
                SELECT value FROM json_each(?) UNION
                SELECT d.source_fact_id FROM fact_dependencies d JOIN ancestors a ON d.fact_id=a.id LIMIT 4097)
                SELECT id FROM ancestors''',(json.dumps(sorted(parents)),))]
            if len(ancestors)>4096:
                raise ValueError('live_fact_lineage_capacity')
            origins = list(db.execute('''SELECT fact_id,conversation_id FROM fact_sources
                WHERE fact_id IN (SELECT value FROM json_each(?)) LIMIT 4097''',(json.dumps(ancestors),)))
            if len(origins)>4096 or {row[0] for row in origins}!=set(ancestors):
                raise ValueError('live_fact_lineage_unavailable')
            conversations.update(row[1] for row in origins)
        if len(conversations)>4096:
            raise ValueError('live_conversation_lineage_capacity')
        return conversations

    def _issue(self, db, namespace, connection, event_id, expires_at, sources, profiles=(), associated=(), activity_sources=()):
        client = self._connection(db,namespace,connection)
        event_id = canonical_id(event_id)
        from src.memory.fact_ledger import MemorySource, FactDependency
        if (type(sources) is not tuple or not sources or len(sources)>256
                or any(type(source) is not MemorySource or any(type(dep) is not FactDependency
                    for dep in source.fact_dependencies) for source in sources)
                or sum(len(source.fact_dependencies) for source in sources)>256
                or type(expires_at) not in (int,float) or not math.isfinite(expires_at)
                or expires_at <= self.ledger.clock()):
            return None
        if profiles:
            if associated or activity_sources or expires_at>self.ledger.clock()+300:
                return None
            expected = self._profile_sources(db,profiles)
            if expected!=set(sources) or not self.ledger.activity_sources_available(sources):
                return None
        elif not self.ledger.sources_available(sources) or expires_at>min(source.expires_at for source in sources):
            return None
        if associated:
            expected = self._profile_sources(db,associated)
            if (type(activity_sources) is not tuple or expected!=set(activity_sources)
                    or not self.ledger.activity_sources_available(activity_sources)):
                return None
        elif activity_sources:
            return None
        all_sources = set(sources) | set(activity_sources)
        if len(all_sources)>256 or sum(len(source.fact_dependencies) for source in all_sources)>256:
            return None
        conversations = self._lineage(db,all_sources)
        total = db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0]
        count = db.execute('SELECT COUNT(*) FROM live_deliveries WHERE owner_namespace=? AND client_id=?',(namespace,client)).fetchone()[0]
        if total>=self.capacity or count>=self.client_capacity:
            return None
        ticket = LiveTicket(str(uuid4()),str(uuid4()))
        db.execute('INSERT INTO live_deliveries(ticket,client_id,connection_id,event_id,nonce,issued_at,expires_at,owner_namespace) VALUES (?,?,?,?,?,?,?,?)',
            (ticket.ticket,client,connection,event_id,ticket.nonce,self.ledger.clock(),expires_at,namespace))
        db.executemany('INSERT OR IGNORE INTO live_delivery_sources VALUES (?,?)',
            [(ticket.ticket,conversation) for conversation in conversations])
        digest = hashlib.sha256(json.dumps(sorted(conversations)).encode()).hexdigest()
        db.execute('INSERT INTO live_delivery_lineage VALUES (?,1,?)',(ticket.ticket,digest))
        return ticket

    async def issue(self, connection: str, event_id: str, expires_at: float, sources: tuple, *, reader: ReaderBinding) -> LiveTicket | None:
        """Commit a content-free ticket before handing bytes to the transport."""
        return await self._call(self._issue,connection,event_id,expires_at,sources,reader=reader)

    async def issue_profiles(self, connection: str, event_id: str, expires_at: float, sources: tuple, profiles: tuple, *, reader: ReaderBinding) -> LiveTicket | None:
        """Only exact confirmed profile revisions may authorize metadata past raw TTL."""
        if not profiles:
            return None
        return await self._call(self._issue,connection,event_id,expires_at,sources,profiles,reader=reader)

    async def issue_associations(self, connection: str, event_id: str, expires_at: float,
            sources: tuple, profiles: tuple, activity_sources: tuple, *, reader: ReaderBinding) -> LiveTicket | None:
        """Keep suggestion parents without extending the candidate's raw lifetime."""
        if not profiles:
            return None
        return await self._call(self._issue,connection,event_id,expires_at,sources,(),profiles,activity_sources,reader=reader)

    def _ack(self, db, namespace, connection, receipts):
        client = self._connection(db,namespace,connection)
        if type(receipts) is not tuple or not 1<=len(receipts)<=16:
            raise ValueError('invalid_live_ack')
        for ticket,nonce in receipts:
            canonical_id(ticket)
            canonical_id(nonce)
            db.execute('DELETE FROM live_deliveries WHERE ticket=? AND nonce=? AND client_id=? AND connection_id=? AND owner_namespace=?',
                (ticket,nonce,client,connection,namespace))

    async def ack(self, connection: str, receipts: tuple[tuple[str,str],...], *, reader: ReaderBinding) -> None:
        """Retire only matching capabilities; unknown or duplicate ACKs are inert."""
        await self._call(self._ack,connection,receipts,reader=reader)

    def _disconnect(self, db, namespace, connection):
        canonical_id(connection)
        if db.execute('SELECT 1 FROM live_connections WHERE connection_id=? AND runtime_id=? AND owner_namespace=?',
                (connection,self.runtime_id,namespace)).fetchone() is None:
            return
        self._departed.add((namespace,connection))
        db.execute('DELETE FROM live_resets WHERE connection_id=?',(connection,))
        db.execute('DELETE FROM live_connections WHERE connection_id=? AND runtime_id=? AND owner_namespace=?',(connection,self.runtime_id,namespace))

    async def disconnect(self, connection: str, *, reader: ReaderBinding) -> None:
        """Release the connection lease, preserving all unacknowledged tickets."""
        await self._call(self._disconnect,connection,reader=reader,release=True)

    def _pending(self, db, conversation):
        canonical_id(conversation)
        attributed,unknown = set(),set()
        ticket,roots,version,digest = None,[],None,None
        def verify():
            if ticket is not None and (not roots or version!=1 or digest!=hashlib.sha256(json.dumps(roots).encode()).hexdigest()):
                unknown.add(ticket)
        for row in db.execute('''SELECT d.ticket,p.version,p.source_digest,s.conversation_id FROM live_deliveries d
                LEFT JOIN live_delivery_lineage p ON p.ticket=d.ticket
                LEFT JOIN live_delivery_sources s ON s.ticket=d.ticket ORDER BY d.ticket,s.conversation_id'''):
            if row[0]!=ticket:
                verify()
                ticket,version,digest,roots = row[0],row[1],row[2],[]
            if row[3] is not None:
                roots.append(row[3])
            if row[3]==conversation:
                attributed.add(ticket)
        verify()
        return len(attributed | unknown)

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
