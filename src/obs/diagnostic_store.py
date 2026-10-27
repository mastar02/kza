"""Durable deadlines and recoverable JSONL projections for numeric diagnostics."""

from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL


@dataclass(frozen=True)
class DiagnosticRecord:
    """Admission metadata and a previously validated, content-free payload."""

    identifier: str
    recorded_at: float
    expires_at: float
    payload: dict = field(repr=False)


@dataclass(frozen=True)
class DiagnosticPublication:
    """A new row may be sent once; rebuilding local files never replays TCP."""

    line: str
    expires_at: float


class DiagnosticStore:
    """Own one catalog and two bounded file projections under a writer lease."""

    def __init__(self, path: Path, *, clock, retention_days: float, rotate_bytes: int, project):
        if type(retention_days) not in (int, float) or retention_days <= 0:
            raise ValueError('invalid_diagnostic_retention')
        try:
            seconds = float(retention_days*86400)
        except OverflowError:
            raise ValueError('invalid_diagnostic_retention') from None
        if not math.isfinite(seconds):
            raise ValueError('invalid_diagnostic_retention')
        if type(rotate_bytes) is not int or rotate_bytes <= 0:
            raise ValueError('invalid_diagnostic_rotation')
        self.path, self.clock, self.seconds = path, clock, seconds
        self.catalog = path.with_suffix(path.suffix+'.sqlite3')
        self._project, self._rotate_bytes = project, rotate_bytes
        self._lock = threading.RLock()
        self._lease = None
        self._initialized = False
        self._closed = False
        self._status = {'ready': False, 'rows': 0, 'quarantines': 0, 'next_at': None}

    def _now(self):
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('invalid_diagnostic_clock')
        return float(now)

    def record(self, payload: dict) -> DiagnosticRecord | None:
        """Stamp the deadline before a post, executor or native writer can wait."""
        safe = self._project(payload)
        if safe is None:
            return None
        now = self._now()
        return DiagnosticRecord(str(uuid4()), now, now+self.seconds, safe)

    def status(self) -> dict:
        """Read cached counts without disk, private paths or payloads."""
        return {**self._status, 'writer_lease': self._lease is not None}

    def _acquire(self):
        if self._closed:
            raise RuntimeError('diagnostic_store_closed')
        if self._lease is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink() or self.catalog.is_symlink():
            raise RuntimeError('diagnostic_target_symlink')
        fd = os.open(str(self.catalog)+'.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError('diagnostic_writer_busy') from None
        except BaseException:
            os.close(fd)
            raise
        try:
            catalog_fd = os.open(self.catalog, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.close(catalog_fd)
        except BaseException:
            os.close(fd)
            raise
        self._lease = fd

    def _sync_directory(self):
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _connection(self):
        db = sqlite3.connect(self.catalog)
        db.execute('PRAGMA secure_delete=ON')
        return db

    @staticmethod
    def _meta(db, name):
        return db.execute('SELECT value FROM state WHERE key=?', (name,)).fetchone()[0]

    @staticmethod
    def _set(db, name, value):
        db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (name, value))

    def _line(self, identifier, born, expiry, payload):
        doc = dict(payload)
        doc.update({'record_id': identifier, 'recorded_at': born, 'expires_at': expiry,
            '@timestamp': datetime.fromtimestamp(born, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'service': 'kza-voice'})
        return json.dumps(doc, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    def _decode(self, line, path, index):
        if len(line) > 65536:
            raise ValueError('unverified_diagnostic_row')
        doc = json.loads(line)
        metadata = {'@timestamp', 'service', 'record_id', 'recorded_at', 'expires_at'}
        if type(doc) is not dict or type(doc.get('diagnostic_schema')) is not int or doc['diagnostic_schema'] != 1:
            raise ValueError('unverified_diagnostic_row')
        body = {key: value for key, value in doc.items() if key not in metadata}
        safe = self._project(body)
        if safe != body or doc.get('service') != 'kza-voice':
            raise ValueError('unverified_diagnostic_row')
        if type(doc.get('@timestamp')) is not str or len(doc['@timestamp']) > 64:
            raise ValueError('unverified_diagnostic_timestamp')
        stamp = datetime.fromisoformat(doc['@timestamp'].replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError('unverified_diagnostic_timestamp')
        born = doc.get('recorded_at', stamp.timestamp())
        if type(born) not in (int, float) or not math.isfinite(born) or abs(born-stamp.timestamp()) > .001 or born > self._now():
            raise ValueError('unverified_diagnostic_timestamp')
        expiry = doc.get('expires_at', born+self.seconds)
        if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= born:
            raise ValueError('unverified_diagnostic_deadline')
        identity = doc.get('record_id', str(uuid5(NAMESPACE_URL,
            path.name+':'+str(index)+':'+hashlib.sha256(line.encode()).hexdigest())))
        if type(identity) is not str:
            raise ValueError('unverified_diagnostic_identifier')
        UUID(identity)
        return identity, born, min(expiry, born+self.seconds), safe

    def _import(self, db, path, partition):
        if not path.exists():
            return
        if path.is_symlink():
            raise RuntimeError('diagnostic_target_symlink')
        db.execute('SAVEPOINT import_file')
        try:
            with path.open('rb') as stream:
                index = 0
                while encoded_line := stream.readline(65537):
                    line = encoded_line.decode('utf-8')
                    identifier, born, expiry, body = self._decode(line, path, index)
                    encoded = self._line(identifier, born, expiry, body)
                    db.execute('INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,?,?)',
                        (identifier, born, expiry, json.dumps(body), partition, len(encoded.encode())+1))
                    index += 1
        except (ValueError, KeyError, TypeError, UnicodeError):
            db.execute('ROLLBACK TO import_file')
            destination = path.with_name(path.name+'.quarantine-'+str(uuid4()))
            db.execute('INSERT INTO quarantines VALUES (?,?,0)', (path.name, destination.name))
            db.commit()
            path.rename(destination)
            self._sync_directory()
            db.execute('UPDATE quarantines SET moved=1 WHERE source=?', (path.name,))
            db.commit()
        else:
            db.execute('RELEASE import_file')

    def _initialize(self, db):
        if self._initialized:
            return
        application = db.execute('PRAGMA application_id').fetchone()[0]
        version = db.execute('PRAGMA user_version').fetchone()[0]
        if application == 0 and not db.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
            db.execute('PRAGMA application_id=1264206897')
            db.execute('PRAGMA user_version=1')
        elif application != 1264206897 or version != 1:
            raise RuntimeError('unsupported_diagnostic_catalog')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value INTEGER NOT NULL);
            INSERT OR IGNORE INTO state VALUES ('dirty',1),('imported',0),('active_bytes',0);
            CREATE TABLE IF NOT EXISTS metrics(id TEXT PRIMARY KEY,born REAL NOT NULL,
                expires REAL NOT NULL,payload TEXT NOT NULL,partition INTEGER NOT NULL,bytes INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS diagnostic_expiry ON metrics(expires);
            CREATE TABLE IF NOT EXISTS quarantines(source TEXT PRIMARY KEY,destination TEXT NOT NULL,moved INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS staged(name TEXT PRIMARY KEY);
        ''')
        # A recorded quarantine intent survives an interrupted rename.
        for source, destination in db.execute('SELECT source,destination FROM quarantines WHERE moved=0').fetchall():
            if source not in {self.path.name, self.path.name+'.1'} or Path(destination).name != destination:
                raise RuntimeError('invalid_diagnostic_quarantine')
            prefix = source+'.quarantine-'
            if not destination.startswith(prefix):
                raise RuntimeError('invalid_diagnostic_quarantine')
            UUID(destination[len(prefix):])
            original, archived = self.path.parent/source, self.path.parent/destination
            if original.exists() and not archived.exists():
                original.rename(archived)
            if not archived.exists():
                raise RuntimeError('diagnostic_quarantine_missing')
            self._sync_directory()
            db.execute('UPDATE quarantines SET moved=1 WHERE source=?', (source,))
        if not self._meta(db, 'imported'):
            self._import(db, self.path, 0)
            self._import(db, self.path.with_suffix(self.path.suffix+'.1'), 1)
            self._set(db, 'imported', 1)
        changed = db.execute('UPDATE metrics SET expires=born+? WHERE expires>born+?',
            (self.seconds, self.seconds)).rowcount
        if changed:
            self._set(db, 'dirty', 1)
        self._cleanup_staged(db)
        # Rebuild on restart; the catalog, not the old cache, grants visibility.
        self._set(db, 'dirty', 1)
        db.commit()
        self._initialized = True

    def _cleanup_staged(self, db):
        # Retry in the same process too: a failed unlink is still owned work.
        for (name,) in db.execute('SELECT name FROM staged').fetchall():
            if Path(name).name != name or not name.startswith('.diag-') or not name.endswith('.tmp'):
                raise RuntimeError('invalid_diagnostic_stage')
            (self.path.parent/name).unlink(missing_ok=True)
            db.execute('DELETE FROM staged WHERE name=?', (name,))
        db.commit()

    def _expire(self, db):
        self._cleanup_staged(db)
        if db.execute('DELETE FROM metrics WHERE expires<=?', (self._now(),)).rowcount:
            self._set(db, 'dirty', 1)
        db.commit()

    def _publish(self, db):
        if not self._meta(db, 'dirty'):
            return
        for partition, target in ((0, self.path), (1, self.path.with_suffix(self.path.suffix+'.1'))):
            fd, name = tempfile.mkstemp(prefix='.diag-', suffix='.tmp', dir=self.path.parent)
            stage = Path(name)
            try:
                db.execute('INSERT INTO staged VALUES (?)', (stage.name,))
                db.commit()
                stream = os.fdopen(fd, 'w', encoding='utf-8')
                fd = None
                with stream:
                    for identifier, born, expiry, payload in db.execute(
                        'SELECT id,born,expires,payload FROM metrics WHERE partition=? ORDER BY born,id', (partition,)):
                        stream.write(self._line(identifier, born, expiry, json.loads(payload))+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(stage, target)
            finally:
                if fd is not None:
                    os.close(fd)
                stage.unlink(missing_ok=True)
                db.execute('DELETE FROM staged WHERE name=?', (stage.name,))
                db.commit()
        self._sync_directory()
        self._set(db, 'active_bytes', self.path.stat().st_size)
        self._set(db, 'dirty', 0)
        db.commit()

    def _snapshot(self, db):
        count, next_at = db.execute('SELECT COUNT(*),MIN(expires) FROM metrics').fetchone()
        self._status = {'ready': not self._meta(db, 'dirty'), 'rows': count, 'next_at': next_at,
            'quarantines': db.execute('SELECT COUNT(*) FROM quarantines').fetchone()[0]}

    def prune(self) -> None:
        """Persist tighter deadlines before repairing either file projection."""
        with self._lock:
            self._status = {**self._status, 'ready': False}
            self._acquire()
            with closing(self._connection()) as db:
                self._initialize(db)
                self._expire(db)
                self._publish(db)
                self._snapshot(db)

    def append(self, record: DiagnosticRecord) -> DiagnosticPublication | None:
        """Commit admission once; a failed file write leaves a durable dirty cache."""
        with self._lock:
            self._status = {**self._status, 'ready': False}
            self._acquire()
            with closing(self._connection()) as db:
                self._initialize(db)
                self._expire(db)
                self._publish(db)
                expiry = min(record.expires_at, record.recorded_at+self.seconds)
                if expiry <= self._now() or record.recorded_at > self._now():
                    self._snapshot(db)
                    return None
                if db.execute('SELECT 1 FROM metrics WHERE id=?', (record.identifier,)).fetchone():
                    self._snapshot(db)
                    return None
                body = self._project(record.payload)
                if body != record.payload:
                    raise ValueError('unverified_diagnostic_payload')
                line = self._line(record.identifier, record.recorded_at, expiry, body)
                if expiry <= self._now():
                    self._snapshot(db)
                    return None
                rotate = self._meta(db, 'active_bytes') >= self._rotate_bytes
                if rotate:
                    db.execute('DELETE FROM metrics WHERE partition=1')
                    db.execute('UPDATE metrics SET partition=1 WHERE partition=0')
                inserted = db.execute('INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,0,?)',
                    (record.identifier, record.recorded_at, expiry, json.dumps(body), len(line.encode())+1)).rowcount
                self._set(db, 'dirty', 1)
                db.commit()
                if rotate:
                    self._publish(db)
                elif inserted:
                    with self.path.open('a', encoding='utf-8') as stream:
                        stream.write(line+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._set(db, 'active_bytes', self._meta(db, 'active_bytes')+len(line.encode())+1)
                    self._set(db, 'dirty', 0)
                    db.commit()
                self._expire(db)
                self._publish(db)
                self._snapshot(db)
                return DiagnosticPublication(line, expiry) if inserted and expiry > self._now() else None

    def close(self) -> None:
        """Release only after the emitter drains both write and retention owners."""
        with self._lock:
            self._closed = True
            if self._lease is not None:
                fcntl.flock(self._lease, fcntl.LOCK_UN)
                os.close(self._lease)
                self._lease = None
