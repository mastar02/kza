"""Offline rehearsal: copy legacy SQLite and quarantine an explicit selection.

No record enters facts, source provenance, proposals or index jobs. Legacy
insertion metadata remains opaque; copying it does not establish extraction,
identity or confirmation. This tool is never called by runtime startup.
"""

from contextlib import AsyncExitStack, closing
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from urllib.parse import quote
from uuid import UUID, uuid4

from src.core.owned_work import OwnedBlockingWork
from src.memory.privacy_services import PrivacyServices


@dataclass(frozen=True)
class QuarantineCopyReceipt:
    """Content-free proof of an isolated, inference-ineligible copy."""

    batch_id: str
    snapshot_records: int
    selected_records: int
    quarantined_records: int
    unselected_records: int
    vector_records: int
    source_unchanged: bool
    replayed: bool


_BATCHES = """CREATE TABLE IF NOT EXISTS legacy_quarantine_batches_v1 (
    id TEXT PRIMARY KEY, snapshot_sha256 TEXT NOT NULL, selection_sha256 TEXT NOT NULL,
    collection_id TEXT NOT NULL, imported_at TEXT NOT NULL, records INTEGER NOT NULL)"""
_RECORDS = """CREATE TABLE IF NOT EXISTS legacy_quarantine_records_v1 (
    batch_id TEXT NOT NULL REFERENCES legacy_quarantine_batches_v1(id),
    record_id TEXT NOT NULL, state TEXT NOT NULL CHECK(state='quarantined'),
    document TEXT NOT NULL, metadata_json TEXT NOT NULL, vector BLOB NOT NULL,
    encoding TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(batch_id,record_id))"""


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(path: Path) -> dict:
    result = {}
    for suffix in ('', '-wal'):
        candidate = Path(str(path)+suffix)
        result[suffix] = _hash(candidate) if candidate.is_file() else None
    return result


def _sync_directory(root: Path) -> None:
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(root: Path, manifest: dict) -> None:
    temporary = root/('manifest-'+str(uuid4())+'.part')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(_encoded(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root/'manifest.json')
    _sync_directory(root)


def _read_json(path: Path, limit: int = 8*1024*1024):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        body = stream.read(limit+1)
    if len(body) > limit:
        raise ValueError('legacy_manifest_limit')
    return json.loads(body)


def _readonly(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect('file:'+quote(str(path))+'?mode=ro'+('&immutable=1' if immutable else ''), uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA trusted_schema=OFF')
    except Exception:
        db.close()
        raise
    db.row_factory = sqlite3.Row
    return db


def _collection(db, name: str):
    rows = db.execute('SELECT id,dimension FROM collections WHERE name=?', (name,)).fetchall()
    if len(rows) != 1:
        raise ValueError('legacy_collection_unavailable')
    identifier, dimension = rows[0]
    UUID(identifier)
    if type(dimension) is not int or not 1 <= dimension <= 16384:
        raise ValueError('legacy_vector_dimension')
    return identifier, dimension


def _selection(values) -> list[str]:
    if type(values) is not list or not 1 <= len(values) <= 100000:
        raise ValueError('legacy_selection_limit')
    if any(type(value) is not str or not 1 <= len(value) <= 512 for value in values):
        raise ValueError('legacy_selection_invalid')
    if len(set(values)) != len(values):
        raise ValueError('legacy_selection_duplicates')
    return sorted(values)


def _prepare(preview: Path, root: Path) -> dict:
    prior = _read_json(preview)
    selected = _selection([row['id'] for row in prior['record_metadata']])
    source = Path(prior['source']['path']).resolve(strict=True)
    if not source.is_file():
        raise ValueError('legacy_source_not_regular')
    collection = prior['source']['collection']
    if type(collection) is not str or not collection or len(collection) > 512:
        raise ValueError('legacy_collection_invalid')
    root = root.resolve()
    if root == source or root.is_relative_to(source.parent) or source.is_relative_to(root):
        raise ValueError('legacy_output_overlaps_source')
    # Exclusive creation prevents accidentally migrating an installed database.
    root.mkdir(mode=0o700)
    manifest = {'format': 'kza-legacy-quarantine-copy-v1', 'batch_id': str(uuid4()),
        'stage': 'copying', 'copied_at': datetime.now(UTC).isoformat(),
        'source': str(source), 'collection': collection, 'selected_ids': selected,
        'selection_sha256': hashlib.sha256(_encoded(selected)).hexdigest()}
    _write_manifest(root, manifest)
    before = _fingerprints(source)
    partial = root/'legacy.sqlite3.part'
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    started = time.monotonic()

    def progress(*_):
        if time.monotonic()-started > 30:
            raise TimeoutError('legacy_snapshot_deadline')

    with closing(_readonly(source)) as original, closing(sqlite3.connect(partial)) as copied:
        original.backup(copied, pages=128, progress=progress, sleep=.01)
    descriptor = os.open(partial, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, root/'legacy.sqlite3')
    _sync_directory(root)
    after = _fingerprints(source)
    with closing(_readonly(root/'legacy.sqlite3', immutable=True)) as snapshot:
        snapshot.set_progress_handler(lambda: time.monotonic()-started > 30, 10000)
        if snapshot.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('legacy_snapshot_corrupt')
        collection_id, _ = _collection(snapshot, collection)
        records, unique = snapshot.execute('''SELECT COUNT(*),COUNT(DISTINCT e.embedding_id)
            FROM embeddings e JOIN segments s ON s.id=e.segment_id WHERE s.collection=?''', (collection_id,)).fetchone()
        found = 0
        for offset in range(0, len(selected), 128):
            batch = selected[offset:offset+128]
            found += len(snapshot.execute('''SELECT e.embedding_id FROM embeddings e
                JOIN segments s ON s.id=e.segment_id WHERE s.collection=? AND e.embedding_id IN
                (SELECT value FROM json_each(?))''', (collection_id, _encoded(batch).decode())).fetchall())
        if records != unique or found != len(selected):
            raise ValueError('legacy_selection_not_in_snapshot')
    manifest.update(stage='snapshot_ready', snapshot_sha256=_hash(root/'legacy.sqlite3'),
        snapshot_records=records, source_before=before, source_after=after,
        source_unchanged=before == after)
    _write_manifest(root, manifest)
    return manifest


def _load(root: Path) -> dict:
    manifest = _read_json(root/'manifest.json')
    if manifest['format'] != 'kza-legacy-quarantine-copy-v1' or manifest['stage'] not in ('snapshot_ready', 'quarantined'):
        raise ValueError('legacy_copy_incomplete')
    UUID(manifest['batch_id'])
    selected = _selection(manifest['selected_ids'])
    if (type(manifest['snapshot_records']) is not int or manifest['snapshot_records'] < len(selected)
        or type(manifest['source_unchanged']) is not bool
        or type(manifest['collection']) is not str or not 1 <= len(manifest['collection']) <= 512):
        raise ValueError('legacy_manifest_invalid')
    if selected != manifest['selected_ids'] or hashlib.sha256(_encoded(selected)).hexdigest() != manifest['selection_sha256']:
        raise ValueError('legacy_selection_changed')
    if (root/'legacy.sqlite3').is_symlink() or _hash(root/'legacy.sqlite3') != manifest['snapshot_sha256']:
        raise ValueError('legacy_snapshot_changed')
    return manifest


def _private_root(root: Path) -> None:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('legacy_copy_not_private')


def _admit_candidate(root: Path) -> None:
    _private_root(root)
    for name in ('candidate.sqlite3', 'candidate.sqlite3.lock'):
        path = root/name
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o077:
                raise ValueError('legacy_candidate_not_private')
        else:
            os.close(descriptor)


def _record(db, collection_id, dimension, identifier, topic):
    rows = db.execute('''SELECT e.id,e.seq_id,e.created_at FROM embeddings e JOIN segments s
        ON s.id=e.segment_id WHERE s.collection=? AND e.embedding_id=?''', (collection_id, identifier)).fetchall()
    if len(rows) != 1:
        raise ValueError('legacy_record_unavailable')
    row_id, sequence, inserted_at = rows[0]
    metadata = [list(row) for row in db.execute('''SELECT key,string_value,int_value,float_value,bool_value
        FROM embedding_metadata WHERE id=? ORDER BY rowid''', (row_id,))]
    documents = [row[1] for row in metadata if row[0] == 'chroma:document']
    if len(documents) != 1 or type(documents[0]) is not str or len(documents[0].encode()) > 1024*1024:
        raise ValueError('legacy_document_unavailable')
    document = documents[0]
    metadata = [row for row in metadata if row[0] != 'chroma:document']
    arrays = [list(row) for row in db.execute('''SELECT key,string_value,int_value,float_value,bool_value
        FROM embedding_metadata_array WHERE id=? ORDER BY rowid''', (row_id,))]
    vector_row = db.execute('''SELECT seq_id,vector,encoding FROM embeddings_queue
        WHERE topic=? AND id=? ORDER BY seq_id DESC LIMIT 1''', (topic, identifier)).fetchone()
    if isinstance(sequence, bytes):
        sequence = int.from_bytes(sequence, 'big')
    if vector_row is None or vector_row[0] != sequence or vector_row[2] != 'FLOAT32' or type(vector_row[1]) is not bytes or len(vector_row[1]) != dimension*4:
        raise ValueError('legacy_vector_unavailable')
    vector = vector_row[1]
    encoded = _encoded({'legacy_insertion_metadata': inserted_at, 'sequence': sequence,
        'metadata': metadata, 'arrays': arrays}).decode()
    if len(encoded.encode()) > 1024*1024:
        raise ValueError('legacy_metadata_limit')
    digest = hashlib.sha256(_encoded([identifier, document, encoded, 'FLOAT32'])+vector).hexdigest()
    return document, encoded, vector, 'FLOAT32', digest


def _import(ledger, root: Path, manifest: dict) -> QuarantineCopyReceipt:
    if _hash(root/'legacy.sqlite3') != manifest['snapshot_sha256']:
        raise ValueError('legacy_snapshot_changed')
    with closing(_readonly(root/'legacy.sqlite3', immutable=True)) as snapshot, ledger._transaction() as db:
        collection_id, dimension = _collection(snapshot, manifest['collection'])
        topics = [row[0] for row in snapshot.execute('SELECT DISTINCT topic FROM embeddings_queue WHERE topic LIKE ?',
            ('%/'+collection_id,)).fetchmany(2)]
        if len(topics) != 1:
            raise ValueError('legacy_vector_topic_unavailable')
        started = time.monotonic()
        snapshot.set_progress_handler(lambda: time.monotonic()-started > 30, 10000)
        db.execute(_BATCHES)
        db.execute(_RECORDS)
        existing = db.execute('SELECT * FROM legacy_quarantine_batches_v1 WHERE id=?', (manifest['batch_id'],)).fetchone()
        replayed = existing is not None
        if existing is not None:
            if (existing['snapshot_sha256'] != manifest['snapshot_sha256']
                or existing['selection_sha256'] != manifest['selection_sha256']
                or existing['collection_id'] != collection_id
                or existing['records'] != len(manifest['selected_ids'])):
                raise ValueError('legacy_batch_changed')
        else:
            if db.execute('SELECT COUNT(*) FROM facts').fetchone()[0]:
                raise ValueError('legacy_copy_requires_fresh_ledger')
            db.execute('INSERT INTO legacy_quarantine_batches_v1 VALUES (?,?,?,?,?,?)',
                (manifest['batch_id'], manifest['snapshot_sha256'], manifest['selection_sha256'],
                 collection_id, datetime.now(UTC).isoformat(), len(manifest['selected_ids'])))
        for identifier in manifest['selected_ids']:
            if time.monotonic()-started > 30:
                raise TimeoutError('legacy_import_deadline')
            data = _record(snapshot, collection_id, dimension, identifier, topics[0])
            if replayed:
                stored = db.execute('''SELECT document,metadata_json,vector,encoding,digest FROM
                    legacy_quarantine_records_v1 WHERE batch_id=? AND record_id=? AND state='quarantined' ''',
                    (manifest['batch_id'], identifier)).fetchone()
                if stored is None or tuple(stored) != data:
                    # A retry must not silently recreate a removed/reviewed row.
                    raise ValueError('legacy_quarantine_changed')
            else:
                db.execute('INSERT INTO legacy_quarantine_records_v1 VALUES (?,?,?,?,?,?,?,?)',
                    (manifest['batch_id'], identifier, 'quarantined', *data))
        count = db.execute('SELECT COUNT(*) FROM legacy_quarantine_records_v1 WHERE batch_id=?', (manifest['batch_id'],)).fetchone()[0]
        if count != len(manifest['selected_ids']):
            raise ValueError('legacy_quarantine_count_changed')
        if _hash(root/'legacy.sqlite3') != manifest['snapshot_sha256']:
            raise ValueError('legacy_snapshot_changed')
    return QuarantineCopyReceipt(manifest['batch_id'], manifest['snapshot_records'], count, count,
        manifest['snapshot_records']-count, count, manifest['source_unchanged'], replayed)


async def stage_legacy_quarantine(preview: str | None, output: str, *, resume: bool = False) -> QuarantineCopyReceipt:
    """Rehearse the authorized selection in a fresh private copy, never in-place.

    Args:
        preview: Private H0 metadata preview selecting exact legacy IDs.
        output: New directory for a snapshot, manifest and isolated policy DB.
        resume: Verify and finish that same copy without opening its source.

    Returns:
        Counts only; no text, vector, identity or original record ID escapes.
    """
    root = Path(output).absolute()
    async with AsyncExitStack() as shutdown:
        work = OwnedBlockingWork()
        shutdown.push_async_callback(work.aclose)
        if resume:
            await work.run(_private_root, root)
            manifest = await work.run(_load, root)
        else:
            if preview is None:
                raise ValueError('legacy_preview_required')
            manifest = await work.run(_prepare, Path(preview), root)
        await work.run(_admit_candidate, root)
        privacy = PrivacyServices(str(root/'candidate.sqlite3'))
        shutdown.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        # Import/verification belongs to the leased authority's actual worker.
        result = await privacy.acall(_import, privacy.ledger, root, manifest)
        manifest['stage'] = 'quarantined'
        await work.run(_write_manifest, root, manifest)
        return result
