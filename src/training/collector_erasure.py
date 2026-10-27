"""Owned erasure of registered collector files; unknown legacy data stays pending."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
import os
from pathlib import Path
import stat
from uuid import UUID

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.wakeword.clip_store import ClipTarget, CollectorKind, WakeClipStore, clip_sources


@contextmanager
def _directory(root):
    path = Path(root)
    if not path.is_absolute() or str(path) != root or '..' in path.parts or root == '/':
        raise ValueError('invalid_collector_root')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


def _identity(descriptor):
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class CollectorPage:
    """Internal metadata progress; no raw filenames enter a public cursor."""

    result: ErasureResult
    more: bool = False
    waiting: bool = False


class CollectorAudit:
    """Walk one registered root with two descriptors and bounded native pages."""

    def __init__(self, root, owned, nested):
        self.root, self.owned, self.nested = root, owned, nested
        self.context = _directory(root)
        self.root_fd = self.context.__enter__()
        self.policy_fd = None
        self.policy_identity = None
        try:
            self.root_identity = _identity(self.root_fd)
            self.root_entries = os.scandir(self.root_fd)
        except BaseException:
            self.context.__exit__(None,None,None)
            raise
        self.policy_entries = None
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root_entries.close()
        if self.policy_entries is not None:
            self.policy_entries.close()
        if self.policy_fd is not None:
            os.close(self.policy_fd)
        self.context.__exit__(None, None, None)

    def check_identity(self):
        with _directory(self.root) as current:
            if _identity(current) != self.root_identity:
                raise ValueError('collector_root_replaced')
            if self.policy_identity is not None:
                info = os.stat('policy_v1', dir_fd=current, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self.policy_identity:
                    raise ValueError('collector_policy_directory_replaced')

    def page(self, ledger, target, conversation_id):
        self.check_identity()
        for _ in range(128):
            policy = self.policy_entries is not None
            try:
                entry = next(self.policy_entries if policy else self.root_entries)
            except StopIteration:
                if policy:
                    self.policy_entries.close()
                    self.policy_entries = None
                    continue
                self.check_identity()
                return True, False
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not policy:
                if entry.name == 'policy_v1' and self.owned and stat.S_ISDIR(info.st_mode):
                    self.policy_fd = os.open('policy_v1', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.root_fd)
                    self.policy_identity = _identity(self.policy_fd)
                    self.policy_entries = os.scandir(self.policy_fd)
                elif stat.S_ISDIR(info.st_mode) and str(Path(self.root)/entry.name) in self.nested:
                    # A distinct durable target owns that subtree; every
                    # operation also requires its own independent receipt.
                    continue
                else:
                    return False, False
            else:
                if not stat.S_ISREG(info.st_mode):
                    return False, False
                extension = next((suffix for suffix in ('.part','.jsonl','.json','.wav') if entry.name.endswith(suffix)),None)
                if extension is None:
                    return False, False
                identifier = entry.name[:-len(extension)]
                try:
                    if str(UUID(identifier)) != identifier:
                        return False, False
                except ValueError:
                    return False, False
                with ledger._transaction() as db:
                    row = db.execute('SELECT target_id,sources_json FROM wake_clips_v1 WHERE id=?', (identifier,)).fetchone()
                if row is None:
                    # It may have been unlinked by independent retention after
                    # scandir observed it. Never attribute an extant orphan.
                    try:
                        os.stat(entry.name, dir_fd=self.policy_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    return False, False
                owner = self.owned.get(row['target_id'])
                if owner is None or extension not in ('.part',owner.extension):
                    return False, False
                sources = clip_sources(row['sources_json'])
                if target is not None and owner.id == target.id and any(s.conversation_id == conversation_id for s in sources):
                    raise RuntimeError('collector_locator_reappeared')
        return False, True


class CollectorEraser:
    """Drain known files and native raw scopes without deleting unknown legacy."""

    def __init__(self, store: WakeClipStore, *, close_timeout_s: float = 1):
        store.privacy.require_available()
        self.store, self.privacy = store, store.privacy
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='collector-erasure')
        self._work = OwnedBlockingWork(worker=self._offload, capacity=4)
        self._audits = {}
        self._callbacks = {}
        self._closed = False
        self._closing = None
        self._close_timeout_s = close_timeout_s
        self.privacy.add_borrower(self.begin_close)

    async def _offload(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, copy_context().run, partial(function, *args))

    def _targets(self, step):
        if step.participant is not ErasureParticipant.COLLECTORS or step.version != 1:
            raise ValueError('unsupported_collector_target')
        rows = self.privacy.ledger.erasures.targets(ErasureParticipant.COLLECTORS)
        if not any(row['id'] == step.target_id and row['version'] == step.version and row['descriptor'] == step.descriptor for row in rows):
            raise ValueError('unregistered_collector_target')
        if not self.privacy.ledger.conversation_committed_revoked(step.conversation_id):
            raise RuntimeError('collector_erasure_authority_unavailable')
        with self.privacy.ledger._transaction() as db:
            owned = db.execute('''SELECT 1 FROM erasure_steps s JOIN conversation_erasures o ON o.id=s.operation_id
                WHERE s.operation_id=? AND s.target_id=? AND s.participant='collectors' AND s.version=?
                AND s.state='running' AND s.claim_token=? AND o.conversation_id=?''',
                (step.operation_id,step.target_id,step.version,step.claim_token,step.conversation_id)).fetchone()
            if owned is None:
                raise ValueError('collector_claim_not_owned')
        return rows

    def _page(self, step):
        rows = self._targets(step)
        descriptor = step.descriptor
        kind = descriptor.get('kind')
        if kind == 'quarantine_locator_coverage':
            if set(descriptor) != {'kind','batch_id'}:
                raise ValueError('invalid_quarantine_coverage')
            UUID(descriptor['batch_id'])
            known = any(row['version'] == 1 and row['descriptor'].get('kind') == 'legacy_quarantine'
                and row['descriptor'].get('batch_id') == descriptor['batch_id'] for row in rows)
            return CollectorPage(ErasureResult(known, pending_count=0 if known else None))
        if kind == 'legacy_quarantine':
            if set(descriptor) != {'kind','root','batch_id'}:
                raise ValueError('invalid_quarantine_target')
            UUID(descriptor['batch_id'])
            with self.privacy.ledger._transaction() as db:
                count = db.execute('SELECT COUNT(*) FROM legacy_quarantine_records_v1 WHERE batch_id=?', (descriptor['batch_id'],)).fetchone()[0]
            if count:
                return CollectorPage(ErasureResult(False, pending_count=count))
            try:
                with _directory(descriptor['root']) as directory:
                    for filename in ('legacy.sqlite3', 'legacy.sqlite3.part'):
                        try:
                            os.stat(filename, dir_fd=directory, follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        return CollectorPage(ErasureResult(False))
            except FileNotFoundError:
                pass
            return CollectorPage(ErasureResult(True, pending_count=0))
        if set(descriptor) != {'kind','root'}:
            raise ValueError('invalid_collector_target')
        managed = kind in set(CollectorKind)
        if not managed and kind != 'legacy_training':
            raise ValueError('unknown_collector_target')
        target = ClipTarget(step.target_id, descriptor['root'], CollectorKind(kind)) if managed else None
        if managed and not (step.cursor or '').startswith('a:'):
            cursor = step.cursor.split(':') if step.cursor else None
            if cursor is not None and (len(cursor) != 3 or cursor[0] != 's'
                or any(not value.isascii() or not value.isdecimal() or len(value)>19 for value in cursor[1:])):
                raise ValueError('invalid_collector_cursor')
            with self.store._files:
                with self.privacy.ledger._transaction() as db:
                    last, high = (int(cursor[1]), int(cursor[2])) if cursor else (0, db.execute(
                        'SELECT COALESCE(MAX(sequence),0) FROM wake_clips_v1 WHERE target_id=?', (target.id,)).fetchone()[0])
                    if not 0 <= last <= high <= 2**63-1:
                        raise ValueError('invalid_collector_cursor')
                    batch = db.execute('SELECT * FROM wake_clips_v1 WHERE target_id=? AND sequence>? AND sequence<=? ORDER BY sequence LIMIT 128',
                        (target.id,last,high)).fetchall()
                blocked = None
                for row in batch:
                    sources = clip_sources(row['sources_json'])
                    if not 0 < row['expires_at'] <= min(s.expires_at for s in sources):
                        raise ValueError('invalid_collector_locator')
                    affected = any(s.conversation_id == step.conversation_id for s in sources)
                    if affected or row['expires_at'] <= self.privacy.ledger.clock() or self.privacy.ledger.sources_committed_unavailable(sources):
                        if self.store._writing(row['id']):
                            blocked = row['sequence'] if blocked is None else blocked
                            continue
                        # File I/O never holds the sole authority transaction.
                        # The immutable locator remains durable until unlink
                        # and directory fsync both actually finish.
                        self.store._unlink(target,row['id'])
                        with self.privacy.ledger._transaction() as db:
                            changed = db.execute('DELETE FROM wake_clips_v1 WHERE id=? AND target_id=? AND sequence=? AND sources_json=?',
                                (row['id'],target.id,row['sequence'],row['sources_json'])).rowcount
                            if not changed and db.execute('SELECT 1 FROM wake_clips_v1 WHERE id=?', (row['id'],)).fetchone():
                                raise RuntimeError('collector_locator_changed')
                if blocked is not None:
                    return CollectorPage(ErasureResult(False,f's:{max(last,blocked-1)}:{high}'),waiting=True)
                if len(batch) == 128:
                    return CollectorPage(ErasureResult(False,f's:{batch[-1]["sequence"]}:{high}'),more=True)
            if self.store.conversation_busy(target.id,step.conversation_id):
                return CollectorPage(ErasureResult(False,f's:{high}:{high}'),waiting=True)
        key = (step.operation_id,step.target_id)
        if managed and self.store.conversation_busy(target.id,step.conversation_id):
            return CollectorPage(ErasureResult(False,step.cursor),waiting=True)
        audit = self._audits.get(key)
        if audit is None:
            nested = {row['descriptor'].get('root') for row in rows if row['id'] != step.target_id}
            owned = {row['id']:ClipTarget(row['id'],descriptor['root'],CollectorKind(row['descriptor']['kind']))
                for row in rows if row['version']==1 and set(row['descriptor'])=={'kind','root'}
                and row['descriptor']['root']==descriptor['root'] and row['descriptor']['kind'] in set(CollectorKind)}
            try:
                audit = CollectorAudit(descriptor['root'],owned,nested)
            except FileNotFoundError:
                return CollectorPage(ErasureResult(True,pending_count=0))
            self._audits[key] = audit
        complete, more = audit.page(self.privacy.ledger,target,step.conversation_id)
        if not more:
            audit.close()
            self._audits.pop(key,None)
        return CollectorPage(ErasureResult(complete,'a:scan' if more else None,0 if complete else None),more=more)

    async def _clean(self, step):
        # A previous observer/restore cannot use a cursor to hide raw metadata
        # behind it. Page cursors only advance inside this retained callback.
        current = replace(step,cursor=None)
        try:
            while not self._closed:
                for writer in tuple(self.store._writers):
                    writer.prune()
                page = await self._work.run(self._page,current)
                if not page.more and not page.waiting:
                    return page.result
                current = replace(current,cursor=page.result.cursor)
                if page.waiting:
                    await asyncio.sleep(.01)
            return ErasureResult(False,current.cursor)
        finally:
            audit = self._audits.pop((step.operation_id,step.target_id),None)
            if audit is not None:
                # The retained callback owns this finalizer even when all
                # bounded page slots are occupied; close must not need another.
                await self._offload(audit.close)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Keep actual cleanup alive after observer cancellation or timeout."""
        key = (step.operation_id,step.target_id,step.claim_token)
        if self._closed or key not in self._callbacks and len(self._callbacks) >= 4:
            raise RuntimeError('collector_erasure_unavailable')
        if step.cursor is not None and step.cursor != 'a:scan':
            values = step.cursor.split(':')
            if (len(values) != 3 or values[0] != 's'
                or any(not value.isascii() or not value.isdecimal() or len(value)>19 for value in values[1:])
                or not 0 <= int(values[1]) <= int(values[2]) <= 2**63-1):
                raise ValueError('invalid_collector_cursor')
        task = self._callbacks.get(key)
        if task is None:
            task = asyncio.create_task(self._clean(step))
            self._callbacks[key] = task
            def finished(done):
                self._callbacks.pop(key,None)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        finally:
            task = step = None

    def begin_close(self) -> asyncio.Task:
        """Stop admission and retain cleanup until its native resources exit."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*tuple(self._callbacks.values()),return_exceptions=True)
                await self._work.begin_close()
                for audit in self._audits.values():
                    audit.close()
                self._audits.clear()
                self._executor.shutdown(wait=False)
                while any(thread.is_alive() for thread in tuple(self._executor._threads)):
                    await asyncio.sleep(.005)
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation without releasing unfinished native ownership."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('collector_erasure_still_running') from None

    def resource_status(self) -> dict:
        """Expose counters without filenames, source IDs or private data."""
        return {'closed':self._closed,'callbacks':len(self._callbacks),'audits':len(self._audits),
            'native_operations':self._work.status()['operations']}


async def build_collector_erasure(config: dict, *, store: WakeClipStore,
                                  owner: ErasureCoordinator, shutdown: AsyncExitStack) -> CollectorEraser:
    """Register current/historical learning roots even with training disabled."""
    driver = CollectorEraser(store)
    shutdown.push_async_callback(driver.aclose)
    training = config.get('training',{})
    nightly = training.get('nightly',{})
    roots = (training.get('habits',{}).get('data_dir','data/habit_training'),
        nightly.get('data_dir','data/nightly_training'),
        training.get('lora',{}).get('output_dir','models/lora_adapters'),
        nightly.get('output_dir','models/lora_adapters/nightly'),
        nightly.get('whisper_finetune',{}).get('output_dir','models/lora_adapters/whisper'))
    for root in roots:
        descriptor = await driver._work.run(lambda root=root: {'kind':'legacy_training','root':str(Path(root).resolve())})
        await store.privacy.acall(store.ledger.erasures.register_target,ErasureParticipant.COLLECTORS,1,descriptor)
    def register_quarantine_guards():
        with store.ledger._transaction() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='legacy_quarantine_batches_v1'").fetchone():
                for row in db.execute('SELECT id FROM legacy_quarantine_batches_v1'):
                    store.ledger.erasures._register_target(db,ErasureParticipant.COLLECTORS,1,
                        {'kind':'quarantine_locator_coverage','batch_id':row['id']})
    await store.privacy.acall(register_quarantine_guards)
    owner.register(ErasureParticipant.COLLECTORS,driver.clean,versions=frozenset({1}),begin_close=driver.begin_close)
    return driver
