"""Durable capture/clip cleanup over registered current and historical stores."""

import asyncio
import os
from pathlib import Path
from functools import partial

from src.ambient.retention import CaptureRetention
from src.ambient.store import AmbientStore
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.memory.privacy_services import PrivacyServices


class CaptureEraser:
    """Wait actual encoders before retiring rows and their durable file locators."""

    def __init__(self, privacy: PrivacyServices, *, retention_hours: float = 48):
        self.privacy = privacy
        self._archivers = []
        self._closed = False
        self._closing: asyncio.Task | None = None
        self.retention = CaptureRetention(privacy, audio_inflight=self._is_writing, archive_contains=self._contains_audio, retention_hours=retention_hours)

    async def attach(self, ambient_path) -> None:
        """Bind the actual archive owner before the ambient graph starts producers."""
        if self._closed:
            raise RuntimeError('capture_erasure_closed')
        archiver = ambient_path.transcriber._archiver
        if archiver is not None and archiver not in self._archivers:
            store = ambient_path.store
            if archiver._privacy is not self.privacy or store.source_policy is not self.privacy.ledger or archiver._store is not store:
                raise ValueError('capture_archive_authority_mismatch')
            descriptor = await self.privacy.acall(lambda: {
                'db_path': str(Path(store.db_path).resolve()), 'audio_dir': str(archiver.base_dir.resolve())})
            store_root = await self.privacy.acall(lambda: str(store.audio_dir.resolve()))
            if store_root != descriptor['audio_dir']:
                raise ValueError('capture_archive_location_mismatch')
            await self.privacy.acall(self.privacy.ledger.erasures.register_target, ErasureParticipant.CAPTURES, 1, descriptor)
            if self._closed:
                raise RuntimeError('capture_erasure_closed')
            self._archivers.append(archiver)
            store.audio_inflight = self._is_writing
            await self.retention.watch_store(store, descriptor['db_path'])

    def _is_writing(self, path: str) -> bool:
        return any(archiver.is_writing(path) for archiver in self._archivers)

    def _contains_audio(self, path: str) -> bool:
        # Path.rglob suppresses traversal errors on Python 3.13. An unreadable
        # directory is unknown, never evidence that no clip remains.
        try:
            os.stat(path)
        except FileNotFoundError:
            return False
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise RuntimeError('capture_archive_symlink')
                if entry.is_dir(follow_symlinks=False):
                    if self._contains_audio(entry.path):
                        return True
                elif entry.name.endswith('.flac'):
                    return True
        return False

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Reconcile one location; failures leave its journal claim pending."""
        descriptor = step.descriptor
        if step.participant is not ErasureParticipant.CAPTURES or step.version != 1 or set(descriptor) != {'db_path', 'audio_dir'}:
            raise ValueError('invalid_capture_erasure_target')
        if any(type(value) is not str or not Path(value).is_absolute() for value in descriptor.values()):
            raise ValueError('invalid_capture_erasure_location')
        if not self.privacy.ledger.conversation_revoked(step.conversation_id):
            raise PermissionError('capture_erasure_requires_revocation')
        await asyncio.gather(*(a.drain_conversation(step.conversation_id) for a in self._archivers))
        targets = await self.privacy.acall(self.privacy.ledger.erasures.targets, ErasureParticipant.CAPTURES)
        # A database can retain locators from before keep_audio.dir changed.
        # Authorize only roots registered for that same database, never paths
        # supplied by a row or an unknown version of a storage descriptor.
        roots = tuple(target['descriptor']['audio_dir'] for target in targets if target['version'] == 1
            and set(target['descriptor']) == {'db_path', 'audio_dir'}
            and target['descriptor']['db_path'] == descriptor['db_path'])
        store = AmbientStore(db_path=descriptor['db_path'], audio_dir=descriptor['audio_dir'],
            source_policy=self.privacy.ledger, clock=self.privacy.ledger.clock,
            audio_inflight=self._is_writing, cleanup_audio_roots=roots, disk_worker=self.retention._disk.run,
            file_guard=partial(self.retention.guard_files, descriptor['db_path']))
        try:
            if not await store.init(existing_only=True):
                # A missing database cannot prove its archive directory empty.
                # Keep the location blocked if any clips lost their locators.
                clips = await asyncio.to_thread(self._contains_audio, descriptor['audio_dir'])
                if clips:
                    raise RuntimeError('capture_archive_without_database')
                return ErasureResult(True)
            progress = await store.erase_conversation(step.conversation_id)
            return ErasureResult(progress['pending'] == 0, pending_count=progress['pending'])
        finally:
            await store.close()

    def begin_close(self) -> asyncio.Task:
        """Retain actual encoders even when their callers or close observers cancel."""
        self._closed = True
        self.retention.stop_admission()
        if self._closing is None:
            async def finish():
                await asyncio.gather(*(a.begin_close() for a in self._archivers))
                await self.retention.begin_close()
            self._closing = asyncio.create_task(finish())
        return self._closing


async def build_capture_erasure(config: dict, *, privacy: PrivacyServices,
                                owner: ErasureCoordinator) -> CaptureEraser:
    """Register capture locations even if ambient capture/archiving is disabled."""
    settings = config.get('ambient', {}) or {}
    path = os.fspath(settings.get('db_path', './data/ambient.db'))
    if not path or path == ':memory:' or path.startswith('file:'):
        raise ValueError('capture_policy_requires_persistent_file')
    descriptor = await privacy.acall(lambda: {
        'db_path': str(Path(path).resolve()),
        'audio_dir': str(Path((settings.get('keep_audio', {}) or {}).get('dir', './data/ambient_audio')).resolve())})
    driver = CaptureEraser(privacy, retention_hours=settings.get('retention_hours', 48))
    owner.register(ErasureParticipant.CAPTURES, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
    await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.CAPTURES, 1, descriptor)
    await driver.retention.refresh()
    driver.retention.start()
    return driver
