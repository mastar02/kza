"""Real wake writer, durable catalog and privacy authority with synthetic sources."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from src.memory.privacy_services import PrivacyServices
from src.wakeword.clip_store import ClipTarget, WakeClipStore
from src.wakeword.wake_clip_writer import WakeClipWriter
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


@dataclass
class ClipGraph:
    privacy: PrivacyServices
    store: WakeClipStore
    target: ClipTarget
    writer: WakeClipWriter
    clock: Clock

    def source(self, *, ttl=48*3600):
        return memory_source(now=self.clock.now, ttl=ttl)

    async def rows(self):
        def read():
            with self.privacy.ledger._transaction() as db:
                return [dict(row) for row in db.execute('SELECT * FROM wake_clips_v1 ORDER BY sequence')]
        return await self.store._work.run(read)

    def files(self):
        return list((Path(self.target.root)/'policy_v1').glob('*.wav'))

    async def idle(self):
        await until(lambda: not self.writer.resource_status()['busy'] and not self.writer.resource_status()['queued'])


@asynccontextmanager
async def graph(tmp_path, *, background=False, **options):
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    clock.now += 1
    store = WakeClipStore(privacy, close_timeout_s=.03)
    await store._work.run(store.initialize)
    target = await store._work.run(store.register, tmp_path/'captured')
    writer = WakeClipWriter(target.root, store=store, target=target, close_timeout_s=.03, **options)
    if background:
        store.start()
    try:
        yield ClipGraph(privacy, store, target, writer, clock)
    finally:
        await privacy.begin_close()
