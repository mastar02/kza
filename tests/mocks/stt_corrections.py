"""Actual correction collector sharing the application's real file/SQLite owner."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

import numpy as np

from src.memory.fact_ledger import MemoryOrigin
from src.training.stt_correction_collector import STTCorrectionCollector
from src.wakeword.clip_store import CollectorKind
from tests.mocks.wake_clips import ClipGraph, graph as clip_graph


@dataclass
class CorrectionGraph:
    base: ClipGraph
    collector: STTCorrectionCollector

    @property
    def privacy(self):
        return self.base.privacy

    @property
    def store(self):
        return self.base.store

    @property
    def clock(self):
        return self.base.clock

    def sources(self, *, original_ttl=48*3600, feedback_ttl=48*3600):
        return (self.base.source(ttl=original_ttl),
                replace(self.base.source(ttl=feedback_ttl), origin=MemoryOrigin.INTERACTION))

    async def add(self, *, sources=None, **kwargs):
        original, feedback = sources or self.sources()
        arguments = dict(audio=np.ones(1600,dtype=np.float32)*.2,
                         original_text='synthetic original', corrected_text='synthetic corrected',
                         sources=(original,), correction_sources=(feedback,))
        arguments.update(kwargs)
        return await self.collector.add_correction(**arguments)

    async def rows(self):
        def read():
            with self.privacy.ledger._transaction() as db:
                return [dict(row) for row in db.execute('''SELECT a.*,c.original_text,c.corrected_text,
                    c.user_id,c.sample_rate,c.used FROM wake_clips_v1 a JOIN stt_corrections_v1 c ON c.id=a.id
                    WHERE a.target_id=? ORDER BY sequence''', (self.collector._target.id,))]
        return await self.store._work.run(read)


@asynccontextmanager
async def graph(tmp_path, *, background=False):
    async with clip_graph(tmp_path, background=background) as base:
        target = await base.store._work.run(base.store.register, tmp_path/'corrections', CollectorKind.STT_CORRECTIONS)
        collector = STTCorrectionCollector(target.root, store=base.store, target=target, close_timeout_s=.03)
        yield CorrectionGraph(base, collector)
