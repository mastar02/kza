"""Actual conversation records and file ownership, with synthetic capture sources."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

from src.memory.producer_context import producer_context
from src.training.conversation_collector import ConversationCollector
from src.wakeword.clip_store import CollectorKind
from tests.mocks.wake_clips import ClipGraph, graph as clip_graph


@dataclass
class ConversationGraph:
    base: ClipGraph
    collector: ConversationCollector

    @property
    def store(self):
        return self.base.store

    @property
    def privacy(self):
        return self.base.privacy

    @property
    def clock(self):
        return self.base.clock

    def sources(self, *, input_ttl=48*3600, response_ttl=48*3600):
        return self.base.source(ttl=input_ttl), self.base.source(ttl=response_ttl)

    def feedback(self, primary):
        return replace(self.base.source(),conversation_id=primary.conversation_id)

    async def add(self, *, sources=None, room='synthetic-room', **kwargs):
        primary,response = sources or self.sources()
        arguments = dict(user_input='synthetic input',assistant_response='synthetic response',
                         input_sources=(primary,),response_sources=(response,))
        arguments.update(kwargs)
        with producer_context(room,(primary,)):
            return await self.collector.add_turn(**arguments)

    async def rows(self):
        def read():
            with self.privacy.ledger._transaction() as db:
                return [dict(row) for row in db.execute('''SELECT a.*,t.group_id,t.quality,t.digest
                    FROM wake_clips_v1 a JOIN conversation_turns_v1 t ON t.id=a.id
                    WHERE a.target_id=? ORDER BY a.sequence''', (self.collector._target.id,))]
        return await self.store._work.run(read)


@asynccontextmanager
async def graph(tmp_path, *, background=False):
    async with clip_graph(tmp_path,background=background) as base:
        target = await base.store._work.run(base.store.register,tmp_path/'conversations',CollectorKind.CONVERSATIONS)
        collector = ConversationCollector(target.root,store=base.store,target=target,close_timeout_s=.03)
        yield ConversationGraph(base,collector)
