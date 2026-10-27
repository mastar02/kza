"""Actual journal, all collector writers and isolated configured legacy roots."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import numpy as np

from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import producer_context
from src.training.collector_erasure import build_collector_erasure
from src.training.conversation_collector import ConversationCollector
from src.training.stt_correction_collector import STTCorrectionCollector
from src.wakeword.clip_store import CollectorKind, WakeClipStore
from src.wakeword.wake_clip_writer import WakeClipWriter
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source

ACTOR = MemoryReviewer('synthetic-admin')


def configuration(root):
    """Keep every current and historical dataset inside the test directory."""
    return {'rooms':{'wake_word':{'capture_clips':{'enabled':False,'dir':str(root/'wake')}}},
        'training':{'conversations_dir':str(root/'conversations'),'exports_dir':str(root/'exports'),
            'stt_corrections':{'enabled':False,'data_dir':str(root/'stt')},
            'habits':{'data_dir':str(root/'habits')},'lora':{'output_dir':str(root/'models')},
            'nightly':{'enabled':False,'data_dir':str(root/'nightly'),'output_dir':str(root/'models/nightly'),
                'whisper_finetune':{'output_dir':str(root/'models/whisper')}}}}


@dataclass
class CollectorGraph:
    privacy: object
    store: object
    owner: object
    driver: object
    wake: object
    stt: object
    conversations: object
    targets: dict
    clock: object
    config: dict

    def source(self, **kwargs):
        return memory_source(now=self.clock.now,**kwargs)

    async def add(self, kind, source):
        if kind=='wake':
            assert self.wake.submit('synthetic-room',.99,np.ones(1600,dtype=np.float32)*.2,sources=(source,))
            async with asyncio.timeout(2):
                while True:
                    records=await self.records()
                    matching=[row for row in records if row['target_id']==self.targets[kind].id
                        and source.conversation_id in row['sources_json']]
                    if matching and self.wake._busy is None and not self.wake._queue:
                        return matching[-1]['id']
                    await asyncio.sleep(.001)
        if kind=='stt':
            return await self.stt.add_correction(audio=np.ones(1600,dtype=np.float32)*.2,
                original_text='synthetic original',corrected_text='synthetic corrected',
                sources=(source,),correction_sources=(source,))
        with producer_context('synthetic-room',(source,)):
            identifier=await self.conversations.add_turn('synthetic input','synthetic response',
                input_sources=(source,),response_sources=(source,))
        if kind=='conversations':
            return identifier
        view,=await self.conversations.get_turns(conversation_id=source.conversation_id)
        assert await self.conversations.mark_response(view,'good',sources=(source,))
        return await self.conversations.export_for_training(self.targets['exports'].root)

    async def records(self):
        def read():
            with self.privacy.ledger._transaction() as db:
                return [dict(row) for row in db.execute('SELECT * FROM wake_clips_v1 ORDER BY sequence')]
        return await self.privacy.acall(read)

    async def receipt(self, source):
        return await self.owner.request(source.conversation_id,ACTOR)

    async def steps(self, source):
        return [row for row in (await self.receipt(source))['steps'] if row['participant']=='collectors']

    async def settle(self, source):
        async with asyncio.timeout(3):
            while True:
                await self.owner.reconcile()
                steps=await self.steps(source)
                if steps and all(row['state']=='complete' for row in steps):
                    return steps
                self.clock.now+=.02
                await asyncio.sleep(.001)


@asynccontextmanager
async def graph(root):
    clock=Clock()
    async with AsyncExitStack() as stack:
        privacy=PrivacyServices(str(root/'privacy.db'),clock=clock,close_timeout_s=.03)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        clock.now+=1
        store=WakeClipStore(privacy,close_timeout_s=.03)
        stack.push_async_callback(store.aclose)
        await store._work.run(store.initialize)
        kinds={'wake':CollectorKind.WAKE_CLIPS,'stt':CollectorKind.STT_CORRECTIONS,
            'conversations':CollectorKind.CONVERSATIONS,'exports':CollectorKind.CONVERSATION_EXPORTS}
        targets={name:await store._work.run(store.register,root/name,kind) for name,kind in kinds.items()}
        owner=ErasureCoordinator(privacy,close_timeout_s=.03)
        stack.push_async_callback(owner.aclose)
        config=configuration(root)
        driver=await build_collector_erasure(config,store=store,owner=owner,shutdown=stack)
        wake=WakeClipWriter(targets['wake'].root,store=store,target=targets['wake'],close_timeout_s=.03)
        stt=STTCorrectionCollector(targets['stt'].root,store=store,target=targets['stt'],close_timeout_s=.03)
        conversations=ConversationCollector(targets['conversations'].root,store=store,target=targets['conversations'],close_timeout_s=.03)
        yield CollectorGraph(privacy,store,owner,driver,wake,stt,conversations,targets,clock,config)
