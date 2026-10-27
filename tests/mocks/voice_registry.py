"""Synthetic encoder and real leased SQLite voice registry; no model or GPU."""

import asyncio
from contextlib import AsyncExitStack,asynccontextmanager

import numpy as np

from src.core.logging import get_context
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_registry import build_voice_registry
from src.dashboard.live_erasure import build_live_services
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source

ACTOR = VoiceReviewer('synthetic-admin')
VECTOR = np.ones(192,dtype=np.float32)
AUDIO = np.linspace(-.1,.1,16000,dtype=np.float32)


class Encoder:
    model_name = 'synthetic-ecapa'
    sample_rate = 16000
    _model_type = 'synthetic'

    def __init__(self):
        self.calls = 0

    def get_embedding(self,audio):
        assert get_context()=={}
        self.calls+=1
        return VECTOR.copy()


@asynccontextmanager
async def graph(tmp_path):
    async with AsyncExitStack() as stack:
        clock = Clock(2_000_000_000.)
        privacy = PrivacyServices(str(tmp_path/'privacy.db'),clock=clock)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        voices = await build_voice_registry(privacy=privacy,owner=owner,shutdown=stack)
        yield privacy,owner,voices,clock


async def seed(voices,clock,*,ttl=60):
    sources = [memory_source(now=clock.now,ttl=ttl) for _ in range(3)]
    identities = [await voices.acall(voices.ledger.put,'synthetic-model',VECTOR.copy(),1.,(source,)) for source in sources]
    return sources,tuple(identities)


async def idle(voices):
    async with asyncio.timeout(3):
        while voices.resource_status()['pending']:
            await asyncio.sleep(.001)


@asynccontextmanager
async def web_graph(tmp_path):
    """Use the actual shared socket receipts and voice authority for HTTP reads."""
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        async with AsyncExitStack() as stack:
            bus = await build_live_services({},privacy=privacy,owner=owner,shutdown=stack)
            yield privacy,owner,voices,clock,bus
