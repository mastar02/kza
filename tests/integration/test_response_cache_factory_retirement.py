"""The graph owns its warmup cache and earcon until actual output retirement."""

from contextlib import AsyncExitStack
from weakref import ref

import numpy as np
import pytest

from src.main import build_application
from tests.integration.test_application_graph import leaves as graph_leaves
from tests.mocks.application_graph import offline_config, offline_home


leaves = graph_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_graph_retires_owned_cache_and_earcon_with_wrappers_retained(
    leaves, tmp_path, monkeypatch, close_path,
):
    import src.tts.response_cache as module

    earcon_refs = []

    def load_earcon(_path):
        audio = np.full(512, .125, dtype=np.float32)
        earcon_refs.append(ref(audio))
        return audio, 24000

    monkeypatch.setattr(module, 'CANONICAL_PHRASES', ['Listo'])
    monkeypatch.setattr(module, 'TEMPLATES', {})
    monkeypatch.setattr(module, 'load_earcon', load_earcon)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config['tts']['response_cache'] = {'enabled': True}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            response = graph.voice_services.responses
            await graph.warmup()
            cache = response._response_cache
            assert cache.size() == 1
            cached_audio = ref(cache.get('Listo').audio)
            assert cached_audio() is not None and earcon_refs[0]() is not None
            if close_path == 'factory_stack':
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert cache.size() == 0
            assert cache._tts is cache._synthesize_callback is None
            assert response._response_cache is response._earcon_audio is None
            assert cached_audio() is None and earcon_refs[0]() is None
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert observations['service_calls'] == 0
