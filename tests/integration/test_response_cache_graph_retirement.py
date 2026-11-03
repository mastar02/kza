"""Actual graph response-cache retirement; synthetic SDK/output leaves only.

These assertions use the pre-J public API so the old graph fails on retained
PCM, detached startup work or borrowed-resource mutation, rather than a missing
new lifecycle method. No capture, hardware output or graph.run is started.
"""

import asyncio
import threading
from contextlib import AsyncExitStack
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

import src.tts.piper_tts as output_module
import src.tts.response_cache as cache_module
from src.hooks import HookRegistry, TtsPayload, execute_after_event
from src.main import build_application
from src.pipeline.response_handler import ResponseHandler
from src.tts.response_cache import ResponseCache
from tests.integration.test_tts_graph_retirement import (
    assert_retained,
    cancel_close_twice,
    close_graph,
    selected_config,
    selected_kokoro as kokoro_fixture,
)
from tests.mocks.application_graph import OfflineTTS, offline_home
from tests.mocks.audio_output import AudioBackend

selected_kokoro = kokoro_fixture


pytestmark = pytest.mark.asyncio
PHRASE = "Listo"


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


@pytest.fixture
def cached_output(selected_kokoro, monkeypatch):
    sdk, factories = selected_kokoro
    record = SimpleNamespace(caches=[], audios=[], earcons=[], backend=AudioBackend())
    monkeypatch.setattr(output_module, "sd", record.backend)
    monkeypatch.setattr(cache_module, "CANONICAL_PHRASES", (PHRASE,))
    monkeypatch.setattr(cache_module, "TEMPLATES", {})
    initialize = ResponseCache.__init__
    synthesize = ResponseCache._synthesize

    def remember(cache, *args, **kwargs):
        initialize(cache, *args, **kwargs)
        record.caches.append(cache)

    async def remember_audio(cache, text):
        audio = await synthesize(cache, text)
        if audio is not None:
            record.audios.append(ref(audio))
        return audio

    def load_earcon(_path):
        audio = np.full(1024, .0625, dtype=np.float32)
        record.earcons.append(ref(audio))
        return audio, 24000

    monkeypatch.setattr(ResponseCache, "__init__", remember)
    monkeypatch.setattr(ResponseCache, "_synthesize", remember_audio)
    monkeypatch.setattr(cache_module, "load_earcon", load_earcon)
    return sdk, factories, record


def cache_config(tmp_path, url):
    config = selected_config(tmp_path, url)
    config["tts"]["response_cache"] = {"enabled": True}
    return config


def audio_reference(cache):
    cached = cache.get(PHRASE)
    assert cached is not None and len(cached.audio) > 0
    return ref(cached.audio)


def assert_cache_retired(cache, responses, record, audio=None):
    assert cache.size() == 0 and cache.get(PHRASE) is None
    assert cache._tts is cache._synthesize_callback is None
    assert responses._response_cache is responses._earcon_audio is None
    assert record.earcons and all(item() is None for item in record.earcons)
    if audio is not None:
        assert audio() is None


async def settle(responses, *tasks):
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await responses.wait_closed()


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
@pytest.mark.parametrize("phase", ["native_postlude", "before_publication"])
async def test_unattached_warmup_cache_is_owned_before_build_and_cannot_publish_after_close(
    cached_output, tmp_path, monkeypatch, close_path, phase,
):
    sdk, factories, record = cached_output
    entered, release = asyncio.Event(), asyncio.Event()
    native_entered, native_release = threading.Event(), threading.Event()
    warmup = closing = None
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(cache_config(tmp_path, url), shutdown=stack, factories=factories)
            responses = graph.voice_services.responses
            if phase == "native_postlude":
                finished = responses._worker_finished

                def hold_native(session):
                    borrowed = graph.models.tts.kokoro._model
                    try:
                        finished(session)
                        native_entered.set()
                        assert native_release.wait(5), "cache synthesis postlude was not released"
                        assert borrowed is sdk.models[0]()
                    finally:
                        borrowed = None

                monkeypatch.setattr(responses, "_worker_finished", hold_native)
            else:
                synthesize = ResponseCache._synthesize

                async def hold_publication(cache, text):
                    audio = await synthesize(cache, text)
                    try:
                        entered.set()
                        # A callback already returning PCM may resist cancellation.
                        while not release.is_set():
                            try:
                                await asyncio.shield(release.wait())
                            except asyncio.CancelledError:
                                continue
                        return audio
                    finally:
                        audio = None

                monkeypatch.setattr(ResponseCache, "_synthesize", hold_publication)
            warmup = asyncio.create_task(graph.warmup())
            try:
                await until(native_entered.is_set if phase == "native_postlude" else entered.is_set)
                assert len(record.caches) == 1
                cache = record.caches[0]
                assert responses._response_cache is None and cache.size() == 0
                if phase == "native_postlude":
                    assert responses.get_delivery_status()["workers"] == 0
                else:
                    assert record.audios and record.audios[0]() is not None
                closing = close_graph(close_path, graph, stack)
                await assert_retained(graph, closing)
                await cancel_close_twice(closing)
                assert not warmup.done()
                assert record.earcons[0]() is responses._earcon_audio
            finally:
                native_release.set()
                release.set()
                await settle(responses, warmup, closing)
            assert_cache_retired(cache, responses, record)
            assert all(item() is None for item in record.audios)
            assert sdk.models[0]() is None
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_successful_partial_cache_build_is_registered_even_when_warmup_build_raises(
    cached_output, tmp_path, monkeypatch, close_path,
):
    _, factories, record = cached_output
    build = ResponseCache.build

    async def fail_after_one_real_phrase(cache):
        await build(cache)
        assert cache.size() == 1 and audio_reference(cache)() is not None
        raise RuntimeError("synthetic failure after one actual cache publication")

    monkeypatch.setattr(ResponseCache, "build", fail_after_one_real_phrase)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(cache_config(tmp_path, url), shutdown=stack, factories=factories)
            responses = graph.voice_services.responses
            await graph.warmup()
            assert len(record.caches) == 1 and responses._response_cache is None
            assert record.audios
            cache = record.caches[0]
            await close_graph(close_path, graph, stack)
            assert_cache_retired(cache, responses, record)
            assert all(item() is None for item in record.audios)
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
@pytest.mark.parametrize("phase", ["earcon_playback_postlude", "owned_hook"])
async def test_pcm_drop_waits_for_real_playback_or_owned_hook_but_not_registry_peer(
    cached_output, tmp_path, monkeypatch, close_path, phase,
):
    sdk, factories, record = cached_output
    entered, release = threading.Event(), threading.Event()
    hook_entered, hook_release = asyncio.Event(), asyncio.Event()
    peer_entered, peer_release = asyncio.Event(), asyncio.Event()
    speech = closing = None
    registry = HookRegistry()

    async def after_tts(payload):
        if payload.text == "synthetic unrelated peer":
            peer_entered.set()
            await peer_release.wait()
        else:
            hook_entered.set()
            await hook_release.wait()

    registry.register_after("tts", after_tts)
    peers = set()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(cache_config(tmp_path, url), shutdown=stack, factories=factories)
            responses = graph.voice_services.responses
            await graph.warmup()
            cache = record.caches[0]
            audio = audio_reference(cache)
            responses.hooks = registry
            try:
                execute_after_event(registry, "tts", TtsPayload(0, "synthetic unrelated peer", None, 0, True))
                await until(peer_entered.is_set)
                peers = set(registry._after_tasks)
                assert len(peers) == 1
                if phase == "earcon_playback_postlude":
                    play = output_module.StreamingAudioPlayer.play_stream

                    def hold_playback(player, source, *args, **kwargs):
                        result = play(player, source, *args, **kwargs)
                        entered.set()
                        assert release.wait(5), "actual playback postlude was not released"
                        return result

                    monkeypatch.setattr(output_module.StreamingAudioPlayer, "play_stream", hold_playback)
                    speech = asyncio.create_task(responses.aplay_earcon())
                else:
                    speech = asyncio.create_task(responses.aspeak(PHRASE))
                await until(entered.is_set if phase == "earcon_playback_postlude" else hook_entered.is_set)
                assert len(sdk.native_threads) == 1  # Only the actual cache build synthesized.
                assert record.backend.outputs and all(output.closed for output in record.backend.outputs)
                closing = close_graph(close_path, graph, stack)
                await assert_retained(graph, closing)
                await cancel_close_twice(closing)
                # Sealed lookup may deny new borrowers; retained PCM must still
                # remain owned until the admitted playback/hook really exits.
                assert cache._cache and audio() is not None
                assert responses._response_cache is cache
                assert record.earcons[0]() is responses._earcon_audio
                assert all(not task.done() for task in peers)
                release.set()
                hook_release.set()
                await settle(responses, speech, closing)
                assert_cache_retired(cache, responses, record, audio)
                assert all(not task.done() and not task.cancelled() for task in peers)
            finally:
                release.set()
                hook_release.set()
                peer_release.set()
                await settle(responses, speech, closing)
                await asyncio.gather(*peers, *tuple(registry._after_tasks), return_exceptions=True)
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_borrowed_cache_and_its_tts_remain_usable_after_graph_detaches_them(
    cached_output, tmp_path, monkeypatch, close_path,
):
    _, factories, record = cached_output
    peer = OfflineTTS()
    borrowed = ResponseCache(peer)
    await borrowed.build()
    audio = audio_reference(borrowed)
    initialize = ResponseHandler.__init__

    def with_borrowed_cache(responses, *args, **kwargs):
        assert "response_cache" not in kwargs
        initialize(responses, *args, response_cache=borrowed, **kwargs)

    monkeypatch.setattr(ResponseHandler, "__init__", with_borrowed_cache)
    try:
        async with offline_home() as (url, observations):
            async with AsyncExitStack() as stack:
                config = cache_config(tmp_path, url)
                config["tts"]["response_cache"]["enabled"] = False
                graph = await build_application(config, shutdown=stack, factories=factories)
                responses = graph.voice_services.responses
                assert responses._response_cache is borrowed and borrowed.size() == 1
                await close_graph(close_path, graph, stack)
                assert responses._response_cache is responses._earcon_audio is None
                assert borrowed.size() == 1 and audio() is not None
                assert borrowed._tts is peer and borrowed._synthesize_callback is None
                assert borrowed.get(PHRASE).audio is audio()
                assert peer.synthesize("synthetic peer after graph shutdown")[0].size > 0
                assert record.earcons and record.earcons[0]() is None
            assert observations["service_calls"] == 0
    finally:
        close = getattr(type(borrowed), "aclose", None)
        if callable(close):
            await close(borrowed)
