"""Reusable pause of real file collectors keeps native work and history owned."""
from __future__ import annotations

import asyncio
from pathlib import Path
import threading

import pytest

from tests.mocks.conversations import graph as conversation_graph
from tests.mocks.stt_corrections import graph as correction_graph
from tests.mocks.wake_clips import until

pytestmark = pytest.mark.asyncio
KINDS = ('corrections', 'conversations')


def graph(kind, path):
    return correction_graph(path) if kind == 'corrections' else conversation_graph(path)


async def views(kind, collector):
    if kind == 'corrections':
        return await collector.get_training_pairs()
    return await collector.get_turns()


def raw_fields(kind, view):
    if kind == 'corrections':
        return view.original_text, view.corrected_text, view.audio_path
    return view.user_input, view.assistant_response


def short_sources(kind, g):
    if kind == 'corrections':
        return g.sources(original_ttl=2)
    return g.sources(input_ttl=2)


@pytest.mark.parametrize('kind', KINDS)
async def test_pause_and_repeated_resume_keep_history_and_original_deadlines(tmp_path, kind):
    async with graph(kind, tmp_path) as g:
        identifier = await g.add()
        assert identifier
        history, = await views(kind, g.collector)
        before = raw_fields(kind, history)
        expiry = history.expires_at
        rows = await g.rows()
        assert history.available and all(before)
        for _ in range(2):
            g.collector.pause_admission()
            await g.collector.drain_pause()
            assert await g.add() is None
            retained = await views(kind, g.collector)
            assert identifier in {view.id for view in retained}
            assert history.available and raw_fields(kind, history) == before
            assert history.expires_at == expiry
            assert await g.rows() == rows
            g.collector.resume_admission()
            assert not g.collector.resource_status()['closed']
            fresh = await g.add()
            assert fresh and fresh != identifier
            # Successful writes prove the original collector remains reusable.
            rows = await g.rows()
        assert identifier in {view.id for view in await views(kind, g.collector)}
        assert raw_fields(kind, history) == before and history.expires_at == expiry


@pytest.mark.parametrize('kind', KINDS)
async def test_pause_retains_cancelled_native_encoder_and_cancelled_pause_observer(tmp_path, monkeypatch, kind):
    async with graph(kind, tmp_path) as g:
        historical = await g.add()
        history, = await views(kind, g.collector)
        before = raw_fields(kind, history)
        expiry = history.expires_at
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        workers = []
        encode = g.collector._encode

        def held(job):
            result = encode(job)
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5), 'native collector encode was not released'
            returned.set()
            return result

        monkeypatch.setattr(g.collector, '_encode', held)
        adding = asyncio.create_task(g.add())
        observer = second = None
        try:
            await until(entered.is_set)
            adding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await adding
            g.collector.pause_admission()
            observer = asyncio.create_task(g.collector.drain_pause())
            await until(lambda: g.collector.resource_status()['pause_pending'])
            owner = g.collector._pausing
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            g.collector.pause_admission()
            second = asyncio.create_task(g.collector.drain_pause())
            await until(lambda: g.collector.resource_status()['operations'] > 0)
            assert g.collector._pausing is owner and not owner.done()
            assert not second.done() and not returned.is_set()
            assert len(workers) == 1 and workers[0].ident is not None
            assert g.collector.resource_status()['active_writes'] == 1
            assert g.privacy.resource_status()['writer_lease']
            with pytest.raises(RuntimeError, match='pause_pending'):
                g.collector.resume_admission()
            assert await g.add() is None
            assert history.available and raw_fields(kind, history) == before
        finally:
            release.set()
            await asyncio.gather(adding, return_exceptions=True)
            if observer is not None:
                await asyncio.gather(observer, return_exceptions=True)
            if second is not None:
                await second
            monkeypatch.setattr(g.collector, '_encode', encode)
        assert returned.is_set()
        assert g.collector.resource_status()['active_writes'] == 0
        assert g.collector.resource_status()['pending_writes'] == 0
        assert g.collector.resource_status()['operations'] == 0
        assert {row['id'] for row in await g.rows()} == {historical}
        assert raw_fields(kind, history) == before and history.expires_at == expiry
        g.collector.resume_admission()
        assert await g.add()
        assert historical in {view.id for view in await views(kind, g.collector)}


@pytest.mark.parametrize('kind', KINDS)
async def test_original_ttl_expires_during_pause_and_resume_cannot_revive_the_view(tmp_path, kind):
    async with graph(kind, tmp_path) as g:
        sources = short_sources(kind, g)
        assert await g.add(sources=sources)
        history, = await views(kind, g.collector)
        before = raw_fields(kind, history)
        expiry = history.expires_at
        paths = {path for path in Path(g.collector._target.root).rglob('*') if path.is_file()}
        assert all(before) and paths
        g.collector.pause_admission()
        await g.collector.drain_pause()
        assert history.available and history.expires_at == expiry
        g.clock.now = expiry
        assert raw_fields(kind, history) == ('',) * len(before)
        assert not history.available and not await views(kind, g.collector)
        assert paths.issubset({path for path in Path(g.collector._target.root).rglob('*') if path.is_file()})
        g.collector.resume_admission()
        assert not history.available and raw_fields(kind, history) == ('',) * len(before)
        assert await g.add()
        assert history.id not in {view.id for view in await views(kind, g.collector)}
        assert history.expires_at == expiry
