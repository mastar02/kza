"""WAV format, independent bucket rotation and bounded sourced collection."""

import asyncio
from pathlib import Path
import threading
from uuid import UUID
import wave

import numpy as np
import pytest

from src.memory.producer_context import producer_context
from tests.mocks.wake_clips import graph, until

pytestmark = pytest.mark.asyncio


async def test_writes_valid_wav_with_opaque_name_and_original_lineage(tmp_path):
    async with graph(tmp_path) as g:
        source = g.source()
        audio = np.sin(np.linspace(0, 440*2*np.pi, 16000)).astype(np.float32)*.5
        with producer_context('synthetic-room', (source,)):
            assert g.writer.submit('synthetic-room', .61, audio)
        await g.idle()
        path, = g.files()
        assert str(UUID(path.stem)) == path.stem
        assert 'synthetic-room' not in path.name
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()) == (1,16000,2,16000)
        row, = await g.rows()
        assert row['room_id'] == 'synthetic-room' and row['score'] == .61
        assert row['captured_at'] == source.captured_at and row['expires_at'] == source.expires_at
        assert path.stat().st_mode & 0o777 == 0o600


async def test_clips_audio_out_of_range(tmp_path):
    async with graph(tmp_path) as g:
        assert g.writer.submit('synthetic-room', .9, np.full(1600, 1.7, dtype=np.float32), sources=(g.source(),))
        await g.idle()
        with wave.open(str(g.files()[0]), 'rb') as wav:
            assert np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).max() == 32767


async def test_accepts_plain_list(tmp_path):
    async with graph(tmp_path) as g:
        assert g.writer.submit('synthetic-room', .5, [0.]*1600, sources=(g.source(),))
        await g.idle()
        assert len(g.files()) == 1


async def test_rotation_keeps_newest_managed_clips_and_legacy_unchanged(tmp_path):
    async with graph(tmp_path, max_files=5) as g:
        legacy = Path(g.target.root)/'synthetic-legacy.wav'
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b'synthetic legacy')
        for i in range(8):
            assert g.writer.submit('synthetic-room', .4+i/100, np.zeros(160), sources=(g.source(),))
            await g.idle()
        assert len(g.files()) == 5
        assert [r['score'] for r in await g.rows()] == pytest.approx([.43,.44,.45,.46,.47])
        assert legacy.read_bytes() == b'synthetic legacy'


@pytest.mark.parametrize('bucket', ['accepted', 'rejected', 'missed'])
async def test_bucket_and_missing_score_are_metadata_not_file_identity(tmp_path, bucket):
    async with graph(tmp_path) as g:
        if bucket == 'missed':
            assert g.writer.submit_missed('synthetic-room', -1., np.zeros(1600), sources=(g.source(),))
        else:
            assert g.writer.submit('synthetic-room', .43, np.zeros(1600), accepted=bucket=='accepted', sources=(g.source(),))
        await g.idle()
        row, = await g.rows()
        assert row['bucket'] == bucket
        assert row['score'] == (-1. if bucket == 'missed' else .43)
        assert len(g.files()) == 1


@pytest.mark.parametrize('bucket', ['rejected', 'missed'])
async def test_rotation_is_independent_by_bucket(tmp_path, bucket):
    async with graph(tmp_path, max_files=5, max_rejected_files=3, max_missed_files=3) as g:
        assert g.writer.submit('synthetic-room', .8, np.zeros(160), sources=(g.source(),))
        await g.idle()
        for i in range(6):
            if bucket == 'missed':
                assert g.writer.submit_missed('synthetic-room', -1., np.zeros(160), sources=(g.source(),))
            else:
                assert g.writer.submit('synthetic-room', .4+i/100, np.zeros(160), accepted=False, sources=(g.source(),))
            await g.idle()
        rows = await g.rows()
        assert sum(r['bucket'] == bucket for r in rows) == 3
        assert sum(r['bucket'] == 'accepted' for r in rows) == 1


async def test_stop_is_idempotent_and_rejects_later_input(tmp_path):
    async with graph(tmp_path) as g:
        await g.writer.aclose()
        await g.writer.aclose()
        assert not g.writer.resource_status()['native_running']
        assert not g.writer.submit('synthetic-room', .5, np.zeros(160), sources=(g.source(),))


async def test_full_queue_never_blocks_and_does_not_alias_caller_waveform(tmp_path, monkeypatch):
    async with graph(tmp_path, queue_size=1) as g:
        entered, release = threading.Event(), threading.Event()
        encode = g.writer._encode
        def blocked(job):
            entered.set()
            assert release.wait(3)
            encode(job)
        monkeypatch.setattr(g.writer, '_encode', blocked)
        try:
            original = np.ones(160, dtype=np.float32)*.5
            assert g.writer.submit('synthetic-room', .5, original, sources=(g.source(),))
            await until(entered.is_set)
            original[:] = 0
            assert g.writer.submit('synthetic-room', .6, np.zeros(160), sources=(g.source(),))
            assert not g.writer.submit('synthetic-room', .7, np.zeros(160), sources=(g.source(),))
            assert g.writer.resource_status()['queued'] == 1
            assert await asyncio.wait_for(asyncio.sleep(0, result=True), .1)
        finally:
            release.set()
            await g.idle()
        row = (await g.rows())[0]
        with wave.open(str(Path(g.target.root)/'policy_v1'/f"{row['id']}.wav"), 'rb') as wav:
            assert np.frombuffer(wav.readframes(160), dtype=np.int16).max() > 16000
