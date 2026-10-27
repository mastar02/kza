"""Correction format, validation, available counts and actual used receipts."""

from pathlib import Path
from uuid import UUID
import wave

import numpy as np
import pytest

from src.training.stt_correction_collector import STTCorrection
from tests.mocks.stt_corrections import graph

pytestmark = pytest.mark.asyncio


async def test_add_correction(tmp_path):
    async with graph(tmp_path) as g:
        identifier = await g.add(user_id='synthetic-reviewer', confidence=.85)
        assert str(UUID(identifier)) == identifier
        path = Path(g.collector.data_dir)/'policy_v1'/(identifier+'.wav')
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getnframes()) == (16000,1,2,1600)
        row, = await g.rows()
        assert row['original_text'] == 'synthetic original'
        assert row['corrected_text'] == 'synthetic corrected'
        assert row['user_id'] == 'synthetic-reviewer' and row['score'] == .85
        assert not row['used'] and path.stat().st_mode & 0o777 == 0o600
        assert not list(g.collector.data_dir.rglob('metadata.json'))


async def test_add_correction_too_long(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add(audio=np.ones(16000*60)) is None
        assert not await g.rows()


async def test_add_correction_empty_text(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add(corrected_text='   ') is None
        assert not await g.rows()


async def test_count_empty(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.collector.get_corrections_count() == 0


async def test_count_correct(tmp_path):
    async with graph(tmp_path) as g:
        for i in range(3):
            assert await g.add(original_text=f'synthetic original {i}')
        assert await g.collector.get_corrections_count() == 3


async def test_get_training_pairs(tmp_path):
    async with graph(tmp_path) as g:
        for i in range(3):
            assert await g.add(corrected_text=f'synthetic corrected {i}')
        pairs = await g.collector.get_training_pairs()
        assert len(pairs) == 3
        assert all(isinstance(pair, STTCorrection) and pair.available for pair in pairs)
        assert [pair.corrected_text for pair in pairs] == [f'synthetic corrected {i}' for i in range(3)]


async def test_mark_used(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add()
        pairs = await g.collector.get_training_pairs()
        deadline = pairs[0].expires_at
        await g.collector.mark_used(pairs)
        assert pairs[0].used and pairs[0].expires_at == deadline
        assert not await g.collector.get_training_pairs()
        assert await g.collector.get_corrections_count() == 0
