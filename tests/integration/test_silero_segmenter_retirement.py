"""Segmenter and predictor ownership are independent explicit choices."""

import asyncio

import numpy as np
import pytest

from src.ambient.room_producer import RoomProducer
from src.ambient.segmenter import UtteranceSegmenter, make_silero_predictor
from tests.integration.test_silero_factory_retirement import install_torch


AUDIO = np.ones((1280, 1), dtype=np.float32)


@pytest.mark.asyncio
@pytest.mark.parametrize('owns_segmenter', [False, True])
@pytest.mark.parametrize('owns_predictor', [False, True])
async def test_segmenter_and_predictor_ownership_are_independent(
    monkeypatch, owns_segmenter, owns_predictor,
):
    models = install_torch(monkeypatch)
    predictor = make_silero_predictor()
    segmenter = UtteranceSegmenter(predictor, owns_vad_predictor=owns_predictor)
    room = RoomProducer('synthetic', lambda: segmenter,
                        owns_segmenter=owns_segmenter, require_cleanup=True)
    try:
        await room._call(room._ensure_segmenter)
        await room.begin_close()
        assert room._segmenter is None
        if owns_segmenter and owns_predictor:
            assert models[0]() is None
            with pytest.raises(RuntimeError, match='^silero_closed$'):
                predictor(AUDIO[:, 0])
        else:
            assert models[0]() is not None
            assert predictor(AUDIO[:, 0]) == .8
        if owns_segmenter:
            assert segmenter._vad_predict is None
            with pytest.raises(RuntimeError, match='^ambient_segmenter_closed$'):
                segmenter.feed(1., AUDIO)
        else:
            assert segmenter._vad_predict is predictor
            segmenter.feed(1., AUDIO)
    finally:
        await room.begin_close()
        await asyncio.to_thread(segmenter.close)
        await predictor.aclose()
    assert models[0]() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('owns_segmenter', [False, True])
async def test_rejected_owned_candidate_stays_registered_until_actual_cleanup(
    monkeypatch, owns_segmenter,
):
    models = install_torch(monkeypatch)
    predictor = make_silero_predictor()

    class RejectedSegmenter(UtteranceSegmenter):
        @property
        def oldest_timestamp(self):
            raise AttributeError('synthetic_missing_cleanup_metadata')

    segmenter = RejectedSegmenter(predictor, owns_vad_predictor=True)
    room = RoomProducer('synthetic', lambda: segmenter,
                        owns_segmenter=owns_segmenter, require_cleanup=True)
    try:
        with pytest.raises(RuntimeError):
            await room._call(room._ensure_segmenter)
        assert (room._segmenter is segmenter) == owns_segmenter
        assert models[0]() is not None
        await room.begin_close()
        assert (models[0]() is None) == owns_segmenter
        if not owns_segmenter:
            assert predictor(AUDIO[:, 0]) == .8
    finally:
        await room.begin_close()
        await asyncio.to_thread(segmenter.close)
        await predictor.aclose()
    assert models[0]() is None


@pytest.mark.asyncio
async def test_each_discard_retires_owned_model_but_utterance_reset_reuses_it(monkeypatch):
    models = install_torch(monkeypatch)
    segmenters = []

    def factory():
        segmenter = UtteranceSegmenter(
            make_silero_predictor(), owns_vad_predictor=True,
            vad_col=0, min_speech_ms=80, preroll_ms=0, max_segment_s=.08,
        )
        segmenters.append(segmenter)
        return segmenter

    room = RoomProducer('synthetic', factory, owns_segmenter=True, require_cleanup=True)
    try:
        await room._call(room._ensure_segmenter)
        first = segmenters[0]
        await room._call(first.reset)
        assert await room._call(first.feed, 1., AUDIO) == []
        assert len(await room._call(first.feed, 1.08, AUDIO)) == 1
        assert models[0]() is not None and len(models) == 1
        await room.discard_private()
        assert models[0]() is None and first._vad_predict is None
        await room._call(room._ensure_segmenter)
        assert len(models) == 2 and models[1]() is not None
        assert segmenters[1] is not first
        await room.discard_private()
        assert all(model() is None for model in models)
        assert all(segmenter._vad_predict is None for segmenter in segmenters)
    finally:
        await room.begin_close()
        for segmenter in segmenters:
            await asyncio.to_thread(segmenter.close)
