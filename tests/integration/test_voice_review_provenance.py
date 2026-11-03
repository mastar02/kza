"""Technical metadata follows the existing owned PCM and source lifetime."""

import asyncio
from dataclasses import replace
import threading

import numpy as np
import pytest

from src.evaluation.provenance import (
    CaptureLane, CaptureProvenance, CaptureSource, DSPCertainty, DSPObservation,
    FrameInterval,
)
from src.users.voice_capture_metadata import decode_audio_metadata, decode_transcript_metadata
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import ACTOR, AUDIO, Encoder, graph, idle


pytestmark = pytest.mark.asyncio


def provenance(frames, channel=0):
    return CaptureProvenance(
        CaptureSource.LIVE_INPUT, 'synthetic-input', CaptureLane.AMBIENT,
        4, 12, 2, channel, 16000, frames,
        DSPObservation(DSPCertainty.DECLARED, 'synthetic-before'),
    )


def metadata_rows(voices):
    with voices.privacy.ledger._transaction() as db:
        return [tuple(row) for row in db.execute('''SELECT fingerprint_id,
            audio_metadata_json,transcript_metadata_json FROM voice_capture_reviews
            ORDER BY fingerprint_id''')]


async def test_owned_observation_detaches_both_channel_descriptors_and_preview_span(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        entered, release = threading.Event(), threading.Event()

        class Held(Encoder):
            def get_embedding(self, audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)

        encoder = Held()
        voices.register_encoder(encoder)
        audio = np.tile(AUDIO, 8)
        first = provenance(audio.size)
        text = replace(first, selected_channel=1)
        try:
            assert voices.observe(encoder, audio, (memory_source(now=clock.now),),
                transcript='Synthetic full segment', audio_provenance=first,
                transcript_provenance=text, transcript_span=FrameInterval(0, audio.size),
                transcript_truncated=False)
            await until(entered.is_set)
            audio.fill(0)
            # Even a producer bypassing its frozen DTO cannot relabel the queued job.
            object.__setattr__(first, 'selected_channel', 1)
            object.__setattr__(first.dsp, 'declared_revision', 'synthetic-after')
        finally:
            release.set()
            await idle(voices)
        [(identity, raw_audio, raw_text)] = await voices.acall(metadata_rows, voices)
        audio_source, preview = decode_audio_metadata(raw_audio)
        text_source, text_span, truncated = decode_transcript_metadata(raw_text)
        assert audio_source.selected_channel == 0 and text_source.selected_channel == 1
        assert audio_source.dsp.declared_revision == text_source.dsp.declared_revision == 'synthetic-before'
        assert preview == FrameInterval(0, 96000)
        assert text_span == FrameInterval(0, 128000) and truncated is False
        assert audio_source.frame_count == text_source.frame_count == 128000
        public = await voices.acall(voices.ledger.review_capture, identity, ACTOR)
        assert public['clip_duration_s'] == 6 and public['duration_s'] == 8
        assert 'audio_provenance' not in public and 'transcript_provenance' not in public


async def test_retry_cannot_upgrade_or_relabel_first_review_metadata(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now, ttl=60)
        first = provenance(AUDIO.size)
        assert voices.observe(encoder, AUDIO, (source,), audio_provenance=first)
        await idle(voices)
        before = await voices.acall(metadata_rows, voices)
        clock.now += 10
        assert voices.observe(encoder, AUDIO, (source,),
            audio_provenance=replace(first, selected_channel=1, stream_generation=5),
            transcript_truncated=True)
        await idle(voices)
        assert await voices.acall(metadata_rows, voices) == before


async def test_expired_owned_provenance_cannot_publish_a_late_review(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        entered, release = threading.Event(), threading.Event()

        class Held(Encoder):
            def get_embedding(self, audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)

        encoder = Held()
        voices.register_encoder(encoder)
        try:
            assert voices.observe(encoder, AUDIO, (memory_source(now=clock.now, ttl=10),),
                audio_provenance=provenance(AUDIO.size), transcript_truncated=False)
            await until(entered.is_set)
            clock.now += 10
        finally:
            release.set()
            await idle(voices)
        assert await voices.acall(metadata_rows, voices) == []
        assert not (await voices.acall(voices.ledger.snapshot))['fingerprints']


@pytest.mark.parametrize('damage', ['frames', 'rate', 'dictionary', 'text_pair', 'partial_text', 'flag'])
async def test_invalid_metadata_is_rejected_before_admitting_pcm(tmp_path, damage):
    async with graph(tmp_path) as (_, _, voices, clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        values = {'audio_provenance': provenance(AUDIO.size)}
        if damage == 'frames':
            values['audio_provenance'] = provenance(AUDIO.size + 1)
        elif damage == 'rate':
            values['audio_provenance'] = replace(values['audio_provenance'], sample_rate_hz=8000)
        elif damage == 'dictionary':
            values['audio_provenance'] = {'frame_count': AUDIO.size}
        elif damage == 'text_pair':
            values['transcript_span'] = FrameInterval(0, AUDIO.size)
        elif damage == 'partial_text':
            values['transcript_provenance'] = provenance(AUDIO.size, 1)
            values['transcript_span'] = FrameInterval(1, AUDIO.size)
        else:
            values['transcript_truncated'] = 1
        assert voices.observe(encoder, AUDIO, (memory_source(now=clock.now),), **values) is False
        assert voices.resource_status()['pending'] == 0 and encoder.calls == 0
        assert await voices.acall(metadata_rows, voices) == []


async def test_known_text_truncation_does_not_invent_legacy_source_metadata(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        assert voices.observe(encoder, AUDIO, (memory_source(now=clock.now),),
            transcript='Synthetic retained prefix', transcript_truncated=True)
        await idle(voices)
        [(_, audio, text)] = await voices.acall(metadata_rows, voices)
        assert audio is None
        assert decode_transcript_metadata(text) == (None, None, True)


async def test_close_keeps_metadata_owned_until_native_exit_then_clears_it(tmp_path):
    async with graph(tmp_path) as (privacy, _, voices, clock):
        entered, release = threading.Event(), threading.Event()

        class Held(Encoder):
            def get_embedding(self, audio):
                entered.set()
                assert release.wait(3)
                return super().get_embedding(audio)

        encoder = Held()
        voices.register_encoder(encoder)
        try:
            descriptor = provenance(AUDIO.size)
            assert voices.observe(encoder, AUDIO, (memory_source(now=clock.now),),
                audio_provenance=descriptor, transcript_provenance=replace(descriptor, selected_channel=1),
                transcript_span=FrameInterval(0, AUDIO.size), transcript_truncated=False)
            await until(entered.is_set)
            [retained] = list(voices._jobs.values())
            closing = voices.begin_close()
            await asyncio.sleep(0)
            assert not closing.done() and retained.audio_provenance is not None
            assert retained.transcript_provenance is not None and retained.audio is not None
        finally:
            release.set()
            await voices.begin_close()
        assert retained.audio is None and retained.audio_provenance is None
        assert retained.transcript_provenance is retained.transcript_span is retained.transcript_truncated is None
        assert await privacy.acall(metadata_rows, voices) == []
