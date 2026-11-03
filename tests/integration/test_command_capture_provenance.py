"""Private proposal: actual command transport with synthetic model/hardware leaves."""

import asyncio
from contextvars import ContextVar
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from src.audio.microphone_status import CaptureObservation
from src.ambient.textual_wake import TextualWakeDetector
from src.core.capture_opening import CaptureOpening, capture_opening_context
from src.evaluation.provenance import CaptureLane, CaptureSource, DSPCertainty, DSPObservation, FrameInterval
from src.memory.fact_ledger import MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT, producer_context
from src.pipeline.command_event import CapturedTranscript, CommandEvent, TranscriptSource
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream, _CaptureHolder, _pcm_context
from src.pipeline.request_router import RequestRouter
from src.stt.whisper_fast import STTResult
from src.wakeword.whisper_wake import WhisperWakeDetector
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.memory_sources import memory_source


def opening(name='synthetic-opening', channels=2):
    return CaptureOpening(CaptureSource.LIVE_INPUT, name, 4, 16000, channels,
        DSPObservation(DSPCertainty.DECLARED, 'synthetic-before'))


def selected(frames, channel=0, *, source=None):
    return (source or opening()).project(lane=CaptureLane.COMMAND, segment_generation=7,
        selected_channel=channel, frame_count=frames, input_ref=uuid4().hex)


def test_digest_reuse_and_actual_stt_window_are_independent_and_detached():
    audio = np.zeros(16000, dtype=np.float32)
    original = selected(32000, 1)
    transcript = CapturedTranscript.from_audio('Synthetic full window', TranscriptSource.WAKE, audio,
        transcript_provenance=original, transcript_span=FrameInterval(0, 32000), transcript_truncated=False)
    object.__setattr__(original, 'selected_channel', 0)
    object.__setattr__(original.dsp, 'declared_revision', 'synthetic-after')
    assert transcript.covers(audio) and transcript.samples == 16000
    assert transcript.transcript_provenance.frame_count == 32000
    assert transcript.transcript_provenance.selected_channel == 1
    assert transcript.transcript_provenance.dsp.declared_revision == 'synthetic-before'
    assert transcript.transcript_span == FrameInterval(0, 32000)
    unknown = CapturedTranscript.from_audio('Legacy text', TranscriptSource.WAKE, audio)
    assert unknown.covers(audio) and unknown.transcript_provenance is None and unknown.transcript_span is None
    assert unknown.transcript_truncated is None


class Registry:
    def __init__(self):
        self.calls = []

    def register_encoder(self, identifier):
        return threading.Lock()

    def observe(self, identifier, audio, sources, **kwargs):
        self.calls.append((sources, kwargs))
        return True


class STT:
    def __init__(self, text='Synthetic primary result'):
        self.text = text
        self.calls = 0

    def transcribe_with_confidence(self, audio, sample_rate=16000):
        self.calls += 1
        return STTResult(text=self.text, elapsed_ms=1.)


def processor(stt, registry):
    return CommandProcessor(stt, speaker_identifier=object(), user_manager=SimpleNamespace(
        voice_review_managed=True, voice_registry=registry))


@pytest.mark.asyncio
@pytest.mark.parametrize('reused', [False, True])
async def test_processor_preserves_actual_primary_or_independent_reused_stt_metadata(reused):
    registry, stt = Registry(), STT()
    command = processor(stt, registry)
    audio = np.zeros(16000, dtype=np.float32)
    audio_source, text_source = selected(16000), selected(32000, 1)
    source = memory_source(now=2_000_000_000.)
    options = ({'pretranscribed_text': 'Synthetic reused result', 'transcript_provenance': text_source,
        'transcript_span': FrameInterval(0, 32000), 'transcript_truncated': False} if reused else {})
    try:
        with producer_context('synthetic-room', (source,)):
            await command.process_command(audio, use_parallel=False, audio_provenance=audio_source, **options)
        [(sources, values)] = registry.calls
        assert sources == (source,) and values['audio_provenance'] == audio_source
        assert values['transcript_provenance'] == (text_source if reused else audio_source)
        assert values['transcript_span'] == FrameInterval(0, 32000 if reused else 16000)
        assert values['transcript_truncated'] is False
        assert stt.calls == (0 if reused else 1)
    finally:
        await command.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('length,prior,expected', [
    (8192, None, None), (8193, None, True), (8192, False, False), (10, True, True),
])
async def test_direct_pretranscribed_text_stays_unknown_and_prefix_truncation_is_explicit(length, prior, expected):
    registry, stt = Registry(), STT()
    command = processor(stt, registry)
    try:
        await command.process_command(np.zeros(16000, dtype=np.float32), use_parallel=False,
            pretranscribed_text='x' * length, audio_provenance=selected(16000), transcript_truncated=prior)
        [(_, values)] = registry.calls
        assert values['transcript_provenance'] is None and values['transcript_span'] is None
        assert values['transcript_truncated'] is expected
        assert len(values['transcript']) == min(length, 8192) and stt.calls == 0
    finally:
        await command.aclose()


@pytest.mark.asyncio
async def test_cancelled_observer_and_close_retain_actual_primary_without_late_observation():
    entered, release = threading.Event(), threading.Event()

    class Held(STT):
        def transcribe_with_confidence(self, audio, sample_rate=16000):
            entered.set()
            assert release.wait(3)
            return super().transcribe_with_confidence(audio, sample_rate)

    registry = Registry()
    command = processor(Held(), registry)
    pending = asyncio.create_task(command.process_command(np.zeros(16000, dtype=np.float32),
        use_parallel=False, audio_provenance=selected(16000)))
    closing = None
    try:
        await until(entered.is_set)
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        closing = command.begin_close()
        await asyncio.sleep(0)
        assert not closing.done() and registry.calls == []
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        if closing is not None:
            await closing
        await command.aclose()
    assert registry.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('orchestrated', [False, True])
@pytest.mark.parametrize('covers', [False, True])
async def test_router_passes_both_descriptors_only_for_matching_reused_text(orchestrated, covers):
    router = object.__new__(RequestRouter)
    router._memory_source = ContextVar('synthetic-source', default=None)
    router._response_sources = ContextVar('synthetic-response', default=())
    router._turn_outcome = ContextVar('synthetic-turn', default=None)
    router._closed = False
    router._source_policy = router.memory = router.llm_command_router = None
    router.orchestrator_enabled = orchestrated
    router._orchestrator = object() if orchestrated else None
    router._complete_turn = lambda result: None
    router._current_source_available = lambda: True
    router._active_sources = lambda: (router._memory_source.get(),)
    router._privacy = SimpleNamespace(source_for_capture=lambda capture_id, captured_at, room_id, **kwargs:
        MemorySource(capture_id, kwargs['conversation_id'], captured_at, kwargs['expires_at'], kwargs['origin']))
    calls = []

    async def dispatched(audio, **kwargs):
        calls.append((kwargs, PRODUCER_CONTEXT.get().sources))
        return {'text': '', 'response': '', 'intent': 'cancelled', 'success': False}

    router._process_command_orchestrated = router._process_command_legacy = dispatched
    audio = np.zeros(16000, dtype=np.float32)
    audio_source, text_source = selected(16000), selected(32000, 1)
    transcript = CapturedTranscript.from_audio('Synthetic reused text', TranscriptSource.WAKE, audio,
        transcript_provenance=text_source, transcript_span=FrameInterval(0, 32000), transcript_truncated=False)
    if not covers:
        audio[0] = .1
    source = memory_source(now=2_000_000_000.)
    event = CommandEvent(audio, 'synthetic-room', transcript=transcript, capture_complete=True,
        audio_provenance=audio_source, capture_id=source.capture_id, conversation_id=source.conversation_id,
        captured_at=source.captured_at, source_expires_at=source.expires_at)
    await router._process_received_command(event)
    [(values, sources)] = calls
    assert sources[0].capture_id == source.capture_id and sources[0].expires_at == source.expires_at
    assert values['audio_provenance'] == audio_source
    assert values['pretranscribed_text'] == ('Synthetic reused text' if covers else None)
    assert values['transcript_provenance'] == (text_source if covers else None)
    assert values['transcript_span'] == (FrameInterval(0, 32000) if covers else None)


class Wake:
    def __init__(self):
        self.trigger = True
        self.resets = 0

    def detect(self, audio):
        return ('nexa', 1.) if self.trigger else None

    def reset(self):
        self.resets += 1


def room_loop(channel=0):
    loop = object.__new__(MultiRoomAudioLoop)
    loop.sample_rate = 16000
    loop._capture_rooms = {'synthetic-room': SimpleNamespace(require_live=lambda: None)}
    loop.barge_in_enabled = False
    loop._preroll_maxchunks = 2
    loop.native_model_locks = {}
    loop._wake_vad_predictors = {}
    loop._guard = loop._wake_clip_writer = loop._loop = None
    loop.follow_up = SimpleNamespace(is_active=False, start_conversation=lambda: None)
    loop._should_accept_wakeword = lambda *args, **kwargs: True
    wake = Wake()
    echo = SimpleNamespace(is_safe_to_listen=True, should_process_audio=lambda audio: (True, 'speech'))
    room = RoomStream('synthetic-room', 0, wake, echo, capture_channel=channel)
    return loop, room


@pytest.mark.parametrize('width,actual', [(1, 0), (2, 1)])
def test_command_descriptor_records_actual_selected_column_after_fallback(width, actual):
    loop, room = room_loop(1)
    seed = opening(channels=width)
    raw = np.column_stack([np.full(1280, .1 + col * .1, dtype=np.float32) for col in range(width)])
    with capture_opening_context(seed):
        loop._make_audio_processor(room)(raw, len(raw), 0., None)
    audio = np.array(room.audio_buffer, dtype=np.float32)
    metadata = loop._capture_provenance(room, audio)
    assert metadata.selected_channel == actual and metadata.source_channels == width
    assert metadata.frame_count == audio.size == 1280
    assert np.array_equal(audio, raw[:, actual])


@pytest.mark.parametrize('packets,known', [(2, False), (3, True)])
def test_unknown_preroll_remains_unknown_until_its_actual_pcm_is_evicted(packets, known):
    loop, room = room_loop()
    seed = opening()
    process = loop._make_audio_processor(room)
    raw = np.full((1280, 2), .1, dtype=np.float32)
    for index in range(packets):
        room.wake_detector.trigger = index == packets - 1
        with capture_opening_context(None if index == 0 else seed):
            process(raw, len(raw), 0., None)
    metadata = loop._capture_provenance(room, np.array(room.audio_buffer, dtype=np.float32))
    assert (metadata is not None) is known
    assert len(room.audio_buffer) == 2560


@pytest.mark.parametrize('change', ['opening', 'column'])
def test_known_conflict_resets_capture_before_mixing_pcm(change):
    loop, room = room_loop()
    first = opening()
    process = loop._make_audio_processor(room)
    with capture_opening_context(first):
        process(np.full((1280, 2), .1, dtype=np.float32), 1280, 0., None)
    previous_generation = room.capture_generation
    second = opening('another-opening') if change == 'opening' else first
    room.capture_channel = 1 if change == 'column' else 0
    raw = np.full((1280, 2), .2, dtype=np.float32)
    with capture_opening_context(second):
        process(raw, 1280, 0., None)
    audio = np.array(room.audio_buffer, dtype=np.float32)
    metadata = loop._capture_provenance(room, audio)
    assert room.wake_detector.resets == 1 and room.capture_generation > previous_generation
    assert audio.size == 1280 and np.array_equal(audio, raw[:, room.capture_channel])
    assert metadata.source_ref.startswith(second.source_ref + ':')
    assert metadata.selected_channel == room.capture_channel


def test_early_snapshot_captures_its_own_pcm_descriptor_before_later_mutation():
    loop, room = room_loop()
    room.listening = True
    room.audio_buffer = list(np.ones(16000, dtype=np.float32))
    room.audio_metadata = [(16000, opening(), 0)]
    audio, metadata = loop._early_snapshot(room, room.capture_generation)
    room.audio_buffer.extend(np.ones(1280, dtype=np.float32))
    room.audio_metadata.append((1280, opening(), 0))
    assert audio.size == metadata.frame_count == 16000
    assert len(room.audio_buffer) == 17280


def test_opening_is_created_only_from_real_handle_properties_not_telemetry_fallbacks():
    loop, room = room_loop(1)
    holder = _CaptureHolder(CaptureObservation(generation=9, channels=2))
    loop._microphone_captures = {room.room_id: holder}
    loop._configure_stream(room, SimpleNamespace())
    assert holder.observation.channels == 2 and holder.opening is None
    loop._configure_stream(room, SimpleNamespace(samplerate=16000., channels=2, dtype='float32', device=0))
    assert holder.opening.channels == 2 and holder.opening.sample_rate_hz == 16000
    assert holder.opening.stream_generation == 9


def test_selected_channel_conflicts_are_detected_even_with_an_unknown_contribution():
    seed = opening()
    assert _pcm_context([(10, None, None), (10, seed, 0)])[0] is None
    assert _pcm_context([(10, None, None), (10, seed, 0), (10, seed, 1)])[2] is True


class WakeModel:
    def __init__(self):
        self.inputs = []

    def transcribe(self, audio, **kwargs):
        self.inputs.append(audio.copy())
        return [SimpleNamespace(text='nexa prende la luz', no_speech_prob=0., avg_logprob=-.1)], None


def whisper(decisions):
    model = WakeModel()
    detector = WhisperWakeDetector(model, ['nexa'], silence_end_ms=80, min_utterance_ms=1,
        pre_roll_ms=80, min_rms=0.)
    detector._loaded = True
    speech = iter(decisions)
    detector._is_speech = lambda audio: next(speech)
    return detector, model


def test_whisper_records_original_stt_window_and_distinct_cropped_pcm_anchor():
    detector, model = whisper([True, False])
    seed = opening()
    chunk = np.full(8000, .1, dtype=np.float32)
    assert detector.detect_with_provenance(chunk, opening=seed, selected_channel=1) is None
    assert detector.detect_with_provenance(chunk, opening=seed, selected_channel=1) == ('nexa', 1.)
    command_opening, channel, audio_metadata = detector.pop_pending_command_metadata()
    cropped = detector.pop_pending_command_audio()
    text_metadata, text_span, truncated = detector.pop_pending_transcript_metadata()
    assert detector.pop_pending_text() == 'nexa prende la luz'
    [actual_stt] = model.inputs
    assert command_opening == seed and channel == 1
    assert np.array_equal(cropped, actual_stt[6400:])
    assert text_metadata.frame_count == actual_stt.size == 24000
    assert audio_metadata.frame_count == cropped.size == 17600
    assert text_span == FrameInterval(0, 24000) and truncated is False
    assert text_metadata.source_ref != audio_metadata.source_ref
    assert text_metadata.selected_channel == audio_metadata.selected_channel == 1
    assert detector.pop_pending_command_metadata() == detector.pop_pending_transcript_metadata() == (None, None, None)
    detector.reset()
    assert not detector._pre_roll_metadata and detector._utterance_metadata == []
    assert detector._last_stt_provenance is None


def test_direct_whisper_caller_never_guesses_a_source_or_column():
    detector, _ = whisper([True, False])
    chunk = np.full(8000, .1, dtype=np.float32)
    detector.detect(chunk)
    assert detector.detect(chunk) == ('nexa', 1.)
    assert detector.pop_pending_command_metadata() == (None, None, None)
    assert detector.pop_pending_transcript_metadata() == (None, None, False)


def test_unknown_whisper_pcm_cannot_be_relabelled_by_a_later_known_chunk():
    detector, _ = whisper([True, False])
    chunk = np.full(8000, .1, dtype=np.float32)
    detector.detect(chunk)
    assert detector.detect_with_provenance(chunk, opening=opening(), selected_channel=1) == ('nexa', 1.)
    assert detector.pop_pending_command_metadata() == (None, None, None)
    assert detector.pop_pending_transcript_metadata() == (None, None, False)


def test_whisper_detaches_opening_and_dsp_before_retaining_pcm():
    detector, _ = whisper([True, False])
    seed = opening()
    chunk = np.full(8000, .1, dtype=np.float32)
    detector.detect_with_provenance(chunk, opening=seed, selected_channel=1)
    object.__setattr__(seed, 'stream_generation', 5)
    object.__setattr__(seed.dsp, 'declared_revision', 'synthetic-after')
    # An independent equal original descriptor keeps the owned old window coherent.
    assert detector.detect_with_provenance(chunk, opening=opening(), selected_channel=1) == ('nexa', 1.)
    provenance, _, _ = detector.pop_pending_transcript_metadata()
    assert provenance.stream_generation == 4 and provenance.dsp.declared_revision == 'synthetic-before'


def test_whisper_recurrent_fence_survives_unknown_pcm_eviction_and_resets_before_new_opening():
    detector, _ = whisper([False, False, False])
    events = []
    detector._vad = SimpleNamespace(reset_states=lambda: events.append('reset'))
    detector._is_speech = lambda audio: events.append('infer') or False
    chunk = np.full(8000, .1, dtype=np.float32)
    detector.detect_with_provenance(chunk, opening=opening('A'), selected_channel=0)
    detector.detect(chunk)  # deque maxlen=1 now contains only an unknown chunk.
    assert detector._pre_roll_metadata[0][0] is None
    detector.detect_with_provenance(chunk, opening=opening('B'), selected_channel=0)
    assert events == ['infer', 'infer', 'reset', 'infer']
    assert detector._pre_roll_metadata[0][0].source_ref == 'B'


@pytest.mark.asyncio
async def test_processor_owned_task_does_not_alias_metadata_while_primary_is_held():
    entered, release = threading.Event(), threading.Event()

    class Held(STT):
        def transcribe_with_confidence(self, audio, sample_rate=16000):
            entered.set()
            assert release.wait(3)
            return super().transcribe_with_confidence(audio, sample_rate)

    registry = Registry()
    command = processor(Held(), registry)
    original = selected(16000)
    pending = asyncio.create_task(command.process_command(np.zeros(16000, dtype=np.float32),
        use_parallel=False, audio_provenance=original))
    try:
        await until(entered.is_set)
        object.__setattr__(original, 'selected_channel', 1)
        object.__setattr__(original.dsp, 'declared_revision', 'synthetic-after')
    finally:
        release.set()
        await pending
        await command.aclose()
    [(_, values)] = registry.calls
    assert values['audio_provenance'].selected_channel == values['transcript_provenance'].selected_channel == 0
    assert values['audio_provenance'].dsp.declared_revision == 'synthetic-before'


@pytest.mark.asyncio
async def test_textual_wake_transports_explicit_ambient_pairs_without_changing_source_identity():
    received = []

    async def dispatch(event):
        received.append(event)
        return {}

    wake = TextualWakeDetector(dispatch, lambda room: 0., now_fn=lambda: 100.)
    audio = np.zeros(16000, dtype=np.float32)
    metadata = opening().project(lane=CaptureLane.AMBIENT, segment_generation=7,
        selected_channel=1, frame_count=audio.size, input_ref=uuid4().hex)
    source = memory_source(now=2_000_000_000.)
    assert await wake.maybe_dispatch('synthetic-room', 'nexa prende la luz', 'live', None, audio,
        vad_prob=1., memory_source=source, audio_provenance=metadata,
        transcript_provenance=metadata, transcript_span=FrameInterval(0, audio.size), transcript_truncated=False)
    [event] = received
    assert event.capture_id == source.capture_id and event.conversation_id == source.conversation_id
    assert event.source_expires_at == source.expires_at and event.captured_at == source.captured_at
    assert event.audio_provenance == event.transcript.transcript_provenance == metadata
    assert event.audio_provenance.lane is CaptureLane.AMBIENT
    assert event.transcript.covers(audio) and event.transcript.transcript_truncated is False
