"""Unit tests for inline 'Nexa <action>' extraction and continuous segmenter overlap."""
from __future__ import annotations

import asyncio
import numpy as np

from src.ambient.inline_wake import extract_all_inline_commands, extract_inline_command
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.textual_wake import TextualWakeDetector


def test_extract_inline_command_isolates_mid_conversation_clause():
    text = "che mirá que ayer fui al súper, listo, Nexa prendé la luz del living, y después te cuento lo que pasó"
    match = extract_inline_command(text)
    assert match is not None
    assert match.embedded_in_conversation is True
    assert match.command_text == "Nexa prendé la luz del living"
    assert "ayer fui al súper" in match.prefix_text
    assert "después te cuento" in match.suffix_text


def test_extract_inline_command_handles_sentence_boundary_and_direct_command():
    direct = extract_inline_command("Nexa apagá la luz de la cocina")
    assert direct is not None
    assert direct.embedded_in_conversation is False
    assert direct.command_text == "Nexa apagá la luz de la cocina"

    with_period = extract_inline_command("estaba viendo la tele. Nexa bajá la persiana del living. Bueno che seguimos")
    assert with_period is not None
    assert with_period.embedded_in_conversation is True
    assert with_period.command_text == "Nexa bajá la persiana del living"


def test_extract_inline_command_does_not_cut_immediately_after_wake_period():
    match = extract_inline_command("ah sí, Nexa. Prendé el aire del living. Bueno dale")
    assert match is not None
    assert match.command_text == "Nexa. Prendé el aire del living"
    assert match.embedded_in_conversation is True


def test_extract_inline_command_respects_fuzzy_denylist_and_bigrams():
    assert extract_inline_command("che nena apagá la luz") is None
    assert extract_inline_command("el nexo entre ambos es claro") is None

    bigram = extract_inline_command("y te decía que next up prende la luz del escritorio, bueno nada")
    assert bigram is not None
    assert bigram.matched_variant == "next up"
    assert bigram.command_text == "next up prende la luz del escritorio"
    assert bigram.embedded_in_conversation is True


def test_textual_wake_dispatches_extracted_clause_with_inline_flag_and_speech_vad():
    dispatched = []

    async def fake_dispatch(event):
        dispatched.append(event)
        return {"success": True}

    detector = TextualWakeDetector(
        dispatch_fn=fake_dispatch,
        last_acoustic_command_ts_fn=lambda _room: 0.0,
        min_vad=0.50,
    )
    audio = np.zeros(1600, dtype=np.float32)

    # Diluted mean vad_prob=0.38 (below 0.50), but speech-only vad_speech_prob=0.82 -> passes!
    ok = asyncio.run(
        detector.maybe_dispatch(
            room_id="living",
            text="che listo, Nexa prendé la luz del living, bueno te seguía contando",
            source="live",
            speaker="gabriel",
            audio=audio,
            vad_prob=0.38,
            vad_speech_prob=0.82,
        )
    )
    assert ok is True
    assert len(dispatched) == 1
    ev = dispatched[0]
    assert ev.wake_text == "Nexa prendé la luz del living"
    assert ev.inline_conversation is True
    assert ev.textual_wake is True


def test_segmenter_overlap_carries_trailing_audio_and_tracks_vad_speech_prob():
    # Each chunk is 1280 samples = 80ms @ 16kHz. max_segment_s=0.48s = 6 chunks.
    # First 6 chunks (0..5) are speech -> forced max_segment_s close (complete=False),
    # carrying chunks 4 & 5 (values 0.05, 0.06) as overlap into the next segment.
    # Next 4 chunks: 2 speech + 2 silence (160ms >= close_silence_ms) -> complete=True.
    probs = iter([0.9, 0.85, 0.8, 0.9, 0.88, 0.87, 0.92, 0.91, 0.05, 0.05])
    seg = UtteranceSegmenter(
        vad_predict=lambda _chunk: next(probs, 0.0),
        sample_rate=16000,
        speech_threshold=0.5,
        close_silence_ms=160,
        preroll_ms=0,
        max_segment_s=0.48,
        min_speech_ms=160,
        overlap_ms=160,  # 2 chunks carried into next segment on forced max_segment_s close
    )

    emitted = []
    for i in range(10):
        chunk = np.full((1280, 2), float(i + 1) * 0.01, dtype=np.float32)
        emitted.extend(seg.feed(ts=100.0 + i * 0.08, chunk=chunk))

    assert len(emitted) == 2
    first, second = emitted[0], emitted[1]
    assert first.complete is False
    assert first.vad_speech_prob is not None and first.vad_speech_prob > 0.8

    # Second segment starts with the 2 overlapped chunks (indices 4, 5 -> values 0.05, 0.06)
    assert second.complete is True
    assert np.isclose(second.audio[0, 0], 0.05)
    assert np.isclose(second.audio[1280, 0], 0.06)
    # Speech-only VAD is much higher than diluted overall vad_prob (which includes silence tail)
    assert second.vad_speech_prob is not None
    assert second.vad_speech_prob > second.vad_prob + 0.25


def test_command_capture_inline_skips_followup_and_incomplete_releases_dedup():
    from unittest.mock import MagicMock

    from src.pipeline.ambient_guard import AmbientGuard, AmbientGuardConfig
    from src.pipeline.command_event import CommandEvent
    from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop

    follow_up = MagicMock()
    guard = AmbientGuard(AmbientGuardConfig(enabled=True))
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=follow_up,
        ambient_guard=guard,
    )

    async def accepted_cb(_event):
        return {"success": True, "text": "Nexa prendé la luz del living"}

    loop.on_command(accepted_cb)
    inline_ev = CommandEvent(
        audio=np.zeros(1600, dtype=np.float32),
        room_id="living",
        textual_wake=True,
        inline_conversation=True,
    )
    asyncio.run(loop._dispatch_command(inline_ev))
    follow_up.start_conversation.assert_not_called()

    async def incomplete_cb(_event):
        return {"success": False, "error": "gate_rejected:incomplete_capture"}

    loop.on_command(incomplete_cb)
    acoustic_ev = CommandEvent(audio=np.zeros(1600, dtype=np.float32), room_id="living")
    asyncio.run(loop._dispatch_command(acoustic_ev))
    assert loop.last_command_dispatch_ts("living") == 0.0


def test_extract_all_inline_commands_handles_back_to_back_commands_with_or_without_punctuation():
    punctuated = (
        "Hola, hola, hola, hola. Nexa, prendé la luz del escritorio. "
        "Nexa, ¿qué temperatura hace? Dale a tu cuerpo, alegría, macarena."
    )
    matches = extract_all_inline_commands(punctuated)
    assert [m.command_text for m in matches] == [
        "Nexa, prendé la luz del escritorio",
        "Nexa, ¿qué temperatura hace?",
    ]
    assert all(m.embedded_in_conversation for m in matches)

    unpunctuated = (
        "hola hola nexa prendé la luz del escritorio nexa que temperatura hace? "
        "bueno te seguía contando"
    )
    matches2 = extract_all_inline_commands(unpunctuated)
    assert [m.command_text for m in matches2] == [
        "nexa prendé la luz del escritorio",
        "nexa que temperatura hace?",
    ]


def test_textual_wake_dispatches_multiple_inline_commands_in_single_utterance():
    dispatched = []
    missed_calls = []

    async def fake_dispatch(event):
        dispatched.append(event)
        return {"success": True}

    detector = TextualWakeDetector(
        dispatch_fn=fake_dispatch,
        last_acoustic_command_ts_fn=lambda _room: 0.0,
        min_vad=0.50,
        on_missed_fn=lambda room, _audio: missed_calls.append(room),
    )
    audio = np.zeros(1600, dtype=np.float32)
    text = (
        "Hola, hola. Nexa, prendé la luz del escritorio. "
        "Nexa, ¿qué temperatura hace? ¿Cómo estás, che?"
    )
    ok = asyncio.run(
        detector.maybe_dispatch(
            room_id="escritorio",
            text=text,
            source="live",
            speaker="gabriel",
            audio=audio,
            vad_prob=0.80,
        )
    )
    assert ok is True
    assert len(missed_calls) == 1
    assert [ev.wake_text for ev in dispatched] == [
        "Nexa, prendé la luz del escritorio",
        "Nexa, ¿qué temperatura hace?",
    ]
    assert all(ev.inline_conversation for ev in dispatched)
