"""Unit tests for tools/stt_ab_compare.py (2ch split + 3-engine A/B + HTML + verdict)."""
from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import numpy as np
import soundfile as sf

from src.stt.whisper_fast import STTResult
from tools.stt_ab_compare import (
    ClipSpec,
    collect_clips_from_db,
    collect_clips_from_dir,
    evaluate_clips,
    main,
    read_clip_channels,
    summarize_verdict,
)


class _DummyEngine:
    def __init__(self, responses: dict[str, str], elapsed_ms: float = 25.0):
        self._responses = responses
        self._elapsed_ms = elapsed_ms
        self.calls = 0

    def transcribe_with_confidence(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        self.calls += 1
        # Distinguish ch0 (low rms ~0.05) vs ch1 (higher rms ~0.3) vs silence (<0.005)
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms < 0.005:
            key = "silence"
        elif rms < 0.15:
            key = "ch0"
        else:
            key = "ch1"
        return STTResult(
            text=self._responses.get(key, ""),
            elapsed_ms=self._elapsed_ms,
            compression_ratio=0.95,
        )


def _write_2ch_flac(path: Path, ch0_amp: float = 0.08, ch1_amp: float = 0.35, sr: int = 16000) -> None:
    n = int(sr * 0.2)
    t = np.linspace(0, 0.2, n, endpoint=False, dtype=np.float32)
    ch0 = ch0_amp * np.sin(2 * np.pi * 440 * t)
    ch1 = ch1_amp * np.sin(2 * np.pi * 660 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([ch0, ch1], axis=1), sr, format="FLAC")


def test_read_clip_channels_splits_2ch_and_resamples(tmp_path):
    clip_path = tmp_path / "sample.wav"
    n = 4800  # 0.1s at 48kHz
    ch0 = np.full(n, 0.1, dtype=np.float32)
    ch1 = np.full(n, 0.4, dtype=np.float32)
    sf.write(str(clip_path), np.stack([ch0, ch1], axis=1), 48000, format="WAV")

    channels, dur = read_clip_channels(clip_path)
    assert set(channels.keys()) == {"ch0", "ch1"}
    assert len(channels["ch0"]) == 1600
    assert abs(dur - 0.1) < 0.01
    assert np.allclose(channels["ch0"], 0.1, atol=1e-2)
    assert np.allclose(channels["ch1"], 0.4, atol=1e-2)


def test_collect_clips_from_dir_discovers_rooms_and_limits(tmp_path):
    _write_2ch_flac(tmp_path / "living" / "1.flac")
    _write_2ch_flac(tmp_path / "cocina" / "2.flac")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    all_clips = collect_clips_from_dir(tmp_path, limit=0)
    assert len(all_clips) == 2
    assert {c.room_id for c in all_clips} == {"living", "cocina"}

    limited = collect_clips_from_dir(tmp_path, limit=1)
    assert len(limited) == 1


def test_collect_clips_from_db_stratifies_and_skips_tv(tmp_path):
    audio_rel = Path("audio/living/10.flac")
    _write_2ch_flac(tmp_path / audio_rel)
    db_path = tmp_path / "ambient.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE utterances (
            id INTEGER PRIMARY KEY, room_id TEXT, vad_prob REAL,
            text TEXT, source TEXT, audio_path TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO utterances VALUES (10, 'living', 0.72, 'nexa prende la luz', 'live', ?)",
        (str(audio_rel),),
    )
    conn.execute(
        "INSERT INTO utterances VALUES (11, 'living', 0.85, 'tv noise', 'tv', ?)",
        (str(audio_rel),),
    )
    conn.commit()
    conn.close()

    clips = collect_clips_from_db(db_path, per_bucket=5, seed=7, audio_base=tmp_path)
    assert len(clips) == 1
    assert clips[0].clip_id == "utt_10"
    assert clips[0].room_id == "living"
    assert clips[0].vad_prob == 0.72


def test_evaluate_clips_emits_2ch_wavs_hybrid_veto_and_html(tmp_path):
    speech_clip = tmp_path / "in" / "living" / "speech.flac"
    silence_clip = tmp_path / "in" / "living" / "silence.flac"
    _write_2ch_flac(speech_clip, ch0_amp=0.08, ch1_amp=0.35)
    _write_2ch_flac(silence_clip, ch0_amp=0.0, ch1_amp=0.0)

    parakeet = _DummyEngine({"ch0": "apaga la luz", "ch1": "apaga la luz living", "silence": ""})
    whisper = _DummyEngine(
        {"ch0": "apagá la luz", "ch1": "apagá la luz del living", "silence": "¡Gracias por ver!"}
    )

    out_dir = tmp_path / "out"
    clips = [
        ClipSpec(clip_id="c_speech", source_path=speech_clip, room_id="living", vad_prob=0.81),
        ClipSpec(clip_id="c_silence", source_path=silence_clip, room_id="living", vad_prob=0.05),
    ]
    payload = evaluate_clips(
        clips,
        out_dir,
        parakeet=parakeet,
        whisper=whisper,
        langid_fn=lambda txt: ("es", 0.99) if "luz" in txt else ("en", 0.6),
    )

    assert payload["clip_count"] == 2
    assert (out_dir / "comparison.json").is_file()
    assert (out_dir / "index.html").is_file()
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((out_dir / "comparison.json").stat().st_mode) == 0o600

    # Verify both split WAV files exist and are playable mono 16k
    for cid in ("c_speech", "c_silence"):
        for ch in ("ch0", "ch1"):
            wav_file = out_dir / "audio" / f"{cid}_{ch}.wav"
            assert wav_file.is_file()
            data, sr = sf.read(str(wav_file))
            assert sr == 16000 and data.ndim == 1

    speech_item = payload["items"][0]
    assert speech_item["channels"]["ch1"]["engines"]["whisper"]["text"] == "apagá la luz del living"
    assert speech_item["channels"]["ch1"]["engines"]["hybrid"]["text"] == "apagá la luz del living"
    assert speech_item["channels"]["ch1"]["engines"]["hybrid"]["vetoed"] is False

    silence_item = payload["items"][1]
    assert silence_item["channels"]["ch0"]["engines"]["whisper"]["text"] == "¡Gracias por ver!"
    assert silence_item["channels"]["ch0"]["engines"]["parakeet"]["text"] == ""
    assert silence_item["channels"]["ch0"]["engines"]["hybrid"]["text"] == ""
    assert silence_item["channels"]["ch0"]["engines"]["hybrid"]["vetoed"] is True

    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "c_speech" in html
    assert "Exportar verdict.json" in html
    assert payload["summary"]["ch1"]["hybrid"]["veto_pct"] == 50.0


def test_summarize_verdict_and_cli(tmp_path, capsys):
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "run_id": "ab_test",
                "clip_count": 3,
                "votes": {
                    "c1": {"channel_winner": "ch1", "engine_winner": "hybrid", "note": "mucho más limpio"},
                    "c2": {"channel_winner": "ch1", "engine_winner": "hybrid"},
                    "c3": {"channel_winner": "ch0", "engine_winner": "whisper"},
                },
            }
        ),
        encoding="utf-8",
    )

    agg = summarize_verdict(json.loads(verdict_path.read_text(encoding="utf-8")))
    assert agg["channel_leader"] == "ch1"
    assert agg["engine_leader"] == "hybrid"
    assert agg["voted_items"] == 3
    assert agg["notes_count"] == 1

    rc = main(["--summarize", str(verdict_path)])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["engine_leader"] == "hybrid"


def test_build_engines_retires_cleanly_under_resource_session(monkeypatch):
    from types import SimpleNamespace

    from src.core.resource_retirement import ResourceSession
    from tools.stt_ab_compare import _build_engines

    closed = []

    class _StubBackend:
        def __init__(self, name):
            self.name = name
            self.loaded = False

        def load(self):
            self.loaded = True

        def close(self):
            closed.append(self.name)

    monkeypatch.setattr("src.stt.parakeet_stt.ParakeetSTT", lambda **kw: _StubBackend("parakeet"))
    monkeypatch.setattr("src.stt.whisper_fast.FastWhisperSTT", lambda **kw: _StubBackend("whisper"))

    args = SimpleNamespace(
        parakeet_model="pk",
        language="es",
        whisper_model="wh",
        whisper_device="cpu",
        whisper_compute="int8",
        whisper_beam_size=1,
        whisper_prompt=None,
    )
    with ResourceSession(error_prefix="test_ab") as session:
        pk, wh = _build_engines(args, session)
        assert pk.loaded and wh.loaded

    assert set(closed) == {"parakeet", "whisper"}
