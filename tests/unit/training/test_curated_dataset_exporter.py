"""Unit tests for curated dataset exporter (STT + WakeWord + LLM) and CLI."""
from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.clip_io import ArchiveChannelMode
from src.training.curated_dataset_exporter import CurationPolicy, export_curated_datasets
from tools.curate_ambient_dataset import main as curate_cli_main


def _make_2ch_flac(path: Path, val_ch0: float = 0.1, val_ch1: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ch0 = np.full(3200, val_ch0, dtype=np.float32)
    ch1 = np.full(3200, val_ch1, dtype=np.float32)
    sf.write(str(path), np.stack([ch0, ch1], axis=1), 16000, format="FLAC")


def _seed_db(tmp_path: Path) -> Path:
    _make_2ch_flac(tmp_path / "audio/living/1.flac", 0.1, 0.4)
    _make_2ch_flac(tmp_path / "audio/living/2.flac", 0.2, 0.5)
    _make_2ch_flac(tmp_path / "audio/living/3.flac", 0.3, 0.6)

    db_path = tmp_path / "ambient.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE utterances (
            id INTEGER PRIMARY KEY, room_id TEXT, t0 REAL, t1 REAL,
            text TEXT, speaker TEXT, source TEXT, vad_prob REAL,
            lang TEXT, lang_ok INTEGER, audio_path TEXT, conversation_id TEXT
        )"""
    )
    rows = [
        (1, "living", 100.0, 102.0, "che mirá, Nexa prendé la luz del living, bueno seguimos",
         "gabriel", "live", 0.85, "es", 1, "audio/living/1.flac", "conv-1"),
        (2, "living", 103.0, 105.0, "y después mañana vamos al supermercado",
         "gabriel", "live", 0.80, "es", 1, "audio/living/2.flac", "conv-1"),
        (3, "living", 110.0, 112.0, "suscríbete al canal",
         "unknown", "tv", 0.70, "es", 1, "audio/living/3.flac", "conv-2"),
    ]
    conn.executemany("INSERT INTO utterances VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db_path


def test_curation_fails_closed_without_explicit_selection(tmp_path):
    db_path = _seed_db(tmp_path)
    with pytest.raises(ValueError, match="curation_requires_explicit_selection"):
        export_curated_datasets(db_path, tmp_path / "out", CurationPolicy())


def test_export_curated_datasets_builds_three_targets_and_enforces_privacy(tmp_path):
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / "curated"

    policy = CurationPolicy(
        approved_speakers=frozenset({"gabriel"}),
        spanish_only=True,
        channel_mode=ArchiveChannelMode.MONO_CH1,
    )
    summary = export_curated_datasets(
        db_path,
        out_dir,
        policy,
        audio_base=tmp_path,
        corrections={2: "y después mañana vamos al súper"},
    )

    assert summary.stt_samples == 2
    assert summary.wake_positives == 1
    assert summary.wake_negatives == 1
    assert summary.llm_conversations == 1
    assert summary.skipped_missing_audio == 0

    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((out_dir / "stt" / "metadata.jsonl").stat().st_mode) == 0o600

    # Verify ch1 (0.4) was extracted instead of ch0 (0.1)
    wav_data, sr = sf.read(str(out_dir / "stt" / "audio" / "utt_1.wav"), dtype="float32")
    assert sr == 16000 and wav_data.ndim == 1
    assert np.allclose(wav_data, 0.4, atol=1e-2)

    # Verify human correction applied and TV excluded
    stt_rows = [
        json.loads(line)
        for line in (out_dir / "stt" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["utterance_id"] for r in stt_rows] == [1, 2]
    assert stt_rows[1]["text"] == "y después mañana vamos al súper"

    # Verify wakeword buckets
    assert (out_dir / "wakeword" / "positive" / "utt_1.wav").is_file()
    assert (out_dir / "wakeword" / "negative" / "utt_2.wav").is_file()

    # Verify LLM conversation export preserves inline action metadata
    llm_rows = [
        json.loads(line)
        for line in (out_dir / "llm" / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(llm_rows) == 1
    assert llm_rows[0]["conversation_id"] == "conv-1"
    assert llm_rows[0]["inline_actions"] == [
        {"utterance_id": 1, "command": "Nexa prendé la luz del living"}
    ]


def test_curate_cli_runs_end_to_end(tmp_path, capsys):
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / "cli_out"
    rc = curate_cli_main([
        "--db", str(db_path),
        "--out", str(out_dir),
        "--audio-base", str(tmp_path),
        "--conversation-id", "conv-1",
        "--channel-mode", "mono_ch1",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stt_samples"] == 2
    assert payload["wake_positives"] == 1
    assert payload["llm_conversations"] == 1


def test_export_includes_parakeet_alt_text_review_flag_and_multi_inline_commands(tmp_path):
    import asyncio
    from src.ambient.store import AmbientStore
    from src.ambient.types import AmbientUtterance

    _make_2ch_flac(tmp_path / "audio/escritorio/1.flac", 0.3, 0.1)
    db_path = tmp_path / "ambient_new.db"

    async def seed():
        store = AmbientStore(db_path=str(db_path), retention_hours=48.0)
        await store.init()
        utt_id = await store.add(
            AmbientUtterance(
                room_id="escritorio",
                t0=1000.0,
                t1=1020.0,
                text=(
                    "Hola. Nexa, prendé la luz del escritorio. "
                    "Nexa, ¿qué temperatura hace? Hoy es un día maravilloso, ¿no?"
                ),
                alt_text=(
                    "Hola. Nexa, aprender a la luz del escritorio. "
                    "Nexa, ¿qué temperatura hace? Hoy es un día maravilloso para morirse, ¿no?"
                ),
                agreement_ratio=0.79,
                speaker="gabriel",
                source="live",
                vad_prob=0.85,
                lang="es",
                lang_ok=True,
                audio_path="audio/escritorio/1.flac",
            )
        )
        await store.close()
        return utt_id

    utt_id = asyncio.run(seed())
    # Verify AmbientStore stays strictly compliant with SCHEMA_CONTRACT_EPOCH=2
    from scripts import stage_runtime
    conn = sqlite3.connect(db_path)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        stage_runtime._schema_contract(conn, names, "data/ambient.db")
    finally:
        conn.close()

    out_dir = tmp_path / "curated_new"
    summary = export_curated_datasets(
        db_path,
        out_dir,
        CurationPolicy(approved_utterance_ids=frozenset({utt_id})),
        audio_base=tmp_path,
        alt_metadata={
            utt_id: {
                "alt_text": (
                    "Hola. Nexa, aprender a la luz del escritorio. "
                    "Nexa, ¿qué temperatura hace? Hoy es un día maravilloso para morirse, ¿no?"
                ),
                "agreement_ratio": 0.79,
            }
        },
    )
    assert summary.stt_samples == 1
    assert summary.wake_positives == 1

    # Default channel_mode is mono_ch0 (0.3)
    wav_data, _ = sf.read(str(out_dir / "stt" / "audio" / f"utt_{utt_id}.wav"), dtype="float32")
    assert np.allclose(wav_data, 0.3, atol=1e-2)

    stt_row = json.loads((out_dir / "stt" / "metadata.jsonl").read_text(encoding="utf-8").strip())
    assert "para morirse" in stt_row["alt_text_parakeet"]
    assert stt_row["agreement_ratio"] == 0.79
    assert stt_row["needs_review"] is True

    llm_row = json.loads((out_dir / "llm" / "conversations.jsonl").read_text(encoding="utf-8").strip())
    assert llm_row["inline_actions"] == [
        {"utterance_id": utt_id, "command": "Nexa, prendé la luz del escritorio"},
        {"utterance_id": utt_id, "command": "Nexa, ¿qué temperatura hace?"},
    ]
    assert llm_row["turns"][0]["needs_review"] is True
