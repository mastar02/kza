"""Unit tests for TrainingDatasetHub and offline needs_review HTML reviewer."""
from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.audio.clip_io import ArchiveChannelMode
from src.training.curated_dataset_exporter import CurationPolicy
from src.training.dataset_hub import HubConfig, TrainingDatasetHub, load_corrections_file
from tools.curate_ambient_dataset import main as curate_cli_main
from tools.training_review import build_review_html, render_word_diff


def _write_wav(path: Path, val: float = 0.2, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        data = np.full(3200, val, dtype=np.float32)
    else:
        data = np.stack(
            [np.full(3200, val, dtype=np.float32), np.full(3200, val * 0.5, dtype=np.float32)],
            axis=1,
        )
    sf.write(str(path), data, 16000, format="WAV")


def _seed_ambient_db(tmp_path: Path) -> tuple[Path, int, int]:
    _write_wav(tmp_path / "audio/escritorio/1.wav", val=0.3, channels=2)
    _write_wav(tmp_path / "audio/escritorio/2.wav", val=0.25, channels=2)
    db_path = tmp_path / "ambient.db"

    async def seed():
        store = AmbientStore(db_path=str(db_path), retention_hours=48.0)
        await store.init()
        u1 = await store.add(
            AmbientUtterance(
                room_id="escritorio",
                t0=1000.0,
                t1=1020.0,
                text="Nexa, prendé la luz del escritorio. Hoy es maravilloso, ¿no?",
                alt_text="Nexa, aprender la luz del escritorio. Hoy es maravilloso para morirse, ¿no?",
                agreement_ratio=0.78,
                speaker="gabriel",
                source="live",
                vad_prob=0.85,
                lang="es",
                lang_ok=True,
                audio_path="audio/escritorio/1.wav",
            )
        )
        u2 = await store.add(
            AmbientUtterance(
                room_id="escritorio",
                t0=1025.0,
                t1=1030.0,
                text="Buenos días, ¿todo bien?",
                alt_text="Buenos días, ¿todo bien?",
                agreement_ratio=1.0,
                speaker="gabriel",
                source="live",
                vad_prob=0.90,
                lang="es",
                lang_ok=True,
                audio_path="audio/escritorio/2.wav",
            )
        )
        await store.close()
        return u1, u2

    u1, u2 = asyncio.run(seed())
    return db_path, u1, u2


def test_hub_fails_closed_without_explicit_source(tmp_path):
    hub = TrainingDatasetHub()
    with pytest.raises(ValueError, match="hub_requires_explicit_selection"):
        hub.build(HubConfig(out_dir=tmp_path / "hub"))


def test_hub_unifies_all_four_sources_and_generates_needs_review_html(tmp_path):
    db_path, u1, u2 = _seed_ambient_db(tmp_path)

    # Extra WakeClipWriter buckets
    wake_dir = tmp_path / "wake_clips"
    _write_wav(wake_dir / "accepted" / "a1.wav", val=0.4)
    _write_wav(wake_dir / "missed" / "m1.wav", val=0.35)
    _write_wav(wake_dir / "rejected" / "r1.wav", val=0.1)

    # Extra STT correction audio
    corr_wav = tmp_path / "corr_audio" / "c1.wav"
    _write_wav(corr_wav, val=0.5)

    # Extra curated conversation JSONL
    ext_conv = tmp_path / "ext_conv.jsonl"
    ext_conv.write_text(
        json.dumps({"instruction": "prendé el aire", "input": "living", "output": "Listo, prendí el aire."}) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "hub"
    summary = TrainingDatasetHub().build(
        HubConfig(
            out_dir=out_dir,
            ambient_db=db_path,
            ambient_policy=CurationPolicy(approved_speakers=frozenset({"gabriel"})),
            audio_base=tmp_path,
            alt_metadata={
                u1: {
                    "alt_text": "Nexa, aprender la luz del escritorio. Hoy es maravilloso para morirse, ¿no?",
                    "agreement_ratio": 0.78,
                },
                u2: {"alt_text": "Buenos días, ¿todo bien?", "agreement_ratio": 1.0},
            },
            wake_clips_dir=wake_dir,
            stt_correction_records=(
                {"audio_path": str(corr_wav), "original_text": "nexa prende", "corrected_text": "Nexa, prendé"},
            ),
            conversation_jsonl_paths=(ext_conv,),
            channel_mode=ArchiveChannelMode.MONO_CH0,
        )
    )

    assert summary.stt_samples == 3  # 2 ambient + 1 stt_correction
    assert summary.stt_needs_review == 1  # only u1 (agreement 0.78 < 0.90)
    assert summary.stt_human_verified == 1
    assert summary.wake_positives == 3  # u1 + accepted/a1 + missed/m1
    assert summary.wake_negatives == 2  # u2 + rejected/r1
    assert summary.llm_conversations == 2  # 1 ambient conv + 1 external conv
    assert set(summary.sources_used) == {"ambient_db", "stt_corrections", "wake_clips", "conversation_jsonl"}
    assert stat.S_IMODE(out_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((out_dir / "hub_manifest.json").stat().st_mode) == 0o600

    # Reviewer HTML shows ONLY needs_review clip (u1) with highlighted diff
    html_path, count = build_review_html(out_dir)
    assert count == 1
    assert stat.S_IMODE(html_path.stat().st_mode) == 0o600
    content = html_path.read_text(encoding="utf-8")
    assert "para morirse" in content
    assert "diff-w" in content and "diff-p" in content

    # Apply exported corrections.json via CLI -> clears needs_review
    corr_json = tmp_path / "corrections.json"
    corr_json.write_text(
        json.dumps({str(u1): "Nexa, prendé la luz del escritorio. Hoy es un día maravilloso para morirse, ¿no?"}),
        encoding="utf-8",
    )
    assert load_corrections_file(corr_json)[u1].endswith("para morirse, ¿no?")

    out2 = tmp_path / "hub_corrected"
    rc = curate_cli_main([
        "--db", str(db_path),
        "--out", str(out2),
        "--audio-base", str(tmp_path),
        "--speaker", "gabriel",
        "--corrections", str(corr_json),
    ])
    assert rc == 0
    manifest2 = json.loads((out2 / "hub_manifest.json").read_text(encoding="utf-8"))
    assert manifest2["stt_needs_review"] == 0
    assert manifest2["stt_human_verified"] == 1


def test_render_word_diff_highlights_disagreements():
    w_html, p_html = render_word_diff("Hoy es maravilloso", "Hoy es para morirse")
    assert "Hoy es" in w_html
    assert '<mark class="diff-w">maravilloso</mark>' in w_html
    assert '<mark class="diff-p">para morirse</mark>' in p_html
