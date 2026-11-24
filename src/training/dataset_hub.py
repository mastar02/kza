"""Unified privacy-safe TrainingDatasetHub combining all curated training sources.

Consolidates the four previously disconnected household data silos into a single
canonical, fail-closed, private (``0700``/``0600``) curated layout outside Git:

  - ``stt/metadata.jsonl`` + ``stt/audio/*.wav`` (16 kHz mono ``ch0``)
    from ``ambient.db`` (with ``alt_text_parakeet``, ``agreement_ratio``,
    ``needs_review``) plus explicitly supplied human ``STTCorrection`` records.
  - ``wakeword/positive/*.wav`` + ``wakeword/negative/*.wav`` + ``manifest.json``
    from ambient utterances plus explicitly supplied ``WakeClipWriter`` buckets
    (``accepted/`` + ``missed/`` -> positive, ``rejected/`` -> negative).
  - ``llm/conversations.jsonl``
    from ambient continuous conversations (with multi-inline ``Nexa`` actions)
    plus explicitly supplied curated conversation JSONL files.
  - ``hub_manifest.json``
    content-free summary and readiness counts for the three modular trainers
    (Whisper LoRA, Wakeword ONNX, LLM QLoRA on 2x RTX 3070).

Privacy invariant:
  Fails closed unless at least one explicit operator approval filter or
  explicitly curated source is supplied. Preserves the ``<= 48h`` TTL contract
  of ``AmbientStore`` / ``AudioArchiver``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from src.audio.clip_io import (
    TARGET_SAMPLE_RATE,
    ArchiveChannelMode,
    resample_linear_16k,
    select_archive_audio,
    write_private_audio,
    write_private_bytes,
)
from src.training.curated_dataset_exporter import (
    CurationPolicy,
    export_curated_datasets,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubConfig:
    """Explicit opt-in inputs for one curated dataset hub build."""

    out_dir: Path
    ambient_db: Path | None = None
    ambient_policy: CurationPolicy | None = None
    audio_base: Path | None = None
    corrections: dict[int, str] = field(default_factory=dict)
    alt_metadata: dict[int, dict] = field(default_factory=dict)
    # Optional explicitly approved extra sources (never auto-discovered):
    wake_clips_dir: Path | None = None
    stt_correction_records: tuple[dict, ...] = ()
    conversation_jsonl_paths: tuple[Path, ...] = ()
    channel_mode: ArchiveChannelMode = ArchiveChannelMode.MONO_CH0

    def has_explicit_source(self) -> bool:
        ambient_ok = (
            self.ambient_db is not None
            and self.ambient_policy is not None
            and not self.ambient_policy.requires_explicit_selection()
        )
        return bool(
            ambient_ok
            or self.wake_clips_dir is not None
            or self.stt_correction_records
            or self.conversation_jsonl_paths
        )


@dataclass(frozen=True)
class HubBuildSummary:
    """Content-free summary of a curated dataset hub build."""

    out_dir: str
    stt_samples: int
    stt_needs_review: int
    stt_human_verified: int
    wake_positives: int
    wake_negatives: int
    llm_conversations: int
    skipped_missing_audio: int
    sources_used: tuple[str, ...]
    ready_for_stt_lora: bool
    ready_for_wakeword: bool
    ready_for_llm_qlora: bool


def load_corrections_file(path: Path) -> dict[int, str]:
    """Load ``{utterance_id: corrected_text}`` from a reviewer JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corrections_file_must_be_object")
    parsed: dict[int, str] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            text = str(val.get("text") or val.get("corrected_text") or "").strip()
        else:
            text = str(val or "").strip()
        if text:
            parsed[int(key)] = text
    return parsed


def _read_normalized_mono(audio_path: Path, channel_mode: ArchiveChannelMode) -> tuple[np.ndarray, float]:
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    resampled = resample_linear_16k(data, sr)
    selected = select_archive_audio(resampled, channel_mode)
    if selected.ndim == 2:
        selected = np.ascontiguousarray(selected[:, 0])
    duration_s = round(selected.shape[0] / TARGET_SAMPLE_RATE, 3)
    return selected, duration_s


def _append_jsonl_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class TrainingDatasetHub:
    """Orchestrates fail-closed curation across ambient, wake clips, STT and LLM sources."""

    MIN_STT_LORA_SAMPLES = 50
    MIN_WAKE_POSITIVES = 30
    MIN_WAKE_NEGATIVES = 30
    MIN_LLM_CONVERSATIONS = 20

    def build(self, config: HubConfig) -> HubBuildSummary:
        """Build or refresh a private curated dataset hub directory."""
        if not config.has_explicit_source():
            raise ValueError("hub_requires_explicit_selection")

        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(out_dir, 0o700)

        sources_used: list[str] = []
        skipped_audio = 0

        # 1. Ambient continuous conversations (ambient.db)
        if config.ambient_db is not None and config.ambient_policy is not None:
            amb_summary = export_curated_datasets(
                config.ambient_db,
                out_dir,
                config.ambient_policy,
                audio_base=config.audio_base,
                corrections=config.corrections,
                alt_metadata=config.alt_metadata,
            )
            skipped_audio += amb_summary.skipped_missing_audio
            sources_used.append("ambient_db")

        stt_records = _append_jsonl_lines(out_dir / "stt" / "metadata.jsonl")
        for rec in stt_records:
            rec.setdefault("source_origin", "ambient_db")
            if rec.get("utterance_id") in config.corrections:
                rec["human_verified"] = True
                rec["needs_review"] = False

        wake_manifest_path = out_dir / "wakeword" / "manifest.json"
        if wake_manifest_path.is_file():
            wake_manifest = json.loads(wake_manifest_path.read_text(encoding="utf-8"))
            wake_items: list[dict] = list(wake_manifest.get("items", []))
        else:
            wake_items = []

        llm_records = _append_jsonl_lines(out_dir / "llm" / "conversations.jsonl")

        # 2. Explicit human STT corrections (e.g. from STTCorrectionCollector export)
        if config.stt_correction_records:
            sources_used.append("stt_corrections")
            for idx, item in enumerate(config.stt_correction_records, start=1):
                corrected = str(item.get("corrected_text") or item.get("text") or "").strip()
                raw_audio = item.get("audio_path")
                if not corrected or not raw_audio:
                    skipped_audio += 1
                    continue
                audio_path = Path(raw_audio)
                if not audio_path.is_absolute() and config.audio_base is not None:
                    audio_path = config.audio_base / audio_path
                if not audio_path.is_file():
                    skipped_audio += 1
                    continue
                try:
                    mono, duration_s = _read_normalized_mono(audio_path, config.channel_mode)
                except Exception as error:
                    logger.warning("Skipping unreadable STT correction audio %d: %s", idx, type(error).__name__)
                    skipped_audio += 1
                    continue
                rel = Path("stt") / "audio" / f"corr_{idx}.wav"
                write_private_audio(
                    out_dir / rel, np.clip(mono, -1.0, 1.0), TARGET_SAMPLE_RATE, fmt="WAV", subtype="PCM_16"
                )
                stt_records.append(
                    {
                        "utterance_id": f"corr_{idx}",
                        "audio_path": str(rel),
                        "text": corrected,
                        "original_text": str(item.get("original_text") or "").strip() or None,
                        "room_id": item.get("room_id"),
                        "speaker": item.get("user_id") or item.get("speaker") or "unknown",
                        "duration_s": duration_s,
                        "source_origin": "stt_correction",
                        "human_verified": True,
                        "needs_review": False,
                    }
                )

        # 3. Explicit wakeword clip buckets (accepted/ + missed/ -> positive, rejected/ -> negative)
        if config.wake_clips_dir is not None and config.wake_clips_dir.is_dir():
            sources_used.append("wake_clips")
            bucket_map = (
                ("accepted", "positive"),
                ("missed", "positive"),
                ("positive", "positive"),
                ("rejected", "negative"),
                ("negative", "negative"),
            )
            clip_counter = 0
            for src_bucket, dst_label in bucket_map:
                src_dir = config.wake_clips_dir / src_bucket
                if not src_dir.is_dir():
                    continue
                for wav_file in sorted(src_dir.glob("*.wav")):
                    try:
                        mono, duration_s = _read_normalized_mono(wav_file, config.channel_mode)
                    except Exception as error:
                        logger.warning("Skipping unreadable wake clip %s: %s", wav_file.name, type(error).__name__)
                        skipped_audio += 1
                        continue
                    clip_counter += 1
                    rel = Path("wakeword") / dst_label / f"clip_{src_bucket}_{clip_counter}.wav"
                    write_private_audio(
                        out_dir / rel, np.clip(mono, -1.0, 1.0), TARGET_SAMPLE_RATE, fmt="WAV", subtype="PCM_16"
                    )
                    wake_items.append(
                        {
                            "utterance_id": f"clip_{src_bucket}_{clip_counter}",
                            "label": dst_label,
                            "audio_path": str(rel),
                            "duration_s": duration_s,
                            "source_origin": f"wake_clips:{src_bucket}",
                        }
                    )

        # 4. Explicit curated LLM conversation JSONL files
        if config.conversation_jsonl_paths:
            sources_used.append("conversation_jsonl")
            for conv_path in config.conversation_jsonl_paths:
                if not conv_path.is_file():
                    continue
                for idx, entry in enumerate(_append_jsonl_lines(conv_path), start=1):
                    if "turns" in entry:
                        entry.setdefault("source_origin", "conversation_jsonl")
                        llm_records.append(entry)
                    elif "instruction" in entry and "output" in entry:
                        llm_records.append(
                            {
                                "conversation_id": f"ext_{conv_path.stem}_{idx}",
                                "turn_count": 1,
                                "source_origin": "conversation_jsonl",
                                "inline_actions": [],
                                "turns": [
                                    {
                                        "utterance_id": f"ext_{idx}",
                                        "speaker": "user",
                                        "text": str(entry["instruction"]),
                                        "assistant_response": str(entry["output"]),
                                        "context": str(entry.get("input") or ""),
                                    }
                                ],
                            }
                        )

        # Persist unified manifests (0600)
        stt_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in stt_records)
        if stt_jsonl:
            stt_jsonl += "\n"
        write_private_bytes(out_dir / "stt" / "metadata.jsonl", stt_jsonl.encode("utf-8"))

        wake_pos = sum(1 for item in wake_items if item.get("label") == "positive")
        wake_neg = sum(1 for item in wake_items if item.get("label") == "negative")
        write_private_bytes(
            wake_manifest_path,
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "channel_mode": str(config.channel_mode),
                    "positives": wake_pos,
                    "negatives": wake_neg,
                    "items": wake_items,
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        llm_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in llm_records)
        if llm_jsonl:
            llm_jsonl += "\n"
        write_private_bytes(out_dir / "llm" / "conversations.jsonl", llm_jsonl.encode("utf-8"))

        stt_needs_review = sum(1 for r in stt_records if r.get("needs_review") is True)
        stt_verified = sum(1 for r in stt_records if r.get("human_verified") is True)
        summary = HubBuildSummary(
            out_dir=str(out_dir),
            stt_samples=len(stt_records),
            stt_needs_review=stt_needs_review,
            stt_human_verified=stt_verified,
            wake_positives=wake_pos,
            wake_negatives=wake_neg,
            llm_conversations=len(llm_records),
            skipped_missing_audio=skipped_audio,
            sources_used=tuple(sources_used),
            ready_for_stt_lora=len(stt_records) >= self.MIN_STT_LORA_SAMPLES,
            ready_for_wakeword=(
                wake_pos >= self.MIN_WAKE_POSITIVES and wake_neg >= self.MIN_WAKE_NEGATIVES
            ),
            ready_for_llm_qlora=len(llm_records) >= self.MIN_LLM_CONVERSATIONS,
        )
        write_private_bytes(
            out_dir / "hub_manifest.json",
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **asdict(summary),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )
        return summary
