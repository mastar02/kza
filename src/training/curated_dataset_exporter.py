"""Privacy-safe curated dataset exporter from reviewed ambient conversations.

Preserves the ``<= 48h`` TTL contract of ``AmbientStore`` / ``AudioArchiver``:
raw household captures expire normally unless explicitly curated and promoted
by the operator (by conversation ID, utterance ID, or consented speaker list)
into isolated ``0700``/``0600`` training directories outside Git:

1. ``stt/``      — 16 kHz mono WAVs (selected channel) + ``metadata.jsonl``
                   for Whisper Rioplatense fine-tuning.
2. ``wakeword/`` — ``positive/`` (clips containing verified wake mention) and
                   ``negative/`` (clean real conversation hard-negatives) for
                   ``scripts/train_custom_wake.py``.
3. ``llm/``      — ``conversations.jsonl`` grouped by ``conversation_id`` with
                   ordered turns and detected inline ``Nexa <action>`` clauses.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3

import numpy as np
import soundfile as sf

from src.ambient.inline_wake import extract_all_inline_commands
from src.audio.clip_io import (
    TARGET_SAMPLE_RATE,
    ArchiveChannelMode,
    resample_linear_16k,
    select_archive_audio,
    write_private_audio,
    write_private_bytes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurationPolicy:
    """Explicit opt-in filter controlling which utterances may be promoted."""

    approved_conversation_ids: frozenset[str] = frozenset()
    approved_utterance_ids: frozenset[int] = frozenset()
    approved_speakers: frozenset[str] = frozenset()
    min_vad_prob: float = 0.0
    spanish_only: bool = False
    exclude_sources: frozenset[str] = frozenset({"tv", "self"})
    channel_mode: ArchiveChannelMode = ArchiveChannelMode.MONO_CH0
    review_agreement_threshold: float = 0.90

    def requires_explicit_selection(self) -> bool:
        """Fail closed unless at least one explicit approval filter is supplied."""
        return not (
            self.approved_conversation_ids
            or self.approved_utterance_ids
            or self.approved_speakers
        )


@dataclass(frozen=True)
class CurationSummary:
    """Counts emitted by one curated export run (metadata only, no transcripts)."""

    stt_samples: int
    wake_positives: int
    wake_negatives: int
    llm_conversations: int
    skipped_missing_audio: int
    output_dir: str


def _load_approved_rows(db_path: Path, policy: CurationPolicy) -> list[dict]:
    if policy.requires_explicit_selection():
        raise ValueError("curation_requires_explicit_selection")

    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(utterances)").fetchall()}
        select_cols = [
            "id", "room_id", "t0", "t1", "text", "speaker", "source",
            "vad_prob", "lang", "lang_ok", "audio_path", "conversation_id",
        ]
        for opt in ("alt_text", "agreement_ratio"):
            if opt in cols:
                select_cols.append(opt)
        rows = [
            dict(r)
            for r in conn.execute(
                f"SELECT {', '.join(select_cols)} FROM utterances ORDER BY t0 ASC, id ASC"  # nosec B608 -- fixed whitelist of column names
            ).fetchall()
        ]
    finally:
        conn.close()

    selected: list[dict] = []
    for row in rows:
        if row.get("source") in policy.exclude_sources:
            continue
        vad = row.get("vad_prob")
        if vad is not None and float(vad) < policy.min_vad_prob:
            continue
        if policy.spanish_only and row.get("lang_ok") == 0:
            continue

        conv_ok = (
            bool(policy.approved_conversation_ids)
            and row.get("conversation_id") in policy.approved_conversation_ids
        )
        utt_ok = (
            bool(policy.approved_utterance_ids)
            and int(row["id"]) in policy.approved_utterance_ids
        )
        speaker_ok = (
            bool(policy.approved_speakers)
            and row.get("speaker") in policy.approved_speakers
        )
        if not (conv_ok or utt_ok or speaker_ok):
            continue
        selected.append(row)
    return selected


def _read_selected_mono(audio_path: Path, channel_mode: ArchiveChannelMode) -> tuple[np.ndarray, float]:
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    resampled = resample_linear_16k(data, sr)
    selected = select_archive_audio(resampled, channel_mode)
    if selected.ndim == 2:
        selected = np.ascontiguousarray(selected[:, 0])
    duration_s = round(selected.shape[0] / TARGET_SAMPLE_RATE, 3)
    return selected, duration_s


def export_curated_datasets(
    db_path: Path,
    out_dir: Path,
    policy: CurationPolicy,
    *,
    audio_base: Path | None = None,
    corrections: dict[int, str] | None = None,
    alt_metadata: dict[int, dict] | None = None,
) -> CurationSummary:
    """Export explicitly approved ambient conversations into STT, wakeword, and LLM datasets.

    Args:
        db_path: Read-only path to ``ambient.db``.
        out_dir: Private destination directory (created with mode ``0700``).
        policy: Explicit opt-in selection and channel policy.
        audio_base: Base directory for resolving relative ``audio_path`` entries.
        corrections: Optional ``{utterance_id: human_corrected_text}`` overrides.
        alt_metadata: Optional ``{utterance_id: {"alt_text": ..., "agreement_ratio": ...}}``
            sidecar metadata (used when ``ambient.db`` stays on SCHEMA_CONTRACT_EPOCH=2).

    Returns:
        ``CurationSummary`` with sample counts across the 3 datasets.
    """
    rows = _load_approved_rows(db_path, policy)
    base = audio_base if audio_base is not None else Path.cwd()
    corrections = corrections or {}
    alt_metadata = alt_metadata or {}

    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)

    stt_records: list[dict] = []
    wake_manifest: list[dict] = []
    conversations: dict[str, list[dict]] = defaultdict(list)

    stt_count = wake_pos = wake_neg = skipped_audio = 0

    for row in rows:
        utt_id = int(row["id"])
        text = (corrections.get(utt_id) if utt_id in corrections else row.get("text") or "").strip()
        inlines = extract_all_inline_commands(text) if text else []
        first_inline = inlines[0] if inlines else None
        has_wake = bool(inlines)
        extra_meta = alt_metadata.get(utt_id) or {}
        alt_text = row.get("alt_text") or extra_meta.get("alt_text")
        agreement = row.get("agreement_ratio")
        if agreement is None:
            agreement = extra_meta.get("agreement_ratio")
        needs_review = (
            utt_id not in corrections
            and agreement is not None
            and float(agreement) < policy.review_agreement_threshold
        )

        conv_key = row.get("conversation_id") or f"room_{row.get('room_id') or 'unknown'}"
        if text:
            turn_entry = {
                "utterance_id": utt_id,
                "t0": row.get("t0"),
                "t1": row.get("t1"),
                "room_id": row.get("room_id"),
                "speaker": row.get("speaker") or "unknown",
                "text": text,
                "vad_prob": row.get("vad_prob"),
                "inline_command": first_inline.command_text if first_inline is not None else None,
                "inline_commands": [item.command_text for item in inlines],
                "embedded_in_conversation": any(item.embedded_in_conversation for item in inlines),
            }
            if alt_text:
                turn_entry["alt_text_parakeet"] = alt_text
            if agreement is not None:
                turn_entry["agreement_ratio"] = round(float(agreement), 4)
                turn_entry["needs_review"] = needs_review
            conversations[conv_key].append(turn_entry)

        raw_audio = row.get("audio_path")
        if not raw_audio:
            skipped_audio += 1
            continue
        resolved = Path(raw_audio)
        if not resolved.is_absolute():
            resolved = base / resolved
        if not resolved.is_file():
            skipped_audio += 1
            continue

        try:
            mono, duration_s = _read_selected_mono(resolved, policy.channel_mode)
        except Exception as error:
            logger.warning("Skipping unreadable curated audio %s: %s", utt_id, type(error).__name__)
            skipped_audio += 1
            continue

        # 1. STT fine-tuning dataset (requires non-empty text)
        if text:
            stt_rel = Path("stt") / "audio" / f"utt_{utt_id}.wav"
            write_private_audio(
                out_dir / stt_rel, np.clip(mono, -1.0, 1.0), TARGET_SAMPLE_RATE, fmt="WAV", subtype="PCM_16"
            )
            stt_rec = {
                "utterance_id": utt_id,
                "audio_path": str(stt_rel),
                "text": text,
                "room_id": row.get("room_id"),
                "speaker": row.get("speaker") or "unknown",
                "duration_s": duration_s,
                "vad_prob": row.get("vad_prob"),
                "conversation_id": row.get("conversation_id"),
            }
            if alt_text:
                stt_rec["alt_text_parakeet"] = alt_text
            if agreement is not None:
                stt_rec["agreement_ratio"] = round(float(agreement), 4)
                stt_rec["needs_review"] = needs_review
            stt_records.append(stt_rec)
            stt_count += 1

        # 2. Wakeword dataset (positives with verified wake vs real-speech hard negatives)
        bucket = "positive" if has_wake else "negative"
        wake_rel = Path("wakeword") / bucket / f"utt_{utt_id}.wav"
        write_private_audio(
            out_dir / wake_rel, np.clip(mono, -1.0, 1.0), TARGET_SAMPLE_RATE, fmt="WAV", subtype="PCM_16"
        )
        wake_manifest.append(
            {
                "utterance_id": utt_id,
                "label": bucket,
                "audio_path": str(wake_rel),
                "room_id": row.get("room_id"),
                "duration_s": duration_s,
                "matched_variant": first_inline.matched_variant if first_inline is not None else None,
            }
        )
        if has_wake:
            wake_pos += 1
        else:
            wake_neg += 1

    # 3. Write manifests & LLM continuous-conversation dataset
    stt_jsonl = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in stt_records)
    if stt_jsonl:
        stt_jsonl += "\n"
    write_private_bytes(out_dir / "stt" / "metadata.jsonl", stt_jsonl.encode("utf-8"))

    write_private_bytes(
        out_dir / "wakeword" / "manifest.json",
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "channel_mode": str(policy.channel_mode),
                "positives": wake_pos,
                "negatives": wake_neg,
                "items": wake_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    )

    llm_lines = []
    for conv_id, turns in sorted(conversations.items()):
        inline_actions = [
            {"utterance_id": t["utterance_id"], "command": cmd}
            for t in turns
            for cmd in t.get("inline_commands", ())
        ]
        llm_lines.append(
            json.dumps(
                {
                    "conversation_id": conv_id,
                    "turn_count": len(turns),
                    "inline_actions": inline_actions,
                    "turns": turns,
                },
                ensure_ascii=False,
            )
        )
    llm_body = ("\n".join(llm_lines) + "\n") if llm_lines else ""
    write_private_bytes(out_dir / "llm" / "conversations.jsonl", llm_body.encode("utf-8"))

    return CurationSummary(
        stt_samples=stt_count,
        wake_positives=wake_pos,
        wake_negatives=wake_neg,
        llm_conversations=len(conversations),
        skipped_missing_audio=skipped_audio,
        output_dir=str(out_dir),
    )
