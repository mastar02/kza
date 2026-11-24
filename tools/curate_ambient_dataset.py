#!/usr/bin/env python3
"""CLI for explicit, privacy-safe curation of ambient conversations into 3 datasets.

Exports approved utterances/conversations from ``ambient.db`` (within its <=48h
TTL) into ``0700``/``0600`` training folders for:
  - ``stt/``      (Whisper Rioplatense fine-tuning)
  - ``wakeword/`` (positive/negative clips for ``nexa.onnx``)
  - ``llm/``      (continuous conversations + inline Nexa actions for LoRA/habits)

Fails closed unless at least one explicit approval filter is passed
(``--conversation-id``, ``--utterance-id``, or ``--speaker``).

Usage:
    PYTHONPATH=. python3 tools/curate_ambient_dataset.py \\
        --db data/ambient.db --out data/curated_export \\
        --speaker gabriel --channel-mode mono_ch1 --spanish-only
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.clip_io import ArchiveChannelMode  # noqa: E402
from src.training.curated_dataset_exporter import CurationPolicy  # noqa: E402
from src.training.dataset_hub import (  # noqa: E402
    HubConfig,
    TrainingDatasetHub,
    load_corrections_file,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None, help="Path to ambient.db")
    ap.add_argument("--out", type=Path, required=True, help="Private output directory (0700)")
    ap.add_argument("--audio-base", type=Path, default=None, help="Base dir for relative audio_path")
    ap.add_argument("--conversation-id", action="append", default=[], help="Approved conversation_id (repeatable)")
    ap.add_argument("--utterance-id", action="append", type=int, default=[], help="Approved utterance id (repeatable)")
    ap.add_argument("--speaker", action="append", default=[], help="Consented speaker id (repeatable)")
    ap.add_argument("--min-vad", type=float, default=0.0, help="Minimum vad_prob filter")
    ap.add_argument("--spanish-only", action="store_true", help="Exclude rows flagged lang_ok=0")
    ap.add_argument(
        "--channel-mode",
        choices=[m.value for m in ArchiveChannelMode],
        default=ArchiveChannelMode.MONO_CH0.value,
        help="Channel to extract from 2ch archived FLACs (default: mono_ch0)",
    )
    ap.add_argument(
        "--review-threshold",
        type=float,
        default=0.90,
        help="Flag rows with agreement_ratio < threshold as needs_review",
    )
    ap.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Optional corrections.json exported by tools/training_review.py",
    )
    ap.add_argument(
        "--wake-clips-dir",
        type=Path,
        default=None,
        help="Optional explicitly approved WakeClipWriter directory (accepted/missed/rejected)",
    )
    ap.add_argument(
        "--conversation-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional explicitly approved conversation JSONL file (repeatable)",
    )
    args = ap.parse_args(argv)

    channel_mode = ArchiveChannelMode(args.channel_mode)
    policy = (
        CurationPolicy(
            approved_conversation_ids=frozenset(args.conversation_id),
            approved_utterance_ids=frozenset(args.utterance_id),
            approved_speakers=frozenset(args.speaker),
            min_vad_prob=args.min_vad,
            spanish_only=args.spanish_only,
            channel_mode=channel_mode,
            review_agreement_threshold=args.review_threshold,
        )
        if args.db is not None
        else None
    )
    corrections = load_corrections_file(args.corrections) if args.corrections is not None else {}

    hub = TrainingDatasetHub()
    try:
        summary = hub.build(
            HubConfig(
                out_dir=args.out,
                ambient_db=args.db,
                ambient_policy=policy,
                audio_base=args.audio_base,
                corrections=corrections,
                wake_clips_dir=args.wake_clips_dir,
                conversation_jsonl_paths=tuple(args.conversation_jsonl),
                channel_mode=channel_mode,
            )
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
