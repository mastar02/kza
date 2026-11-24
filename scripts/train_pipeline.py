#!/usr/bin/env python3
"""Unified CLI for KZA's 3 curated training pipelines (2x RTX 3070).

Drives offline training strictly from an operator-curated ``TrainingDatasetHub``
directory (produced by ``tools/curate_ambient_dataset.py`` + reviewed via
``tools/training_review.py``):

  python3 scripts/train_pipeline.py status   --hub data/curated_hub
  python3 scripts/train_pipeline.py whisper  --hub data/curated_hub --target turbo --dry-run
  python3 scripts/train_pipeline.py wakeword --hub data/curated_hub --name nexa --dry-run
  python3 scripts/train_pipeline.py llm      --hub data/curated_hub --gpus 0,1 --dry-run
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src as _g1_src  # noqa: E402,F401 -- G1 (U3b-0e): refuses unmanaged run under a managed root (exit 78)

from src.training.pipelines import (  # noqa: E402
    LlmQloraConfig,
    LlmQloraPipeline,
    WakewordPipeline,
    WakewordPipelineConfig,
    WhisperLoraConfig,
    WhisperLoraPipeline,
)

_WHISPER_TARGETS = {
    "turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Inspect hub_manifest.json readiness")
    p_status.add_argument("--hub", type=Path, required=True)

    p_wh = sub.add_parser("whisper", help="Fine-tune Whisper Rioplatense LoRA + CTranslate2")
    p_wh.add_argument("--hub", type=Path, required=True)
    p_wh.add_argument("--out", type=Path, default=Path("./data/training_outputs/whisper"))
    p_wh.add_argument("--target", choices=sorted(_WHISPER_TARGETS), default="turbo")
    p_wh.add_argument("--gpu", type=int, default=0)
    p_wh.add_argument("--epochs", type=int, default=3)
    p_wh.add_argument("--min-samples", type=int, default=20)
    p_wh.add_argument("--include-unreviewed", action="store_true")
    p_wh.add_argument("--dry-run", action="store_true")

    p_wk = sub.add_parser("wakeword", help="Train custom openWakeWord ONNX from curated hub")
    p_wk.add_argument("--hub", type=Path, required=True)
    p_wk.add_argument("--out", type=Path, default=Path("./data/training_outputs/wakeword"))
    p_wk.add_argument("--name", default="nexa")
    p_wk.add_argument("--musan", type=Path, default=None)
    p_wk.add_argument("--gpu", type=int, default=0)
    p_wk.add_argument("--epochs", type=int, default=30)
    p_wk.add_argument("--min-positives", type=int, default=30)
    p_wk.add_argument("--min-negatives", type=int, default=30)
    p_wk.add_argument("--dry-run", action="store_true")

    p_llm = sub.add_parser("llm", help="Fine-tune Qwen2.5 QLoRA on 2x RTX 3070")
    p_llm.add_argument("--hub", type=Path, required=True)
    p_llm.add_argument("--out", type=Path, default=Path("./data/training_outputs/lora"))
    p_llm.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p_llm.add_argument("--gpus", default="0,1")
    p_llm.add_argument("--epochs", type=int, default=3)
    p_llm.add_argument("--min-examples", type=int, default=10)
    p_llm.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)

    try:
        if args.command == "status":
            manifest = args.hub / "hub_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"missing_hub_manifest:{manifest}")
            print(manifest.read_text(encoding="utf-8"))
            return 0

        if args.command == "whisper":
            plan = WhisperLoraPipeline(
                WhisperLoraConfig(
                    curated_dir=args.hub,
                    output_dir=args.out,
                    base_model=_WHISPER_TARGETS[args.target],
                    gpu_index=args.gpu,
                    epochs=args.epochs,
                    min_samples=args.min_samples,
                    include_unreviewed=args.include_unreviewed,
                )
            ).run(dry_run=args.dry_run)
            print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
            return 0

        if args.command == "wakeword":
            plan = WakewordPipeline(
                WakewordPipelineConfig(
                    curated_dir=args.hub,
                    wakeword_name=args.name,
                    output_dir=args.out,
                    musan_dir=args.musan,
                    gpu_index=args.gpu,
                    epochs=args.epochs,
                    min_positives=args.min_positives,
                    min_negatives=args.min_negatives,
                )
            ).run(dry_run=args.dry_run)
            print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
            return 0

        if args.command == "llm":
            gpus = tuple(int(g.strip()) for g in args.gpus.split(",") if g.strip())
            plan = LlmQloraPipeline(
                LlmQloraConfig(
                    curated_dir=args.hub,
                    output_dir=args.out,
                    base_model=args.base_model,
                    gpus=gpus,
                    epochs=args.epochs,
                    min_examples=args.min_examples,
                )
            ).run(dry_run=args.dry_run)
            print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
