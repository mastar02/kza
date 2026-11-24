"""Unified openWakeWord ONNX training pipeline consuming TrainingDatasetHub.

Consolidates the fragmented wakeword training scripts (``train_custom_wake``,
``train_nexa``, ``train_che_bo``, ``train_wakeword``) into a single reusable
pipeline parameterized by ``wakeword_name`` and backed by curated hub clips
(``wakeword/positive/`` + ``wakeword/negative/``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from src.audio.clip_io import write_private_bytes


@dataclass(frozen=True)
class WakewordPipelineConfig:
    """Configuration for unified openWakeWord ONNX training."""

    curated_dir: Path
    wakeword_name: str = "nexa"
    output_dir: Path = Path("./data/training_outputs/wakeword")
    musan_dir: Path | None = None
    n_synthetic: int = 4000
    epochs: int = 30
    batch_size: int = 64
    min_positives: int = 30
    min_negatives: int = 30
    gpu_index: int = 0


@dataclass(frozen=True)
class WakewordPipelinePlan:
    """Validated clip inventory and artifact paths for wakeword training."""

    wakeword_name: str
    positive_clips: int
    negative_clips: int
    musan_available: bool
    n_synthetic: int
    output_onnx: str
    manifest_copy: str


class WakewordPipeline:
    """Validates curated wakeword buckets and runs unified openWakeWord training."""

    def __init__(self, config: WakewordPipelineConfig):
        self.config = config

    def _load_manifest(self) -> tuple[list[Path], list[Path]]:
        manifest_path = self.config.curated_dir / "wakeword" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing_wakeword_manifest:{manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        positives: list[Path] = []
        negatives: list[Path] = []
        for item in payload.get("items", []):
            rel = item.get("audio_path")
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = self.config.curated_dir / path
            if not path.is_file():
                continue
            if item.get("label") == "positive":
                positives.append(path.resolve())
            elif item.get("label") == "negative":
                negatives.append(path.resolve())
        return positives, negatives

    def prepare(self) -> WakewordPipelinePlan:
        """Validate curated positive/negative counts and write the run manifest."""
        positives, negatives = self._load_manifest()
        if len(positives) < self.config.min_positives:
            raise ValueError(
                f"insufficient_wake_positives:{len(positives)}<{self.config.min_positives}"
            )
        if len(negatives) < self.config.min_negatives:
            raise ValueError(
                f"insufficient_wake_negatives:{len(negatives)}<{self.config.min_negatives}"
            )

        out = self.config.output_dir / self.config.wakeword_name
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(out, 0o700)

        musan_ok = bool(self.config.musan_dir and self.config.musan_dir.is_dir())
        onnx_path = out / f"{self.config.wakeword_name}.onnx"
        manifest_copy = out / "resolved_clips.json"
        write_private_bytes(
            manifest_copy,
            json.dumps(
                {
                    "wakeword_name": self.config.wakeword_name,
                    "positives": [str(p) for p in positives],
                    "negatives": [str(p) for p in negatives],
                    "musan_dir": str(self.config.musan_dir) if musan_ok else None,
                    "n_synthetic": self.config.n_synthetic,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "output_onnx": str(onnx_path),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        plan = WakewordPipelinePlan(
            wakeword_name=self.config.wakeword_name,
            positive_clips=len(positives),
            negative_clips=len(negatives),
            musan_available=musan_ok,
            n_synthetic=self.config.n_synthetic,
            output_onnx=str(onnx_path),
            manifest_copy=str(manifest_copy),
        )
        write_private_bytes(
            out / "plan.json",
            json.dumps(asdict(plan), ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return plan

    def run(self, *, dry_run: bool = False) -> WakewordPipelinePlan:
        """Prepare the dataset plan and optionally invoke ``WakeWordTrainer``."""
        plan = self.prepare()
        if dry_run:
            return plan
        from src.wakeword.trainer import WakeWordTrainer

        trainer = WakeWordTrainer(
            data_dir=str(self.config.curated_dir),
            models_dir=str(self.config.output_dir / self.config.wakeword_name),
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.config.gpu_index))
        os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
        result = trainer.train(
            wake_word_name=self.config.wakeword_name,
            epochs=self.config.epochs,
        )
        if not result:
            raise RuntimeError("wakeword_training_failed")
        return plan
