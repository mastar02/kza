"""Unit tests for the 3 modular 2x RTX 3070 training pipelines and unified CLI."""
from __future__ import annotations

import ast
import json
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.train_pipeline import main as train_pipeline_main
from src.training.pipelines import (
    LlmQloraConfig,
    LlmQloraPipeline,
    WakewordPipeline,
    WakewordPipelineConfig,
    WhisperLoraConfig,
    WhisperLoraPipeline,
)


def _seed_hub(tmp_path: Path) -> Path:
    hub = tmp_path / "curated_hub"
    (hub / "stt" / "audio").mkdir(parents=True)
    (hub / "wakeword" / "positive").mkdir(parents=True)
    (hub / "wakeword" / "negative").mkdir(parents=True)
    (hub / "llm").mkdir(parents=True)

    stt_lines = []
    for i in range(1, 6):
        wav = hub / "stt" / "audio" / f"utt_{i}.wav"
        sf.write(str(wav), np.zeros(3200, dtype=np.float32), 16000, format="WAV")
        stt_lines.append(
            json.dumps(
                {
                    "utterance_id": i,
                    "audio_path": f"stt/audio/utt_{i}.wav",
                    "text": f"Nexa, prendé la luz {i}",
                    "duration_s": 0.2,
                    "needs_review": (i == 5),
                },
                ensure_ascii=False,
            )
        )
    (hub / "stt" / "metadata.jsonl").write_text("\n".join(stt_lines) + "\n", encoding="utf-8")

    wake_items = []
    for i in range(1, 4):
        p = hub / "wakeword" / "positive" / f"p_{i}.wav"
        n = hub / "wakeword" / "negative" / f"n_{i}.wav"
        sf.write(str(p), np.zeros(3200, dtype=np.float32), 16000, format="WAV")
        sf.write(str(n), np.zeros(3200, dtype=np.float32), 16000, format="WAV")
        wake_items.append({"utterance_id": f"p{i}", "label": "positive", "audio_path": f"wakeword/positive/p_{i}.wav"})
        wake_items.append({"utterance_id": f"n{i}", "label": "negative", "audio_path": f"wakeword/negative/n_{i}.wav"})
    (hub / "wakeword" / "manifest.json").write_text(
        json.dumps({"positives": 3, "negatives": 3, "items": wake_items}),
        encoding="utf-8",
    )

    conv_lines = [
        json.dumps(
            {
                "conversation_id": "conv-1",
                "turn_count": 2,
                "inline_actions": [
                    {"utterance_id": 1, "command": "Nexa, prendé la luz del escritorio"},
                    {"utterance_id": 1, "command": "Nexa, ¿qué temperatura hace?"},
                ],
                "turns": [
                    {
                        "utterance_id": 1,
                        "room_id": "escritorio",
                        "speaker": "gabriel",
                        "text": "Hola. Nexa, prendé la luz del escritorio. Nexa, ¿qué temperatura hace?",
                        "inline_commands": [
                            "Nexa, prendé la luz del escritorio",
                            "Nexa, ¿qué temperatura hace?",
                        ],
                    },
                    {
                        "utterance_id": 2,
                        "room_id": "escritorio",
                        "speaker": "gabriel",
                        "text": "Prendé el aire",
                        "assistant_response": "Listo, prendí el aire del escritorio.",
                    },
                ],
            },
            ensure_ascii=False,
        )
    ]
    (hub / "llm" / "conversations.jsonl").write_text("\n".join(conv_lines) + "\n", encoding="utf-8")
    (hub / "hub_manifest.json").write_text(json.dumps({"stt_samples": 5}), encoding="utf-8")
    return hub


def test_whisper_lora_pipeline_filters_needs_review_and_generates_valid_script(tmp_path):
    hub = _seed_hub(tmp_path)
    out = tmp_path / "out_whisper"

    plan = WhisperLoraPipeline(
        WhisperLoraConfig(
            curated_dir=hub,
            output_dir=out,
            base_model="openai/whisper-large-v3-turbo",
            min_samples=3,
        )
    ).run(dry_run=True)

    assert plan.total_samples == 4  # skips utt_5 (needs_review=True)
    assert plan.skipped_needs_review == 1
    assert plan.train_samples + plan.val_samples == 4
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    script = Path(plan.script_path).read_text(encoding="utf-8")
    ast.parse(script)
    assert "ct2-transformers-converter" in script
    assert "openai/whisper-large-v3-turbo" in script


def test_wakeword_pipeline_validates_counts_and_prepares_manifest(tmp_path):
    hub = _seed_hub(tmp_path)
    with pytest.raises(ValueError, match="insufficient_wake_positives"):
        WakewordPipeline(
            WakewordPipelineConfig(curated_dir=hub, output_dir=tmp_path / "wk_fail", min_positives=10)
        ).prepare()

    plan = WakewordPipeline(
        WakewordPipelineConfig(
            curated_dir=hub,
            output_dir=tmp_path / "wk_ok",
            wakeword_name="nexa",
            min_positives=3,
            min_negatives=3,
        )
    ).run(dry_run=True)

    assert plan.positive_clips == 3
    assert plan.negative_clips == 3
    assert plan.output_onnx.endswith("nexa.onnx")
    assert Path(plan.manifest_copy).is_file()


def test_llm_qlora_pipeline_builds_inline_and_dialogue_examples_for_2x_3070(tmp_path):
    hub = _seed_hub(tmp_path)
    out = tmp_path / "out_llm"

    plan = LlmQloraPipeline(
        LlmQloraConfig(
            curated_dir=hub,
            output_dir=out,
            min_examples=2,
        )
    ).run(dry_run=True)

    assert plan.gpus == (0, 1)
    assert plan.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert plan.training_examples == 2
    assert plan.inline_action_examples == 2
    script = Path(plan.script_path).read_text(encoding="utf-8")
    ast.parse(script)
    assert "BitsAndBytesConfig" in script
    assert "Qwen/Qwen2.5-7B-Instruct" in script


def test_unified_train_pipeline_cli_dry_runs_all_subcommands(tmp_path, capsys):
    hub = _seed_hub(tmp_path)

    assert train_pipeline_main(["status", "--hub", str(hub)]) == 0
    assert json.loads(capsys.readouterr().out)["stt_samples"] == 5

    assert train_pipeline_main([
        "whisper", "--hub", str(hub), "--out", str(tmp_path / "w"), "--target", "large-v3",
        "--min-samples", "3", "--dry-run",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["base_model"] == "openai/whisper-large-v3"

    assert train_pipeline_main([
        "wakeword", "--hub", str(hub), "--out", str(tmp_path / "wk"), "--name", "nexa",
        "--min-positives", "3", "--min-negatives", "3", "--dry-run",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["wakeword_name"] == "nexa"

    assert train_pipeline_main([
        "llm", "--hub", str(hub), "--out", str(tmp_path / "l"), "--gpus", "0,1",
        "--min-examples", "2", "--dry-run",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["gpus"] == [0, 1]
