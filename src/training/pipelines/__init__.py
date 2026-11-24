"""Modular offline training pipelines for KZA on 2x RTX 3070 (8 GB each).

All three pipelines consume the canonical private directory produced by
``src.training.dataset_hub.TrainingDatasetHub``:
  - ``WhisperLoraPipeline`` -> Rioplatense Whisper LoRA + CTranslate2 export
  - ``WakewordPipeline``    -> Unified openWakeWord ONNX trainer (nexa / custom)
  - ``LlmQloraPipeline``    -> 2x RTX 3070 QLoRA trainer for Qwen2.5-7B/3B
"""
from __future__ import annotations

from src.training.pipelines.llm_qlora import LlmQloraConfig, LlmQloraPipeline
from src.training.pipelines.wakeword_pipeline import WakewordPipeline, WakewordPipelineConfig
from src.training.pipelines.whisper_lora import WhisperLoraConfig, WhisperLoraPipeline

__all__ = [
    "WhisperLoraConfig",
    "WhisperLoraPipeline",
    "WakewordPipelineConfig",
    "WakewordPipeline",
    "LlmQloraConfig",
    "LlmQloraPipeline",
]
