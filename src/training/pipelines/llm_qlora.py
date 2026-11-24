"""2x RTX 3070 (8 GB) QLoRA training pipeline for KZA continuous conversations & inline actions.

Consumes ``<curated_dir>/llm/conversations.jsonl`` produced by ``TrainingDatasetHub``
and converts continuous multi-turn household chats + inline ``Nexa <action>``
annotations into clean instruction-tuning JSONL targeting ``Qwen/Qwen2.5-7B-Instruct``
(or ``Qwen/Qwen2.5-3B-Instruct``) across the real 2x RTX 3070 hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from src.audio.clip_io import write_private_bytes
from src.core.paths import TRAINING_OUTPUT_LORA


@dataclass(frozen=True)
class LlmQloraConfig:
    """Configuration calibrated for 2x RTX 3070 (8 GB VRAM each)."""

    curated_dir: Path
    output_dir: Path = Path(TRAINING_OUTPUT_LORA)
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    gpus: tuple[int, ...] = (0, 1)
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    min_examples: int = 10


@dataclass(frozen=True)
class LlmQloraPlan:
    """Prepared LLM dataset and runnable QLoRA script metadata."""

    base_model: str
    gpus: tuple[int, ...]
    conversations_read: int
    training_examples: int
    inline_action_examples: int
    dataset_path: str
    script_path: str
    adapter_output_dir: str


def build_llm_examples_from_conversations(conversations: list[dict]) -> tuple[list[dict], int]:
    """Convert curated ``llm/conversations.jsonl`` entries into instruction examples.

    Produces two complementary example types:
    1. Direct turn pairs when ``assistant_response`` is present (curated dialogues).
    2. Inline command extraction / contextual grounding examples when continuous
       conversation turns contain ``inline_commands`` (teaching the local router
       to isolate ``Nexa <action>`` from surrounding Rioplatense chat).
    """
    examples: list[dict] = []
    inline_count = 0
    for conv in conversations:
        turns = conv.get("turns") or []
        prior_context: list[str] = []
        for turn in turns:
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = str(turn.get("speaker") or "unknown")
            room = str(turn.get("room_id") or "")
            context_prefix = "\n".join(prior_context[-4:])

            assistant_resp = str(turn.get("assistant_response") or "").strip()
            if assistant_resp:
                examples.append(
                    {
                        "instruction": text,
                        "input": str(turn.get("context") or context_prefix),
                        "output": assistant_resp,
                        "category": "dialogue",
                    }
                )

            inline_cmds = list(turn.get("inline_commands") or [])
            if not inline_cmds and turn.get("inline_command"):
                inline_cmds = [str(turn["inline_command"])]
            if inline_cmds:
                inline_count += len(inline_cmds)
                ctx_parts = [p for p in (f"room={room}" if room else "", f"speaker={speaker}", context_prefix) if p]
                examples.append(
                    {
                        "instruction": (
                            "Extraé los comandos dirigidos a Nexa dentro de la siguiente "
                            "transcripción continua en español rioplatense."
                        ),
                        "input": f"{' | '.join(ctx_parts)}\nTranscripción: {text}",
                        "output": json.dumps({"commands": inline_cmds}, ensure_ascii=False),
                        "category": "inline_command_extraction",
                    }
                )

            prior_context.append(f"{speaker}: {text}")
    return examples, inline_count


class LlmQloraPipeline:
    """Prepares and runs 2x RTX 3070 QLoRA fine-tuning from curated hub conversations."""

    def __init__(self, config: LlmQloraConfig):
        self.config = config

    def prepare(self) -> LlmQloraPlan:
        """Build the private training JSONL and QLoRA runner script."""
        conv_path = self.config.curated_dir / "llm" / "conversations.jsonl"
        if not conv_path.is_file():
            raise FileNotFoundError(f"missing_curated_llm_conversations:{conv_path}")

        conversations = [
            json.loads(line)
            for line in conv_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        examples, inline_count = build_llm_examples_from_conversations(conversations)
        if len(examples) < self.config.min_examples:
            raise ValueError(
                f"insufficient_llm_examples:{len(examples)}<{self.config.min_examples}"
            )

        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(out, 0o700)

        dataset_path = out / "llm_train.jsonl"
        write_private_bytes(
            dataset_path,
            ("\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples) + "\n").encode("utf-8"),
        )
        adapter_dir = out / "qwen_qlora_adapter"
        script_path = out / "run_llm_qlora.py"
        write_private_bytes(
            script_path,
            self._render_script(dataset_path, adapter_dir).encode("utf-8"),
        )

        plan = LlmQloraPlan(
            base_model=self.config.base_model,
            gpus=tuple(self.config.gpus),
            conversations_read=len(conversations),
            training_examples=len(examples),
            inline_action_examples=inline_count,
            dataset_path=str(dataset_path),
            script_path=str(script_path),
            adapter_output_dir=str(adapter_dir),
        )
        write_private_bytes(
            out / "plan.json",
            json.dumps(asdict(plan), ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return plan

    def _render_script(self, dataset_path: Path, adapter_dir: Path) -> str:
        c = self.config
        return f'''#!/usr/bin/env python3
"""Auto-generated QLoRA trainer calibrated for 2x RTX 3070 (8 GB)."""
import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = {c.base_model!r}
DATASET_PATH = Path({str(dataset_path)!r})
ADAPTER_DIR = Path({str(adapter_dir)!r})

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r={c.lora_r},
        lora_alpha={c.lora_alpha},
        lora_dropout={c.lora_dropout},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    rows = [json.loads(l) for l in DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    def fmt(r):
        user = f"{{r['instruction']}}\\n{{r['input']}}".strip()
        text = (
            f"<|im_start|>user\\n{{user}}<|im_end|>\\n"
            f"<|im_start|>assistant\\n{{r['output']}}<|im_end|>"
        )
        return tokenizer(text, truncation=True, max_length={c.max_seq_length})

    ds = Dataset.from_list(rows).map(fmt, remove_columns=list(rows[0].keys()))
    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs={c.epochs},
        per_device_train_batch_size={c.batch_size},
        gradient_accumulation_steps={c.gradient_accumulation_steps},
        learning_rate={c.learning_rate},
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

if __name__ == "__main__":
    main()
'''

    def run(self, *, dry_run: bool = False) -> LlmQloraPlan:
        """Prepare the plan and optionally run QLoRA across configured GPUs."""
        plan = self.prepare()
        if dry_run:
            return plan
        gpu_csv = ",".join(str(g) for g in self.config.gpus)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_csv)
        subprocess.run([sys.executable, plan.script_path], env=env, check=True)
        return plan
