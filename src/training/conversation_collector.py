"""
Conversation Collector Module
Recolecta conversaciones para fine-tuning del modelo.

El flujo es:
1. Recolectar conversaciones durante el uso normal
2. El usuario puede marcar respuestas como buenas/malas
3. Exportar en formato para entrenamiento LoRA
4. Entrenar con los datos recolectados
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from enum import StrEnum

logger = logging.getLogger(__name__)


class ResponseQuality(StrEnum):
    """Calidad de la respuesta marcada por el usuario"""
    UNMARKED = "unmarked"
    GOOD = "good"      # Respuesta correcta y útil
    BAD = "bad"        # Respuesta incorrecta o inútil
    CORRECTED = "corrected"  # Usuario proporcionó corrección


@dataclass
class ConversationTurn:
    """Un turno de conversación (user + assistant)"""
    timestamp: float
    user_input: str
    assistant_response: str
    intent: str | None = None
    entities_used: list[str] = field(default_factory=list)
    quality: ResponseQuality = ResponseQuality.UNMARKED
    correction: str | None = None  # Si el usuario corrigió
    user_name: str | None = None
    latency_ms: float | None = None


@dataclass
class Conversation:
    """Una conversación completa (múltiples turnos)"""
    id: str
    started_at: float
    turns: list[ConversationTurn] = field(default_factory=list)
    user_name: str | None = None
    metadata: dict = field(default_factory=dict)


class ConversationCollector:
    """
    Recolecta y gestiona conversaciones para entrenamiento.

    Features:
    - Guardar conversaciones automáticamente
    - Marcar respuestas como buenas/malas
    - Proporcionar correcciones
    - Exportar en formato LoRA/Alpaca
    - Estadísticas de calidad
    """

    def __init__(
        self,
        data_dir: str = "./data/conversations",
        auto_save_interval: int = 10,  # Guardar cada N turnos
        max_conversations_in_memory: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save_interval = auto_save_interval
        self.max_conversations_in_memory = max_conversations_in_memory

        self._current_conversation: Conversation | None = None
        self._conversations: list[Conversation] = []
        self._turns_since_save = 0

        self._load_recent()

    def _load_recent(self):
        """Cargar conversaciones recientes"""
        conv_files = sorted(self.data_dir.glob("conv_*.json"), reverse=True)

        for conv_file in conv_files[:self.max_conversations_in_memory]:
            try:
                with open(conv_file) as f:
                    data = json.load(f)

                conv = Conversation(
                    id=data["id"],
                    started_at=data["started_at"],
                    user_name=data.get("user_name"),
                    metadata=data.get("metadata", {})
                )

                for turn_data in data.get("turns", []):
                    turn = ConversationTurn(
                        timestamp=turn_data["timestamp"],
                        user_input=turn_data["user_input"],
                        assistant_response=turn_data["assistant_response"],
                        intent=turn_data.get("intent"),
                        entities_used=turn_data.get("entities_used", []),
                        quality=ResponseQuality(turn_data.get("quality", "unmarked")),
                        correction=turn_data.get("correction"),
                        user_name=turn_data.get("user_name"),
                        latency_ms=turn_data.get("latency_ms")
                    )
                    conv.turns.append(turn)

                self._conversations.append(conv)

            except Exception as e:
                logger.error(f"Error cargando {conv_file}: {e}")

        logger.info(f"Cargadas {len(self._conversations)} conversaciones")

    def start_conversation(self, user_name: str = None) -> str:
        """Iniciar nueva conversación"""
        conv_id = f"conv_{int(time.time() * 1000)}"

        self._current_conversation = Conversation(
            id=conv_id,
            started_at=time.time(),
            user_name=user_name
        )

        logger.debug(f"Conversación iniciada: {conv_id}")
        return conv_id

    def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        intent: str = None,
        entities_used: list[str] = None,
        user_name: str = None,
        latency_ms: float = None
    ):
        """Agregar turno a la conversación actual"""
        if self._current_conversation is None:
            self.start_conversation(user_name)

        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_used=entities_used or [],
            user_name=user_name,
            latency_ms=latency_ms
        )

        self._current_conversation.turns.append(turn)
        self._turns_since_save += 1

        # Auto-save periódico
        if self._turns_since_save >= self.auto_save_interval:
            self._save_current()
            self._turns_since_save = 0

    def mark_last_response(self, quality: str, correction: str = None):
        """
        Marcar la última respuesta con calidad.

        Args:
            quality: "good", "bad", o "corrected"
            correction: Respuesta correcta si quality es "corrected"
        """
        if not self._current_conversation or not self._current_conversation.turns:
            return False

        last_turn = self._current_conversation.turns[-1]

        if quality == "good":
            last_turn.quality = ResponseQuality.GOOD
        elif quality == "bad":
            last_turn.quality = ResponseQuality.BAD
        elif quality == "corrected" and correction:
            last_turn.quality = ResponseQuality.CORRECTED
            last_turn.correction = correction
        else:
            return False

        self._save_current()
        logger.info(f"Respuesta marcada como: {quality}")
        return True

    def end_conversation(self):
        """Finalizar conversación actual"""
        if self._current_conversation and self._current_conversation.turns:
            self._save_current()
            self._conversations.append(self._current_conversation)

            # Limpiar memoria si hay muchas
            if len(self._conversations) > self.max_conversations_in_memory:
                self._conversations = self._conversations[-self.max_conversations_in_memory:]

        self._current_conversation = None

    def _save_current(self):
        """Guardar conversación actual a disco"""
        if not self._current_conversation:
            return

        conv = self._current_conversation
        filepath = self.data_dir / f"{conv.id}.json"

        data = {
            "id": conv.id,
            "started_at": conv.started_at,
            "user_name": conv.user_name,
            "metadata": conv.metadata,
            "turns": [
                {
                    "timestamp": t.timestamp,
                    "user_input": t.user_input,
                    "assistant_response": t.assistant_response,
                    "intent": t.intent,
                    "entities_used": t.entities_used,
                    "quality": t.quality.value,
                    "correction": t.correction,
                    "user_name": t.user_name,
                    "latency_ms": t.latency_ms
                }
                for t in conv.turns
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_for_training(
        self,
        output_path: str = "./data/training_data.jsonl",
        format: str = "alpaca",
        only_marked: bool = True,
        include_corrections: bool = True
    ) -> dict:
        """
        Exportar conversaciones en formato para entrenamiento.

        Args:
            output_path: Ruta de salida
            format: "alpaca", "sharegpt", o "conversations"
            only_marked: Solo incluir turnos marcados como buenos
            include_corrections: Incluir correcciones como ejemplos

        Returns:
            Estadísticas de exportación
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        exported = 0
        good_count = 0
        corrected_count = 0

        with open(output_file, "w") as f:
            for conv in self._conversations:
                for turn in conv.turns:
                    # Filtrar según calidad
                    if only_marked:
                        if turn.quality == ResponseQuality.UNMARKED:
                            continue
                        if turn.quality == ResponseQuality.BAD:
                            continue

                    # Determinar la respuesta a usar
                    response = turn.assistant_response
                    if turn.quality == ResponseQuality.CORRECTED and turn.correction:
                        if include_corrections:
                            response = turn.correction
                            corrected_count += 1
                        else:
                            continue
                    elif turn.quality == ResponseQuality.GOOD:
                        good_count += 1

                    # Formatear según el formato solicitado
                    if format == "alpaca":
                        entry = {
                            "instruction": turn.user_input,
                            "input": "",
                            "output": response
                        }
                    elif format == "sharegpt":
                        entry = {
                            "conversations": [
                                {"from": "human", "value": turn.user_input},
                                {"from": "gpt", "value": response}
                            ]
                        }
                    else:  # conversations
                        entry = {
                            "user": turn.user_input,
                            "assistant": response,
                            "intent": turn.intent,
                            "timestamp": turn.timestamp
                        }

                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    exported += 1

        stats = {
            "exported": exported,
            "good_responses": good_count,
            "corrected_responses": corrected_count,
            "format": format,
            "output_path": str(output_file)
        }

        logger.info(f"Exportadas {exported} entradas para entrenamiento")
        return stats

    def get_stats(self) -> dict:
        """Obtener estadísticas de las conversaciones"""
        total_turns = sum(len(c.turns) for c in self._conversations)

        quality_counts = {
            "unmarked": 0,
            "good": 0,
            "bad": 0,
            "corrected": 0
        }

        for conv in self._conversations:
            for turn in conv.turns:
                quality_counts[turn.quality.value] += 1

        return {
            "total_conversations": len(self._conversations),
            "total_turns": total_turns,
            "current_conversation_turns": len(self._current_conversation.turns) if self._current_conversation else 0,
            "quality_distribution": quality_counts,
            "marked_for_training": quality_counts["good"] + quality_counts["corrected"],
            "data_dir": str(self.data_dir)
        }

    def get_training_commands(self) -> str:
        """
        Retorna comandos de voz para el sistema de feedback.
        El usuario puede decir estas frases para marcar respuestas.
        """
        return """
Comandos de feedback:
- "Eso estuvo bien" / "Buena respuesta" → Marca como buena
- "Eso estuvo mal" / "Mala respuesta" → Marca como mala
- "Debiste decir [corrección]" → Guarda corrección

Estos datos se usan para mejorar la IA con el tiempo.
"""


class LoRATrainer:
    """
    Entrena el modelo con LoRA usando los datos recolectados.

    LoRA (Low-Rank Adaptation) permite fine-tuning eficiente
    usando solo una fracción de los parámetros del modelo.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir: str = "./models/lora_adapters",
        device: str = "cuda:0"
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        training_data_path: str,
        adapter_name: str = "custom_adapter",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ) -> dict:
        """
        Entrenar adapter LoRA.

        Args:
            training_data_path: Ruta a los datos de entrenamiento (.jsonl)
            adapter_name: Nombre del adapter resultante
            epochs: Número de épocas
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
            lora_r: Rango de LoRA (menor = más ligero)
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA

        Returns:
            Información del entrenamiento
        """
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import load_dataset
            import torch

        except ImportError as e:
            logger.error(f"Dependencias no instaladas: {e}")
            return {
                "success": False,
                "error": "Instala: pip install transformers peft datasets accelerate"
            }

        logger.info(f"Iniciando entrenamiento LoRA para {self.base_model}")
        logger.info(f"Datos: {training_data_path}")

        # Cargar dataset
        dataset = load_dataset("json", data_files=training_data_path)

        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Función de tokenización
        def tokenize_function(examples):
            # Formato Alpaca
            prompts = []
            for instruction, output in zip(examples["instruction"], examples["output"]):
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                prompts.append(prompt)

            return tokenizer(
                prompts,
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Cargar modelo base
        logger.info("Cargando modelo base...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Configurar LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Configurar entrenamiento
        output_path = self.output_dir / adapter_name
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
        )

        # Entrenar
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer
        )

        logger.info("Iniciando entrenamiento...")
        trainer.train()

        # Guardar adapter
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Adapter guardado en: {output_path}")

        return {
            "success": True,
            "adapter_path": str(output_path),
            "epochs": epochs,
            "trainable_params": model.num_parameters(only_trainable=True)
        }

    def list_adapters(self) -> list[dict]:
        """Listar adapters entrenados"""
        adapters = []

        for adapter_dir in self.output_dir.iterdir():
            if adapter_dir.is_dir():
                config_file = adapter_dir / "adapter_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)

                    adapters.append({
                        "name": adapter_dir.name,
                        "path": str(adapter_dir),
                        "lora_r": config.get("r"),
                        "lora_alpha": config.get("lora_alpha"),
                        "base_model": config.get("base_model_name_or_path")
                    })

        return adapters
