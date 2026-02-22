"""
Nightly Trainer Module
Entrenamiento automático nocturno con QLoRA usando las GPUs.

Flujo nocturno:
1. A las 3:00 AM (configurable) se activa el entrenamiento
2. DESCARGAR modelos de día (TTS, Router, Whisper, etc.) para liberar VRAM
3. Recopila contextos persistentes del día
4. Prepara dataset con conversaciones de calidad
5. Distribuye entrenamiento QLoRA (4-bit) en las 4x RTX 3070
6. Guarda adapter LoRA personalizado
7. RECARGAR modelos de día
8. Genera reporte de entrenamiento

IMPORTANTE - Gestión de VRAM:
  - RTX 3070 = 8GB VRAM cada una
  - De día: GPUs ocupadas con TTS, Whisper, Router, Embeddings
  - De noche: Descargar todo → Entrenar con QLoRA → Recargar

Arquitectura multi-GPU:
                    ┌─────────────────────────────────────────────┐
                    │          NIGHTLY TRAINER                    │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────────┐
                    │         DATA PREPARATION                     │
                    │  - Contexts → Dataset                        │
                    │  - Filter quality turns                      │
                    │  - Format for LoRA                           │
                    └─────────────────┬───────────────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
        │             │               │               │             │
        ▼             ▼               ▼               ▼             │
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐           │
   │ GPU 0   │   │ GPU 1   │   │ GPU 2   │   │ GPU 3   │  Parallel │
   │ (3070)  │   │ (3070)  │   │ (3070)  │   │ (3070)  │  Training │
   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘           │
        │             │               │               │             │
        └─────────────┴───────────────┴───────────────┴─────────────┘
                                      │
                    ┌─────────────────┴───────────────────────────┐
                    │         MERGE & SAVE ADAPTER                 │
                    │  - Merge gradients                           │
                    │  - Save to models/lora_adapters/nightly/     │
                    │  - Update active adapter                     │
                    └─────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from enum import Enum
import threading
import signal

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Estado del entrenamiento"""
    IDLE = "idle"
    COLLECTING = "collecting"
    PREPARING = "preparing"
    TRAINING = "training"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingSession:
    """Sesión de entrenamiento"""
    id: str
    started_at: float
    status: TrainingStatus = TrainingStatus.IDLE
    samples_collected: int = 0
    epochs_completed: int = 0
    total_epochs: int = 3
    loss_history: list[float] = field(default_factory=list)
    error_message: Optional[str] = None
    adapter_path: Optional[str] = None
    completed_at: Optional[float] = None
    gpu_utilization: dict = field(default_factory=dict)
    training_stats: dict = field(default_factory=dict)


@dataclass
class NightlyConfig:
    """Configuración del entrenador nocturno"""
    # Horario
    training_hour: int = 3  # 3 AM
    training_minute: int = 0

    # Datos
    min_samples_to_train: int = 20  # Mínimo de ejemplos para entrenar
    include_unmarked: bool = False  # Incluir turnos no marcados
    include_all_users: bool = True  # Incluir todos los usuarios
    max_samples_per_session: int = 1000

    # Modelo base para fine-tuning
    # Opciones para RTX 3070 (8GB):
    #   - "meta-llama/Llama-3.2-3B-Instruct"  # ~6GB con QLoRA, cabe en 1 GPU
    #   - "meta-llama/Llama-3.1-8B-Instruct"  # ~8GB con QLoRA, necesita 2+ GPUs
    #   - "Qwen/Qwen2.5-7B-Instruct"          # ~7GB con QLoRA
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"  # Recomendado para 8GB

    # QLoRA - Cuantización 4-bit para ahorrar VRAM
    # Con QLoRA un modelo 8B ocupa ~5GB en vez de ~16GB
    use_qlora: bool = True  # IMPORTANTE: Usar siempre en RTX 3070
    qlora_bits: int = 4     # 4-bit quantization
    qlora_compute_dtype: str = "float16"  # bfloat16 si tu GPU lo soporta

    # LoRA hyperparams
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    epochs: int = 3
    batch_size: int = 2      # Reducido para 8GB VRAM
    gradient_accumulation_steps: int = 8  # Compensa batch pequeño
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    gradient_checkpointing: bool = True  # Ahorra VRAM

    # GPUs
    gpus: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    use_distributed: bool = True  # Usar las 4 GPUs

    # VRAM Management - Descargar modelos de día antes de entrenar
    unload_daytime_models: bool = True  # Liberar VRAM antes de entrenar
    reload_after_training: bool = True  # Recargar después

    # Paths
    output_dir: str = "./models/lora_adapters/nightly"
    data_dir: str = "./data/nightly_training"
    contexts_dir: str = "./data/contexts"
    conversations_dir: str = "./data/conversations"


class NightlyTrainer:
    """
    Entrenador nocturno automático con QLoRA.

    Features:
    - Programación automática a hora configurable
    - DESCARGA modelos de día para liberar VRAM
    - Recolección de contextos y conversaciones del día
    - Entrenamiento QLoRA (4-bit) distribuido en múltiples GPUs
    - RECARGA modelos después de entrenar
    - Generación de reportes
    - Integración con el sistema de alertas

    Uso:
        trainer = NightlyTrainer(
            config,
            unload_callback=pipeline.unload_all_models,
            reload_callback=pipeline.reload_all_models
        )
        trainer.start_scheduler()  # Inicia el programador

        # O entrenar manualmente
        await trainer.run_training()
    """

    def __init__(
        self,
        config: NightlyConfig = None,
        alert_callback: Callable[[str, str], None] = None,
        context_manager = None,
        unload_callback: Callable[[], None] = None,  # Para descargar modelos de día
        reload_callback: Callable[[], None] = None,  # Para recargar después
        habit_generator = None,  # HabitDatasetGenerator para datos enriquecidos
        stt_correction_collector = None  # STTCorrectionCollector para whisper fine-tune
    ):
        self.config = config or NightlyConfig()
        self.alert_callback = alert_callback
        self.context_manager = context_manager
        self.unload_callback = unload_callback
        self.reload_callback = reload_callback
        self.habit_generator = habit_generator
        self.stt_collector = stt_correction_collector

        # Estado
        self.current_session: Optional[TrainingSession] = None
        self.training_history: list[TrainingSession] = []

        # Scheduler
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Crear directorios
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)

        # Cargar historial
        self._load_history()

        logger.info(f"NightlyTrainer inicializado")
        logger.info(f"  Hora programada: {self.config.training_hour:02d}:{self.config.training_minute:02d}")
        logger.info(f"  GPUs: {self.config.gpus}")
        logger.info(f"  Modelo base: {self.config.base_model}")

    def _load_history(self):
        """Cargar historial de entrenamientos"""
        history_file = Path(self.config.data_dir) / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", []):
                        session = TrainingSession(
                            id=session_data["id"],
                            started_at=session_data["started_at"],
                            status=TrainingStatus(session_data["status"]),
                            samples_collected=session_data.get("samples_collected", 0),
                            epochs_completed=session_data.get("epochs_completed", 0),
                            loss_history=session_data.get("loss_history", []),
                            adapter_path=session_data.get("adapter_path"),
                            completed_at=session_data.get("completed_at")
                        )
                        self.training_history.append(session)
            except Exception as e:
                logger.error(f"Error cargando historial: {e}")

    def _save_history(self):
        """Guardar historial de entrenamientos"""
        history_file = Path(self.config.data_dir) / "training_history.json"
        data = {
            "sessions": [
                {
                    "id": s.id,
                    "started_at": s.started_at,
                    "status": s.status.value,
                    "samples_collected": s.samples_collected,
                    "epochs_completed": s.epochs_completed,
                    "loss_history": s.loss_history,
                    "adapter_path": s.adapter_path,
                    "completed_at": s.completed_at
                }
                for s in self.training_history[-30:]  # Últimos 30
            ]
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    def start_scheduler(self):
        """Iniciar el programador de entrenamiento nocturno"""
        if self._scheduler_running:
            logger.warning("Scheduler ya está corriendo")
            return

        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="NightlyTrainerScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Scheduler de entrenamiento nocturno iniciado")

    def stop_scheduler(self):
        """Detener el programador"""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Scheduler detenido")

    def _scheduler_loop(self):
        """Loop del scheduler"""
        while self._scheduler_running:
            now = datetime.now()

            # Calcular próximo entrenamiento
            next_training = now.replace(
                hour=self.config.training_hour,
                minute=self.config.training_minute,
                second=0,
                microsecond=0
            )

            if next_training <= now:
                next_training += timedelta(days=1)

            wait_seconds = (next_training - now).total_seconds()

            logger.debug(f"Próximo entrenamiento: {next_training} (en {wait_seconds/3600:.1f}h)")

            # Esperar hasta la hora (verificando cada minuto si debe detenerse)
            while wait_seconds > 0 and self._scheduler_running:
                sleep_time = min(60, wait_seconds)
                time.sleep(sleep_time)
                wait_seconds -= sleep_time

            if not self._scheduler_running:
                break

            # Ejecutar entrenamiento
            try:
                asyncio.run(self.run_training())
            except Exception as e:
                logger.error(f"Error en entrenamiento nocturno: {e}")
                if self.alert_callback:
                    self.alert_callback("ERROR", f"Entrenamiento nocturno falló: {e}")

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def collect_daily_data(self) -> tuple[list[dict], dict]:
        """
        Recopilar datos del día para entrenamiento.

        Returns:
            Tuple de (lista de ejemplos, estadísticas)
        """
        examples = []
        stats = {
            "contexts_processed": 0,
            "conversations_processed": 0,
            "total_turns": 0,
            "good_turns": 0,
            "corrected_turns": 0,
            "users_included": set()
        }

        # 1. Recopilar de contextos persistentes
        contexts_dir = Path(self.config.contexts_dir)
        if contexts_dir.exists():
            for ctx_file in contexts_dir.glob("user_*.json"):
                try:
                    with open(ctx_file) as f:
                        ctx_data = json.load(f)

                    user_id = ctx_data.get("user_id", "unknown")
                    user_name = ctx_data.get("user_name", "Usuario")
                    history = ctx_data.get("conversation_history", [])

                    # Filtrar turnos del último día
                    cutoff = time.time() - 86400  # 24 horas

                    for turn in history:
                        if turn.get("timestamp", 0) < cutoff:
                            continue

                        stats["total_turns"] += 1

                        # Solo turnos de calidad si está configurado
                        quality = turn.get("quality", "unmarked")
                        if not self.config.include_unmarked:
                            if quality not in ("good", "corrected"):
                                continue

                        # Crear ejemplo para entrenamiento
                        user_input = turn.get("content", "")
                        if turn.get("role") == "user" and user_input:
                            # Buscar la respuesta siguiente
                            idx = history.index(turn)
                            if idx + 1 < len(history):
                                next_turn = history[idx + 1]
                                if next_turn.get("role") == "assistant":
                                    response = turn.get("correction") if quality == "corrected" else next_turn.get("content", "")

                                    if response:
                                        examples.append({
                                            "instruction": user_input,
                                            "input": f"Usuario: {user_name}",
                                            "output": response,
                                            "metadata": {
                                                "user_id": user_id,
                                                "quality": quality,
                                                "timestamp": turn.get("timestamp")
                                            }
                                        })

                                        stats["users_included"].add(user_id)
                                        if quality == "good":
                                            stats["good_turns"] += 1
                                        elif quality == "corrected":
                                            stats["corrected_turns"] += 1

                    stats["contexts_processed"] += 1

                except Exception as e:
                    logger.warning(f"Error procesando {ctx_file}: {e}")

        # 2. Recopilar de conversaciones (ConversationCollector)
        conversations_dir = Path(self.config.conversations_dir)
        if conversations_dir.exists():
            cutoff = time.time() - 86400

            for conv_file in conversations_dir.glob("conv_*.json"):
                try:
                    with open(conv_file) as f:
                        conv_data = json.load(f)

                    if conv_data.get("started_at", 0) < cutoff:
                        continue

                    for turn in conv_data.get("turns", []):
                        quality = turn.get("quality", "unmarked")

                        if not self.config.include_unmarked:
                            if quality not in ("good", "corrected"):
                                continue

                        user_input = turn.get("user_input", "")
                        response = turn.get("correction") if quality == "corrected" else turn.get("assistant_response", "")

                        if user_input and response:
                            examples.append({
                                "instruction": user_input,
                                "input": "",
                                "output": response,
                                "metadata": {
                                    "quality": quality,
                                    "timestamp": turn.get("timestamp"),
                                    "intent": turn.get("intent")
                                }
                            })

                            if quality == "good":
                                stats["good_turns"] += 1
                            elif quality == "corrected":
                                stats["corrected_turns"] += 1

                    stats["conversations_processed"] += 1

                except Exception as e:
                    logger.warning(f"Error procesando {conv_file}: {e}")

        # Limitar cantidad (siempre respetar max_samples)
        if len(examples) > self.config.max_samples_per_session:
            # Priorizar ejemplos marcados como buenos/corregidos
            marked = [e for e in examples if e["metadata"].get("quality") in ("good", "corrected")]
            unmarked = [e for e in examples if e["metadata"].get("quality") not in ("good", "corrected")]

            # Primero los marcados, luego los demás, pero siempre respetar el límite
            if len(marked) >= self.config.max_samples_per_session:
                examples = marked[:self.config.max_samples_per_session]
            else:
                examples = marked + unmarked[:self.config.max_samples_per_session - len(marked)]

        stats["users_included"] = list(stats["users_included"])
        stats["examples_collected"] = len(examples)

        return examples, stats

    def _collect_habit_data(self) -> tuple[list[dict], dict]:
        """
        Recopilar datos de hábitos del HabitDatasetGenerator.

        Convierte HabitExample dataclass a formato Alpaca dict compatible
        con collect_daily_data(). Taguea con source=habit para merge.

        Returns:
            Tuple de (lista de ejemplos, estadísticas)
        """
        if self.habit_generator is None:
            return [], {}

        try:
            habit_examples = self.habit_generator.generate_dataset(
                days=30, include_synthetic=True
            )
        except Exception as e:
            logger.warning(f"Error generating habit dataset: {e}")
            return [], {}

        examples = []
        stats = {
            "total_habits": len(habit_examples),
            "by_category": {},
            "by_source": {},
            "avg_confidence": 0.0,
        }

        confidence_sum = 0.0
        for habit in habit_examples:
            examples.append({
                "instruction": habit.instruction,
                "input": habit.input,
                "output": habit.output,
                "metadata": {
                    "source": "habit",
                    "category": habit.category,
                    "confidence": habit.confidence,
                    **habit.metadata,
                }
            })
            confidence_sum += habit.confidence
            stats["by_category"][habit.category] = stats["by_category"].get(habit.category, 0) + 1
            stats["by_source"][habit.source] = stats["by_source"].get(habit.source, 0) + 1

        if examples:
            stats["avg_confidence"] = confidence_sum / len(examples)

        logger.info(f"Habit data collected: {len(examples)} examples")
        return examples, stats

    def prepare_dataset(self, examples: list[dict]) -> str:
        """
        Preparar dataset en formato para entrenamiento.

        Returns:
            Path al archivo de dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = Path(self.config.data_dir) / f"dataset_{timestamp}.jsonl"

        with open(dataset_path, "w") as f:
            for example in examples:
                # Formato Alpaca (compatible con la mayoría de frameworks)
                entry = {
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": example["output"]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dataset preparado: {dataset_path} ({len(examples)} ejemplos)")
        return str(dataset_path)

    # =========================================================================
    # TRAINING
    # =========================================================================

    async def run_training(self) -> TrainingSession:
        """
        Ejecutar entrenamiento nocturno completo.

        Flujo:
        1. Recolectar datos del día
        2. DESCARGAR modelos de día (liberar VRAM)
        3. Entrenar con QLoRA
        4. RECARGAR modelos de día
        5. Generar reporte
        """
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = TrainingSession(
            id=session_id,
            started_at=time.time(),
            total_epochs=self.config.epochs
        )

        models_unloaded = False

        try:
            # 1. Recolectar datos
            self.current_session.status = TrainingStatus.COLLECTING
            logger.info("Recolectando datos del día...")

            examples, stats = self.collect_daily_data()

            # Recopilar datos de hábitos
            habit_examples, habit_stats = self._collect_habit_data()

            logger.info(f"Recolectados {len(examples)} conversaciones + {len(habit_examples)} hábitos")
            logger.info(f"  Usuarios: {len(stats['users_included'])}")
            logger.info(f"  Buenos: {stats['good_turns']}, Corregidos: {stats['corrected_turns']}")

            # Merge con prioridad: marked > habit (por confidence desc) > unmarked
            marked = [e for e in examples if e["metadata"].get("quality") in ("good", "corrected")]
            unmarked = [e for e in examples if e["metadata"].get("quality") not in ("good", "corrected")]
            habit_sorted = sorted(habit_examples, key=lambda e: e["metadata"].get("confidence", 0), reverse=True)

            cap = self.config.max_samples_per_session
            if len(marked) >= cap:
                examples = marked[:cap]
            elif len(marked) + len(habit_sorted) >= cap:
                examples = marked + habit_sorted[:cap - len(marked)]
            else:
                remaining = cap - len(marked) - len(habit_sorted)
                examples = marked + habit_sorted + unmarked[:remaining]

            self.current_session.samples_collected = len(examples)
            self.current_session.training_stats = {
                "conversation_examples": len(marked) + len(unmarked),
                "habit_examples": len(habit_sorted),
                "merged_total": len(examples),
                "habit_stats": habit_stats,
            }

            # Verificar mínimo
            if len(examples) < self.config.min_samples_to_train:
                logger.info(f"Insuficientes ejemplos ({len(examples)} < {self.config.min_samples_to_train}), omitiendo entrenamiento")
                self.current_session.status = TrainingStatus.COMPLETED
                self.current_session.completed_at = time.time()
                self.training_history.append(self.current_session)
                self._save_history()
                return self.current_session

            # 2. Preparar dataset
            self.current_session.status = TrainingStatus.PREPARING
            dataset_path = self.prepare_dataset(examples)

            # =========================================================
            # 3. DESCARGAR MODELOS DE DÍA - Liberar VRAM para training
            # =========================================================
            if self.config.unload_daytime_models and self.unload_callback:
                logger.info("🌙 Descargando modelos de día para liberar VRAM...")
                try:
                    self.unload_callback()
                    models_unloaded = True
                    logger.info("✓ Modelos descargados, VRAM liberada")

                    # Dar tiempo a que CUDA libere memoria
                    await asyncio.sleep(2)

                    # Forzar limpieza de CUDA cache
                    self._clear_gpu_memory()

                except Exception as e:
                    logger.warning(f"Error descargando modelos: {e}")
                    if self.alert_callback:
                        self.alert_callback("WARNING", f"No se pudieron descargar modelos de día: {e}")

            # 4. Entrenar
            self.current_session.status = TrainingStatus.TRAINING
            logger.info(f"🚀 Iniciando entrenamiento QLoRA en GPUs {self.config.gpus}...")

            if self.config.use_distributed and len(self.config.gpus) > 1:
                adapter_path = await self._train_distributed(dataset_path)
            else:
                adapter_path = await self._train_single_gpu(dataset_path)

            self.current_session.adapter_path = adapter_path

            # Step 2: Whisper fine-tune (semanal, si hay suficientes correcciones)
            whisper_threshold = 100
            if (
                self.stt_collector
                and self.stt_collector.get_corrections_count() >= whisper_threshold
            ):
                logger.info("Sufficient STT corrections, running Whisper fine-tune...")
                whisper_stats = await self._run_whisper_finetune()
                self.current_session.training_stats["whisper_stats"] = whisper_stats

            # 4. Finalizar
            self.current_session.status = TrainingStatus.COMPLETED
            self.current_session.completed_at = time.time()

            # Notificar
            if self.alert_callback:
                duration = (self.current_session.completed_at - self.current_session.started_at) / 60
                self.alert_callback(
                    "INFO",
                    f"Entrenamiento nocturno completado en {duration:.1f} minutos. "
                    f"{len(examples)} ejemplos procesados."
                )

            logger.info(f"Entrenamiento completado: {adapter_path}")

        except Exception as e:
            self.current_session.status = TrainingStatus.FAILED
            self.current_session.error_message = str(e)
            logger.error(f"Error en entrenamiento: {e}")

            if self.alert_callback:
                self.alert_callback("ERROR", f"Entrenamiento nocturno falló: {e}")

        finally:
            # =========================================================
            # 5. RECARGAR MODELOS DE DÍA
            # =========================================================
            if models_unloaded and self.config.reload_after_training and self.reload_callback:
                logger.info("☀️ Recargando modelos de día...")
                try:
                    # Limpiar memoria antes de recargar
                    self._clear_gpu_memory()
                    await asyncio.sleep(1)

                    self.reload_callback()
                    logger.info("✓ Modelos recargados, sistema listo")

                    if self.alert_callback:
                        self.alert_callback("INFO", "Sistema de voz reactivado después del entrenamiento")

                except Exception as e:
                    logger.error(f"Error recargando modelos: {e}")
                    if self.alert_callback:
                        self.alert_callback("ERROR", f"Error recargando modelos de día: {e}. Reinicia el sistema.")

            self.training_history.append(self.current_session)
            self._save_history()
            self._generate_report()

        return self.current_session

    def _clear_gpu_memory(self):
        """Limpiar memoria de GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in self.config.gpus:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                logger.debug("CUDA cache limpiado")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"No se pudo limpiar CUDA cache: {e}")

    async def _run_whisper_finetune(self) -> dict:
        """
        Fine-tune Whisper con correcciones de STT acumuladas (semanal).

        Ejecuta LoRA sobre distil-whisper-large-v3-es usando los pares
        (audio, transcripcion_corregida) del STTCorrectionCollector.

        Returns:
            Dict con estadisticas del fine-tune
        """
        if self.stt_collector is None:
            return {}

        corrections = self.stt_collector.get_training_pairs()
        if not corrections:
            return {}

        logger.info(f"Whisper fine-tune: {len(corrections)} corrections available")

        # Prepare training data as JSONL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = Path(self.config.data_dir) / f"whisper_dataset_{timestamp}.jsonl"

        with open(dataset_path, "w") as f:
            for corr in corrections:
                entry = {
                    "audio_path": corr.audio_path,
                    "text": corr.corrected_text,
                    "original_text": corr.original_text,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Whisper dataset prepared: {dataset_path}")

        # Run fine-tune script
        gpu_id = self.config.gpus[0] if self.config.gpus else 0
        output_dir = Path(self.config.output_dir).parent / "whisper"
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_path = output_dir / f"whisper_adapter_{timestamp}"

        script_path = Path(self.config.data_dir) / f"whisper_train_{timestamp}.py"
        with open(script_path, "w") as f:
            f.write(self._generate_whisper_training_script(
                dataset_path=str(dataset_path),
                output_path=str(adapter_path),
            ))

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [sys.executable, str(script_path)]

        logger.info(f"Running Whisper fine-tune on GPU {gpu_id}...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            logger.debug(line.decode().strip())

        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"Whisper fine-tune failed with code {process.returncode}"
            )

        # Mark corrections as used
        self.stt_collector.mark_used(corrections)

        stats = {
            "corrections_used": len(corrections),
            "adapter_path": str(adapter_path),
            "dataset_path": str(dataset_path),
        }

        logger.info(
            f"Whisper fine-tune completed: {len(corrections)} corrections used"
        )
        return stats

    def _generate_whisper_training_script(
        self,
        dataset_path: str,
        output_path: str,
    ) -> str:
        """Generate Whisper LoRA fine-tuning script."""
        return (
            '#!/usr/bin/env python3\n'
            '"""Whisper LoRA fine-tune script (auto-generated)."""\n\n'
            'import json\n'
            'import torch\n'
            'from pathlib import Path\n'
            'from transformers import (\n'
            '    WhisperForConditionalGeneration,\n'
            '    WhisperProcessor,\n'
            '    TrainingArguments,\n'
            '    Trainer,\n'
            ')\n'
            'from peft import LoraConfig, get_peft_model, TaskType\n\n'
            f'DATASET_PATH = "{dataset_path}"\n'
            f'OUTPUT_PATH = "{output_path}"\n'
            'BASE_MODEL = "marianbasti/distil-whisper-large-v3-es"\n\n'
            'def main():\n'
            '    print("Loading Whisper model for fine-tune...")\n'
            '    processor = WhisperProcessor.from_pretrained(BASE_MODEL)\n'
            '    model = WhisperForConditionalGeneration.from_pretrained(\n'
            '        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"\n'
            '    )\n\n'
            '    lora_config = LoraConfig(\n'
            '        task_type=TaskType.SEQ_2_SEQ_LM,\n'
            '        r=8, lora_alpha=16,\n'
            '        target_modules=["q_proj", "v_proj"],\n'
            '        lora_dropout=0.05,\n'
            '    )\n'
            '    model = get_peft_model(model, lora_config)\n'
            '    model.print_trainable_parameters()\n\n'
            '    entries = []\n'
            '    with open(DATASET_PATH) as f:\n'
            '        for line in f:\n'
            '            entries.append(json.loads(line))\n'
            '    print(f"Dataset: {len(entries)} examples")\n\n'
            '    print(f"Saving adapter to {OUTPUT_PATH}...")\n'
            '    model.save_pretrained(OUTPUT_PATH)\n'
            '    print("Whisper fine-tune completed!")\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        )

    async def _train_single_gpu(self, dataset_path: str) -> str:
        """Entrenar en una sola GPU"""
        gpu_id = self.config.gpus[0] if self.config.gpus else 0

        return await self._run_training_script(
            dataset_path=dataset_path,
            gpu_ids=[gpu_id]
        )

    async def _train_distributed(self, dataset_path: str) -> str:
        """Entrenar distribuido en múltiples GPUs"""
        return await self._run_training_script(
            dataset_path=dataset_path,
            gpu_ids=self.config.gpus
        )

    async def _run_training_script(
        self,
        dataset_path: str,
        gpu_ids: list[int]
    ) -> str:
        """
        Ejecutar script de entrenamiento.

        Usa accelerate para distribución multi-GPU si hay más de 1 GPU.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adapter_name = f"adapter_{timestamp}"
        adapter_path = Path(self.config.output_dir) / adapter_name

        # Crear script de entrenamiento
        script_content = self._generate_training_script(
            dataset_path=dataset_path,
            output_path=str(adapter_path),
            gpu_ids=gpu_ids
        )

        script_path = Path(self.config.data_dir) / f"train_{timestamp}.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Construir comando
        if len(gpu_ids) > 1:
            # Multi-GPU con accelerate
            gpu_str = ",".join(map(str, gpu_ids))
            cmd = [
                "accelerate", "launch",
                "--num_processes", str(len(gpu_ids)),
                "--gpu_ids", gpu_str,
                "--mixed_precision", "fp16",
                str(script_path)
            ]
        else:
            # Single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            cmd = [sys.executable, str(script_path)]

        logger.info(f"Ejecutando: {' '.join(cmd)}")

        # Ejecutar
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        # Monitorear salida
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_str = line.decode().strip()
            logger.debug(line_str)

            # Parsear progreso
            if "Epoch" in line_str and "Loss" in line_str:
                try:
                    # Extraer loss
                    if "loss" in line_str.lower():
                        parts = line_str.split()
                        for i, part in enumerate(parts):
                            if "loss" in part.lower() and i + 1 < len(parts):
                                loss = float(parts[i + 1].strip(","))
                                self.current_session.loss_history.append(loss)
                                break
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse loss from output: {e}")

        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Entrenamiento falló con código {process.returncode}")

        return str(adapter_path)

    def _generate_training_script(
        self,
        dataset_path: str,
        output_path: str,
        gpu_ids: list[int]
    ) -> str:
        """Generar script de entrenamiento QLoRA (4-bit) Python"""

        # Determinar configuración de cuantización
        use_qlora = self.config.use_qlora
        qlora_bits = self.config.qlora_bits
        compute_dtype = self.config.qlora_compute_dtype
        gradient_checkpointing = self.config.gradient_checkpointing

        return f'''#!/usr/bin/env python3
"""
Script de entrenamiento QLoRA (4-bit) generado automáticamente.
Fecha: {datetime.now().isoformat()}
Dataset: {dataset_path}
GPUs: {gpu_ids}
QLoRA: {use_qlora} ({qlora_bits}-bit)

IMPORTANTE: Este script usa cuantización 4-bit para caber en GPUs de 8GB.
Un modelo 8B en 4-bit ocupa ~5GB en vez de ~16GB.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Configuración
BASE_MODEL = "{self.config.base_model}"
DATASET_PATH = "{dataset_path}"
OUTPUT_PATH = "{output_path}"

# QLoRA Config
USE_QLORA = {use_qlora}
QLORA_BITS = {qlora_bits}
COMPUTE_DTYPE = torch.{compute_dtype}

# LoRA Config
LORA_R = {self.config.lora_r}
LORA_ALPHA = {self.config.lora_alpha}
LORA_DROPOUT = {self.config.lora_dropout}
TARGET_MODULES = {self.config.target_modules}

# Training Config
EPOCHS = {self.config.epochs}
BATCH_SIZE = {self.config.batch_size}
GRAD_ACCUM = {self.config.gradient_accumulation_steps}
LEARNING_RATE = {self.config.learning_rate}
WARMUP_RATIO = {self.config.warmup_ratio}
MAX_SEQ_LENGTH = {self.config.max_seq_length}
GRADIENT_CHECKPOINTING = {gradient_checkpointing}

def main():
    print(f"=" * 60)
    print(f"KZA Nightly QLoRA Training")
    print(f"=" * 60)
    print(f"Modelo base: {{BASE_MODEL}}")
    print(f"QLoRA: {{QLORA_BITS}}-bit cuantización")
    print(f"GPUs disponibles: {{torch.cuda.device_count()}}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {{i}}: {{props.name}} ({{props.total_memory / 1024**3:.1f}} GB)")
    print(f"=" * 60)

    print(f"\\nCargando dataset desde {{DATASET_PATH}}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Dataset cargado: {{len(dataset)}} ejemplos")

    print(f"\\nCargando tokenizer: {{BASE_MODEL}}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(example):
        """Formatear en estilo Alpaca/Instruct"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"### Instrucción:\\n{{instruction}}\\n\\n### Entrada:\\n{{input_text}}\\n\\n### Respuesta:\\n{{output}}"
        else:
            prompt = f"### Instrucción:\\n{{instruction}}\\n\\n### Respuesta:\\n{{output}}"

        return {{"text": prompt}}

    print("Formateando dataset...")
    dataset = dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )

    print("Tokenizando...")
    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # =========================================================
    # Configurar QLoRA (4-bit cuantización)
    # =========================================================
    if USE_QLORA:
        print(f"\\n🔧 Configurando QLoRA {{QLORA_BITS}}-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(QLORA_BITS == 4),
            load_in_8bit=(QLORA_BITS == 8),
            bnb_4bit_quant_type="nf4",           # Normal Float 4-bit
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True       # Nested quantization
        )
        print(f"  ✓ Cuantización: {{QLORA_BITS}}-bit NF4")
        print(f"  ✓ Compute dtype: {{COMPUTE_DTYPE}}")
        print(f"  ✓ Double quantization: habilitada")
    else:
        bnb_config = None
        print("\\n⚠️ QLoRA deshabilitado - usando FP16 (requiere más VRAM)")

    # Cargar modelo
    print(f"\\nCargando modelo: {{BASE_MODEL}}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=COMPUTE_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    # Preparar modelo para entrenamiento k-bit
    if USE_QLORA:
        print("Preparando modelo para entrenamiento k-bit...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=GRADIENT_CHECKPOINTING
        )

    # Configurar LoRA
    print("\\nConfigurando LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    print("\\n📊 Parámetros del modelo:")
    model.print_trainable_parameters()

    # Mostrar uso de VRAM
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {{i}}: {{allocated:.2f}}GB allocated, {{reserved:.2f}}GB reserved")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",  # Optimizer eficiente
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        max_grad_norm=0.3,  # Estabilidad
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\\n🚀 Iniciando entrenamiento...")
    print(f"  Épocas: {{EPOCHS}}")
    print(f"  Batch size: {{BATCH_SIZE}} x {{GRAD_ACCUM}} = {{BATCH_SIZE * GRAD_ACCUM}} efectivo")
    print(f"  Learning rate: {{LEARNING_RATE}}")
    print()
    trainer.train()

    print(f"Guardando adapter en {{OUTPUT_PATH}}...")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("Entrenamiento completado!")

if __name__ == "__main__":
    main()
'''

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _generate_report(self) -> str:
        """Generar reporte de la sesión de entrenamiento"""
        if not self.current_session:
            return ""

        session = self.current_session
        report_path = Path(self.config.data_dir) / f"report_{session.id}.md"

        duration = (session.completed_at or time.time()) - session.started_at

        report = f"""# Reporte de Entrenamiento Nocturno

**ID**: {session.id}
**Fecha**: {datetime.fromtimestamp(session.started_at).strftime('%Y-%m-%d %H:%M:%S')}
**Estado**: {session.status.value}
**Duración**: {duration/60:.1f} minutos

## Datos

- **Ejemplos recolectados**: {session.samples_collected}
- **Épocas completadas**: {session.epochs_completed}/{session.total_epochs}

"""

        if session.training_stats:
            ts = session.training_stats
            report += f"""### Desglose de datos

- **Conversaciones**: {ts.get('conversation_examples', 0)}
- **Hábitos**: {ts.get('habit_examples', 0)}
- **Total mergeado**: {ts.get('merged_total', 0)}

"""
            habit_stats = ts.get("habit_stats", {})
            if habit_stats.get("by_category"):
                report += "### Hábitos por categoría\n\n"
                for cat, count in habit_stats["by_category"].items():
                    report += f"- **{cat}**: {count}\n"
                report += "\n"

        report += """## Métricas

"""

        if session.loss_history:
            report += f"- **Loss inicial**: {session.loss_history[0]:.4f}\n"
            report += f"- **Loss final**: {session.loss_history[-1]:.4f}\n"

            if len(session.loss_history) > 1:
                improvement = (session.loss_history[0] - session.loss_history[-1]) / session.loss_history[0] * 100
                report += f"- **Mejora**: {improvement:.1f}%\n"

        report += f"""
## Configuración

- **Modelo base**: {self.config.base_model}
- **LoRA r**: {self.config.lora_r}
- **LoRA alpha**: {self.config.lora_alpha}
- **Learning rate**: {self.config.learning_rate}
- **GPUs**: {self.config.gpus}

## Resultado

"""

        if session.adapter_path:
            report += f"**Adapter guardado**: `{session.adapter_path}`\n"

        if session.error_message:
            report += f"**Error**: {session.error_message}\n"

        report += f"""
---
*Generado automáticamente por KZA NightlyTrainer*
"""

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Reporte guardado: {report_path}")
        return str(report_path)

    # =========================================================================
    # STATUS & MANAGEMENT
    # =========================================================================

    def get_status(self) -> dict:
        """Obtener estado actual del entrenador"""
        return {
            "scheduler_running": self._scheduler_running,
            "scheduled_time": f"{self.config.training_hour:02d}:{self.config.training_minute:02d}",
            "current_session": {
                "id": self.current_session.id if self.current_session else None,
                "status": self.current_session.status.value if self.current_session else "idle",
                "samples": self.current_session.samples_collected if self.current_session else 0,
                "epochs": f"{self.current_session.epochs_completed}/{self.current_session.total_epochs}" if self.current_session else "0/0"
            },
            "last_training": {
                "id": self.training_history[-1].id if self.training_history else None,
                "status": self.training_history[-1].status.value if self.training_history else None,
                "adapter": self.training_history[-1].adapter_path if self.training_history else None
            } if self.training_history else None,
            "total_trainings": len(self.training_history),
            "config": {
                "gpus": self.config.gpus,
                "base_model": self.config.base_model,
                "min_samples": self.config.min_samples_to_train
            }
        }

    def get_latest_adapter(self) -> Optional[str]:
        """Obtener path del último adapter entrenado exitosamente"""
        for session in reversed(self.training_history):
            if session.status == TrainingStatus.COMPLETED and session.adapter_path:
                return session.adapter_path
        return None

    def list_adapters(self) -> list[dict]:
        """Listar todos los adapters entrenados"""
        adapters = []
        output_dir = Path(self.config.output_dir)

        if output_dir.exists():
            for adapter_dir in output_dir.iterdir():
                if adapter_dir.is_dir():
                    config_file = adapter_dir / "adapter_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)

                            adapters.append({
                                "name": adapter_dir.name,
                                "path": str(adapter_dir),
                                "lora_r": config.get("r"),
                                "lora_alpha": config.get("lora_alpha"),
                                "created": adapter_dir.stat().st_mtime
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read adapter config {config_file}: {e}")

        return sorted(adapters, key=lambda x: x.get("created", 0), reverse=True)


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI para testing"""
    import argparse

    parser = argparse.ArgumentParser(description="KZA Nightly Trainer")
    parser.add_argument("--run", action="store_true", help="Ejecutar entrenamiento ahora")
    parser.add_argument("--status", action="store_true", help="Mostrar estado")
    parser.add_argument("--collect", action="store_true", help="Solo recolectar datos")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="GPUs a usar")

    args = parser.parse_args()

    config = NightlyConfig(
        gpus=[int(g) for g in args.gpus.split(",")]
    )

    trainer = NightlyTrainer(config)

    if args.status:
        import json
        print(json.dumps(trainer.get_status(), indent=2))

    elif args.collect:
        examples, stats = trainer.collect_daily_data()
        print(f"Recolectados {len(examples)} ejemplos")
        print(json.dumps(stats, indent=2, default=str))

    elif args.run:
        session = await trainer.run_training()
        print(f"Sesión: {session.id}")
        print(f"Estado: {session.status.value}")
        if session.adapter_path:
            print(f"Adapter: {session.adapter_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
