"""
Tests para NightlyTrainer - Entrenamiento automático nocturno con QLoRA
"""

import pytest
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.training.nightly_trainer import (
    NightlyTrainer,
    NightlyConfig,
    TrainingStatus,
    TrainingSession,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para tests"""
    output_dir = str(tmp_path / "adapters")
    data_dir = str(tmp_path / "data")
    contexts_dir = str(tmp_path / "contexts")
    conversations_dir = str(tmp_path / "conversations")
    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "contexts_dir": contexts_dir,
        "conversations_dir": conversations_dir,
    }


@pytest.fixture
def config(temp_dir):
    """Configuración de test"""
    return NightlyConfig(
        training_hour=3,
        training_minute=0,
        min_samples_to_train=5,
        include_unmarked=True,
        max_samples_per_session=100,
        base_model="test-model",
        epochs=1,
        batch_size=2,
        gpus=[0],
        use_distributed=False,
        output_dir=temp_dir["output_dir"],
        data_dir=temp_dir["data_dir"],
        contexts_dir=temp_dir["contexts_dir"],
        conversations_dir=temp_dir["conversations_dir"],
    )


@pytest.fixture
def trainer(config):
    """NightlyTrainer con config de test"""
    return NightlyTrainer(config=config)


@pytest.fixture
def trainer_with_callbacks(config):
    """NightlyTrainer con callbacks mock"""
    alert_cb = MagicMock()
    unload_cb = MagicMock()
    reload_cb = MagicMock()

    return NightlyTrainer(
        config=config,
        alert_callback=alert_cb,
        unload_callback=unload_cb,
        reload_callback=reload_cb,
    )


def _create_conversation_file(conversations_dir, turns):
    """Helper para crear archivo de conversación"""
    Path(conversations_dir).mkdir(parents=True, exist_ok=True)
    conv_data = {
        "id": f"conv_{int(time.time() * 1000)}",
        "started_at": time.time(),
        "user_name": "Mastar",
        "turns": turns,
    }
    filepath = Path(conversations_dir) / f"{conv_data['id']}.json"
    with open(filepath, "w") as f:
        json.dump(conv_data, f)
    return filepath


def _create_context_file(contexts_dir, user_id, history):
    """Helper para crear archivo de contexto"""
    Path(contexts_dir).mkdir(parents=True, exist_ok=True)
    ctx_data = {
        "user_id": user_id,
        "user_name": "Mastar",
        "conversation_history": history,
    }
    filepath = Path(contexts_dir) / f"user_{user_id}.json"
    with open(filepath, "w") as f:
        json.dump(ctx_data, f)
    return filepath


# =========================================================================
# Tests de configuración
# =========================================================================

class TestConfiguration:
    def test_default_config(self):
        """Verificar configuración por defecto"""
        config = NightlyConfig()
        assert config.training_hour == 3
        assert config.use_qlora is True
        assert config.qlora_bits == 4
        assert config.gradient_checkpointing is True
        assert len(config.gpus) == 4

    def test_custom_config(self, config):
        """Verificar configuración personalizada"""
        assert config.min_samples_to_train == 5
        assert config.gpus == [0]
        assert config.use_distributed is False

    def test_directories_created(self, trainer, temp_dir):
        """Verificar que se crean los directorios"""
        assert Path(temp_dir["output_dir"]).exists()
        assert Path(temp_dir["data_dir"]).exists()


# =========================================================================
# Tests de recolección de datos
# =========================================================================

class TestDataCollection:
    def test_collect_empty(self, trainer):
        """Recolección sin datos produce lista vacía"""
        examples, stats = trainer.collect_daily_data()
        assert len(examples) == 0
        assert stats["total_turns"] == 0

    def test_collect_from_conversations(self, trainer, config):
        """Recolección desde archivos de conversación"""
        turns = [
            {
                "timestamp": time.time(),
                "user_input": "Prende la luz",
                "assistant_response": "Luz encendida",
                "quality": "good",
                "intent": "light_control",
            },
            {
                "timestamp": time.time(),
                "user_input": "Apagá todo",
                "assistant_response": "Todo apagado",
                "quality": "unmarked",
            },
        ]
        _create_conversation_file(config.conversations_dir, turns)

        examples, stats = trainer.collect_daily_data()
        assert len(examples) == 2
        assert stats["conversations_processed"] == 1

    def test_collect_only_marked(self, config):
        """Recolección solo de turnos marcados"""
        config.include_unmarked = False
        trainer = NightlyTrainer(config=config)

        turns = [
            {
                "timestamp": time.time(),
                "user_input": "Prende la luz",
                "assistant_response": "Luz encendida",
                "quality": "good",
            },
            {
                "timestamp": time.time(),
                "user_input": "Hola",
                "assistant_response": "Hola",
                "quality": "unmarked",
            },
        ]
        _create_conversation_file(config.conversations_dir, turns)

        examples, stats = trainer.collect_daily_data()
        assert len(examples) == 1

    def test_collect_from_contexts(self, trainer, config):
        """Recolección desde contextos persistentes"""
        history = [
            {
                "role": "user",
                "content": "Poné el aire a 22",
                "timestamp": time.time(),
                "quality": "good",
            },
            {
                "role": "assistant",
                "content": "Aire puesto a 22 grados",
                "timestamp": time.time(),
            },
        ]
        _create_context_file(config.contexts_dir, "mastar", history)

        examples, stats = trainer.collect_daily_data()
        assert len(examples) >= 1
        assert stats["contexts_processed"] >= 1

    def test_collect_corrected_uses_correction(self, trainer, config):
        """Verificar que turnos corregidos usan la corrección"""
        turns = [
            {
                "timestamp": time.time(),
                "user_input": "Poné el aire",
                "assistant_response": "¿A cuánto?",
                "quality": "corrected",
                "correction": "Pongo el aire a 22 como siempre",
            },
        ]
        _create_conversation_file(config.conversations_dir, turns)

        examples, stats = trainer.collect_daily_data()
        assert len(examples) == 1
        assert examples[0]["output"] == "Pongo el aire a 22 como siempre"

    def test_collect_respects_max_samples(self, config):
        """Verificar límite de muestras"""
        config.max_samples_per_session = 3
        trainer = NightlyTrainer(config=config)

        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Comando {i}",
                "assistant_response": f"Respuesta {i}",
                "quality": "unmarked",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        examples, stats = trainer.collect_daily_data()
        assert len(examples) <= 3


# =========================================================================
# Tests de preparación de dataset
# =========================================================================

class TestDatasetPreparation:
    def test_prepare_dataset(self, trainer):
        """Verificar preparación de dataset JSONL"""
        examples = [
            {
                "instruction": "Prende la luz",
                "input": "",
                "output": "Luz encendida",
                "metadata": {},
            },
            {
                "instruction": "Apagá el aire",
                "input": "Escritorio",
                "output": "Aire apagado",
                "metadata": {},
            },
        ]

        path = trainer.prepare_dataset(examples)
        assert Path(path).exists()

        with open(path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            entry = json.loads(lines[0])
            assert entry["instruction"] == "Prende la luz"
            assert entry["output"] == "Luz encendida"


# =========================================================================
# Tests de entrenamiento
# =========================================================================

class TestTraining:
    @pytest.mark.asyncio
    async def test_training_insufficient_data(self, trainer, config):
        """Entrenamiento con datos insuficientes se omite"""
        config.min_samples_to_train = 100

        session = await trainer.run_training()
        assert session.status == TrainingStatus.COMPLETED
        assert session.samples_collected == 0

    @pytest.mark.asyncio
    async def test_training_unloads_models(self, trainer_with_callbacks, config):
        """Verificar que descarga modelos antes de entrenar"""
        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Comando {i}",
                "assistant_response": f"Respuesta {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        with patch.object(trainer_with_callbacks, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = str(Path(config.output_dir) / "test_adapter")
            session = await trainer_with_callbacks.run_training()

            trainer_with_callbacks.unload_callback.assert_called_once()
            trainer_with_callbacks.reload_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_generates_report(self, trainer, config):
        """Verificar generación de reporte"""
        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Comando {i}",
                "assistant_response": f"Respuesta {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = str(Path(config.output_dir) / "adapter_test")
            await trainer.run_training()

        reports = list(Path(config.data_dir).glob("report_*.md"))
        assert len(reports) == 1

    @pytest.mark.asyncio
    async def test_training_failure_handled(self, trainer_with_callbacks, config):
        """Verificar manejo de errores en entrenamiento"""
        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Cmd {i}",
                "assistant_response": f"Resp {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        with patch.object(trainer_with_callbacks, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.side_effect = RuntimeError("GPU error")
            session = await trainer_with_callbacks.run_training()

        assert session.status == TrainingStatus.FAILED
        assert "GPU error" in session.error_message
        trainer_with_callbacks.alert_callback.assert_called()


# =========================================================================
# Tests de historial y estado
# =========================================================================

class TestHistory:
    def test_save_load_history(self, trainer, config):
        """Verificar persistencia del historial"""
        session = TrainingSession(
            id="test_session",
            started_at=time.time(),
            status=TrainingStatus.COMPLETED,
            samples_collected=50,
            adapter_path="/models/test",
            completed_at=time.time(),
        )
        trainer.training_history.append(session)
        trainer._save_history()

        trainer2 = NightlyTrainer(config=config)
        assert len(trainer2.training_history) == 1
        assert trainer2.training_history[0].id == "test_session"

    def test_get_status(self, trainer):
        """Verificar reporte de estado"""
        status = trainer.get_status()
        assert "scheduler_running" in status
        assert status["scheduled_time"] == "03:00"

    def test_get_latest_adapter(self, trainer):
        """Verificar obtención del último adapter"""
        assert trainer.get_latest_adapter() is None

        session = TrainingSession(
            id="s1",
            started_at=time.time(),
            status=TrainingStatus.COMPLETED,
            adapter_path="/models/adapter_1",
        )
        trainer.training_history.append(session)
        assert trainer.get_latest_adapter() == "/models/adapter_1"

    def test_list_adapters(self, trainer, config):
        """Verificar listado de adapters"""
        adapter_dir = Path(config.output_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump({"r": 16, "lora_alpha": 32}, f)

        adapters = trainer.list_adapters()
        assert len(adapters) == 1
        assert adapters[0]["lora_r"] == 16


# =========================================================================
# Tests del script de entrenamiento generado
# =========================================================================

class TestTrainingScript:
    def test_generate_script(self, trainer):
        """Verificar generación del script de entrenamiento"""
        script = trainer._generate_training_script(
            dataset_path="/data/test.jsonl",
            output_path="/models/test_adapter",
            gpu_ids=[0],
        )
        assert "QLoRA" in script
        assert "BitsAndBytesConfig" in script
        assert "LoraConfig" in script
        assert "test.jsonl" in script

    def test_script_has_qlora_config(self, trainer):
        """Verificar que el script usa cuantización 4-bit"""
        script = trainer._generate_training_script(
            dataset_path="/data/test.jsonl",
            output_path="/models/test",
            gpu_ids=[0],
        )
        assert "load_in_4bit" in script
        assert "nf4" in script
        assert "paged_adamw_8bit" in script


# =========================================================================
# Tests del scheduler
# =========================================================================

class TestScheduler:
    def test_start_stop_scheduler(self, trainer):
        """Verificar inicio y detención del scheduler"""
        trainer.start_scheduler()
        assert trainer._scheduler_running is True
        trainer.stop_scheduler()
        assert trainer._scheduler_running is False

    def test_double_start(self, trainer):
        """Verificar que no se puede iniciar dos veces"""
        trainer.start_scheduler()
        trainer.start_scheduler()
        trainer.stop_scheduler()


# =========================================================================
# Tests de integración con HabitDatasetGenerator
# =========================================================================

class TestHabitIntegration:
    """Tests para la integración de HabitDatasetGenerator en NightlyTrainer"""

    def _make_habit_example(self, instruction="Prende la luz", confidence=0.85):
        """Helper para crear HabitExample mock"""
        from src.training.habit_dataset_generator import HabitExample
        return HabitExample(
            instruction=instruction,
            input="Usuario: Mastar",
            output="Luz encendida",
            category="preference",
            source="pattern",
            confidence=confidence,
            metadata={"user_id": "mastar"},
        )

    def test_collect_habit_data(self, config):
        """Verificar que _collect_habit_data convierte HabitExamples a dicts Alpaca"""
        mock_generator = MagicMock()
        mock_generator.generate_dataset.return_value = [
            self._make_habit_example("Prende la luz", 0.9),
            self._make_habit_example("Apagá el aire", 0.7),
        ]

        trainer = NightlyTrainer(config=config, habit_generator=mock_generator)
        examples, stats = trainer._collect_habit_data()

        assert len(examples) == 2
        assert examples[0]["instruction"] == "Prende la luz"
        assert examples[0]["metadata"]["source"] == "habit"
        assert examples[0]["metadata"]["confidence"] == 0.9
        assert stats["total_habits"] == 2
        mock_generator.generate_dataset.assert_called_once_with(
            days=30, include_synthetic=True
        )

    def test_collect_habit_data_none_generator(self, trainer):
        """Sin habit_generator retorna vacío sin error"""
        assert trainer.habit_generator is None
        examples, stats = trainer._collect_habit_data()
        assert examples == []
        assert stats == {}

    def test_collect_habit_data_error(self, config):
        """Exception en habit_generator no bloquea el training"""
        mock_generator = MagicMock()
        mock_generator.generate_dataset.side_effect = RuntimeError("DB corrupted")

        trainer = NightlyTrainer(config=config, habit_generator=mock_generator)
        examples, stats = trainer._collect_habit_data()

        assert examples == []
        assert stats == {}

    @pytest.mark.asyncio
    async def test_run_training_merges_habits(self, config):
        """run_training combina conversaciones + hábitos en el dataset"""
        # Crear conversaciones
        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Comando {i}",
                "assistant_response": f"Respuesta {i}",
                "quality": "good",
            }
            for i in range(5)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        # Crear habit generator mock
        mock_generator = MagicMock()
        mock_generator.generate_dataset.return_value = [
            self._make_habit_example(f"Hábito {i}", 0.8)
            for i in range(5)
        ]

        trainer = NightlyTrainer(config=config, habit_generator=mock_generator)

        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = str(Path(config.output_dir) / "test_adapter")
            session = await trainer.run_training()

        assert session.status == TrainingStatus.COMPLETED
        assert session.samples_collected == 10  # 5 conv + 5 habit
        assert session.training_stats["conversation_examples"] == 5
        assert session.training_stats["habit_examples"] == 5

    @pytest.mark.asyncio
    async def test_habits_push_over_minimum(self, config):
        """Hábitos + conversaciones alcanzan min_samples_to_train"""
        config.min_samples_to_train = 8

        # Solo 3 conversaciones (insuficiente por sí solas)
        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Cmd {i}",
                "assistant_response": f"Resp {i}",
                "quality": "good",
            }
            for i in range(3)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        # 6 hábitos que alcanzan el mínimo combinado
        mock_generator = MagicMock()
        mock_generator.generate_dataset.return_value = [
            self._make_habit_example(f"Hab {i}", 0.85)
            for i in range(6)
        ]

        trainer = NightlyTrainer(config=config, habit_generator=mock_generator)

        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = str(Path(config.output_dir) / "adapter")
            session = await trainer.run_training()

        # Debería haber entrenado (3 + 6 = 9 >= 8)
        assert session.status == TrainingStatus.COMPLETED
        assert session.samples_collected == 9
        assert mock_train.called


# =========================================================================
# Tests de Whisper fine-tune
# =========================================================================

class TestWhisperFinetune:
    """Tests para el paso de Whisper fine-tune en NightlyTrainer"""

    def _make_mock_collector(self, count: int):
        """Helper: crea STTCorrectionCollector mock con N correcciones"""
        from src.training.stt_correction_collector import STTCorrection

        mock_collector = MagicMock()
        mock_collector.get_corrections_count.return_value = count
        mock_collector.get_training_pairs.return_value = [
            STTCorrection(
                audio_path=f"/tmp/audio_{i}.wav",
                original_text=f"original {i}",
                corrected_text=f"corrected {i}",
                timestamp="2025-01-01T00:00:00",
            )
            for i in range(count)
        ]
        return mock_collector

    @pytest.mark.asyncio
    async def test_whisper_finetune_skips_below_threshold(self, config):
        """No entrena Whisper si hay <100 correcciones"""
        mock_collector = self._make_mock_collector(50)

        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Cmd {i}",
                "assistant_response": f"Resp {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        trainer = NightlyTrainer(
            config=config, stt_correction_collector=mock_collector
        )

        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train:
            mock_train.return_value = str(Path(config.output_dir) / "adapter")
            session = await trainer.run_training()

        assert session.status == TrainingStatus.COMPLETED
        # Whisper fine-tune should NOT have been called
        mock_collector.get_training_pairs.assert_not_called()

    @pytest.mark.asyncio
    async def test_whisper_finetune_runs_above_threshold(self, config):
        """Entrena Whisper si hay >=100 correcciones"""
        mock_collector = self._make_mock_collector(100)

        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Cmd {i}",
                "assistant_response": f"Resp {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        trainer = NightlyTrainer(
            config=config, stt_correction_collector=mock_collector
        )

        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train, \
             patch.object(trainer, '_run_whisper_finetune', new_callable=AsyncMock) as mock_whisper:
            mock_train.return_value = str(Path(config.output_dir) / "adapter")
            mock_whisper.return_value = {"corrections_used": 100}
            session = await trainer.run_training()

        assert session.status == TrainingStatus.COMPLETED
        mock_whisper.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_training_includes_whisper_stats(self, config):
        """run_training incluye whisper_stats en training_stats"""
        mock_collector = self._make_mock_collector(150)

        turns = [
            {
                "timestamp": time.time(),
                "user_input": f"Cmd {i}",
                "assistant_response": f"Resp {i}",
                "quality": "good",
            }
            for i in range(10)
        ]
        _create_conversation_file(config.conversations_dir, turns)

        trainer = NightlyTrainer(
            config=config, stt_correction_collector=mock_collector
        )

        whisper_result = {"corrections_used": 150, "adapter_path": "/models/whisper"}
        with patch.object(trainer, '_run_training_script', new_callable=AsyncMock) as mock_train, \
             patch.object(trainer, '_run_whisper_finetune', new_callable=AsyncMock) as mock_whisper:
            mock_train.return_value = str(Path(config.output_dir) / "adapter")
            mock_whisper.return_value = whisper_result
            session = await trainer.run_training()

        assert session.status == TrainingStatus.COMPLETED
        assert "whisper_stats" in session.training_stats
        assert session.training_stats["whisper_stats"]["corrections_used"] == 150
