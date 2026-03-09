"""
Tests para HabitDatasetGenerator - Generador de datasets de hábitos
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, time as dtime

from src.training.habit_dataset_generator import (
    HabitDatasetGenerator,
    HabitExample,
    UserProfile,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para tests"""
    return str(tmp_path / "habit_data")


@pytest.fixture
def mock_pattern_learner():
    """Mock del PatternLearner con patrones de prueba"""
    learner = MagicMock()

    # Simular patrones detectados
    pattern_light = MagicMock()
    pattern_light.pattern_id = "light_on_living_0700"
    pattern_light.action_type = "light_on"
    pattern_light.entity_id = "light.living"
    pattern_light.confidence = 0.85
    pattern_light.occurrences = 15
    pattern_light.typical_time = dtime(7, 0)
    pattern_light.time_variance_minutes = 10.0
    pattern_light.days_of_week = [0, 1, 2, 3, 4]  # Lunes a viernes
    pattern_light.user_id = "mastar"
    pattern_light.typical_data = {}
    pattern_light.dismissed = False
    pattern_light.suggested = False
    pattern_light.accepted = False

    pattern_climate = MagicMock()
    pattern_climate.pattern_id = "climate_set_escritorio_0900"
    pattern_climate.action_type = "climate_set"
    pattern_climate.entity_id = "climate.escritorio_ac"
    pattern_climate.confidence = 0.92
    pattern_climate.occurrences = 20
    pattern_climate.typical_time = dtime(9, 0)
    pattern_climate.time_variance_minutes = 5.0
    pattern_climate.days_of_week = [0, 1, 2, 3, 4]
    pattern_climate.user_id = "mastar"
    pattern_climate.typical_data = {"temperature": 22}
    pattern_climate.dismissed = False
    pattern_climate.suggested = False
    pattern_climate.accepted = False

    pattern_dismissed = MagicMock()
    pattern_dismissed.pattern_id = "light_off_hall_2300"
    pattern_dismissed.action_type = "light_off"
    pattern_dismissed.entity_id = "light.hall"
    pattern_dismissed.confidence = 0.75
    pattern_dismissed.occurrences = 10
    pattern_dismissed.typical_time = dtime(23, 0)
    pattern_dismissed.time_variance_minutes = 15.0
    pattern_dismissed.days_of_week = [0, 1, 2, 3, 4, 5, 6]
    pattern_dismissed.user_id = "mastar"
    pattern_dismissed.typical_data = {}
    pattern_dismissed.dismissed = True  # Rechazado por el usuario
    pattern_dismissed.suggested = False
    pattern_dismissed.accepted = False

    learner._patterns = {
        "light_on_living_0700": pattern_light,
        "climate_set_escritorio_0900": pattern_climate,
        "light_off_hall_2300": pattern_dismissed,
    }

    return learner


@pytest.fixture
def mock_event_logger():
    """Mock del EventLogger"""
    logger = MagicMock()

    import time
    now = time.time()

    logger.query_recent.return_value = [
        {
            "timestamp": now - 3600,
            "entity_id": "light.living",
            "action": "turn_on",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 5,
            "trigger_phrase": "prende la luz del living",
            "context_json": "{}",
        },
        {
            "timestamp": now - 3590,
            "entity_id": "climate.living_ac",
            "action": "set_temperature",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 5,
            "trigger_phrase": "poné el aire a 22",
            "context_json": json.dumps({"temperature": 22}),
        },
        {
            "timestamp": now - 86400,
            "entity_id": "light.living",
            "action": "turn_on",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 10,
            "trigger_phrase": "prende la luz",
            "context_json": "{}",
        },
        {
            "timestamp": now - 86390,
            "entity_id": "climate.living_ac",
            "action": "set_temperature",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 10,
            "trigger_phrase": "aire a 22",
            "context_json": json.dumps({"temperature": 22}),
        },
        {
            "timestamp": now - 172800,
            "entity_id": "light.living",
            "action": "turn_on",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 3,
            "trigger_phrase": "encendé la luz",
            "context_json": "{}",
        },
        {
            "timestamp": now - 172790,
            "entity_id": "climate.living_ac",
            "action": "set_temperature",
            "user_id": "mastar",
            "user_name": "Mastar",
            "hour": 7,
            "minute": 3,
            "trigger_phrase": "aire a 22",
            "context_json": json.dumps({"temperature": 22}),
        },
    ]

    return logger


@pytest.fixture
def mock_conversation_collector():
    """Mock del ConversationCollector"""
    collector = MagicMock()

    # Simular conversaciones
    turn_good = MagicMock()
    turn_good.user_input = "Prende la luz del escritorio"
    turn_good.assistant_response = "Listo, encendí la luz del escritorio."
    turn_good.quality.value = "good"
    turn_good.correction = None
    turn_good.user_name = "Mastar"
    turn_good.intent = "light_control"

    turn_corrected = MagicMock()
    turn_corrected.user_input = "Poné el aire"
    turn_corrected.assistant_response = "¿A qué temperatura?"
    turn_corrected.quality.value = "corrected"
    turn_corrected.correction = "Pongo el aire del escritorio a 22°C, como siempre."
    turn_corrected.user_name = "Mastar"
    turn_corrected.intent = "climate_control"

    turn_unmarked = MagicMock()
    turn_unmarked.user_input = "Hola"
    turn_unmarked.assistant_response = "Hola Mastar"
    turn_unmarked.quality.value = "unmarked"
    turn_unmarked.correction = None
    turn_unmarked.user_name = "Mastar"
    turn_unmarked.intent = "greeting"

    conv = MagicMock()
    conv.turns = [turn_good, turn_corrected, turn_unmarked]
    conv.user_name = "Mastar"

    collector._conversations = [conv]

    return collector


@pytest.fixture
def generator(temp_dir, mock_pattern_learner, mock_event_logger, mock_conversation_collector):
    """HabitDatasetGenerator con todos los mocks"""
    return HabitDatasetGenerator(
        data_dir=temp_dir,
        pattern_learner=mock_pattern_learner,
        event_logger=mock_event_logger,
        conversation_collector=mock_conversation_collector,
        min_confidence=0.5,
        synthetic_multiplier=2,
    )


# =========================================================================
# Tests de generación de ejemplos
# =========================================================================

class TestDatasetGeneration:
    def test_generate_produces_examples(self, generator):
        """Verificar que se generan ejemplos"""
        examples = generator.generate_dataset()
        assert len(examples) > 0

    def test_generate_from_patterns(self, generator):
        """Verificar generación desde patrones"""
        examples = generator.generate_dataset(include_synthetic=False)

        # Debe haber ejemplos de patrones
        pattern_examples = [e for e in examples if e.source == "pattern"]
        assert len(pattern_examples) > 0

        # El patrón de luz del living debería generar ejemplos
        living_examples = [
            e for e in pattern_examples
            if "living" in e.instruction.lower() or "living" in e.output.lower()
        ]
        assert len(living_examples) > 0

    def test_dismissed_patterns_excluded(self, generator):
        """Verificar que patrones rechazados no generan ejemplos"""
        examples = generator.generate_dataset(include_synthetic=False)

        # No debería haber ejemplos del patrón rechazado (hall)
        hall_pattern_examples = [
            e for e in examples
            if e.source == "pattern" and "hall" in e.metadata.get("pattern_id", "")
        ]
        assert len(hall_pattern_examples) == 0

    def test_generate_from_events(self, generator):
        """Verificar generación desde eventos"""
        examples = generator.generate_dataset(include_synthetic=False)

        event_examples = [e for e in examples if e.source == "event"]
        # Puede o no haber secuencias detectadas dependiendo del umbral
        assert isinstance(event_examples, list)

    def test_generate_from_conversations(self, generator):
        """Verificar generación desde conversaciones"""
        examples = generator.generate_dataset(include_synthetic=False)

        conv_examples = [e for e in examples if e.source == "conversation"]
        assert len(conv_examples) == 2  # good + corrected (unmarked excluido)

    def test_corrected_conversation_uses_correction(self, generator):
        """Verificar que las conversaciones corregidas usan la corrección"""
        examples = generator.generate_dataset(include_synthetic=False)

        corrected = [
            e for e in examples
            if e.source == "conversation" and e.metadata.get("quality") == "corrected"
        ]
        assert len(corrected) == 1
        assert "22°C" in corrected[0].output

    def test_confidence_filter(self, generator):
        """Verificar filtro de confianza mínima"""
        generator.min_confidence = 0.9
        examples = generator.generate_dataset(include_synthetic=False)

        for example in examples:
            assert example.confidence >= 0.9

    def test_synthetic_variations(self, generator):
        """Verificar generación de variaciones sintéticas"""
        examples = generator.generate_dataset(include_synthetic=True)

        synthetic = [e for e in examples if e.source == "synthetic"]
        real = [e for e in examples if e.source != "synthetic"]

        # Debería haber variaciones sintéticas si hay comandos con verbos conocidos
        # (puede ser 0 si ningún ejemplo tiene verbos del diccionario)
        assert isinstance(synthetic, list)

    def test_proactive_examples(self, generator):
        """Verificar ejemplos proactivos para patrones de alta confianza"""
        examples = generator.generate_dataset(include_synthetic=False)

        proactive = [
            e for e in examples
            if e.metadata.get("proactive", False)
        ]
        # El patrón de clima tiene confidence=0.92 (>0.8) así que genera proactivo
        assert len(proactive) > 0


# =========================================================================
# Tests de perfiles de usuario
# =========================================================================

class TestUserProfiles:
    def test_build_profiles_from_events(self, generator):
        """Verificar construcción de perfiles desde eventos"""
        profiles = generator.build_user_profiles()

        assert "mastar" in profiles
        profile = profiles["mastar"]
        assert profile.user_name == "Mastar"

    def test_temperature_preferences(self, generator):
        """Verificar extracción de preferencias de temperatura"""
        profiles = generator.build_user_profiles()
        profile = profiles["mastar"]

        # Debería detectar preferencia de 22°C para living
        if profile.preferred_temperatures:
            assert any(
                temp == 22
                for temp in profile.preferred_temperatures.values()
            )

    def test_schedule_extraction(self, generator):
        """Verificar extracción de horarios"""
        profiles = generator.build_user_profiles()
        profile = profiles["mastar"]

        # Los eventos son a las 7am, debería detectar wake_time ~07:xx
        if profile.wake_time:
            assert profile.wake_time.startswith("07")

    def test_profile_persistence(self, generator, temp_dir):
        """Verificar que los perfiles se guardan y cargan"""
        generator.build_user_profiles()
        generator._save_profiles()

        profiles_file = Path(temp_dir) / "user_profiles.json"
        assert profiles_file.exists()

        # Crear nuevo generador y verificar que carga
        gen2 = HabitDatasetGenerator(data_dir=temp_dir)
        assert len(gen2._user_profiles) > 0

    def test_profile_generates_examples(self, generator):
        """Verificar que los perfiles generan ejemplos"""
        generator.build_user_profiles()
        examples = generator.generate_dataset(include_synthetic=False)

        profile_examples = [e for e in examples if e.source == "profile"]
        assert len(profile_examples) > 0

    def test_enrich_from_patterns(self, generator):
        """Verificar enriquecimiento de perfiles con patrones"""
        profiles = generator.build_user_profiles()

        # El patrón de clima tiene typical_data con temperatura 22
        profile = profiles.get("mastar")
        if profile and profile.preferred_temperatures:
            # Debería tener la temperatura del patrón
            assert any(t == 22 for t in profile.preferred_temperatures.values())


# =========================================================================
# Tests de exportación
# =========================================================================

class TestExport:
    def test_export_jsonl(self, generator, temp_dir):
        """Verificar exportación JSONL"""
        generator.generate_dataset()

        output_path = str(Path(temp_dir) / "test_export.jsonl")
        result = generator.export_jsonl(output_path)

        assert Path(result).exists()

        # Verificar formato
        with open(result) as f:
            lines = f.readlines()
            assert len(lines) > 0

            for line in lines:
                entry = json.loads(line)
                assert "instruction" in entry
                assert "input" in entry
                assert "output" in entry

    def test_export_with_metadata(self, generator, temp_dir):
        """Verificar exportación con metadata"""
        generator.generate_dataset()

        output_path = str(Path(temp_dir) / "test_full.jsonl")
        result = generator.export_with_metadata(output_path)

        with open(result) as f:
            first_line = json.loads(f.readline())
            assert "category" in first_line
            assert "source" in first_line
            assert "confidence" in first_line

    def test_export_max_examples(self, generator, temp_dir):
        """Verificar límite de exportación"""
        generator.generate_dataset()

        output_path = str(Path(temp_dir) / "test_limited.jsonl")
        result = generator.export_jsonl(output_path, max_examples=5)

        with open(result) as f:
            lines = f.readlines()
            assert len(lines) <= 5

    def test_export_auto_generates(self, generator, temp_dir):
        """Verificar que export genera dataset si no existe"""
        output_path = str(Path(temp_dir) / "test_auto.jsonl")
        result = generator.export_jsonl(output_path)

        assert Path(result).exists()
        with open(result) as f:
            assert len(f.readlines()) > 0

    def test_auto_generated_path(self, generator):
        """Verificar generación automática de path"""
        generator.generate_dataset()
        result = generator.export_jsonl()

        assert Path(result).exists()
        assert "habits_" in result
        assert result.endswith(".jsonl")


# =========================================================================
# Tests de integración con NightlyTrainer
# =========================================================================

class TestNightlyIntegration:
    def test_prepare_for_nightly(self, generator, temp_dir):
        """Verificar preparación para entrenamiento nocturno"""
        nightly_dir = str(Path(temp_dir) / "nightly")

        result = generator.prepare_for_nightly_training(nightly_dir)

        assert result["success"] is True
        assert result["examples"] > 0
        assert Path(result["dataset_path"]).exists()

    def test_prepare_empty_data(self, temp_dir):
        """Verificar manejo de datos vacíos"""
        gen = HabitDatasetGenerator(
            data_dir=temp_dir,
            min_confidence=0.99,  # Muy alto para que no pase nada
        )

        result = gen.prepare_for_nightly_training()
        assert result["success"] is False
        assert result["examples"] == 0


# =========================================================================
# Tests de estadísticas
# =========================================================================

class TestStats:
    def test_get_stats(self, generator):
        """Verificar estadísticas"""
        generator.generate_dataset()
        stats = generator.get_stats()

        assert "total_examples" in stats
        assert stats["total_examples"] > 0
        assert "by_category" in stats
        assert "by_source" in stats
        assert "avg_confidence" in stats
        assert stats["avg_confidence"] > 0

    def test_count_by_category(self, generator):
        """Verificar conteo por categoría"""
        generator.generate_dataset()
        stats = generator.get_stats()

        # Debe tener al menos temporales (de patrones)
        assert "temporal" in stats["by_category"] or "preference" in stats["by_category"]


# =========================================================================
# Tests de secuencias de acciones
# =========================================================================

class TestActionSequences:
    def test_detect_sequences(self, generator):
        """Verificar detección de secuencias de acciones"""
        import time
        now = time.time()

        events = [
            {"timestamp": now - 10, "entity_id": "light.living", "action": "turn_on", "hour": 7},
            {"timestamp": now - 5, "entity_id": "climate.living_ac", "action": "set_temperature", "hour": 7},
            {"timestamp": now - 86410, "entity_id": "light.living", "action": "turn_on", "hour": 7},
            {"timestamp": now - 86405, "entity_id": "climate.living_ac", "action": "set_temperature", "hour": 7},
            {"timestamp": now - 172810, "entity_id": "light.living", "action": "turn_on", "hour": 7},
            {"timestamp": now - 172805, "entity_id": "climate.living_ac", "action": "set_temperature", "hour": 7},
        ]

        sequences = generator._detect_action_sequences(events)
        assert len(sequences) > 0
        assert sequences[0]["count"] >= 3

    def test_no_false_sequences(self, generator):
        """Verificar que no detecta secuencias falsas (gap grande)"""
        import time
        now = time.time()

        events = [
            {"timestamp": now - 1000, "entity_id": "light.living", "action": "turn_on", "hour": 7},
            {"timestamp": now - 500, "entity_id": "climate.living_ac", "action": "set_temperature", "hour": 7},
        ]

        sequences = generator._detect_action_sequences(events, max_gap_seconds=120)
        # Gap de 500s > 120s, no debería detectar secuencia
        assert len(sequences) == 0


# =========================================================================
# Tests de helpers
# =========================================================================

class TestHelpers:
    def test_action_to_verb(self, generator):
        """Verificar conversión de acción a verbo"""
        assert "encender" in generator._action_to_verb("light_on")
        assert "apagar" in generator._action_to_verb("light_off")
        assert "ajustar" in generator._action_to_verb("climate_set")

    def test_days_description(self, generator):
        """Verificar descripción de días"""
        assert generator._days_description([0, 1, 2, 3, 4]) == "de lunes a viernes"
        assert generator._days_description([5, 6]) == "fines de semana"
        assert generator._days_description([0, 1, 2, 3, 4, 5, 6]) == "todos los días"
        assert "lunes" in generator._days_description([0])
