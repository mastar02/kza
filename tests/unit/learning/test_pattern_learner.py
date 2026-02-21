"""
Tests para PatternLearner - Detección de patrones de comportamiento
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, time as dtime
from unittest.mock import MagicMock

from src.learning.pattern_learner import (
    PatternLearner,
    ActionRecord,
    DetectedPattern,
    RoutineSuggestion,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para datos del learner"""
    return str(tmp_path / "patterns")


@pytest.fixture
def learner(temp_dir):
    """PatternLearner con configuración de test"""
    return PatternLearner(
        data_dir=temp_dir,
        min_occurrences=3,
        max_time_variance=30,
    )


def _add_recurring_action(learner, action_type, entity_id, hour, minute,
                          days, user_id="mastar", data=None):
    """Helper para agregar acciones recurrentes"""
    for day in days:
        for _ in range(2):  # 2 ocurrencias por día
            now = datetime.now().replace(
                hour=hour, minute=minute
            )
            record = ActionRecord(
                action_type=action_type,
                entity_id=entity_id,
                timestamp=now,
                user_id=user_id,
                day_of_week=day,
                hour=hour,
                minute=minute,
                data=data or {},
                trigger="voice",
            )
            learner._action_history.append(record)


# =========================================================================
# Tests de registro de acciones
# =========================================================================

class TestActionRecording:
    def test_record_action(self, learner):
        """Verificar registro de acciones"""
        learner.record_action(
            action_type="light_on",
            entity_id="light.living",
            user_id="mastar",
            trigger="voice",
        )
        assert len(learner._action_history) == 1
        assert learner._action_history[0].action_type == "light_on"
        assert learner._action_history[0].entity_id == "light.living"

    def test_record_with_data(self, learner):
        """Verificar registro con datos adicionales"""
        learner.record_action(
            action_type="climate_set",
            entity_id="climate.escritorio_ac",
            user_id="mastar",
            data={"temperature": 22},
        )
        assert learner._action_history[0].data["temperature"] == 22

    def test_auto_save(self, learner, temp_dir):
        """Verificar auto-guardado periódico"""
        for i in range(10):
            learner.record_action(
                action_type="light_on",
                entity_id=f"light.test_{i}",
            )
        
        history_file = Path(temp_dir) / "action_history.json"
        assert history_file.exists()


# =========================================================================
# Tests de detección de patrones
# =========================================================================

class TestPatternDetection:
    @pytest.mark.asyncio
    async def test_detect_simple_pattern(self, learner):
        """Detectar patrón simple: misma acción a misma hora"""
        _add_recurring_action(
            learner,
            action_type="light_on",
            entity_id="light.living",
            hour=7,
            minute=0,
            days=[0, 1, 2, 3, 4],  # L-V
        )

        patterns = await learner.analyze_patterns()
        assert len(patterns) > 0
        
        pattern = patterns[0]
        assert pattern.action_type == "light_on"
        assert pattern.entity_id == "light.living"
        assert pattern.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_no_pattern_insufficient_data(self, learner):
        """No detectar patrón con datos insuficientes"""
        learner.record_action("light_on", "light.living", "mastar")
        
        patterns = await learner.analyze_patterns()
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_no_pattern_high_variance(self, learner):
        """No detectar patrón con alta varianza temporal"""
        # Acciones a horas muy diferentes
        for hour in [6, 10, 15, 20, 23]:
            record = ActionRecord(
                action_type="light_on",
                entity_id="light.living",
                timestamp=datetime.now().replace(hour=hour),
                hour=hour,
                minute=0,
                day_of_week=0,
            )
            learner._action_history.append(record)

        patterns = await learner.analyze_patterns()
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_pattern_with_data(self, learner):
        """Detectar patrón con datos típicos"""
        _add_recurring_action(
            learner,
            action_type="climate_set",
            entity_id="climate.escritorio_ac",
            hour=9,
            minute=0,
            days=[0, 1, 2, 3, 4],
            data={"temperature": 22},
        )

        patterns = await learner.analyze_patterns()
        assert len(patterns) > 0
        
        pattern = patterns[0]
        assert "temperature" in pattern.typical_data
        assert pattern.typical_data["temperature"] == 22

    @pytest.mark.asyncio
    async def test_dismissed_pattern_not_redetected(self, learner):
        """Patrones rechazados no se vuelven a detectar"""
        _add_recurring_action(
            learner, "light_on", "light.living", 7, 0, [0, 1, 2, 3, 4]
        )

        # Primera detección
        patterns = await learner.analyze_patterns()
        assert len(patterns) > 0

        # Rechazar el patrón
        learner._dismissed_patterns.add(patterns[0].pattern_id)

        # Limpiar y re-detectar
        learner._patterns.clear()
        patterns = await learner.analyze_patterns()
        assert len(patterns) == 0


# =========================================================================
# Tests de sugerencias
# =========================================================================

class TestSuggestions:
    @pytest.mark.asyncio
    async def test_generate_suggestion(self, learner):
        """Verificar generación de sugerencia"""
        _add_recurring_action(
            learner, "light_off", "light.living", 23, 0, [0, 1, 2, 3, 4]
        )

        patterns = await learner.analyze_patterns()
        assert len(patterns) > 0

        suggestion = learner.generate_suggestion(patterns[0])
        assert suggestion is not None
        assert suggestion.routine_name is not None
        assert len(suggestion.actions) > 0
        assert suggestion.trigger["type"] == "time"

    @pytest.mark.asyncio
    async def test_suggestion_text(self, learner):
        """Verificar texto de sugerencia para el usuario"""
        _add_recurring_action(
            learner, "light_off", "light.living", 23, 0, [0, 1, 2, 3, 4]
        )

        patterns = await learner.analyze_patterns()
        suggestion = learner.generate_suggestion(patterns[0])

        text = learner.get_suggestion_text(suggestion)
        assert "He notado" in text
        assert "23:00" in text or "living" in text.lower()

    @pytest.mark.asyncio
    async def test_accept_suggestion(self, learner):
        """Verificar aceptación de sugerencia"""
        _add_recurring_action(
            learner, "light_off", "light.living", 23, 0, [0, 1, 2, 3, 4]
        )

        patterns = await learner.analyze_patterns()
        suggestion = learner.generate_suggestion(patterns[0])

        routine = learner.accept_suggestion(suggestion.suggestion_id)
        assert routine is not None
        assert "name" in routine
        assert "triggers" in routine
        assert "actions" in routine
        assert routine["created_by"] == "pattern_learner"

    @pytest.mark.asyncio
    async def test_dismiss_suggestion(self, learner):
        """Verificar rechazo de sugerencia"""
        _add_recurring_action(
            learner, "light_off", "light.living", 23, 0, [0, 1, 2, 3, 4]
        )

        patterns = await learner.analyze_patterns()
        suggestion = learner.generate_suggestion(patterns[0])

        learner.dismiss_suggestion(suggestion.suggestion_id)
        
        assert suggestion.suggestion_id not in learner._pending_suggestions
        assert patterns[0].pattern_id in learner._dismissed_patterns

    def test_get_pending_suggestions(self, learner):
        """Verificar lista de sugerencias pendientes"""
        assert len(learner.get_pending_suggestions()) == 0


# =========================================================================
# Tests de persistencia
# =========================================================================

class TestPersistence:
    def test_save_and_load(self, learner, temp_dir):
        """Verificar guardado y carga de datos"""
        for i in range(10):
            learner.record_action(
                action_type="light_on",
                entity_id="light.living",
                user_id="mastar",
            )
        learner._save_data()

        learner2 = PatternLearner(data_dir=temp_dir, min_occurrences=3)
        assert len(learner2._action_history) == 10

    def test_save_dismissed(self, learner, temp_dir):
        """Verificar persistencia de patrones rechazados"""
        learner._dismissed_patterns.add("test_pattern_1")
        learner._save_data()

        learner2 = PatternLearner(data_dir=temp_dir)
        assert "test_pattern_1" in learner2._dismissed_patterns


# =========================================================================
# Tests de callbacks
# =========================================================================

class TestCallbacks:
    @pytest.mark.asyncio
    async def test_pattern_detected_callback(self, learner):
        """Verificar callback de patrón detectado"""
        detected = []
        learner.on_pattern_detected(lambda p: detected.append(p))

        _add_recurring_action(
            learner, "light_on", "light.living", 7, 0, [0, 1, 2, 3, 4]
        )

        await learner.analyze_patterns()
        assert len(detected) > 0

    @pytest.mark.asyncio
    async def test_suggestion_ready_callback(self, learner):
        """Verificar callback de sugerencia lista"""
        suggestions = []
        learner.on_suggestion_ready(lambda s: suggestions.append(s))

        # El callback se llama desde run_analysis_loop, no desde analyze_patterns
        # Solo verificamos que se registra
        assert learner._on_suggestion_ready is not None


# =========================================================================
# Tests de estado
# =========================================================================

class TestStatus:
    def test_get_status(self, learner):
        """Verificar reporte de estado"""
        status = learner.get_status()
        assert "total_actions_recorded" in status
        assert "patterns_detected" in status
        assert "pending_suggestions" in status

    @pytest.mark.asyncio
    async def test_get_patterns(self, learner):
        """Verificar listado de patrones"""
        _add_recurring_action(
            learner, "light_on", "light.living", 7, 0, [0, 1, 2, 3, 4]
        )
        await learner.analyze_patterns()

        patterns = learner.get_patterns()
        assert len(patterns) > 0
        assert "pattern_id" in patterns[0]
        assert "confidence" in patterns[0]


# =========================================================================
# Tests de helpers
# =========================================================================

class TestHelpers:
    def test_action_names(self, learner):
        """Verificar nombres de acciones"""
        assert learner._get_action_name("light_on") == "Encender"
        assert learner._get_action_name("light_off") == "Apagar"
        assert learner._get_action_name("climate_set") == "Ajustar clima"

    def test_days_description(self, learner):
        """Verificar descripción de días"""
        assert learner._get_days_description([0, 1, 2, 3, 4]) == "de lunes a viernes"
        assert learner._get_days_description([5, 6]) == "los fines de semana"
        assert "todos los días" in learner._get_days_description(list(range(7)))

    def test_days_to_names(self, learner):
        """Verificar conversión de días a nombres"""
        names = learner._days_to_names([0, 4, 6])
        assert names == ["mon", "fri", "sun"]
