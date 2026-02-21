"""
Tests for RequestRouter - command routing logic extracted from VoicePipeline.

Tests ensure that:
1. Initialization stores all dependencies correctly
2. Orchestrated path delegates to MultiUserOrchestrator
3. Legacy path processes commands through the full pipeline
4. Empty transcription returns early
5. Feedback detection works for positive/negative phrases
6. Sync command detection works
7. Enrollment command detection works
8. Cache stores and retrieves repeated queries
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
import numpy as np

from src.pipeline.request_router import RequestRouter, PermissionResult


# ============================================================
# Helpers
# ============================================================

def _make_cmd_result(text="enciende la luz", user=None, emotion=None, timings=None):
    """Create a mock CommandProcessor result."""
    return {
        "text": text,
        "user": user,
        "emotion": emotion,
        "timings": timings or {"stt": 50.0}
    }


def _make_mock_user(user_id="user1", name="Test User", perm_name="ADULT"):
    """Create a mock user object."""
    user = MagicMock()
    user.user_id = user_id
    user.name = name
    user.permission_level = MagicMock()
    user.permission_level.name = perm_name
    user.permission_level.value = 3
    return user


def _make_mock_emotion(emotion_str="neutral"):
    """Create a mock emotion object."""
    emotion = MagicMock()
    emotion.emotion = emotion_str
    emotion.response_adjustment = None
    return emotion


def _make_dispatch_result(
    intent="domotics",
    response="Luz encendida",
    success=True,
    action=None,
    path_value="fast",
    timings=None
):
    """Create a mock orchestrator dispatch result."""
    dr = MagicMock()
    dr.intent = intent
    dr.response = response
    dr.success = success
    dr.action = action
    dr.was_queued = False
    dr.queue_position = None
    dr.timings = timings or {"dispatch": 10.0}
    dr.path = MagicMock()
    dr.path.value = path_value
    return dr


def _build_router(**overrides):
    """Build a RequestRouter with sensible mock defaults."""
    defaults = {
        "command_processor": MagicMock(),
        "response_handler": MagicMock(),
        "audio_manager": MagicMock(),
        "orchestrator": None,
        "orchestrator_enabled": False,
        "chroma_sync": MagicMock(),
        "ha_client": MagicMock(),
        "llm_reasoner": MagicMock(),
        "fast_router": None,
        "memory_manager": None,
        "user_manager": None,
        "enrollment": None,
        "conversation_collector": None,
        "command_learner": None,
        "event_logger": None,
        "suggestion_engine": None,
        "latency_monitor": None,
        "features": None,
        "voice_routine_handler": None,
        "routine_manager": MagicMock(),
        "vector_search_threshold": 0.65,
        "latency_target_ms": 300,
        "suggestion_interval": 50,
        "cache_max_size": 100,
    }
    defaults.update(overrides)

    # Make command_processor.process_command async by default
    if not isinstance(defaults["command_processor"].process_command, AsyncMock):
        defaults["command_processor"].process_command = AsyncMock(
            return_value=_make_cmd_result()
        )

    # Make routine_manager.handle async by default
    if defaults["routine_manager"] and not isinstance(
        defaults["routine_manager"].handle, AsyncMock
    ):
        defaults["routine_manager"].handle = AsyncMock(
            return_value={"handled": False}
        )

    # Make ha_client.call_service_ws async by default
    if defaults["ha_client"] and not isinstance(
        defaults["ha_client"].call_service_ws, AsyncMock
    ):
        defaults["ha_client"].call_service_ws = AsyncMock(return_value=True)

    # Make ha_client.get_domotics_entities async
    if defaults["ha_client"] and not isinstance(
        defaults["ha_client"].get_domotics_entities, AsyncMock
    ):
        defaults["ha_client"].get_domotics_entities = AsyncMock(return_value=[])

    return RequestRouter(**defaults)


# ============================================================
# Tests
# ============================================================

class TestRequestRouterInit:
    """Test initialization and dependency storage."""

    def test_request_router_init(self):
        """Verify all dependencies are stored correctly."""
        cp = MagicMock()
        rh = MagicMock()
        am = MagicMock()
        orch = MagicMock()
        chroma = MagicMock()
        ha = MagicMock()
        llm = MagicMock()

        router = RequestRouter(
            command_processor=cp,
            response_handler=rh,
            audio_manager=am,
            orchestrator=orch,
            orchestrator_enabled=True,
            chroma_sync=chroma,
            ha_client=ha,
            llm_reasoner=llm,
            routine_manager=MagicMock(),
        )

        assert router.command_processor is cp
        assert router.response_handler is rh
        assert router.audio_manager is am
        assert router._orchestrator is orch
        assert router.orchestrator_enabled is True
        assert router.chroma is chroma
        assert router.ha is ha
        assert router.llm is llm
        assert router._query_cache == {}
        assert router._command_count == 0
        assert router._pending_suggestion is None
        assert router._last_response is None

    def test_init_with_defaults(self):
        """Verify default config values."""
        router = _build_router()
        assert router.vector_search_threshold == 0.65
        assert router.latency_target_ms == 300
        assert router.suggestion_interval == 50
        assert router._cache_max_size == 100


class TestProcessCommandOrchestrated:
    """Test orchestrated (multi-user) path."""

    @pytest.mark.asyncio
    async def test_process_command_orchestrated_path(self):
        """When orchestrator enabled, delegates to orchestrator."""
        mock_user = _make_mock_user()
        mock_emotion = _make_mock_emotion()
        dispatch = _make_dispatch_result()

        orch = MagicMock()
        orch.process = AsyncMock(return_value=dispatch)

        cp = MagicMock()
        cp.process_command = AsyncMock(return_value=_make_cmd_result(
            text="enciende la luz",
            user=mock_user,
            emotion=mock_emotion
        ))

        am = MagicMock()
        am.detect_source_zone.return_value = "living_room"

        router = _build_router(
            command_processor=cp,
            audio_manager=am,
            orchestrator=orch,
            orchestrator_enabled=True,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await router.process_command(audio)

        assert result["success"] is True
        assert result["text"] == "enciende la luz"
        assert result["intent"] == "domotics"
        assert result["path"] == "fast"
        orch.process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_orchestrated_empty_text_returns_early(self):
        """Empty transcription returns early in orchestrated path."""
        orch = MagicMock()
        orch.process = AsyncMock()

        cp = MagicMock()
        cp.process_command = AsyncMock(return_value=_make_cmd_result(text="   "))

        router = _build_router(
            command_processor=cp,
            orchestrator=orch,
            orchestrator_enabled=True,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await router.process_command(audio)

        assert result["text"] == "   "
        assert result["success"] is False
        orch.process.assert_not_awaited()


class TestProcessCommandLegacy:
    """Test legacy (single-user) path."""

    @pytest.mark.asyncio
    async def test_process_command_legacy_path(self):
        """When orchestrator disabled, uses legacy vector search path."""
        cp = MagicMock()
        cp.process_command = AsyncMock(return_value=_make_cmd_result(
            text="enciende la luz"
        ))

        chroma = MagicMock()
        chroma.search_command.return_value = {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
            "description": "Luz de la sala encendida",
            "similarity": 0.92,
            "data": None,
        }

        ha = MagicMock()
        ha.call_service_ws = AsyncMock(return_value=True)
        ha.get_domotics_entities = AsyncMock(return_value=[])

        router = _build_router(
            command_processor=cp,
            chroma_sync=chroma,
            ha_client=ha,
            orchestrator_enabled=False,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await router.process_command(audio)

        assert result["success"] is True
        assert result["intent"] == "domotics"
        assert result["action"]["entity_id"] == "light.living_room"
        ha.call_service_ws.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_command_empty_text(self):
        """Empty transcription returns early in legacy path."""
        cp = MagicMock()
        cp.process_command = AsyncMock(return_value=_make_cmd_result(text=""))

        router = _build_router(
            command_processor=cp,
            orchestrator_enabled=False,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await router.process_command(audio)

        assert result["text"] == ""
        assert result["success"] is False


class TestFeedbackDetection:
    """Test feedback phrase detection."""

    def test_feedback_good_detected(self):
        """Positive feedback phrase detected."""
        collector = MagicMock()
        router = _build_router(conversation_collector=collector)
        router._last_response = "some response"

        fb = router._check_feedback("buena respuesta")
        assert fb["is_feedback"] is True
        assert "Gracias" in fb["response"]
        collector.mark_last_response.assert_called_once_with("good")

    def test_feedback_bad_detected(self):
        """Negative feedback phrase detected."""
        collector = MagicMock()
        router = _build_router(conversation_collector=collector)
        router._last_response = "some response"

        fb = router._check_feedback("mala respuesta")
        assert fb["is_feedback"] is True
        assert "mejorar" in fb["response"]
        collector.mark_last_response.assert_called_once_with("bad")

    def test_feedback_correction_detected(self):
        """Correction feedback detected."""
        collector = MagicMock()
        router = _build_router(conversation_collector=collector)
        router._last_response = "some response"

        fb = router._check_feedback("debiste decir hola mundo")
        assert fb["is_feedback"] is True
        assert "hola mundo" in fb["response"]
        collector.mark_last_response.assert_called_once_with("corrected", "hola mundo")

    def test_no_feedback_without_last_response(self):
        """No feedback if no previous response exists."""
        collector = MagicMock()
        router = _build_router(conversation_collector=collector)
        # _last_response is None by default

        fb = router._check_feedback("buena respuesta")
        assert fb["is_feedback"] is False

    def test_no_feedback_normal_text(self):
        """Normal text not detected as feedback."""
        collector = MagicMock()
        router = _build_router(conversation_collector=collector)
        router._last_response = "some response"

        fb = router._check_feedback("enciende la luz de la sala")
        assert fb["is_feedback"] is False


class TestSyncCommandDetection:
    """Test sync command detection."""

    @pytest.mark.parametrize("text", [
        "sincroniza",
        "sincronizar comandos",
        "actualiza los comandos",
        "sync",
    ])
    def test_sync_command_detection(self, text):
        """Sync phrases are detected correctly."""
        router = _build_router()
        assert router._is_sync_command(text) is True

    def test_non_sync_command(self):
        """Non-sync text is not detected."""
        router = _build_router()
        assert router._is_sync_command("enciende la luz") is False


class TestEnrollmentCommandDetection:
    """Test enrollment command detection."""

    @pytest.mark.parametrize("text", [
        "agregar persona",
        "nuevo usuario",
        "registrar persona",
        "add user",
    ])
    def test_enrollment_command_detection(self, text):
        """Enrollment phrases are detected correctly."""
        router = _build_router()
        assert router._is_enrollment_command(text) is True

    def test_non_enrollment_command(self):
        """Non-enrollment text is not detected."""
        router = _build_router()
        assert router._is_enrollment_command("enciende la luz") is False


class TestCache:
    """Test query cache behavior."""

    def test_cache_stores_and_retrieves(self):
        """Cache stores a value and retrieves it."""
        router = _build_router(cache_max_size=10)
        router._add_to_cache("enciende luz", {"entity_id": "light.sala"})

        assert "enciende luz" in router._query_cache
        assert router._query_cache["enciende luz"]["entity_id"] == "light.sala"

    def test_cache_evicts_oldest(self):
        """Cache evicts oldest entry when full."""
        router = _build_router(cache_max_size=3)
        router._add_to_cache("cmd1", {"id": 1})
        router._add_to_cache("cmd2", {"id": 2})
        router._add_to_cache("cmd3", {"id": 3})
        # This should evict cmd1
        router._add_to_cache("cmd4", {"id": 4})

        assert "cmd1" not in router._query_cache
        assert "cmd4" in router._query_cache
        assert len(router._query_cache) == 3


class TestPermissionCheck:
    """Test permission checking."""

    def test_no_user_manager_allows_all(self):
        """Without user manager, all actions are allowed."""
        router = _build_router(user_manager=None)
        result = router._check_permission(MagicMock(), "light.sala", "turn_on")
        assert isinstance(result, PermissionResult)
        assert result.allowed is True

    def test_permission_delegates_to_user_manager(self):
        """With user manager, delegates permission check."""
        um = MagicMock()
        check_result = MagicMock()
        check_result.allowed = True
        um.check_entity_permission.return_value = check_result

        router = _build_router(user_manager=um)
        user = MagicMock()
        result = router._check_permission(user, "light.sala", "turn_on")

        assert result.allowed is True
        um.check_entity_permission.assert_called_once_with(user, "light.sala", "turn_on")


class TestSuggestionHandling:
    """Test automation suggestion response handling."""

    def test_accept_suggestion(self):
        """Accept suggestion when user says 'si'."""
        engine = MagicMock()
        engine.respond_to_suggestion.return_value = {"message": "configurada"}

        router = _build_router(suggestion_engine=engine)
        router._pending_suggestion = MagicMock()
        router._pending_suggestion.id = "s1"

        result = router._handle_suggestion_response("si")
        assert result["handled"] is True
        assert "automatizacion creada" in result["response"]
        assert router._pending_suggestion is None

    def test_reject_suggestion(self):
        """Reject suggestion when user says 'no'."""
        engine = MagicMock()
        router = _build_router(suggestion_engine=engine)
        router._pending_suggestion = MagicMock()
        router._pending_suggestion.id = "s1"

        result = router._handle_suggestion_response("no quiero")
        assert result["handled"] is True
        assert "no creare" in result["response"]
        assert router._pending_suggestion is None

    def test_snooze_suggestion(self):
        """Snooze suggestion when user says 'despues'."""
        engine = MagicMock()
        router = _build_router(suggestion_engine=engine)
        router._pending_suggestion = MagicMock()
        router._pending_suggestion.id = "s1"

        result = router._handle_suggestion_response("despues")
        assert result["handled"] is True
        assert "manana" in result["response"]

    def test_no_pending_suggestion(self):
        """No handling if no suggestion is pending."""
        router = _build_router()
        result = router._handle_suggestion_response("si")
        assert result["handled"] is False


class TestLatencyLogging:
    """Test latency logging."""

    def test_log_latency_within_target(self):
        """Latency within target logs OK."""
        monitor = MagicMock()
        router = _build_router(latency_monitor=monitor, latency_target_ms=300)

        result = {
            "latency_ms": 150.0,
            "timings": {"stt": 50.0, "vector_search": 30.0},
            "user": {"name": "Test"},
            "intent": "domotics"
        }
        router._log_latency(result)
        monitor.record.assert_called_once()

    def test_log_latency_without_monitor(self):
        """Latency logging works even without monitor (just logs)."""
        router = _build_router(latency_monitor=None)
        result = {
            "latency_ms": 500.0,
            "timings": {"stt": 200.0},
            "user": None,
            "intent": "conversation"
        }
        # Should not raise
        router._log_latency(result)


class TestBuildPrompt:
    """Test prompt building."""

    @pytest.mark.asyncio
    async def test_build_prompt_basic(self):
        """Basic prompt includes user text and system instructions."""
        ha = MagicMock()
        ha.get_domotics_entities = AsyncMock(return_value=[])

        router = _build_router(ha_client=ha, memory_manager=None)
        prompt = await router._build_prompt("que hora es")

        assert "que hora es" in prompt
        assert "asistente de hogar" in prompt.lower()
        assert "Usuario:" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_memory(self):
        """Prompt includes memory context when available."""
        ha = MagicMock()
        ha.get_domotics_entities = AsyncMock(return_value=[])

        memory = MagicMock()
        memory.format_context_for_prompt.return_value = "User likes warm lights"

        router = _build_router(ha_client=ha, memory_manager=memory)
        prompt = await router._build_prompt("que hora es")

        assert "User likes warm lights" in prompt
