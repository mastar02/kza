"""
Tests for slim VoicePipeline constraints.

Verifies that VoicePipeline is a thin orchestrator with:
- At most 11 __init__ parameters (including self)
- No passthrough methods — callers access components directly
- Small code footprint (~200 lines)
"""

import sys
import inspect
from unittest.mock import MagicMock, AsyncMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
import numpy as np


class TestSlimPipelineConstraints:
    """Verify VoicePipeline stays thin after refactoring."""

    def test_voice_pipeline_has_few_params(self):
        """VoicePipeline.__init__ must have at most 11 params (self + 10)."""
        from src.pipeline.voice_pipeline import VoicePipeline

        sig = inspect.signature(VoicePipeline.__init__)
        # self + max 10 params (down from 39)
        assert len(sig.parameters) <= 11, (
            f"VoicePipeline has {len(sig.parameters)} params, expected <= 11"
        )

    def test_voice_pipeline_line_count(self):
        """VoicePipeline module should be roughly 200 lines or fewer."""
        from src.pipeline import voice_pipeline

        source = inspect.getsource(voice_pipeline)
        line_count = len(source.splitlines())
        assert line_count <= 250, (
            f"voice_pipeline.py has {line_count} lines, expected <= 250"
        )

    def test_no_internal_component_construction(self):
        """VoicePipeline must NOT construct AudioManager, EchoSuppressor, etc."""
        from src.pipeline import voice_pipeline

        source = inspect.getsource(voice_pipeline)

        # These classes should NOT be constructed inside VoicePipeline
        forbidden_constructions = [
            "AudioManager(",
            "EchoSuppressor(",
            "MultiUserOrchestrator(",
            "FollowUpMode(",
            "AudioEventDetector(",
            "EntityLearner(",
            "PatternLearner(",
            "MorningBriefing(",
        ]
        for cls in forbidden_constructions:
            assert cls not in source, (
                f"VoicePipeline should not construct {cls} — inject it instead"
            )

    def test_no_passthrough_methods(self):
        """VoicePipeline must NOT have passthrough methods."""
        from src.pipeline.voice_pipeline import VoicePipeline

        # These passthrough methods should be removed
        removed_methods = [
            "get_orchestrator_stats",
            "get_active_contexts",
            "clear_user_context",
            "get_queue_size",
            "cancel_user_requests",
            "get_automation_suggestions",
            "get_event_stats",
            "analyze_patterns",
            "who_is_home",
            "is_anyone_home",
            "get_presence_summary",
            "get_zone_occupancy",
            "is_user_home",
            "get_user_location",
            "get_all_routines",
            "get_routine_stats",
            "execute_routine_by_name",
            "save_routines",
            "load_routines",
            "should_process_without_wake_word",
            "start_conversation",
            "end_conversation",
            "notify_user_spoke",
            "notify_kza_responded",
            "speak_with_echo_suppression",
            "get_echo_suppressor_stats",
            "set_echo_suppression_enabled",
            "sync_entities_from_ha",
            "resolve_entity",
            "learn_entity_alias",
            "register_user",
            "get_user_name",
            "get_learned_entities",
            "record_user_action",
            "analyze_user_patterns",
            "accept_routine_suggestion",
            "dismiss_routine_suggestion",
            "get_pending_suggestions",
            "deliver_morning_briefing",
            "should_deliver_briefing",
            "configure_user_briefing",
            "get_new_features_status",
            "configure_ambient_event",
        ]
        for method_name in removed_methods:
            assert not hasattr(VoicePipeline, method_name), (
                f"VoicePipeline still has passthrough method '{method_name}' — remove it"
            )

    def test_required_methods_exist(self):
        """VoicePipeline must keep its core methods."""
        from src.pipeline.voice_pipeline import VoicePipeline

        required = ["__init__", "run", "stop", "process_command", "test_from_text"]
        for method_name in required:
            assert hasattr(VoicePipeline, method_name), (
                f"VoicePipeline is missing required method '{method_name}'"
            )


class TestSlimPipelineCreation:
    """Verify the slim VoicePipeline can be instantiated with the new signature."""

    @pytest.fixture
    def slim_deps(self):
        """Create mock dependencies for slim VoicePipeline."""
        audio_loop = MagicMock()
        audio_loop.start = AsyncMock()
        audio_loop.run = AsyncMock()
        audio_loop.stop = AsyncMock()
        audio_loop.on_command = MagicMock()
        audio_loop.on_post_command = MagicMock()

        command_processor = MagicMock()
        command_processor.load_models = MagicMock()

        request_router = MagicMock()
        request_router.process_command = AsyncMock(return_value={
            "text": "test", "success": True
        })

        response_handler = MagicMock()

        feature_manager = MagicMock()
        feature_manager.start = AsyncMock()
        feature_manager.stop = AsyncMock()

        chroma_sync = MagicMock()
        chroma_sync.initialize = MagicMock()
        chroma_sync.get_stats = MagicMock(return_value={"commands_phrases": 50})
        chroma_sync.search_command = MagicMock(return_value=None)

        memory_manager = MagicMock()
        memory_manager.initialize = MagicMock()

        orchestrator = MagicMock()
        orchestrator.start = AsyncMock()
        orchestrator.stop = AsyncMock()

        return {
            "audio_loop": audio_loop,
            "command_processor": command_processor,
            "request_router": request_router,
            "response_handler": response_handler,
            "feature_manager": feature_manager,
            "chroma_sync": chroma_sync,
            "memory_manager": memory_manager,
            "orchestrator": orchestrator,
        }

    def test_instantiation_with_all_deps(self, slim_deps):
        """VoicePipeline can be created with all dependencies."""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(**slim_deps)

        assert vp.audio_loop is slim_deps["audio_loop"]
        assert vp.command_processor is slim_deps["command_processor"]
        assert vp.request_router is slim_deps["request_router"]
        assert vp.response_handler is slim_deps["response_handler"]
        assert vp.features is slim_deps["feature_manager"]

    def test_instantiation_minimal(self):
        """VoicePipeline can be created with only required deps."""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(
            audio_loop=MagicMock(),
            command_processor=MagicMock(),
            request_router=MagicMock(),
            response_handler=MagicMock(),
            feature_manager=MagicMock(),
        )

        assert vp.chroma is None
        assert vp.memory is None
        assert vp._orchestrator is None

    @pytest.mark.asyncio
    async def test_process_command_delegates_to_router(self, slim_deps):
        """process_command must delegate to request_router."""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(**slim_deps)
        audio = np.zeros(16000, dtype=np.float32)
        result = await vp.process_command(audio)

        slim_deps["request_router"].process_command.assert_called_once_with(audio)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_stop_stops_all_components(self, slim_deps):
        """stop() must stop orchestrator, audio_loop, and features."""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(**slim_deps)
        vp._running = True

        await vp.stop()

        slim_deps["orchestrator"].stop.assert_called_once()
        slim_deps["audio_loop"].stop.assert_called_once()
        slim_deps["feature_manager"].stop.assert_called_once()
        assert vp._running is False
