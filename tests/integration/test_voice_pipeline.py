"""
Integration tests for Voice Pipeline.

These tests verify the pipeline's end-to-end behavior using mocks
for external dependencies. After the RequestRouter extraction,
process_command delegates to RequestRouter.
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
import numpy as np


class TestVoicePipelineIntegration:
    """Integration tests for the voice pipeline"""

    @pytest.fixture
    def pipeline(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm):
        """Create a pipeline with mocked dependencies and a RequestRouter"""
        from src.pipeline.voice_pipeline import VoicePipeline
        from src.pipeline.request_router import RequestRouter

        vp = VoicePipeline(
            stt=mock_stt,
            tts=mock_tts,
            chroma_sync=mock_chroma,
            routine_manager=mock_routine_manager,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            fast_router=None,
            latency_target_ms=300,
            orchestrator_enabled=False,  # Use legacy path for integration tests
        )

        # Create RequestRouter and wire it into the pipeline
        router = RequestRouter(
            command_processor=vp.command_processor,
            response_handler=vp.response_handler,
            audio_manager=vp.audio_manager,
            orchestrator_enabled=False,
            chroma_sync=mock_chroma,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            routine_manager=mock_routine_manager,
            vector_search_threshold=vp.vector_search_threshold,
            latency_target_ms=vp.latency_target_ms,
        )
        vp.request_router = router
        return vp

    @pytest.mark.asyncio
    async def test_process_domotics_command(self, pipeline, sample_audio):
        """Test processing a domotics command"""
        # Configure mock to return a light command via command_processor
        pipeline.command_processor.process_command = AsyncMock(return_value={
            "text": "prende la luz del living",
            "user": None,
            "emotion": None,
            "timings": {"stt": 50.0}
        })

        # Override chroma search to return entity_id matching mock HA client
        pipeline.request_router.chroma.search_command.return_value = {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
            "description": "Prendiendo luz del living",
            "similarity": 0.95,
        }

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == "prende la luz del living"
        assert result["intent"] == "domotics"
        assert result["success"] is True
        assert "timings" in result

    @pytest.mark.asyncio
    async def test_process_sync_command(self, pipeline, sample_audio):
        """Test processing a sync command"""
        pipeline.command_processor.process_command = AsyncMock(return_value={
            "text": "sincroniza los comandos",
            "user": None,
            "emotion": None,
            "timings": {"stt": 40.0}
        })

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "sync"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_routine_command(self, pipeline, sample_audio):
        """Test processing a routine command"""
        pipeline.command_processor.process_command = AsyncMock(return_value={
            "text": "crea una rutina para las 7am",
            "user": None,
            "emotion": None,
            "timings": {"stt": 45.0}
        })
        pipeline.request_router.routines.handle = AsyncMock(return_value={
            "handled": True,
            "response": "Rutina creada",
            "success": True
        })

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "routine"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_empty_transcription(self, pipeline, sample_audio):
        """Test handling empty transcription"""
        pipeline.command_processor.process_command = AsyncMock(return_value={
            "text": "",
            "user": None,
            "emotion": None,
            "timings": {"stt": 30.0}
        })

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == ""
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_latency_tracking(self, pipeline, sample_audio):
        """Test that latency is tracked"""
        pipeline.command_processor.process_command = AsyncMock(return_value={
            "text": "prende la luz",
            "user": None,
            "emotion": None,
            "timings": {"stt": 50.0}
        })

        result = await pipeline.process_command(sample_audio)

        assert "latency_ms" in result
        assert result["latency_ms"] > 0
        assert "timings" in result
        assert "stt" in result["timings"]

    @pytest.mark.asyncio
    async def test_no_router_returns_error(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm, sample_audio):
        """Test that pipeline without request_router returns error"""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(
            stt=mock_stt,
            tts=mock_tts,
            chroma_sync=mock_chroma,
            routine_manager=mock_routine_manager,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            fast_router=None,
        )
        # No request_router set
        result = await vp.process_command(sample_audio)

        assert result["success"] is False
        assert "error" in result


class TestVoicePipelineCache:
    """Test caching functionality (now in RequestRouter)"""

    @pytest.fixture
    def router(self):
        """Create a RequestRouter for cache testing"""
        from src.pipeline.request_router import RequestRouter

        return RequestRouter(
            command_processor=MagicMock(),
            response_handler=MagicMock(),
            audio_manager=MagicMock(),
            routine_manager=MagicMock(),
            cache_max_size=100,
        )

    def test_cache_size_limit(self, router):
        """Test that cache respects size limit"""
        for i in range(router._cache_max_size + 10):
            router._add_to_cache(f"key_{i}", {"value": i})

        assert len(router._query_cache) <= router._cache_max_size


class TestSyncCommandDetection:
    """Test sync command detection (now in RequestRouter)"""

    @pytest.fixture
    def router(self):
        from src.pipeline.request_router import RequestRouter

        return RequestRouter(
            command_processor=MagicMock(),
            response_handler=MagicMock(),
            audio_manager=MagicMock(),
            routine_manager=MagicMock(),
        )

    @pytest.mark.parametrize("command", [
        "sincroniza los comandos",
        "actualiza la base de datos",
        "refresca todo",
        "sync",
        "aprende los comandos"
    ])
    def test_sync_command_detection(self, router, command):
        """Test various sync command phrases"""
        assert router._is_sync_command(command) is True

    @pytest.mark.parametrize("command", [
        "prende la luz",
        "que hora es",
        "hola",
        "crea una rutina"
    ])
    def test_non_sync_command_detection(self, router, command):
        """Test that non-sync commands are not detected as sync"""
        assert router._is_sync_command(command) is False
