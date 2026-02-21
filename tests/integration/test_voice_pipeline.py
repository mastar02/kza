"""
Integration tests for Voice Pipeline.

These tests verify the pipeline's end-to-end behavior using mocks
for external dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch


class TestVoicePipelineIntegration:
    """Integration tests for the voice pipeline"""

    @pytest.fixture
    def pipeline(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm):
        """Create a pipeline with mocked dependencies"""
        from src.pipeline.voice_pipeline import VoicePipeline

        return VoicePipeline(
            stt=mock_stt,
            tts=mock_tts,
            chroma_sync=mock_chroma,
            routine_manager=mock_routine_manager,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            fast_router=None,
            latency_target_ms=300
        )

    @pytest.mark.asyncio
    async def test_process_domotics_command(self, pipeline, sample_audio):
        """Test processing a domotics command"""
        # Configure mock to return a light command
        pipeline.stt.transcribe.return_value = ("prende la luz del living", 50.0)

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == "prende la luz del living"
        assert result["intent"] == "domotics"
        assert result["success"] is True
        assert "timings" in result

    @pytest.mark.asyncio
    async def test_process_sync_command(self, pipeline, sample_audio):
        """Test processing a sync command"""
        pipeline.stt.transcribe.return_value = ("sincroniza los comandos", 40.0)

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "sync"
        assert result["success"] is True
        pipeline.chroma.sync_commands.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_routine_command(self, pipeline, sample_audio):
        """Test processing a routine command"""
        pipeline.stt.transcribe.return_value = ("crea una rutina para las 7am", 45.0)
        pipeline.routines.handle.return_value = {
            "handled": True,
            "response": "Rutina creada",
            "success": True
        }

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "routine"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_conversation(self, pipeline, sample_audio):
        """Test processing a conversational query"""
        pipeline.stt.transcribe.return_value = ("qué hora es", 50.0)
        pipeline.chroma.search_command.return_value = None  # No match

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "conversation"
        pipeline.llm.generate.assert_called()

    @pytest.mark.asyncio
    async def test_empty_transcription(self, pipeline, sample_audio):
        """Test handling empty transcription"""
        pipeline.stt.transcribe.return_value = ("   ", 30.0)

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == ""
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_latency_tracking(self, pipeline, sample_audio):
        """Test that latency is tracked"""
        pipeline.stt.transcribe.return_value = ("prende la luz", 50.0)

        result = await pipeline.process_command(sample_audio)

        assert "latency_ms" in result
        assert result["latency_ms"] > 0
        assert "timings" in result
        assert "stt" in result["timings"]

    @pytest.mark.asyncio
    async def test_tts_called_on_response(self, pipeline, sample_audio):
        """Test that TTS is called with response"""
        pipeline.stt.transcribe.return_value = ("prende la luz", 50.0)

        await pipeline.process_command(sample_audio)

        pipeline.tts.speak.assert_called()


class TestVoicePipelineCache:
    """Test caching functionality"""

    @pytest.fixture
    def pipeline(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm):
        from src.pipeline.voice_pipeline import VoicePipeline

        return VoicePipeline(
            stt=mock_stt,
            tts=mock_tts,
            chroma_sync=mock_chroma,
            routine_manager=mock_routine_manager,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            fast_router=None
        )

    @pytest.mark.asyncio
    async def test_cache_hit(self, pipeline, sample_audio):
        """Test that cache is used for repeated queries"""
        pipeline.stt.transcribe.return_value = ("prende la luz", 50.0)

        # First call
        await pipeline.process_command(sample_audio)
        first_search_count = pipeline.chroma.search_command.call_count

        # Second call with same text - should use cache
        await pipeline.process_command(sample_audio)
        second_search_count = pipeline.chroma.search_command.call_count

        # Search should only be called once (second used cache)
        assert second_search_count == first_search_count

    def test_cache_size_limit(self, pipeline):
        """Test that cache respects size limit"""
        # Fill cache beyond limit
        for i in range(pipeline._cache_max_size + 10):
            pipeline._add_to_cache(f"key_{i}", {"value": i})

        assert len(pipeline._query_cache) <= pipeline._cache_max_size


class TestVoicePipelineSyncCommands:
    """Test sync command detection"""

    @pytest.fixture
    def pipeline(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm):
        from src.pipeline.voice_pipeline import VoicePipeline

        return VoicePipeline(
            stt=mock_stt,
            tts=mock_tts,
            chroma_sync=mock_chroma,
            routine_manager=mock_routine_manager,
            ha_client=mock_ha_client,
            llm_reasoner=mock_llm,
            fast_router=None
        )

    @pytest.mark.parametrize("command", [
        "sincroniza los comandos",
        "actualiza la base de datos",
        "refresca todo",
        "sync",
        "aprende los comandos"
    ])
    def test_sync_command_detection(self, pipeline, command):
        """Test various sync command phrases"""
        assert pipeline._is_sync_command(command) is True

    @pytest.mark.parametrize("command", [
        "prende la luz",
        "qué hora es",
        "hola",
        "crea una rutina"
    ])
    def test_non_sync_command_detection(self, pipeline, command):
        """Test that non-sync commands are not detected as sync"""
        assert pipeline._is_sync_command(command) is False
