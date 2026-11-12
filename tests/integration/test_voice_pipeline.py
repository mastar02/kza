"""
Integration tests for Voice Pipeline.

These tests verify the pipeline's end-to-end behavior using mocks
for external dependencies. After the slim-pipeline refactor,
VoicePipeline receives pre-built components via DI.
"""

import sys
from unittest.mock import MagicMock, AsyncMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
from tests.mocks.home_control import selected_actions

from tests.mocks.responses import response_handler_mock
from src.pipeline.command_processor import ProcessedCommand
from src.core.task_outcome import PersistenceState, TaskStatus
from src.vectordb.chroma_sync import CommandSyncError


class TestVoicePipelineIntegration:
    """Integration tests for the voice pipeline"""

    @pytest.fixture
    def pipeline(self, mock_stt, mock_tts, mock_chroma, mock_routine_manager, mock_ha_client, mock_llm):
        """Create a slim pipeline with pre-built mocked components."""
        from src.pipeline.voice_pipeline import VoicePipeline
        from src.pipeline.command_processor import CommandProcessor
        from src.pipeline.response_handler import ResponseHandler
        from src.pipeline.audio_manager import AudioManager
        from src.pipeline.request_router import RequestRouter

        # Build components externally (as main.py will do)
        command_processor = CommandProcessor(
            stt=mock_stt,
            sample_rate=16000,
        )

        response_handler = ResponseHandler(
            tts=mock_tts,
        )

        audio_manager = AudioManager(
            sample_rate=16000,
        )

        router = RequestRouter(
            command_processor=command_processor,
            response_handler=response_handler,
            audio_manager=audio_manager,
            orchestrator_enabled=False,
            chroma_sync=mock_chroma,
            ha_client=mock_ha_client,
            home_actions=selected_actions(mock_ha_client, ("light.living_room",)),
            llm_reasoner=mock_llm,
            routine_manager=mock_routine_manager,
            vector_search_threshold=0.65,
            latency_target_ms=300,
        )

        audio_loop = MagicMock()
        audio_loop.start = AsyncMock()
        audio_loop.run = AsyncMock()
        audio_loop.stop = AsyncMock()
        audio_loop.on_command = MagicMock()

        feature_manager = MagicMock()
        feature_manager.start = AsyncMock()
        feature_manager.stop = AsyncMock()

        vp = VoicePipeline(
            audio_loop=audio_loop,
            command_processor=command_processor,
            request_router=router,
            response_handler=response_handler,
            feature_manager=feature_manager,
            chroma_sync=mock_chroma,
        )
        return vp

    @pytest.mark.asyncio
    async def test_process_domotics_command(self, pipeline, sample_audio):
        """Test processing a domotics command"""
        # Configure mock to return a light command via command_processor
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="nexa prende la luz del living",
            timings={"stt": 50.0},
            success=True,
        ))

        # Override chroma search to return entity_id matching mock HA client
        pipeline.request_router.chroma.search_command.return_value = {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
            "description": "Prendiendo luz del living",
            "similarity": 0.95,
        }

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == "nexa prende la luz del living"
        assert result["intent"] == "domotics"
        assert result["success"] is True
        assert "timings" in result

    @pytest.mark.asyncio
    async def test_process_sync_command(self, pipeline, sample_audio):
        """An authorized sync is awaited and its count reaches the reply."""
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="nexa sincroniza los comandos",
            timings={"stt": 40.0},
            success=True,
        ))
        # Voice turns bind no account today; authorize explicitly to reach the sync.
        pipeline.request_router._may_sync_commands = lambda user: True

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "sync"
        assert result["task_completed"] is True
        assert result["response_completed"] is False
        assert result["success"] is False
        # U3b-0d2: the coroutine is awaited, and its count reaches the reply.
        pipeline.request_router.chroma.sync_commands.assert_awaited_once()
        assert result["response"] == "Listo, actualicé 100 comandos"

    @pytest.mark.asyncio
    async def test_unidentified_speaker_cannot_sync(self, pipeline, sample_audio):
        """S1: without an identified ADULT the index is never touched."""
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="nexa sincroniza los comandos", timings={"stt": 40.0}, success=True,
        ))
        pipeline.request_router._run_command_sync = AsyncMock()

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "sync" and result["task_completed"] is False
        assert result["response"] == "Solo un adulto identificado puede sincronizar los comandos."
        pipeline.request_router.chroma.sync_commands.assert_not_awaited()
        pipeline.request_router._run_command_sync.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_casual_update_phrase_does_not_sync(self, pipeline, sample_audio):
        """S1: 'actualiza la lista de compras' is not a sync request."""
        router = pipeline.request_router
        router._may_sync_commands = lambda user: True
        assert router._is_sync_command("nexa actualiza la lista de compras") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error,status,persistence,response", [
        (CommandSyncError("chroma_sync_failed"), TaskStatus.FAILED, PersistenceState.NOT_WRITTEN,
         "No pude sincronizar los comandos; los anteriores siguen activos."),
        (CommandSyncError("chroma_sync_reasoner_unavailable"), TaskStatus.FAILED,
         PersistenceState.NOT_WRITTEN, "No pude sincronizar los comandos; los anteriores siguen activos."),
        (CommandSyncError("chroma_sync_already_running"), TaskStatus.FAILED,
         PersistenceState.NOT_WRITTEN, "Ya estoy sincronizando los comandos."),
        (CommandSyncError("chroma_sync_aborted"), TaskStatus.CANCELLED, PersistenceState.NOT_WRITTEN,
         "Cancelé la sincronización; los comandos anteriores siguen activos."),
        (CommandSyncError("chroma_sync_switch_uncertain", index_kept=False), TaskStatus.FAILED,
         PersistenceState.UNKNOWN, "No pude sincronizar los comandos"),
        (RuntimeError("unexpected"), TaskStatus.FAILED, PersistenceState.UNKNOWN,
         "No pude sincronizar los comandos"),
    ])
    async def test_failed_sync_command_is_reported_not_raised(
            self, pipeline, sample_audio, error, status, persistence, response):
        """S3: the reported persistence says whether the previous index was kept."""
        router = pipeline.request_router
        router._may_sync_commands = lambda user: True
        router.chroma.sync_commands = AsyncMock(side_effect=error)
        outcome = {}
        original = router._run_command_sync

        async def capture(result):
            await original(result)
            outcome.update(result)
        router._run_command_sync = capture
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="nexa sincroniza los comandos", timings={"stt": 40.0}, success=True,
        ))

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "sync"
        assert result["task_completed"] is False
        assert result["response"] == response
        assert outcome["_task_outcome"].status is status
        assert outcome["_task_outcome"].persistence is persistence

    @pytest.mark.asyncio
    async def test_process_routine_command(self, pipeline, sample_audio):
        """Test processing a routine command"""
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="nexa crea una rutina para las 7am",
            timings={"stt": 45.0},
            success=True,
        ))
        pipeline.request_router.routines.handle = AsyncMock(return_value={
            "handled": True,
            "response": "Rutina creada",
            "success": True
        })

        result = await pipeline.process_command(sample_audio)

        assert result["intent"] == "routine"
        assert result["handled"] is True
        assert result["task_completed"] is False
        assert result["outcome"]["status"] == "unknown"
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_empty_transcription(self, pipeline, sample_audio):
        """Test handling empty transcription"""
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="",
            timings={"stt": 30.0},
            success=False,
        ))

        result = await pipeline.process_command(sample_audio)

        assert result["text"] == ""
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_latency_tracking(self, pipeline, sample_audio):
        """Test that latency is tracked"""
        pipeline.command_processor.process_command = AsyncMock(return_value=ProcessedCommand(
            text="prende la luz",
            timings={"stt": 50.0},
            success=True,
        ))

        result = await pipeline.process_command(sample_audio)

        assert "latency_ms" in result
        assert result["latency_ms"] > 0
        assert "timings" in result
        assert "stt" in result["timings"]

    @pytest.mark.asyncio
    async def test_no_router_returns_error(self, sample_audio):
        """Test that pipeline without request_router returns error"""
        from src.pipeline.voice_pipeline import VoicePipeline

        vp = VoicePipeline(
            audio_loop=MagicMock(),
            command_processor=MagicMock(),
            request_router=None,
            response_handler=response_handler_mock(),
            feature_manager=MagicMock(),
        )
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
            response_handler=response_handler_mock(),
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
            response_handler=response_handler_mock(),
            audio_manager=MagicMock(),
            routine_manager=MagicMock(),
        )

    @pytest.mark.parametrize("command", [
        "sincroniza los comandos",
        "sincronizá los comandos",
        "actualizá los comandos",
        "nexa actualizar los comandos",
    ])
    def test_sync_command_detection(self, router, command):
        """Test the exact sync phrases"""
        assert router._is_sync_command(command) is True

    @pytest.mark.parametrize("command", [
        "actualiza la base de datos", "refresca todo", "sync", "aprende los comandos",
        "actualiza la lista de compras",
    ])
    def test_loose_phrases_no_longer_sync(self, router, command):
        """S1: substrings that used to trigger a rebuild do not"""
        assert router._is_sync_command(command) is False

    @pytest.mark.parametrize("command", [
        "prende la luz",
        "que hora es",
        "hola",
        "crea una rutina"
    ])
    def test_non_sync_command_detection(self, router, command):
        """Test that non-sync commands are not detected as sync"""
        assert router._is_sync_command(command) is False
