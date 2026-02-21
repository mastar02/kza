"""
Tests for RequestDispatcher - Request routing and classification.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.orchestrator.dispatcher import (
    RequestDispatcher,
    PathType,
    DispatchResult,
)
from src.orchestrator.priority_queue import Priority, PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB sync"""
    chroma = MagicMock()
    chroma.search_command = MagicMock(return_value={
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living",
        "description": "Luz del living encendida",
        "similarity": 0.92
    })
    return chroma


@pytest.fixture
def mock_ha():
    """Mock Home Assistant client"""
    ha = MagicMock()
    ha.call_service = MagicMock(return_value=True)
    return ha


@pytest.fixture
def mock_routine_manager():
    """Mock routine manager"""
    manager = MagicMock()
    manager.handle = AsyncMock(return_value={
        "handled": False,
        "response": "",
        "success": False
    })
    return manager


@pytest.fixture
def mock_router():
    """Mock fast router"""
    router = MagicMock()
    router.generate = MagicMock(return_value=["Respuesta rápida"])
    return router


@pytest.fixture
def dispatcher(mock_chroma, mock_ha, mock_routine_manager, mock_router):
    """Create dispatcher with mocks"""
    return RequestDispatcher(
        chroma_sync=mock_chroma,
        ha_client=mock_ha,
        routine_manager=mock_routine_manager,
        router=mock_router,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
        vector_threshold=0.65
    )


class TestPathType:
    """Tests for PathType enum"""

    def test_path_types_exist(self):
        assert PathType.FAST_DOMOTICS
        assert PathType.FAST_ROUTINE
        assert PathType.FAST_ROUTER
        assert PathType.FAST_MUSIC
        assert PathType.SLOW_MUSIC
        assert PathType.SLOW_LLM


class TestDispatchResult:
    """Tests for DispatchResult dataclass"""

    def test_create_result(self):
        result = DispatchResult(
            path=PathType.FAST_DOMOTICS,
            priority=Priority.HIGH,
            success=True,
            response="Luz encendida"
        )
        assert result.path == PathType.FAST_DOMOTICS
        assert result.success is True

    def test_result_to_dict(self):
        result = DispatchResult(
            path=PathType.FAST_DOMOTICS,
            priority=Priority.HIGH,
            success=True,
            response="Test",
            intent="domotics",
            timings={"vector_search": 15.0}
        )
        data = result.to_dict()
        assert data["path"] == "fast_domotics"
        assert data["priority"] == "HIGH"
        assert data["success"] is True


class TestRequestClassification:
    """Tests for request classification"""

    def test_classify_domotics_prende(self, dispatcher):
        path, priority = dispatcher._classify_request("prende la luz del living")
        assert path == PathType.FAST_DOMOTICS
        assert priority == Priority.HIGH

    def test_classify_domotics_apaga(self, dispatcher):
        path, priority = dispatcher._classify_request("apaga el aire")
        assert path == PathType.FAST_DOMOTICS
        assert priority == Priority.HIGH

    def test_classify_domotics_sube(self, dispatcher):
        path, priority = dispatcher._classify_request("sube las persianas")
        assert path == PathType.FAST_DOMOTICS
        assert priority == Priority.HIGH

    def test_classify_routine(self, dispatcher):
        path, priority = dispatcher._classify_request("crea una rutina para la noche")
        assert path == PathType.FAST_ROUTINE
        assert priority == Priority.MEDIUM

    def test_classify_simple_query(self, dispatcher):
        path, priority = dispatcher._classify_request("que hora es")
        assert path == PathType.FAST_ROUTER
        assert priority == Priority.MEDIUM

    def test_classify_conversation(self, dispatcher):
        path, priority = dispatcher._classify_request("explícame la teoría de la relatividad")
        assert path == PathType.SLOW_LLM
        assert priority == Priority.LOW


class TestMusicClassification:
    """Tests for music-related classification"""

    def test_classify_music_direct_artist(self, dispatcher):
        # Add music dispatcher mock
        dispatcher.music = MagicMock()

        path, priority = dispatcher._classify_request("pon música de bad bunny")
        assert path == PathType.FAST_MUSIC
        assert priority == Priority.HIGH

    def test_classify_music_control(self, dispatcher):
        dispatcher.music = MagicMock()

        path, priority = dispatcher._classify_request("pausa la música")
        assert path == PathType.FAST_MUSIC

    def test_classify_music_context(self, dispatcher):
        dispatcher.music = MagicMock()

        path, priority = dispatcher._classify_request("pon música para cocinar")
        assert path == PathType.SLOW_MUSIC
        assert priority == Priority.MEDIUM

    def test_classify_music_mood(self, dispatcher):
        dispatcher.music = MagicMock()

        path, priority = dispatcher._classify_request("pon algo tranquilo")
        assert path == PathType.SLOW_MUSIC

    def test_no_music_without_dispatcher(self, dispatcher):
        # Without music dispatcher, should fall through
        dispatcher.music = None

        path, priority = dispatcher._classify_request("pon música de taylor swift")
        # Should not be classified as music
        assert path != PathType.FAST_MUSIC
        assert path != PathType.SLOW_MUSIC


class TestDispatchFastPath:
    """Tests for fast path dispatching"""

    @pytest.mark.asyncio
    async def test_dispatch_domotics_success(self, dispatcher, mock_chroma, mock_ha):
        result = await dispatcher.dispatch(
            user_id="user_1",
            text="prende la luz del living",
            zone_id="living"
        )

        assert result.success is True
        assert result.path == PathType.FAST_DOMOTICS
        assert result.intent == "domotics"
        assert "vector_search" in result.timings
        mock_ha.call_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_domotics_not_found(self, dispatcher, mock_chroma):
        mock_chroma.search_command.return_value = None

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="prende algo que no existe",
            zone_id="living"
        )

        # Should fall back to router or slow path
        assert result.path in [PathType.FAST_ROUTER, PathType.SLOW_LLM]

    @pytest.mark.asyncio
    async def test_dispatch_router_simple(self, dispatcher, mock_chroma, mock_router):
        mock_chroma.search_command.return_value = None

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="que hora es",
            zone_id="living"
        )

        assert result.path == PathType.FAST_ROUTER
        assert result.success is True
        mock_router.generate.assert_called()


class TestDispatchSpecialCommands:
    """Tests for special command handling"""

    @pytest.mark.asyncio
    async def test_dispatch_cancel(self, dispatcher):
        # First enqueue something
        dispatcher.queue.enqueue("user_1", "Test request", Priority.LOW)

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="cancela",
            zone_id="living"
        )

        assert result.success is True
        assert "cancelad" in result.response.lower()

    @pytest.mark.asyncio
    async def test_dispatch_sync(self, dispatcher, mock_chroma):
        mock_chroma.sync_commands = AsyncMock(return_value=100)

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="sincroniza los comandos",
            zone_id="living"
        )

        assert result.path == PathType.SYNC

    @pytest.mark.asyncio
    async def test_dispatch_confirmation_yes(self, dispatcher):
        # Set up pending confirmation
        dispatcher.context_manager.get_or_create("user_1", "Test")
        dispatcher.context_manager.set_pending_confirmation("user_1", {
            "action": "create_routine",
            "routine_name": "test"
        })

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="si confirmo",
            zone_id="living"
        )

        assert result.path == PathType.FAST_ROUTINE
        assert result.intent == "confirmation"

    @pytest.mark.asyncio
    async def test_dispatch_confirmation_no(self, dispatcher):
        dispatcher.context_manager.get_or_create("user_1", "Test")
        dispatcher.context_manager.set_pending_confirmation("user_1", {
            "action": "create_routine"
        })

        result = await dispatcher.dispatch(
            user_id="user_1",
            text="no cancela",
            zone_id="living"
        )

        assert result.path == PathType.FAST_ROUTINE
        # Confirmation should be cleared
        assert dispatcher.context_manager.get_pending_confirmation("user_1") is None


class TestDispatchContext:
    """Tests for context management during dispatch"""

    @pytest.mark.asyncio
    async def test_creates_context(self, dispatcher):
        await dispatcher.dispatch(
            user_id="new_user",
            user_name="Juan",
            text="hola",
            zone_id="living"
        )

        ctx = dispatcher.context_manager.get("new_user")
        assert ctx is not None
        assert ctx.user_name == "Juan"
        assert ctx.zone_id == "living"

    @pytest.mark.asyncio
    async def test_updates_zone(self, dispatcher):
        # First request from living
        await dispatcher.dispatch(
            user_id="user_1",
            text="hola",
            zone_id="living"
        )

        # Second request from kitchen
        await dispatcher.dispatch(
            user_id="user_1",
            text="hola",
            zone_id="kitchen"
        )

        ctx = dispatcher.context_manager.get("user_1")
        assert ctx.zone_id == "kitchen"


class TestDispatchStats:
    """Tests for dispatcher statistics"""

    @pytest.mark.asyncio
    async def test_stats_tracking(self, dispatcher):
        # Make some requests
        await dispatcher.dispatch("user_1", "prende la luz", zone_id="living")
        await dispatcher.dispatch("user_2", "que hora es", zone_id="kitchen")

        stats = dispatcher.get_stats()
        assert stats["total_requests"] == 2
        assert stats["fast_path"] >= 2

    @pytest.mark.asyncio
    async def test_stats_by_path(self, dispatcher):
        await dispatcher.dispatch("user_1", "prende la luz", zone_id="living")

        stats = dispatcher.get_stats()
        assert stats["by_path"][PathType.FAST_DOMOTICS] >= 1


class TestDispatchBatch:
    """Tests for batch dispatching"""

    @pytest.mark.asyncio
    async def test_dispatch_batch(self, dispatcher):
        requests = [
            {"user_id": "user_1", "text": "prende la luz", "zone_id": "living"},
            {"user_id": "user_2", "text": "que hora es", "zone_id": "kitchen"},
        ]

        results = await dispatcher.dispatch_batch(requests)

        assert len(results) == 2
        assert all(r.success for r in results)
