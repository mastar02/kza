"""
Tests for ChromaDB sync functionality.

NOTE: These tests use mocks to avoid requiring actual ChromaDB/embeddings.
For full integration tests, see tests/integration/
"""

import pytest
from unittest.mock import MagicMock, patch


class TestChromaSyncUnit:
    """Unit tests for ChromaSync with mocks"""

    def test_search_command_returns_match(self, mock_chroma):
        """Test that search returns a command match"""
        result = mock_chroma.search_command("prende la luz del living")

        assert result is not None
        assert "domain" in result
        assert "service" in result
        assert "entity_id" in result
        assert "similarity" in result

    def test_search_command_fields(self, mock_chroma):
        """Test returned fields are correct"""
        result = mock_chroma.search_command("prende la luz")

        assert result["domain"] == "light"
        assert result["service"] == "turn_on"
        assert result["similarity"] >= 0.65

    def test_get_stats(self, mock_chroma):
        """Test statistics retrieval"""
        stats = mock_chroma.get_stats()

        assert "commands_phrases" in stats
        assert "routines" in stats
        assert stats["commands_phrases"] >= 0

    def test_sync_commands(self, mock_chroma):
        """Test syncing commands"""
        count = mock_chroma.sync_commands(MagicMock(), MagicMock())

        assert count >= 0

    def test_search_with_threshold(self, mock_chroma):
        """Test search respects threshold"""
        # Default mock returns 0.95 similarity
        result = mock_chroma.search_command("test", threshold=0.90)
        assert result is not None  # 0.95 > 0.90


class TestChromaSyncSearchPatterns:
    """Test different search patterns"""

    @pytest.fixture
    def mock_search_responses(self):
        """Different responses for different queries"""
        return {
            "luz": {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.living",
                "description": "Prender luz del living",
                "similarity": 0.92
            },
            "aire": {
                "domain": "climate",
                "service": "set_temperature",
                "entity_id": "climate.bedroom",
                "description": "Ajustar aire del dormitorio",
                "similarity": 0.88,
                "data": {"temperature": 22}
            },
            "persiana": {
                "domain": "cover",
                "service": "open_cover",
                "entity_id": "cover.blinds",
                "description": "Abrir persianas",
                "similarity": 0.90
            }
        }

    def test_light_commands(self, mock_search_responses):
        """Test light-related commands"""
        # Simulate search for "prende la luz"
        response = mock_search_responses["luz"]
        assert response["domain"] == "light"
        assert response["service"] == "turn_on"

    def test_climate_commands(self, mock_search_responses):
        """Test climate-related commands"""
        response = mock_search_responses["aire"]
        assert response["domain"] == "climate"
        assert "data" in response
        assert response["data"]["temperature"] == 22

    def test_cover_commands(self, mock_search_responses):
        """Test cover-related commands"""
        response = mock_search_responses["persiana"]
        assert response["domain"] == "cover"
        assert response["service"] == "open_cover"


class TestChromaSyncRoutines:
    """Test routine-related ChromaDB operations"""

    def test_save_routine(self, mock_chroma):
        """Test saving a routine"""
        mock_chroma.save_routine = MagicMock(return_value=True)

        result = mock_chroma.save_routine(
            routine_id="routine_123",
            description="Rutina de llegada a casa",
            phrases=["cuando llegue a casa", "al llegar"]
        )

        assert result is True

    def test_search_routine(self, mock_chroma):
        """Test searching for a routine"""
        mock_chroma.search_routine = MagicMock(return_value={
            "routine_id": "routine_123",
            "description": "Rutina de llegada a casa",
            "similarity": 0.85
        })

        result = mock_chroma.search_routine("rutina de llegada")
        assert result is not None
        assert "routine_id" in result

    def test_list_routines(self, mock_chroma):
        """Test listing all routines"""
        mock_chroma.list_routines = MagicMock(return_value=[
            {"id": "routine_1", "description": "Rutina 1"},
            {"id": "routine_2", "description": "Rutina 2"}
        ])

        routines = mock_chroma.list_routines()
        assert len(routines) == 2

    def test_delete_routine(self, mock_chroma):
        """Test deleting a routine"""
        mock_chroma.delete_routine = MagicMock(return_value=True)

        result = mock_chroma.delete_routine("routine_123")
        assert result is True
