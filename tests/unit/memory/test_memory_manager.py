"""
Tests for Memory Manager module.
Tests short-term memory, long-term memory, and preferences storage.
"""

import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.memory.memory_manager import (
    ShortTermMemory,
    LongTermMemory,
    PreferencesStore,
    MemoryManager,
    ConversationTurn,
    UserPreference,
    MemoryFact,
)


class TestConversationTurn:
    """Test ConversationTurn dataclass"""

    def test_conversation_turn_creation(self):
        """Test creating a conversation turn"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input="Hola",
            assistant_response="¿Cómo estás?",
            intent="greeting",
            entities_used=["user"]
        )

        assert turn.user_input == "Hola"
        assert turn.assistant_response == "¿Cómo estás?"
        assert turn.intent == "greeting"
        assert turn.entities_used == ["user"]

    def test_conversation_turn_defaults(self):
        """Test ConversationTurn with default values"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input="test",
            assistant_response="response"
        )

        assert turn.intent is None
        assert turn.entities_used == []


class TestUserPreference:
    """Test UserPreference dataclass"""

    def test_user_preference_creation(self):
        """Test creating a user preference"""
        pref = UserPreference(
            key="favorite_music",
            value="jazz",
            confidence=0.95,
            last_updated=time.time(),
            source="explicit"
        )

        assert pref.key == "favorite_music"
        assert pref.value == "jazz"
        assert pref.confidence == 0.95
        assert pref.source == "explicit"


class TestMemoryFact:
    """Test MemoryFact dataclass"""

    def test_memory_fact_creation(self):
        """Test creating a memory fact"""
        fact = MemoryFact(
            content="Juan tiene un gato",
            category="personal",
            confidence=0.8,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0
        )

        assert fact.content == "Juan tiene un gato"
        assert fact.category == "personal"
        assert fact.confidence == 0.8


class TestShortTermMemory:
    """Test ShortTermMemory class"""

    def test_init_default_max_turns(self):
        """Test initialization with default max_turns"""
        stm = ShortTermMemory()
        assert stm.max_turns == 10
        assert len(stm) == 0

    def test_init_custom_max_turns(self):
        """Test initialization with custom max_turns"""
        stm = ShortTermMemory(max_turns=5)
        assert stm.max_turns == 5

    def test_add_turn(self):
        """Test adding a conversation turn"""
        stm = ShortTermMemory(max_turns=3)

        stm.add_turn(
            user_input="¿Cuál es tu nombre?",
            assistant_response="Soy Jarvis",
            intent="identity"
        )

        assert len(stm) == 1

    def test_add_multiple_turns(self):
        """Test adding multiple turns"""
        stm = ShortTermMemory(max_turns=3)

        for i in range(5):
            stm.add_turn(
                user_input=f"Pregunta {i}",
                assistant_response=f"Respuesta {i}"
            )

        # Should respect max_turns
        assert len(stm) == 3

    def test_get_recent_n_turns(self):
        """Test retrieving recent N turns"""
        stm = ShortTermMemory(max_turns=10)

        for i in range(5):
            stm.add_turn(
                user_input=f"Input {i}",
                assistant_response=f"Output {i}"
            )

        recent = stm.get_recent(n=3)
        assert len(recent) == 3
        assert recent[-1].user_input == "Input 4"

    def test_format_for_prompt(self):
        """Test formatting memory for prompt inclusion"""
        stm = ShortTermMemory()

        stm.add_turn(
            user_input="¿Hola?",
            assistant_response="¡Hola!"
        )
        stm.add_turn(
            user_input="¿Cómo estás?",
            assistant_response="Bien, gracias"
        )

        formatted = stm.format_for_prompt(n=2)

        assert "Conversación reciente:" in formatted
        assert "¿Hola?" in formatted
        assert "¡Hola!" in formatted

    def test_format_for_prompt_empty(self):
        """Test formatting empty memory"""
        stm = ShortTermMemory()
        formatted = stm.format_for_prompt()
        assert formatted == ""

    def test_clear_memory(self):
        """Test clearing short-term memory"""
        stm = ShortTermMemory()

        for i in range(5):
            stm.add_turn(f"Input {i}", f"Output {i}")

        assert len(stm) == 5

        stm.clear()

        assert len(stm) == 0

    def test_len_dunder(self):
        """Test __len__ method"""
        stm = ShortTermMemory(max_turns=5)

        stm.add_turn("User: test", "Assistant: response")
        assert len(stm) == 1

        stm.add_turn("User: test2", "Assistant: response2")
        assert len(stm) == 2


class TestLongTermMemory:
    """Test LongTermMemory class with mocks"""

    def test_init(self):
        """Test LongTermMemory initialization"""
        mock_client = MagicMock()
        ltm = LongTermMemory(mock_client)

        assert ltm._client == mock_client
        assert ltm._collection_name == "user_memories"
        assert ltm._collection is None

    def test_initialize(self):
        """Test initializing collection"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        assert ltm._collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once()

    def test_store_fact(self):
        """Test storing a fact"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        doc_id = ltm.store_fact(
            fact="Juan vive en Buenos Aires",
            category="personal",
            confidence=0.9
        )

        assert doc_id.startswith("fact_")
        mock_collection.add.assert_called_once()

        # Check that metadata contains expected fields
        call_args = mock_collection.add.call_args
        metadatas = call_args.kwargs["metadatas"][0]
        assert metadatas["category"] == "personal"
        assert metadatas["confidence"] == 0.9

    def test_store_fact_with_metadata(self):
        """Test storing fact with custom metadata"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        ltm.store_fact(
            fact="Test fact",
            category="preference",
            confidence=0.7,
            metadata={"source": "user_input"}
        )

        call_args = mock_collection.add.call_args
        metadatas = call_args.kwargs["metadatas"][0]
        assert metadatas["source"] == "user_input"

    def test_search_relevant(self):
        """Test searching for relevant facts"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock search results
        mock_collection.query.return_value = {
            "documents": [["Fact 1", "Fact 2"]],
            "metadatas": [[
                {"category": "personal", "confidence": 0.9},
                {"category": "preference", "confidence": 0.7}
            ]],
            "distances": [[0.2, 0.5]],
            "ids": [["fact_1", "fact_2"]]
        }
        mock_collection.get.return_value = {
            "metadatas": [{"access_count": 0}]
        }

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        results = ltm.search_relevant(
            query="¿Dónde vive Juan?",
            n_results=5,
            min_confidence=0.5
        )

        assert len(results) == 2
        assert results[0]["content"] == "Fact 1"

    def test_get_by_category(self):
        """Test retrieving facts by category"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_collection.get.return_value = {
            "documents": ["Personal fact 1", "Personal fact 2"],
            "metadatas": [
                {"category": "personal", "confidence": 0.9},
                {"category": "personal", "confidence": 0.85}
            ]
        }

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        facts = ltm.get_by_category("personal", limit=10)

        assert len(facts) == 2
        assert facts[0]["content"] == "Personal fact 1"

    def test_delete_fact(self):
        """Test deleting a fact"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        result = ltm.delete_fact("fact_123")

        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["fact_123"])

    def test_get_stats(self):
        """Test getting memory statistics"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 42

        ltm = LongTermMemory(mock_client)
        ltm.initialize()

        stats = ltm.get_stats()

        assert stats["total_facts"] == 42
        assert stats["collection"] == "user_memories"


class TestPreferencesStore:
    """Test PreferencesStore class"""

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))

            assert store.file_path == file_path
            assert len(store._preferences) == 0

    def test_save_and_load_preferences(self):
        """Test saving and loading preferences"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store1 = PreferencesStore(str(file_path))
            store1.set("favorite_artist", "Queen", confidence=0.95, source="explicit")

            # Load in new instance
            store2 = PreferencesStore(str(file_path))
            value = store2.get("favorite_artist")

            assert value == "Queen"

    def test_set_preference(self):
        """Test setting a preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))
            store.set("theme", "dark", confidence=0.9, source="explicit")

            assert store.get("theme") == "dark"

    def test_get_nonexistent_preference(self):
        """Test getting non-existent preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))
            value = store.get("nonexistent")

            assert value is None

    def test_get_all_preferences(self):
        """Test getting all preferences"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))
            store.set("pref1", "value1")
            store.set("pref2", "value2")

            all_prefs = store.get_all()

            assert all_prefs["pref1"] == "value1"
            assert all_prefs["pref2"] == "value2"

    def test_delete_preference(self):
        """Test deleting a preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))
            store.set("temp_pref", "temp_value")

            assert store.get("temp_pref") == "temp_value"

            result = store.delete("temp_pref")

            assert result is True
            assert store.get("temp_pref") is None

    def test_delete_nonexistent_preference(self):
        """Test deleting non-existent preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "preferences.json"

            store = PreferencesStore(str(file_path))
            result = store.delete("nonexistent")

            assert result is False


class TestMemoryManager:
    """Test MemoryManager class"""

    def test_init(self):
        """Test MemoryManager initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            prefs_path = Path(tmpdir) / "prefs.json"

            mm = MemoryManager(
                mock_client,
                preferences_path=str(prefs_path),
                max_short_term_turns=15
            )

            assert isinstance(mm.short_term, ShortTermMemory)
            assert isinstance(mm.long_term, LongTermMemory)
            assert isinstance(mm.preferences, PreferencesStore)
            assert mm.short_term.max_turns == 15

    def test_initialize(self):
        """Test initializing memory manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection

            mm = MemoryManager(mock_client)
            mm.initialize()

            mock_client.get_or_create_collection.assert_called()

    def test_record_interaction(self):
        """Test recording an interaction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection

            mm = MemoryManager(mock_client)

            mm.record_interaction(
                user_input="¿Hola?",
                assistant_response="¡Hola!",
                intent="greeting"
            )

            assert len(mm.short_term) == 1

    def test_record_interaction_with_fact_extraction(self):
        """Test recording interaction with fact extractor"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection

            mm = MemoryManager(mock_client)

            # Mock fact extractor
            mock_extractor = MagicMock()
            mock_extractor.extract.return_value = [
                {"content": "Juan vive en Buenos Aires", "category": "personal", "confidence": 0.9}
            ]
            mm.set_fact_extractor(mock_extractor)

            mm.record_interaction(
                user_input="Vivo en Buenos Aires",
                assistant_response="Entendido"
            )

            mock_extractor.extract.assert_called()
            mock_collection.add.assert_called()

    def test_build_context(self):
        """Test building context"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

            mm = MemoryManager(mock_client)
            mm.short_term.add_turn("Hola", "¡Hola!")
            mm.preferences.set("favorite_artist", "The Beatles")

            context = mm.build_context("¿Cuál es tu artista favorito?")

            assert "short_term" in context
            assert "relevant_facts" in context
            assert "preferences" in context

    def test_format_context_for_prompt(self):
        """Test formatting context for prompt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "documents": [["Fact 1"]],
                "metadatas": [[{"category": "personal"}]],
                "distances": [[0.2]],
                "ids": [["fact_1"]]
            }
            mock_collection.get.return_value = {
                "metadatas": [{"access_count": 0}]
            }

            mm = MemoryManager(mock_client)
            mm.preferences.set("theme", "dark")

            formatted = mm.format_context_for_prompt("Query")

            assert "Preferencias del usuario" in formatted or "theme" in formatted

    def test_remember_preference_explicit(self):
        """Test remembering an explicit preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()

            mm = MemoryManager(mock_client)
            mm.remember_preference("color", "blue", explicit=True)

            pref = mm.preferences._preferences["color"]
            assert pref.confidence == 0.95
            assert pref.source == "explicit"

    def test_remember_preference_inferred(self):
        """Test remembering an inferred preference"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()

            mm = MemoryManager(mock_client)
            mm.remember_preference("color", "red", explicit=False)

            pref = mm.preferences._preferences["color"]
            assert pref.confidence == 0.7
            assert pref.source == "inferred"

    def test_get_stats(self):
        """Test getting memory statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_collection.count.return_value = 5

            prefs_path = str(Path(tmpdir) / "preferences.json")
            mm = MemoryManager(mock_client, preferences_path=prefs_path)
            mm.short_term.add_turn("Input", "Output")
            mm.preferences.set("pref1", "value1")

            stats = mm.get_stats()

            assert stats["short_term_turns"] == 1
            assert stats["preferences_count"] == 1
            assert "long_term" in stats


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager"""

    def test_full_conversation_cycle(self):
        """Test a full conversation cycle with memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

            mm = MemoryManager(mock_client)
            mm.initialize()

            # Simulate conversation
            mm.record_interaction("Hola", "¡Hola! ¿Cómo estás?")
            mm.record_interaction("Estoy bien", "Me alegra escuchar eso")
            mm.remember_preference("language", "es", explicit=True)

            # Verify state
            assert len(mm.short_term) == 2
            assert mm.preferences.get("language") == "es"

            context = mm.build_context("¿Cómo estoy?")
            assert context["preferences"]["language"] == "es"
