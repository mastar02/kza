"""
Tests for ContextManager - Per-user conversation context.
"""

import pytest
import time
from unittest.mock import patch

from src.orchestrator.context_manager import (
    ContextManager,
    UserContext,
    ConversationTurn,
    MusicPreferences
)


class TestConversationTurn:
    """Tests for ConversationTurn dataclass"""

    def test_create_turn(self):
        turn = ConversationTurn(
            role="user",
            content="Prende la luz"
        )
        assert turn.role == "user"
        assert turn.content == "Prende la luz"
        assert turn.intent is None
        assert turn.entities == []

    def test_turn_with_intent(self):
        turn = ConversationTurn(
            role="assistant",
            content="Luz encendida",
            intent="domotics",
            entities=["light.living"]
        )
        assert turn.intent == "domotics"
        assert "light.living" in turn.entities

    def test_turn_to_dict(self):
        turn = ConversationTurn(
            role="user",
            content="Test",
            intent="test"
        )
        data = turn.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert "timestamp" in data

    def test_turn_from_dict(self):
        data = {
            "role": "user",
            "content": "Test",
            "timestamp": 12345.0,
            "intent": "domotics",
            "entities": ["light.test"]
        }
        turn = ConversationTurn.from_dict(data)
        assert turn.role == "user"
        assert turn.timestamp == 12345.0


class TestMusicPreferences:
    """Tests for MusicPreferences dataclass"""

    def test_default_preferences(self):
        prefs = MusicPreferences()
        assert prefs.favorite_genres == []
        assert prefs.favorite_artists == []
        assert prefs.default_energy is None

    def test_custom_preferences(self):
        prefs = MusicPreferences(
            favorite_genres=["rock", "jazz"],
            favorite_artists=["Coldplay"],
            default_energy=0.7
        )
        assert "rock" in prefs.favorite_genres
        assert prefs.default_energy == 0.7

    def test_to_dict(self):
        prefs = MusicPreferences(favorite_genres=["pop"])
        data = prefs.to_dict()
        assert data["favorite_genres"] == ["pop"]
        assert "default_energy" in data


class TestUserContext:
    """Tests for UserContext dataclass"""

    def test_create_context(self):
        ctx = UserContext(
            user_id="user_1",
            user_name="Juan",
            zone_id="living"
        )
        assert ctx.user_id == "user_1"
        assert ctx.user_name == "Juan"
        assert ctx.zone_id == "living"
        assert len(ctx.conversation_history) == 0

    def test_add_turn(self):
        ctx = UserContext(user_id="user_1", user_name="Test")
        ctx.add_turn("user", "Hola")
        ctx.add_turn("assistant", "Hola, ¿en qué te ayudo?")

        assert len(ctx.conversation_history) == 2
        assert ctx.turns_count == 2
        assert ctx.conversation_history[0].role == "user"
        assert ctx.conversation_history[1].role == "assistant"

    def test_max_history_limit(self):
        ctx = UserContext(user_id="user_1", user_name="Test", max_history=3)

        # Add 5 turns
        for i in range(5):
            ctx.add_turn("user", f"Message {i}")

        # Should only keep last 3
        assert len(ctx.conversation_history) == 3
        assert ctx.conversation_history[0].content == "Message 2"
        assert ctx.conversation_history[-1].content == "Message 4"

    def test_get_history_for_prompt(self):
        ctx = UserContext(user_id="user_1", user_name="Test")
        ctx.add_turn("user", "Pregunta")
        ctx.add_turn("assistant", "Respuesta")

        messages = ctx.get_history_for_prompt()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_clear_history(self):
        ctx = UserContext(user_id="user_1", user_name="Test")
        ctx.add_turn("user", "Test")
        ctx.clear_history()

        assert len(ctx.conversation_history) == 0

    def test_last_active_updated(self):
        ctx = UserContext(user_id="user_1", user_name="Test")
        initial_time = ctx.last_active

        time.sleep(0.01)
        ctx.add_turn("user", "Test")

        assert ctx.last_active > initial_time

    def test_music_preferences(self):
        ctx = UserContext(
            user_id="user_1",
            user_name="Test",
            music_preferences=MusicPreferences(favorite_genres=["rock"])
        )
        assert "rock" in ctx.music_preferences.favorite_genres


class TestContextManager:
    """Tests for ContextManager"""

    def test_create_manager(self):
        manager = ContextManager(max_history=10, inactive_timeout=300)
        assert manager.max_history == 10
        assert manager.inactive_timeout == 300

    def test_get_or_create_new(self):
        manager = ContextManager()
        ctx = manager.get_or_create("user_1", "Juan", "living")

        assert ctx.user_id == "user_1"
        assert ctx.user_name == "Juan"
        assert ctx.zone_id == "living"

    def test_get_or_create_existing(self):
        manager = ContextManager()

        # Create context
        ctx1 = manager.get_or_create("user_1", "Juan", "living")
        ctx1.add_turn("user", "Hola")

        # Get same context
        ctx2 = manager.get_or_create("user_1", "Juan", "kitchen")

        # Should be same context with updated zone
        assert ctx2 is ctx1
        assert len(ctx2.conversation_history) == 1
        assert ctx2.zone_id == "kitchen"  # Zone updated

    def test_get_nonexistent(self):
        manager = ContextManager()
        ctx = manager.get("nonexistent")
        assert ctx is None

    def test_remove_context(self):
        manager = ContextManager()
        manager.get_or_create("user_1", "Juan")

        manager.remove_context("user_1")

        ctx = manager.get("user_1")
        assert ctx is None

    def test_clear_user_history(self):
        manager = ContextManager()
        ctx = manager.get_or_create("user_1", "Juan")
        manager.add_turn("user_1", "user", "Test")

        manager.clear_user_history("user_1")

        ctx = manager.get("user_1")
        assert ctx is not None
        assert len(ctx.conversation_history) == 0

    def test_build_prompt(self):
        manager = ContextManager()
        manager.get_or_create("user_1", "Juan")
        manager.add_turn("user_1", "user", "¿Qué es Python?")
        manager.add_turn("user_1", "assistant", "Python es un lenguaje de programación.")

        prompt = manager.build_prompt("user_1", "¿Y para qué sirve?")

        assert "Juan" in prompt
        assert "¿Qué es Python?" in prompt
        assert "Python es un lenguaje" in prompt
        assert "¿Y para qué sirve?" in prompt

    def test_multiple_users(self):
        manager = ContextManager()

        ctx1 = manager.get_or_create("user_1", "Juan", "living")
        ctx2 = manager.get_or_create("user_2", "María", "kitchen")

        ctx1.add_turn("user", "Mensaje de Juan")
        ctx2.add_turn("user", "Mensaje de María")

        # Contexts should be independent
        assert len(ctx1.conversation_history) == 1
        assert len(ctx2.conversation_history) == 1
        assert ctx1.conversation_history[0].content == "Mensaje de Juan"
        assert ctx2.conversation_history[0].content == "Mensaje de María"

    def test_get_stats(self):
        manager = ContextManager()
        manager.get_or_create("user_1", "Juan")
        manager.get_or_create("user_2", "María")
        manager.add_turn("user_1", "user", "Test 1")
        manager.add_turn("user_1", "assistant", "Response 1")
        manager.add_turn("user_2", "user", "Test 2")

        stats = manager.get_stats()
        assert stats["active_contexts"] == 2
        assert stats["total_turns"] == 3

    def test_pending_confirmation(self):
        manager = ContextManager()
        manager.get_or_create("user_1", "Juan")

        # Set pending confirmation
        confirmation = {"routine": "test", "action": "create"}
        manager.set_pending_confirmation("user_1", confirmation)

        ctx = manager.get("user_1")
        assert ctx.pending_confirmation == confirmation

        # Clear confirmation
        manager.clear_pending_confirmation("user_1")
        ctx = manager.get("user_1")
        assert ctx.pending_confirmation is None

    def test_get_pending_confirmation(self):
        manager = ContextManager()
        manager.get_or_create("user_1", "Juan")
        manager.set_pending_confirmation("user_1", {"test": True})

        result = manager.get_pending_confirmation("user_1")
        assert result == {"test": True}

        # Non-existent user
        result = manager.get_pending_confirmation("nonexistent")
        assert result is None


class TestContextManagerCleanup:
    """Tests for automatic cleanup of inactive contexts"""

    def test_cleanup_inactive(self):
        # Create manager with very short timeout
        manager = ContextManager(inactive_timeout=0.05)

        # Create context
        manager.get_or_create("user_1", "Juan")

        # Wait for timeout
        time.sleep(0.1)

        # Run cleanup
        cleaned = manager.cleanup_inactive()

        assert cleaned == 1
        assert manager.get("user_1") is None

    def test_cleanup_keeps_active(self):
        manager = ContextManager(inactive_timeout=1.0)

        # Create context
        manager.get_or_create("user_1", "Juan")

        # Run cleanup immediately (context is still active)
        cleaned = manager.cleanup_inactive()

        assert cleaned == 0
        assert manager.get("user_1") is not None

    def test_thread_safety(self):
        """Test that context manager is thread-safe"""
        import threading

        manager = ContextManager()
        errors = []

        def worker(user_id):
            try:
                for i in range(10):
                    ctx = manager.get_or_create(user_id, f"User{user_id}")
                    ctx.add_turn("user", f"Message {i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"user_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = manager.get_stats()
        assert stats["active_contexts"] == 5
