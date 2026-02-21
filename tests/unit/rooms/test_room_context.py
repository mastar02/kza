"""
Tests para Room Context Module
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from src.rooms.room_context import (
    RoomConfig,
    RoomContext,
    RoomContextManager,
    ContextSource,
    create_default_rooms,
)


@pytest.fixture
def room_configs():
    """Configuraciones de habitaciones para tests"""
    return [
        RoomConfig(
            room_id="living",
            name="Living",
            display_name="el living",
            mic_device_index=2,
            bt_adapter="hci0",
            default_light="light.living",
            default_climate="climate.living_ac",
            default_cover="cover.living_persiana",
            aliases=["living", "sala", "salón"],
        ),
        RoomConfig(
            room_id="escritorio",
            name="Escritorio",
            display_name="el escritorio",
            mic_device_index=3,
            bt_adapter="hci1",
            default_light="light.escritorio",
            default_climate="climate.escritorio_ac",
            aliases=["escritorio", "oficina", "estudio"],
        ),
        RoomConfig(
            room_id="cocina",
            name="Cocina",
            display_name="la cocina",
            mic_device_index=4,
            bt_adapter="hci2",
            default_light="light.cocina",
            default_fan="fan.cocina_extractor",
            aliases=["cocina", "kitchen"],
        ),
        RoomConfig(
            room_id="hall",
            name="Hall",
            display_name="el hall",
            mic_device_index=5,
            bt_adapter="hci3",
            default_light="light.hall",
            aliases=["hall", "pasillo", "entrada"],
        ),
        RoomConfig(
            room_id="bano",
            name="Baño",
            display_name="el baño",
            mic_device_index=6,
            bt_adapter="hci4",
            default_light="light.bano",
            default_fan="fan.bano_extractor",
            aliases=["baño", "bathroom"],
        ),
    ]


@pytest.fixture
def mock_presence_detector():
    """Mock del PresenceDetector"""
    detector = MagicMock()
    detector.get_user_zone = MagicMock(return_value=None)
    detector.get_zone_occupancy = MagicMock(return_value=None)
    return detector


@pytest.fixture
def manager(room_configs, mock_presence_detector):
    """RoomContextManager con habitaciones configuradas"""
    mgr = RoomContextManager(
        presence_detector=mock_presence_detector,
        fallback_room="living"
    )
    for config in room_configs:
        mgr.add_room(config)
    return mgr


# =========================================================================
# Tests de configuración
# =========================================================================

class TestRoomConfiguration:
    def test_add_rooms(self, manager):
        """Verificar que se agregan las 5 habitaciones"""
        rooms = manager.get_all_rooms()
        assert len(rooms) == 5
        assert "living" in rooms
        assert "escritorio" in rooms
        assert "cocina" in rooms
        assert "hall" in rooms
        assert "bano" in rooms

    def test_room_aliases(self, manager):
        """Verificar resolución de aliases"""
        assert manager.get_room_by_alias("sala") is not None
        assert manager.get_room_by_alias("sala").room_id == "living"
        assert manager.get_room_by_alias("oficina").room_id == "escritorio"
        assert manager.get_room_by_alias("pasillo").room_id == "hall"
        assert manager.get_room_by_alias("kitchen").room_id == "cocina"
        assert manager.get_room_by_alias("bathroom").room_id == "bano"

    def test_default_rooms_creation(self):
        """Verificar create_default_rooms"""
        rooms = create_default_rooms()
        assert len(rooms) == 5
        room_ids = [r.room_id for r in rooms]
        assert "living" in room_ids
        assert "escritorio" in room_ids
        assert "hall" in room_ids
        assert "cocina" in room_ids
        assert "bano" in room_ids


# =========================================================================
# Tests de resolución de habitación
# =========================================================================

class TestRoomResolution:
    def test_resolve_by_mic_zone(self, manager):
        """Resolver habitación por zona del micrófono"""
        ctx = manager.resolve_room(mic_zone_id="cocina")
        assert ctx is not None
        assert ctx.room_id == "cocina"
        assert ctx.source == ContextSource.MICROPHONE
        assert ctx.confidence == 0.7

    def test_resolve_by_mic_device_index(self, manager):
        """Resolver habitación por device index del micrófono"""
        ctx = manager.resolve_room(mic_device_index=4)
        assert ctx is not None
        assert ctx.room_id == "cocina"

    def test_resolve_by_spoken_room(self, manager):
        """Resolver habitación mencionada en el comando"""
        ctx = manager.resolve_room(spoken_room="la cocina")
        assert ctx is not None
        assert ctx.room_id == "cocina"
        assert ctx.source == ContextSource.MANUAL
        assert ctx.confidence == 1.0

    def test_spoken_room_overrides_mic(self, manager):
        """La habitación hablada tiene prioridad sobre el mic"""
        ctx = manager.resolve_room(
            mic_zone_id="living",
            spoken_room="cocina"
        )
        assert ctx.room_id == "cocina"
        assert ctx.source == ContextSource.MANUAL

    def test_resolve_with_bt_confirmation(self, manager, mock_presence_detector):
        """Confirmar habitación con BT aumenta confianza"""
        mock_presence_detector.get_user_zone.return_value = "escritorio"

        ctx = manager.resolve_room(
            mic_zone_id="escritorio",
            user_id="mastar"
        )
        assert ctx.room_id == "escritorio"
        assert ctx.source == ContextSource.BOTH
        assert ctx.confidence == 1.0

    def test_resolve_mic_bt_discrepancy(self, manager, mock_presence_detector):
        """Cuando mic y BT no coinciden, confiar en mic"""
        mock_presence_detector.get_user_zone.return_value = "living"

        ctx = manager.resolve_room(
            mic_zone_id="cocina",
            user_id="mastar"
        )
        assert ctx.room_id == "cocina"  # Mic wins
        assert ctx.source == ContextSource.MICROPHONE
        assert ctx.confidence == 0.7

    def test_resolve_bt_only(self, manager, mock_presence_detector):
        """Resolver solo por BT cuando no hay mic"""
        mock_presence_detector.get_user_zone.return_value = "hall"

        ctx = manager.resolve_room(user_id="mastar")
        assert ctx.room_id == "hall"
        assert ctx.source == ContextSource.BLUETOOTH
        assert ctx.confidence == 0.6

    def test_resolve_fallback(self, manager):
        """Fallback al living cuando no se puede determinar"""
        ctx = manager.resolve_room()
        assert ctx.room_id == "living"
        assert ctx.confidence == 0.2

    def test_resolve_returns_none_without_fallback(self, room_configs):
        """Sin fallback, retorna None si no se puede resolver"""
        mgr = RoomContextManager(fallback_room=None)
        for config in room_configs:
            mgr.add_room(config)

        ctx = mgr.resolve_room()
        assert ctx is None


# =========================================================================
# Tests de resolución de entities
# =========================================================================

class TestEntityResolution:
    def test_resolve_light_from_context(self, manager):
        """Resolver light entity desde contexto"""
        ctx = manager.resolve_room(mic_zone_id="cocina")
        entity = manager.resolve_entity("light", room_context=ctx)
        assert entity == "light.cocina"

    def test_resolve_climate_from_context(self, manager):
        """Resolver climate entity desde contexto"""
        ctx = manager.resolve_room(mic_zone_id="living")
        entity = manager.resolve_entity("climate", room_context=ctx)
        assert entity == "climate.living_ac"

    def test_resolve_entity_spoken_room(self, manager):
        """Resolver entity por habitación hablada"""
        entity = manager.resolve_entity("light", spoken_room="escritorio")
        assert entity == "light.escritorio"

    def test_resolve_fan_cocina(self, manager):
        """Resolver fan de la cocina"""
        ctx = manager.resolve_room(mic_zone_id="cocina")
        entity = manager.resolve_entity("fan", room_context=ctx)
        assert entity == "fan.cocina_extractor"

    def test_resolve_nonexistent_entity(self, manager):
        """Resolver entity que no existe en la habitación"""
        ctx = manager.resolve_room(mic_zone_id="hall")
        entity = manager.resolve_entity("climate", room_context=ctx)
        assert entity is None  # Hall no tiene aire acondicionado


# =========================================================================
# Tests de contexto
# =========================================================================

class TestRoomContext:
    def test_context_has_entities(self, manager):
        """Verificar que el contexto incluye entities"""
        ctx = manager.resolve_room(mic_zone_id="living")
        assert "light" in ctx.entities
        assert "climate" in ctx.entities
        assert "cover" in ctx.entities
        assert ctx.entities["light"] == "light.living"

    def test_context_display_name(self, manager):
        """Verificar nombres para TTS"""
        ctx = manager.resolve_room(mic_zone_id="cocina")
        assert ctx.display_name == "la cocina"

        ctx = manager.resolve_room(mic_zone_id="bano")
        assert ctx.display_name == "el baño"

    def test_high_confidence(self, manager, mock_presence_detector):
        """Verificar is_high_confidence"""
        mock_presence_detector.get_user_zone.return_value = "living"
        ctx = manager.resolve_room(mic_zone_id="living", user_id="mastar")
        assert ctx.is_high_confidence is True

    def test_low_confidence(self, manager):
        """Fallback no es alta confianza"""
        ctx = manager.resolve_room()
        assert ctx.is_high_confidence is False

    def test_context_user_tracking(self, manager):
        """Verificar que se trackea el usuario"""
        ctx = manager.resolve_room(mic_zone_id="cocina", user_id="mastar")
        assert ctx.user_id == "mastar"

    def test_user_room_history(self, manager):
        """Verificar historial de habitaciones del usuario"""
        manager.resolve_room(mic_zone_id="living", user_id="mastar")
        manager.resolve_room(mic_zone_id="cocina", user_id="mastar")
        manager.resolve_room(mic_zone_id="escritorio", user_id="mastar")

        room = manager.get_user_room("mastar")
        # Debería ser la última habitación (escritorio) via historial
        # (sin BT activo, cae al historial)
        assert room == "escritorio"


# =========================================================================
# Tests de TTS speaker
# =========================================================================

class TestTTSSpeaker:
    def test_get_tts_speaker(self, manager):
        """Obtener speaker correcto para la habitación"""
        ctx = manager.resolve_room(mic_zone_id="living")
        # En room_configs del fixture no tiene tts_speaker
        # pero create_default_rooms() sí lo tiene
        speaker = manager.get_tts_speaker(ctx)
        # fixture no define tts_speaker, así que es None
        assert speaker is None

    def test_get_tts_speaker_with_default_rooms(self):
        """Verificar speaker con configuración por defecto"""
        rooms = create_default_rooms()
        mgr = RoomContextManager()
        for room in rooms:
            mgr.add_room(room)

        ctx = mgr.resolve_room(mic_zone_id="living")
        speaker = mgr.get_tts_speaker(ctx)
        assert speaker == "media_player.living_speaker"


# =========================================================================
# Tests de room changed callback
# =========================================================================

class TestRoomChanged:
    def test_room_changed_callback(self, manager):
        """Verificar callback cuando cambia de habitación"""
        changes = []
        manager.on_room_changed(
            lambda old, new, user: changes.append((old, new, user))
        )

        manager.resolve_room(mic_zone_id="living", user_id="mastar")
        manager.resolve_room(mic_zone_id="cocina", user_id="mastar")

        assert len(changes) == 1
        assert changes[0] == ("living", "cocina", "mastar")


# =========================================================================
# Tests de summary
# =========================================================================

class TestSummary:
    def test_room_summary(self, manager):
        """Verificar resumen de habitaciones"""
        summary = manager.get_room_summary()
        assert len(summary) == 5
        assert summary["living"]["name"] == "Living"
        assert summary["living"]["mic_active"] is True
        assert summary["living"]["bt_active"] is True
        assert "light" in summary["living"]["entities"]
