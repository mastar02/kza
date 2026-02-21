"""
Tests for Speaker Enrollment - Voice-based learning system

Tests cover:
1. Intent detection from voice commands
2. Speaker naming and renaming
3. Room/floor assignment
4. Group management
5. Device discovery
6. Confirmation flows
7. Alias management
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.spotify.speaker_enrollment import (
    SpeakerEnrollment, EnrollmentIntent, EnrollmentCommand, PendingDevice
)
from src.spotify.speaker_groups import SpeakerGroupManager, Speaker, SpeakerGroup, GroupType


# ============================================================
# Mock Spotify Device
# ============================================================

@dataclass
class MockSpotifyDevice:
    """Mock Spotify device for testing"""
    id: str
    name: str
    type: str = "Speaker"
    is_active: bool = False
    volume_percent: int = 50


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_spotify_client():
    """Mock Spotify client"""
    client = MagicMock()
    client.get_devices = AsyncMock(return_value=[
        MockSpotifyDevice(id="dev_1", name="Echo Kitchen", is_active=True),
        MockSpotifyDevice(id="dev_2", name="Living Room Speaker"),
        MockSpotifyDevice(id="dev_3", name="Bedroom Echo"),
    ])
    return client


@pytest.fixture
def speaker_manager():
    """Empty speaker manager"""
    return SpeakerGroupManager()


@pytest.fixture
def manager_with_speakers():
    """Speaker manager with some speakers configured"""
    manager = SpeakerGroupManager()
    manager.add_speaker(Speaker(
        id="cocina",
        name="Cocina",
        spotify_device_id="dev_1",
        room="cocina",
        floor="planta_baja",
        aliases=["kitchen"]
    ))
    manager.add_speaker(Speaker(
        id="sala",
        name="Sala",
        spotify_device_id="dev_2",
        room="sala",
        floor="planta_baja",
        is_default=True,
        aliases=["living"]
    ))
    return manager


@pytest.fixture
def enrollment(mock_spotify_client, speaker_manager):
    """SpeakerEnrollment instance"""
    return SpeakerEnrollment(
        spotify_client=mock_spotify_client,
        speaker_manager=speaker_manager,
        auto_save=False
    )


@pytest.fixture
def enrollment_with_speakers(mock_spotify_client, manager_with_speakers):
    """Enrollment with pre-configured speakers"""
    return SpeakerEnrollment(
        spotify_client=mock_spotify_client,
        speaker_manager=manager_with_speakers,
        auto_save=False
    )


# ============================================================
# Intent Detection Tests
# ============================================================

class TestIntentDetection:
    """Tests for detecting intents from voice commands"""

    def test_detect_name_speaker(self, enrollment):
        """Should detect NAME_SPEAKER intent"""
        commands = [
            ("esta bocina se llama cocina", "cocina"),
            ("bocina se llama sala", "sala"),
        ]
        for cmd, expected_name in commands:
            result = enrollment.detect_intent(cmd)
            assert result.intent == EnrollmentIntent.NAME_SPEAKER, f"Failed for: {cmd}"
            assert expected_name in result.new_name.lower(), f"Expected {expected_name} in {result.new_name}"

    def test_detect_rename_speaker(self, enrollment):
        """Should detect RENAME_SPEAKER intent"""
        result = enrollment.detect_intent("renombra la cocina a cocina principal")
        assert result.intent == EnrollmentIntent.RENAME_SPEAKER
        assert result.speaker_name == "cocina"
        assert result.new_name == "cocina principal"

    def test_detect_set_floor(self, enrollment):
        """Should detect SET_FLOOR intent"""
        commands = [
            ("la cocina está en la planta baja", "planta_baja"),
            ("la sala está en la planta alta", "planta_alta"),
            ("el dormitorio está arriba", "planta_alta"),
            ("el garage está abajo", "planta_baja"),
        ]
        for cmd, expected_floor in commands:
            result = enrollment.detect_intent(cmd)
            assert result.intent == EnrollmentIntent.SET_FLOOR, f"Failed for: {cmd}"
            assert result.floor == expected_floor, f"Expected {expected_floor} for: {cmd}"

    def test_detect_create_group(self, enrollment):
        """Should detect CREATE_GROUP intent"""
        result = enrollment.detect_intent("crea un grupo llamado área social con cocina, sala y comedor")
        assert result.intent == EnrollmentIntent.CREATE_GROUP
        assert result.group_name == "área social"
        assert "cocina" in result.speaker_list
        assert "sala" in result.speaker_list
        assert "comedor" in result.speaker_list

    def test_detect_add_to_group(self, enrollment):
        """Should detect ADD_TO_GROUP intent"""
        result = enrollment.detect_intent("agrega la cocina al grupo social")
        assert result.intent == EnrollmentIntent.ADD_TO_GROUP
        assert result.speaker_name == "cocina"
        assert result.group_name == "social"

    def test_detect_remove_from_group(self, enrollment):
        """Should detect REMOVE_FROM_GROUP intent"""
        result = enrollment.detect_intent("quita la cocina del grupo social")
        assert result.intent == EnrollmentIntent.REMOVE_FROM_GROUP
        assert result.speaker_name == "cocina"
        assert result.group_name == "social"

    def test_detect_delete_group(self, enrollment):
        """Should detect DELETE_GROUP intent"""
        result = enrollment.detect_intent("elimina el grupo social")
        assert result.intent == EnrollmentIntent.DELETE_GROUP
        assert result.group_name == "social"

    def test_detect_discover_devices(self, enrollment):
        """Should detect DISCOVER_DEVICES intent"""
        commands = [
            "busca bocinas nuevas",
            "detecta bocinas",
            "hay bocinas nuevas",
        ]
        for cmd in commands:
            result = enrollment.detect_intent(cmd)
            assert result.intent == EnrollmentIntent.DISCOVER_DEVICES, f"Failed for: {cmd}"

    def test_detect_list_speakers(self, enrollment):
        """Should detect LIST_SPEAKERS intent"""
        commands = [
            "qué bocinas tengo",
            "lista de bocinas",
            "mis bocinas",
        ]
        for cmd in commands:
            result = enrollment.detect_intent(cmd)
            assert result.intent == EnrollmentIntent.LIST_SPEAKERS, f"Failed for: {cmd}"

    def test_detect_list_groups(self, enrollment):
        """Should detect LIST_GROUPS intent"""
        result = enrollment.detect_intent("qué grupos tengo")
        assert result.intent == EnrollmentIntent.LIST_GROUPS

    def test_detect_add_alias(self, enrollment):
        """Should detect ADD_ALIAS intent"""
        result = enrollment.detect_intent("la cocina también se llama kitchen")
        assert result.intent == EnrollmentIntent.ADD_ALIAS
        assert result.speaker_name == "cocina"
        assert result.alias == "kitchen"

    def test_detect_set_default(self, enrollment):
        """Should detect SET_DEFAULT intent"""
        result = enrollment.detect_intent("la sala es la bocina principal")
        assert result.intent == EnrollmentIntent.SET_DEFAULT
        assert result.speaker_name == "sala"

    def test_detect_forget_speaker(self, enrollment):
        """Should detect FORGET_SPEAKER intent"""
        result = enrollment.detect_intent("olvida la bocina del garage")
        assert result.intent == EnrollmentIntent.FORGET_SPEAKER
        assert "garage" in result.speaker_name

    def test_detect_unknown(self, enrollment):
        """Should return UNKNOWN for unrecognized commands"""
        result = enrollment.detect_intent("esto no tiene sentido")
        assert result.intent == EnrollmentIntent.UNKNOWN


# ============================================================
# Speaker List Parsing Tests
# ============================================================

class TestSpeakerListParsing:
    """Tests for parsing speaker lists from text"""

    def test_parse_with_commas_and_y(self, enrollment):
        """Should parse 'a, b y c' format"""
        result = enrollment._parse_speaker_list("cocina, sala y dormitorio")
        assert result == ["cocina", "sala", "dormitorio"]

    def test_parse_with_y_only(self, enrollment):
        """Should parse 'a y b' format"""
        result = enrollment._parse_speaker_list("cocina y sala")
        assert result == ["cocina", "sala"]

    def test_parse_single(self, enrollment):
        """Should parse single speaker"""
        result = enrollment._parse_speaker_list("cocina")
        assert result == ["cocina"]


# ============================================================
# Name Speaker Tests
# ============================================================

class TestNameSpeaker:
    """Tests for naming speakers"""

    @pytest.mark.asyncio
    async def test_name_active_device(self, enrollment, mock_spotify_client):
        """Should name the currently active device"""
        result = await enrollment.process("esta bocina se llama cocina")

        assert result["success"] is True
        assert result["action"] == "speaker_named"
        assert "cocina" in enrollment.speakers.speakers

    @pytest.mark.asyncio
    async def test_name_no_active_device(self, enrollment, mock_spotify_client):
        """Should fail if no active device"""
        mock_spotify_client.get_devices.return_value = [
            MockSpotifyDevice(id="dev_1", name="Echo Kitchen", is_active=False),
        ]

        result = await enrollment.process("esta bocina se llama cocina")

        assert result["success"] is False
        assert "no encontré" in result["response"].lower() or "activa" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_name_uses_pending_device(self, enrollment, mock_spotify_client):
        """Should use pending device if no active"""
        mock_spotify_client.get_devices.return_value = []

        # Add a pending device
        enrollment._pending_devices["pending_123"] = PendingDevice(
            spotify_device=MockSpotifyDevice(id="pending_123", name="Test Speaker"),
            discovered_at=1234567890.0
        )

        result = await enrollment.process("esta bocina se llama test")

        assert result["success"] is True
        assert "test" in enrollment.speakers.speakers


# ============================================================
# Rename Speaker Tests
# ============================================================

class TestRenameSpeaker:
    """Tests for renaming speakers"""

    @pytest.mark.asyncio
    async def test_rename_existing_speaker(self, enrollment_with_speakers):
        """Should rename existing speaker"""
        result = await enrollment_with_speakers.process("renombra la cocina a cocina principal")

        assert result["success"] is True
        assert result["action"] == "speaker_renamed"
        assert enrollment_with_speakers.speakers.speakers["cocina"].name == "Cocina Principal"

    @pytest.mark.asyncio
    async def test_rename_nonexistent_speaker(self, enrollment_with_speakers):
        """Should fail for nonexistent speaker"""
        result = await enrollment_with_speakers.process("renombra la garage a garaje")

        assert result["success"] is False
        assert "no encontré" in result["response"].lower()


# ============================================================
# Room/Floor Assignment Tests
# ============================================================

class TestLocationAssignment:
    """Tests for room and floor assignment"""

    @pytest.mark.asyncio
    async def test_set_floor(self, enrollment_with_speakers):
        """Should set speaker floor"""
        result = await enrollment_with_speakers.process("la cocina está en la planta alta")

        assert result["success"] is True
        assert enrollment_with_speakers.speakers.speakers["cocina"].floor == "planta_alta"

    @pytest.mark.asyncio
    async def test_set_floor_nonexistent_speaker(self, enrollment_with_speakers):
        """Should fail for nonexistent speaker"""
        result = await enrollment_with_speakers.process("el garage está en la planta baja")

        assert result["success"] is False


# ============================================================
# Group Management Tests
# ============================================================

class TestGroupManagement:
    """Tests for group creation and management"""

    @pytest.mark.asyncio
    async def test_create_group(self, enrollment_with_speakers):
        """Should create group with speakers"""
        result = await enrollment_with_speakers.process(
            "crea un grupo llamado área social con cocina y sala"
        )

        assert result["success"] is True
        assert result["action"] == "group_created"
        assert "área_social" in enrollment_with_speakers.speakers.groups

    @pytest.mark.asyncio
    async def test_create_group_partial_speakers(self, enrollment_with_speakers):
        """Should create group with valid speakers only"""
        result = await enrollment_with_speakers.process(
            "crea un grupo llamado test con cocina, sala y nonexistent"
        )

        assert result["success"] is True
        group = enrollment_with_speakers.speakers.groups.get("test")
        assert group is not None
        assert len(group.speaker_ids) == 2  # Only cocina and sala

    @pytest.mark.asyncio
    async def test_add_speaker_to_group(self, enrollment_with_speakers):
        """Should add speaker to existing group"""
        # First create a group
        enrollment_with_speakers.speakers.create_group(
            id="social",
            name="Social",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina"]
        )

        result = await enrollment_with_speakers.process("agrega la sala al grupo social")

        assert result["success"] is True
        assert "sala" in enrollment_with_speakers.speakers.groups["social"].speaker_ids

    @pytest.mark.asyncio
    async def test_remove_speaker_from_group(self, enrollment_with_speakers):
        """Should remove speaker from group"""
        enrollment_with_speakers.speakers.create_group(
            id="social",
            name="Social",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina", "sala"]
        )

        result = await enrollment_with_speakers.process("quita la cocina del grupo social")

        assert result["success"] is True
        assert "cocina" not in enrollment_with_speakers.speakers.groups["social"].speaker_ids

    @pytest.mark.asyncio
    async def test_delete_group_requires_confirmation(self, enrollment_with_speakers):
        """Should require confirmation to delete group"""
        enrollment_with_speakers.speakers.create_group(
            id="social",
            name="Social",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina"]
        )

        result = await enrollment_with_speakers.process("elimina el grupo social")

        assert result["success"] is True
        assert result.get("needs_confirmation") is True
        assert enrollment_with_speakers._pending_action is not None

    @pytest.mark.asyncio
    async def test_delete_group_after_confirmation(self, enrollment_with_speakers):
        """Should delete group after confirmation"""
        enrollment_with_speakers.speakers.create_group(
            id="social",
            name="Social",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina"]
        )

        # Request delete
        await enrollment_with_speakers.process("elimina el grupo social")

        # Confirm
        result = await enrollment_with_speakers.process("sí")

        assert result["success"] is True
        assert result["action"] == "group_deleted"
        assert "social" not in enrollment_with_speakers.speakers.groups

    @pytest.mark.asyncio
    async def test_cannot_delete_everywhere_group(self, enrollment_with_speakers):
        """Should not allow deleting everywhere group"""
        result = await enrollment_with_speakers.process("elimina el grupo toda la casa")

        assert result["success"] is False


# ============================================================
# Device Discovery Tests
# ============================================================

class TestDeviceDiscovery:
    """Tests for Spotify device discovery"""

    @pytest.mark.asyncio
    async def test_discover_new_devices(self, enrollment, mock_spotify_client):
        """Should discover new devices"""
        result = await enrollment.process("busca bocinas nuevas")

        assert result["success"] is True
        assert result["action"] in ["discover_found", "discover_one"]
        assert len(enrollment._pending_devices) > 0

    @pytest.mark.asyncio
    async def test_discover_no_devices(self, enrollment, mock_spotify_client):
        """Should handle no devices found"""
        mock_spotify_client.get_devices.return_value = []

        result = await enrollment.process("busca bocinas nuevas")

        assert result["success"] is False
        assert "no encontré" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_discover_all_configured(self, enrollment_with_speakers, mock_spotify_client):
        """Should handle when all devices are configured"""
        mock_spotify_client.get_devices.return_value = [
            MockSpotifyDevice(id="dev_1", name="Kitchen"),  # Already configured
            MockSpotifyDevice(id="dev_2", name="Living"),   # Already configured
        ]

        result = await enrollment_with_speakers.process("busca bocinas nuevas")

        assert result["success"] is True
        assert "ya están configuradas" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_suggest_name_from_device(self, enrollment):
        """Should suggest name based on device name"""
        suggested = enrollment._suggest_name("Echo Kitchen Speaker")
        assert suggested == "cocina"

        suggested = enrollment._suggest_name("Living Room Echo")
        assert suggested == "sala"

        suggested = enrollment._suggest_name("Random Device")
        assert suggested is None


# ============================================================
# Listing Tests
# ============================================================

class TestListingCommands:
    """Tests for list commands"""

    @pytest.mark.asyncio
    async def test_list_speakers(self, enrollment_with_speakers):
        """Should list all speakers"""
        result = await enrollment_with_speakers.process("qué bocinas tengo")

        assert result["success"] is True
        assert result["count"] == 2
        assert "cocina" in result["response"].lower()
        assert "sala" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_list_speakers_empty(self, enrollment):
        """Should handle empty speaker list"""
        result = await enrollment.process("qué bocinas tengo")

        assert result["success"] is True
        assert "no tienes" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_list_groups(self, enrollment_with_speakers):
        """Should list custom groups"""
        enrollment_with_speakers.speakers.create_group(
            id="social",
            name="Social",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina"]
        )

        result = await enrollment_with_speakers.process("qué grupos tengo")

        assert result["success"] is True
        assert "social" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_list_pending_devices(self, enrollment):
        """Should list pending devices"""
        enrollment._pending_devices["dev_1"] = PendingDevice(
            spotify_device=MockSpotifyDevice(id="dev_1", name="Test Speaker"),
            discovered_at=1234567890.0
        )

        result = await enrollment.process("hay bocinas sin configurar")

        assert result["success"] is True
        assert result["count"] == 1


# ============================================================
# Alias Tests
# ============================================================

class TestAliasManagement:
    """Tests for alias management"""

    @pytest.mark.asyncio
    async def test_add_alias(self, enrollment_with_speakers):
        """Should add alias to speaker"""
        result = await enrollment_with_speakers.process(
            "la cocina también se llama kitchen"
        )

        assert result["success"] is True
        speaker = enrollment_with_speakers.speakers.speakers["cocina"]
        assert "kitchen" in speaker.aliases

    @pytest.mark.asyncio
    async def test_add_alias_nonexistent_speaker(self, enrollment_with_speakers):
        """Should fail for nonexistent speaker"""
        result = await enrollment_with_speakers.process(
            "la garage también se llama garaje"
        )

        assert result["success"] is False


# ============================================================
# Default Speaker Tests
# ============================================================

class TestDefaultSpeaker:
    """Tests for default speaker management"""

    @pytest.mark.asyncio
    async def test_set_default_speaker(self, enrollment_with_speakers):
        """Should set speaker as default"""
        result = await enrollment_with_speakers.process("la cocina es la bocina principal")

        assert result["success"] is True
        assert enrollment_with_speakers.speakers.speakers["cocina"].is_default is True
        assert enrollment_with_speakers.speakers.speakers["sala"].is_default is False


# ============================================================
# Forget Speaker Tests
# ============================================================

class TestForgetSpeaker:
    """Tests for forgetting speakers"""

    @pytest.mark.asyncio
    async def test_forget_requires_confirmation(self, enrollment_with_speakers):
        """Should require confirmation to forget"""
        result = await enrollment_with_speakers.process("olvida la bocina cocina")

        assert result["success"] is True
        assert result.get("needs_confirmation") is True

    @pytest.mark.asyncio
    async def test_forget_after_confirmation(self, enrollment_with_speakers):
        """Should forget speaker after confirmation"""
        await enrollment_with_speakers.process("olvida la bocina cocina")
        result = await enrollment_with_speakers.process("sí")

        assert result["success"] is True
        assert result["action"] == "speaker_forgotten"
        assert "cocina" not in enrollment_with_speakers.speakers.speakers


# ============================================================
# Confirmation Flow Tests
# ============================================================

class TestConfirmationFlow:
    """Tests for confirmation flows"""

    @pytest.mark.asyncio
    async def test_cancel_pending_action(self, enrollment_with_speakers):
        """Should cancel pending action"""
        enrollment_with_speakers.speakers.create_group(
            id="test",
            name="Test",
            group_type=GroupType.ZONE,
            speaker_ids=["cocina"]
        )

        await enrollment_with_speakers.process("elimina el grupo test")
        result = await enrollment_with_speakers.process("no, cancela")

        assert result["success"] is True
        assert result["action"] == "cancelled"
        assert "test" in enrollment_with_speakers.speakers.groups

    @pytest.mark.asyncio
    async def test_confirm_nothing_pending(self, enrollment):
        """Should handle confirm with nothing pending"""
        # Without a pending action, "sí" is detected as UNKNOWN
        result = await enrollment.process("sí")

        # The UNKNOWN intent returns the generic "no entendí" message
        assert result["success"] is False
        assert result["action"] == "unknown"

    @pytest.mark.asyncio
    async def test_confirm_suggested_name(self, enrollment, mock_spotify_client):
        """Should accept suggested name on confirm"""
        # Discover devices first
        await enrollment.process("busca bocinas nuevas")

        # If there's a suggested name, confirm it
        if enrollment._pending_action and enrollment._pending_action.get("type") == "name_suggested":
            result = await enrollment.process("sí")
            assert result["success"] is True


# ============================================================
# Auto-Discovery Tests
# ============================================================

class TestAutoDiscovery:
    """Tests for background auto-discovery"""

    @pytest.mark.asyncio
    async def test_check_new_devices(self, enrollment_with_speakers, mock_spotify_client):
        """Should detect new devices in background"""
        # Add a new device not in configured list
        mock_spotify_client.get_devices.return_value = [
            MockSpotifyDevice(id="dev_1", name="Kitchen"),  # Configured
            MockSpotifyDevice(id="dev_2", name="Living"),   # Configured
            MockSpotifyDevice(id="dev_new", name="New Speaker"),  # New!
        ]

        message = await enrollment_with_speakers.check_new_devices()

        assert message is not None
        assert "bocina nueva" in message.lower()

    @pytest.mark.asyncio
    async def test_check_no_new_devices(self, enrollment_with_speakers, mock_spotify_client):
        """Should return None if no new devices"""
        mock_spotify_client.get_devices.return_value = [
            MockSpotifyDevice(id="dev_1", name="Kitchen"),
            MockSpotifyDevice(id="dev_2", name="Living"),
        ]

        message = await enrollment_with_speakers.check_new_devices()

        assert message is None

    @pytest.mark.asyncio
    async def test_check_handles_api_error(self, enrollment, mock_spotify_client):
        """Should handle API errors gracefully"""
        mock_spotify_client.get_devices.side_effect = Exception("API Error")

        message = await enrollment.check_new_devices()

        assert message is None


# ============================================================
# Persistence Tests
# ============================================================

class TestPersistence:
    """Tests for configuration persistence"""

    @pytest.mark.asyncio
    async def test_auto_save_on_change(self, mock_spotify_client, manager_with_speakers):
        """Should auto-save when enabled"""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "speakers.json"

            enrollment = SpeakerEnrollment(
                spotify_client=mock_spotify_client,
                speaker_manager=manager_with_speakers,
                auto_save=True,
                config_path=config_path
            )

            await enrollment.process("la cocina está en la planta alta")

            assert config_path.exists()

    @pytest.mark.asyncio
    async def test_no_save_when_disabled(self, mock_spotify_client, manager_with_speakers):
        """Should not save when auto_save is False"""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "speakers.json"

            enrollment = SpeakerEnrollment(
                spotify_client=mock_spotify_client,
                speaker_manager=manager_with_speakers,
                auto_save=False,
                config_path=config_path
            )

            await enrollment.process("la cocina está en la planta alta")

            assert not config_path.exists()


# ============================================================
# Context Memory Tests
# ============================================================

class TestContextMemory:
    """Tests for conversation context memory"""

    @pytest.mark.asyncio
    async def test_remembers_last_mentioned_speaker(self, enrollment_with_speakers):
        """Should remember last mentioned speaker"""
        # First mention cocina
        await enrollment_with_speakers.process("la cocina está en el comedor")

        # Now set floor without specifying speaker
        result = await enrollment_with_speakers.process("está en la planta alta")

        # Should apply to cocina (last mentioned)
        assert enrollment_with_speakers.speakers.speakers["cocina"].floor == "planta_alta"
