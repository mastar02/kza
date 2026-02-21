"""
Tests for Speaker Groups - Multi-room audio management

Tests cover:
1. Speaker management (add, remove, lookup)
2. Group creation and management
3. Zone resolution from voice commands
4. Voice command parsing
5. Configuration persistence
"""

import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.spotify.speaker_groups import (
    Speaker, SpeakerGroup, SpeakerGroupManager, GroupType
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_speakers():
    """Sample speakers for testing"""
    return [
        Speaker(
            id="kitchen",
            name="Bocina Cocina",
            spotify_device_id="spotify_kitchen_123",
            room="cocina",
            floor="planta_baja",
            aliases=["cocina", "kitchen"]
        ),
        Speaker(
            id="living",
            name="Bocina Sala",
            spotify_device_id="spotify_living_456",
            room="sala",
            floor="planta_baja",
            is_default=True,
            aliases=["living", "sala de estar"]
        ),
        Speaker(
            id="bedroom",
            name="Bocina Dormitorio",
            spotify_device_id="spotify_bedroom_789",
            room="dormitorio",
            floor="planta_alta",
            aliases=["cuarto", "habitación", "bedroom"]
        ),
        Speaker(
            id="bathroom",
            name="Bocina Baño",
            spotify_device_id=None,  # No tiene Spotify
            room="baño",
            floor="planta_alta",
            supports_spotify=False
        ),
    ]


@pytest.fixture
def speaker_manager(sample_speakers):
    """SpeakerGroupManager with sample speakers"""
    manager = SpeakerGroupManager()
    for speaker in sample_speakers:
        manager.add_speaker(speaker)
    return manager


@pytest.fixture
def manager_with_groups(speaker_manager):
    """Manager with predefined groups"""
    speaker_manager.create_group(
        id="downstairs",
        name="Planta Baja",
        group_type=GroupType.FLOOR,
        speaker_ids=["kitchen", "living"],
        aliases=["abajo", "primer piso"]
    )
    speaker_manager.create_group(
        id="upstairs",
        name="Planta Alta",
        group_type=GroupType.FLOOR,
        speaker_ids=["bedroom", "bathroom"],
        aliases=["arriba", "segundo piso"]
    )
    return speaker_manager


# ============================================================
# Speaker Management Tests
# ============================================================

class TestSpeakerManagement:
    """Tests for individual speaker management"""

    def test_add_speaker(self, speaker_manager):
        """Should add speaker correctly"""
        new_speaker = Speaker(
            id="garage",
            name="Bocina Garage",
            room="garage"
        )
        speaker_manager.add_speaker(new_speaker)
        assert "garage" in speaker_manager.speakers
        assert speaker_manager.get_speaker("garage").name == "Bocina Garage"

    def test_remove_speaker(self, speaker_manager):
        """Should remove speaker and update groups"""
        speaker_manager.remove_speaker("kitchen")
        assert "kitchen" not in speaker_manager.speakers
        # Should be removed from "everywhere" group
        everywhere = speaker_manager.get_group("everywhere")
        assert "kitchen" not in everywhere.speaker_ids

    def test_get_speaker_by_id(self, speaker_manager):
        """Should get speaker by ID"""
        speaker = speaker_manager.get_speaker("living")
        assert speaker is not None
        assert speaker.name == "Bocina Sala"

    def test_get_speaker_by_spotify_id(self, speaker_manager):
        """Should get speaker by Spotify device ID"""
        speaker = speaker_manager.get_speaker_by_spotify_id("spotify_kitchen_123")
        assert speaker is not None
        assert speaker.id == "kitchen"

    def test_get_speakers_by_room(self, speaker_manager):
        """Should get all speakers in a room"""
        kitchen_speakers = speaker_manager.get_speakers_by_room("cocina")
        assert len(kitchen_speakers) == 1
        assert kitchen_speakers[0].id == "kitchen"

    def test_get_speakers_by_floor(self, speaker_manager):
        """Should get all speakers on a floor"""
        downstairs = speaker_manager.get_speakers_by_floor("planta_baja")
        assert len(downstairs) == 2
        assert set(s.id for s in downstairs) == {"kitchen", "living"}

    def test_get_default_speaker(self, speaker_manager):
        """Should return default speaker"""
        default = speaker_manager.get_default_speaker()
        assert default is not None
        assert default.is_default is True
        assert default.id == "living"

    def test_update_spotify_device_id(self, speaker_manager):
        """Should update Spotify device ID"""
        speaker_manager.update_spotify_device_id("bathroom", "new_spotify_id")
        speaker = speaker_manager.get_speaker("bathroom")
        assert speaker.spotify_device_id == "new_spotify_id"


# ============================================================
# Speaker Matching Tests
# ============================================================

class TestSpeakerMatching:
    """Tests for speaker name matching"""

    def test_matches_direct_name(self, sample_speakers):
        """Should match by direct name"""
        kitchen = sample_speakers[0]
        assert kitchen.matches_name("cocina") is True
        assert kitchen.matches_name("Cocina") is True  # Case insensitive

    def test_matches_room_name(self, sample_speakers):
        """Should match by room name"""
        kitchen = sample_speakers[0]
        assert kitchen.matches_name("cocina") is True

    def test_matches_alias(self, sample_speakers):
        """Should match by alias"""
        bedroom = sample_speakers[2]
        assert bedroom.matches_name("cuarto") is True
        assert bedroom.matches_name("habitación") is True

    def test_no_match_wrong_name(self, sample_speakers):
        """Should not match wrong name"""
        kitchen = sample_speakers[0]
        assert kitchen.matches_name("dormitorio") is False


# ============================================================
# Group Management Tests
# ============================================================

class TestGroupManagement:
    """Tests for speaker group management"""

    def test_create_group(self, speaker_manager):
        """Should create group correctly"""
        group = speaker_manager.create_group(
            id="social",
            name="Área Social",
            group_type=GroupType.ZONE,
            speaker_ids=["kitchen", "living"]
        )
        assert group.id == "social"
        assert len(group.speaker_ids) == 2

    def test_create_group_validates_speakers(self, speaker_manager):
        """Should only include valid speakers"""
        group = speaker_manager.create_group(
            id="mixed",
            name="Mixed",
            group_type=GroupType.ZONE,
            speaker_ids=["kitchen", "nonexistent", "living"]
        )
        assert len(group.speaker_ids) == 2
        assert "nonexistent" not in group.speaker_ids

    def test_everywhere_group_auto_created(self, speaker_manager):
        """Should auto-create 'everywhere' group"""
        everywhere = speaker_manager.get_group("everywhere")
        assert everywhere is not None
        assert everywhere.group_type == GroupType.EVERYWHERE

    def test_everywhere_group_includes_all_speakers(self, speaker_manager):
        """Everywhere group should include all speakers"""
        everywhere = speaker_manager.get_group("everywhere")
        assert len(everywhere.speaker_ids) == len(speaker_manager.speakers)

    def test_get_group_speakers(self, manager_with_groups):
        """Should get speakers from group"""
        speakers = manager_with_groups.get_group_speakers("downstairs")
        assert len(speakers) == 2
        assert set(s.id for s in speakers) == {"kitchen", "living"}

    def test_add_speaker_to_group(self, manager_with_groups):
        """Should add speaker to existing group"""
        manager_with_groups.add_speaker_to_group("bedroom", "downstairs")
        group = manager_with_groups.get_group("downstairs")
        assert "bedroom" in group.speaker_ids

    def test_remove_speaker_from_group(self, manager_with_groups):
        """Should remove speaker from group"""
        manager_with_groups.remove_speaker_from_group("kitchen", "downstairs")
        group = manager_with_groups.get_group("downstairs")
        assert "kitchen" not in group.speaker_ids


# ============================================================
# Zone Resolution Tests
# ============================================================

class TestZoneResolution:
    """Tests for resolving voice commands to zones"""

    def test_resolve_single_speaker(self, speaker_manager):
        """Should resolve to single speaker"""
        target = speaker_manager.resolve_target("cocina")
        assert target is not None
        assert target["type"] == "speaker"
        assert target["speaker"].id == "kitchen"

    def test_resolve_group(self, manager_with_groups):
        """Should resolve to group"""
        target = manager_with_groups.resolve_target("planta baja")
        assert target is not None
        assert target["type"] == "group"
        assert len(target["speakers"]) == 2

    def test_resolve_everywhere(self, speaker_manager):
        """Should resolve 'toda la casa'"""
        everywhere_queries = [
            "toda la casa",
            "everywhere",
            "todas partes",
            "todos los cuartos"
        ]
        for query in everywhere_queries:
            target = speaker_manager.resolve_target(query)
            assert target is not None, f"Failed for: {query}"
            assert target["type"] == "group"
            assert len(target["speakers"]) == len(speaker_manager.speakers)

    def test_resolve_by_alias(self, manager_with_groups):
        """Should resolve by alias"""
        target = manager_with_groups.resolve_target("abajo")  # alias for downstairs
        assert target is not None
        assert target["type"] == "group"

    def test_resolve_by_room(self, speaker_manager):
        """Should resolve by room name"""
        target = speaker_manager.resolve_target("sala")
        assert target is not None
        assert target["speakers"][0].room == "sala"

    def test_resolve_returns_spotify_ids(self, speaker_manager):
        """Should return Spotify device IDs"""
        target = speaker_manager.resolve_target("cocina")
        assert "spotify_device_ids" in target
        assert "spotify_kitchen_123" in target["spotify_device_ids"]

    def test_resolve_excludes_no_spotify(self, speaker_manager):
        """Should handle speakers without Spotify"""
        target = speaker_manager.resolve_target("baño")
        assert target is not None
        assert len(target["spotify_device_ids"]) == 0

    def test_resolve_unknown_returns_none(self, speaker_manager):
        """Should return None for unknown zone"""
        target = speaker_manager.resolve_target("garaje inexistente")
        assert target is None

    def test_resolve_default(self, speaker_manager):
        """Should resolve default speaker"""
        target = speaker_manager.resolve_default()
        assert target is not None
        assert target["speaker"].is_default is True


# ============================================================
# Voice Command Parsing Tests
# ============================================================

class TestVoiceCommandParsing:
    """Tests for parsing zones from voice commands"""

    def test_parse_zone_from_command(self, speaker_manager):
        """Should extract zone from command"""
        target, cleaned = speaker_manager.parse_zone_from_command(
            "pon música de Bad Bunny en la cocina"
        )
        assert target is not None
        assert target["speaker"].id == "kitchen"
        assert "cocina" not in cleaned.lower()
        assert "bad bunny" in cleaned.lower()

    def test_parse_zone_with_el(self, speaker_manager):
        """Should handle 'en el' pattern"""
        target, cleaned = speaker_manager.parse_zone_from_command(
            "reproduce jazz en el living"
        )
        assert target is not None

    def test_parse_zone_toda_la_casa(self, speaker_manager):
        """Should detect 'toda la casa'"""
        target, cleaned = speaker_manager.parse_zone_from_command(
            "pon música en toda la casa"
        )
        assert target is not None
        assert target["type"] == "group"

    def test_parse_no_zone(self, speaker_manager):
        """Should return None when no zone specified"""
        target, cleaned = speaker_manager.parse_zone_from_command(
            "pon música de Bad Bunny"
        )
        assert target is None
        assert cleaned == "pon música de Bad Bunny"

    def test_parse_removes_por_favor(self, speaker_manager):
        """Should clean 'por favor' from zone"""
        target, cleaned = speaker_manager.parse_zone_from_command(
            "pon jazz en la cocina por favor"
        )
        assert target is not None
        assert target["speaker"].id == "kitchen"


# ============================================================
# Configuration Persistence Tests
# ============================================================

class TestConfigPersistence:
    """Tests for saving/loading configuration"""

    def test_save_and_load_config(self, manager_with_groups):
        """Should save and load configuration"""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "speakers.json"

            # Save
            manager_with_groups.save_config(config_path)
            assert config_path.exists()

            # Load into new manager
            new_manager = SpeakerGroupManager(config_path=config_path)

            # Verify speakers
            assert len(new_manager.speakers) == len(manager_with_groups.speakers)
            assert "kitchen" in new_manager.speakers

            # Verify groups (excluding auto-generated 'everywhere')
            assert "downstairs" in new_manager.groups
            assert "upstairs" in new_manager.groups

    def test_config_includes_all_speaker_fields(self, speaker_manager):
        """Should persist all speaker fields"""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "speakers.json"
            speaker_manager.save_config(config_path)

            with open(config_path) as f:
                config = json.load(f)

            kitchen = config["speakers"]["kitchen"]
            assert kitchen["spotify_device_id"] == "spotify_kitchen_123"
            assert kitchen["room"] == "cocina"
            assert "cocina" in kitchen["aliases"]


# ============================================================
# Status and Utility Tests
# ============================================================

class TestStatusAndUtilities:
    """Tests for status reporting and utilities"""

    def test_get_status(self, manager_with_groups):
        """Should return correct status"""
        status = manager_with_groups.get_status()
        assert "speakers" in status
        assert "groups" in status
        assert "default_speaker" in status
        assert len(status["speakers"]) == 4

    def test_list_available_targets(self, manager_with_groups):
        """Should list all available targets"""
        targets = manager_with_groups.list_available_targets()
        assert len(targets) > 0
        # Should include speaker names
        assert any("cocina" in t.lower() for t in targets)
        # Should include group names
        assert any("planta" in t.lower() for t in targets)
