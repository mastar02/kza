"""
Tests for MoodMapper - Natural language to Spotify audio features mapping.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.spotify.mood_mapper import (
    MoodMapper,
    MoodProfile,
    AudioFeatures,
    MOOD_PROFILES
)


class TestAudioFeatures:
    """Tests for AudioFeatures dataclass"""

    def test_create_features(self):
        features = AudioFeatures(
            energy=0.8,
            valence=0.6,
            danceability=0.7
        )
        assert features.energy == 0.8
        assert features.valence == 0.6
        assert features.tempo is None

    def test_to_recommendation_params(self):
        features = AudioFeatures(
            energy=0.8,
            valence=0.6,
            tempo=120
        )
        params = features.to_recommendation_params()

        assert params["target_energy"] == 0.8
        assert params["target_valence"] == 0.6
        assert params["target_tempo"] == 120
        assert "target_danceability" not in params

    def test_to_recommendation_params_empty(self):
        features = AudioFeatures()
        params = features.to_recommendation_params()
        assert params == {}

    def test_merge_with(self):
        base = AudioFeatures(energy=0.5, valence=0.5, tempo=100)
        override = AudioFeatures(energy=0.9, danceability=0.8)

        merged = base.merge_with(override)

        assert merged.energy == 0.9  # Override
        assert merged.valence == 0.5  # Base preserved
        assert merged.danceability == 0.8  # New from override
        assert merged.tempo == 100  # Base preserved

    def test_merge_with_tempo_limits(self):
        base = AudioFeatures(min_tempo=80, max_tempo=140)
        override = AudioFeatures(min_tempo=100)

        merged = base.merge_with(override)

        assert merged.min_tempo == 100
        assert merged.max_tempo == 140


class TestMoodProfile:
    """Tests for MoodProfile dataclass"""

    def test_create_profile(self):
        features = AudioFeatures(energy=0.7, valence=0.8)
        profile = MoodProfile(
            name="Happy",
            description="Upbeat music",
            features=features,
            genres=["pop", "dance"],
            keywords=["feliz", "alegre"]
        )

        assert profile.name == "Happy"
        assert profile.shuffle is True
        assert "pop" in profile.genres

    def test_matches_query_positive(self):
        profile = MoodProfile(
            name="Workout",
            description="Exercise music",
            features=AudioFeatures(),
            keywords=["ejercicio", "gym", "entrenar"]
        )

        assert profile.matches_query("pon música para ejercicio") is True
        assert profile.matches_query("algo para el gym") is True

    def test_matches_query_negative(self):
        profile = MoodProfile(
            name="Workout",
            description="Exercise music",
            features=AudioFeatures(),
            keywords=["ejercicio", "gym"]
        )

        assert profile.matches_query("música para dormir") is False

    def test_matches_query_case_insensitive(self):
        profile = MoodProfile(
            name="Test",
            description="Test",
            features=AudioFeatures(),
            keywords=["jazz"]
        )

        assert profile.matches_query("JAZZ") is True
        assert profile.matches_query("Jazz Music") is True


class TestMoodProfiles:
    """Tests for predefined mood profiles"""

    def test_profiles_exist(self):
        assert "happy" in MOOD_PROFILES
        assert "sad" in MOOD_PROFILES
        assert "workout" in MOOD_PROFILES
        assert "focus" in MOOD_PROFILES
        assert "romantic" in MOOD_PROFILES
        assert "dinner" in MOOD_PROFILES
        assert "party" in MOOD_PROFILES

    def test_workout_profile(self):
        workout = MOOD_PROFILES["workout"]
        assert workout.features.energy == 0.9
        assert workout.features.min_tempo == 130
        assert "gym" in workout.keywords

    def test_sleep_profile(self):
        sleep = MOOD_PROFILES["sleep"]
        assert sleep.features.energy == 0.1
        assert sleep.features.max_tempo == 70
        assert "dormir" in sleep.keywords

    def test_dinner_profile(self):
        dinner = MOOD_PROFILES["dinner"]
        assert dinner.features.energy < 0.5
        assert "jazz" in dinner.genres or "bossa-nova" in dinner.genres

    def test_all_profiles_have_required_fields(self):
        for name, profile in MOOD_PROFILES.items():
            assert profile.name, f"{name} missing name"
            assert profile.description, f"{name} missing description"
            assert profile.features is not None, f"{name} missing features"
            assert len(profile.keywords) > 0, f"{name} missing keywords"


class TestMoodMapper:
    """Tests for MoodMapper"""

    def test_create_mapper(self):
        mapper = MoodMapper()
        assert len(mapper.profiles) > 0

    def test_create_mapper_with_llm(self):
        mock_llm = MagicMock()
        mapper = MoodMapper(llm=mock_llm)
        assert mapper.llm is mock_llm

    def test_get_mood_profile_happy(self):
        mapper = MoodMapper()
        profile = mapper.get_mood_profile("pon algo alegre")

        assert profile is not None
        assert profile.name == "Feliz"

    def test_get_mood_profile_workout(self):
        mapper = MoodMapper()
        profile = mapper.get_mood_profile("música para el gym")

        assert profile is not None
        assert profile.name == "Ejercicio"

    def test_get_mood_profile_no_match(self):
        mapper = MoodMapper()
        profile = mapper.get_mood_profile("xyzabc")

        assert profile is None

    def test_get_mood_profile_focus(self):
        mapper = MoodMapper()
        profile = mapper.get_mood_profile("algo para estudiar")

        assert profile is not None
        assert profile.name == "Concentración"

    def test_get_profile_by_name(self):
        mapper = MoodMapper()
        profile = mapper.get_profile_by_name("romantic")

        assert profile is not None
        assert profile.name == "Romántico"

    def test_list_available_moods(self):
        mapper = MoodMapper()
        moods = mapper.list_available_moods()

        assert "happy" in moods
        assert "workout" in moods
        assert len(moods) >= 15

    def test_add_custom_profile(self):
        mapper = MoodMapper()
        custom = MoodProfile(
            name="Custom",
            description="Test profile",
            features=AudioFeatures(energy=0.5),
            keywords=["custom_keyword"]
        )

        mapper.add_custom_profile("custom", custom)

        assert "custom" in mapper.profiles
        profile = mapper.get_mood_profile("play custom_keyword music")
        assert profile is not None
        assert profile.name == "Custom"

    def test_custom_profiles_on_init(self):
        custom = {
            "test": MoodProfile(
                name="Test",
                description="Test",
                features=AudioFeatures(),
                keywords=["test"]
            )
        }
        mapper = MoodMapper(custom_profiles=custom)

        assert "test" in mapper.profiles


class TestMoodMapperExtraction:
    """Tests for artist/track extraction"""

    def test_extract_artist(self):
        mapper = MoodMapper()
        result = mapper.extract_artist_or_track("pon música de Bad Bunny")

        assert result["artist"] == "bad bunny"
        assert result["track"] is None

    def test_extract_artist_spanish(self):
        mapper = MoodMapper()
        result = mapper.extract_artist_or_track("canciones de Taylor Swift")

        assert result["artist"] == "taylor swift"

    def test_extract_track_with_quotes(self):
        mapper = MoodMapper()
        result = mapper.extract_artist_or_track("pon la canción 'Blinding Lights'")

        assert result["track"] == "blinding lights"

    def test_extract_nothing(self):
        mapper = MoodMapper()
        # Use a command that clearly has no artist or track
        result = mapper.extract_artist_or_track("pausa la música")

        assert result["artist"] is None
        assert result["track"] is None


class TestMoodMapperLLM:
    """Tests for LLM interpretation"""

    @pytest.mark.asyncio
    async def test_interpret_no_llm(self):
        mapper = MoodMapper(llm=None)
        result = await mapper.interpret_with_llm("música para una cena romántica")

        assert result is None

    @pytest.mark.asyncio
    async def test_interpret_with_llm_success(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        {
            "mood_name": "romantic_dinner",
            "energy": 0.3,
            "valence": 0.6,
            "danceability": null,
            "acousticness": 0.7,
            "instrumentalness": 0.4,
            "tempo_bpm": 85,
            "genres": ["jazz", "bossa-nova", "soul"],
            "reasoning": "Música suave y romántica para cena"
        }
        """

        mapper = MoodMapper(llm=mock_llm)
        profile = await mapper.interpret_with_llm("música para una cena romántica a la luz de las velas")

        assert profile is not None
        assert profile.name == "romantic_dinner"
        assert profile.features.energy == 0.3
        assert profile.features.acousticness == 0.7
        assert "jazz" in profile.genres

    @pytest.mark.asyncio
    async def test_interpret_with_llm_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "This is not valid JSON"

        mapper = MoodMapper(llm=mock_llm)
        profile = await mapper.interpret_with_llm("test query")

        assert profile is None

    @pytest.mark.asyncio
    async def test_interpret_with_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        mapper = MoodMapper(llm=mock_llm)
        profile = await mapper.interpret_with_llm("test query")

        assert profile is None
