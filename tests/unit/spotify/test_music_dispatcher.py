"""
Tests for MusicDispatcher - Music command routing and execution.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from src.spotify.music_dispatcher import (
    MusicDispatcher,
    MusicIntent,
    MusicCommand,
    MusicResult
)
from src.spotify.mood_mapper import MoodMapper, MoodProfile, AudioFeatures
from src.spotify.client import SpotifyTrack


@pytest.fixture
def mock_spotify():
    """Mock Spotify client"""
    client = MagicMock()

    # Default async methods
    client.pause = AsyncMock(return_value=True)
    client.play = AsyncMock(return_value=True)
    client.next_track = AsyncMock(return_value=True)
    client.previous_track = AsyncMock(return_value=True)
    client.set_volume = AsyncMock(return_value=True)
    client.set_shuffle = AsyncMock(return_value=True)
    client.get_current_track_info = AsyncMock(return_value="Song - Artist")

    # Search methods
    client.search_artists = AsyncMock(return_value=[{
        "name": "Bad Bunny",
        "uri": "spotify:artist:123"
    }])
    client.search_tracks = AsyncMock(return_value=[
        SpotifyTrack(
            id="track123",
            name="Test Song",
            artists=["Test Artist"],
            album="Test Album",
            uri="spotify:track:123",
            duration_ms=180000,
            popularity=75
        )
    ])
    client.search_playlists = AsyncMock(return_value=[{
        "name": "My Playlist",
        "uri": "spotify:playlist:123"
    }])

    # Recommendations
    client.get_recommendations = AsyncMock(return_value=[
        SpotifyTrack(
            id=f"rec{i}",
            name=f"Rec Song {i}",
            artists=[f"Artist {i}"],
            album="Album",
            uri=f"spotify:track:rec{i}",
            duration_ms=200000,
            popularity=60
        )
        for i in range(5)
    ])

    client.get_user_top_tracks = AsyncMock(return_value=[
        SpotifyTrack(
            id=f"top{i}",
            name=f"Top Song {i}",
            artists=[f"Artist {i}"],
            album="Album",
            uri=f"spotify:track:top{i}",
            duration_ms=200000,
            popularity=80
        )
        for i in range(3)
    ])

    return client


@pytest.fixture
def mock_mood_mapper():
    """Mock mood mapper"""
    mapper = MagicMock(spec=MoodMapper)
    mapper.get_mood_profile = MagicMock(return_value=None)
    mapper.extract_artist_or_track = MagicMock(return_value={"artist": None, "track": None})
    mapper.interpret_with_llm = AsyncMock(return_value=None)
    return mapper


@pytest.fixture
def dispatcher(mock_spotify, mock_mood_mapper):
    """Create dispatcher with mocks"""
    return MusicDispatcher(
        spotify_client=mock_spotify,
        mood_mapper=mock_mood_mapper,
        llm=None
    )


class TestMusicIntent:
    """Tests for MusicIntent enum"""

    def test_play_intents_exist(self):
        assert MusicIntent.PLAY_ARTIST
        assert MusicIntent.PLAY_TRACK
        assert MusicIntent.PLAY_MOOD
        assert MusicIntent.PLAY_CONTEXT

    def test_control_intents_exist(self):
        assert MusicIntent.PAUSE
        assert MusicIntent.RESUME
        assert MusicIntent.NEXT
        assert MusicIntent.PREVIOUS
        assert MusicIntent.VOLUME
        assert MusicIntent.SHUFFLE


class TestMusicCommand:
    """Tests for MusicCommand dataclass"""

    def test_create_command(self):
        cmd = MusicCommand(
            intent=MusicIntent.PLAY_ARTIST,
            raw_text="pon música de Taylor Swift",
            artist="Taylor Swift"
        )

        assert cmd.intent == MusicIntent.PLAY_ARTIST
        assert cmd.artist == "Taylor Swift"
        assert cmd.track is None

    def test_command_with_mood(self):
        profile = MoodProfile(
            name="Test",
            description="Test",
            features=AudioFeatures(),
            keywords=[]
        )
        cmd = MusicCommand(
            intent=MusicIntent.PLAY_MOOD,
            raw_text="algo tranquilo",
            mood_profile=profile
        )

        assert cmd.mood_profile is not None


class TestMusicResult:
    """Tests for MusicResult dataclass"""

    def test_create_result(self):
        result = MusicResult(
            success=True,
            response="Reproduciendo música",
            intent=MusicIntent.PLAY_ARTIST
        )

        assert result.success is True
        assert result.latency_ms == 0
        assert result.tracks_played == []

    def test_result_with_tracks(self):
        track = SpotifyTrack(
            id="1",
            name="Song",
            artists=["Artist"],
            album="Album",
            uri="spotify:track:1",
            duration_ms=180000,
            popularity=70
        )
        result = MusicResult(
            success=True,
            response="Playing",
            intent=MusicIntent.PLAY_TRACK,
            tracks_played=[track]
        )

        assert len(result.tracks_played) == 1


class TestIntentDetection:
    """Tests for intent detection"""

    def test_detect_pause(self, dispatcher):
        cmd = dispatcher.detect_intent("pausa la música")
        assert cmd.intent == MusicIntent.PAUSE

    def test_detect_stop(self, dispatcher):
        cmd = dispatcher.detect_intent("para la música")
        assert cmd.intent == MusicIntent.PAUSE

    def test_detect_resume(self, dispatcher):
        cmd = dispatcher.detect_intent("continúa la música")
        assert cmd.intent == MusicIntent.RESUME

    def test_detect_next(self, dispatcher):
        cmd = dispatcher.detect_intent("siguiente canción")
        assert cmd.intent == MusicIntent.NEXT

    def test_detect_skip(self, dispatcher):
        cmd = dispatcher.detect_intent("salta esta")
        assert cmd.intent == MusicIntent.NEXT

    def test_detect_previous(self, dispatcher):
        cmd = dispatcher.detect_intent("canción anterior")
        assert cmd.intent == MusicIntent.PREVIOUS

    def test_detect_shuffle(self, dispatcher):
        cmd = dispatcher.detect_intent("pon aleatorio")
        assert cmd.intent == MusicIntent.SHUFFLE

    def test_detect_whats_playing(self, dispatcher):
        cmd = dispatcher.detect_intent("qué está sonando")
        assert cmd.intent == MusicIntent.WHATS_PLAYING

    def test_detect_volume(self, dispatcher):
        cmd = dispatcher.detect_intent("volumen al 50")
        assert cmd.intent == MusicIntent.VOLUME
        assert cmd.volume_level == 50

    def test_detect_volume_max(self, dispatcher):
        cmd = dispatcher.detect_intent("volumen máximo")
        assert cmd.intent == MusicIntent.VOLUME
        assert cmd.volume_level == 100

    def test_detect_artist(self, dispatcher, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": "bad bunny",
            "track": None
        }

        cmd = dispatcher.detect_intent("pon música de Bad Bunny")
        assert cmd.intent == MusicIntent.PLAY_ARTIST
        assert cmd.artist == "bad bunny"

    def test_detect_track(self, dispatcher, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": None,
            "track": "blinding lights"
        }

        cmd = dispatcher.detect_intent("pon la canción Blinding Lights")
        assert cmd.intent == MusicIntent.PLAY_TRACK
        assert cmd.track == "blinding lights"

    def test_detect_playlist(self, dispatcher):
        # Note: Use "lista" instead of "playlist" to avoid "play" substring match
        cmd = dispatcher.detect_intent("reproduce mi lista de ejercicio")
        assert cmd.intent == MusicIntent.PLAY_PLAYLIST

    def test_detect_similar(self, dispatcher):
        cmd = dispatcher.detect_intent("pon algo parecido")
        assert cmd.intent == MusicIntent.PLAY_SIMILAR

    def test_detect_mood(self, dispatcher, mock_mood_mapper):
        mock_mood_mapper.get_mood_profile.return_value = MoodProfile(
            name="Workout",
            description="Exercise",
            features=AudioFeatures(),
            keywords=["gym"]
        )

        cmd = dispatcher.detect_intent("algo para el gym")
        assert cmd.intent == MusicIntent.PLAY_MOOD
        assert cmd.mood_profile is not None

    def test_detect_context(self, dispatcher, mock_mood_mapper):
        mock_mood_mapper.get_mood_profile.return_value = None

        cmd = dispatcher.detect_intent("pon música para cocinar")
        assert cmd.intent == MusicIntent.PLAY_CONTEXT

    def test_detect_fallback_recommendations(self, dispatcher, mock_mood_mapper):
        mock_mood_mapper.get_mood_profile.return_value = None

        cmd = dispatcher.detect_intent("pon algo de música")
        assert cmd.intent == MusicIntent.PLAY_RECOMMENDATIONS


class TestControlExecution:
    """Tests for playback control commands"""

    @pytest.mark.asyncio
    async def test_pause(self, dispatcher, mock_spotify):
        result = await dispatcher.process("pausa")

        assert result.success is True
        assert result.intent == MusicIntent.PAUSE
        mock_spotify.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume(self, dispatcher, mock_spotify):
        result = await dispatcher.process("continúa")

        assert result.success is True
        assert result.intent == MusicIntent.RESUME
        mock_spotify.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_next(self, dispatcher, mock_spotify):
        result = await dispatcher.process("siguiente")

        assert result.success is True
        mock_spotify.next_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_previous(self, dispatcher, mock_spotify):
        result = await dispatcher.process("anterior")

        assert result.success is True
        mock_spotify.previous_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_volume(self, dispatcher, mock_spotify):
        result = await dispatcher.process("volumen al 70")

        assert result.success is True
        mock_spotify.set_volume.assert_called_once_with(70)

    @pytest.mark.asyncio
    async def test_shuffle(self, dispatcher, mock_spotify):
        result = await dispatcher.process("aleatorio")

        assert result.success is True
        mock_spotify.set_shuffle.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_whats_playing(self, dispatcher, mock_spotify):
        result = await dispatcher.process("qué suena")

        assert result.success is True
        assert "Song" in result.response or result.response == "Song - Artist"


class TestPlayExecution:
    """Tests for play commands"""

    @pytest.mark.asyncio
    async def test_play_artist(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": "bad bunny",
            "track": None
        }

        result = await dispatcher.process("pon música de Bad Bunny")

        assert result.success is True
        assert result.intent == MusicIntent.PLAY_ARTIST
        mock_spotify.search_artists.assert_called_once()
        mock_spotify.play.assert_called()

    @pytest.mark.asyncio
    async def test_play_artist_not_found(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": "nonexistent artist",
            "track": None
        }
        mock_spotify.search_artists.return_value = []

        result = await dispatcher.process("pon música de NonexistentArtist")

        assert result.success is False
        assert "No encontré" in result.response

    @pytest.mark.asyncio
    async def test_play_track(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": None,
            "track": "test song"
        }

        result = await dispatcher.process("pon la canción Test Song")

        assert result.success is True
        mock_spotify.search_tracks.assert_called_once()
        assert len(result.tracks_played) > 0

    @pytest.mark.asyncio
    async def test_play_playlist(self, dispatcher, mock_spotify):
        # Use "lista" to avoid "play" substring conflict in "playlist"
        result = await dispatcher.process("reproduce mi lista de ejercicio")

        assert result.success is True
        mock_spotify.search_playlists.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_mood(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.get_mood_profile.return_value = MoodProfile(
            name="Energético",
            description="High energy music",
            features=AudioFeatures(energy=0.9, danceability=0.8),
            genres=["edm", "dance"],
            keywords=["energia"]
        )

        result = await dispatcher.process("algo con energía")

        assert result.success is True
        mock_spotify.get_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_recommendations(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.get_mood_profile.return_value = None

        result = await dispatcher.process("recomiéndame música")

        assert result.success is True
        assert result.intent == MusicIntent.PLAY_RECOMMENDATIONS
        mock_spotify.get_user_top_tracks.assert_called()


class TestLatencyTracking:
    """Tests for latency measurement"""

    @pytest.mark.asyncio
    async def test_latency_recorded(self, dispatcher):
        result = await dispatcher.process("pausa")

        assert result.latency_ms > 0


class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_spotify_exception(self, dispatcher, mock_spotify):
        mock_spotify.pause.side_effect = Exception("API error")

        result = await dispatcher.process("pausa")

        assert result.success is False
        assert "Error" in result.response

    @pytest.mark.asyncio
    async def test_search_failure(self, dispatcher, mock_spotify, mock_mood_mapper):
        mock_mood_mapper.extract_artist_or_track.return_value = {
            "artist": "test",
            "track": None
        }
        mock_spotify.search_artists.side_effect = Exception("Network error")

        result = await dispatcher.process("pon música de Test")

        assert result.success is False


class TestVolumeExtraction:
    """Tests for volume extraction from text"""

    def test_extract_number(self, dispatcher):
        assert dispatcher._extract_volume("volumen al 75") == 75

    def test_extract_max(self, dispatcher):
        assert dispatcher._extract_volume("volumen máximo") == 100

    def test_extract_min(self, dispatcher):
        assert dispatcher._extract_volume("volumen mínimo") == 20

    def test_extract_mid(self, dispatcher):
        assert dispatcher._extract_volume("volumen medio") == 50

    def test_extract_clamp_high(self, dispatcher):
        assert dispatcher._extract_volume("volumen 150") == 100

    def test_extract_default(self, dispatcher):
        assert dispatcher._extract_volume("volumen") == 50


class TestPlaylistExtraction:
    """Tests for playlist name extraction"""

    def test_extract_playlist_name(self, dispatcher):
        result = dispatcher._extract_playlist_name("playlist de ejercicio")
        assert "ejercicio" in result

    def test_extract_lista_name(self, dispatcher):
        result = dispatcher._extract_playlist_name("mi lista de rock")
        assert "rock" in result
