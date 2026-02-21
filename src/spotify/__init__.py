# Spotify Integration Module
# Control avanzado de Spotify con interpretación de contexto
# Incluye soporte multi-room estilo Alexa

from .client import SpotifyClient
from .auth import SpotifyAuth, TokenManager
from .mood_mapper import MoodMapper, AudioFeatures, MoodProfile
from .music_dispatcher import MusicDispatcher, MusicIntent
from .speaker_groups import SpeakerGroupManager, Speaker, SpeakerGroup, GroupType
from .zone_controller import SpotifyZoneController, PlaybackMode
from .speaker_enrollment import SpeakerEnrollment, EnrollmentIntent, EnrollmentCommand

__all__ = [
    # Client
    "SpotifyClient",
    # Auth
    "SpotifyAuth",
    "TokenManager",
    # Mood Mapping
    "MoodMapper",
    "AudioFeatures",
    "MoodProfile",
    # Dispatcher
    "MusicDispatcher",
    "MusicIntent",
    # Multi-room / Speaker Groups
    "SpeakerGroupManager",
    "Speaker",
    "SpeakerGroup",
    "GroupType",
    # Zone Controller
    "SpotifyZoneController",
    "PlaybackMode",
    # Enrollment
    "SpeakerEnrollment",
    "EnrollmentIntent",
    "EnrollmentCommand",
]
