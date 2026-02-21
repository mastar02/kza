"""
Ambient Module
Detección de eventos ambientales sin wake word
"""

from .audio_event_detector import AudioEventDetector, AudioEventType, AudioEvent

__all__ = ["AudioEventDetector", "AudioEventType", "AudioEvent"]
