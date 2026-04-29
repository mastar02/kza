"""
Test factories for components that require explicit configuration.

Production constructors (SpeakerIdentifier, EmotionDetector, ChromaSync) do not
provide defaults for device/model — config is the single source of truth in
settings.yaml. Tests that exercise behavior unrelated to that config use these
factories to instantiate with safe test defaults (CPU device, canonical models).

Imports are lazy so a test that only needs one factory does not pull in the
others' dependencies (e.g. chromadb).

Usage:
    from tests.factories import make_speaker_identifier, make_emotion_detector

    identifier = make_speaker_identifier()
    detector = make_emotion_detector(sample_rate=22050)
"""


def make_speaker_identifier(**overrides):
    """Build a SpeakerIdentifier with CPU defaults, override any field."""
    from src.users.speaker_identifier import SpeakerIdentifier

    defaults = dict(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        device="cpu",
        similarity_threshold=0.75,
        sample_rate=16000,
    )
    defaults.update(overrides)
    return SpeakerIdentifier(**defaults)


def make_emotion_detector(**overrides):
    """Build an EmotionDetector with CPU defaults, override any field."""
    from src.users.emotion_detector import EmotionDetector

    defaults = dict(
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cpu",
        sample_rate=16000,
    )
    defaults.update(overrides)
    return EmotionDetector(**defaults)


def make_chroma_sync(**overrides):
    """Build a ChromaSync with CPU defaults, override any field."""
    from src.vectordb.chroma_sync import ChromaSync

    defaults = dict(
        chroma_path="/tmp/test_chroma",
        embedder_model="BAAI/bge-m3",
        embedder_device="cpu",
    )
    defaults.update(overrides)
    return ChromaSync(**defaults)
