"""Tests: SpeakerTagger — etiqueta hablante por utterance con ECAPA."""
import asyncio
from unittest.mock import MagicMock

import numpy as np

from src.ambient.speaker_tagger import SpeakerTagger
from src.users.speaker_identifier import SpeakerMatch

SR = 16000


def _match(user_id, conf, known):
    return SpeakerMatch(
        user_id=user_id, confidence=conf,
        embedding=np.zeros(192, dtype=np.float32), is_known=known,
    )


def test_known_speaker():
    ident = MagicMock()
    ident.identify.return_value = _match("gabriel", 0.88, True)
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert speaker == "gabriel"
    assert conf == 0.88


def test_unknown_when_no_enrolled_embeddings():
    ident = MagicMock()
    tagger = SpeakerTagger(identifier=ident, embeddings_loader=lambda: {}, min_audio_s=0.8)
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
    ident.identify.assert_not_called()  # sin enrolados no se gasta GPU


def test_unknown_for_too_short_audio():
    ident = MagicMock()
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR // 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
    ident.identify.assert_not_called()


def test_identify_error_is_unknown_not_crash():
    ident = MagicMock()
    ident.identify.side_effect = RuntimeError("CUDA hiccup")
    tagger = SpeakerTagger(
        identifier=ident,
        embeddings_loader=lambda: {"gabriel": np.ones(192, dtype=np.float32)},
        min_audio_s=0.8,
    )
    speaker, conf = asyncio.run(tagger.tag(np.zeros(SR * 2, dtype=np.float32)))
    assert (speaker, conf) == ("unknown", 0.0)
