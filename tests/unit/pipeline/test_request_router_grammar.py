"""
Tests for _grammar_fastpath_classification routing by target/quality.

Verifies that:
- Domotics commands (quality='full') are routed as is_command=True with correct intent.
- Music/media commands (target='music', quality='full') are also is_command=True.
- Incompatible or conversational text returns None (falls through to LLM router).
"""
import sys
from unittest.mock import MagicMock

# Mock heavy system-level modules before any imports
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import pytest

from src.pipeline.request_router import _grammar_fastpath_classification


@pytest.mark.parametrize("text,intent,is_cmd", [
    ("nexa prendé la luz del escritorio", "turn_on", True),
    ("subí la persiana del cuarto", "open", True),
    # media también es comando válido; "subí el volumen" sin wake = conf 0.70,
    # por eso usamos threshold 0.70 para toda la suite (ver umbral en la llamada).
    ("subí el volumen", "volume_set", True),
    ("abrí la luz", None, None),    # incompat intent/domain → quality='partial' → None
    ("hola qué tal", None, None),   # ninguna señal → quality='none' → None
])
def test_grammar_fastpath_classification(text, intent, is_cmd):
    # Use 0.70 threshold so the music case (conf=0.70) is not cut off while still
    # exercising the quality gate for the incompatible / noise cases.
    cls = _grammar_fastpath_classification(text, 0.70)
    if is_cmd is None:
        assert cls is None
    else:
        assert cls.is_command is True
        assert cls.intent == intent
