"""
Tests for _grammar_fastpath_classification routing by target/quality.

Verifies that:
- Domotics commands (quality='full') are routed as is_command=True with correct intent.
- Music/media commands (target='music', quality='full') are also is_command=True.
- Incompatible or conversational text returns None (falls through to LLM router).
"""
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock heavy system-level modules before any imports
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np
import pytest

from src.pipeline.request_router import _grammar_fastpath_classification
from src.pipeline.command_event import CommandEvent
from src.nlu.llm_router import CommandClassification
from src.pipeline.request_router import RequestRouter
from src.nlu.command_gate import CommandAcceptanceGate


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


# ---------------------------------------------------------------------------
# Helpers compartidos para TestAmbientStrictDisablesWakeBonus
# ---------------------------------------------------------------------------

def _make_cmd_stub(text="prende la luz"):
    """Stub del resultado de command_processor.process_command."""
    cmd = MagicMock()
    cmd.text = text
    cmd.user = None
    cmd.emotion = None
    cmd.timings = {}
    cmd.stt_confidence = None
    return cmd


def _noise_classification():
    """Clasificación de 'ruido' para que el LLM router indique no-comando."""
    return CommandClassification(
        is_command=False,
        confidence=0.1,
        intent="noise",
        entity_hint=None,
        rejection_reason="ambient_noise",
    )


def _make_router_with_llm(wake_acoustically_confirmed: bool):
    """Construye un RequestRouter con wake_acoustically_confirmed dado y un
    llm_command_router mockeado que devuelve ruido (no-comando).
    Devuelve (router, llm_mock).
    """
    # STT stub — devuelve el texto que viene en el CommandEvent (pretranscribed)
    # pero el router lo llama con pretranscribed_text y cmd.text debe coincidir.
    cmd_stub = _make_cmd_stub("prende la luz")
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=cmd_stub)

    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None,
        timings={}, was_queued=False, queue_position=None,
    ))

    llm = MagicMock()
    llm.classify = AsyncMock(return_value=_noise_classification())

    router = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=MagicMock(),
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        # El gate acepta todo (openwakeword ya disparó)
        command_gate=CommandAcceptanceGate(wake_words=()),
        llm_command_router=llm,
        wake_acoustically_confirmed=wake_acoustically_confirmed,
        confidence_threshold=0.75,
    )
    return router, llm


class TestAmbientStrictDisablesWakeBonus:
    """ambient_strict=True debe suprimir el bonus wake_acoustically_confirmed
    en el grammar fast-path.

    'prende la luz' es un comando de 3 palabras que la gramática parsea con
    quality='full' y confidence=0.70 (sin bonus).  Con wake_confirmed=True el
    bonus +0.15 lo lleva a 0.85 ≥ 0.75 → fast-path gana y el LLM NO se llama.
    Con ambient_strict=True el bonus se suprime → conf=0.70 < 0.75 → fast-path
    devuelve None → se llama al llm_command_router.classify.
    """

    @pytest.mark.asyncio
    async def test_strict_event_does_not_get_wake_bonus(self):
        """STRICT: sin bonus el fast-path falla → llm_command_router.classify SÍ se llama."""
        router, llm = _make_router_with_llm(wake_acoustically_confirmed=True)

        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32),
            room_id="escritorio",
            wake_text="prende la luz",
            ambient_strict=True,
        )
        await router.process_command(event)

        assert llm.classify.called, (
            "En STRICT el bonus debe suprimirse → fast-path no clasifica → "
            "el LLMCommandRouter DEBE ser llamado"
        )

    @pytest.mark.asyncio
    async def test_normal_event_keeps_wake_bonus(self):
        """Normal (ambient_strict=False): bonus aplica → fast-path gana → LLM NO se llama."""
        router, llm = _make_router_with_llm(wake_acoustically_confirmed=True)

        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32),
            room_id="escritorio",
            wake_text="prende la luz",
            ambient_strict=False,
        )
        await router.process_command(event)

        assert not llm.classify.called, (
            "En modo normal el bonus aplica → fast-path gana → "
            "el LLMCommandRouter NO debe ser llamado"
        )
