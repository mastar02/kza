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


def _turn_on_classification(confidence: float = 0.9):
    return CommandClassification(
        is_command=True,
        confidence=confidence,
        intent="turn_on",
        entity_hint="light",
        rejection_reason=None,
    )


def _make_router_for(text: str, classification):
    """Router con texto y clasificación LLM arbitrarios (texto SIN parse full
    de gramática → siempre cae al path LLM)."""
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=_make_cmd_stub(text))
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None,
        timings={}, was_queued=False, queue_position=None,
    ))
    llm = MagicMock()
    llm.classify = AsyncMock(return_value=classification)
    router = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=MagicMock(),
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        command_gate=CommandAcceptanceGate(wake_words=()),
        llm_command_router=llm,
        wake_acoustically_confirmed=True,
        confidence_threshold=0.75,
    )
    return router, orch


class TestSetIntentRoutesFastDomotics:
    """Bug 2026-08-06: 'Nexa, luz al 100% del living' — el grammar clasificó
    intent=set conf=1.00, pero el router solo propagaba service_filter para
    turn_on/turn_off. Sin verbo, DOMOTICS_KEYWORDS (solo verbos) no matchea y
    _classify_request mandaba el comando al SLOW_LLM → timeout de 5s — y el
    slow path no tiene tool-calling a HA, así que el brillo por ahí no puede
    ejecutar NUNCA. En HA setear brillo/color ES light.turn_on con atributos,
    por eso set/set_brightness mapean a service_filter='turn_on' (que además
    filtra el vector search al doc correcto, esquivando el antónimo turn_off
    que el retrieval denso no distingue)."""

    @pytest.mark.asyncio
    async def test_grammar_set_propagates_turn_on_service_filter(self):
        # El texto real del bug: sin verbo, con room y slot de brillo.
        router, llm = _make_router_with_llm(wake_acoustically_confirmed=True)
        router.command_processor.process_command = AsyncMock(
            return_value=_make_cmd_stub("Nexa, luz al 100% del living.")
        )

        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32),
            room_id="escritorio",
            wake_text="Nexa, luz al 100% del living.",
        )
        await router.process_command(event)

        assert not llm.classify.called, "el grammar debe ganar (conf 1.00)"
        router._orchestrator.process.assert_awaited_once()
        kwargs = router._orchestrator.process.call_args.kwargs
        assert kwargs["service_filter"] == "turn_on"
        assert kwargs["query_slots"] == {"brightness_pct": 100}

    @pytest.mark.asyncio
    async def test_llm_set_brightness_propagates_turn_on_service_filter(self):
        classification = CommandClassification(
            is_command=True,
            confidence=0.9,
            intent="set_brightness",
            entity_hint="light",
            rejection_reason=None,
            slots={"brightness_pct": 80},
        )
        # Texto que el grammar NO parsea full ("lu" mangled → sin dominio)
        # pero con slot textual real → path LLM con evidencia.
        router, orch = _make_router_for(
            "Nexa, la lu del living al 80 por ciento.", classification
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, la lu del living al 80 por ciento.",
        )
        await router.process_command(event)

        orch.process.assert_awaited_once()
        kwargs = orch.process.call_args.kwargs
        assert kwargs["service_filter"] == "turn_on"
        assert kwargs["query_slots"] == {"brightness_pct": 80}

    @pytest.mark.asyncio
    async def test_grammar_set_color_maps_turn_on(self):
        # Color también es light.turn_on con atributos (review 2026-08-09:
        # la primera versión del fix dejaba color muriendo en el SLOW_LLM).
        router, llm = _make_router_with_llm(wake_acoustically_confirmed=True)
        router.command_processor.process_command = AsyncMock(
            return_value=_make_cmd_stub("Nexa, luz del living en rojo.")
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, luz del living en rojo.",
        )
        await router.process_command(event)

        kwargs = router._orchestrator.process.call_args.kwargs
        assert kwargs["service_filter"] == "turn_on"
        assert kwargs["query_slots"] == {"rgb_color": [255, 0, 0]}

    @pytest.mark.asyncio
    async def test_textual_slot_overrides_llm_hallucinated_value(self):
        # Review 2026-08-09: la evidencia textual no solo habilita el mapeo,
        # REEMPLAZA los slots del LLM — si el texto dice 30 y el 7B devolvió
        # 80, a HA viaja el 30.
        classification = CommandClassification(
            is_command=True, confidence=0.9, intent="set_brightness",
            entity_hint="light", rejection_reason=None,
            slots={"brightness_pct": 80},   # valor alucinado
        )
        router, orch = _make_router_for(
            "Nexa, la lu del living al 30 por ciento.", classification
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, la lu del living al 30 por ciento.",
        )
        await router.process_command(event)

        kwargs = orch.process.call_args.kwargs
        assert kwargs["service_filter"] == "turn_on"
        assert kwargs["query_slots"] == {"brightness_pct": 30}

    @pytest.mark.asyncio
    async def test_llm_set_without_textual_slot_does_not_map(self):
        # Guard de evidencia (review 2026-08-09): un set_brightness del LLM
        # cuyo slot NO es extraíble del texto es la firma de una alucinación
        # del 7B sobre garble — sin evidencia textual no gana fast path
        # (antes de este fix moría en el SLOW_LLM; se preserva ese destino).
        classification = CommandClassification(
            is_command=True, confidence=0.9, intent="set_brightness",
            entity_hint="light", rejection_reason=None,
            slots={"brightness_pct": 50},   # alucinado: el texto no trae slot
        )
        router, orch = _make_router_for(
            "Nexa, quiero las luces distintas.", classification
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, quiero las luces distintas.",
        )
        await router.process_command(event)

        orch.process.assert_awaited_once()
        assert orch.process.call_args.kwargs["service_filter"] is None

    @pytest.mark.asyncio
    async def test_llm_set_brightness_non_light_entity_does_not_map(self):
        # Guard de dominio (review 2026-08-09): "bajá el brillo de la tele"
        # con filtro turn_on domain-agnóstico habría matcheado el doc
        # media_player.turn_on y PRENDIDO la tele en vez de bajarle el brillo.
        classification = CommandClassification(
            is_command=True, confidence=0.9, intent="set_brightness",
            entity_hint="media_player", rejection_reason=None,
            slots={"brightness_pct": 20},
        )
        router, orch = _make_router_for(
            "Nexa, bajá el brillo de la tele al 20 por ciento.", classification
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, bajá el brillo de la tele al 20 por ciento.",
        )
        await router.process_command(event)

        orch.process.assert_awaited_once()
        assert orch.process.call_args.kwargs["service_filter"] is None

    @pytest.mark.asyncio
    async def test_turn_off_propagation_unchanged(self):
        # Regresión: el par binario sigue propagando su propio service.
        classification = CommandClassification(
            is_command=True, confidence=0.9, intent="turn_off",
            entity_hint="light", rejection_reason=None,
        )
        router, orch = _make_router_for(
            "Nexa, apagá la luz.", classification
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, apagá la luz.",
        )
        await router.process_command(event)

        orch.process.assert_awaited_once()
        assert orch.process.call_args.kwargs["service_filter"] == "turn_off"


class TestUnverifiedIntentGuard:
    """El intent binario del LLM debe estar evidenciado por un verbo del texto
    (caso real 2026-06-06: 'apagá'→STT 'pero a la luz'→LLM turn_on conf alta
    → prendió cuando pidieron apagar)."""

    @pytest.mark.asyncio
    async def test_unevidenced_turn_on_rejected(self):
        router, orch = _make_router_for(
            "Nexa, pero a la luz.", _turn_on_classification(0.9)
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, pero a la luz.",
        )
        result = await router.process_command(event)
        assert result["success"] is False
        assert result["intent"] == "unverified_intent:turn_on"
        orch.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_evidenced_turn_on_dispatches(self):
        router, orch = _make_router_for(
            "Nexa, prender el...", _turn_on_classification(0.9)
        )
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, prender el...",
        )
        result = await router.process_command(event)
        orch.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_unevidenced_rejection_plays_earcon(self):
        """El rechazo por verbo no evidenciado debe avisar con EARCON (solo
        sonido, 2026-06-15): para llegar acá el texto pasó el CommandGate y el
        7B dio comando con confianza alta — casi seguro es un humano con el STT
        garbleado. El silence mudo es indistinguible de 'no funciona' (caso real
        2026-06-11 17:57). Reemplaza el reprompt de voz por el earcon."""
        router, orch = _make_router_for(
            "Nexa, pero a la luz.", _turn_on_classification(0.9)
        )
        # Activar earcon con wake_score alto (simula humano plausible)
        router._earcon_cfg = {
            "enabled": True,
            "min_wake_score": 0.0,
            "min_rms": 0.0,
            "reasons": ["unverified_intent"],
        }
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Nexa, pero a la luz.",
            wake_score=0.9,
        )
        result = await router.process_command(event)
        # response vacío: el feedback es el earcon, no texto hablado
        assert result["response"] == "", "solo earcon, sin texto de voz"
        assert result["success"] is False
        assert result["intent"].startswith("unverified_intent")
        router.response_handler.speak.assert_not_called()
        router.response_handler.play_earcon.assert_called_once()

    @pytest.mark.asyncio
    async def test_gate_rejection_stays_silent(self):
        """Los rechazos del CommandGate (TV: 'gracias por ver el video') NO
        deben hablar — son cientos por día con la TV prendida."""
        router, orch = _make_router_for(
            "Gracias por ver el video.", _noise_classification()
        )
        # Gate real con noise phrases activas para que rechace el texto de TV.
        router.command_gate = CommandAcceptanceGate(wake_words=("nexa",))
        event = CommandEvent(
            audio=np.zeros(16000, dtype=np.float32), room_id="escritorio",
            wake_text="Gracias por ver el video.",
        )
        result = await router.process_command(event)
        assert result["success"] is False
        router.response_handler.speak.assert_not_called()
