"""Tests: una captura acústica vacía NO debe suprimir el canal textual 8s.

Incidente 2026-07-25 18:19 ("al tercer intento recién agarró"), trampa de dos
partes:

    18:19:43.046  Wake word in escritorio (nexa: 0.87)      <- acústico OK
    18:19:43.453  [STT-veto] descartado (parakeet=vacío): 'De la luz.'
    18:19:43.453  [CommandProcessor] Text=''                <- comando muerto
    18:19:43.161  [TextualWake] skip decision=dedup_acoustic
                  text='Next up, prende la luz.'            <- red de seguridad ABAJO

El wake textual SÍ había reconocido el comando, pero lo descartó porque
`last_command_dispatch_ts` estaba fresco. Ese ts se setea al **disparar el
wake** (`_dispatch_command`), no al **producir un comando**, así que una
captura vetada o vacía deja el canal textual suprimido durante
`dedup_window_s` (8s) — la red de seguridad se baja exactamente cuando el
camino primario acaba de fallar en silencio.

El ts se sigue seteando ANTES del await (invariante deliberado: un comando
slow-path de varios segundos tiene que ganar la carrera de dedup desde el
primer instante). Lo que cambia es que al terminar, si la captura no produjo
nada, se libera la supresión.

Conservador a propósito: sólo se libera en `outcome == "empty"`. Ruido/TV y
alucinaciones mantienen la supresión — ahí el canal textual no debe entrar.
"""

import sys
import time
from unittest.mock import MagicMock, AsyncMock

import pytest

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.pipeline.command_event import CommandEvent


def _room_stream(room_id: str) -> RoomStream:
    wake = MagicMock()
    wake.load = MagicMock()
    wake.detect = MagicMock(return_value=None)
    wake.get_active_models = MagicMock(return_value=["nexa"])
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio = MagicMock(return_value=(True, "ok"))
    echo.config = MagicMock()
    echo.config.post_speech_buffer_ms = 400
    return RoomStream(
        room_id=room_id, device_index=0,
        wake_detector=wake, echo_suppressor=echo,
    )


def _loop(callback_result: dict) -> MultiRoomAudioLoop:
    follow_up = MagicMock()
    follow_up.is_active = False
    follow_up.follow_up_window = 8.0
    follow_up.start_conversation = MagicMock()
    loop = MultiRoomAudioLoop(
        room_streams={"escritorio": _room_stream("escritorio")},
        follow_up=follow_up,
    )
    loop.on_command(AsyncMock(return_value=callback_result))
    return loop


def _event() -> CommandEvent:
    import numpy as np
    return CommandEvent(
        audio=np.zeros(1600, dtype="float32"),
        room_id="escritorio",
        wake_text="nexa",
        wake_score=0.87,
    )


@pytest.mark.asyncio
async def test_empty_capture_releases_dedup_suppression():
    """El caso del incidente: STT-veto dejó Text='' → liberar el canal textual.

    Falla sin el fix: el ts queda fresco y el wake textual deduplica 8s.
    """
    loop = _loop({"text": "", "intent": "gate_rejected:empty"})

    await loop._dispatch_command(_event())

    assert loop.last_command_dispatch_ts("escritorio") == 0.0, (
        "una captura vacía dejó el canal textual suprimido"
    )


@pytest.mark.asyncio
async def test_accepted_capture_keeps_suppression():
    """Un comando real SÍ debe suprimir el textual (evitar doble ejecución)."""
    loop = _loop({"success": True, "text": "prendé la luz", "intent": "turn_on"})

    await loop._dispatch_command(_event())

    ts = loop.last_command_dispatch_ts("escritorio")
    assert ts > 0.0, "un comando aceptado debe mantener la supresión"
    assert time.monotonic() - ts < 1.0, "el ts debe ser reciente"


@pytest.mark.asyncio
async def test_noise_capture_keeps_suppression():
    """TV/ruido: el canal textual NO debe entrar. Conservador."""
    loop = _loop({
        "text": "y ahora volvemos con el partido",
        "intent": "gate_rejected:noise_phrase:'volvemos con'",
    })

    await loop._dispatch_command(_event())

    assert loop.last_command_dispatch_ts("escritorio") > 0.0, (
        "un reject por ruido no debe liberar el canal textual"
    )


@pytest.mark.asyncio
async def test_hallucination_keeps_suppression():
    """Alucinación de Whisper con texto: mantener supresión."""
    loop = _loop({
        "text": "¡Gracias!",
        "intent": "gate_rejected:filler_word:'gracias'",
    })

    await loop._dispatch_command(_event())

    assert loop.last_command_dispatch_ts("escritorio") > 0.0


@pytest.mark.asyncio
async def test_ts_is_already_set_while_callback_runs():
    """Invariante que NO se puede romper (spec 2026-07-05, review final).

    Un comando slow-path tarda segundos; el ts tiene que existir desde el
    primer instante del dispatch o el canal textual evalúa la misma utterance
    sin ver el dedup y dispara un segundo comando en paralelo.
    """
    seen: list[float] = []

    async def slow_callback(event):
        seen.append(loop.last_command_dispatch_ts("escritorio"))
        return {"success": True, "text": "prendé la luz"}

    loop = _loop({})
    loop.on_command(slow_callback)

    await loop._dispatch_command(_event())

    assert seen and seen[0] > 0.0, (
        "el ts debe estar seteado ANTES de awaitear el callback"
    )


@pytest.mark.asyncio
async def test_stt_veto_also_releases_suppression():
    """Los dos fixes de 2026-07-25 tienen que COMPONER.

    El veto viaja como `gate_rejected:stt_veto` (motivo propio, para que el
    earcon pueda saltear el piso de RMS). Ese cambio no debe romper la
    liberación de la supresión: `classify_outcome` mira el texto vacío ANTES
    del intent, así que sigue clasificando "empty".

    Sin esto, el escenario exacto del incidente seguiría vivo.
    """
    loop = _loop({"text": "", "intent": "gate_rejected:stt_veto"})

    await loop._dispatch_command(_event())

    assert loop.last_command_dispatch_ts("escritorio") == 0.0, (
        "un veto dejó el canal textual suprimido — la trampa sigue viva"
    )


@pytest.mark.asyncio
async def test_callback_exception_keeps_suppression():
    """Si el callback explota no sabemos qué pasó → no liberar (fail-safe)."""
    loop = _loop({})
    loop.on_command(AsyncMock(side_effect=RuntimeError("boom")))

    await loop._dispatch_command(_event())  # no debe propagar

    assert loop.last_command_dispatch_ts("escritorio") > 0.0, (
        "ante un error desconocido hay que mantener la supresión"
    )
