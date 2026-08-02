"""Tests: el [STT-veto] no debe matar comandos en silencio total.

Incidente 2026-07-25 18:19. El wake acústico disparó a 0.87, Whisper dio
'De la luz.' (comando real, parcial), Parakeet devolvió vacío, y el veto
descartó la captura dejando `Text=''`. El usuario no oyó absolutamente nada.

Dos problemas distintos, los dos cubiertos acá:

1. **El veto era indistinguible de "el STT no oyó nada".** `_apply_shadow_veto`
   devuelve `""` y el router reporta `gate_rejected:empty`, el mismo motivo que
   una captura genuinamente muda. Sin esa distinción no se puede ni medir el
   daño ni decidir distinto.

2. **El earcon no sonaba por el piso de RMS.** `should_play_earcon` exige
   `rms >= min_rms` (0.02 en prod) y el RMS real del comando era **0.0105** —
   los comandos reales de este mic viven en 0.010-0.013. El piso existe para
   que el earcon no le suene a la TV distante, pero en un veto **ya tenemos
   evidencia positiva de habla real: el motor primario produjo texto**. Ahí el
   piso de RMS no aplica.

El resto de las compuertas del earcon se respetan: `enabled`, `min_wake_score`
y los prefijos de ruido.
"""

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

from src.pipeline.earcon_gate import should_play_earcon


CFG = {
    "enabled": True,
    "min_wake_score": 0.55,
    "min_rms": 0.02,
    "reasons": ["empty", "empty_after_norm", "high_compression", "low_confidence"],
}


class TestVetoBypassesRmsFloor:
    """Un veto tiene evidencia de habla real → el piso de RMS no aplica."""

    def test_veto_earcons_below_rms_floor(self):
        """El caso exacto del incidente: wake 0.87, rms 0.0105."""
        assert should_play_earcon("stt_veto", 0.87, 0.0105, CFG) is True

    def test_generic_empty_still_respects_rms_floor(self):
        """Sin veto, una captura muda con rms bajo sigue sin earcon.

        Esta es la defensa anti-TV — no se toca.
        """
        assert should_play_earcon("empty", 0.87, 0.0105, CFG) is False

    def test_veto_still_respects_wake_score_floor(self):
        """Un wake débil sigue sin earcon aunque haya veto."""
        assert should_play_earcon("stt_veto", 0.30, 0.0105, CFG) is False

    def test_veto_respects_enabled_flag(self):
        cfg = {**CFG, "enabled": False}
        assert should_play_earcon("stt_veto", 0.87, 0.0105, cfg) is False

    def test_veto_earcons_with_healthy_rms_too(self):
        assert should_play_earcon("stt_veto", 0.87, 0.05, CFG) is True

    def test_veto_works_without_being_listed_in_reasons(self):
        """No debe quedar inerte esperando un cambio de config en producción."""
        cfg = {**CFG, "reasons": ["empty"]}
        assert should_play_earcon("stt_veto", 0.87, 0.0105, cfg) is True

    def test_veto_can_be_disabled_explicitly(self):
        """Escape hatch: si molesta, se apaga por config."""
        cfg = {**CFG, "exclude_reasons": ["stt_veto"]}
        assert should_play_earcon("stt_veto", 0.87, 0.0105, cfg) is False


class TestVetoIsDistinguishable:
    """El veto tiene que viajar como motivo propio, no como 'empty'."""

    @pytest.mark.asyncio
    async def test_veto_sets_flag_on_result(self):
        from src.pipeline.command_processor import CommandProcessor, ProcessedCommand

        result = ProcessedCommand(text="")
        proc = CommandProcessor.__new__(CommandProcessor)
        proc.shadow_veto_timeout_s = 1.0

        shadow = MagicMock()
        shadow.text = ""
        shadow.elapsed_ms = 50.0

        async def _fut():
            return shadow

        out = await proc._apply_shadow_veto(_fut(), "De la luz.", result)

        assert out == "", "el veto debe seguir descartando el texto"
        assert result.stt_vetoed is True, (
            "el veto no quedó registrado en el resultado"
        )

    @pytest.mark.asyncio
    async def test_no_veto_leaves_flag_false(self):
        from src.pipeline.command_processor import CommandProcessor, ProcessedCommand

        result = ProcessedCommand(text="")
        proc = CommandProcessor.__new__(CommandProcessor)
        proc.shadow_veto_timeout_s = 1.0

        shadow = MagicMock()
        shadow.text = "de la luz"
        shadow.elapsed_ms = 50.0

        async def _fut():
            return shadow

        out = await proc._apply_shadow_veto(_fut(), "De la luz.", result)

        assert out == "De la luz."
        assert result.stt_vetoed is False

    def test_processed_command_defaults_to_not_vetoed(self):
        from src.pipeline.command_processor import ProcessedCommand

        assert ProcessedCommand(text="hola").stt_vetoed is False
