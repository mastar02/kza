"""Word-boundary en keywords cortos del dispatcher (2026-06-02).

Charla ambiente NO debe rutear a domótica/cancel por colisión de subcadena
(baja∈trabajamos, pon∈supongo/ponen, sube∈subestimar, para∈preposición,
olvida∈inolvidable), mientras los comandos reales conservan su morfología
(prende∈prender debe seguir matcheando).
Ver: project_nexa_command_detection_rootcause_2026-06-02.
"""
import pytest
from unittest.mock import MagicMock

from src.orchestrator.dispatcher import RequestDispatcher, PathType
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.orchestrator.context_manager import ContextManager


@pytest.fixture
def dispatcher():
    return RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
    )


# --- Charla ambiente NO debe rutear a domótica (colisiones de subcadena) ---
@pytest.mark.parametrize("text", [
    "es la fusion y trabajamos",               # baja ∈ traBAJAmos
    "la mayoria a veces si supongo",           # pon  ∈ suPONgo
    "tengo una goticina mia no lo ponen mas",  # pon  ∈ PONen
    "no quiero subestimar el esfuerzo",        # sube ∈ suBEstimar
])
def test_ambient_chatter_not_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text.lower())
    assert path != PathType.FAST_DOMOTICS, f"{text!r} ruteó a domótica"


# --- Comandos reales SÍ deben rutear a domótica (morfología preservada) ---
@pytest.mark.parametrize("text", [
    "pon la luz del escritorio",
    "poné la luz",
    "baja la persiana",
    "bajá la persiana del living",
    "sube la temperatura",
    "subí la temperatura",
    "prender la luz",     # prende ∈ prender (substring, NO exact)
    "prendé la luz",
    "apagar el aire",     # apaga ∈ apagar
    "apagá la luz",
    "abrí la persiana",
    "encendé la luz",
])
def test_real_commands_still_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text.lower())
    assert path == PathType.FAST_DOMOTICS, f"{text!r} NO ruteó a domótica"


# --- Imperativos voseo con enclítico (rioplatense) SÍ deben rutear a domótica ---
# bajame/bajale/ponele/ponelo: el word-boundary no debe matarlos (theme vowel +
# pronombre enclítico). Ver review adversarial 2026-06-02.
@pytest.mark.parametrize("text", [
    "bajame la persiana",
    "bajale a la luz del living",
    "bajalo un poco",
    "ponele azul a la luz",
    "ponelo en calido",
])
def test_voseo_enclitic_still_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text.lower())
    assert path == PathType.FAST_DOMOTICS, f"{text!r} NO ruteó a domótica"


# --- Cancel: 'para' preposición NO debe cancelar; 'pará' sí ---
@pytest.mark.asyncio
async def test_para_preposition_does_not_cancel(dispatcher):
    ctx = MagicMock()
    ctx.pending_confirmation = None
    res = await dispatcher._check_special_commands("para abrirte todo", "u1", ctx)
    assert res is None or res.intent != "cancel"


@pytest.mark.asyncio
async def test_para_accented_still_cancels(dispatcher):
    ctx = MagicMock()
    ctx.pending_confirmation = None
    res = await dispatcher._check_special_commands("para la musica ya", "u1", ctx)
    # 'para' sin acento ya no cancela; el comando real es 'pará' (con acento)
    assert res is None or res.intent != "cancel"
    res2 = await dispatcher._check_special_commands("pará la musica", "u1", ctx)
    assert res2 is not None and res2.intent == "cancel"


@pytest.mark.asyncio
async def test_inolvidable_does_not_cancel(dispatcher):
    ctx = MagicMock()
    ctx.pending_confirmation = None
    res = await dispatcher._check_special_commands("que noche tan inolvidable", "u1", ctx)
    assert res is None or res.intent != "cancel"


@pytest.mark.asyncio
async def test_olvidate_enclitic_still_cancels(dispatcher):
    # 'olvidate de eso' (olvidá + te) debe seguir cancelando pese al boundary.
    ctx = MagicMock()
    ctx.pending_confirmation = None
    res = await dispatcher._check_special_commands("olvidate de eso", "u1", ctx)
    assert res is not None and res.intent == "cancel"
