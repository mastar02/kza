"""Voice routine creation awaits HA and keeps its LLM and vector work off the loop."""

import json
import threading
import time
from unittest.mock import MagicMock

from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from src.routines.routine_manager import RoutineManager

CREATE = "crea una rutina que cuando salga el sol prenda la luz del living"
H5_REFUSAL = "No pude crear la rutina: La gestión de rutinas necesita el scheduler persistente de KZA."


class _Home:
    """HomeAssistantClient's getters are coroutines."""

    def __init__(self):
        self.reads = 0

    async def get_domotics_entities(self):
        self.reads += 1
        return [{"entity_id": "light.living", "attributes": {"friendly_name": "Living"}}]


class _SlowLLM:
    def __init__(self, entity="light.living"):
        self.entity = entity
        self.on_main = []

    def __call__(self, prompt, max_tokens=1500, temperature=0.1):
        self.on_main.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.3)
        return {"choices": [{"text": json.dumps({
            "name": "Amanecer", "description": "Luz al amanecer",
            "triggers": [{"type": "sun", "config": {"event": "sunrise"}, "description": "al amanecer"}],
            "actions": [{"entity_id": self.entity, "domain": "light", "service": "turn_on",
                         "data": {}, "description": "prender la luz del living"}]})}]}


async def test_voice_creation_reaches_confirmation_and_the_documented_refusal():
    home, llm = _Home(), _SlowLLM()
    manager = RoutineManager(home, MagicMock(), llm)
    probe = LoopLagProbe(LoopLagConfig(interval_s=0.02))
    probe.start()
    proposal = await manager.handle(CREATE)
    await probe.stop()
    assert home.reads == 1 and llm.on_main == [False]
    assert proposal["handled"] and proposal["success"]
    assert proposal["response"].startswith("Voy a crear la rutina 'Amanecer'")
    assert manager.awaiting_confirmation
    assert probe.snapshot()["max_ms"] < 200, probe.snapshot()
    # Writing stays behind the persistent-scheduler boundary (H5): no HA automation.
    confirmed = await manager.handle("sí")
    assert confirmed["handled"] and not confirmed["success"]
    assert confirmed["response"] == H5_REFUSAL
    assert not manager.awaiting_confirmation


async def test_unknown_entity_is_resolved_by_vector_search_off_the_loop():
    chroma = MagicMock()
    on_main = []

    def search(description, threshold):
        on_main.append(threading.current_thread() is threading.main_thread())
        return {"entity_id": "light.living", "domain": "light"}

    chroma.search_command = search
    manager = RoutineManager(_Home(), chroma, _SlowLLM(entity="light.unknown"))
    proposal = await manager.handle(CREATE)
    assert proposal["success"] and on_main == [False]
    assert manager.pending_routine["actions"][0]["entity_id"] == "light.living"


async def test_without_a_reasoner_creation_is_declined_without_reading_ha():
    home = _Home()
    result = await RoutineManager(home, MagicMock(), None).handle(CREATE)
    assert result["handled"] and not result["success"] and home.reads == 0
    assert result["response"] == "No puedo crear rutinas por voz sin el modelo de lenguaje."


async def test_listing_routines_reads_the_index_off_the_loop():
    chroma = MagicMock()
    on_main = []

    def list_routines():
        on_main.append(threading.current_thread() is threading.main_thread())
        return [{"name": "Amanecer"}]

    chroma.list_routines = list_routines
    result = await RoutineManager(_Home(), chroma, _SlowLLM()).handle("qué rutinas tengo")
    assert result["response"] == "Tienes 1 rutinas: Amanecer" and on_main == [False]
