"""Construct a real response/zone/player graph with synthetic endpoints."""

from unittest.mock import MagicMock
from src.audio.zone_manager import Zone, ZoneManager
from src.llm.reasoner import HttpReasoner
from src.pipeline.response_handler import ResponseHandler
from src.tts import piper_tts
from tests.mocks.audio_output import AudioBackend
from tests.mocks.llm_http import Endpoint, SSEBody, event
from tests.mocks.speech import PCMEngine


def voice_graph(monkeypatch, *, parts=None, **options):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, 'sd', backend)
    engine = PCMEngine()
    amplifier = MagicMock()
    amplifier.audio_output_device = 7
    zones = ZoneManager([Zone('zone_origin', 'Origin', 0, 1), Zone('zone_other', 'Other', 1, 2)],
                        ma1260_controller=amplifier)
    endpoint = Endpoint(SSEBody(parts or [event('Una respuesta completa.'), event(finish='stop')]))
    reasoner = HttpReasoner(model='configured-model', api_style='chat', cloud_consent=True)
    monkeypatch.setattr(reasoner, '_new_async_client', endpoint.build)
    owner = ResponseHandler(engine, zones, reasoner, llm_use_filler=False, **options)
    return owner, engine, backend, endpoint, amplifier

