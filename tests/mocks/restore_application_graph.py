"""Public application assembly for compatible restore rehearsals.

Only model/transport/embedding leaves are synthetic. GraphBundle is a view of
owners constructed by main; it never constructs another graph or coordinator.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from uuid import uuid4

from aiohttp import web
from aiohttp.test_utils import TestServer
import chromadb

from src.application import ApplicationGraph
from src.dashboard.authority import DashboardPermission
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemoryReviewer
from src.users.voice_ledger import VoiceReviewer
from src.wakeword.clip_store import ClipTarget, CollectorKind
from tests.mocks.application_graph import offline_home
from tests.mocks.memory_graph import SyntheticEmbedding
from tests.mocks.producer_erasure_graph import (
    configuration as base_configuration,
    construction_sdk_guards,
    nominal_factories,
    until,
)


TOKEN = "synthetic-application-restore-root"
MODEL_ENV = "KZA_APPLICATION_RESTORE_MODEL_KEY"
PARTICIPANTS = frozenset(part.value for part in ErasureParticipant)


@asynccontextmanager
async def local_models():
    requests = []

    async def respond(request):
        requests.append((request.method, request.path))
        if request.path.endswith("/models"):
            return web.json_response({"object": "list", "data": [
                {"id": "synthetic", "object": "model"}]})
        return web.json_response({"id": "synthetic-response", "model": "synthetic",
            "choices": [{"text": "Synthetic compacted summary", "message": {
                "role": "assistant", "content": "Synthetic compacted summary"},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", respond)
    async with TestServer(app, host="127.0.0.1") as server:
        yield str(server.make_url("")).rstrip("/"), requests


class MemorySDK:
    """Retain the installed SDK, replacing only its default embedding leaf."""

    def __init__(self, sdk):
        self.sdk, self.closed = sdk, False

    def get_or_create_collection(self, *args, **kwargs):
        if "embedding_function" not in kwargs:
            kwargs["embedding_function"] = SyntheticEmbedding()
        return self.sdk.get_or_create_collection(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.sdk, name)

    def close(self):
        if not self.closed:
            self.sdk.close()
            self.closed = True


def sdk_leaves(patch, memory_path):
    """Wrap only the configured memory index; keep all real public closes."""
    import torch

    real_client = chromadb.PersistentClient
    clients = []

    def client(*args, **kwargs):
        path = kwargs.get("path", args[0] if args else None)
        sdk = real_client(*args, **kwargs)
        if path is not None and Path(path).resolve() == memory_path.resolve():
            sdk = MemorySDK(sdk)
            clients.append(sdk)
        return sdk

    patch.setattr(chromadb, "PersistentClient", client)

    class CPUEncoder:
        def encode_batch(self, audio):
            value = float(audio.mean().item())
            vector = torch.zeros((1, 1, 192), dtype=torch.float32)
            vector[0, 0, 0 if value < .15 else 1 if value < .25 else 2] = 1
            return vector

    class EncoderClassifier:
        @classmethod
        def from_hparams(cls, *, source, savedir, run_opts):
            assert run_opts["device"] == "cpu"
            return CPUEncoder()

    for name in ("speechbrain", "speechbrain.inference"):
        module = ModuleType(name)
        module.__path__ = []
        patch.setitem(sys.modules, name, module)
    module = ModuleType("speechbrain.inference.speaker")
    module.EncoderClassifier = EncoderClassifier
    patch.setitem(sys.modules, module.__name__, module)
    audio = ModuleType("torchaudio")

    def backends():
        return []

    audio.list_audio_backends = backends
    patch.setitem(sys.modules, "torchaudio", audio)
    return clients


def configuration(root, home_url, model_url):
    """Resolve every policy path and optional owner before inventory sealing."""
    root = Path(root).resolve()
    config = base_configuration(root, home_url, ambient=True, rooms=True)
    config["memory"].update(enabled=True, chroma_path=str(root / "chroma"),
        privacy_db_path=str(root / "privacy.db"),
        preferences_path=str(root / "preferences.json"))
    config["ambient"].update(shadow_mode=False, quality={"enabled": False},
        stt={"engine": "parakeet", "device": "cpu", "parakeet_model": "synthetic-ambient"},
        keep_audio={"enabled": True, "dir": str(root / "audio"), "min_free_mb": 0},
        distill={"llm_url": model_url + "/v1", "destination": "local", "model": "synthetic"})
    config["rooms"]["wake_word"]["capture_clips"].update(enabled=True, dir=str(root / "wake"))
    config["training"]["stt_corrections"].update(enabled=True, data_dir=str(root / "stt"))
    config["training"].update(conversations_dir=str(root / "conversations"),
        exports_dir=str(root / "exports"))
    config["analytics"].update(enabled=True, events_db=str(root / "events.db"),
        suggestions_path=str(root / "suggestions.json"))
    config["observability"].update(enabled=True, metrics_path=str(root / "metrics.jsonl"),
        logstash_host=None)
    config["orchestrator"]["context"] = {
        "compaction": {"enabled": True, "base_url": model_url + "/v1",
            "destination": "local", "model": "synthetic", "api_key_env": MODEL_ENV,
            "threshold_turns": 100},
        "persistence": {"enabled": True, "base_path": str(root / "contexts")},
    }
    return config


@dataclass(frozen=True)
class MemoryView:
    """Compatibility view of main's actual manager, without lifecycle methods."""

    manager: object


@dataclass
class GraphBundle:
    graph: ApplicationGraph
    root: Path
    clock: object
    config: dict
    models: object
    home: dict
    model_requests: list
    sdk_clients: list
    sdk_loads: list
    before_voice_privacies: list
    privacy: object
    owner: object
    memory: MemoryView
    client: object
    capture: object
    ambient: object
    archive: object
    clips: object
    targets: dict
    wake: object
    stt: object
    conversations: object
    manager: object
    persister: object
    events: object
    suggestions: object
    diagnostics: object
    bus: object
    voices: object
    encoder: object
    processor: object
    lease: object
    reader: object
    actor: MemoryReviewer
    voice_actor: VoiceReviewer


class ApplicationPeer:
    """Use the real graph reader and durable hello/reset/ACK protocol."""

    def __init__(self, bus, reader, client=None):
        self.bus, self.reader = bus, reader
        self.client = client or str(uuid4())
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.task = None

    async def message(self, document):
        await self.incoming.put({"type": "websocket.receive", "text": json.dumps(document)})

    async def start(self):
        sub, _ = await self.bus.subscribe()
        self.task = asyncio.create_task(self.bus.browser_stream(sub, self.outgoing.put,
            receiver=self.incoming.get, reader=self.reader))
        await self.message({"type": "live_hello", "version": 1, "client_id": self.client})
        reset = await self.frame("live_reset")
        await self.message({"type": "live_reset_ack", **reset["payload"]})
        await until(lambda: self.bus._retention._states[sub].browser.ready)
        return self

    async def frame(self, kind):
        async with asyncio.timeout(4):
            while True:
                frame = await self.outgoing.get()
                if frame["type"] == kind:
                    return frame

    async def close(self):
        if self.task is not None:
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
            async with asyncio.timeout(4):
                await self.task
            self.task = None


async def _bundle(graph, root, clock, config, models, home, requests, clients,
                  sdk_loads, observed):
    router, loop = graph.pipeline.request_router, graph.pipeline.audio_loop
    stt, conversations = graph.producer_coverage.collectors
    wake = loop._wake_clip_writer
    assert stt is not None and wake is not None
    clips = stt._store
    assert wake._store is conversations._store is clips
    registered = await graph.privacy.acall(
        graph.privacy.ledger.erasures.targets, ErasureParticipant.COLLECTORS)
    exports = [target for target in registered if target["version"] == 1
        and target["descriptor"] == {"kind": CollectorKind.CONVERSATION_EXPORTS.value,
            "root": str(root / "exports")}]
    assert len(exports) == 1
    targets = {"wake": wake._target, "stt": stt._target,
        "conversations": conversations._target,
        "exports": ClipTarget(exports[0]["id"], str(root / "exports"),
            CollectorKind.CONVERSATION_EXPORTS)}
    manager = graph.pipeline._orchestrator.context_manager
    assert manager.persister is not None and manager.compactor is not None
    assert ("GET", "/v1/models") in requests
    ambient = graph.ambient_path.store
    await ambient.init()
    lease = graph.dashboard_authority.resolve(TOKEN)
    permission = frozenset({DashboardPermission.CONVERSATIONS_READ})
    assert lease is not None and lease.current(permission)
    reader = lease.reader_binding(permission)
    assert reader is not None
    actor = MemoryReviewer.from_dashboard(lease.principal,
        lambda: lease.current(frozenset({DashboardPermission.MEMORY_REVIEW})))
    voice_actor = VoiceReviewer.from_dashboard(lease.principal,
        lambda: lease.current(frozenset({DashboardPermission.VOICES_REVIEW,
            DashboardPermission.ADMIN_MANAGE})), person_authority=graph.dashboard_authority)
    diagnostics = graph.erasure_services._drivers[ErasureParticipant.DIAGNOSTICS].callback.__self__
    assert router.event_logger._clock is graph.privacy.ledger.clock
    assert diagnostics.emitter._clock is graph.privacy.ledger.clock
    assert diagnostics.emitter._store.clock is graph.privacy.ledger.clock
    assert graph.producer_erasure.inventory.verify().valid
    return GraphBundle(graph=graph, root=root, clock=clock, config=config,
        models=models, home=home, model_requests=requests, sdk_clients=clients,
        sdk_loads=sdk_loads, before_voice_privacies=observed,
        privacy=graph.privacy, owner=graph.erasure_services,
        memory=MemoryView(graph.pipeline.memory), client=graph.pipeline.memory.long_term._client,
        capture=graph.producer_coverage.capture_erasure, ambient=ambient,
        archive=graph.ambient_path.transcriber._archiver, clips=clips, targets=targets,
        wake=wake, stt=stt, conversations=conversations, manager=manager,
        persister=manager.persister, events=router.event_logger,
        suggestions=router.suggestion_engine, diagnostics=diagnostics,
        bus=graph.pipeline._event_bus, voices=graph.voice_registry,
        encoder=graph.ambient_path.transcriber._tagger._identifier,
        processor=graph.pipeline.command_processor, lease=lease, reader=reader,
        actor=actor, voice_actor=voice_actor)


@asynccontextmanager
async def application_graph(root, monkeypatch, clock, *, gates=(),
                            subject_gate=None, unrelated_gate=None, before_voices=None):
    """Construct actual main owners and close them before loopback leaves leave."""
    releases = (*gates, *((subject_gate,) if subject_gate is not None else ()),
        *((unrelated_gate,) if unrelated_gate is not None else ()))
    try:
        import src.main as main_module

        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if abs(clock() - time.time()) > 30 * 60:
            raise ValueError("application_fixture_clock_must_remain_near_wall_time")
        with monkeypatch.context() as patch:
            patch.setenv("KZA_DASHBOARD_TOKEN", TOKEN)
            patch.setenv(MODEL_ENV, "synthetic-local-key")
            factories, models = nominal_factories(patch, root,
                subject=subject_gate, unrelated=unrelated_gate)
            sdk_loads = construction_sdk_guards(patch)
            clients = sdk_leaves(patch, root / "chroma")
            original_privacy = main_module.build_privacy_services
            original_voices = main_module.build_voice_registry
            observed = []

            async def privacy_factory(config, *, shutdown):
                return await original_privacy(config, shutdown=shutdown, clock=clock)

            async def voice_factory(*, privacy, owner, shutdown):
                observed.append(privacy)
                if before_voices is not None:
                    await before_voices(privacy)
                return await original_voices(privacy=privacy, owner=owner, shutdown=shutdown)

            patch.setattr(main_module, "build_privacy_services", privacy_factory)
            patch.setattr(main_module, "build_voice_registry", voice_factory)
            async with offline_home() as (home_url, home), local_models() as (model_url, requests):
                stack = AsyncExitStack()
                try:
                    config = configuration(root, home_url, model_url)
                    graph = await main_module.build_application(config, shutdown=stack, factories=factories)
                    assert isinstance(graph, ApplicationGraph)
                    assert {part.value for part in graph.erasure_services._drivers} == PARTICIPANTS
                    assert graph.erasure_services.privacy is graph.producer_erasure.privacy is graph.privacy
                    assert graph.pipeline.memory.long_term.ledger is graph.privacy.ledger
                    assert observed == [graph.privacy] and not sdk_loads
                    yield await _bundle(graph, root, clock, config, models, home, requests,
                        clients, sdk_loads, observed)
                finally:
                    for gate in releases:
                        gate.release.set()
                    await stack.aclose()
    finally:
        # Covers setup failures before servers, stack or graph exist.
        for gate in releases:
            gate.release.set()
