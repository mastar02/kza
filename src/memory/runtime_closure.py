"""Observe graph-owned retirement without reviving a closed authority.

This module can be loaded under a trusted bridge alias while ``src`` belongs to
an older C-compatible artifact. It only observes that artifact's nominal owners;
it never patches their ledger, methods, namespace or close outcomes.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from types import MethodType


class RuntimeClosureError(RuntimeError):
    """A content-free failure to establish owned graph closure."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeClosureError("runtime_closure_" + code)


def _finished(task) -> bool:
    return (isinstance(task, asyncio.Task) and task.done()
            and not task.cancelled() and task.exception() is None)


def _adapter_closed(adapter, *, asynchronous: bool = True) -> bool:
    with adapter._adapter_condition:
        admission = adapter._adapter_close_admission
        return bool(adapter._adapter_sealed and admission is not None
                    and admission.finished and admission.failure is None
                    and adapter._adapter_failure is None
                    and not any(lease.active for lease in adapter._adapter_leases)
                    and not adapter._adapter_retained_iterators
                    and (not asynchronous or _finished(adapter._adapter_closing)))


def _native_closed(model, *, asynchronous: bool = True) -> bool:
    """Read positive terminal receipts of the supported C lifecycle classes."""
    from src.llm.hermes_reasoner import HermesCliReasoner
    from src.llm.reasoner import FastRouter, HttpReasoner
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT
    from src.tts.piper_tts import DualTTS, KokoroTTS, PiperTTS
    from src.users.speaker_identifier import SpeakerIdentifier
    from src.wakeword.detector import WakeWordDetector

    if type(model) in (ParakeetSTT, WakeWordDetector):
        prefix = "_parakeet" if type(model) is ParakeetSTT else "_wake"
        with getattr(model, prefix + "_condition"):
            receipt = getattr(model, prefix + "_close")
            released = (model._model is None if type(model) is ParakeetSTT else
                        model._oww_model is None and not model._custom_models
                        and not model._loaded and not model._wake_has_backend)
            return bool(getattr(model, prefix + "_sealed") and receipt is not None
                        and receipt.result == (None,)
                        and not getattr(model, prefix + "_active")
                        and released
                        and (not asynchronous or _finished(getattr(model, prefix + "_closing"))))
    if type(model) is SpeakerIdentifier:
        with model._speaker_condition:
            receipt = model._speaker_close_admission
            return bool(model._speaker_sealed and receipt is not None and receipt.finished
                        and receipt.failure is None and model._speaker_failure is None
                        and not any(lease.active for lease in model._speaker_leases)
                        and model._model is None and model._model_type is None
                        and not model._embeddings_cache
                        and (not asynchronous or _finished(model._speaker_closing)))
    if type(model) is FastWhisperSTT:
        with model._lifecycle:
            return bool(model._closed and model._close_finished.is_set()
                        and not model._close_failed and not model._admitted
                        and model._model is None and _finished(model._closing))
    if type(model) in (DualTTS, KokoroTTS, PiperTTS):
        if not _adapter_closed(model, asynchronous=asynchronous):
            return False
        if type(model) is DualTTS:
            return all(_native_closed(child, asynchronous=False)
                       for child in model._owned_tts_engines)
        return model._model is None if type(model) is KokoroTTS else model._voice is None
    if type(model) is FastRouter:
        return bool(_adapter_closed(model, asynchronous=asynchronous)
                    and model._router_executor is None and not model._router_work
                    and not model._router_native_futures and not model._router_dispatch_uncertain
                    and not model._router_work_failure and model._client is None)
    if type(model) is HttpReasoner:
        state = model._http_lifecycle
        return bool(_adapter_closed(state, asynchronous=asynchronous)
                    and not state._resource_failures and not state._finish_failures
                    and not state._dispatch_uncertain and not state._close_work
                    and not state._async_streams and not state._initializing
                    and not state._bundles and model._client is None
                    and all(item.source is None and item.iterator is None
                            for item in state._streams.values()))
    if type(model) is HermesCliReasoner:
        with model._hermes_condition:
            return bool(model._hermes_sealed and _finished(model._hermes_closing)
                        and not model._hermes_pending_locked()
                        and not model._hermes_failed_locked())
    return False


def _supported_native(model) -> bool:
    from src.llm.hermes_reasoner import HermesCliReasoner
    from src.llm.reasoner import FastRouter, HttpReasoner
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT
    from src.tts.piper_tts import DualTTS, KokoroTTS, PiperTTS
    from src.users.speaker_identifier import SpeakerIdentifier
    from src.wakeword.detector import WakeWordDetector

    if type(model) is DualTTS:
        return (model.qwen3 is None and model.kokoro in model._owned_tts_engines
                and all(_supported_native(child) for child in model._owned_tts_engines))
    return type(model) in (ParakeetSTT, FastWhisperSTT, KokoroTTS, PiperTTS,
                           SpeakerIdentifier, WakeWordDetector, FastRouter,
                           HttpReasoner, HermesCliReasoner)


_EDGES = (
    "pipeline", "privacy", "ledger", "models", "stt_executors", "tts_models",
    "speaker_models", "wake_models", "http_reasoners", "hermes_reasoners",
    "voice_registry", "voice_services", "erasure_services", "producer_coverage",
    "producer_erasure", "inventory", "ambient_path", "fallback_audio_loop",
    "capture_control", "dashboard_authority", "audio_loop", "command_processor",
    "request_router", "response_handler", "_orchestrator", "_event_bus", "memory",
    "chroma", "stt", "shadow_stt", "speaker_id", "emotion_detector", "tts", "llm",
    "router", "llm_command_router", "responses", "context_manager", "manager",
    "long_term", "_privacy", "_source_policy", "source_policy", "_policy",
    "_store", "store", "_stt", "_tagger", "_voice_registry", "_identifier",
    "_archiver", "_tap", "_room_states", "_inputs", "audio_manager",
    "wake_detector", "_wake_models", "_wake_factory", "_http_lifecycle",
    "_router_session", "kokoro", "qwen3", "_stop_services",
)
_MEMBERS = ("_borrowers", "_drivers", "_routers", "_managers", "_engines", "_writers",
            "_owned_tts_engines", "_capture_rooms", "room_streams")
_PRIVACY_BORROWERS = {
    "src.application": ("STTExecutorOwner", "TTSModelOwner", "SpeakerModelOwner",
                        "WakeModelOwner", "HttpReasonerOwner", "HermesReasonerOwner"),
    "src.vectordb.chroma_sync": ("ChromaSync",),
    "src.llm.reasoner": ("FastRouter",),
    "src.memory.services": ("MemoryServices",),
    "src.memory.long_term": ("LongTermMemory",),
    "src.memory.erasure_coordinator": ("ErasureCoordinator",),
    "src.memory.producer_erasure": ("ProducersEraser",),
    "src.memory.capture_control": ("CaptureControl",),
    "src.users.voice_registry": ("VoiceRegistry",),
    "src.wakeword.clip_store": ("WakeClipStore",),
    "src.training.collector_erasure": ("CollectorEraser",),
    "src.ambient.control": ("AmbientControl",),
    "src.ambient.transcriber": ("AmbientPath", "AmbientTranscriber"),
    "src.ambient.distiller": ("Distiller",),
    "src.ambient.audio_event_detector": ("AudioEventDetector",),
    "src.dashboard.live_event_bus": ("LiveEventBus",),
    "src.orchestrator.context_manager": ("ContextManager",),
    "src.analytics.event_logger": ("EventLogger",),
    "src.obs.metrics_emitter": ("MetricsEmitter",),
    "src.pipeline.audio_loop": ("AudioLoop",),
    "src.pipeline.multi_room_audio_loop": ("MultiRoomAudioLoop",),
    "src.pipeline.command_processor": ("CommandProcessor",),
    "src.pipeline.request_router": ("RequestRouter",),
}


def _member_values(container) -> tuple:
    if isinstance(container, dict):
        return tuple(item for pair in container.items() for item in pair)
    return tuple(container)


def _member_tokens(container) -> tuple:
    if isinstance(container, dict):
        return tuple(sorted((id(key), id(value)) for key, value in container.items()))
    return tuple(sorted(map(id, container)))


def _same_edge(current, captured) -> bool:
    if type(current) is MethodType and type(captured) is MethodType:
        return current.__self__ is captured.__self__ and current.__func__ is captured.__func__
    return current is captured


class _ChromaConstruction:
    """Observe the real lazy constructor without replacing SDK ownership."""

    def __init__(self, owner):
        original = owner._construct
        _require(type(original) is MethodType and original.__self__ is owner
                 and original.__func__ is type(owner)._construct, "sdk_constructor_changed")
        with owner._condition:
            _require(not owner._active and not owner._admission_closed
                     and owner._closing is None and not owner._client_failed
                     and not owner._client_closed, "sdk_owner_not_fresh")
            _require(bool(owner._ready) == (owner._client is not None), "sdk_handle_unconfirmed")
        self.owner, self._original = owner, original
        self._lock = threading.RLock()
        self.client = owner._client
        self.started = self.returned = self.client is not None
        self.inflight = self.failed = False
        self._observer = self._observe
        owner._construct = self._observer

    def _observe(self) -> None:
        # initialize() retains its own operation/initializer locks. Its close
        # owner cannot finish until this observation and the real call return.
        with self._lock:
            _require(not self.started and not self.failed
                     and getattr(self.owner._local, "depth", 0) > 0,
                     "sdk_constructor_not_owned")
            self.started = self.inflight = True
        try:
            self._original()
            with self._lock:
                _require(self.owner._client is not None and self.client is None,
                         "sdk_handle_unconfirmed")
                self.client = self.owner._client
                self.returned = True
        except BaseException:
            with self._lock:
                self.failed = True
            raise
        finally:
            with self._lock:
                self.inflight = False

    def before_close(self) -> None:
        with self._lock:
            _require(self.owner._construct is self._observer and not self.failed,
                     "sdk_binding_changed")
            # An admitted initializer can still be running. The real close
            # drains that operation; its actual return must publish the handle.
            if not self.inflight:
                expected = None if self.owner._client_closed else self.client
                _require(self.owner._client is expected, "sdk_binding_changed")

    def verify_closed(self) -> bool:
        with self._lock:
            _require(self.owner._construct is self._observer and not self.failed
                     and not self.inflight, "sdk_construction_unconfirmed")
            state = self.owner.resource_status()
            _require(state["closed"] and state["close_phase"] == "complete"
                     and _finished(self.owner._closing)
                     and not state["operations"] and not state["async_operations"]
                     and not state["client_close_failed"] and self.owner._client is None
                     and not self.owner._ready, "sdk_not_closed")
            if self.started:
                _require(self.returned and self.client is not None and state["client_closed"],
                         "sdk_close_unconfirmed")
            else:
                _require(not self.returned and self.client is None and not state["client_closed"],
                         "sdk_never_opened_unconfirmed")
            return self.started


class _Pins:
    """Strong references to stable DI edges, never native payload fields."""

    def __init__(self, objects, *, model_owners, wake_owner):
        self.attributes, self.members = [], []
        seen = set()
        for obj in objects:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            for name in _EDGES:
                if hasattr(obj, name):
                    self.attributes.append((obj, name, getattr(obj, name)))
            names = _MEMBERS + (("_models",) if any(obj is owner for owner in model_owners)
                                and obj is not wake_owner else ())
            for name in names:
                if hasattr(obj, name):
                    container = getattr(obj, name)
                    self.members.append((obj, name, container, _member_values(container),
                                         _member_tokens(container)))

    def verify(self) -> None:
        for obj, name, value in self.attributes:
            _require(_same_edge(getattr(obj, name), value), "binding_changed")
        for obj, name, container, members, tokens in self.members:
            _require(getattr(obj, name) is container, "binding_changed")
            # WeakSets are unordered; retained members prevent identity reuse.
            _require(len(_member_values(container)) == len(members)
                     and _member_tokens(container) == tokens, "membership_changed")


class RuntimeClosureWitness:
    """Pin live C ownership, then observe its actual normal retirement."""

    @classmethod
    def capture(cls, graph):
        from src.application import ApplicationGraph
        from src.memory.privacy_services import PrivacyServices
        from src.memory.producer_erasure import ProducerInventory, ProducersEraser
        from src.memory.services import MemoryServices
        from src.vectordb.chroma_sync import ChromaSync

        _require(type(graph) is ApplicationGraph and type(graph.privacy) is PrivacyServices,
                 "public_graph_required")
        driver = graph.producer_erasure
        _require(type(driver) is ProducersEraser and type(driver.inventory) is ProducerInventory,
                 "inventory_required")
        inventory = driver.inventory
        _require(inventory.verify().valid, "inventory_invalid")
        self = cls()
        self._failed = self._prepared = False
        self._graph, self._privacy, self._inventory = graph, graph.privacy, inventory
        self._owners = (graph.stt_executors, graph.tts_models, graph.speaker_models,
                        graph.wake_models, graph.http_reasoners, graph.hermes_reasoners)
        borrowers = []
        for reference in graph.privacy._borrowers:
            begin = reference()
            owner = getattr(begin, "__self__", None)
            allowed = _PRIVACY_BORROWERS.get(type(owner).__module__, ())
            module = sys.modules.get(type(owner).__module__)
            _require(any(type(owner) is getattr(module, name, None) for name in allowed),
                     "unsupported_privacy_borrower")
            borrowers.append(owner)
        self._borrowers = tuple(borrowers)
        self._chroma = graph.models.chroma
        _require(type(self._chroma) is ChromaSync and graph.pipeline.chroma is self._chroma,
                 "sdk_owner_required")
        memories = [owner for owner in borrowers if type(owner) is MemoryServices]
        _require(len(memories) == int(graph.pipeline.memory is not None), "memory_owner_missing")
        self._memory = memories[0] if memories else None
        if self._memory is not None:
            _require(self._memory.manager is graph.pipeline.memory and self._memory._owns_client
                     and self._memory._client is not None, "memory_sdk_not_owned")
        self._memory_client = self._memory._client if self._memory is not None else None
        _require(graph.models.emotion_detector is None, "unsupported_emotion_owner")
        self._manager = graph.fallback_audio_loop.audio_manager
        _require(all(_supported_native(model) for model in inventory._models),
                 "unsupported_native_owner")
        self._chroma_construction = _ChromaConstruction(self._chroma)
        self._pins = _Pins((graph, graph.privacy, graph.pipeline, graph.models, driver, inventory,
                            *inventory._bindings, *inventory._models, *self._owners, *borrowers,
                            self._memory, self._chroma), model_owners=self._owners,
                           wake_owner=graph.wake_models)
        self._models = tuple(inventory._models)
        self._wake_models = tuple(graph.wake_models._models.values())
        _require(inventory.verify().valid, "inventory_invalid")
        return self

    async def pre_close(self) -> None:
        """Run first on the owned exit stack, while authority is still live."""
        _require(not self._failed, "witness_failed")
        if self._prepared:
            return
        try:
            self._pins.verify()
            self._chroma_construction.before_close()
            _require(self._memory is None or self._memory._client is self._memory_client,
                     "sdk_binding_changed")
            _require(self._inventory.verify().valid, "inventory_invalid")
            # verify() alone may adopt the reviewed None->registered lazy wake.
            # No other model membership change is accepted by the C inventory.
            self._models = tuple(self._inventory._models)
            _require(all(_supported_native(model) for model in self._models),
                     "unsupported_native_owner")
            self._wake_models = tuple(self._graph.wake_models._models.values())
            self._inputs = tuple((loop, loop._input if hasattr(loop, "_input") else loop._inputs)
                                 for loop in {self._graph.pipeline.audio_loop,
                                              self._graph.fallback_audio_loop})
            self._prepared = True
        except BaseException:
            self._failed = True
            raise

    def verify_closed(self) -> dict:
        """Return content-free proof only after positive owned close receipts."""
        _require(not self._failed and self._prepared, "pre_close_required")
        try:
            self._pins.verify()
            current_wake = tuple(self._graph.wake_models._models.values())
            _require(len(current_wake) == len(self._wake_models)
                     and all(left is right for left, right in zip(current_wake, self._wake_models)),
                     "late_wake_registration")
            _require(self._manager._wake_closed and self._manager.wake_inventory()[2] is None,
                     "wake_admission_not_closed")
            status = self._privacy.resource_status()
            _require(status["close_phase"] == "complete" and not status["writer_lease"]
                     and not status["closing"] and _finished(self._privacy._closing),
                     "privacy_not_closed")
            _require(_finished(self._graph.pipeline._stopping), "pipeline_not_closed")
            for owner in self._owners:
                _require(not owner._capture_unreviewed and _finished(owner._closing),
                         "model_owner_not_closed")
            for model in self._models:
                _require(_native_closed(model), "native_not_closed")
            for loop, inputs in self._inputs:
                _require(_finished(loop._closing), "input_owner_not_closed")
                _require((loop._input if hasattr(loop, "_input") else loop._inputs) is inputs,
                         "input_binding_changed")
                state = inputs.resource_status()
                _require(state["closed"] and state["retirement_confirmed"]
                         and not state["retirement_unknown"] and not state["operations"],
                         "input_not_retired")
            response = self._graph.pipeline.response_handler
            delivery = response.get_delivery_status()
            _require(_finished(response._shutdown_task) and delivery["closed"]
                     and all(value == 0 for key, value in delivery.items() if key != "closed"),
                     "response_not_closed")
            chroma_opened = self._chroma_construction.verify_closed()
            if self._memory is not None:
                memory = self._memory.resource_status()
                _require(memory["closed"] and memory["close_phase"] == "complete"
                         and memory["client_closed"] and not memory["client_close_failed"]
                         and not memory["building"] and not memory["closing"]
                         and self._memory._client is self._memory_client, "memory_not_closed")
            return {"schema_version": 1, "bindings_unchanged": True,
                    "model_owners_closed": True, "graph_closed": True,
                    "privacy_closed": True, "sdk_closed": True, "native_closed": True,
                    "model_owner_count": len(self._owners), "model_count": len(self._models),
                    "sdk_owner_count": 1 + int(self._memory is not None),
                    "chroma_opened": chroma_opened, "chroma_never_opened": not chroma_opened,
                    "input_owner_count": len({self._graph.pipeline.audio_loop,
                                              self._graph.fallback_audio_loop})}
        except BaseException:
            self._failed = True
            raise
