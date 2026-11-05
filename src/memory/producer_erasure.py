"""Nominal, lease-owned producer barriers for one revoked conversation."""
from __future__ import annotations

import asyncio
from contextvars import Context
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import ref

from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep

if TYPE_CHECKING:
    from src.application import ApplicationGraph
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.privacy_services import PrivacyServices

PRODUCER_TARGET = {"kind": "runtime_producers", "protocol": 1}
PRODUCER_ROLES = ("active_capture_loop", "command_processor", "ambient_transcriber",
                  "ambient_distiller", "orchestrator_queue")


class _BindingError(ValueError):
    """A literal first-party failure code, safe to retain as inventory metadata."""


@dataclass(frozen=True)
class ProducerCoverage:
    """Actual builder receipts, never caller-supplied success predicates."""
    fallback_audio_loop: object
    hermes_reasoners: object
    context_erasure: object
    suggestion_erasure: object
    capture_erasure: object
    collectors: tuple[object | None, ...] = ()


@dataclass(frozen=True)
class InventoryObservation:
    """Content-free coverage state; uncertainty has no fabricated finite count."""
    valid: bool
    pending_count: int | None = None
    reason: str | None = None


def _same(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _driver(owner, participant):
    registration = owner._drivers.get(participant)
    return getattr(registration.callback, "__self__", None) if registration else None


def _scope_owned(owner):
    from src.memory.producer_scopes import ProducerScope, ProducerScopes
    scopes = owner._scopes
    if not isinstance(scopes, ProducerScopes):
        return False
    snapshot = scopes._operations()
    if snapshot is None or snapshot.__self__ is not owner:
        return False
    for scope in tuple(scopes._scopes.values()):
        chain = set()
        while scope is not None:
            if not isinstance(scope, ProducerScope) or scope.token in chain:
                return False
            chain.add(scope.token)
            scope = scope._receiver
    return True


def _model_unknown(model):
    """Observe uncertainty without draining known work belonging to others."""
    from src.llm.hermes_reasoner import (_HermesLifecycle, _HERMES_PROCESS_LOCK,
                                       _HERMES_PROCESS_OWNERS)
    from src.llm.reasoner import FastRouter, HttpReasoner
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT
    from src.tts.piper_tts import DualTTS, KokoroTTS
    from src.users.speaker_identifier import SpeakerIdentifier
    from src.wakeword.detector import WakeWordDetector
    if isinstance(model, _HermesLifecycle):
        with model._hermes_condition:
            calls = tuple(model._hermes_calls)
            if model._hermes_close_uncertain or any(call.uncertain for call in calls):
                return True
            tracked = {resource for call in calls for resource in call.process_owners}
            with _HERMES_PROCESS_LOCK:
                resources = model._hermes_process_owners | {
                    resource for resource in _HERMES_PROCESS_OWNERS if resource.reasoner is model}
                return any(not resource.retired and
                           (resource not in tracked or resource.cleanup_finished or resource.phase_codes)
                           for resource in resources)
    if isinstance(model, HttpReasoner):
        state = model._http_lifecycle
        with state._adapter_condition:
            return bool(state._dispatch_uncertain or state._resource_failures or state._finish_failures)
    if isinstance(model, FastRouter):
        with model._adapter_condition:
            executor = model._router_executor
            return bool(model._router_dispatch_uncertain or model._router_work_failure
                        or executor is not None and executor._broken or model._adapter_failure)
    if isinstance(model, FastWhisperSTT):
        with model._lifecycle:
            executor, closing = model._executor, model._closing
            return bool(executor is not None and executor._broken or
                        closing is not None and closing.done() and
                        (closing.cancelled() or closing.exception() is not None))
    if isinstance(model, (DualTTS, KokoroTTS)):
        with model._adapter_condition:
            active = any(lease.active for lease in model._adapter_leases)
            return bool(model._adapter_failure or model._adapter_retained_iterators
                        and (model._adapter_sealed or not active))
    if isinstance(model, SpeakerIdentifier):
        with model._speaker_condition:
            return model._speaker_failure is not None
    if isinstance(model, (ParakeetSTT, WakeWordDetector)):
        prefix = "_parakeet" if isinstance(model, ParakeetSTT) else "_wake"
        with getattr(model, prefix + "_condition"):
            closing = getattr(model, prefix + "_close")
            return closing is not None and closing.result is not None and closing.result[0] is not None
    return True


def _borrower_supported(borrower):
    from src.ambient.ambient_stt import AmbientSTT
    from src.ambient.speaker_tagger import SpeakerTagger
    from src.ambient.transcriber import AmbientTranscriber
    from src.analytics.event_logger import EventLogger
    from src.core.owned_work import OwnedBlockingWork
    from src.llm.reasoner import FastRouter
    from src.memory.long_term import LongTermMemory
    from src.nlu.llm_router import LLMCommandRouter
    from src.orchestrator.context_manager import ContextManager
    from src.pipeline.audio_loop import AudioLoop
    from src.pipeline.command_processor import CommandProcessor
    from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
    from src.pipeline.response_handler import ResponseHandler
    from src.users.voice_registry import VoiceRegistry
    from src.vectordb.chroma_sync import ChromaSync
    return isinstance(borrower, (OwnedBlockingWork, AudioLoop, MultiRoomAudioLoop,
        CommandProcessor, AmbientTranscriber, AmbientSTT, SpeakerTagger, VoiceRegistry,
        ResponseHandler, ContextManager, FastRouter, LLMCommandRouter, EventLogger,
        LongTermMemory, ChromaSync))


class ProducerInventory:
    """Retain members but compare them with the live DI on every use."""
    def __init__(self, graph: ApplicationGraph, coverage: ProducerCoverage) -> None:
        self._graph = ref(graph)
        self.coverage, self.privacy = coverage, graph.privacy
        self._bindings = self._models = self._borrowers = ()
        self._sealed = False
        self._seal_error = "inventory_unsealed"
        self._driver = None
        self._wake_snapshot = self._observed_wake = None

    @classmethod
    def from_graph(cls, graph: ApplicationGraph, coverage: ProducerCoverage) -> ProducerInventory:
        """Seal the actual construction graph without granting source authority."""
        inventory = cls(graph, coverage)
        try:
            bindings, models, borrowers = inventory._read()
        except (AttributeError, TypeError, ValueError) as error:
            inventory._seal_error = str(error) if type(error) is _BindingError else "unsupported_binding"
            return inventory
        inventory._bindings, inventory._models, inventory._borrowers = bindings, models, borrowers
        inventory._wake_snapshot = inventory._observed_wake
        inventory._sealed = True
        return inventory

    def _read(self):
        from src.application import (ApplicationGraph, HermesReasonerOwner, STTExecutorOwner,
            TTSModelOwner, SpeakerModelOwner, WakeModelOwner, HttpReasonerOwner)
        from src.ambient.distiller import Distiller
        from src.ambient.transcriber import AmbientPath, AmbientTranscriber
        from src.orchestrator.dispatcher import MultiUserOrchestrator
        from src.pipeline.audio_loop import AudioLoop
        from src.pipeline.command_processor import CommandProcessor
        from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
        from src.pipeline.request_router import RequestRouter
        from src.pipeline.response_handler import ResponseHandler
        from src.pipeline.voice_pipeline import VoicePipeline
        graph, coverage, privacy = self._graph(), self.coverage, self.privacy
        if (not isinstance(graph, ApplicationGraph) or type(coverage) is not ProducerCoverage
                or graph.producer_coverage is not coverage or graph.privacy is not privacy
                or not privacy.ledger.authority_available):
            raise _BindingError("producer_graph_unavailable")
        pipeline = graph.pipeline
        if not isinstance(pipeline, VoicePipeline):
            raise _BindingError("unsupported_producer_pipeline")
        loop, command = pipeline.audio_loop, pipeline.command_processor
        router, responses, orchestrator = pipeline.request_router, pipeline.response_handler, pipeline._orchestrator
        fallback, ambient = graph.fallback_audio_loop, graph.ambient_path
        if (not isinstance(loop, (AudioLoop, MultiRoomAudioLoop))
                or not isinstance(fallback, AudioLoop) or fallback is not coverage.fallback_audio_loop
                or not isinstance(command, CommandProcessor) or not isinstance(router, RequestRouter)
                or not isinstance(responses, ResponseHandler)
                or router.command_processor is not command or router._orchestrator is not orchestrator
                or router.response_handler is not responses or router._source_policy is not privacy.ledger):
            raise _BindingError("producer_binding_mismatch")
        ambient_config = graph.config.get("ambient", {}) or {}
        if (ambient is None and ambient_config.get("enabled", False)
                and isinstance(loop, MultiRoomAudioLoop)):
            raise _BindingError("missing_ambient_producer")
        if orchestrator is None and (graph.config.get("orchestrator", {}) or {}).get("enabled", True):
            raise _BindingError("missing_orchestrator_producer")
        transcriber = distiller = None
        if ambient is not None:
            if not isinstance(ambient, AmbientPath):
                raise _BindingError("unsupported_ambient_path")
            transcriber, distiller = ambient.transcriber, ambient.distiller
            if (not isinstance(transcriber, AmbientTranscriber)
                    or ambient.store.source_policy is not privacy.ledger
                    or transcriber._store is not ambient.store):
                raise _BindingError("ambient_authority_mismatch")
            if (distiller is None and pipeline.memory is not None
                    and not ambient_config.get("shadow_mode", True)):
                raise _BindingError("missing_distiller_producer")
        members = (loop, command, transcriber, distiller, orchestrator, fallback)
        expected = ((AudioLoop, MultiRoomAudioLoop), (CommandProcessor,), (AmbientTranscriber,),
                    (Distiller,), (MultiUserOrchestrator,), (AudioLoop,))
        for member, types in zip(members, expected):
            if member is None:
                continue
            if not isinstance(member, types) or not _scope_owned(member):
                raise _BindingError("unsupported_producer_owner")
            authority = (member._policy if isinstance(member, Distiller) else
                         member._source_policy if isinstance(member, MultiUserOrchestrator) else member._privacy)
            expected_authority = privacy.ledger if isinstance(member, (Distiller, MultiUserOrchestrator)) else privacy
            if authority is not expected_authority:
                raise _BindingError("producer_authority_mismatch")
        owner = graph.erasure_services
        if owner.privacy is not privacy:
            raise _BindingError("producer_coordinator_mismatch")
        delegates = self._delegates(graph, owner, router, orchestrator, ambient)
        model_owners = (graph.stt_executors, graph.tts_models, graph.speaker_models,
                        graph.wake_models, graph.http_reasoners, graph.hermes_reasoners)
        if graph.hermes_reasoners is not coverage.hermes_reasoners:
            raise _BindingError("producer_model_owner_mismatch")
        nominal_owners = (STTExecutorOwner, TTSModelOwner, SpeakerModelOwner,
                          WakeModelOwner, HttpReasonerOwner, HermesReasonerOwner)
        models, borrowers = [], []
        manager = fallback.audio_manager
        with manager._wake_inventory_lock:
            wake_owner, factory, wake_model = manager.wake_inventory()
            from src.wakeword.detector import WakeWordDetector
            if wake_owner is not graph.wake_models or factory is not WakeWordDetector:
                raise _BindingError("producer_wake_factory_mismatch")
            if wake_model is not None and (type(wake_model) is not WakeWordDetector
                    or not any(value is wake_model for value in wake_owner._models.values())):
                raise _BindingError("producer_wake_registration_missing")
            self._observed_wake = (manager, wake_owner, factory, wake_model)
            for model_owner, nominal in zip(model_owners, nominal_owners):
                if not isinstance(model_owner, nominal) or model_owner._capture_unreviewed:
                    raise _BindingError("unsupported_producer_model_owner")
                if any(not isinstance(model, model_owner._model_types) for model in model_owner._models.values()):
                    raise _BindingError("producer_model_registration_mismatch")
                models.extend(model_owner._models.values())
                for begin in model_owner._borrowers:
                    borrower = getattr(begin, "__self__", None)
                    if not _borrower_supported(borrower):
                        raise _BindingError("unsupported_producer_borrower")
                    borrowers.extend((borrower, begin.__func__))
            leaves = self._used_models(graph, model_owners)
        # FastRouter has its own privacy borrower, rather than a model owner.
        if router.router is not None:
            from src.llm.reasoner import FastRouter
            if not isinstance(router.router, FastRouter):
                raise _BindingError("unsupported_producer_router")
            models.append(router.router)
        classifier = router.llm_command_router
        if classifier is not None:
            from src.nlu.llm_router import LLMCommandRouter
            if (not isinstance(classifier, LLMCommandRouter)
                    or classifier.source_policy is not privacy.ledger
                    or classifier.router is not router.router):
                raise _BindingError("producer_classifier_mismatch")
        bindings = (graph.pipeline, coverage, owner, *members, ambient, router, responses,
                    pipeline._event_bus, classifier, router.llm, manager, factory,
                    *model_owners, *delegates, *leaves)
        return bindings, tuple(models), tuple(borrowers)

    def _used_models(self, graph, owners):
        """Prove that used model leaves and their wrappers share the leased authority."""
        from src.ambient.ambient_stt import AmbientSTT
        from src.ambient.speaker_tagger import SpeakerTagger
        from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
        stt, tts, speaker, wake, http, hermes = owners
        pipeline, models = graph.pipeline, graph.models
        command, response = pipeline.command_processor, pipeline.response_handler
        if (models.stt is not command.stt or models.tts is not response.tts
                or models.speaker_identifier is not command.speaker_id
                or models.emotion_detector is not command.emotion_detector
                or graph.voice_services.responses is not response):
            raise _BindingError("producer_model_alias_mismatch")
        leaves = [models, command.stt, command.shadow_stt, command.speaker_id,
                  command.emotion_detector, response.tts, response.llm]
        def registered(owner, model):
            if model is not None and not any(value is model for value in owner._models.values()):
                raise _BindingError("producer_model_owner_missing")
        registered(stt, command.stt)
        registered(stt, command.shadow_stt)
        registered(speaker, command.speaker_id)
        registered(tts, response.tts)
        for model in (response.llm, pipeline.request_router.llm):
            if model is not None and not any(value is model for owner in (http, hermes)
                                             for value in owner._models.values()):
                raise _BindingError("producer_reasoner_owner_missing")
        loop, fallback = pipeline.audio_loop, graph.fallback_audio_loop
        for owner, members in ((stt, (command, loop, fallback)),
                (speaker, (command, loop, fallback)), (wake, (loop, fallback)),
                (tts, (response,)), (http, (command, response)), (hermes, (command, response))):
            for member in members:
                if not any(getattr(begin, "__self__", None) is member for begin in owner._borrowers):
                    raise _BindingError("producer_required_borrower_missing")
        if isinstance(loop, MultiRoomAudioLoop):
            if loop._stt is not command.stt:
                raise _BindingError("producer_room_stt_mismatch")
            leaves.extend((loop._inputs, *loop._capture_rooms.values()))
            for room in loop.room_streams.values():
                registered(wake, room.wake_detector)
                leaves.extend((room, room.wake_detector))
        if graph.ambient_path is not None:
            transcriber = graph.ambient_path.transcriber
            if (not isinstance(transcriber._stt, AmbientSTT)
                    or not isinstance(transcriber._tagger, SpeakerTagger)
                    or transcriber._stt._source_policy is not self.privacy.ledger
                    or transcriber._tagger._source_policy is not self.privacy.ledger
                    or transcriber._tagger._voice_registry is not graph.voice_registry):
                raise _BindingError("producer_ambient_wrapper_authority_mismatch")
            registered(stt, transcriber._stt._stt)
            registered(speaker, transcriber._tagger._identifier)
            leaves.extend((transcriber._stt, transcriber._stt._stt,
                           transcriber._tagger, transcriber._tagger._identifier,
                           transcriber._tap, transcriber._room_states))
        return tuple(leaves)

    def _delegates(self, graph, owner, router, orchestrator, ambient):
        from src.ambient.erasure import CaptureEraser
        from src.analytics.suggestion_erasure import SuggestionEraser
        from src.dashboard.live_erasure import LiveStreamEraser
        from src.orchestrator.context_erasure import ContextEraser
        from src.training.collector_erasure import CollectorEraser
        from src.training.conversation_collector import ConversationCollector
        from src.training.stt_correction_collector import STTCorrectionCollector
        from src.wakeword.wake_clip_writer import WakeClipWriter
        from src.users.voice_registry import VoiceRegistry
        coverage, privacy = self.coverage, self.privacy
        contexts = _driver(owner, ErasureParticipant.CONTEXTS)
        suggestions = _driver(owner, ErasureParticipant.SUGGESTIONS)
        captures = _driver(owner, ErasureParticipant.CAPTURES)
        collectors = _driver(owner, ErasureParticipant.COLLECTORS)
        streams = _driver(owner, ErasureParticipant.STREAMS)
        voices = _driver(owner, ErasureParticipant.VOICES)
        if (not isinstance(contexts, ContextEraser) or contexts is not coverage.context_erasure
                or not isinstance(suggestions, SuggestionEraser) or suggestions is not coverage.suggestion_erasure
                or not isinstance(captures, CaptureEraser) or captures is not coverage.capture_erasure
                or not isinstance(collectors, CollectorEraser) or not isinstance(streams, LiveStreamEraser)
                or not isinstance(voices, VoiceRegistry) or voices is not graph.voice_registry
                or any(value.privacy is not privacy for value in
                       (contexts, suggestions, captures, collectors, streams, voices))
                or router not in contexts._routers or router not in suggestions._routers
                or streams.bus is not graph.pipeline._event_bus
                or streams.receipts.ledger is not privacy.ledger):
            raise _BindingError("producer_delegate_mismatch")
        manager = orchestrator.context_manager if orchestrator is not None else None
        if manager is not None and (manager not in contexts._managers
                or manager.source_policy is not privacy.ledger
                or manager.persister is not None and manager.persister.source_policy is not privacy.ledger):
            raise _BindingError("producer_context_attachment_missing")
        engine = router.suggestion_engine
        if engine is not None and (engine not in suggestions._engines or engine.store.policy is not privacy.ledger):
            raise _BindingError("producer_suggestion_attachment_missing")
        if ambient is not None:
            archiver = ambient.transcriber._archiver
            if archiver is not None and (archiver not in captures._archivers
                    or archiver._privacy is not privacy or archiver._store is not ambient.store):
                raise _BindingError("producer_capture_attachment_missing")
        if (len(coverage.collectors) != 2
                or router.conversation_collector is not coverage.collectors[1]):
            raise _BindingError("producer_collector_binding_mismatch")
        for collector, nominal in zip(coverage.collectors, (STTCorrectionCollector, ConversationCollector)):
            if collector is not None and (collector._store is not collectors.store
                    or not isinstance(collector, nominal) or collector not in collectors.store._writers
                    or collector._store.privacy is not privacy):
                raise _BindingError("producer_collector_attachment_missing")
        writers = tuple(sorted(collectors.store._writers, key=id))
        wake_writer = getattr(graph.pipeline.audio_loop, "_wake_clip_writer", None)
        for writer in writers:
            if not any(writer is value for value in coverage.collectors):
                if not isinstance(writer, WakeClipWriter) or writer is not wake_writer:
                    raise _BindingError("producer_uncovered_writer")
        if wake_writer is not None and (wake_writer not in writers
                or wake_writer._store is not collectors.store):
            raise _BindingError("producer_wake_writer_attachment_missing")
        return (contexts, suggestions, captures, collectors, streams, voices, manager, engine,
                *coverage.collectors, *writers)

    def verify(self) -> InventoryObservation:
        """Invalidate a changed graph permanently; observe native uncertainty separately."""
        if not self._sealed:
            return InventoryObservation(False, reason=self._seal_error)
        try:
            bindings, models, borrowers = self._read()
            if (not _same(bindings, self._bindings) or not self._wake_registration(models)
                    or not _same(borrowers, self._borrowers)
                    or self._graph().producer_erasure is not self._driver
                    or _driver(self._graph().erasure_services, ErasureParticipant.PRODUCERS) is not self._driver):
                self._sealed = False
                self._seal_error = "binding_replaced"
                return InventoryObservation(False, reason=self._seal_error)
            if any(_model_unknown(model) for model in models):
                return InventoryObservation(False, reason="native_retirement_unknown")
            graph = self._graph()
            from src.pipeline.audio_loop import AudioLoop
            for loop in (graph.pipeline.audio_loop, graph.fallback_audio_loop):
                if isinstance(loop, AudioLoop):
                    if loop._input.retirement_unknown:
                        return InventoryObservation(False, reason="input_retirement_unknown")
                elif loop._inputs.resource_status()["retirement_unknown"]:
                    return InventoryObservation(False, reason="input_retirement_unknown")
            if graph.pipeline.response_handler.get_delivery_status()["failed_barriers"]:
                return InventoryObservation(False, reason="response_retirement_unknown")
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._sealed = False
            self._seal_error = str(error) if type(error) is _BindingError else "unsupported_binding"
            return InventoryObservation(False, reason=self._seal_error)
        return InventoryObservation(True, 0)

    def _wake_registration(self, models) -> bool:
        """Adopt only the reviewed lazy constructor's single registered result."""
        previous, current = self._wake_snapshot, self._observed_wake
        if previous is None or current is None or not _same(previous[:3], current[:3]):
            return False
        if previous[3] is current[3]:
            return _same(models, self._models)
        if previous[3] is not None or current[3] is None:
            return False
        model = current[3]
        if any(value is model for value in self._models):
            return False
        if sum(value is model for value in models) != 1 or not _same(
                tuple(value for value in models if value is not model), self._models):
            return False
        self._models, self._wake_snapshot = models, current
        return True

    async def sweep(self, conversation_id: str) -> InventoryObservation:
        """Prune revoked associations and observe actual per-conversation barriers."""
        if not self.verify().valid:
            return InventoryObservation(False)
        graph = self._graph()
        loop, fallback = graph.pipeline.audio_loop, graph.fallback_audio_loop
        # Avoid queuing cleanup behind a known unrelated room's native call.
        if loop.conversation_busy(conversation_id):
            await loop.prune_private()
        if fallback is not loop and fallback.conversation_busy(conversation_id):
            await fallback.prune_private()
        command = graph.pipeline.command_processor
        command.prune_private()
        ambient = graph.ambient_path
        if ambient is not None:
            ambient.transcriber.prune_private()
        return self.observe(conversation_id)

    def observe(self, conversation_id: str) -> InventoryObservation:
        """No await between the last barrier observation and returning a receipt."""
        if not self.verify().valid:
            return InventoryObservation(False)
        graph = self._graph()
        loop, command, fallback = graph.pipeline.audio_loop, graph.pipeline.command_processor, graph.fallback_audio_loop
        ambient = graph.ambient_path
        members = (loop, command, fallback, ambient.transcriber if ambient else None,
                   ambient.distiller if ambient else None, graph.pipeline._orchestrator)
        pending = sum(bool(member.conversation_busy(conversation_id))
                      for member in dict.fromkeys(members) if member is not None)
        return InventoryObservation(True, pending) if self.verify().valid else InventoryObservation(False)


class ProducersEraser:
    """Own cleanup observers without acquiring authority over upstream shutdown."""
    def __init__(self, privacy: PrivacyServices, inventory: ProducerInventory) -> None:
        self.privacy, self.inventory = privacy, inventory
        self._jobs = {}
        self._closed = False
        self._closing = None

    async def _clean(self, step):
        journal = self.privacy.ledger.erasures
        if not await self.privacy.acall(journal.is_current_claim, step):
            return ErasureResult(False, cursor="stale_claim")
        if not self.privacy.ledger.conversation_committed_revoked(step.conversation_id):
            return ErasureResult(False, cursor="authority_unconfirmed")
        try:
            observation = await self.inventory.sweep(step.conversation_id)
        except (AttributeError, TypeError, ValueError, RuntimeError, PermissionError):
            return ErasureResult(False, cursor="inventory_unknown")
        if not observation.valid:
            return ErasureResult(False, cursor="inventory_unknown")
        if observation.pending_count:
            return ErasureResult(False, pending_count=observation.pending_count)
        if not await self.privacy.acall(journal.is_current_claim, step):
            return ErasureResult(False, cursor="stale_claim")
        try:
            final = self.inventory.observe(step.conversation_id)
        except (AttributeError, TypeError, ValueError, RuntimeError, PermissionError):
            return ErasureResult(False, cursor="inventory_unknown")
        if not final.valid:
            return ErasureResult(False, cursor="inventory_unknown")
        if final.pending_count:
            return ErasureResult(False, pending_count=final.pending_count)
        return ErasureResult(True, pending_count=0)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Keep the actual cleanup alive when a receipt observer is cancelled."""
        if (type(step) is not ErasureStep or step.participant is not ErasureParticipant.PRODUCERS
                or step.version != 1 or step.descriptor != PRODUCER_TARGET):
            return ErasureResult(False, cursor="unsupported_target")
        if self._closed:
            return ErasureResult(False, cursor="inventory_closed")
        key = (step.operation_id, step.target_id, step.version, step.claim_token)
        task = self._jobs.get(key)
        if task is None:
            if len(self._jobs) >= 4:
                return ErasureResult(False, cursor="inventory_capacity")
            task = asyncio.create_task(self._clean(step), context=Context())
            self._jobs[key] = task
            def finished(done):
                self._jobs.pop(key, None)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
        return await asyncio.shield(task)

    def begin_close(self) -> asyncio.Task:
        """Close only own admission and retain the drain of own admitted jobs."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*tuple(self._jobs.values()), return_exceptions=True)
            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Observe retained cleanup without cancelling it on observer exit."""
        await asyncio.shield(self.begin_close())


async def build_producer_erasure(*, privacy: PrivacyServices, owner: ErasureCoordinator,
                                 graph: ApplicationGraph, coverage: ProducerCoverage) -> ProducersEraser:
    """Bind shutdown before the first fallible target-registration await."""
    if owner.privacy is not privacy or graph.privacy is not privacy or graph.erasure_services is not owner:
        raise _BindingError("producer_privacy_mismatch")
    driver = ProducersEraser(privacy, ProducerInventory.from_graph(graph, coverage))
    driver.inventory._driver = driver
    graph.producer_erasure = driver
    owner.register(ErasureParticipant.PRODUCERS, driver.clean, versions=frozenset({1}),
                   begin_close=driver.begin_close)
    privacy.add_borrower(driver.begin_close)
    await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.PRODUCERS,
                        1, PRODUCER_TARGET)
    return driver
