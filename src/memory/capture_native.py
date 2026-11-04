"""Nonterminal observation of the graph's already-owned native capture work.

The caller seals capture/command admission first. Model lease predicates are
checked only after the registered borrowers' actual work receipts have drained;
an inactive model lease alone is not evidence of a returned native callable.
These callbacks retain caches, executors, models and historical read authority.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable

from src.core.owned_work import OwnedBlockingWork


_UNAVAILABLE = "capture_native_retirement_unconfirmed"


async def _tasks_finished(tasks) -> None:
    pending = tuple(task for task in tasks if not task.done())
    try:
        if asyncio.current_task() in pending:
            raise RuntimeError(_UNAVAILABLE)
        if pending:
            await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
    finally:
        tasks = pending = None


async def _snapshot_finished(snapshot) -> None:
    while True:
        tasks = snapshot()
        try:
            if not any(not task.done() for task in tasks):
                return
            await _tasks_finished(tasks)
        finally:
            tasks = None


async def _idle(predicate) -> None:
    while not predicate():
        await asyncio.sleep(.005)


def build_capture_native_drains(
    *, model_owners=(), router=None, responses=None, context_manager=None,
) -> tuple[tuple[Callable[[], Awaitable[None]], ...], bool]:
    """Bind only reviewed graph owners; unknown participants fail closed.

    Returned regular callables run caller guards before returning an awaitable.
    Their retained owner is CaptureControl, not a terminal model closer or a new
    executor. They must run after the input/command/ambient pause acknowledgments.
    """
    from src.ambient.ambient_stt import AmbientSTT
    from src.ambient.speaker_tagger import SpeakerTagger
    from src.ambient.transcriber import AmbientTranscriber
    from src.application import _ModelResourceOwner
    from src.analytics.event_logger import EventLogger
    from src.analytics.suggestion_engine import SuggestionEngine
    from src.memory.long_term import LongTermMemory
    from src.memory.memory_manager import MemoryManager
    from src.nlu.llm_router import LLMCommandRouter
    from src.llm.hermes_reasoner import _HermesLifecycle
    from src.llm.reasoner import FastRouter, HttpReasoner
    from src.orchestrator.context_manager import ContextManager
    from src.pipeline.audio_loop import AudioLoop
    from src.pipeline.command_processor import CommandProcessor
    from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
    from src.pipeline.request_router import RequestRouter
    from src.pipeline.response_handler import ResponseHandler
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT
    from src.tts.piper_tts import DualTTS, KokoroTTS
    from src.users.speaker_identifier import SpeakerIdentifier
    from src.users.voice_registry import VoiceRegistry
    from src.vectordb.chroma_sync import ChromaSync
    from src.wakeword.detector import WakeWordDetector

    guards, borrowers, models = [], {}, {}
    model_receipts = set()
    native_drains, model_drains = [], []
    supported = True
    snapshots = (AudioLoop, MultiRoomAudioLoop, CommandProcessor,
                 AmbientTranscriber, AmbientSTT, SpeakerTagger)

    def unavailable():
        raise RuntimeError(_UNAVAILABLE)

    def add_borrower(borrower):
        nonlocal supported
        if borrower is None or id(borrower) in borrowers:
            return
        borrowers[id(borrower)] = borrower
        if isinstance(borrower, OwnedBlockingWork):
            native_drains.append(borrower.drain_admitted)
        elif isinstance(borrower, snapshots):
            async def drain_snapshot(value=borrower):
                await _snapshot_finished(value.snapshot_operations)
            native_drains.append(drain_snapshot)
        elif isinstance(borrower, VoiceRegistry):
            async def drain_registry(value=borrower):
                while True:
                    await value._work.drain_admitted()
                    await value._disk.drain_admitted()
                    await _tasks_finished(tuple(value._tasks))
                    # Reserved posts can precede their event-loop Task.
                    if not value._jobs and not value._tasks and len(value._free) == 6:
                        return
                    await asyncio.sleep(.005)
            native_drains.append(drain_registry)
        elif isinstance(borrower, ResponseHandler):
            def response_guard(value=borrower):
                if value._after_context.get():
                    raise RuntimeError("voice_close_from_callback")
            guards.append(response_guard)

            async def drain_response(value=borrower):
                await value.cancel()
                await value._native_work.drain_admitted()
                keys = ("pending", "controllers", "workers", "generations",
                        "blocked_generations", "aborts", "native_calls",
                        "native_operations", "native_retirements", "callbacks", "hook_tasks")
                while True:
                    status = value.get_delivery_status()
                    if not any(status[key] for key in keys):
                        if status["failed_barriers"]:
                            raise RuntimeError(_UNAVAILABLE)
                        return
                    await asyncio.sleep(.005)
            native_drains.append(drain_response)
        elif isinstance(borrower, ContextManager):
            async def drain_context(value=borrower):
                while True:
                    tasks = tuple(set(value._compaction_tasks) | set(value._active_compactions))
                    await _tasks_finished(tasks)
                    await value._io_work.drain_admitted()
                    if not any(not task.done() for task in value._compaction_tasks):
                        with value._lock:
                            if not any(not task.done() for task in value._active_compactions):
                                return
            native_drains.append(drain_context)
        elif isinstance(borrower, FastRouter):
            guards.append(borrower._router_require_close_caller)

            async def drain_fast_router(value=borrower):
                while True:
                    closing = value._adapter_closing
                    if closing is not None:
                        # Observe an existing terminal owner; mute never starts it.
                        await asyncio.shield(closing)
                        return
                    with value._adapter_condition:
                        executor = value._router_executor
                        if (value._router_dispatch_uncertain or value._router_work_failure
                                or executor is not None and executor._broken):
                            raise RuntimeError(_UNAVAILABLE)
                        pending = (bool(value._router_work)
                                   or any(lease.active for lease in value._adapter_leases)
                                   or any(not future.done() for future in value._router_native_futures))
                    if not pending:
                        return
                    await asyncio.sleep(.005)
            native_drains.append(drain_fast_router)
        elif isinstance(borrower, LLMCommandRouter):
            add_borrower(borrower._work)
        elif isinstance(borrower, EventLogger):
            async def drain_events(value=borrower):
                await _snapshot_finished(value.snapshot_operations)
            native_drains.append(drain_events)
        elif isinstance(borrower, MemoryManager):
            add_borrower(borrower.long_term)
        elif isinstance(borrower, LongTermMemory):
            async def drain_memory(value=borrower):
                def snapshot():
                    with value._resources_lock:
                        return tuple(value._operations)
                await _snapshot_finished(snapshot)
            native_drains.append(drain_memory)
        elif isinstance(borrower, ChromaSync):
            async def drain_chroma(value=borrower):
                await value._async_work.drain_admitted()
                def idle():
                    with value._condition:
                        return value._active == 0
                await _idle(idle)
            native_drains.append(drain_chroma)
        else:
            supported = False

    for owner in model_owners:
        if not isinstance(owner, _ModelResourceOwner):
            supported = False
            continue
        if owner._capture_unreviewed or owner._closing is not None:
            supported = False
        for model in tuple(owner._models.values()):
            models[id(model)] = model
            if owner._borrowers:
                model_receipts.add(id(model))
        for begin in tuple(owner._borrowers):
            borrower = getattr(begin, "__self__", None)
            if borrower is None:
                supported = False
            else:
                add_borrower(borrower)

    if router is not None:
        if isinstance(router, RequestRouter) and isinstance(router._query_work, OwnedBlockingWork):
            add_borrower(router._query_work)
            add_borrower(router.router)
            add_borrower(router.llm_command_router)
            add_borrower(router.event_logger)
            add_borrower(router._asr_event_logger)
            add_borrower(router.memory)
            if isinstance(router.suggestion_engine, SuggestionEngine):
                add_borrower(router.suggestion_engine.event_logger)
            elif router.suggestion_engine is not None:
                supported = False
        else:
            supported = False
    add_borrower(responses)
    add_borrower(context_manager)

    for model in models.values():
        if isinstance(model, _HermesLifecycle):
            guards.append(model._hermes_require_close_caller)
            model_drains.append(model._hermes_finish)
        elif isinstance(model, FastRouter):
            add_borrower(model)
        elif isinstance(model, HttpReasoner):
            lifecycle = model._http_lifecycle
            guards.append(model._http_require_close_caller)

            async def drain_http(value=lifecycle):
                def idle():
                    with value._adapter_condition:
                        if value._dispatch_uncertain or value._resource_failures or value._finish_failures:
                            raise RuntimeError(_UNAVAILABLE)
                        return (not any(lease.active for lease in value._adapter_leases)
                                and not value._initializing and not value._close_work
                                and not value._streams and not value._async_streams)
                await _idle(idle)
            model_drains.append(drain_http)
        elif isinstance(model, FastWhisperSTT):
            def whisper_guard(value=model):
                caller = threading.current_thread()
                if caller is value._worker_thread or caller is value._close_owner:
                    raise RuntimeError("whisper_close_from_worker")
            guards.append(whisper_guard)

            async def drain_whisper(value=model):
                def idle():
                    with value._lifecycle:
                        executor = value._executor
                        if executor is not None and executor._broken:
                            raise RuntimeError(_UNAVAILABLE)
                        # This admission is released by Future's done callback,
                        # after the pinned invocation actually returned.
                        return value._admitted == 0 and not value._creating_executor
                await _idle(idle)
            model_drains.append(drain_whisper)
        elif isinstance(model, (DualTTS, KokoroTTS)):
            if id(model) not in model_receipts:
                supported = False
            guards.append(model._adapter_require_close_caller)

            async def drain_tts(value=model):
                def idle():
                    with value._adapter_condition:
                        active = any(lease.active for lease in value._adapter_leases)
                        if not active and value._adapter_retained_iterators:
                            raise RuntimeError(_UNAVAILABLE)
                        return not active
                await _idle(idle)
            model_drains.append(drain_tts)
        elif isinstance(model, (ParakeetSTT, SpeakerIdentifier, WakeWordDetector)):
            if id(model) not in model_receipts:
                supported = False
            if isinstance(model, ParakeetSTT):
                prefix = "_parakeet"
            elif isinstance(model, SpeakerIdentifier):
                prefix = "_speaker"
            else:
                prefix = "_wake"
            guards.append(getattr(model, prefix + "_require_close_caller"))

            async def drain_leases(value=model, name=prefix):
                condition = getattr(value, name + "_condition")
                def idle():
                    with condition:
                        return not any(lease.active for lease in getattr(value, name + "_leases"))
                await _idle(idle)
            model_drains.append(drain_leases)
        else:
            supported = False

    async def finish():
        # Keep all known borrowers progressing even if another drain fails.
        failures = await asyncio.gather(*(drain() for drain in native_drains), return_exceptions=True)
        model_failures = await asyncio.gather(*(drain() for drain in model_drains), return_exceptions=True)
        if not supported or any(isinstance(item, BaseException) for item in (*failures, *model_failures)):
            raise RuntimeError(_UNAVAILABLE)

    def drain():
        for guard in guards:
            guard()
        return finish()

    if not supported and not native_drains and not model_drains:
        return (unavailable,), False
    return (drain,), supported
