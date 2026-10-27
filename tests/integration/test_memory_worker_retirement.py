"""Cancelled memory observers must not retain eventual native values."""

import asyncio
import threading
from uuid import uuid4
from weakref import ref

import pytest

from src.memory.producer_scopes import ProducerScopes
from src.memory.fact_ledger import MemoryOrigin, MemorySource
import src.memory.long_term as module
from src.memory.prompt_context import PromptContext
from tests.integration.test_memory_retention import graph, until
from tests.integration.test_prompt_provenance import REVIEWER, command_event, owner_graph, reference, request_router, stored
from tests.integration.test_transient_privacy import graph as manager_graph
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


class Payload:
    """Weak-referenceable synthetic private input/result."""


class Owner:
    def snapshot(self):
        return ()  # The actual memory worker is owned outside this controller.


@pytest.mark.parametrize('entry', ['acall', 'astore_fact', 'asearch_relevant', 'areconcile'])
async def test_cancelled_observer_releases_eventual_result_without_releasing_running_worker(tmp_path, monkeypatch, entry):
    memory, _, _ = graph(tmp_path)
    entered, release = threading.Event(), threading.Event()
    references, errors = [], []

    def native():
        entered.set()
        assert release.wait(3)
        result = Payload()
        references.append(ref(result))
        return result

    method = {'astore_fact': 'store_fact', 'asearch_relevant': 'search_relevant', 'areconcile': 'reconcile'}.get(entry)
    if method:
        monkeypatch.setattr(memory, method, native)
    observer = asyncio.create_task(memory.acall(native) if entry == 'acall' else getattr(memory, entry)())
    try:
        await until(entered.is_set)
        observer.cancel()
        try:
            await observer
        except asyncio.CancelledError as error:
            errors.append(error)  # Deliberately keep the cancelled observer's traceback.
        assert errors and memory.resource_status()['operations'] == 1
        closing = memory.begin_close()
        assert not closing.done()
        release.set()
        await closing
        await until(lambda: memory.resource_status()['operations'] == 0)
        assert references and references[0]() is None
        assert errors[0].__traceback__ is not None
    finally:
        release.set()
        await memory.aclose()


@pytest.mark.parametrize('entry', ['acall', 'astore_fact', 'asearch_relevant'])
async def test_completed_transport_and_cancelled_observer_do_not_retain_inputs(tmp_path, monkeypatch, entry):
    memory, _, _ = graph(tmp_path)
    entered, release = threading.Event(), threading.Event()
    transports, errors = [], []
    to_thread = asyncio.to_thread

    async def transport(function, *args, **kwargs):
        result = await to_thread(function, *args, **kwargs)
        if function is module._invoke_memory:
            transports.append((function, args, kwargs))
        return result

    monkeypatch.setattr(asyncio, 'to_thread', transport)

    def native(*args, **kwargs):
        entered.set()
        assert release.wait(3)

    method = {'astore_fact': 'store_fact', 'asearch_relevant': 'search_relevant'}.get(entry)
    if method:
        monkeypatch.setattr(memory, method, native)
    payload = Payload()
    raw = ref(payload)
    observer = asyncio.create_task(memory.acall(native, payload, private=payload) if entry == 'acall'
                                   else getattr(memory, entry)(payload, private=payload))
    del payload
    try:
        await until(entered.is_set)
        observer.cancel()
        try:
            await observer
        except asyncio.CancelledError as error:
            errors.append(error)
        assert raw() is not None and memory.resource_status()['operations'] == 1
        release.set()
        await memory.begin_close()
        assert errors and transports and raw() is None
    finally:
        release.set()
        await memory.aclose()


@pytest.mark.parametrize('entry', ['acall', 'astore_fact', 'asearch_relevant', 'acall_prompt'])
@pytest.mark.parametrize('rejection', ['closed', 'capacity'])
async def test_rejected_admission_tracebacks_release_inputs(tmp_path, entry, rejection):
    memory, _, _ = graph(tmp_path)
    release = threading.Event()
    blockers = []
    if rejection == 'closed':
        await memory.aclose()
    else:
        blockers = [asyncio.create_task(memory.acall(release.wait, 3)) for _ in range(8)]
        await until(lambda: memory.resource_status()['operations'] == 8)
    payload, errors = Payload(), []
    raw = ref(payload)
    try:
        try:
            if entry in {'acall', 'acall_prompt'}:
                await getattr(memory, entry)(lambda *args, **kwargs: None, payload, private=payload)
            else:
                await getattr(memory, entry)(payload, private=payload)
        except RuntimeError as error:
            errors.append(error)
        del payload
        assert errors and raw() is None
    finally:
        release.set()
        await asyncio.gather(*blockers)
        await memory.aclose()


@pytest.mark.parametrize('entry', ['aprompt_context', 'arecent_prompt_context', 'aformat_context_for_prompt'])
async def test_selected_sources_are_published_before_result_delivery(tmp_path, monkeypatch, entry):
    async with manager_graph(tmp_path) as (manager, memory, clock, _):
        origin, fact = await stored(memory, clock)
        if entry != 'arecent_prompt_context':
            clock.now = fact.expires_at - 60
        capture = memory_source(now=clock.now)
        owner = Owner()
        scopes = ProducerScopes(owner.snapshot)
        entered, release = threading.Event(), threading.Event()
        held, finish = asyncio.Event(), asyncio.Event()
        observations, selected_scopes = [], []
        original = manager.recent_prompt_context if entry == 'arecent_prompt_context' else manager.prompt_context
        if entry == 'arecent_prompt_context':
            manager.short_term.add_turn('Texto anterior.', fact.content, sources=(origin,))

        def select(*args):
            result = original(*args)
            entered.set()
            assert release.wait(3)
            return result

        monkeypatch.setattr(manager, 'recent_prompt_context' if entry == 'arecent_prompt_context' else 'prompt_context', select)

        async def consume():
            with scopes.operation((capture,)) as scope:
                selected_scopes.append(scope)
                result = await getattr(manager, entry)(*([] if entry == 'arecent_prompt_context' else ['dato privado']))
                held.set()
                await finish.wait()
                return bool(result)

        observer = asyncio.create_task(consume())
        try:
            await until(entered.is_set)
            assert scopes.busy(str(uuid4()), memory.ledger)  # Selection still native-owned.
            native = next(iter(memory._operations))
            native.add_done_callback(lambda _: observations.append(selected_scopes[0].snapshot()))
            release.set()
            await asyncio.wait_for(held.wait(), 3)
            assert observations and observations[0][1] == 0
            sources = observations[0][0]
            assert not scopes.busy(str(uuid4()), memory.ledger)
            assert scopes.busy(capture.conversation_id, memory.ledger)
            if entry == 'arecent_prompt_context':
                assert origin in sources and scopes.busy(origin.conversation_id, memory.ledger)
            else:
                assert reference(fact) in sources[0].fact_dependencies
                assert sources[0].expires_at == fact.expires_at
                await memory.acall(memory.ledger.delete_conversation, origin.conversation_id, REVIEWER)
                assert scopes.busy(origin.conversation_id, memory.ledger)
        finally:
            release.set()
            finish.set()
            await observer
        assert scopes.resource_status()['scopes'] == 0


async def test_cancelled_unresolved_prompt_read_keeps_scope_capacity_until_actual_exit(tmp_path, monkeypatch):
    async with manager_graph(tmp_path) as (manager, memory, clock, _):
        capture = memory_source(now=clock.now)
        owner = Owner()
        scopes = ProducerScopes(owner.snapshot, capacity=1)
        entered, release = threading.Event(), threading.Event()
        references, errors = [], []

        def select(query):
            entered.set()
            assert release.wait(3)
            result = PromptContext('Texto privado sintético.')
            references.append(ref(result))
            return result

        monkeypatch.setattr(manager, 'prompt_context', select)

        async def consume():
            with scopes.operation((capture,)):
                await manager.aprompt_context('Consulta privada sintética.')

        observer = asyncio.create_task(consume())
        try:
            await until(entered.is_set)
            observer.cancel()
            try:
                await observer
            except asyncio.CancelledError as error:
                errors.append(error)
            assert scopes.resource_status() == {'scopes': 1, 'capacity': 1, 'unknown': 1}
            assert scopes.busy(str(uuid4()), memory.ledger)
            with pytest.raises(RuntimeError, match='producer_scope_capacity'):
                with scopes.operation((capture,)):
                    raise AssertionError('A cancelled observer released the actual read')
        finally:
            release.set()
            await until(lambda: memory.resource_status()['operations'] == 0)
        assert errors and references and references[0]() is None
        assert scopes.resource_status()['scopes'] == 0


@pytest.mark.parametrize('failure', ['raise', 'invalid-result'])
async def test_native_prompt_failures_cannot_export_raw_arguments_or_tracebacks(tmp_path, monkeypatch, failure):
    async with manager_graph(tmp_path) as (manager, _, _, _):
        references, errors, transports = [], [], []
        to_thread = asyncio.to_thread

        async def transport(function, *args, **kwargs):
            result = await to_thread(function, *args, **kwargs)
            if function is module._invoke_memory:
                transports.append((function, args, kwargs))
            return result

        def select(query):
            raw = Payload()
            references.append(ref(raw))
            if failure == 'raise':
                raise ValueError(raw)
            return raw

        monkeypatch.setattr(asyncio, 'to_thread', transport)
        monkeypatch.setattr(manager, 'prompt_context', select)
        try:
            await manager.aprompt_context('Consulta privada.')
        except RuntimeError as error:
            errors.append(error)
        assert errors and str(errors[0]) == 'memory_prompt_unavailable'
        assert references and references[0]() is None
        assert transports


async def test_concurrent_native_selections_merge_dependencies_without_renewing_expiry(tmp_path):
    async with manager_graph(tmp_path) as (_, memory, clock, _):
        _, first = await stored(memory, clock, 'Primer hecho sintético.')
        _, second = await stored(memory, clock, 'Segundo hecho sintético.')
        capture = memory_source(now=clock.now)
        owner = Owner()
        scopes = ProducerScopes(owner.snapshot)
        barrier = threading.Barrier(2)

        def select(fact):
            barrier.wait(timeout=3)
            return PromptContext(fact.content, facts=(reference(fact),))

        with scopes.operation((capture,)) as scope:
            await asyncio.gather(memory.acall_prompt(select, first), memory.acall_prompt(select, second))
            sources, reads = scope.snapshot()
            assert reads == 0 and sources[0].expires_at == capture.expires_at
            assert set(sources[0].fact_dependencies) == {reference(first), reference(second)}
            assert not scopes.busy(str(uuid4()), memory.ledger)
        assert scopes.resource_status()['scopes'] == 0


async def test_actual_legacy_voice_route_enriches_its_capture_receiver_scope(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch, spoken=True) as (_, manager, memory, clock, _, handler, _, _):
        origin, fact = await stored(memory, clock)
        event = command_event(clock)
        capture = MemorySource(event.capture_id, event.conversation_id, event.captured_at,
                               event.source_expires_at, MemoryOrigin.VOICE)
        owner = Owner()
        scopes = ProducerScopes(owner.snapshot)
        router = request_router(manager, handler)
        try:
            with scopes.operation((capture,)) as scope:
                result = await router.process_command(event)
                assert result['success'] and result['response'] == 'Una respuesta completa.'
                assert reference(fact) in scope.sources[0].fact_dependencies
                assert not scopes.busy(str(uuid4()), memory.ledger)
                await memory.acall(memory.ledger.delete_conversation, origin.conversation_id, REVIEWER)
                assert scopes.busy(origin.conversation_id, memory.ledger)
        finally:
            await router.aclose()
        assert scopes.resource_status()['scopes'] == 0
