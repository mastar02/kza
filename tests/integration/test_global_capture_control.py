"""Global capture policy with real owners and synthetic native leaves."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import threading
from uuid import uuid4

import pytest

from src.ambient.control import build_ambient_control
from src.ambient.types import AmbientUtterance
from src.conversation.follow_up_mode import FollowUpMode
from src.dashboard.authority import DashboardPrincipal
from src.memory.capture_control import CaptureControl, CaptureController
from src.memory.fact_ledger import MemoryOrigin
from src.memory.producer_context import producer_context
from src.pipeline.command_processor import CommandProcessor
from src.users.voice_registry import VoiceRegistry
from tests.integration.test_ambient_pause import ACTOR as AMBIENT_ACTOR, phase
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_command_producer_privacy import Engine
from tests.integration.test_input_callback_retirement import HeldInput, graph as callback_graph, invoke
from tests.integration.test_input_driver_pause import Driver, portaudio
from tests.integration.test_shared_input_retirement import ambient as attached_ambient
from tests.mocks.ambient_graph import Gate
from tests.mocks.voice_registry import AUDIO, Encoder, idle

pytestmark = pytest.mark.asyncio
ROOM = 'synthetic-a'


def actor(current=lambda: True):
    principal = DashboardPrincipal('synthetic-operator', frozenset({'admin:manage'}))
    return CaptureController.from_dashboard(principal, current)


async def capture_phase(control, expected):
    await until(lambda: control.snapshot()['phase'] == expected)
    return control.snapshot()


@dataclass
class Controlled:
    graph: object
    room: object
    control: CaptureControl
    ambient: object
    path: object
    processor: CommandProcessor
    voices: VoiceRegistry
    gate: Gate
    drivers: list

    def source(self, origin=MemoryOrigin.VOICE, *, expires_at=None):
        self.graph.clock.now += .2
        return self.graph.privacy.source_for_capture(
            str(uuid4()), self.graph.clock.now, ROOM,
            origin=origin, expires_at=expires_at,
        )

    def command(self, source):
        with producer_context(ROOM, (source,)):
            return asyncio.create_task(self.processor.process_command(AUDIO.copy()))


@asynccontextmanager
async def controlled(tmp_path, monkeypatch, *, transport='multi'):
    drivers = [Driver() for _ in range(12)]
    portaudio(monkeypatch, drivers)
    async with callback_graph(tmp_path, monkeypatch, transport) as (g, room):
        # The older single-input fixture's minimal FollowUp has no pause hook.
        g.loop.follow_up = FollowUpMode()
        async with AsyncExitStack() as stack:
            path = ambient = None
            if transport == 'multi':
                path = await stack.enter_async_context(attached_ambient(g, tmp_path))
                path.store._clock = g.clock
            gate = Gate()
            processor = CommandProcessor(Engine(gate), privacy=g.privacy, close_timeout_s=.03)
            voices = VoiceRegistry(g.privacy)
            stack.push_async_callback(voices.aclose)
            stack.push_async_callback(processor.aclose)
            if path is not None:
                ambient = await build_ambient_control(
                    privacy=g.privacy, path=path, voices=voices, shutdown=stack,
                )
            control = CaptureControl(g.privacy, g.loop, processor,
                ambient_control=ambient, voices=voices)
            stack.push_async_callback(control.aclose)
            await control.initialize()
            state = control.snapshot()
            if state['target_muted']:
                assert state['ingress_sealed']
            expected = 'muted' if state['target_muted'] else 'running'
            await capture_phase(control, expected)
            g.clock.now += .2
            try:
                yield Controlled(g, room, control, ambient, path, processor, voices, gate, drivers)
            finally:
                gate.release.set()


@pytest.mark.parametrize('permissions', [frozenset(), frozenset({'ambient:control'}),
    frozenset({'evaluation:read'})])
async def test_global_capability_cannot_be_inferred_from_other_permissions(permissions):
    principal = DashboardPrincipal('synthetic-unprivileged', permissions)
    with pytest.raises(PermissionError):
        CaptureController.from_dashboard(principal, lambda: True)


@pytest.mark.parametrize('ambient_first', [False, True])
async def test_global_and_ambient_policies_release_only_their_own_seals(tmp_path, monkeypatch, ambient_first):
    async with controlled(tmp_path, monkeypatch) as g:
        old_voice = g.source()
        old_ambient = g.source(MemoryOrigin.AMBIENT)
        encoder = Encoder()
        g.voices.register_encoder(encoder)
        if ambient_first:
            g.ambient.request(True, g.ambient.snapshot()['revision'], AMBIENT_ACTOR)
            await phase(g.ambient, 'paused')
        g.control.request(True, g.control.snapshot()['revision'], actor())
        muted = await capture_phase(g.control, 'muted')
        assert muted['target_muted'] and muted['persisted_muted']
        assert muted['ingress_sealed'] and muted['total_mute_confirmed']
        assert muted['pending_operations'] == 0
        if not ambient_first:
            g.ambient.request(True, g.ambient.snapshot()['revision'], AMBIENT_ACTOR)
            await phase(g.ambient, 'paused')
            g.ambient.request(False, g.ambient.snapshot()['revision'], AMBIENT_ACTOR)
            await phase(g.ambient, 'running')
            for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
                with pytest.raises(PermissionError):
                    g.source(origin)
            assert not g.voices.observe(encoder, AUDIO, (old_ambient,))
            assert not g.voices.observe(encoder, AUDIO, (old_voice,))

        g.control.request(False, g.control.snapshot()['revision'], actor())
        running = await capture_phase(g.control, 'running')
        assert not running['target_muted'] and not running['persisted_muted']
        assert not running['total_mute_confirmed']
        fresh_voice = g.source()
        assert fresh_voice.conversation_id != old_voice.conversation_id
        assert g.voices.observe(encoder, AUDIO, (fresh_voice,))
        await idle(g.voices)
        if ambient_first:
            assert g.ambient.snapshot()['phase'] == 'paused'
            with pytest.raises(PermissionError):
                g.source(MemoryOrigin.AMBIENT)
            assert not g.voices.observe(encoder, AUDIO, (old_ambient,))
            g.ambient.request(False, g.ambient.snapshot()['revision'], AMBIENT_ACTOR)
            await phase(g.ambient, 'running')
        fresh_ambient = g.source(MemoryOrigin.AMBIENT)
        assert fresh_ambient.conversation_id != old_ambient.conversation_id
        assert g.graph.privacy.sources_available((old_voice, old_ambient))


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_global_ack_waits_for_native_command_after_observer_cancellation(tmp_path, monkeypatch, transport):
    async with controlled(tmp_path, monkeypatch, transport=transport) as g:
        source = g.source()
        g.gate.block()
        command = g.command(source)
        try:
            await until(g.gate.entered.is_set)
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
            assert any(not task.done() for task in g.processor.snapshot_operations())
            g.control.request(True, g.control.snapshot()['revision'], actor())
            await until(lambda: g.control.snapshot()['ingress_sealed'])
            pending = g.control.snapshot()
            assert pending['phase'] == 'muting' and not pending['total_mute_confirmed']
            assert pending['pending_operations'] > 0
            assert g.gate.calls == 1 and g.gate.threads
            with pytest.raises(PermissionError):
                g.source()
            assert g.graph.privacy.sources_available((source,))
        finally:
            g.gate.release.set()
            await asyncio.gather(command, return_exceptions=True)
        await capture_phase(g.control, 'muted')
        assert not any(not task.done() for task in g.processor.snapshot_operations())
        assert g.control.snapshot()['total_mute_confirmed']


@pytest.mark.parametrize('origin', [MemoryOrigin.AMBIENT, MemoryOrigin.VOICE])
async def test_global_ack_includes_real_voice_encoder_and_discards_late_observation(tmp_path, monkeypatch, origin):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    class HeldEncoder(Encoder):
        def get_embedding(self, audio):
            entered.set()
            assert release.wait(4)
            try:
                return super().get_embedding(audio)
            finally:
                returned.set()

    async with controlled(tmp_path, monkeypatch) as g:
        encoder = HeldEncoder()
        g.voices.register_encoder(encoder)
        source = g.source(origin)
        assert g.voices.observe(encoder, AUDIO, (source,))
        try:
            await until(entered.is_set)
            g.control.request(True, g.control.snapshot()['revision'], actor())
            await until(lambda: g.control.snapshot()['ingress_sealed'])
            assert g.voices.pending_origin(origin) == 1
            assert not returned.is_set()
            assert g.control.snapshot()['phase'] == 'muting'
            assert not g.control.snapshot()['total_mute_confirmed']
        finally:
            release.set()
            await idle(g.voices)
        await capture_phase(g.control, 'muted')
        assert returned.is_set() and g.voices.pending_origin(origin) == 0
        snapshot = await g.voices.acall(g.voices.ledger.snapshot)
        assert all(source.capture_id not in row['capture_ids'] for row in snapshot['fingerprints'])


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_global_ack_waits_for_actual_input_callback_exit(tmp_path, monkeypatch, transport):
    async with controlled(tmp_path, monkeypatch, transport=transport) as g:
        audio = HeldInput('copy' if transport == 'single' else 'any')
        failures = []
        returned = threading.Event()

        def push():
            try:
                invoke(g.graph, transport, audio)
            except BaseException as error:
                failures.append(error)
            finally:
                returned.set()

        native = threading.Thread(target=push)
        native.start()
        try:
            await until(audio.entered.is_set)
            assert g.room.callbacks.status()['active'] == 1
            g.control.request(True, g.control.snapshot()['revision'], actor())
            await until(lambda: g.control.snapshot()['ingress_sealed'])
            assert native.is_alive() and not returned.is_set()
            assert g.room.callbacks.status()['active'] == 1
            assert not g.control.snapshot()['total_mute_confirmed']
            assert g.control.snapshot()['pending_operations'] > 0
        finally:
            audio.release.set()
            await asyncio.to_thread(native.join)
        assert returned.is_set() and not native.is_alive() and not failures
        assert g.room.callbacks.status()['active'] == 0
        await capture_phase(g.control, 'muted')


async def test_durable_mute_restores_before_admission_without_renewing_history(tmp_path, monkeypatch):
    async with controlled(tmp_path, monkeypatch) as g:
        source = g.source(MemoryOrigin.AMBIENT, expires_at=g.graph.clock.now + 60)
        await g.path.store.add(AmbientUtterance(ROOM, source.captured_at, source.captured_at + .01,
            text='Synthetic retained history', capture_id=source.capture_id,
            conversation_id=source.conversation_id, expires_at=source.expires_at))
        history = await g.path.store.undistilled_live(require_provenance=True)
        assert len(history) == 1
        g.control.request(True, g.control.snapshot()['revision'], actor())
        await capture_phase(g.control, 'muted')
        revision = g.control.snapshot()['revision']
        assert await g.path.store.undistilled_live(require_provenance=True) == history

    async with controlled(tmp_path, monkeypatch) as rebuilt:
        restored = rebuilt.control.snapshot()
        assert restored['revision'] >= revision and restored['persisted_muted']
        assert restored['target_muted'] and restored['ingress_sealed']
        assert restored['phase'] == 'muted' and restored['total_mute_confirmed']
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            with pytest.raises(PermissionError):
                rebuilt.source(origin)
        rebuilt.graph.clock.now = max(rebuilt.graph.clock.now, source.captured_at + .2)
        assert await rebuilt.path.store.undistilled_live(require_provenance=True) == history
        assert rebuilt.graph.privacy.sources_available((source,))
        rebuilt.control.request(False, restored['revision'], actor())
        await capture_phase(rebuilt.control, 'running')
        fresh = rebuilt.source(MemoryOrigin.AMBIENT)
        assert fresh.conversation_id != source.conversation_id
        assert await rebuilt.path.store.undistilled_live(require_provenance=True) == history
        rebuilt.graph.clock.now = history[0]['expires_at']
        await rebuilt.path.store.purge_expired()
        assert await rebuilt.path.store.undistilled_live(require_provenance=True) == []


async def test_revoked_original_controller_cannot_commit_queued_resume(tmp_path, monkeypatch):
    async with controlled(tmp_path, monkeypatch) as g:
        g.control.request(True, g.control.snapshot()['revision'], actor())
        await capture_phase(g.control, 'muted')
        write = g.control._write_mode
        entered, release = threading.Event(), threading.Event()
        current = [True]
        original = actor(lambda: current[0])

        def held(mode, revision, controller):
            if mode == 'resuming':
                entered.set()
                assert release.wait(4)
            return write(mode, revision, controller)

        monkeypatch.setattr(g.control, '_write_mode', held)
        try:
            g.control.request(False, g.control.snapshot()['revision'], original)
            await until(entered.is_set)
            current[0] = False
            assert g.control.snapshot()['ingress_sealed']
            assert not g.control.snapshot()['total_mute_confirmed']
        finally:
            release.set()
        await capture_phase(g.control, 'failed')
        policy = await g.graph.privacy.acall(g.control._load_policy)
        assert policy[0] == 'muted'
        assert g.control.snapshot()['persisted_muted'] and g.control.snapshot()['ingress_sealed']
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            with pytest.raises(PermissionError):
                g.source(origin)


async def test_real_processor_drain_failure_never_confirms_or_reopens_capture(tmp_path, monkeypatch):
    async with controlled(tmp_path, monkeypatch) as g:
        prune = g.processor.prune_private
        calls = []

        def failed_prune():
            prune()
            calls.append(threading.get_ident())
            raise RuntimeError('synthetic_after_real_command_prune')

        monkeypatch.setattr(g.processor, 'prune_private', failed_prune)
        try:
            g.control.request(True, g.control.snapshot()['revision'], actor())
            await capture_phase(g.control, 'failed')
            state = g.control.snapshot()
            assert calls and state['ingress_sealed'] and state['persisted_muted']
            assert not state['total_mute_confirmed'] and state['error_code']
            for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
                with pytest.raises(PermissionError):
                    g.source(origin)
        finally:
            monkeypatch.setattr(g.processor, 'prune_private', prune)


@pytest.mark.parametrize('committed_mode', ['resuming', 'running'])
async def test_post_commit_failure_compensates_durable_mode_before_rebuild(tmp_path, monkeypatch, committed_mode):
    async with controlled(tmp_path, monkeypatch) as g:
        g.control.request(True, g.control.snapshot()['revision'], actor())
        await capture_phase(g.control, 'muted')
        write, force = g.control._write_mode, g.control._force_muted
        committed, compensated = threading.Event(), threading.Event()

        def interrupted(mode, revision, controller):
            result = write(mode, revision, controller)
            if mode == committed_mode:
                committed.set()
                raise RuntimeError('synthetic_after_durable_capture_commit')
            return result

        def compensate():
            result = force()
            compensated.set()
            return result

        monkeypatch.setattr(g.control, '_write_mode', interrupted)
        monkeypatch.setattr(g.control, '_force_muted', compensate)
        g.control.request(False, g.control.snapshot()['revision'], actor())
        await asyncio.shield(g.control._operation)
        state = g.control.snapshot()
        assert committed.is_set() and compensated.is_set()
        assert state['phase'] == 'failed' and state['ingress_sealed']
        assert state['persisted_muted'] and not state['total_mute_confirmed']
        assert (await g.graph.privacy.acall(g.control._load_policy))[0] == 'muted'
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            with pytest.raises(PermissionError):
                g.source(origin)
    async with controlled(tmp_path, monkeypatch) as rebuilt:
        state = rebuilt.control.snapshot()
        assert state['phase'] == 'muted' and state['persisted_muted']
        assert state['target_muted'] and state['ingress_sealed'] and state['total_mute_confirmed']
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            with pytest.raises(PermissionError):
                rebuilt.source(origin)


async def test_failed_compensation_restores_real_resuming_policy_as_muted(tmp_path, monkeypatch):
    async with controlled(tmp_path, monkeypatch) as g:
        g.control.request(True, g.control.snapshot()['revision'], actor())
        await capture_phase(g.control, 'muted')
        write = g.control._write_mode
        committed, compensation_attempted = threading.Event(), threading.Event()

        def interrupted(mode, revision, controller):
            result = write(mode, revision, controller)
            if mode == 'resuming':
                committed.set()
                raise RuntimeError('synthetic_after_durable_resume_intent')
            return result

        def unavailable():
            compensation_attempted.set()
            raise RuntimeError('synthetic_compensation_unavailable')

        monkeypatch.setattr(g.control, '_write_mode', interrupted)
        monkeypatch.setattr(g.control, '_force_muted', unavailable)
        g.control.request(False, g.control.snapshot()['revision'], actor())
        await asyncio.shield(g.control._operation)
        state = g.control.snapshot()
        assert committed.is_set() and compensation_attempted.is_set()
        assert state['phase'] == 'failed' and state['ingress_sealed']
        assert not state['total_mute_confirmed']
        assert (await g.graph.privacy.acall(g.control._load_policy))[0] == 'resuming'
    async with controlled(tmp_path, monkeypatch) as rebuilt:
        state = rebuilt.control.snapshot()
        assert state['phase'] == 'muted' and state['persisted_muted']
        assert state['target_muted'] and state['ingress_sealed'] and state['total_mute_confirmed']
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            with pytest.raises(PermissionError):
                rebuilt.source(origin)
