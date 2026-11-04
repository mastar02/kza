"""Independent control-path seals compose without releasing another owner."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import threading
from uuid import uuid4

import pytest

from src.ambient.control import build_ambient_control
from src.memory.fact_ledger import MemoryOrigin
from src.users.voice_registry import VoiceRegistry
from tests.integration.test_ambient_pause import ACTOR, phase
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_input_callback_retirement import HeldInput, graph as callback_graph, invoke
from tests.integration.test_shared_input_retirement import ambient as attached_ambient
from tests.mocks.ambient_graph import ambient_graph
from tests.mocks.voice_registry import AUDIO, Encoder, idle


pytestmark = pytest.mark.asyncio
ROOM = "synthetic-a"


def _capture(graph, origin):
    graph.clock.now += .2
    return graph.privacy.source_for_capture(str(uuid4()), graph.clock.now, ROOM, origin=origin)


def _registry(stack, privacy):
    registry = VoiceRegistry(privacy)
    stack.push_async_callback(registry.aclose)
    return registry


class _Counted(HeldInput):
    def __array_finalize__(self, parent):
        super().__array_finalize__(parent)
        self.copies = getattr(parent, "copies", None)

    def copy(self, *args, **kwargs):
        self.copies.append(True)
        return super().copy(*args, **kwargs)


class _InvalidReason:
    def __hash__(self):
        raise AssertionError("invalid reason was hashed")

    def __eq__(self, other):
        raise AssertionError("invalid reason was compared")

    def __repr__(self):
        raise AssertionError("invalid reason was rendered")


async def _push(graph):
    audio = HeldInput("unused").view(_Counted)
    audio.copies = []
    await asyncio.to_thread(invoke, graph, "multi", audio)
    return audio.copies


async def test_explicit_reason_survives_real_ambient_pause_resume_and_blocks_its_copy(tmp_path, monkeypatch):
    async with callback_graph(tmp_path, monkeypatch, "multi") as (g, command):
        async with attached_ambient(g, tmp_path) as path, AsyncExitStack() as stack:
            registry = _registry(stack, g.privacy)
            encoder = Encoder()
            registry.register_encoder(encoder)
            control = await build_ambient_control(
                privacy=g.privacy, path=path, voices=registry, shutdown=stack,
            )
            previous = _capture(g, MemoryOrigin.AMBIENT)
            expires_at = previous.expires_at
            old = path.tap.input_epoch(ROOM)
            scopes = g.privacy.scopes
            reason = object()
            scopes.suspend_origin(MemoryOrigin.AMBIENT, reason=reason)
            registry.suspend_origin(MemoryOrigin.AMBIENT, reason=reason)
            scopes.suspend_origin(MemoryOrigin.AMBIENT, reason=reason)
            assert old.retired

            for component in (scopes, registry):
                for method in (component.suspend_origin, component.resume_origin):
                    with pytest.raises(ValueError) as error:
                        method(MemoryOrigin.AMBIENT, reason=_InvalidReason())
                    assert error.value.args == ("invalid_capture_suspension_reason",)
                    with pytest.raises(ValueError) as error:
                        method(False, reason=reason)
                    assert error.value.args == ("invalid_capture_origin",)

            control.request(True, control.snapshot()["revision"], ACTOR)
            await phase(control, "paused")
            control.request(False, control.snapshot()["revision"], ACTOR)
            await phase(control, "running")
            assert path.tap.input_epoch(ROOM) is old and old.retired
            with pytest.raises(PermissionError):
                _capture(g, MemoryOrigin.AMBIENT)
            assert g.privacy.sources_available((previous,))
            assert previous.expires_at == expires_at
            assert not registry.observe(encoder, AUDIO, (previous,))
            scopes.resume_origin(MemoryOrigin.AMBIENT, reason=object())
            registry.resume_origin(MemoryOrigin.AMBIENT, reason=object())
            assert path.tap.input_epoch(ROOM) is old
            assert len(await _push(g)) == 1  # Command copy, no ambient copy.
            assert path.tap.buffered_count() == 0
            assert command.resource_status()["queued"] == 1

            scopes.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
            fresh = path.tap.input_epoch(ROOM)
            assert fresh is not old and not fresh.retired
            source = _capture(g, MemoryOrigin.AMBIENT)
            assert source.conversation_id != previous.conversation_id
            assert not registry.observe(encoder, AUDIO, (source,))
            registry.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
            assert registry.observe(encoder, AUDIO, (source,))
            await idle(registry)
            g.clock.now += .2
            assert len(await _push(g)) == 2
            assert path.tap.buffered_count() == 1

            def forbidden_clock():
                raise AssertionError("unknown explicit release consulted clock")

            with monkeypatch.context() as patch:
                patch.setattr(g.privacy.ledger, "clock", forbidden_clock)
                scopes.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
                scopes.resume_origin(MemoryOrigin.AMBIENT, reason=object())
                registry.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
                registry.resume_origin(MemoryOrigin.AMBIENT, reason=object())
            assert path.tap.input_epoch(ROOM) is fresh and not fresh.retired


async def test_releasing_explicit_reason_preserves_real_ambient_pause_and_history(tmp_path):
    async with ambient_graph(tmp_path) as g, AsyncExitStack() as stack:
        registry = _registry(stack, g.privacy)
        encoder = Encoder()
        registry.register_encoder(encoder)
        control = await build_ambient_control(
            privacy=g.privacy, path=g.path, voices=registry, shutdown=stack,
        )
        previous = _capture(g, MemoryOrigin.AMBIENT)
        await g.transcriber._handle_segment(ROOM, g.segment())
        history = await g.store.undistilled_live(require_provenance=True)
        assert len(history) == 1
        old = g.tap.input_epoch(ROOM)
        voice_source = _capture(g, MemoryOrigin.VOICE)
        voice_old = g.privacy.scopes.ingress(ROOM, origin=MemoryOrigin.VOICE).current
        voice_reason, reason = object(), object()
        for component in (g.privacy.scopes, registry):
            component.suspend_origin(MemoryOrigin.VOICE, reason=voice_reason)

        control.request(True, control.snapshot()["revision"], ACTOR)
        await phase(control, "paused")
        for component in (g.privacy.scopes, registry):
            component.suspend_origin(MemoryOrigin.AMBIENT, reason=reason)
            component.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
            component.resume_origin(MemoryOrigin.AMBIENT, reason=reason)
            component.resume_origin(MemoryOrigin.AMBIENT, reason=object())
        assert control.snapshot()["phase"] == "paused"
        assert g.tap.input_epoch(ROOM) is old and old.retired
        with pytest.raises(PermissionError):
            _capture(g, MemoryOrigin.AMBIENT)
        assert not registry.observe(encoder, AUDIO, (previous,))
        assert await g.store.undistilled_live(require_provenance=True) == history

        control.request(False, control.snapshot()["revision"], ACTOR)
        await phase(control, "running")
        fresh = g.tap.input_epoch(ROOM)
        assert fresh is not old and not fresh.retired
        ambient_source = _capture(g, MemoryOrigin.AMBIENT)
        assert ambient_source.conversation_id != previous.conversation_id
        assert registry.observe(encoder, AUDIO, (ambient_source,))
        await idle(registry)
        with pytest.raises(PermissionError):
            _capture(g, MemoryOrigin.VOICE)
        assert voice_old.retired
        assert g.privacy.scopes.ingress(ROOM, origin=MemoryOrigin.VOICE).current is voice_old
        assert not registry.observe(encoder, AUDIO, (voice_source,))
        assert g.privacy.sources_available((previous, voice_source))

        g.privacy.scopes.resume_origin(MemoryOrigin.VOICE, reason=voice_reason)
        registry.resume_origin(MemoryOrigin.VOICE, reason=voice_reason)
        fresh_voice = _capture(g, MemoryOrigin.VOICE)
        assert fresh_voice.conversation_id != voice_source.conversation_id
        assert registry.observe(encoder, AUDIO, (fresh_voice,))
        await idle(registry)
        assert await g.store.undistilled_live(require_provenance=True) == history
        g.clock.now = history[0]["expires_at"]
        await g.store.purge_expired()
        assert await g.store.undistilled_live(require_provenance=True) == []


@pytest.mark.parametrize("origin", [MemoryOrigin.AMBIENT, MemoryOrigin.VOICE])
async def test_registry_last_reason_waits_for_real_native_exit_and_only_opens_its_lane(tmp_path, origin):
    async with ambient_graph(tmp_path) as g, AsyncExitStack() as stack:
        registry = _registry(stack, g.privacy)
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()

        class Held(Encoder):
            def get_embedding(self, audio):
                entered.set()
                assert release.wait(5), "actual encoder was not released"
                value = super().get_embedding(audio)
                returned.set()
                return value

        encoder, peer_encoder = Held(), Encoder()
        registry.register_encoder(encoder)
        registry.register_encoder(peer_encoder)
        other = MemoryOrigin.VOICE if origin is MemoryOrigin.AMBIENT else MemoryOrigin.AMBIENT
        source, peer_source = _capture(g, origin), _capture(g, other)
        first, last, peer_reason = object(), object(), object()
        assert registry.observe(encoder, AUDIO, (source,))
        try:
            await until(entered.is_set)
            registry.suspend_origin(origin, reason=first)
            registry.suspend_origin(origin, reason=last)
            assert registry.pending_origin(origin) == 1
            assert registry.observe(peer_encoder, AUDIO, (peer_source,))
            await until(lambda: peer_encoder.calls == 1 and registry.pending_origin(other) == 0)
            registry.suspend_origin(other, reason=peer_reason)
            registry.resume_origin(origin, reason=first)  # Partial: pending is allowed.
            registry.resume_origin(origin, reason=object())
            assert not registry.observe(encoder, AUDIO, (source,))
            with pytest.raises(RuntimeError, match="^voice_lane_still_owned$"):
                registry.resume_origin(origin, reason=last)
            assert not registry.observe(encoder, AUDIO, (source,))
            assert not registry.observe(peer_encoder, AUDIO, (peer_source,))
        finally:
            release.set()
            await idle(registry)

        assert returned.is_set() and registry.pending_origin(origin) == 0
        snapshot = await registry.acall(registry.ledger.snapshot)
        assert all(source.capture_id not in item["capture_ids"] for item in snapshot["fingerprints"])
        assert not registry.observe(encoder, AUDIO, (source,))  # Failed release kept seal.
        registry.resume_origin(origin, reason=last)
        assert registry.observe(encoder, AUDIO, (source,))
        assert not registry.observe(peer_encoder, AUDIO, (peer_source,))
        await idle(registry)
        snapshot = await registry.acall(registry.ledger.snapshot)
        assert any(source.capture_id in item["capture_ids"] for item in snapshot["fingerprints"])
        registry.resume_origin(other, reason=peer_reason)
        assert registry.observe(peer_encoder, AUDIO, (peer_source,))
        await idle(registry)
        registry.suspend_origin(origin, reason=last)
        await registry.aclose()
        with pytest.raises(RuntimeError, match="^voice_registry_closed$"):
            registry.resume_origin(origin, reason=last)
        assert not registry.observe(encoder, AUDIO, (source,))
