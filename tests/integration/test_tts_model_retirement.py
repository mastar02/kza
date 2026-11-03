"""Actual Kokoro/Dual lifecycles with synthetic SDK constructor and iterator.

No weights, capture, playback, SDK download or hardware are used. Event waits
are failure bounds, not performance or heartbeat measurements.
"""

import asyncio
import logging
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

import src.tts.piper_tts as tts_module
from src.tts.piper_tts import DualTTS, KokoroTTS


pytestmark = pytest.mark.asyncio
CANARY = "SYNTHETIC_PRIVATE_TTS_CANARY_6A0E2F"


class BadAudio:
    def cpu(self):
        raise ValueError("synthetic audio conversion failure")


class Pipeline:
    def __init__(self, probe):
        self.probe = probe

    def __call__(self, text, *, voice):
        return self.probe.generate(self, text, voice)


class PipelineProbe:
    def __init__(self, monkeypatch):
        self.ctor_entered = threading.Event()
        self.ctor_release = threading.Event()
        self.infer_entered = threading.Event()
        self.infer_release = threading.Event()
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        self.release_all()
        self.models = []
        self.calls = 0
        self.iterator_exits = 0
        self.bad_audio = False
        self.fail_close = False
        sdk = ModuleType("kokoro")
        sdk.KPipeline = self.construct
        monkeypatch.setitem(sys.modules, "kokoro", sdk)

    def construct(self, **options):
        assert options == {"lang_code": "e", "device": "cpu"}
        model = Pipeline(self)
        self.models.append(weakref.ref(model))
        self.ctor_entered.set()
        assert self.ctor_release.wait(5), "synthetic constructor not released"
        return model

    def generate(self, model, _text, _voice):
        assert model.probe is self
        self.calls += 1
        self.infer_entered.set()
        try:
            assert self.infer_release.wait(5), "synthetic inference not released"
            audio = BadAudio() if self.bad_audio else np.arange(8192, dtype=np.float32)
            yield SimpleNamespace(audio=audio)
        finally:
            self.close_entered.set()
            assert self.close_release.wait(5), "synthetic iterator close not released"
            self.iterator_exits += 1
            if self.fail_close:
                raise RuntimeError("synthetic iterator close failure")

    def release_all(self):
        self.ctor_release.set()
        self.infer_release.set()
        self.close_release.set()


def adapter(kind):
    if kind == "dual":
        return DualTTS(kokoro_config={"device": "cpu", "lang_code": "e"})
    return KokoroTTS(device="cpu", lang_code="e")


def child(tts):
    return tts.kokoro if isinstance(tts, DualTTS) else tts


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(probe, adapters, *, releases=(), tasks=(), streams=()):
    # Always release native leaves before awaiting callers or closing generators.
    probe.release_all()
    for release in releases:
        release.set()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    for stream in streams:
        if stream is not None:
            await asyncio.to_thread(stream.close)
    await asyncio.gather(*(tts.aclose() for tts in adapters), return_exceptions=True)


async def test_concurrent_load_is_once_and_close_retains_constructor(monkeypatch):
    probe = PipelineProbe(monkeypatch)
    probe.ctor_release.clear()
    tts = adapter("kokoro")
    first = second = None
    try:
        first = asyncio.create_task(asyncio.to_thread(tts.load))
        await until(probe.ctor_entered.is_set)
        second = asyncio.create_task(asyncio.to_thread(tts.load))
        await until(lambda: tts._tts_active == 2)
        assert len(probe.models) == 1
        assert probe.models[0]() is not None
        assert tts._model is None

        closing = tts.begin_close()
        assert tts._tts_sealed
        assert not closing.done()
        assert not first.done() and not second.done()
        probe.ctor_release.set()
        assert await asyncio.gather(first, second) == [None, None]
        await asyncio.shield(closing)
        assert len(probe.models) == 1
        assert tts._model is None
        assert probe.models[0]() is None
        with pytest.raises(RuntimeError, match="^tts_closed$"):
            await asyncio.to_thread(tts.load)
        assert len(probe.models) == 1
    finally:
        await finish(probe, [tts], tasks=[first, second])


@pytest.mark.parametrize("entry", ["synthesize", "stream"])
async def test_dual_admitted_delegate_can_enter_child_after_parent_seal(monkeypatch, entry):
    probe = PipelineProbe(monkeypatch)
    tts = adapter("dual")
    owned_child = child(tts)
    entered = threading.Event()
    release = threading.Event()
    method = "synthesize_stream" if entry == "stream" else "synthesize"
    actual = getattr(owned_child, method)

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5), "synthetic delegate not released"
        return actual(*args, **kwargs)

    monkeypatch.setattr(owned_child, method, delayed)
    stream = tts.synthesize_stream(CANARY) if entry == "stream" else None
    caller = None
    try:
        operation = (lambda: next(stream)) if stream is not None else (lambda: tts.synthesize(CANARY))
        caller = asyncio.create_task(asyncio.to_thread(operation))
        await until(entered.is_set)
        assert tts._tts_active == 1
        assert owned_child._tts_active == 0
        closing = tts.begin_close()
        assert not closing.done()
        assert not owned_child._tts_sealed
        release.set()
        result = await caller
        assert len(probe.models) == 1
        assert probe.calls == 1
        if stream is not None:
            assert result.shape == (4096,)
            assert not closing.done()
            assert owned_child._tts_active == 1
            await asyncio.to_thread(stream.close)
        else:
            assert len(result) == 3 and result[2] == "kokoro"
            assert result[0].shape == (8192,)
        await asyncio.shield(closing)
        assert owned_child._model is None
        assert probe.models[0]() is None
        assert owned_child._tts_sealed
    finally:
        await finish(probe, [tts], releases=[release], tasks=[caller], streams=[stream])


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
async def test_stream_closed_before_first_next_has_no_admission_or_inference(monkeypatch, kind):
    probe = PipelineProbe(monkeypatch)
    tts = adapter(kind)
    owned_child = child(tts)
    unopened = tts.synthesize_stream(CANARY)
    after_seal = tts.synthesize_stream(CANARY)
    try:
        unopened.close()
        assert tts._tts_active == 0
        assert owned_child._tts_active == 0
        assert probe.models == [] and probe.calls == 0
        await tts.aclose()
        with pytest.raises(RuntimeError, match="^tts_closed$"):
            next(after_seal)
        assert tts._tts_active == 0
        assert owned_child._tts_active == 0
        assert probe.models == [] and probe.calls == 0
    finally:
        await finish(probe, [tts], streams=[unopened, after_seal])


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
async def test_suspended_stream_retains_model_until_close_without_logging_text(monkeypatch, caplog, kind):
    probe = PipelineProbe(monkeypatch)
    tts = adapter(kind)
    owned_child = child(tts)
    stream = None
    caplog.set_level(logging.DEBUG, logger=tts_module.__name__)
    try:
        await asyncio.to_thread(tts.load)
        await asyncio.to_thread(tts.load)
        assert len(probe.models) == 1
        stream = tts.synthesize_stream(CANARY)
        chunk = await asyncio.to_thread(next, stream)
        assert chunk.shape == (4096,)
        assert probe.iterator_exits == 1
        assert owned_child._tts_active == 1
        assert probe.models[0]() is not None
        closing = tts.begin_close()
        assert not closing.done()
        with pytest.raises(RuntimeError, match="^tts_closed$"):
            await asyncio.to_thread(tts.synthesize, CANARY)
        assert probe.calls == 1
        await asyncio.to_thread(stream.close)
        await asyncio.shield(closing)
        assert owned_child._model is None
        assert probe.models[0]() is None
        assert any('Kokoro TTS completed in ' in record.getMessage() for record in caplog.records)
        assert not any(CANARY in record.getMessage() for record in caplog.records)
    finally:
        await finish(probe, [tts], streams=[stream])


async def test_conversion_error_waits_for_actual_sdk_iterator_finally(monkeypatch):
    probe = PipelineProbe(monkeypatch)
    probe.bad_audio = True
    probe.close_release.clear()
    tts = adapter("kokoro")
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(tts.synthesize, CANARY))
        await until(probe.close_entered.is_set)
        assert tts._tts_active == 1
        assert probe.iterator_exits == 0
        assert not caller.done()
        assert probe.models[0]() is not None
        closing = tts.begin_close()
        assert not closing.done()
        probe.close_release.set()
        result, = await asyncio.gather(caller, return_exceptions=True)
        assert isinstance(result, ValueError)
        assert str(result) == "synthetic audio conversion failure"
        assert probe.iterator_exits == 1
        # This test intentionally holds the error, but it must not retain a model.
        result = None
        caller = None
        await asyncio.shield(closing)
        assert tts._model is None
        assert probe.models[0]() is None
    finally:
        await finish(probe, [tts], tasks=[caller])


async def test_failed_sdk_iterator_close_seals_and_reports_retained_failure(monkeypatch):
    probe = PipelineProbe(monkeypatch)
    probe.bad_audio = True
    probe.fail_close = True
    tts = adapter("kokoro")
    try:
        with pytest.raises(RuntimeError, match="^synthetic iterator close failure$"):
            await asyncio.to_thread(tts.synthesize, CANARY)
        assert probe.iterator_exits == 1
        assert tts._tts_sealed
        assert len(tts._tts_retained_iterators) == 1
        assert tts._model is not None
        with pytest.raises(RuntimeError, match="^tts_closed$"):
            await asyncio.to_thread(tts.synthesize, CANARY)
        with pytest.raises(RuntimeError, match="^tts_iterator_close_failed$"):
            await tts.aclose()
        assert tts._model is not None
        assert probe.models[0]() is not None
        assert len(tts._tts_retained_iterators) == 1
        assert probe.calls == 1
    finally:
        # The deliberately poisoned dummy model remains retained, not falsely freed.
        await finish(probe, [tts])


async def test_dual_close_does_not_close_borrowed_replacement_or_peer(monkeypatch):
    probe = PipelineProbe(monkeypatch)
    tts = adapter("dual")
    originally_owned = child(tts)
    borrowed = adapter("kokoro")
    peer = adapter("kokoro")
    try:
        await asyncio.to_thread(borrowed.load)
        await asyncio.to_thread(peer.load)
        assert len(probe.models) == 2
        tts.kokoro = borrowed
        audio, _, engine = await asyncio.to_thread(tts.synthesize, CANARY)
        assert engine == "kokoro" and audio.shape == (8192,)
        await tts.aclose()
        assert originally_owned._tts_sealed
        assert originally_owned._model is None
        assert tts.kokoro is None
        assert not borrowed._tts_sealed and not peer._tts_sealed
        assert all(model_ref() is not None for model_ref in probe.models)
        for survivor in (borrowed, peer):
            surviving_audio, _ = await asyncio.to_thread(survivor.synthesize, CANARY)
            assert surviving_audio.shape == (8192,)
        assert len(probe.models) == 2
    finally:
        await finish(probe, [tts, borrowed, peer])


async def test_optional_backend_without_retirement_contract_is_explicit_failure(monkeypatch):
    probe = PipelineProbe(monkeypatch)

    class UnsupportedBackend:
        def __init__(self, **_options):
            pass

    monkeypatch.setattr(tts_module, "Qwen3TTS", UnsupportedBackend)
    tts = DualTTS(
        kokoro_config={"device": "cpu", "lang_code": "e"},
        qwen3_config={"synthetic": True},
    )
    unsupported = tts.qwen3
    try:
        with pytest.raises(RuntimeError, match="^tts_backend_close_unsupported$"):
            await tts.aclose()
        assert tts.qwen3 is unsupported
        assert unsupported in tts._owned_tts_engines
        assert tts.kokoro._tts_sealed
        assert probe.models == []
    finally:
        await finish(probe, [tts])
