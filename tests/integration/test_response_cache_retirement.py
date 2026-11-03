"""Actual cache/work ownership, synthetic TTS/callback leaves; no SDK/model IO."""

import asyncio
from contextvars import Context, ContextVar
import threading
import weakref

import numpy as np
import pytest

import src.tts.response_cache as cache_module
from src.tts.response_cache import ResponseCache

pytestmark = pytest.mark.asyncio


class TTS:
    sample_rate = 16000

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.output = np.arange(8, dtype=np.float64)
        self.calls = []
        self.fail_for = set()
        self.on_call = None
        self.close_calls = 0

    def synthesize(self, text):
        self.calls.append(text)
        self.entered.set()
        assert self.release.wait(5), "synthetic synthesis not released"
        if self.on_call is not None:
            self.on_call()
        if text in self.fail_for:
            raise ValueError("synthetic phrase failure")
        return self.output, 1.0, "synthetic"

    def close(self):
        self.close_calls += 1


class HeldArray(np.ndarray):
    def __new__(cls, entered, release):
        value = np.arange(8, dtype=np.float64).view(cls)
        value.entered, value.release = entered, release
        return value

    def astype(self, dtype, *args, **kwargs):
        assert dtype is np.float32 and args == () and kwargs == {}
        self.entered.set()
        assert self.release.wait(5), "synthetic conversion not released"
        return np.array(self, dtype=np.float32, copy=True, subok=False)


def small_cache(monkeypatch, tts, *, phrases=("Listo", "Dale"), callback=None):
    monkeypatch.setattr(cache_module, "CANONICAL_PHRASES", list(phrases))
    monkeypatch.setattr(cache_module, "TEMPLATES", {})
    return ResponseCache(tts, synthesize_callback=callback)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(cache, *, tasks=(), threading_releases=(), async_releases=()):
    for release in threading_releases:
        release.set()
    for release in async_releases:
        release.set()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await cache.aclose()


@pytest.mark.parametrize("phase", ["synthesize", "conversion"])
async def test_close_waits_actual_native_step_and_discards_late_pcm(monkeypatch, phase):
    tts = TTS()
    converted, convert_release = threading.Event(), threading.Event()
    convert_release.set()
    if phase == "synthesize":
        tts.release.clear()
        entered = tts.entered
    else:
        convert_release.clear()
        entered = converted
        tts.output = HeldArray(converted, convert_release)
    cache = small_cache(monkeypatch, tts)
    building = None
    try:
        building = asyncio.create_task(cache.build())
        await until(entered.is_set)
        owner = cache.begin_close()
        assert owner is cache.begin_close() and not owner.done()
        assert cache._tts is tts and cache.size() == 0
        assert cache.get("Listo") is None
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await progressed.wait()  # Loop scheduling, not numeric C03 evidence.
        await asyncio.sleep(0)
        assert not owner.done() and cache._tts is tts
        tts.release.set()
        convert_release.set()
        await building
        await asyncio.shield(owner)
        assert cache.size() == 0 and cache._tts is cache._synthesize_callback is None
        assert tts.calls == ["Listo"] and tts.close_calls == 0
        with pytest.raises(RuntimeError, match="^response_cache_closed$"):
            await cache.build()
    finally:
        await finish(cache, tasks=[building], threading_releases=[tts.release, convert_release])


async def test_coalesced_build_and_repeated_close_observer_cancel_keep_native_owned(monkeypatch):
    tts = TTS()
    tts.release.clear()
    cache = small_cache(monkeypatch, tts)
    first = second = observer = None
    started = asyncio.Event()

    async def observe():
        started.set()
        await cache.aclose()

    try:
        first = asyncio.create_task(cache.build())
        second = asyncio.create_task(cache.build())
        await until(tts.entered.is_set)
        assert tts.calls == ["Listo"]
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        owner = cache.begin_close()
        observer = asyncio.create_task(observe())
        await started.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        assert not owner.done() and not observer.done() and not first.done()
        assert cache._tts is tts
        tts.release.set()
        await first
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.shield(owner)
        assert cache.size() == 0 and tts.calls == ["Listo"]
        assert cache._tts is None and tts.close_calls == 0
    finally:
        await finish(cache, tasks=[first, second, observer], threading_releases=[tts.release])


async def test_rebuild_preserves_old_caller_array_and_never_closes_borrowed_tts(monkeypatch):
    tts = TTS()
    cache = small_cache(monkeypatch, tts)
    peer = None
    try:
        await cache.build()
        old = cache.get(" ¡LISTO! ")
        old_dale = cache.get("Dale")
        before = old.audio.copy()
        assert old.audio.dtype == np.float32 and old.sample_rate == 16000
        assert old.duration_s == 8 / 16000 and old.text == "Listo"
        tts.output = np.full(8, 9.0, dtype=np.float64)
        tts.fail_for.add("Dale")
        await cache.build()  # A later build really synthesizes again.
        assert tts.calls == ["Listo", "Dale", "Listo", "Dale"]
        assert cache.size() == 2 and cache.get("Dale") is old_dale
        assert cache.get("Listo") is not old
        assert np.array_equal(cache.get("Listo").audio, np.full(8, 9, dtype=np.float32))
        await cache.aclose()
        assert cache.size() == 0 and cache.get("Listo") is None
        assert np.array_equal(old.audio, before)  # No mutation of caller's loan.
        assert cache._tts is cache._synthesize_callback is None
        tts.fail_for.clear()
        peer = small_cache(monkeypatch, tts, phrases=("Listo",))
        await peer.build()
        assert peer.size() == 1 and tts.close_calls == 0
    finally:
        await cache.aclose()
        if peer is not None:
            await peer.aclose()


async def test_fallback_order_sync_returned_coroutine_and_callback_no_fallback(monkeypatch):
    calls = []

    class Fallback:
        sample_rate = 8000

        async def synthesize_to_array(self, _text):
            calls.append("to_array")
            raise ValueError("synthetic method failure")

        def synthesize(self, _text):
            calls.append("synthesize")
            return "not PCM", 1.0

        def generate(self, _text):
            calls.append("generate")

            async def resolve():
                await asyncio.sleep(0)
                return ["metadata", np.arange(4, dtype=np.int16)]

            return resolve()

    tts = Fallback()
    cache = small_cache(monkeypatch, tts, phrases=("Listo",))
    callback_cache = None
    try:
        await cache.build()
        assert calls == ["to_array", "synthesize", "generate"]
        record = cache.get("listo")
        assert np.array_equal(record.audio, np.arange(4, dtype=np.float32))
        assert record.sample_rate == 8000 and record.duration_s == 4 / 8000

        async def failed_callback(_text):
            raise ValueError("synthetic callback failure")

        callback_cache = small_cache(monkeypatch, tts, phrases=("Listo",), callback=failed_callback)
        await callback_cache.build()
        assert callback_cache.size() == 0
        assert calls == ["to_array", "synthesize", "generate"]
    finally:
        await cache.aclose()
        if callback_cache is not None:
            await callback_cache.aclose()


async def test_self_build_rejected_and_inactive_inherited_child_may_later_close(monkeypatch):
    tts = TTS()
    recorded = []
    release_child = asyncio.Event()
    child = None
    cache = None

    async def callback(_text):
        nonlocal child
        try:
            # The timeout safely breaks the old self-join if this guard regresses.
            async with asyncio.timeout(0.1):
                await cache.build()
        except BaseException as error:
            recorded.append((type(error), str(error)))
        for function in (cache.require_close_caller, cache.begin_close):
            try:
                function()
            except RuntimeError as error:
                recorded.append((type(error), str(error)))

        async def later_close():
            await release_child.wait()
            cache.require_close_caller()
            await cache.aclose()

        child = asyncio.create_task(later_close())
        return np.arange(4, dtype=np.float32)

    cache = small_cache(monkeypatch, tts, phrases=("Listo",), callback=callback)
    try:
        await cache.build()
        assert recorded == [
            (RuntimeError, "response_cache_build_from_operation"),
            (RuntimeError, "response_cache_close_from_operation"),
            (RuntimeError, "response_cache_close_from_operation"),
        ]
        assert not cache._closed and cache.size() == 1
        release_child.set()
        await child
        assert cache.size() == 0 and cache._tts is cache._synthesize_callback is None
        assert tts.calls == []
    finally:
        await finish(cache, tasks=[child], async_releases=[release_child])


async def test_native_pin_keeps_guard_after_callback_cancel_until_real_native_exit(monkeypatch):
    tts = TTS()
    native_entered, native_release = threading.Event(), threading.Event()
    native_exited = threading.Event()
    callback_release, child_release = asyncio.Event(), asyncio.Event()
    callback_exited = asyncio.Event()
    native = child = building = None
    native_errors = []
    cache = None

    async def callback(_text):
        nonlocal native, child
        pin = cache.retain_current_operation()
        assert pin is not None

        def worker():
            try:
                native_entered.set()
                assert native_release.wait(5), "synthetic external native borrower not released"
                try:
                    cache.require_close_caller()
                except RuntimeError as error:
                    native_errors.append(str(error))
                else:
                    native_errors.append("guard missing")
            finally:
                native_exited.set()

        def native_finished(done):
            if not done.cancelled() and native_exited.is_set():
                pin.release()
                pin.release()  # Receipt release is explicitly idempotent.

        async def later_close():
            await child_release.wait()
            cache.require_close_caller()
            assert cache.retain_current_operation() is None
            await cache.aclose()

        native = asyncio.create_task(asyncio.to_thread(worker))
        native.add_done_callback(native_finished, context=Context())
        child = asyncio.create_task(later_close())
        await callback_release.wait()
        callback_exited.set()
        raise asyncio.CancelledError  # Mirrors an early interrupted async observer.

    cache = small_cache(monkeypatch, tts, phrases=("Listo",), callback=callback)
    try:
        assert cache.retain_current_operation() is None
        building = asyncio.create_task(cache.build())
        await until(native_entered.is_set)
        cache.request_close()
        callback_release.set()
        await callback_exited.wait()
        # The parent operation has entered its final drain. Its actual work
        # task remains live until the independently released native receipt.
        assert any(not task.done() for task in cache._work._tasks)
        assert not building.done()
        assert not native.done() and cache._tts is tts
        native_release.set()
        await native  # Owner drains this external borrower before cache.aclose.
        await building
        assert native_errors == ["response_cache_close_from_operation"]
        child_release.set()
        await child
        assert cache._tts is cache._synthesize_callback is None and cache.size() == 0
    finally:
        await finish(cache, tasks=[building, native, child], threading_releases=[native_release],
                     async_releases=[callback_release, child_release])


async def test_seal_during_synthesis_postlude_prevents_cache_publication(monkeypatch):
    tts = TTS()
    cache = small_cache(monkeypatch, tts, phrases=("Listo",))
    entered, release = asyncio.Event(), asyncio.Event()
    actual = cache._synthesize

    async def postlude(text):
        audio = await actual(text)
        entered.set()
        await release.wait()
        return audio

    monkeypatch.setattr(cache, "_synthesize", postlude)
    building = None
    try:
        building = asyncio.create_task(cache.build())
        await entered.wait()
        owner = cache.begin_close()
        assert not owner.done() and cache._tts is tts
        assert cache.size() == 0
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await progressed.wait()
        await asyncio.sleep(0)
        assert not owner.done() and cache._tts is tts
        release.set()
        await building
        await asyncio.shield(owner)
        assert cache.size() == 0 and cache.get("Listo") is None
        assert cache._tts is cache._synthesize_callback is None
    finally:
        await finish(cache, tasks=[building], async_releases=[release])


async def test_close_disposes_callback_off_loop_context_empty_and_guards_reentry(monkeypatch):
    tts = TTS()
    marker = ContextVar("synthetic_cache_closer_context", default=None)
    observed = []

    class Callback:
        cache = None

        async def __call__(self, _text):
            assert marker.get() is None
            return np.arange(4, dtype=np.float32)

        def __del__(self):
            observed.append(marker.get())
            try:
                self.cache.require_close_caller()
            except RuntimeError as error:
                observed.append(str(error))

    callback = Callback()
    callback_ref = weakref.ref(callback)
    cache = small_cache(monkeypatch, tts, phrases=("Listo",), callback=callback)
    callback.cache = cache
    callback = None
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    token = marker.set("synthetic caller context")
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        await cache.build()
        assert callback_ref() is not None
        owner = cache.begin_close()
        assert owner is cache.begin_close()
        await asyncio.shield(owner)
        assert observed == [None, "response_cache_close_from_owner"]
        assert callback_ref() is None
        assert cache._tts is cache._synthesize_callback is None
        assert tts.close_calls == 0
    finally:
        loop.set_task_factory(previous_factory)
        marker.reset(token)
        await cache.aclose()
