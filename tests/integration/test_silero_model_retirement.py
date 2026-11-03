"""Real predictor/factory, synthetic Torch leaves; no Hub/cache/model access."""

import asyncio
from contextvars import ContextVar
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

import src.ambient.segmenter as segmenter_module
from src.ambient.segmenter import make_silero_predictor

pytestmark = pytest.mark.asyncio


class Gates:
    def __init__(self):
        self.load_entered, self.load_release = threading.Event(), threading.Event()
        self.eval_entered, self.eval_release = threading.Event(), threading.Event()
        self.call_entered, self.call_release = threading.Event(), threading.Event()
        self.item_entered, self.item_release = threading.Event(), threading.Event()
        self.reset_entered, self.reset_release = threading.Event(), threading.Event()
        self.drop_entered, self.drop_release = threading.Event(), threading.Event()
        self.release_all()
        self.windows = []
        self.resets = 0
        self.failure = None

    def release_all(self):
        for event in (self.load_release, self.eval_release, self.call_release,
                      self.item_release, self.reset_release, self.drop_release):
            event.set()


class Utilities:
    def __init__(self, model):
        self.model = model


class Window:
    def __init__(self, probe, array):
        self.probe, self.array = probe, array

    def __del__(self):
        if self.probe.hold_window_drop:
            gates = self.probe.gates[0]
            gates.drop_entered.set()
            assert gates.drop_release.wait(5), "synthetic tensor destructor not released"


class Probability:
    def __init__(self, gates, value):
        self.gates, self.value = gates, value

    def item(self):
        self.gates.item_entered.set()
        assert self.gates.item_release.wait(5), "synthetic item not released"
        return self.value


class Model:
    def __init__(self, probe, gates):
        self.probe, self.gates = probe, gates

    def eval(self):
        self.gates.eval_entered.set()
        assert self.gates.eval_release.wait(5), "synthetic eval not released"
        failure = self.gates.failure
        if failure == "eval":
            raise RuntimeError("synthetic SDK eval failure")
        if failure == "interrupt":
            raise KeyboardInterrupt("synthetic SDK control")
        if failure == "exit":
            raise SystemExit(7)
        if failure == "cancel":
            raise asyncio.CancelledError("synthetic SDK control")
        if failure == "generator":
            raise GeneratorExit("synthetic SDK control")
        return self

    def __call__(self, window, sample_rate):
        assert sample_rate == 16000 and window.array.dtype == np.float32
        assert window.array.shape == (512,) and self.probe.local.grad > 0
        if self.probe.on_call is not None:
            self.probe.on_call()
        self.gates.windows.append(window.array.copy())
        self.gates.call_entered.set()
        assert self.gates.call_release.wait(5), "synthetic inference not released"
        return Probability(self.gates, 0.2 if len(self.gates.windows) % 2 else 0.8)

    def reset_states(self):
        self.gates.reset_entered.set()
        assert self.gates.reset_release.wait(5), "synthetic reset not released"
        self.gates.resets += 1

    def __del__(self):
        if self.probe.on_drop is not None:
            self.probe.on_drop()


class NoGrad:
    def __init__(self, probe):
        self.probe = probe

    def __enter__(self):
        self.probe.local.grad = getattr(self.probe.local, "grad", 0) + 1

    def __exit__(self, *_args):
        self.probe.local.grad -= 1


class TorchProbe:
    def __init__(self, monkeypatch):
        self.gates = [Gates(), Gates()]
        self.models, self.utils, self.owners, self.options = [], [], [], []
        self.local = threading.local()
        self.on_drop = self.on_call = None
        self.hold_window_drop = False
        torch = ModuleType("torch")
        torch.hub = SimpleNamespace(load=self.load)
        torch.from_numpy = lambda value: Window(self, value)
        torch.no_grad = lambda: NoGrad(self)
        monkeypatch.setitem(sys.modules, "torch", torch)
        self.torch = torch
        original = segmenter_module._SileroPredictor.__init__

        def remember_owner(owner, *args, **kwargs):
            original(owner, *args, **kwargs)
            self.owners.append(owner)

        monkeypatch.setattr(segmenter_module._SileroPredictor, "__init__", remember_owner)

    def load(self, **options):
        assert options == {"repo_or_dir": "snakers4/silero-vad", "model": "silero_vad", "trust_repo": True}
        gates = self.gates[len(self.models)]
        model = Model(self, gates)
        utils = Utilities(model)
        self.models.append(weakref.ref(model))
        self.utils.append(weakref.ref(utils))
        self.options.append(options)
        gates.load_entered.set()
        assert gates.load_release.wait(5), "synthetic Hub load not released"
        if gates.failure == "hub":
            raise RuntimeError("synthetic Hub load failure")
        return model, utils

    def release_all(self):
        for gates in self.gates:
            gates.release_all()


def pcm(frames=1024, dtype=np.float32):
    return np.arange(frames, dtype=dtype)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(probe, *, tasks=()):
    probe.release_all()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await asyncio.gather(*(owner.aclose() for owner in probe.owners), return_exceptions=True)


async def test_factories_serialize_hub_and_eval_but_each_predictor_is_independent(monkeypatch):
    probe = TorchProbe(monkeypatch)
    gates = probe.gates[0]
    gates.eval_release.clear()
    first = second = None
    try:
        first = asyncio.create_task(asyncio.to_thread(make_silero_predictor))
        await until(gates.eval_entered.is_set)
        assert len(probe.owners) == 1 and probe.owners[0]._silero_active == 1
        second = asyncio.create_task(asyncio.to_thread(make_silero_predictor))
        await until(lambda: len(probe.owners) == 2 and probe.owners[1]._silero_active == 1)
        assert len(probe.models) == 1  # B cannot enter Hub while A eval is held.
        gates.eval_release.set()
        a, b = await asyncio.gather(first, second)
        assert callable(a) and callable(a.reset) and a is not b
        assert len(probe.models) == 2 and all(ref() is not None for ref in probe.models)
        assert all(ref() is None for ref in probe.utils)
        await asyncio.to_thread(a._load)
        await asyncio.to_thread(b._load)
        assert len(probe.models) == 2
        assert await asyncio.to_thread(a, pcm()) == 0.8
        assert len(probe.gates[0].windows) == 2 and probe.gates[1].windows == []
    finally:
        await finish(probe, tasks=[first, second])


async def test_retiring_one_predictor_does_not_wait_for_peer_global_load(monkeypatch):
    probe = TorchProbe(monkeypatch)
    second = None
    try:
        a = await asyncio.to_thread(make_silero_predictor)
        probe.gates[1].load_release.clear()
        second = asyncio.create_task(asyncio.to_thread(make_silero_predictor))
        await until(probe.gates[1].load_entered.is_set)
        owner = a.begin_close()
        await until(owner.done)
        owner.result()
        assert not second.done() and probe.models[1]() is not None
        assert probe.models[0]() is None
        probe.gates[1].load_release.set()
        b = await second
        assert await asyncio.to_thread(b, pcm()) == 0.8
        assert not b._silero_sealed
        with pytest.raises(RuntimeError, match="^silero_closed$"):
            await asyncio.to_thread(a, pcm())
        assert len(probe.models) == 2
    finally:
        await finish(probe, tasks=[second])


@pytest.mark.parametrize("phase", ["call", "item", "reset", "tensor_cleanup"])
async def test_native_and_last_tensor_cleanup_keep_retirement_pending(monkeypatch, phase):
    probe = TorchProbe(monkeypatch)
    caller = None
    try:
        predictor = await asyncio.to_thread(make_silero_predictor)
        gates = probe.gates[0]
        if phase == "reset":
            gates.reset_release.clear()
            entered, call = gates.reset_entered, predictor.reset
        elif phase == "tensor_cleanup":
            probe.hold_window_drop = True
            gates.drop_release.clear()
            entered, call = gates.drop_entered, lambda: predictor(pcm(512))
        else:
            getattr(gates, phase + "_release").clear()
            entered = getattr(gates, phase + "_entered")
            def call():
                return predictor(pcm(512))
        caller = asyncio.create_task(asyncio.to_thread(call))
        await until(entered.is_set)
        owner = predictor.begin_close()
        assert owner is predictor.begin_close()
        assert predictor._silero_active == 1
        assert not caller.done() and not owner.done()
        assert probe.models[0]() is not None
        probe.release_all()
        result = await caller
        assert result == (None if phase == "reset" else 0.2)
        await asyncio.shield(owner)
        assert predictor._model is predictor._utils is predictor._torch is None
        assert probe.models[0]() is None and probe.utils[0]() is None
    finally:
        await finish(probe, tasks=[caller])


async def test_callable_float_windows_no_grad_max_silence_and_reusable_reset(monkeypatch, caplog):
    probe = TorchProbe(monkeypatch)
    try:
        predictor = await asyncio.to_thread(make_silero_predictor)
        audio = pcm(1025, np.float64)
        result = await asyncio.to_thread(predictor, audio)
        assert type(result) is float and result == 0.8
        assert len(probe.gates[0].windows) == 2
        assert np.array_equal(probe.gates[0].windows[0], audio[:512].astype(np.float32))
        assert np.array_equal(probe.gates[0].windows[1], audio[512:1024].astype(np.float32))
        assert await asyncio.to_thread(predictor, pcm(511)) == 0.0
        assert len(probe.gates[0].windows) == 2
        assert any("tratado como silencio" in record.getMessage() for record in caplog.records)
        await asyncio.to_thread(predictor.reset)
        assert probe.gates[0].resets == 1 and not predictor._silero_sealed
        assert await asyncio.to_thread(predictor, pcm(512)) == 0.2
        assert len(probe.models) == 1
    finally:
        await finish(probe)


async def test_repeated_close_observer_cancellation_keeps_real_call_owned(monkeypatch):
    probe = TorchProbe(monkeypatch)
    caller = observer = None
    started = asyncio.Event()
    try:
        predictor = await asyncio.to_thread(make_silero_predictor)
        probe.gates[0].call_release.clear()
        caller = asyncio.create_task(asyncio.to_thread(predictor, pcm(512)))
        await until(probe.gates[0].call_entered.is_set)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        owner = predictor.begin_close()

        async def observe():
            started.set()
            await predictor.aclose()

        observer = asyncio.create_task(observe())
        await started.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        assert not owner.done() and not observer.done()
        assert predictor._silero_active == 1 and probe.models[0]() is not None
        probe.release_all()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.shield(owner)
        assert probe.models[0]() is None
    finally:
        await finish(probe, tasks=[caller, observer])


@pytest.mark.parametrize("failure", ["hub", "eval", "late"])
async def test_retained_factory_error_does_not_retain_sdk_model_or_utils(monkeypatch, failure):
    probe = TorchProbe(monkeypatch)
    probe.gates[0].failure = failure
    if failure == "late":
        actual_load = segmenter_module._SileroPredictor._load

        def late_failure(owner):
            actual_load(owner)
            raise RuntimeError("synthetic failure after native initialization")

        monkeypatch.setattr(segmenter_module._SileroPredictor, "_load", late_failure)
    retained = None
    try:
        try:
            await asyncio.to_thread(make_silero_predictor)
        except RuntimeError as error:
            retained = error
        assert retained is not None and str(retained) == "silero_initialization_failed"
        assert retained.__cause__ is retained.__context__ is None
        owner = probe.owners[0]  # Even a separately retained wrapper owns no SDK ref.
        assert owner._model is owner._utils is owner._torch is None
        assert owner._silero_sealed and owner._silero_close_done.is_set()
        assert probe.models[0]() is None and probe.utils[0]() is None
        trace = retained.__traceback__
        while trace is not None:
            assert trace.tb_frame.f_code.co_name not in {"eval", "load", "_load", "late_failure"}
            if trace.tb_frame.f_code.co_name == "make_silero_predictor":
                assert trace.tb_frame.f_locals["predictor"] is None
                assert trace.tb_frame.f_locals["torch"] is None
            trace = trace.tb_next
    finally:
        await finish(probe)


@pytest.mark.parametrize("failure,expected", [
    ("interrupt", KeyboardInterrupt),
    ("exit", SystemExit),
    ("cancel", asyncio.CancelledError),
    ("generator", GeneratorExit),
])
async def test_control_failure_keeps_type_without_retaining_sdk_traceback(monkeypatch, failure, expected):
    probe = TorchProbe(monkeypatch)
    probe.gates[0].failure = failure
    retained = None
    try:
        # Await the coroutine directly: do not spawn an unhandled control Task.
        try:
            await asyncio.to_thread(make_silero_predictor)
        except BaseException as error:
            retained = error
        assert type(retained) is expected
        assert retained.__cause__ is retained.__context__ is None
        if expected is SystemExit:
            assert retained.code == 7
        assert probe.models[0]() is None and probe.utils[0]() is None
        assert probe.owners[0]._model is probe.owners[0]._torch is None
    finally:
        await finish(probe)


async def test_native_and_closer_reentry_fail_without_self_wait_and_context_leak(monkeypatch):
    probe = TorchProbe(monkeypatch)
    errors, observed = [], []
    marker = ContextVar("synthetic_silero_closer_context", default=None)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    token = marker.set("synthetic caller context")
    try:
        predictor = await asyncio.to_thread(make_silero_predictor)

        def from_call():
            for function in (predictor.wait_closed, predictor.close, predictor.begin_close):
                try:
                    function()
                except RuntimeError as error:
                    errors.append(str(error))

        probe.on_call = from_call
        assert await asyncio.to_thread(predictor, pcm(512)) == 0.2
        assert errors == ["silero_close_from_worker"] * 3
        assert not predictor._silero_sealed
        probe.on_call = None

        def from_drop():
            observed.append(marker.get())
            try:
                predictor.wait_closed()
            except RuntimeError as error:
                observed.append(str(error))

        probe.on_drop = from_drop
        loop.set_task_factory(asyncio.eager_task_factory)
        owner = predictor.begin_close()
        assert owner is predictor.begin_close()
        await asyncio.shield(owner)
        assert observed == [None, "silero_close_from_owner"]
        assert probe.models[0]() is None
        with pytest.raises(RuntimeError, match="^silero_closed$"):
            await asyncio.to_thread(predictor.reset)
        with pytest.raises(RuntimeError, match="^silero_closed$"):
            await asyncio.to_thread(predictor._load)
        assert len(probe.models) == 1
    finally:
        loop.set_task_factory(previous_factory)
        marker.reset(token)
        probe.on_call = probe.on_drop = None
        await finish(probe)
