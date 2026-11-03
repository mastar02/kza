"""Actual SpeakerIdentifier lifecycle with bounded synthetic SDK leaves.

No Torch/SpeechBrain code, weights, capture or hardware execute here. Event
timeouts are failure bounds; these cases make no C03 or VRAM-release claim.
"""

import asyncio
from contextvars import ContextVar
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

from src.users.speaker_identifier import SpeakerIdentifier, SpeakerMatch


pytestmark = pytest.mark.asyncio


class InputTensor:
    def __init__(self, audio):
        self.audio = audio

    def unsqueeze(self, axis):
        assert axis == 0
        return self


class OutputTensor:
    def __init__(self, probe):
        self.probe = probe

    def squeeze(self):
        return self

    def cpu(self):
        if self.probe.conversion_phase == "cpu":
            self.probe.hold_conversion()
        return self

    def numpy(self):
        if self.probe.conversion_phase == "numpy":
            self.probe.hold_conversion()
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class Encoder:
    def __init__(self, probe):
        self.probe = probe

    def encode_batch(self, tensor):
        assert isinstance(tensor, InputTensor)
        assert tensor.audio.dtype == np.float32
        self.probe.calls += 1
        self.probe.infer_entered.set()
        assert self.probe.infer_release.wait(5), "synthetic inference not released"
        return OutputTensor(self.probe)


class SDKProbe:
    def __init__(self, monkeypatch):
        self.ctor_entered = threading.Event()
        self.ctor_release = threading.Event()
        self.infer_entered = threading.Event()
        self.infer_release = threading.Event()
        self.conversion_entered = threading.Event()
        self.conversion_release = threading.Event()
        self.release_all()
        self.models = []
        self.calls = 0
        self.conversion_phase = None
        self.conversion_exited = False
        speechbrain = ModuleType("speechbrain")
        speechbrain.__path__ = []
        inference = ModuleType("speechbrain.inference")
        inference.__path__ = []
        speaker = ModuleType("speechbrain.inference.speaker")
        speaker.EncoderClassifier = SimpleNamespace(from_hparams=self.construct)
        speechbrain.inference = inference
        inference.speaker = speaker
        torch = ModuleType("torch")
        torch.from_numpy = InputTensor
        torchaudio = ModuleType("torchaudio")
        torchaudio.list_audio_backends = lambda: []
        for name, module in (
            ("speechbrain", speechbrain),
            ("speechbrain.inference", inference),
            ("speechbrain.inference.speaker", speaker),
            ("torch", torch),
            ("torchaudio", torchaudio),
        ):
            monkeypatch.setitem(sys.modules, name, module)

    def construct(self, **options):
        assert options == {
            "source": "speechbrain/synthetic",
            "savedir": "./models/speaker/speechbrain_synthetic",
            "run_opts": {"device": "cpu"},
        }
        model = Encoder(self)
        self.models.append(weakref.ref(model))
        self.ctor_entered.set()
        assert self.ctor_release.wait(5), "synthetic constructor not released"
        return model

    def hold_conversion(self):
        self.conversion_entered.set()
        assert self.conversion_release.wait(5), "synthetic conversion not released"
        self.conversion_exited = True

    def release_all(self):
        self.ctor_release.set()
        self.infer_release.set()
        self.conversion_release.set()


def identifier():
    return SpeakerIdentifier("speechbrain/synthetic", "cpu")


def pcm():
    return np.zeros(1600, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(probe, models, *, tasks=(), releases=()):
    # Release every native barrier before awaiting native/closer owners.
    probe.release_all()
    for release in releases:
        release.set()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await asyncio.gather(*(model.aclose() for model in models), return_exceptions=True)


async def test_concurrent_load_is_once_and_retains_constructor_until_actual_exit(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.ctor_release.clear()
    model = identifier()
    first = second = None
    try:
        first = asyncio.create_task(asyncio.to_thread(model.load))
        await until(probe.ctor_entered.is_set)
        second = asyncio.create_task(asyncio.to_thread(model.load))
        await until(lambda: model._speaker_active == 2)
        assert len(probe.models) == 1
        assert probe.models[0]() is not None
        assert model._model is None and model._model_type is None
        owner = model.begin_close()
        assert model._speaker_sealed
        assert owner is model.begin_close()
        assert not owner.done()
        assert not first.done() and not second.done()
        probe.ctor_release.set()
        assert await asyncio.gather(first, second) == [None, None]
        await asyncio.shield(owner)
        assert len(probe.models) == 1
        assert model._model is None and model._model_type is None
        assert probe.models[0]() is None
        with pytest.raises(RuntimeError, match="^speaker_closed$"):
            await asyncio.to_thread(model.load)
        assert len(probe.models) == 1
    finally:
        await finish(probe, [model], tasks=[first, second])


@pytest.mark.parametrize("phase", ["inference", "cpu", "numpy"])
async def test_native_inference_and_conversion_keep_admission_until_real_exit(monkeypatch, phase):
    probe = SDKProbe(monkeypatch)
    model = identifier()
    if phase == "inference":
        probe.infer_release.clear()
        entered = probe.infer_entered
    else:
        probe.conversion_phase = phase
        probe.conversion_release.clear()
        entered = probe.conversion_entered
    caller = None
    try:
        await asyncio.to_thread(model.load)
        await asyncio.to_thread(model.load)
        assert len(probe.models) == 1
        caller = asyncio.create_task(asyncio.to_thread(model.get_embedding, pcm()))
        await until(entered.is_set)
        owner = model.begin_close()
        assert model._speaker_active == 1
        assert not caller.done() and not owner.done()
        assert probe.models[0]() is not None
        assert model._model_type == "speechbrain"
        if phase != "inference":
            assert not probe.conversion_exited
        probe.release_all()
        output = await caller
        assert np.array_equal(output, [1.0, 0.0, 0.0])
        if phase != "inference":
            assert probe.conversion_exited
        await asyncio.shield(owner)
        assert model._speaker_active == 0
        assert model._model is None and probe.models[0]() is None
        assert model.compute_similarity(output, output) == 1.0
    finally:
        await finish(probe, [model], tasks=[caller])


async def test_enrollment_admitted_before_seal_finishes_later_samples(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model = identifier()
    first_sample_done = threading.Event()
    continue_samples = threading.Event()
    actual = model.get_embedding
    seen = 0

    def between_samples(audio):
        nonlocal seen
        result = actual(audio)
        seen += 1
        if seen == 1:
            first_sample_done.set()
            assert continue_samples.wait(5), "synthetic enrollment not released"
        return result

    monkeypatch.setattr(model, "get_embedding", between_samples)
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(model.create_enrollment_embedding, [pcm(), pcm()]))
        await until(first_sample_done.is_set)
        assert probe.calls == 1
        assert model._speaker_active == 1
        owner = model.begin_close()
        assert not owner.done()
        continue_samples.set()
        result = await caller
        assert np.array_equal(result, [1.0, 0.0, 0.0])
        assert seen == 2 and probe.calls == 2
        assert len(probe.models) == 1
        await asyncio.shield(owner)
        assert model._model is None and probe.models[0]() is None
    finally:
        await finish(probe, [model], tasks=[caller], releases=[continue_samples])


@pytest.mark.parametrize("operation", ["identify", "verify"])
async def test_composed_operation_keeps_nested_embedding_admitted_after_seal(monkeypatch, operation):
    probe = SDKProbe(monkeypatch)
    model = identifier()
    entered = threading.Event()
    release = threading.Event()
    actual = model.get_embedding

    def delayed(audio):
        entered.set()
        assert release.wait(5), "synthetic embedding admission not released"
        return actual(audio)

    monkeypatch.setattr(model, "get_embedding", delayed)
    enrolled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    call = ((lambda: model.identify(pcm(), {"person": enrolled})) if operation == "identify"
            else (lambda: model.verify(pcm(), enrolled)))
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(call))
        await until(entered.is_set)
        assert model._speaker_active == 1
        owner = model.begin_close()
        assert not owner.done() and probe.models == []
        release.set()
        result = await caller
        if operation == "identify":
            assert isinstance(result, SpeakerMatch)
            assert result.user_id == "person" and result.is_known
            assert result.confidence == 1.0
        else:
            assert result == (True, 1.0)
        await asyncio.shield(owner)
        assert len(probe.models) == 1 and probe.models[0]() is None
    finally:
        await finish(probe, [model], tasks=[caller], releases=[release])


async def test_repeated_observer_cancellation_retains_native_and_close_owners(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.infer_release.clear()
    model = identifier()
    caller = observer = None
    started = asyncio.Event()

    async def observe_close():
        started.set()
        await model.aclose()

    try:
        caller = asyncio.create_task(asyncio.to_thread(model.get_embedding, pcm()))
        await until(probe.infer_entered.is_set)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        owner = model.begin_close()
        observer = asyncio.create_task(observe_close())
        await started.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        assert not observer.done() and not owner.done()
        assert owner is model.begin_close()
        assert model._speaker_active == 1
        assert probe.models[0]() is not None
        # Scheduling continues while the actual native call is held.
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await progressed.wait()
        probe.infer_release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.shield(owner)
        assert model._speaker_active == 0
        assert model._model is None and probe.models[0]() is None
        assert probe.calls == 1
    finally:
        await finish(probe, [model], tasks=[caller, observer])


async def test_closed_instance_rejects_reload_without_closing_peer_or_outputs(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model, peer = identifier(), identifier()
    try:
        output = await asyncio.to_thread(model.get_embedding, pcm())
        await asyncio.to_thread(peer.load)
        cache_value = np.array([7.0], dtype=np.float32)
        cache_ref = weakref.ref(cache_value)
        model._embeddings_cache["synthetic"] = cache_value
        cache_value = None
        peer._embeddings_cache["peer"] = np.array([8.0], dtype=np.float32)
        await model.aclose()
        assert model._model is None and model._embeddings_cache == {}
        assert cache_ref() is None
        assert probe.models[0]() is None and probe.models[1]() is not None
        assert not peer._speaker_sealed
        for call in (
            model.load,
            lambda: model.get_embedding(pcm()),
            lambda: model.identify(pcm(), {}),
            lambda: model.verify(pcm(), output),
            lambda: model.create_enrollment_embedding([pcm()]),
        ):
            with pytest.raises(RuntimeError, match="^speaker_closed$"):
                await asyncio.to_thread(call)
        peer_output = await asyncio.to_thread(peer.get_embedding, pcm())
        assert np.array_equal(peer_output, output)
        assert model.compute_similarity(output, output) == 1.0
        assert "peer" in peer._embeddings_cache
        assert len(probe.models) == 2
    finally:
        await finish(probe, [model, peer])


async def test_closer_is_context_empty_and_reentrant_close_fails_without_self_wait(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model = identifier()
    marker = ContextVar("synthetic_speaker_closer_context", default=None)
    observed = []

    class ReentrantCache(dict):
        def clear(self):
            observed.append(marker.get())
            try:
                model.wait_closed()
            except RuntimeError as error:
                observed.append(str(error))
            super().clear()

    token = marker.set("synthetic caller context")
    try:
        await asyncio.to_thread(model.load)
        model._embeddings_cache = ReentrantCache()
        owner = model.begin_close()
        assert owner is model.begin_close()
        await asyncio.shield(owner)
        assert observed == [None, "speaker_close_from_owner"]
        assert model._model is None and probe.models[0]() is None
    finally:
        marker.reset(token)
        await finish(probe, [model])


@pytest.mark.parametrize('fallback', [False, True])
async def test_resemblyzer_and_speechbrain_import_fallback_retain_actual_inference(
    monkeypatch, fallback,
):
    probe = SDKProbe(monkeypatch)
    probe.infer_release.clear()

    class VoiceEncoder:
        def __init__(self, *, device):
            assert device == 'cpu'
            probe.models.append(weakref.ref(self))

        def embed_utterance(self, audio):
            assert audio.dtype == np.float32
            probe.calls += 1
            probe.infer_entered.set()
            assert probe.infer_release.wait(5), 'synthetic Resemblyzer was not released'
            return np.array([1., 0., 0.], dtype=np.float32)

    sdk = ModuleType('resemblyzer')
    sdk.VoiceEncoder = VoiceEncoder
    monkeypatch.setitem(sys.modules, 'resemblyzer', sdk)
    if fallback:
        monkeypatch.delattr(sys.modules['speechbrain.inference.speaker'], 'EncoderClassifier')
    model = SpeakerIdentifier('speechbrain/synthetic' if fallback else 'resemblyzer', 'cpu')
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(model.get_embedding, pcm()))
        await until(probe.infer_entered.is_set)
        owner = model.begin_close()
        assert model._model_type == 'resemblyzer' and not owner.done()
        assert len(probe.models) == 1 and probe.models[0]() is not None
        probe.infer_release.set()
        output = await caller
        await asyncio.shield(owner)
        assert np.array_equal(output, [1., 0., 0.])
        assert model._model is None and probe.models[0]() is None
        with pytest.raises(RuntimeError, match='^speaker_closed$'):
            await asyncio.to_thread(model.load)
        assert len(probe.models) == probe.calls == 1
    finally:
        await finish(probe, [model], tasks=[caller])
