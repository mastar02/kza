"""Actual enrollment and actual SpeakerIdentifier with synthetic SDK/WAV leaves."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave
import weakref

import numpy as np
import pytest

from tests.integration.test_speaker_model_retirement import Encoder, InputTensor, OutputTensor, SDKProbe


def project_root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "scripts/enroll_voice.py").is_file())


@pytest.fixture
def enrollment(monkeypatch, tmp_path):
    probe = SDKProbe(monkeypatch)
    spec = importlib.util.spec_from_file_location("synthetic_enrollment", project_root() / "scripts/enroll_voice.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "samples"
    folder.mkdir()
    adapters = []
    state = SimpleNamespace(
        failures={}, gates={}, threads=[], owners=[], load_receipts=[],
        arrays=[], tensors=[], vectors=[], means=[], save_copies=[], payloads=[], actor_entered={},
        embedding_values=[], embedding_index=0, confidence=1.0, windows=[],
        terminal_events=[], events=[], info=[], errors=[], peers=[], peer_refs=[], peer_load=False,
        output=tmp_path / "outputs", tmp=tmp_path, folder=folder,
    )

    def hold(phase):
        gate = threading.Event(), threading.Event()
        state.gates[phase] = gate
        return gate

    def hit(phase):
        if phase in state.gates:
            entered, release = state.gates[phase]
            entered.set()
            assert release.wait(5), f"synthetic {phase} not released"
        factory = state.failures.get(phase)
        if factory is not None:
            raise factory()

    def wav(name="sample.wav", samples=16000, rate=16000, channels=1, values=None):
        with wave.open(str(folder / name), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            data = np.full(samples * channels, 100, dtype=np.int16) if values is None else np.asarray(values, dtype=np.int16)
            output.writeframes(data.tobytes())

    wav()
    actual = module.SpeakerIdentifier

    def construct(*args, **kwargs):
        adapter = actual(*args, **kwargs)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(module, "SpeakerIdentifier", construct)
    from src.core import resource_retirement as resources

    actual_owner = module.ResourceOwner

    def resource_owner(*args, **kwargs):
        owner = actual_owner(*args, **kwargs)
        state.owners.append(owner)
        original_run = owner._closer.run
        phase = "speaker-postlude" if owner.error_prefix == "enrollment_speaker" else "vad-postlude"
        entered = threading.Event()
        state.actor_entered[owner.error_prefix] = entered

        def native_run():
            entered.set()
            original_run()
            hit(phase)

        owner._closer.run = native_run
        return owner

    monkeypatch.setattr(module, "ResourceOwner", resource_owner)

    def observe_load(kind):
        prefix = "enrollment_speaker" if kind == "speaker" else "enrollment_vad"
        owner = next(owner for owner in state.owners if owner.error_prefix == prefix)
        assert owner in resources._OWNERS
        assert owner.resource is not None
        assert any(lease.active and lease.owner is threading.current_thread() for lease in owner._operations)
        if kind == "speaker":
            assert owner.resource is adapters[-1] and owner.resource._model is None
            assert owner._caller_guard is not None
        else:
            assert owner.resource.model is None and owner.resource.utilities is None
        state.load_receipts.append(kind)

    original_sdk = sys.modules["speechbrain.inference.speaker"].EncoderClassifier.from_hparams

    def sdk_construct(**options):
        if not state.peer_load:
            observe_load("speaker")
        model = original_sdk(**options)
        if not state.peer_load:
            hit("load")
        return model

    monkeypatch.setattr(sys.modules["speechbrain.inference.speaker"].EncoderClassifier, "from_hparams", sdk_construct)
    original_encode, original_numpy = Encoder.encode_batch, OutputTensor.numpy

    def encode(model, tensor):
        result = original_encode(model, tensor)
        hit("embedding")
        return result

    def to_numpy(tensor):
        result = original_numpy(tensor)
        if state.embedding_values:
            result = np.asarray(state.embedding_values[state.embedding_index], dtype=np.float32).copy()
            state.embedding_index += 1
        state.vectors.append(weakref.ref(result))
        return result

    def speaker_drop(_model):
        hit("speaker-drop")

    monkeypatch.setattr(Encoder, "encode_batch", encode)
    monkeypatch.setattr(Encoder, "__del__", speaker_drop, raising=False)
    monkeypatch.setattr(OutputTensor, "numpy", to_numpy)
    vad_models, utilities = [], []

    class Utilities:
        def __del__(self):
            hit("utilities-drop")

    class VAD:
        def __call__(self, tensor, sample_rate):
            assert sample_rate == 16000 and tensor.audio.shape == (512,)
            state.windows.append(float(tensor.audio[0]))
            hit("trim")
            return SimpleNamespace(item=lambda: state.confidence)

        def __del__(self):
            hit("vad-drop")

    def hub_load(**options):
        assert options == {"repo_or_dir": "snakers4/silero-vad", "model": "silero_vad", "trust_repo": True}
        observe_load("vad")
        model, utility = VAD(), Utilities()
        vad_models.append(weakref.ref(model))
        utilities.append(weakref.ref(utility))
        hit("hub-load")
        return model, utility

    torch = sys.modules["torch"]
    torch.hub = SimpleNamespace(load=hub_load)
    torch.no_grad = nullcontext

    def from_numpy(audio):
        state.arrays.append(weakref.ref(audio))
        tensor = InputTensor(audio)
        state.tensors.append(weakref.ref(tensor))
        return tensor

    torch.from_numpy = from_numpy
    original_load_wav = module.load_wav

    def load_wav(path):
        audio = original_load_wav(path)
        if audio is not None:
            state.arrays.append(weakref.ref(audio))
        return audio

    monkeypatch.setattr(module, "load_wav", load_wav)
    original_result = module._EnrollmentResult

    def result(*args, **kwargs):
        value = original_result(*args, **kwargs)
        if value.mean_emb is not None:
            state.means.append(weakref.ref(value.mean_emb))
        return value

    monkeypatch.setattr(module, "_EnrollmentResult", result)

    class NumpyLeaves:
        def __getattr__(self, name):
            return getattr(np, name)

        def mean(self, values, *args, **kwargs):
            value = np.mean(values, *args, **kwargs)
            if kwargs.get("axis") == 0:
                state.means.append(weakref.ref(value))
                hit("mean")
            return value

        def save(self, path, array):
            state.save_copies.append(weakref.ref(array))
            state.events.append("npy")
            assert all(owner._confirmed and owner._handle.is_done() for owner in state.owners)
            hit("write")
            return np.save(path, array)

    monkeypatch.setattr(module, "np", NumpyLeaves())
    terminal = []
    original_info = module.logger.info

    def log_info(message, *args, **kwargs):
        state.info.append(message)
        if message.startswith("dry-run"):
            terminal.append({
                "speaker_released": all(adapter._model is None for adapter in adapters),
                "vad_released": all(reference() is None for reference in vad_models),
                "utilities_released": all(reference() is None for reference in utilities),
            })
        if message.startswith(("dry-run", "Guardado:", "Metadata:", "Listo.")):
            state.terminal_events.append(message)
            assert all(owner._confirmed and owner._handle.is_done() for owner in state.owners)
        if message.startswith("Embedding promedio"):
            hit("report")
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(module.logger, "info", log_info)

    original_error = module.logger.error

    def log_error(message, *args, **kwargs):
        state.errors.append(message)
        return original_error(message, *args, **kwargs)

    monkeypatch.setattr(module.logger, "error", log_error)

    def run(*extra, dry_run=True):
        monkeypatch.setattr(sys, "argv", ["enroll_voice.py", "--user", "synthetic", "--dir", str(folder),
                                         "--model", "speechbrain/synthetic", "--device", "cpu",
                                         "--output-dir", str(state.output),
                                         *(["--dry-run"] if dry_run else []), *extra])
        return module.main()

    def start(*extra, dry_run=True):
        outcome = SimpleNamespace(value=None, error=None, done=threading.Event())

        def work():
            try:
                outcome.value = run(*extra, dry_run=dry_run)
            except BaseException as error:
                outcome.error = error
            finally:
                outcome.done.set()

        thread = threading.Thread(target=work, daemon=True, name="enrollment-test-caller")
        state.threads.append(thread)
        thread.start()
        return outcome

    def new_peer():
        peer = actual("speechbrain/synthetic", "cpu")
        state.peer_load = True
        try:
            peer.load()
        finally:
            state.peer_load = False
        state.peers.append(peer)
        state.peer_refs.append(weakref.ref(peer._model))
        return peer

    try:
        state.module, state.probe, state.adapters = module, probe, adapters
        state.models, state.utilities, state.terminal = vad_models, utilities, terminal
        state.run, state.start, state.hold, state.wav = run, start, hold, wav
        state.new_peer = new_peer
        yield state
    finally:
        probe.release_all()
        for _, release in state.gates.values():
            release.set()
        failures = []
        for thread in state.threads:
            try:
                thread.join(5)
                assert thread._handle.is_done(), "synthetic caller did not exit natively"
            except BaseException as error:
                failures.append(error)
        for adapter in [*adapters, *state.peers]:
            try:
                adapter.close()
            except BaseException as error:
                failures.append(error)
        for owner in state.owners:
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("enrollment fixture cleanup failed", failures)


def test_enrollment_dry_run_retires_actual_speaker_with_visible_wrapper(enrollment):
    assert enrollment.run() == 0
    assert len(enrollment.adapters) == 1 and len(enrollment.probe.models) == 1
    assert enrollment.adapters[0]._model is None
    assert enrollment.probe.models[0]() is None
    assert enrollment.terminal == [{"speaker_released": True, "vad_released": True, "utilities_released": True}]


def test_enrollment_terminal_dry_run_follows_direct_hub_reference_retirement(enrollment):
    assert enrollment.run("--vad-trim") == 0
    assert len(enrollment.models) == len(enrollment.utilities) == 1
    assert enrollment.terminal == [{"speaker_released": True, "vad_released": True, "utilities_released": True}]


def _retired(enrollment):
    assert all(adapter._model is None and adapter._embeddings_cache == {} for adapter in enrollment.adapters)
    owned_models = [reference for reference in enrollment.probe.models if reference not in enrollment.peer_refs]
    for references in (
        owned_models, enrollment.models, enrollment.utilities, enrollment.arrays,
        enrollment.tensors, enrollment.vectors, enrollment.means,
        enrollment.save_copies, enrollment.payloads,
    ):
        assert all(reference() is None for reference in references), "owned payload remained reachable"
    for owner in enrollment.owners:
        assert owner._confirmed and owner.resource is None and owner._caller_guard is None
        assert owner._target_exited.is_set() and owner._handle.is_done()


def _failure(function):
    try:
        function()
    except BaseException as error:
        return error
    raise AssertionError("synthetic failure did not reach the owning boundary")


def _leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaves(child)]
    return [error]


class _RawError(RuntimeError):
    pass


class _RawInterrupt(KeyboardInterrupt):
    pass


class _RawExit(SystemExit):
    pass


def _two_orthogonal(enrollment):
    enrollment.wav("second.wav")
    enrollment.embedding_values = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def _previous_artifacts(enrollment):
    enrollment.output.mkdir()
    npy = enrollment.output / "synthetic_voice.npy"
    metadata = enrollment.output / "synthetic_voice.json"
    npy.write_bytes(b"previous-npy")
    metadata.write_text('{"prior": true}')
    return npy, metadata


def test_enrollment_no_valid_audio_retires_direct_hub_before_return_one(enrollment):
    enrollment.confidence = 0.0
    assert enrollment.run("--vad-trim", dry_run=False) == 1
    assert enrollment.adapters == [] and enrollment.probe.models == []
    assert len(enrollment.models) == len(enrollment.utilities) == 1
    assert enrollment.load_receipts == ["vad"]
    assert not enrollment.output.exists()
    _retired(enrollment)


def test_enrollment_dry_run_precedes_low_consistency_rejection(enrollment):
    _two_orthogonal(enrollment)
    assert enrollment.run("--min-consistency", "0.9") == 0
    assert not enrollment.output.exists()
    assert not any("REJECTED" in message for message in enrollment.errors)
    _retired(enrollment)


@pytest.mark.parametrize("force, expected", [(False, 2), (True, 0)])
def test_enrollment_orthogonal_samples_reject_or_force_same_normalized_mean(enrollment, force, expected):
    _two_orthogonal(enrollment)
    assert enrollment.run(*(["--force"] if force else []), dry_run=False) == expected
    if force:
        saved = np.load(enrollment.output / "synthetic_voice.npy")
        assert saved.dtype == np.float32
        np.testing.assert_allclose(saved, np.array([2**-0.5, 2**-0.5, 0.0], dtype=np.float32))
        metadata = json.loads((enrollment.output / "synthetic_voice.json").read_text())
        assert metadata["intra_consistency"] == 0.0 and metadata["n_samples"] == 2
    else:
        assert not enrollment.output.exists()
        assert any("REJECTED" in message for message in enrollment.errors)
    _retired(enrollment)


def test_enrollment_exact_metadata_and_npy_then_timestamp_then_json(enrollment, monkeypatch):
    _two_orthogonal(enrollment)
    original_write = Path.write_text

    def now(zone):
        assert zone is enrollment.module.timezone.utc
        assert (enrollment.output / "synthetic_voice.npy").exists()
        enrollment.events.append("metadata")
        return SimpleNamespace(isoformat=lambda: "2026-11-03T00:00:00+00:00")

    def write(path, text, *args, **kwargs):
        if path.name == "synthetic_voice.json":
            enrollment.events.append("json")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(enrollment.module, "datetime", SimpleNamespace(now=now))
    monkeypatch.setattr(Path, "write_text", write)
    assert enrollment.run("--force", dry_run=False) == 0
    assert enrollment.events == ["npy", "metadata", "json"]
    metadata = json.loads((enrollment.output / "synthetic_voice.json").read_text())
    assert metadata == {
        "user_id": "synthetic", "model": "speechbrain/synthetic", "device": "cpu",
        "embedding_dim": 3, "n_samples": 2, "n_skipped_short": 0,
        "min_duration_s": 1.0, "intra_consistency": 0.0, "source_dirs": ["samples"],
        "vad_trim": False, "created_at": "2026-11-03T00:00:00+00:00",
    }
    _retired(enrollment)


def test_enrollment_registers_and_adopts_both_real_owners_before_sdk_load(enrollment):
    assert enrollment.run("--vad-trim", "--min-duration", "99") == 0
    assert enrollment.load_receipts == ["vad", "speaker"]
    assert len(enrollment.windows) == 31
    assert len(enrollment.owners) == 2
    _retired(enrollment)


def test_enrollment_retirement_keeps_independent_real_speaker_usable(enrollment):
    peer = enrollment.new_peer()
    peer_model = weakref.ref(peer._model)
    assert enrollment.run() == 0
    _retired(enrollment)
    assert peer_model() is peer._model
    np.testing.assert_array_equal(peer.get_embedding(np.ones(16000, dtype=np.float32)), [1.0, 0.0, 0.0])
    assert peer_model() is not None


@pytest.mark.parametrize("phase", ["npy", "json", "source_dirs", "timestamp"])
def test_enrollment_partial_publication_keeps_existing_artifacts(enrollment, monkeypatch, phase):
    npy, metadata = _previous_artifacts(enrollment)
    if phase == "npy":
        enrollment.failures["write"] = lambda: OSError("synthetic NPY failure")
    elif phase == "json":
        original = Path.write_text

        def write(path, text, *args, **kwargs):
            if path == metadata:
                assert npy.read_bytes() != b"previous-npy"
                raise OSError("synthetic JSON failure")
            return original(path, text, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)
    elif phase == "source_dirs":
        new_root = enrollment.tmp / "different-root"
        new_root.mkdir()
        monkeypatch.setattr(enrollment.module, "ROOT", new_root)
    else:
        def now(_zone):
            assert npy.read_bytes() != b"previous-npy"
            raise RuntimeError("synthetic timestamp failure")

        monkeypatch.setattr(enrollment.module, "datetime", SimpleNamespace(now=now))
    retained = _failure(lambda: enrollment.run(dry_run=False))
    assert isinstance(retained, RuntimeError)
    assert "enrollment_publish_failed" in str(retained)
    assert metadata.read_text() == '{"prior": true}'
    if phase == "npy":
        assert npy.read_bytes() == b"previous-npy"
    else:
        np.testing.assert_array_equal(np.load(npy), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert not enrollment.terminal_events
    _retired(enrollment)


@pytest.mark.parametrize("phase", ["hub-load", "load", "trim", "embedding", "mean", "report", "write"])
@pytest.mark.parametrize("kind", ["error", "interrupt", "exit-payload"])
def test_enrollment_retained_fresh_failure_does_not_retain_owned_payloads(enrollment, phase, kind):
    def fresh():
        payload = np.full(32, 0.375, dtype=np.float32)
        enrollment.payloads.append(weakref.ref(payload))
        error = {"error": _RawError, "interrupt": _RawInterrupt, "exit-payload": _RawExit}[kind](payload)
        return error

    enrollment.failures[phase] = fresh
    retained = _failure(lambda: enrollment.run("--vad-trim", dry_run=False))
    leaves = _leaves(retained)
    if kind == "error":
        assert all(type(leaf) is RuntimeError for leaf in leaves)
        assert all("0.375" not in str(leaf) for leaf in leaves)
    elif kind == "interrupt":
        assert all(type(leaf) is KeyboardInterrupt and leaf.args == () for leaf in leaves)
    else:
        assert all(type(leaf) is SystemExit and leaf.code == 1 for leaf in leaves)
    assert not enrollment.terminal_events
    assert not (enrollment.output / "synthetic_voice.json").exists()
    _retired(enrollment)


@pytest.mark.parametrize("phase", ["load", "embedding"])
def test_enrollment_preserves_safe_system_exit_code_after_retirement(enrollment, phase):
    enrollment.failures[phase] = lambda: SystemExit(7)
    retained = _failure(lambda: enrollment.run("--vad-trim"))
    assert all(type(leaf) is SystemExit and leaf.code == 7 for leaf in _leaves(retained))
    _retired(enrollment)


def _complete(enrollment, outcome):
    assert outcome.done.wait(5), "enrollment caller did not complete after test gate release"
    for thread in enrollment.threads:
        thread.join(1)
        assert thread._handle.is_done()
    assert outcome.error is None, outcome.error
    assert outcome.value == 0


def test_enrollment_starts_both_native_closers_while_first_destructor_is_held(enrollment):
    speaker_entered, speaker_release = enrollment.hold("speaker-drop")
    vad_entered, vad_release = enrollment.hold("vad-postlude")
    outcome = enrollment.start("--vad-trim", dry_run=False)
    assert speaker_entered.wait(5), "actual speaker model destructor did not enter"
    assert vad_entered.wait(5), "second real closer did not pass target exit"
    owners = {owner.error_prefix: owner for owner in enrollment.owners}
    speaker, vad = owners["enrollment_speaker"], owners["enrollment_vad"]
    assert speaker._handle.ident and vad._handle.ident
    assert not speaker._handle.is_done() and not vad._handle.is_done()
    assert vad._target_exited.is_set()
    assert not outcome.done.is_set() and not enrollment.output.exists()
    assert enrollment.terminal_events == []
    vad_release.set()
    vad._handle.join(1)
    assert vad._handle.is_done()
    assert not outcome.done.is_set() and not enrollment.output.exists()
    speaker_release.set()
    _complete(enrollment, outcome)
    _retired(enrollment)


@pytest.mark.parametrize("kind", ["speaker", "vad"])
def test_enrollment_publication_waits_for_real_closer_native_postlude(enrollment, kind):
    entered, release = enrollment.hold(f"{kind}-postlude")
    outcome = enrollment.start("--vad-trim", dry_run=False)
    assert entered.wait(5), "actual native closer did not reach postlude"
    held_prefix = f"enrollment_{kind}"
    other_prefix = "enrollment_vad" if kind == "speaker" else "enrollment_speaker"
    assert enrollment.actor_entered[other_prefix].wait(5)
    owners = {owner.error_prefix: owner for owner in enrollment.owners}
    owners[other_prefix]._handle.join(1)
    assert owners[other_prefix]._handle.is_done()
    held = owners[held_prefix]
    assert held._target_exited.is_set() and not held._handle.is_done()
    assert all(reference() is None for reference in enrollment.probe.models)
    assert all(reference() is None for reference in [*enrollment.models, *enrollment.utilities])
    assert not outcome.done.is_set() and not enrollment.output.exists()
    assert enrollment.terminal_events == []
    release.set()
    _complete(enrollment, outcome)
    _retired(enrollment)


def test_enrollment_retains_native_embedding_before_starting_resource_closers(enrollment):
    enrollment.probe.infer_release.clear()
    outcome = enrollment.start("--vad-trim", dry_run=False)
    assert enrollment.probe.infer_entered.wait(5), "actual synthetic encoder did not enter"
    assert all(not owner._start_called for owner in enrollment.owners)
    assert enrollment.probe.models[0]() is enrollment.adapters[0]._model
    assert enrollment.models[0]() is not None and enrollment.utilities[0]() is not None
    assert not outcome.done.is_set() and not enrollment.output.exists()
    assert enrollment.terminal_events == []
    enrollment.probe.infer_release.set()
    _complete(enrollment, outcome)
    _retired(enrollment)
