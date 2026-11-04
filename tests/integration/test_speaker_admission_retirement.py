"""Immutable baseline probes. Project/SDK imports occur only when root runs them."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


def _project_root():
    return next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "src/users/speaker_identifier.py").is_file()
    )


def _active(adapter):
    leases = vars(adapter).get("_speaker_leases")
    if leases is not None:
        return sum(lease.active for lease in leases)
    return vars(adapter)["_speaker_active"]


def _reserved(adapter):
    token = vars(adapter).get("_speaker_close_admission")
    return token is not None or vars(adapter).get("_speaker_close_started", False)


class _ExitFault:
    """Release the actual lock, then interrupt one selected metadata boundary."""

    def __init__(self, condition, predicate):
        self.condition = condition
        self.predicate = predicate
        self.fired = False

    def __enter__(self):
        self.condition.__enter__()
        return self

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if not self.fired and self.predicate():
            self.fired = True
            raise KeyboardInterrupt
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)


def _descriptor(error):
    return {"type": type(error).__name__, "text": str(error)}


def _emit(mode, reason, **scalars):
    print("SPEAKER_PROBE=" + json.dumps({"mode": mode, "reason": reason, **scalars}), flush=True)


def _native_exit(thread, timeout=1.0):
    thread.join(timeout)
    return thread._handle.is_done()


def _fixtures():
    fixture = runpy.run_path(str(_project_root() / "tests/integration/test_speaker_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["SDKProbe"](patch)
    return fixture, patch, probe, fixture["identifier"]()


def _baseline_probe(mode):
    fixture, patch, probe, adapter = _fixtures()
    if mode != "load":
        adapter.load()
        assert len(probe.models) == 1
    baseline_calls = probe.calls
    fault = _ExitFault(
        adapter._speaker_condition,
        (lambda: _reserved(adapter)) if mode == "close" else (lambda: _active(adapter) == 1),
    )
    adapter._speaker_condition = fault
    first = None
    try:
        if mode == "load":
            adapter.load()
        elif mode == "embedding":
            adapter.get_embedding(fixture["pcm"]())
        else:
            adapter.close()
    except BaseException as error:
        first = _descriptor(error)
    assert fault.fired, "fault did not reach the actual publication boundary"
    assert first is not None, "publication interrupt was silently lost"
    if mode != "close":
        assert first["type"] == "KeyboardInterrupt", first
        assert probe.calls == baseline_calls, "inference body entered before admission completed"
        assert len(probe.models) == (0 if mode == "load" else 1)
    else:
        assert first["type"] in {"KeyboardInterrupt", "RuntimeError"}, first

    result = []
    entered = threading.Event()

    def observe_close():
        entered.set()
        try:
            adapter.close()
        except BaseException as error:
            result.append(_descriptor(error))

    observer = threading.Thread(target=observe_close, daemon=True, name="speaker-red-observer")
    observer.start()
    assert entered.wait(1), "external close observer was not dispatched"
    if not _native_exit(observer):
        _emit(
            mode, "observer_hang", fault_fired=fault.fired,
            inference_calls=probe.calls - baseline_calls,
            constructor_calls=len(probe.models), active=_active(adapter),
            model_retained=adapter._model is not None,
            native_exited=False, first_type=first["type"],
        )
        os._exit(1)

    assert not observer.is_alive()
    if mode == "close":
        assert result == [{"type": "RuntimeError", "text": "speaker_close_failed"}], result
        assert adapter._model is not None and probe.models[0]() is adapter._model
        token = vars(adapter).get("_speaker_close_admission")
        assert token is not None and token.finished and token.failure == "speaker_close_failed"
    else:
        assert result == [], result
        assert _active(adapter) == 0
        assert adapter._model is None and adapter._model_type is None
        assert adapter._embeddings_cache == {}
        assert all(reference() is None for reference in probe.models)
    _emit(mode, "confirmed", native_exited=True, first_type=first["type"])
    patch.undo()


def _run_probe(mode):
    script = (
        "import os, runpy, sys; "
        "module = runpy.run_path(sys.argv[1]); "
        "module['_baseline_probe'](sys.argv[2]); "
        "os._exit(0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
        cwd=_project_root(), capture_output=True, text=True, timeout=15,
    )
    records = [
        json.loads(line.removeprefix("SPEAKER_PROBE="))
        for line in completed.stdout.splitlines() if line.startswith("SPEAKER_PROBE=")
    ]
    assert records, f"probe fixture failed before a receipt: {completed.stdout}\n{completed.stderr}"
    receipt = records[-1]
    assert completed.returncode == 0, f"{receipt}\n{completed.stderr}"
    assert receipt["mode"] == mode and receipt["reason"] == "confirmed", receipt
    assert receipt["native_exited"] is True


def test_speaker_load_admission_condition_exit_interrupt_does_not_strand_close():
    _run_probe("load")


def test_speaker_embedding_admission_condition_exit_interrupt_does_not_strand_close():
    _run_probe("embedding")


def test_speaker_close_reservation_condition_exit_interrupt_notifies_followers():
    _run_probe("close")


class _TLSFault:
    def __init__(self, local, fail_at=None):
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "fail_at", fail_at)
        object.__setattr__(self, "sets", 0)
        object.__setattr__(self, "fired", False)
        object.__setattr__(self, "armed", False)

    def arm(self):
        object.__setattr__(self, "armed", True)

    def __getattr__(self, name):
        return getattr(self.local, name)

    def __setattr__(self, name, value):
        if name == "lease":
            object.__setattr__(self, "sets", self.sets + 1)
            if not self.fired and (self.armed or self.sets == self.fail_at):
                object.__setattr__(self, "fired", True)
                object.__setattr__(self, "armed", False)
                raise KeyboardInterrupt
        setattr(self.local, name, value)


class _WaitCondition:
    def __init__(self, condition, adapter, fail_notify=False):
        self.condition = condition
        self.adapter = adapter
        self.fail_notify = fail_notify
        self.fired = False
        self.wait_entered = threading.Event()

    def __enter__(self):
        self.condition.__enter__()
        return self

    def __exit__(self, *args):
        return self.condition.__exit__(*args)

    def wait(self, timeout=None):
        self.wait_entered.set()
        return self.condition.wait(timeout)

    def notify_all(self):
        if self.fail_notify and not self.fired and _active(self.adapter) == 0:
            self.fired = True
            raise KeyboardInterrupt
        return self.condition.notify_all()

    def __getattr__(self, name):
        return getattr(self.condition, name)


class _SetFault:
    def __init__(self, event, error_type):
        self.event = event
        self.error_type = error_type
        self.fired = False

    def set(self):
        if not self.fired:
            self.fired = True
            raise self.error_type
        return self.event.set()

    def __getattr__(self, name):
        return getattr(self.event, name)


def _spawn(function, name):
    errors = []

    def run():
        try:
            function()
        except BaseException as error:
            errors.append(_descriptor(error))

    thread = threading.Thread(target=run, daemon=True, name=name)
    thread.start()
    return thread, errors


def _require_exit(thread, mode, phase):
    if not _native_exit(thread):
        _emit(mode, "observer_hang", phase=phase, native_exited=False)
        os._exit(1)


def _assert_retired(adapter, probe):
    assert _active(adapter) == 0
    assert adapter._speaker_current() is None
    assert adapter._model is None and adapter._model_type is None
    assert adapter._embeddings_cache == {}
    assert all(reference() is None for reference in probe.models)


def _tls_probe(operation, phase):
    fixture, patch, probe, adapter = _fixtures()
    if operation == "embedding":
        adapter.load()
    local = _TLSFault(adapter._speaker_local, 1 if phase == "publication" else 2)
    adapter._speaker_local = local
    call = adapter.load if operation == "load" else lambda: adapter.get_embedding(fixture["pcm"]())
    caller, errors = _spawn(call, "speaker-tls-caller")
    _require_exit(caller, "tls", "caller")
    assert local.fired and errors == [{"type": "KeyboardInterrupt", "text": ""}], errors
    closer, close_errors = _spawn(adapter.close, "speaker-tls-closer")
    _require_exit(closer, "tls", "close")
    assert close_errors == [], close_errors
    _assert_retired(adapter, probe)
    assert len(probe.models) == (0 if operation == "load" and phase == "publication" else 1)
    assert probe.calls == (1 if operation == "embedding" and phase == "restoration" else 0)
    _emit("tls", "confirmed", operation=operation, phase=phase, native_exited=True)
    patch.undo()


def _notify_probe():
    fixture, patch, probe, adapter = _fixtures()
    adapter.load()
    condition = _WaitCondition(adapter._speaker_condition, adapter, fail_notify=True)
    adapter._speaker_condition = condition
    probe.infer_release.clear()
    caller, errors = _spawn(lambda: adapter.get_embedding(fixture["pcm"]()), "speaker-notify-caller")
    assert probe.infer_entered.wait(5), "actual synthetic encoder did not enter"
    closer, close_errors = _spawn(adapter.close, "speaker-notify-closer")
    assert condition.wait_entered.wait(1), "closer did not enter actual Condition.wait"
    assert adapter._model is not None and not closer._handle.is_done()
    probe.infer_release.set()
    _require_exit(caller, "notify", "caller")
    _require_exit(closer, "notify", "close")
    assert condition.fired
    assert errors == [{"type": "KeyboardInterrupt", "text": ""}], errors
    assert close_errors == [], close_errors
    _assert_retired(adapter, probe)
    _emit("notify", "confirmed", native_exited=True)
    patch.undo()


def _completion_probe(error_name):
    fixture, patch, probe, adapter = _fixtures()
    adapter.load()
    error_type = {"RuntimeError": RuntimeError, "KeyboardInterrupt": KeyboardInterrupt}[error_name]
    fault = _SetFault(adapter._speaker_close_done, error_type)
    adapter._speaker_close_done = fault
    first, first_errors = _spawn(adapter.close, "speaker-completion-first")
    _require_exit(first, "completion", "first-close")
    assert fault.fired
    assert first_errors == [{"type": "RuntimeError", "text": "speaker_close_failed"}], first_errors
    assert all(reference() is None for reference in probe.models)
    second, second_errors = _spawn(adapter.close, "speaker-completion-second")
    _require_exit(second, "completion", "follower")
    assert second_errors == first_errors, second_errors
    token = adapter._speaker_close_admission
    assert token.finished and token.failure == "speaker_close_failed"
    _emit("completion", "confirmed", error_name=error_name, native_exited=True)
    patch.undo()


def _nested_probe(interrupt_inner):
    fixture, patch, probe, adapter = _fixtures()
    local = _TLSFault(adapter._speaker_local)
    adapter._speaker_local = local
    condition = _WaitCondition(adapter._speaker_condition, adapter)
    adapter._speaker_condition = condition
    outer = []
    calls = []
    controls = []
    second_entered = threading.Event()
    second_release = threading.Event()
    original_encode = fixture["Encoder"].encode_batch
    original_construct = sys.modules["speechbrain.inference.speaker"].EncoderClassifier.from_hparams
    original_embedding = adapter.get_embedding

    def guard_rejects():
        try:
            adapter.close()
        except RuntimeError as error:
            assert str(error) == "speaker_close_from_worker"
        else:
            raise AssertionError("active nested lease allowed self-close")

    def construct(**options):
        lease = adapter._speaker_current()
        assert lease is not None and lease.active
        assert lease.thread is threading.current_thread()
        outer.append(lease)
        assert _active(adapter) == 1
        guard_rejects()
        return original_construct(**options)

    def encode(model, tensor):
        lease = adapter._speaker_current()
        assert lease is outer[0] and lease.active
        assert lease.thread is threading.current_thread()
        assert _active(adapter) == 1
        calls.append(lease)
        guard_rejects()
        if len(calls) == 1:
            adapter.request_close()
            if interrupt_inner:
                local.arm()
        else:
            second_entered.set()
            assert second_release.wait(5), "nested synthetic inference not released"
        return original_encode(model, tensor)

    def embedding(audio):
        try:
            return original_embedding(audio)
        except KeyboardInterrupt:
            controls.append("KeyboardInterrupt")
            assert adapter._speaker_current() is outer[0] and outer[0].active
            assert _active(adapter) == 1
            guard_rejects()
            return fixture["np"].array([1.0, 0.0, 0.0], dtype=fixture["np"].float32)

    patch.setattr(sys.modules["speechbrain.inference.speaker"].EncoderClassifier, "from_hparams", construct)
    patch.setattr(fixture["Encoder"], "encode_batch", encode)
    patch.setattr(adapter, "get_embedding", embedding)

    def enroll():
        result = adapter.create_enrollment_embedding([fixture["pcm"](), fixture["pcm"]()])
        assert fixture["np"].array_equal(result, [1.0, 0.0, 0.0])

    caller, errors = _spawn(enroll, "speaker-nested-caller")
    assert second_entered.wait(5), "second real nested embedding did not enter"
    closer, close_errors = _spawn(adapter.close, "speaker-nested-closer")
    assert condition.wait_entered.wait(1), "closer did not wait for the real outer lease"
    assert _active(adapter) == 1 and outer[0].active
    assert not closer._handle.is_done() and probe.models[0]() is adapter._model
    second_release.set()
    _require_exit(caller, "nested", "caller")
    _require_exit(closer, "nested", "close")
    assert errors == [] and close_errors == [], (errors, close_errors)
    assert controls == (["KeyboardInterrupt"] if interrupt_inner else [])
    assert len(calls) == 2 and calls[0] is calls[1] is outer[0]
    assert not outer[0].active
    _assert_retired(adapter, probe)
    _emit("nested", "confirmed", interrupt_inner=interrupt_inner, native_exited=True)
    patch.undo()


def _external_owner_probe(fail_close):
    from src.core import resource_retirement as resources

    fixture, patch, probe, adapter = _fixtures()
    owner = resources.ResourceOwner(
        adapter, owns_resource=True, error_prefix="enrollment_speaker",
        caller_guard=adapter._speaker_require_close_caller,
    )
    adapter.load()
    if fail_close:
        fault = _ExitFault(adapter._speaker_condition, lambda: _reserved(adapter))
        adapter._speaker_condition = fault
        observer, errors = _spawn(owner.close, "speaker-external-owner")
        _require_exit(observer, "external-owner", "owner-observer")
        assert fault.fired and errors
        assert owner._handle.is_done() and owner._target_exited.is_set()
        assert not owner._confirmed and owner.resource is adapter
        assert owner._caller_guard is not None
        assert owner in resources._OWNERS and owner in resources._FAILED_OWNERS
        assert probe.models[0]() is adapter._model
        token = adapter._speaker_close_admission
        assert token.finished and token.failure == "speaker_close_failed"
        repeated, repeated_errors = _spawn(owner.close, "speaker-external-repeat")
        _require_exit(repeated, "external-owner", "repeat")
        assert repeated_errors == errors
        assert owner.resource is adapter and probe.models[0]() is adapter._model
    else:
        original_encode = fixture["Encoder"].encode_batch
        guards = []

        def encode(model, tensor):
            assert not owner._operations, "adapter caller must be outside the generic operation lease"
            try:
                owner.close()
            except RuntimeError as error:
                guards.append(str(error))
            else:
                raise AssertionError("adapter worker was allowed to close external owner")
            assert owner._start_admission is None and owner._handle.ident == 0
            return original_encode(model, tensor)

        patch.setattr(fixture["Encoder"], "encode_batch", encode)
        caller, errors = _spawn(lambda: adapter.get_embedding(fixture["pcm"]()), "speaker-external-caller")
        _require_exit(caller, "external-owner", "caller")
        assert errors == [] and guards == ["speaker_close_from_worker"]
        observer, close_errors = _spawn(owner.close, "speaker-external-closer")
        _require_exit(observer, "external-owner", "owner-close")
        assert close_errors == []
        assert owner._confirmed and owner.resource is None and owner._caller_guard is None
        assert owner._handle.is_done() and owner._target_exited.is_set()
        _assert_retired(adapter, probe)
    _emit("external-owner", "confirmed", fail_close=fail_close, native_exited=True)
    patch.undo()


def _run_candidate(function, *arguments):
    script = (
        "import json, os, runpy, sys; "
        "module = runpy.run_path(sys.argv[1]); "
        "module[sys.argv[2]](*json.loads(sys.argv[3])); "
        "os._exit(0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), function, json.dumps(arguments)],
        cwd=_project_root(), capture_output=True, text=True, timeout=20,
    )
    records = [
        json.loads(line.removeprefix("SPEAKER_PROBE="))
        for line in completed.stdout.splitlines() if line.startswith("SPEAKER_PROBE=")
    ]
    assert records, f"probe failed before a receipt: {completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, f"{records[-1]}\n{completed.stderr}"
    assert records[-1]["reason"] == "confirmed" and records[-1]["native_exited"] is True


@pytest.mark.parametrize("operation", ["load", "embedding"])
@pytest.mark.parametrize("phase", ["publication", "restoration"])
def test_speaker_tls_interrupt_releases_the_actual_admission(operation, phase):
    _run_candidate("_tls_probe", operation, phase)


@pytest.mark.parametrize("interrupt_inner", [False, True])
def test_speaker_nested_operations_borrow_one_live_lease(interrupt_inner):
    _run_candidate("_nested_probe", interrupt_inner)


def test_speaker_lost_cleanup_notification_does_not_strand_waiting_close():
    _run_candidate("_notify_probe")


@pytest.mark.parametrize("error_name", ["RuntimeError", "KeyboardInterrupt"])
def test_speaker_close_completion_notification_failure_is_finite(error_name):
    _run_candidate("_completion_probe", error_name)


@pytest.mark.parametrize("fail_close", [False, True])
def test_speaker_external_owner_uses_adapter_guard_and_retains_failed_close(fail_close):
    _run_candidate("_external_owner_probe", fail_close)
