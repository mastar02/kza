"""G01: an older slot application's own historical worker vs its own writer probe.

The copied-slot rollback rehearsal runs the earliest managed-capable revision
(80b47f5). Its own admission probes the delegated source state with the original
non-blocking flock while its own historical diagnostic maintenance can hold that
state's ``metrics_policy_v1.jsonl.sqlite3.lock``. The rehearsal harness backports
the 431e44e mutex; these tests cover both the original flock probe and the binding's
SQLite authority capture.
"""
from contextlib import closing
import fcntl
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from scripts import stage_runtime
from src.obs.diagnostic_store import DiagnosticStore
from tests.mocks import managed_slot_graph
from tests.mocks.managed_slot_graph import serialize_older_history

OLD_STATE_LOCKS = ("privacy.db.lock", "dashboard-auth.db.lease", "metrics_policy_v1.jsonl.sqlite3.lock")


def _original_probe(state: Path) -> None:
    """80b47f5's _assert_writers_closed: the same flock probe with no shared mutex."""
    descriptors = []
    try:
        for path in sorted(state.rglob("*")):
            if path.is_file() and path.name.endswith((".lock", ".lease")):
                descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                descriptors.append(descriptor)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("runtime_admission_writer_active") from None
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


class _HistoricalDiagnostics:
    """80b47f5's DiagnosticServices shape: _invoke_history opens, prunes and closes."""

    def __init__(self):
        self.held, self.release = threading.Event(), threading.Event()

    def _invoke_history(self, path, conversation):
        descriptor = os.open(str(path) + ".sqlite3.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.held.set()
            if not self.release.wait(30):
                raise AssertionError("history_release_missing")
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return 0
        finally:
            os.close(descriptor)


def _older_application(state):
    stage = SimpleNamespace(_assert_writers_closed=_original_probe,
                            _binding_snapshot=stage_runtime._binding_snapshot)
    admission = SimpleNamespace(_stage=lambda: stage)
    diagnostics = SimpleNamespace(DiagnosticServices=type("DiagnosticServices", (_HistoricalDiagnostics,), {}))
    for name in OLD_STATE_LOCKS:
        (state / name).touch(mode=0o600)
    return stage, admission, diagnostics


def _run(function, *args):
    outcome = {}

    def target():
        try:
            outcome["value"] = function(*args)
        except BaseException as error:  # noqa: BLE001 - the thread reports any outcome
            outcome["error"] = error
    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


def test_final_probe_waits_for_the_applications_own_historical_catalog_close(tmp_path, monkeypatch):
    state = tmp_path / "old/state/data/graph"
    state.mkdir(parents=True)
    stage, admission, diagnostics = _older_application(state)
    assert serialize_older_history(monkeypatch, admission, diagnostics) is True
    services = diagnostics.DiagnosticServices()
    history, history_outcome = _run(services._invoke_history, state / "metrics_policy_v1.jsonl", None)
    assert services.held.wait(30)
    probe, probe_outcome = _run(stage._assert_writers_closed, state)
    # Unserialized, the probe reaches the held flock at once (the G01 refusal).
    probe.join(0.5)
    # Serialized, it is still waiting for the worker's shared lock: no outcome yet.
    assert probe.is_alive() and probe_outcome == {}, probe_outcome
    services.release.set()
    history.join(30)
    probe.join(30)
    assert not history.is_alive() and not probe.is_alive()
    assert history_outcome == {"value": 0}
    assert probe_outcome == {"value": None}, probe_outcome


def test_binding_snapshot_excludes_transient_staged_publication(tmp_path, monkeypatch):
    state = tmp_path / "state"
    config = state / "config"
    config.mkdir(parents=True)
    privacy = config / "privacy.yaml"
    privacy.write_text("consent: true\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "release.json").write_text("{}")
    manifest = {"git_commit": "80b47f55", "python_magic": "test"}
    metrics = state / "metrics_policy_v1.jsonl"

    class HistoricalDiagnostics:
        def _invoke_history(self, path, conversation):
            with closing(DiagnosticStore(path, clock=lambda: 0, retention_days=1,
                                         rotate_bytes=1024, project=dict)) as store:
                store.prune()

    services = HistoricalDiagnostics()
    services._invoke_history(metrics, None)
    metrics.unlink()  # A missing projection forces the next prune to publish.
    monkeypatch.setattr(stage_runtime, "_checked_slot", lambda root, name: (runtime, state, manifest))
    monkeypatch.setattr(stage_runtime, "_assert_writers_closed", _original_probe)
    baseline = stage_runtime._binding_snapshot(tmp_path, "old")

    probe_passed = threading.Event()
    writer_blocked_or_staged = threading.Event()
    authority_captured = threading.Event()
    lock = threading.RLock()
    capture = stage_runtime.capture_state_authority
    replace = os.replace

    class ObservedGate:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                writer_blocked_or_staged.set()
                lock.acquire()

        def __exit__(self, *_exc):
            lock.release()

    def capture_after_writer_attempt(state):
        # The real binding snapshot calls this after the writer probe succeeds.
        probe_passed.set()
        assert writer_blocked_or_staged.wait(30)
        try:
            return capture(state)
        finally:
            authority_captured.set()

    def pause_at_publish(source, target):
        if target == metrics:
            # _publish has committed the INSERT and has yet to DELETE the row.
            writer_blocked_or_staged.set()
            assert authority_captured.wait(30)
        return replace(source, target)

    def run_history():
        assert probe_passed.wait(30)
        services._invoke_history(metrics, None)

    admission = SimpleNamespace(_stage=lambda: stage_runtime)
    diagnostics = SimpleNamespace(DiagnosticServices=HistoricalDiagnostics)
    tables = baseline["authority"]["projection"]["tables"]
    assert tables[metrics.name + ".sqlite3"]["staged"]["row_hashes"] == []
    with monkeypatch.context() as patch:
        patch.setattr(managed_slot_graph, "threading", SimpleNamespace(RLock=ObservedGate))
        patch.setattr(stage_runtime, "capture_state_authority", capture_after_writer_attempt)
        patch.setattr(os, "replace", pause_at_publish)
        assert serialize_older_history(patch, admission, diagnostics) is True
        history, history_outcome = _run(run_history)
        snapshot, snapshot_outcome = _run(stage_runtime._binding_snapshot, tmp_path, "old")
        snapshot.join(30)
        authority_captured.set()
        history.join(30)
        assert not snapshot.is_alive() and not history.is_alive(), (snapshot_outcome, history_outcome)

    assert history_outcome == {"value": None}, history_outcome
    assert snapshot_outcome == {"value": baseline}, snapshot_outcome
    assert metrics.is_file()
    assert stage_runtime._binding_snapshot(tmp_path, "old") == baseline
    privacy.write_text("consent: false\n")
    changed = stage_runtime._binding_snapshot(tmp_path, "old")
    assert changed["authority"]["digest"] != baseline["authority"]["digest"]


def test_backport_still_refuses_a_real_writer_outside_the_historical_worker(tmp_path, monkeypatch):
    state = tmp_path / "old/state/data/graph"
    state.mkdir(parents=True)
    stage, admission, diagnostics = _older_application(state)
    assert serialize_older_history(monkeypatch, admission, diagnostics) is True
    for name in OLD_STATE_LOCKS:
        descriptor = os.open(state / name, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
                stage._assert_writers_closed(state)
        finally:
            os.close(descriptor)
    stage._assert_writers_closed(state)


def test_application_with_the_gate_is_left_unchanged(monkeypatch):
    from src.obs import diagnostic_erasure

    def untouched():
        raise AssertionError("an application with the gate must not be patched")
    admission = SimpleNamespace(_stage=untouched)
    original = diagnostic_erasure.DiagnosticServices._invoke_history
    assert serialize_older_history(monkeypatch, admission, diagnostic_erasure) is False
    assert diagnostic_erasure.DiagnosticServices._invoke_history is original


def test_undo_restores_the_original_probe_snapshot_and_worker(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    stage, admission, diagnostics = _older_application(state)
    snapshot = stage._binding_snapshot
    worker = diagnostics.DiagnosticServices._invoke_history
    patch = pytest.MonkeyPatch()
    assert serialize_older_history(patch, admission, diagnostics) is True
    assert stage._assert_writers_closed is not _original_probe
    assert stage._binding_snapshot is not snapshot
    patch.undo()
    assert stage._assert_writers_closed is _original_probe
    assert stage._binding_snapshot is snapshot
    assert diagnostics.DiagnosticServices._invoke_history is worker


def test_current_probe_keeps_the_original_refusal_code(tmp_path):
    """The backport wraps the probe; it never changes the code a real writer gets."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "metrics_policy_v1.jsonl.sqlite3.lock").touch(mode=0o600)
    descriptor = os.open(state / "metrics_policy_v1.jsonl.sqlite3.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
            stage_runtime._assert_writers_closed(state)
        with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
            _original_probe(state)
    finally:
        os.close(descriptor)
