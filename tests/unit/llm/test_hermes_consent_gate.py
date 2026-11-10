"""OQ-8/M2: withdrawal kills running Hermes process groups (real inert child)."""
import os
import signal
import subprocess
import sys
import threading

import pytest

from src.core.egress import CloudConsentGate, ConsentWithdrawalSource, EgressDenied
from src.llm import hermes_reasoner as module
from src.llm.error_classifier import classify_error
from src.llm.hermes_reasoner import HermesCliReasoner, _HermesProcessOwner
from src.llm.types import ErrorKind

SCRIPT = """#!{python}
import sys, time
prompt = sys.argv[sys.argv.index("-z") + 1]
if prompt == "block":
    time.sleep(120)
print("respuesta sintetica")
"""


@pytest.fixture
def hermes(tmp_path):
    binary = tmp_path / "fake-hermes"
    binary.write_text(SCRIPT.format(python=sys.executable))
    binary.chmod(0o700)
    gate = CloudConsentGate()
    reasoner = HermesCliReasoner(binary_path=str(binary), cloud_consent=True, timeout_s=30.0, consent_gate=gate)
    return reasoner, gate


@pytest.fixture
def kills(monkeypatch):
    calls = []
    real = os.killpg

    def killpg(pgid, sig):
        calls.append((pgid, sig))
        return real(pgid, sig)
    monkeypatch.setattr(module.os, "killpg", killpg)
    return calls


def _adopted(monkeypatch):
    adopted, owners = threading.Event(), []
    real = _HermesProcessOwner.adopt

    def adopt(self, proc):
        real(self, proc)
        owners.append(self)
        adopted.set()
    monkeypatch.setattr(_HermesProcessOwner, "adopt", adopt)
    return adopted, owners


def _run_in_thread(reasoner, prompt):
    result = {}

    def target():
        try:
            result["value"] = reasoner._run(prompt)
        except BaseException as error:  # noqa: BLE001 - the oracle inspects it
            result["error"] = error
    thread = threading.Thread(target=target)
    thread.start()
    return thread, result


def test_withdraw_kills_the_tracked_process_group(hermes, kills, monkeypatch):
    reasoner, gate = hermes
    adopted, owners = _adopted(monkeypatch)
    thread, result = _run_in_thread(reasoner, "block")
    assert adopted.wait(10)
    pgid = owners[0].pgid
    assert pgid == owners[0].proc.pid
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    thread.join(20)
    assert not thread.is_alive()
    assert (pgid, signal.SIGKILL) in kills
    error = result["error"]
    # F3: the kill surfaces as consent withdrawn, from the -9 exit (not the timeout).
    assert type(error) is EgressDenied and str(error) == "cloud_consent_withdrawn"
    assert "exit=-9" in str(error.__cause__)
    assert classify_error(error) is ErrorKind.CONSENT_WITHDRAWN
    assert owners[0].retired


def test_withdrawal_between_track_and_adopt_kills_the_new_group(hermes, kills, monkeypatch):
    """F2: the subscriber runs before adopt(); the launcher must see the withdrawal."""
    reasoner, gate = hermes
    real = subprocess.Popen
    launched = []

    def popen(*args, **kwargs):
        proc = real(*args, **kwargs)
        launched.append(proc)
        gate.withdraw(ConsentWithdrawalSource.ACTOR)  # Owner tracked, proc not yet adopted.
        return proc
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$") as caught:
        reasoner._run("block")
    assert (launched[0].pid, signal.SIGKILL) in kills
    assert "exit=-9" in str(caught.value.__cause__)


def test_killed_run_surfaces_consent_withdrawn(hermes, monkeypatch):
    """F3: every failure path checks the gate first (nonzero exit here)."""
    reasoner, gate = hermes
    real = subprocess.Popen

    def popen(cmd, **kwargs):
        cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
        proc = real(cmd, **kwargs)
        proc.wait()  # Exited on its own (reaped): the withdrawal kills nothing.
        gate.withdraw(ConsentWithdrawalSource.ACTOR)
        return proc
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$") as caught:
        reasoner._run("hola")
    assert "exit=3" in str(caught.value.__cause__)


def test_late_successful_output_is_discarded(hermes, monkeypatch):
    reasoner, gate = hermes
    real = reasoner._record_usage

    def record(path, elapsed_ms):
        gate.withdraw(ConsentWithdrawalSource.ACTOR)  # After communicate, before return.
        return real(path, elapsed_ms)
    monkeypatch.setattr(reasoner, "_record_usage", record)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        reasoner._run("hola")


def test_open_gate_returns_output(hermes):
    reasoner, _gate = hermes
    text, _tokens = reasoner._run("hola")
    assert text == "respuesta sintetica"


def test_no_popen_after_withdrawal(hermes, monkeypatch):
    reasoner, gate = hermes
    calls = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: calls.append(a))
    gate.withdraw(ConsentWithdrawalSource.ACTOR)
    with pytest.raises(EgressDenied, match="^cloud_consent_withdrawn$"):
        reasoner._run("hola")
    assert calls == []


def test_no_kill_after_leader_reaped(kills, tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate()
    owner = _HermesProcessOwner(str(tmp_path / "usage.json"))
    try:
        owner.adopt(proc)
        assert proc.returncode == 0
        owner.kill_group()
        assert kills == []
    finally:
        owner.retire(terminate=False)


def test_consented_reasoner_requires_a_gate(tmp_path):
    """N4: a consented (always CLOUD) Hermes reasoner cannot be built without the gate."""
    with pytest.raises(ValueError, match="^consent_gate_required$"):
        HermesCliReasoner(binary_path="hermes", cloud_consent=True)
    assert HermesCliReasoner(binary_path="hermes", cloud_consent=False).policy.consent_gate is None
