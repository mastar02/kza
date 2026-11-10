"""OQ-8 D1: admin-only cloud-consent withdrawal endpoint (no voice/persona path)."""
import asyncio
from contextlib import AsyncExitStack, ExitStack
from pathlib import Path
import re

import pytest
from fastapi import FastAPI

from src.core.egress import CloudConsentGate
from src.core.runtime_admission import RuntimeAdmission
from src.core.state_configuration import ConfigurationChanges, ConsentWithdrawalOutcome, MarkerRecordState
from src.dashboard.admin import register_admin_routes
from src.dashboard.authority import DashboardRole
from tests.mocks.dashboard_sessions import session_client

TOKEN = "test-secret-token-1234567890"
RAW = b"reasoner:\n  cloud:\n    consent: true\nother: untouched\n"
SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", TOKEN)


def _bearer(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def consent_app(tmp_path):
    with ExitStack() as outer:
        def create(*, changes=True, gate=True):
            state = tmp_path / "state"
            (state / "config").mkdir(parents=True, exist_ok=True)
            (state / "config" / "settings.yaml").write_bytes(RAW)
            built = {}

            def app_factory(authority):
                admission = RuntimeAdmission({"run_id": "synthetic-control-leaf"}, {"state_root": str(state)},
                                             str(state / "data" / "privacy.db"), config_digest="0" * 64)
                admission.revalidate = lambda: None
                consent_gate = CloudConsentGate()
                stack = AsyncExitStack()
                owner = ConfigurationChanges(authority, admission, consent_gate)
                owner.install_on(stack)
                app = FastAPI()
                app.state.dashboard_authority = authority
                register_admin_routes(app, configuration_changes=owner if changes else None,
                                      consent_gate=consent_gate if gate else None)
                built.update(owner=owner, gate=consent_gate, authority=authority, stack=stack)

                async def cleanup():
                    try:
                        await stack.aclose()
                    except RuntimeError as error:  # A pending write without a graph cannot commit.
                        assert str(error) == "configuration_graph_close_unconfirmed"
                return app, cleanup
            client = outer.enter_context(session_client(app_factory, tmp_path / f"auth-{len(built)}.db"))
            return client, built
        yield create


def _issue(client, built, role):
    authority = built["authority"]
    root = authority.resolve(f"Bearer {TOKEN}")
    issued = client.portal.call(lambda: authority.issue_credential(
        root, subject=f"synthetic-{role.value}", role=role, expires_at=authority._now() + 600))
    return issued


def test_no_session_is_401_and_gate_untouched(consent_app):
    client, built = consent_app()
    assert client.post("/api/admin/cloud-consent/withdraw").status_code == 401
    assert client.get("/api/admin/cloud-consent").status_code == 401
    assert built["gate"].withdrawn is False and built["owner"].pending is None


def test_non_admin_is_403_and_gate_untouched(consent_app):
    client, built = consent_app()
    issued = _issue(client, built, DashboardRole.HOME)
    for response in (client.post("/api/admin/cloud-consent/withdraw", headers=_bearer(issued.token)),
                     client.get("/api/admin/cloud-consent", headers=_bearer(issued.token))):
        assert response.status_code == 403
    assert built["gate"].withdrawn is False and built["owner"].pending is None


def test_admin_post_closes_gate_and_returns_the_outcome(consent_app):
    client, built = consent_app()
    response = client.post("/api/admin/cloud-consent/withdraw", headers=_bearer())
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["gate_closed"] is True and body["marker"] == "unmanaged"
    assert body["pending_write"] is True and body["failure"] is None
    assert built["gate"].withdrawn is True and built["owner"].pending is not None
    assert "nube" in body["message"]
    assert "subject" not in response.text and TOKEN not in response.text


def test_get_status_reports_the_pending_write_cause(consent_app):
    client, built = consent_app()
    issued = _issue(client, built, DashboardRole.OPERATOR)
    assert client.post("/api/admin/cloud-consent/withdraw", headers=_bearer(issued.token)).status_code == 200
    status = client.get("/api/admin/cloud-consent", headers=_bearer())
    assert status.status_code == 200 and status.headers["cache-control"] == "no-store"
    body = status.json()
    assert body["gate_withdrawn"] is True and body["gate_source"] == "actor"
    assert body["pending_write"] is True and body["pending_write_cause"] is None
    authority = built["authority"]
    root = authority.resolve(f"Bearer {TOKEN}")
    assert client.portal.call(lambda: authority.revoke_credential(root, issued.identity))
    body = client.get("/api/admin/cloud-consent", headers=_bearer()).json()
    assert body["pending_write_cause"] == "actor_lease_expired"
    assert set(body) >= {"boot_decision_basis", "marker", "registration_failure", "last_settled"}


def test_post_without_configuration_owner_closes_gate_unmanaged(consent_app):
    client, built = consent_app(changes=False)
    response = client.post("/api/admin/cloud-consent/withdraw", headers=_bearer())
    assert response.status_code == 200
    body = response.json()
    assert body["gate_closed"] is True and body["marker"] == "unmanaged"
    assert body["failure"] == "configuration_changes_unavailable" and body["pending_write"] is False
    assert built["gate"].withdrawn is True
    status = client.get("/api/admin/cloud-consent", headers=_bearer()).json()
    assert status["gate_withdrawn"] is True and status["marker"] is None


def test_post_without_owner_or_gate_is_503(consent_app):
    client, built = consent_app(changes=False, gate=False)
    assert client.post("/api/admin/cloud-consent/withdraw", headers=_bearer()).status_code == 503
    assert built["gate"].withdrawn is False


def test_closed_owner_is_409_and_gate_untouched(consent_app):
    client, built = consent_app()
    built["owner"]._closed = True
    assert client.post("/api/admin/cloud-consent/withdraw", headers=_bearer()).status_code == 409
    assert built["gate"].withdrawn is False
    built["owner"]._closed = False


def test_registration_failed_message_states_degraded_run(consent_app, monkeypatch):
    client, built = consent_app()

    async def failed(actor):
        return ConsentWithdrawalOutcome(True, MarkerRecordState.REGISTRATION_FAILED, None,
                                        "output_registration_destination")
    monkeypatch.setattr(built["owner"], "withdraw_cloud_consent", failed)
    body = client.post("/api/admin/cloud-consent/withdraw", headers=_bearer()).json()
    assert body["marker"] == "registration_failed"
    assert "degradado" in body["message"] and "bloqueado" in body["message"]


def test_blocking_step_runs_off_loop(consent_app, monkeypatch):
    client, built = consent_app()
    seen = []
    real = ConfigurationChanges._queue_pending

    def spy(self, actor):
        try:
            asyncio.get_running_loop()
            seen.append("loop")
        except RuntimeError:
            seen.append("worker")
        return real(self, actor)
    monkeypatch.setattr(ConfigurationChanges, "_queue_pending", spy)
    assert client.post("/api/admin/cloud-consent/withdraw", headers=_bearer()).status_code == 200
    assert seen == ["worker"]


def test_withdrawal_is_reachable_only_from_the_admin_dashboard():
    """Voice, persona, speaker-ID, NLU and LLM tools cannot reach the withdrawal."""
    callers, withdrawers = [], []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(SRC).as_posix()
        if re.search(r"\bwithdraw_cloud_consent\b", text) and relative != "core/state_configuration.py":
            callers.append(relative)
        if re.search(r"ConsentWithdrawalSource\.ACTOR\b", text):
            withdrawers.append(relative)
    assert callers == ["dashboard/admin.py"]
    assert sorted(withdrawers) == ["core/state_configuration.py", "dashboard/admin.py"]
