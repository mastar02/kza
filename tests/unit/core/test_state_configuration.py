"""Pending/denial contracts; successful Graph retirement is integration-only."""
from contextlib import AsyncExitStack
from dataclasses import FrozenInstanceError
import pytest

from src.core.egress import CloudConsentGate
from src.core.runtime_admission import RuntimeAdmission
from src.core.state_configuration import ConfigurationChanges, MarkerRecordState
from src.dashboard.authority import DashboardAuthority, DashboardRole


RAW = b'reasoner:\n  cloud:\n    consent: true\n  api_key_env: "${SYNTHETIC_KEY}"\nother: untouched\n'


@pytest.fixture
async def configured(tmp_path, monkeypatch):
    """Real auth SQLite; admission verification is an isolated control leaf.

    These tests never attach a fake Graph, confirm SDK cleanup or produce a
    successful write/closed admission receipt. RuntimeAdmission has its own
    real controller/integration coverage.
    """
    container = tmp_path / "state"
    config = container / "config" / "settings.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(RAW)
    authority = DashboardAuthority(tmp_path / "auth.db", expected_token=lambda: "synthetic-root")
    await authority.initialize()
    lease = authority.resolve("Bearer synthetic-root")
    assert lease is not None and lease.current(frozenset({"admin:manage"}))
    admission = RuntimeAdmission({"run_id": "synthetic-control-leaf"},
        {"state_root": str(container)}, str(container / "data" / "graph" / "privacy.db"),
        config_digest="0" * 64)
    checks = []
    monkeypatch.setattr(admission, "revalidate", lambda: checks.append("checked"))
    try:
        yield authority, lease, admission, config, checks
    finally:
        await authority.aclose()


@pytest.mark.asyncio
async def test_queue_returns_only_frozen_pending_and_prior_error_preserves_raw_yaml(configured):
    authority, actor, admission, path, checks = configured
    gate = CloudConsentGate()
    owner = ConfigurationChanges(authority, admission, gate)
    original = ValueError("original-owned-exit")
    with pytest.raises(ValueError) as caught:
        async with AsyncExitStack() as stack:
            owner.install_on(stack)
            outcome = await owner.withdraw_cloud_consent(actor)
            # Unmanaged (synthetic admission): no journal, the in-run gate still closes.
            assert outcome.marker is MarkerRecordState.UNMANAGED and outcome.failure is None
            assert outcome.gate_closed is True and gate.withdrawn is True
            pending = outcome.pending
            assert pending.state == "pending" and pending.next_generation_only is True
            assert pending.previous_consent is True and owner.receipt is None
            again = await owner.withdraw_cloud_consent(actor)
            assert again.pending is pending and again.marker is MarkerRecordState.ALREADY_RECORDED
            with pytest.raises(FrozenInstanceError):
                pending.state = "committed"
            assert path.read_bytes() == RAW and checks
            raise original
    assert caught.value is original and path.read_bytes() == RAW and owner.receipt is None


@pytest.mark.asyncio
async def test_partial_factory_cannot_commit_without_actual_graph(configured):
    authority, actor, admission, path, _checks = configured
    owner = ConfigurationChanges(authority, admission, CloudConsentGate())
    with pytest.raises(RuntimeError, match="configuration_graph_close_unconfirmed"):
        async with AsyncExitStack() as stack:
            owner.install_on(stack)
            assert (await owner.withdraw_cloud_consent(actor)).pending is not None
    assert path.read_bytes() == RAW and owner.receipt is None


@pytest.mark.asyncio
async def test_foreign_actual_authority_cannot_queue(configured, tmp_path):
    authority, _actor, admission, path, _checks = configured
    foreign = DashboardAuthority(tmp_path / "foreign.db", expected_token=lambda: "foreign-root")
    await foreign.initialize()
    try:
        actor = foreign.resolve("Bearer foreign-root")
        gate = CloudConsentGate()
        async with AsyncExitStack() as stack:
            owner = ConfigurationChanges(authority, admission, gate)
            owner.install_on(stack)
            with pytest.raises(PermissionError, match="configuration_actor_denied"):
                await owner.withdraw_cloud_consent(actor)
        assert path.read_bytes() == RAW and owner.pending is owner.receipt is None
        assert gate.withdrawn is False
    finally:
        await foreign.aclose()


@pytest.mark.asyncio
async def test_revoked_original_credential_cannot_replace_its_pending_request(configured):
    authority, root, admission, path, _checks = configured
    issued = await authority.issue_credential(root, subject="synthetic-configuration-operator",
        role=DashboardRole.OPERATOR, expires_at=authority._now() + 600)
    actor = authority.resolve(issued.token)
    owner = ConfigurationChanges(authority, admission, CloudConsentGate())
    with pytest.raises(ValueError, match="preserve-pending"):
        async with AsyncExitStack() as stack:
            owner.install_on(stack)
            pending = (await owner.withdraw_cloud_consent(actor)).pending
            assert await authority.revoke_credential(root, issued.identity)
            assert actor.current() is False
            with pytest.raises(PermissionError, match="configuration_actor_denied"):
                await owner.withdraw_cloud_consent(actor)
            assert owner.pending is pending and owner.receipt is None
            raise ValueError("preserve-pending")
    assert path.read_bytes() == RAW


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["file", "directory"])
async def test_symbolic_configuration_location_is_rejected_before_queue(configured, tmp_path, component):
    authority, actor, admission, path, _checks = configured
    outside = tmp_path / "outside"
    outside.mkdir()
    copied = outside / "settings.yaml"
    copied.write_bytes(RAW)
    path.unlink()
    if component == "file":
        path.symlink_to(copied)
    else:
        path.parent.rmdir()
        path.parent.symlink_to(outside, target_is_directory=True)
    gate = CloudConsentGate()
    async with AsyncExitStack() as stack:
        owner = ConfigurationChanges(authority, admission, gate)
        owner.install_on(stack)
        outcome = await owner.withdraw_cloud_consent(actor)
        # The step-4 refusal no longer raises: the gate is closed and the marker kept first.
        assert outcome.failure == "configuration_path_invalid" and outcome.pending is None
        assert outcome.gate_closed is True and gate.withdrawn is True
        assert outcome.marker is MarkerRecordState.UNMANAGED
        assert owner.pending is owner.receipt is None and copied.read_bytes() == RAW


@pytest.mark.asyncio
async def test_changed_yaml_cannot_refresh_the_original_pending_sha(configured):
    authority, actor, admission, path, _checks = configured
    owner = ConfigurationChanges(authority, admission, CloudConsentGate())
    changed = RAW.replace(b"untouched", b"new-external-edit")
    with pytest.raises(ValueError, match="preserve-cas"):
        async with AsyncExitStack() as stack:
            owner.install_on(stack)
            pending = (await owner.withdraw_cloud_consent(actor)).pending
            path.write_bytes(changed)
            again = await owner.withdraw_cloud_consent(actor)
            assert again.failure == "configuration_file_changed" and again.pending is pending
            assert owner.pending is pending and owner.receipt is None
            raise ValueError("preserve-cas")
    assert path.read_bytes() == changed


@pytest.mark.asyncio
async def test_nonboolean_consent_cannot_be_treated_as_a_grant(configured):
    authority, actor, admission, path, _checks = configured
    malformed = RAW.replace(b"consent: true", b'consent: "true"')
    path.write_bytes(malformed)
    gate = CloudConsentGate()
    async with AsyncExitStack() as stack:
        owner = ConfigurationChanges(authority, admission, gate)
        owner.install_on(stack)
        outcome = await owner.withdraw_cloud_consent(actor)
        assert outcome.failure == "configuration_consent_invalid" and outcome.pending is None
        assert outcome.gate_closed is True and gate.withdrawn is True
    assert path.read_bytes() == malformed and owner.pending is owner.receipt is None


@pytest.mark.asyncio
async def test_unmanaged_owner_never_creates_a_configuration_path(configured):
    authority, actor, _admission, path, _checks = configured
    gate = CloudConsentGate()
    async with AsyncExitStack() as stack:
        owner = ConfigurationChanges(authority, None, gate)
        owner.install_on(stack)
        outcome = await owner.withdraw_cloud_consent(actor)
        assert outcome.failure == "configuration_admission_required" and outcome.pending is None
        assert outcome.marker is MarkerRecordState.UNMANAGED and gate.withdrawn is True
    assert path.read_bytes() == RAW and owner.pending is owner.receipt is None


@pytest.mark.asyncio
async def test_no_queued_change_or_foreign_graph_cannot_make_a_commit(configured):
    authority, _actor, admission, path, _checks = configured
    async with AsyncExitStack() as stack:
        owner = ConfigurationChanges(authority, admission)
        owner.install_on(stack)
        with pytest.raises(RuntimeError, match="configuration_public_graph_required"):
            owner.attach_graph(object())
        with pytest.raises(RuntimeError, match="configuration_owner_already_installed"):
            owner.install_on(stack)
    assert path.read_bytes() == RAW and owner.pending is owner.receipt is None
