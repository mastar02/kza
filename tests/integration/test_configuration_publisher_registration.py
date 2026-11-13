"""Managed configuration publisher: ConfigurationChanges under the V4 journal.

As in the account and family registration tests, admission validation and the
verified-instance registry are synthetic seams. The DashboardAuthority lease,
PrivacyServices close, ApplicationGraph, ConfigurationChanges owner, its writer
thread and the owned AsyncExitStack order are real. Compiled ancestry
(native-03 advance-N) is the separate real-binding oracle.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from scripts import stage_runtime as stage
from src.application import ApplicationGraph
from src.core import runtime_admission
from src.core import state_configuration
from src.core.egress import CloudConsentGate
from src.core.state_configuration import ConfigurationChanges, MarkerRecordState, PendingConsentWithdrawal
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import build_privacy_services
from src.users.user_manager import UserManager
from tests.integration.test_account_publisher_registration import _voice_links

RAW = b'reasoner:\n  cloud:\n    consent: true\n  api_key_env: "${SYNTHETIC_KEY}"\nother: untouched\n'
TOKEN = "synthetic-configuration-root"


def v4():
    from src.memory import output_registration_v4
    return output_registration_v4


def own_temporaries(monkeypatch, *, reuse: bool) -> list:
    """Record each own mkstemp name; with reuse, report ext4 inode-number reuse.

    ext4 hands a freed inode number to the next allocation at once, so a
    substitute written at the own name after an unlink can carry the own
    (st_dev, st_ino). With reuse=True every platform reports exactly that for
    any other file at an own name, so cleanup can only tell them apart by the
    pinned own inode, never by its number.
    """
    created, owned = [], {}
    real_mkstemp, real_lstat = state_configuration.tempfile.mkstemp, os.lstat

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        info = os.fstat(descriptor)
        owned[name] = (info.st_dev, info.st_ino)
        created.append(name)
        return descriptor, name

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        own = owned.get(os.fspath(path))
        if own is None or (info.st_dev, info.st_ino) == own:
            return info
        fields = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        fields.update(st_dev=own[0], st_ino=own[1])
        return SimpleNamespace(**fields)

    monkeypatch.setattr(state_configuration.tempfile, "mkstemp", mkstemp)
    if reuse:
        monkeypatch.setattr(os, "lstat", lstat)
    return created


@asynccontextmanager
async def configuration_graph(tmp_path, monkeypatch, *, dashboard=True, attach=True, accounts=False,
                              raw=RAW, binding_sha=None, privacy=True, poison_before_configuration=None):
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    data = state / "data"
    data.mkdir(parents=True)
    settings = state / "config" / "settings.yaml"
    settings.parent.mkdir()
    settings.write_bytes(raw)
    config = {
        "memory": {"privacy_db_path": str(data / "privacy.db")},
        "dashboard": {"enabled": dashboard, "auth": {"db_path": str(data / "authority.db")}},
        "routines": {"db_path": str(data / "routines.db")},
        "analytics": {"events_db": str(data / "events.db"), "suggestions_path": str(data / "suggestions.json")},
        "speaker_id": {"users_path": str(data / "users.json")},
        **{name: {"db_path": str(data / (name + ".db"))} for name in ("lists", "reminders")},
    }
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    projection = binding_sha or hashlib.sha256(raw).hexdigest()
    binding = {"state_root": str(state),
               "authority": {"projection": {"configuration": {"config/settings.yaml": projection}}}}
    admission = runtime_admission.RuntimeAdmission(ticket, binding, config["memory"]["privacy_db_path"],
                                                   runtime_admission._config_digest(config))
    checks = []
    monkeypatch.setattr(admission, "revalidate", lambda: checks.append("full"))
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: checks.append("live"))
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    owner = await admission.begin_destination_registration(config)
    g = SimpleNamespace(owner=owner, stack=stack, config=config, state=state, data=data, settings=settings,
                        admission=admission, checks=checks, closed=False, error=None, changes=None, observed=[],
                        authority=None, actor=None, privacy=None, users=None, api=None, gate=CloudConsentGate())
    try:
        if dashboard:
            g.authority = DashboardAuthority(config["dashboard"]["auth"]["db_path"], expected_token=lambda: TOKEN)
            stack.push_async_callback(g.authority.aclose)
            await registration.attach_output(g.authority, "authority_sessions")
            stack.push_async_exit(g_observer(g))
            g.changes = ConfigurationChanges(g.authority, admission, g.gate)
            g.changes.install_on(stack)
            if poison_before_configuration:
                # An unrelated producer poisons the epoch after privacy closed,
                # just before the configuration owner's exit runs.
                stack.push_async_callback(poison, owner, poison_before_configuration)
            if attach:
                await owner.attach_configuration(g.changes, g.authority)
            await g.authority.initialize()
            g.actor = g.authority.resolve("Bearer " + TOKEN)
        if accounts:
            g.users = UserManager(config["speaker_id"]["users_path"], voice_registry=_voice_links())
            if dashboard:
                g.api = DashboardAPI(authority=g.authority, user_manager=g.users)
                stack.push_async_callback(g.api.stop)
                await owner.attach_accounts(g.users, g.api)
            else:
                await owner.accounts_disabled(g.users)
        if privacy:
            g.privacy = await build_privacy_services(config, shutdown=stack)
        yield g
    finally:
        await shutdown(g)


async def poison(owner, cause: str) -> None:
    if cause == "failed":
        owner._failed = True
    else:  # The installed admission is no longer a verified one.
        runtime_admission._verified_admissions[owner.admission] = False


def g_observer(g):
    """An exit directly below the configuration owner: records what that exit raised."""
    async def observe(exc_type, exc, traceback) -> bool:
        g.observed.append(exc)
        return False
    return observe


def config_error(g) -> BaseException | None:
    """Exactly what ConfigurationChanges' owned exit raised (later exits may add their own)."""
    observed, = g.observed
    return observed


async def shutdown(g, exc: BaseException | None = None) -> BaseException | None:
    """Run the real owned stack once; return what it raised."""
    if g.closed:
        return g.error
    g.closed = True
    try:
        await g.stack.__aexit__(None if exc is None else type(exc), exc, None)
    except BaseException as error:  # noqa: BLE001 - the oracle inspects every outcome
        g.error = error
    return g.error


def application_graph(g, **overrides) -> ApplicationGraph:
    """A real ApplicationGraph; fields the pin and attach_graph never read hold sentinels."""
    sentinel = object()
    values = dict(config=g.config, pipeline=sentinel, ha_client=sentinel, home_actions=sentinel,
                  privacy=g.privacy, voice_registry=sentinel, voice_services=sentinel, home_services=sentinel,
                  home_capabilities=sentinel, list_manager=sentinel, reminder_manager=sentinel,
                  routine_scheduler=None, dashboard=g.api, models=sentinel, warmup_models=sentinel,
                  dashboard_authority=g.authority, configuration_changes=g.changes)
    values.update(overrides)
    return ApplicationGraph(**values)


def events(owner):
    return [event for event in (base._read(p)[0] for p in sorted(owner._directory.glob("[0-9]*.json")))
            if event["kind"] == "configuration_publisher"]


def phases(owner):
    return [event["phase"] for event in events(owner)]


def identity(path):
    info = os.lstat(path)
    return {"device": info.st_dev, "inode": info.st_ino}


def temporaries(g):
    return sorted(p.name for p in g.settings.parent.glob(".consent-*"))


def errors(error):
    """The raised exception, its exception-group members and its context chain."""
    pending, seen = [error], []
    while pending:
        item = pending.pop()
        if item is None or any(item is other for other in seen):
            continue
        seen.append(item)
        pending.extend(getattr(item, "exceptions", ()))
        pending.append(item.__context__)
    return seen


def messages(error):
    return [str(item) for item in errors(error)]


def drained(owner):
    return v4().chain(owner._directory)[2]["configuration"]


def assert_poisoned(g, *, last="opening", closing="output_registration_epoch_incomplete"):
    assert g.owner._failed
    assert g.owner._closing.done() and g.owner._closing.exception() is not None
    if closing is not None:
        assert str(g.owner._closing.exception()) == closing
    assert "drained" not in phases(g.owner) and phases(g.owner)[-1] == last


async def queue(g, actor=None):
    """OQ-8: gate, durable marker, then the queued shutdown write (one call)."""
    outcome = await g.changes.withdraw_cloud_consent(actor or g.actor)
    assert outcome.gate_closed is True and g.gate.withdrawn is True and outcome.failure is None
    assert outcome.marker in (MarkerRecordState.DURABLE, MarkerRecordState.ALREADY_RECORDED)
    pending = outcome.pending
    assert pending.state == "pending" and g.changes.receipt is None
    return pending


def graph_ready(g):
    g.changes.attach_graph(application_graph(g))


# ----------------------------------------------------------------------
# T05: attachment before any write.
# ----------------------------------------------------------------------

async def test_attach_records_intent_and_attached_before_any_write(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        owner = g.owner
        before = (g.settings.read_bytes(), identity(g.settings))
        assert phases(owner) == ["intent", "attached"] and owner._configuration.phase == "attached"
        plan = events(owner)[0]["plan"]
        info = os.lstat(g.settings)
        assert plan == {"kind": "atomic_yaml", "producer": "configuration", "role": "publisher",
            "representation": "kza-settings-yaml-raw-v1", "path": str(g.settings), "parent": str(g.settings.parent),
            "state_identity": identity(g.state), "parent_identity": identity(g.settings.parent),
            "initial_identity": identity(g.settings), "initial_mode": stat.S_IMODE(info.st_mode),
            "initial_sha256": hashlib.sha256(RAW).hexdigest(), "before_first_publish": True,
            "write_window": "after_graph_close",
            "temporary_scope": {"directory": str(g.settings.parent), "prefix": ".consent-", "allocator": "mkstemp"},
            "classification": "protected_policy_input"}
        assert g.changes._output_coordinator is owner
        assert (g.settings.read_bytes(), identity(g.settings)) == before and temporaries(g) == []
        journal = b"".join(p.read_bytes() for p in owner._directory.iterdir())
        assert b"untouched" not in journal and b"SYNTHETIC_KEY" not in journal
        await shutdown(g)
        assert g.error is None and phases(owner) == ["intent", "attached", "drained"]


@pytest.mark.parametrize("shape", ["owner_type", "foreign_authority", "unattached_authority", "other_admission",
                                   "closed_owner", "pending_owner", "duplicate", "symlinked_settings",
                                   "missing_settings"])
async def test_attach_rejects_wrong_owner_shapes_without_creating_files(tmp_path, monkeypatch, shape):
    async with configuration_graph(tmp_path, monkeypatch, attach=shape == "duplicate") as g:
        owner, changes, authority = g.owner, g.changes, g.authority
        journal = sorted(p.name for p in owner._directory.iterdir())
        if shape == "owner_type":
            changes = SimpleNamespace(**vars(changes))
        elif shape == "foreign_authority":
            authority = DashboardAuthority(g.data / "foreign.db")
            changes = ConfigurationChanges(authority, g.admission)
            changes.install_on(AsyncExitStack())
        elif shape == "unattached_authority":
            authority = DashboardAuthority(g.data / "foreign.db")
            changes._authority = authority
        elif shape == "other_admission":
            other = runtime_admission.RuntimeAdmission(dict(g.admission._ticket), dict(g.admission._binding),
                                                       g.admission._privacy_path, g.admission._config_digest)
            changes = ConfigurationChanges(authority, other)
            changes.install_on(AsyncExitStack())
        elif shape == "closed_owner":
            changes._closed = True
        elif shape == "pending_owner":
            # OQ-8: the public API can no longer queue without an attached,
            # durable marker; the attach guard is exercised on the state seam.
            changes._pending = PendingConsentWithdrawal("0" * 64, True, "synthetic")
        elif shape == "symlinked_settings":
            outside = tmp_path / "outside.yaml"
            outside.write_bytes(RAW)
            g.settings.unlink()
            g.settings.symlink_to(outside)
        elif shape == "missing_settings":
            g.settings.unlink()
        listing = sorted(p.name for p in g.settings.parent.iterdir())
        with pytest.raises(RuntimeError) as error:
            await owner.attach_configuration(changes, authority)
        expected = {"symlinked_settings": "output_registration_destination",
                    "missing_settings": "output_registration_destination"}.get(shape, "output_registration_configuration_owner")
        assert str(error.value) == expected and owner._failed
        assert sorted(p.name for p in g.settings.parent.iterdir()) == listing
        if shape != "duplicate":
            assert sorted(p.name for p in owner._directory.iterdir()) == journal and phases(owner) == []
        else:
            assert phases(owner) == ["intent", "attached"]
        if shape == "closed_owner":
            changes._closed = False


async def test_attach_rejects_settings_that_differ_from_the_admission_binding(tmp_path, monkeypatch):
    """F7: an unregistered writer between admission and attach is refused before intent."""
    async with configuration_graph(tmp_path, monkeypatch, attach=False, binding_sha="0" * 64) as g:
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_configuration_binding$"):
            await g.owner.attach_configuration(g.changes, g.authority)
        assert g.owner._failed and phases(g.owner) == [] and g.settings.read_bytes() == RAW


# ----------------------------------------------------------------------
# T06-T09: the four clean outcomes.
# ----------------------------------------------------------------------

async def test_close_without_pending_drains_no_pending(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        before = (g.settings.read_bytes(), identity(g.settings))
        plan = events(g.owner)[0]["plan"]
        assert await shutdown(g) is None
        assert phases(g.owner) == ["intent", "attached", "drained"]
        assert events(g.owner)[-1]["aggregate"] == {
            "outcome": "no_pending", "attempts": 0, "last_committed_identity": plan["initial_identity"],
            "last_committed_sha256": plan["initial_sha256"], "writer_returned": True, "open_temp_count": 0,
            "cleanup_uncertain": False}
        assert (g.settings.read_bytes(), identity(g.settings)) == before and temporaries(g) == []
        assert g.owner._closing.exception() is None and g.changes.registration_failure is None


async def test_pending_withdrawal_commits_after_opening_through_real_closed_privacy(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        pending = await queue(g)
        assert pending.previous_consent is True
        allocations, original = [], state_configuration.tempfile.mkstemp

        def mkstemp(*args, **kwargs):
            allocations.append(phases(g.owner)[-1])
            result = original(*args, **kwargs)
            allocations.append(identity(result[1]))
            return result
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", mkstemp)
        assert await shutdown(g) is None
        assert allocations[0] == "opening"
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
        persisted = yaml.safe_load(g.settings.read_text())
        assert persisted["reasoner"]["cloud"]["consent"] is False and persisted["other"] == "untouched"
        aggregate = drained(g.owner)["aggregate"]
        assert aggregate["outcome"] == "committed" and aggregate["attempts"] == 1
        assert aggregate["last_committed_identity"] == allocations[1] == identity(g.settings)
        assert aggregate["last_committed_sha256"] == hashlib.sha256(g.settings.read_bytes()).hexdigest()
        receipt = g.changes.receipt
        assert receipt.consent is False and receipt.config_sha256 == aggregate["last_committed_sha256"]
        assert receipt.previous_config_sha256 == hashlib.sha256(RAW).hexdigest()
        assert temporaries(g) == [] and g.owner._closing.exception() is None


async def test_identical_bytes_commit_is_recorded_as_committed(tmp_path, monkeypatch):
    """F1: an already-withdrawn file re-encodes to the same bytes on a new inode."""
    value = {"reasoner": {"cloud": {"consent": False}}, "other": "untouched"}
    raw = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    async with configuration_graph(tmp_path, monkeypatch, raw=raw) as g:
        graph_ready(g)
        initial = identity(g.settings)
        assert (await queue(g)).previous_consent is False
        assert await shutdown(g) is None
        aggregate = drained(g.owner)["aggregate"]
        assert g.settings.read_bytes() == raw and identity(g.settings) != initial
        assert aggregate["outcome"] == "committed"
        assert aggregate["last_committed_sha256"] == hashlib.sha256(raw).hexdigest()
        assert aggregate["last_committed_identity"] == identity(g.settings)


async def test_revoked_actor_after_opening_is_refused_before_temporary(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        issued = await g.authority.issue_credential(g.actor, subject="synthetic-configuration-operator",
            role=DashboardRole.OPERATOR, expires_at=g.authority._now() + 600)
        operator = g.authority.resolve(issued.token)
        await queue(g, operator)
        assert await g.authority.revoke_credential(g.actor, issued.identity)
        error = await shutdown(g)
        assert type(error) is PermissionError and str(error) == "configuration_actor_denied"
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "refused_before_temporary"
        assert drained(g.owner)["aggregate"]["attempts"] == 0
        assert g.settings.read_bytes() == RAW and temporaries(g) == []
        assert g.owner._closing.exception() is None and g.changes.receipt is None


async def test_settings_changed_after_attach_poisons_at_opening(tmp_path, monkeypatch):
    """Fail closed: a change after attach comes from an unregistered writer."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        changed = RAW.replace(b"untouched", b"external-edit")
        g.settings.write_bytes(changed)
        await shutdown(g)
        assert "output_registration_destination_changed" in messages(config_error(g))
        assert g.changes.registration_failure == "output_registration_destination_changed"
        assert g.settings.read_bytes() == changed and temporaries(g) == [] and g.changes.receipt is None
        assert_poisoned(g, last="withdrawn")


async def test_graph_close_unconfirmed_drains_refused_before_temporary_from_attached(tmp_path, monkeypatch):
    """F2: the owner's own refusal before opening still gets exactly one terminal."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        await queue(g)  # No ApplicationGraph was attached: close cannot be confirmed.
        error = await shutdown(g)
        assert type(error) is RuntimeError and str(error) == "configuration_graph_close_unconfirmed"
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "refused_before_temporary"
        assert g.settings.read_bytes() == RAW and g.owner._closing.exception() is None


async def test_failed_graph_with_pending_is_not_attempted(tmp_path, monkeypatch):
    """F2: a pending change abandoned by a failed graph is never reported as no_pending."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        # The stack observes the failure; the caller still owns re-raising it.
        assert await shutdown(g, RuntimeError("synthetic graph failure")) is None
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "not_attempted_graph_failed"
        assert g.settings.read_bytes() == RAW and g.owner._closing.exception() is None


@pytest.mark.parametrize("failure", ["replace_error", "admission_after_temporary"])
async def test_failure_after_temporary_unlinks_only_own_temp_and_drains_clean_failed(tmp_path, monkeypatch, failure):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        created, raised = [], []
        original_mkstemp, original_replace = state_configuration.tempfile.mkstemp, os.replace

        def mkstemp(*args, **kwargs):
            result = original_mkstemp(*args, **kwargs)
            created.append(result[1])
            return result

        def replace(source, destination, *args, **kwargs):
            if failure == "replace_error" and Path(destination) == g.settings:
                raise OSError("synthetic replace failure")
            return original_replace(source, destination, *args, **kwargs)

        def revalidate():
            if failure == "admission_after_temporary" and created and not raised:
                raised.append(True)
                raise RuntimeError("synthetic admission revoked")
            g.checks.append("full")
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", mkstemp)
        monkeypatch.setattr(state_configuration.os, "replace", replace)
        monkeypatch.setattr(g.admission, "revalidate", revalidate)
        error = await shutdown(g)
        assert type(error) is RuntimeError and str(error) == "configuration_write_unconfirmed"
        assert len(created) == 1 and not os.path.lexists(created[0]) and temporaries(g) == []
        assert g.settings.read_bytes() == RAW
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
        aggregate = drained(g.owner)["aggregate"]
        assert aggregate["outcome"] == "clean_failed" and aggregate["attempts"] == 1
        assert g.owner._closing.exception() is None and g.changes.receipt is None


# ----------------------------------------------------------------------
# T10: uncertain publication and the extent.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_inode", "reused_inode_number"])
async def test_substituted_temporary_is_never_replaced_or_deleted(tmp_path, monkeypatch, reuse):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        created = own_temporaries(monkeypatch, reuse=reuse)

        def revalidate():
            if created and Path(created[0]).read_bytes() != b"canary":
                Path(created[0]).unlink()
                Path(created[0]).write_bytes(b"canary")  # A new inode at the own name.
        monkeypatch.setattr(g.admission, "revalidate", revalidate)
        await shutdown(g)
        assert "configuration_write_unconfirmed" in messages(config_error(g))
        assert Path(created[0]).read_bytes() == b"canary" and g.settings.read_bytes() == RAW
        assert g.changes.registration_failure == "output_registration_temporary_changed"
        assert_poisoned(g)


async def test_destination_substituted_after_replace_is_uncertain(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        original = os.replace

        def replace(source, destination, *args, **kwargs):
            result = original(source, destination, *args, **kwargs)
            if Path(destination) == g.settings:
                foreign = g.settings.with_name("foreign.yaml")
                foreign.write_bytes(b"foreign: true\n")
                original(foreign, g.settings)
            return result
        monkeypatch.setattr(state_configuration.os, "replace", replace)
        await shutdown(g)
        assert "configuration_write_unconfirmed" in messages(config_error(g))
        assert g.settings.read_bytes() == b"foreign: true\n" and g.changes.receipt is None
        assert g.changes.registration_failure == "output_registration_destination_changed"
        assert_poisoned(g)


async def test_directory_fsync_failure_after_replace_is_uncertain(tmp_path, monkeypatch):
    """F9: a published but undurable replacement is never reported as committed."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        replaced, original_replace, original_fsync = [], os.replace, os.fsync

        def replace(source, destination, *args, **kwargs):
            result = original_replace(source, destination, *args, **kwargs)
            if Path(destination) == g.settings:
                replaced.append(True)
            return result

        def fsync(descriptor):
            if replaced == [True] and stat.S_ISDIR(os.fstat(descriptor).st_mode):
                replaced.append("failed")
                raise OSError("synthetic directory fsync failure")
            return original_fsync(descriptor)
        monkeypatch.setattr(state_configuration.os, "replace", replace)
        monkeypatch.setattr(state_configuration.os, "fsync", fsync)
        await shutdown(g)
        assert replaced == [True, "failed"] and "configuration_write_unconfirmed" in messages(config_error(g))
        assert yaml.safe_load(g.settings.read_text())["reasoner"]["cloud"]["consent"] is False
        assert g.changes.receipt is None
        assert g.changes.registration_failure == "output_registration_publication_unconfirmed"
        assert_poisoned(g)


@pytest.mark.parametrize("misuse", ["reuse", "foreign_owner"])
async def test_extent_is_single_use_and_bound_to_its_owner(tmp_path, monkeypatch, misuse):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        extent = await registration.configuration_opening(g.changes)
        path, roots, current, _mode, _raw, digest = g.changes._read()
        if misuse == "reuse":
            extent.before_temporary(g.changes, path, roots, current, digest)
            assert not g.owner._failed
        owner = g.changes if misuse == "reuse" else ConfigurationChanges(g.authority, g.admission)
        with pytest.raises(registration.OutputRegistrationError):
            extent.before_temporary(owner, path, roots, current, digest)
        assert g.owner._failed and temporaries(g) == []
        await shutdown(g)
        # Poisoned before privacy closed: that earlier exit fails first, so the
        # configuration exit sees a failed graph and never writes.
        assert g.settings.read_bytes() == RAW and type(config_error(g)) is registration.OutputRegistrationError
        assert_poisoned(g, closing=None)


async def test_managed_owner_without_attachment_is_refused_before_temporary(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch, attach=False) as g:
        graph_ready(g)
        # OQ-8: without an attached publisher the marker cannot be recorded: the
        # gate stays closed in-run, nothing is queued and the epoch is poisoned.
        outcome = await g.changes.withdraw_cloud_consent(g.actor)
        assert outcome.gate_closed is True and g.gate.withdrawn is True
        assert outcome.marker is MarkerRecordState.REGISTRATION_FAILED and outcome.pending is None
        assert outcome.failure == "output_registration_required" and g.owner._failed
        calls = []
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", lambda *a, **k: calls.append(a))
        await shutdown(g)
        assert "configuration_withdrawal_unconfirmed" in messages(config_error(g))
        assert calls == [] and g.settings.read_bytes() == RAW and phases(g.owner) == []
        assert g.changes.registration_failure == "output_registration_required"


async def test_repeated_observer_cancellation_drains_once(tmp_path, monkeypatch):
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        entered, release, calls = threading.Event(), threading.Event(), []
        original = state_configuration.tempfile.mkstemp

        def mkstemp(*args, **kwargs):
            calls.append(True)
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", mkstemp)
        task = asyncio.create_task(shutdown(g))
        assert await asyncio.to_thread(entered.wait, 10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
        release.set()
        error = await task
        assert isinstance(error, asyncio.CancelledError)
        assert calls == [True] and phases(g.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "committed"
        assert yaml.safe_load(g.settings.read_text())["reasoner"]["cloud"]["consent"] is False
        assert g.changes.receipt is not None and g.owner._closing.exception() is None


async def test_drained_after_replace_and_later_revalidation_succeed(tmp_path, monkeypatch):
    """F6 positive oracle: the post-replace terminal needs only live checks.

    Full revalidation after the replace is not forbidden; the real binding
    (which never recomputes the configuration projection) is proven by the
    compiled advance-N run.
    """
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        marks, original = [], os.replace

        def replace(source, destination, *args, **kwargs):
            result = original(source, destination, *args, **kwargs)
            if Path(destination) == g.settings:
                marks.append((len(g.checks), base._read(g.owner._directory / "HEAD.json")[0]["revision"]))
            return result
        monkeypatch.setattr(state_configuration.os, "replace", replace)
        assert await shutdown(g) is None
        (checks_at_replace, revision_at_replace), = marks
        assert drained(g.owner)["revision"] > revision_at_replace
        assert "live" in g.checks[checks_at_replace:]
        g.admission.revalidate()
        g.owner._current_pins()
        assert v4().chain(g.owner._directory)[2]["configuration"]["phase"] == "drained"


# ----------------------------------------------------------------------
# T14: composed pin on the real ApplicationGraph.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["same", "swapped_changes", "foreign_authority", "missing_changes",
                                   "replaced_after_attach"])
async def test_composed_pin_uses_the_real_application_graph(tmp_path, monkeypatch, shape):
    async with configuration_graph(tmp_path, monkeypatch, accounts=True) as g:
        if shape == "same":
            graph = application_graph(g)
            g.changes.attach_graph(graph)
        elif shape == "swapped_changes":
            other = ConfigurationChanges(g.authority, g.admission)
            graph = application_graph(g, configuration_changes=other)
            other.attach_graph(graph)
        elif shape == "foreign_authority":
            graph = application_graph(g, dashboard_authority=DashboardAuthority(g.data / "foreign.db"))
        elif shape == "replaced_after_attach":
            # The graph is a mutable dataclass: the owner joined it, then the field moved.
            graph = application_graph(g)
            g.changes.attach_graph(graph)
            graph.configuration_changes = ConfigurationChanges(g.authority, g.admission)
        else:
            graph = application_graph(g, configuration_changes=None)
        assert type(graph) is ApplicationGraph
        expected = ("output_registration_composition_incomplete" if shape == "same"
                    else "output_registration_configuration_owner")
        with pytest.raises(registration.OutputRegistrationError) as error:
            await registration.composed(g.owner, graph)
        assert str(error.value) == expected and g.owner._failed


@pytest.mark.parametrize("shape", ["never_attached", "attached_to_other_graph"])
async def test_composed_pin_requires_attach_graph_on_the_same_graph(tmp_path, monkeypatch, shape):
    async with configuration_graph(tmp_path, monkeypatch, accounts=True) as g:
        graph = application_graph(g)
        if shape == "attached_to_other_graph":
            g.changes.attach_graph(application_graph(g))
        assert g.changes._graph is not graph
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_configuration_owner$"):
            await registration.composed(g.owner, graph)


@pytest.mark.parametrize("case", ["selected_unattached", "unselected_not_disabled"])
async def test_configuration_completeness_is_checked_in_the_loop(tmp_path, monkeypatch, case):
    """Isolate the loop branch: selection is reduced to the configuration row."""
    dashboard = case == "selected_unattached"
    async with configuration_graph(tmp_path, monkeypatch, dashboard=dashboard, attach=False) as g:
        monkeypatch.setattr(g.owner, "_selected", {"configuration": dashboard})
        journal = sorted(p.name for p in g.owner._directory.iterdir())
        graph = application_graph(g)
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_composition_incomplete$"):
            await registration.composed(g.owner, graph)
        assert sorted(p.name for p in g.owner._directory.iterdir()) == journal


def test_application_graph_still_exposes_the_fields_the_configuration_pin_reads():
    """Drift guard: a rename must fail here, not only in the compiled graph run."""
    import dataclasses
    names = {field.name for field in dataclasses.fields(ApplicationGraph)}
    assert {"configuration_changes", "dashboard_authority", "dashboard", "privacy"} <= names


# ----------------------------------------------------------------------
# T15: dashboard disabled.
# ----------------------------------------------------------------------

async def test_disabled_dashboard_records_disabled_and_rejects_attach_or_duplicate(tmp_path, monkeypatch):
    for second in ("disabled_again", "attach", "attach_selected"):
        async with configuration_graph(tmp_path / second, monkeypatch, dashboard=False) as g:
            owner = g.owner
            assert owner._selected["configuration"] is False
            await owner.configuration_disabled()
            state = drained(owner)
            assert {key: state[key] for key in ("phase", "plan", "producer", "path")} == {
                "phase": "disabled", "plan": None, "producer": "configuration", "path": str(g.settings)}
            journal = sorted(p.name for p in owner._directory.iterdir())
            authority = DashboardAuthority(g.data / "authority.db")
            changes = ConfigurationChanges(authority, g.admission)
            changes.install_on(AsyncExitStack())
            if second == "attach_selected":
                monkeypatch.setitem(owner._selected, "configuration", True)
            with pytest.raises(registration.OutputRegistrationError, match="^output_registration_configuration_owner$"):
                if second == "disabled_again":
                    await owner.configuration_disabled()
                else:
                    await owner.attach_configuration(changes, authority)
            assert owner._failed and owner._configuration is None
            assert sorted(p.name for p in owner._directory.iterdir()) == journal
    async with configuration_graph(tmp_path / "clean", monkeypatch, dashboard=False) as g:
        await g.owner.configuration_disabled()
        assert await shutdown(g) is None and g.owner._closing.exception() is None


async def test_disabled_pin_requires_no_configuration_owner_on_the_graph(tmp_path, monkeypatch):
    for shape in ("absent", "present"):
        async with configuration_graph(tmp_path / shape, monkeypatch, dashboard=False, accounts=True) as g:
            await g.owner.configuration_disabled()
            present = ConfigurationChanges(None, g.admission) if shape == "present" else None
            graph = application_graph(g, configuration_changes=present)
            expected = ("^output_registration_composition_incomplete$" if shape == "absent"
                        else "^output_registration_configuration_owner$")
            with pytest.raises(registration.OutputRegistrationError, match=expected):
                await registration.composed(g.owner, graph)


# ----------------------------------------------------------------------
# T18/T19: OQ-8 fail closed; T20: OQ-5 unmanaged cleanup.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cause", ["failed", "unverified"])
async def test_registration_failure_blocks_the_withdrawal_and_is_recorded(tmp_path, monkeypatch, cause):
    async with configuration_graph(tmp_path, monkeypatch, poison_before_configuration=cause) as g:
        graph_ready(g)
        pending = await queue(g)
        calls = []
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", lambda *a, **k: calls.append(a))
        await shutdown(g)
        assert "output_registration_coordinator" in messages(config_error(g))
        assert g.changes.registration_failure == "output_registration_coordinator"
        # The withdrawal is not written and never reported as committed.
        assert calls == [] and g.settings.read_bytes() == RAW and temporaries(g) == []
        assert g.changes.receipt is None and g.changes.pending is pending and pending.state == "pending"
        assert phases(g.owner) == ["intent", "attached", "withdrawn"]


async def test_epoch_poisoned_before_graph_close_also_leaves_the_withdrawal_unwritten(tmp_path, monkeypatch):
    """OQ-8, earlier poisoning: the graph close fails first; nothing is attempted."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        calls = []
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", lambda *a, **k: calls.append(a))
        g.owner._failed = True
        error = await shutdown(g)
        # The earlier exit's error passes through our exit unchanged.
        assert error is not None and str(config_error(g)) == "output_registration_registry_unavailable"
        assert calls == [] and g.settings.read_bytes() == RAW and g.changes.receipt is None
        assert phases(g.owner) == ["intent", "attached", "withdrawn"]


async def test_registration_failure_withholds_the_receipt_so_no_next_generation_starts(tmp_path, monkeypatch):
    """OQ-8: no closed receipt => the admission head stays running => no next run."""
    receipts = []
    monkeypatch.setattr(runtime_admission, "_stage",
                        lambda: SimpleNamespace(write_admission_receipt=lambda *a: receipts.append(a)))
    async with configuration_graph(tmp_path, monkeypatch, poison_before_configuration="failed") as g:
        graph_ready(g)
        await queue(g)
        g.admission._graph = object()  # Any receipt attempt past the exc_type check would fail loudly.
        await shutdown(g)
        assert "output_registration_coordinator" in messages(config_error(g)) and receipts == []
        assert g.settings.read_bytes() == RAW
    # The unchanged controller contract: a run without a closed receipt keeps
    # its head 'running', and no next generation is admitted from it.
    root = tmp_path / "controller"
    (root / "admission").mkdir(parents=True)
    stage._cas_head(root, None, "a" * 64, "running", uuid4().hex)
    with pytest.raises(ValueError, match="runtime_admission_not_closed"):
        with stage.AdmissionController(root):
            pass


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_inode", "reused_inode_number"])
async def test_unmanaged_cleanup_never_deletes_a_substituted_temporary(tmp_path, monkeypatch, reuse):
    """OQ-5: identity-checked cleanup also applies without a coordinator."""
    state = tmp_path / "state"
    settings = state / "config" / "settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(RAW)
    (state / "data").mkdir()
    config = {"memory": {"privacy_db_path": str(state / "data" / "privacy.db")}}
    admission = runtime_admission.RuntimeAdmission({"run_id": "unmanaged-control-leaf"}, {"state_root": str(state)},
                                                   config["memory"]["privacy_db_path"], config_digest="0" * 64)
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {})
    monkeypatch.setattr(base, "_ACTIVE", set())
    original_replace = os.replace
    created = own_temporaries(monkeypatch, reuse=reuse)

    def revalidate():
        if created and Path(created[0]).read_bytes() != b"canary":
            Path(created[0]).unlink()
            Path(created[0]).write_bytes(b"canary")

    def replace(source, destination, *args, **kwargs):
        if Path(destination) == settings:
            raise OSError("synthetic replace failure")
        return original_replace(source, destination, *args, **kwargs)
    monkeypatch.setattr(admission, "revalidate", revalidate)
    monkeypatch.setattr(state_configuration.os, "replace", replace)
    authority = DashboardAuthority(tmp_path / "auth.db", expected_token=lambda: TOKEN)
    await authority.initialize()
    stack = AsyncExitStack()
    try:
        gate = CloudConsentGate()
        owner = ConfigurationChanges(authority, admission, gate)
        owner.install_on(stack)
        privacy = await build_privacy_services(config, shutdown=stack)
        g = SimpleNamespace(config=config, privacy=privacy, api=None, authority=authority, changes=owner)
        owner.attach_graph(application_graph(g))
        outcome = await owner.withdraw_cloud_consent(authority.resolve("Bearer " + TOKEN))
        assert outcome.marker is MarkerRecordState.UNMANAGED and outcome.pending is not None
        assert gate.withdrawn is True
        with pytest.raises(RuntimeError, match="^configuration_write_unconfirmed$"):
            await stack.__aexit__(None, None, None)
    finally:
        await authority.aclose()
    assert Path(created[0]).read_bytes() == b"canary" and settings.read_bytes() == RAW
    assert owner.registration_failure is None


# ----------------------------------------------------------------------
# Independent review R1-R6: guards that must each fail a test on their own.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("change", ["in_place", "new_inode", "after_commit"])
async def test_unregistered_change_is_caught_by_the_drained_observation(tmp_path, monkeypatch, change):
    """R1: the terminal must match the real file, also for outcomes that wrote nothing."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        foreign = RAW.replace(b"untouched", b"unregistered-edit")
        if change == "after_commit":
            await queue(g)
            state, real_replace, real_fsync = [], os.replace, os.fsync

            def replace(source, destination, *args, **kwargs):
                result = real_replace(source, destination, *args, **kwargs)
                if Path(destination) == g.settings:
                    state.append("replaced")
                return result

            def fsync(descriptor):
                result = real_fsync(descriptor)
                if state == ["replaced"] and stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    state.append("swapped")  # Durable commit, then an unregistered writer.
                    sibling = g.settings.with_name("unregistered.yaml")
                    sibling.write_bytes(foreign)
                    real_replace(sibling, g.settings)
                return result
            monkeypatch.setattr(state_configuration.os, "replace", replace)
            monkeypatch.setattr(state_configuration.os, "fsync", fsync)
        elif change == "in_place":
            g.settings.write_bytes(foreign)  # Same inode, other bytes, nothing pending.
        else:
            sibling = g.settings.with_name("unregistered.yaml")
            sibling.write_bytes(RAW)  # Same bytes on a new inode, nothing pending.
            os.replace(sibling, g.settings)
        await shutdown(g)
        assert "output_registration_destination_changed" in messages(config_error(g))
        assert g.changes.registration_failure == "output_registration_destination_changed"
        assert_poisoned(g, last="opening" if change == "after_commit" else "attached")


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_inode", "reused_inode_number"])
async def test_unproven_cleanup_after_a_failed_replace_poisons(tmp_path, monkeypatch, reuse):
    """R2: a name left at the own temp path is never drained as clean_failed."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        real_replace = os.replace
        created = own_temporaries(monkeypatch, reuse=reuse)

        def replace(source, destination, *args, **kwargs):
            if Path(destination) == g.settings:
                # before_replace already accepted the own temp; now its name is taken.
                Path(source).unlink()
                Path(source).write_bytes(b"canary")
                raise OSError("synthetic replace failure")
            return real_replace(source, destination, *args, **kwargs)
        monkeypatch.setattr(state_configuration.os, "replace", replace)
        await shutdown(g)
        assert "configuration_write_unconfirmed" in messages(config_error(g))
        assert Path(created[0]).read_bytes() == b"canary" and g.settings.read_bytes() == RAW
        assert g.changes.registration_failure == "output_registration_cleanup_unproven"
        assert_poisoned(g)


async def test_destination_swapped_after_the_owner_check_is_never_replaced(tmp_path, monkeypatch):
    """R3: the extent rechecks the destination after the owner's own pinned read."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        created, swapped, real_mkstemp, real_read = [], [], state_configuration.tempfile.mkstemp, g.changes._read

        def mkstemp(*args, **kwargs):
            result = real_mkstemp(*args, **kwargs)
            created.append(result[1])
            return result

        def read():
            result = real_read()
            if created and not swapped:
                swapped.append(True)  # Same bytes on a new inode, after the owner's check.
                sibling = g.settings.with_name("operator.yaml")
                sibling.write_bytes(RAW)
                os.replace(sibling, g.settings)
            return result
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", mkstemp)
        monkeypatch.setattr(g.changes, "_read", read)
        await shutdown(g)
        assert swapped == [True] and "configuration_write_unconfirmed" in messages(config_error(g))
        # The operator's file was not overwritten by the withdrawal.
        assert g.settings.read_bytes() == RAW and not os.path.lexists(created[0])
        assert g.changes.registration_failure == "output_registration_temporary_changed"
        assert_poisoned(g)


@pytest.mark.parametrize("exit_", ["task_not_created", "result_unavailable"])
async def test_writer_that_cannot_start_or_report_still_drains_once(tmp_path, monkeypatch, exit_):
    """R5: the two rare writer exits still record exactly one terminal."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        real = asyncio.to_thread
        failure = RuntimeError("synthetic executor unavailable")

        def owned(function):
            return getattr(function, "__func__", None) is ConfigurationChanges._commit

        def to_thread_sync(function, *args, **kwargs):
            if owned(function):
                raise failure
            return real(function, *args, **kwargs)

        async def to_thread_async(function, *args, **kwargs):
            if owned(function):
                raise failure
            return await real(function, *args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread",
                            to_thread_sync if exit_ == "task_not_created" else to_thread_async)
        error = await shutdown(g)
        assert error is failure and config_error(g) is failure
        assert phases(g.owner) == ["intent", "attached", "withdrawn", "opening", "drained"]
        assert drained(g.owner)["aggregate"]["outcome"] == "refused_before_temporary"
        assert g.settings.read_bytes() == RAW and temporaries(g) == [] and g.changes.receipt is None
        assert g.owner._closing.exception() is None and g.changes.registration_failure is None


async def test_cancelled_opening_records_the_dropped_change(tmp_path, monkeypatch):
    """R6: a pending change dropped by cancellation during opening is recorded."""
    async with configuration_graph(tmp_path, monkeypatch) as g:
        graph_ready(g)
        await queue(g)
        entered, release = threading.Event(), threading.Event()
        real = registration.OutputRegistrationV2._append_configuration

        def append(self, phase, plan, aggregate=None):
            if phase == "opening":
                entered.set()
                assert release.wait(10)
            return real(self, phase, plan, aggregate)
        monkeypatch.setattr(registration.OutputRegistrationV2, "_append_configuration", append)
        calls = []
        monkeypatch.setattr(state_configuration.tempfile, "mkstemp", lambda *a, **k: calls.append(a))
        task = asyncio.create_task(shutdown(g))
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        await task
        assert isinstance(config_error(g), asyncio.CancelledError)
        assert g.changes.registration_failure == "output_registration_opening_cancelled"
        assert calls == [] and g.settings.read_bytes() == RAW and g.changes.receipt is None
        assert g.owner._failed and "drained" not in phases(g.owner)
