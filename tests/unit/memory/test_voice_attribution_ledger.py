"""U3b-4a-1: consents, verification, voice-owned proposals, «No es mío» counter, switch (design u3b-4 v4).

Every case runs on a real FactLedger file with injected scores (no audio, no
model). Refusals compare the database file bytes, so 'nothing changed' is
literal.
"""

import ast
import copy
import json
import logging
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from src.memory.fact_ledger import (
    FactLedger, FactState, MemoryAssigner, MemoryOrigin, MemoryOwnerActor, MemoryOwnerReader, MemoryReviewer,
    MemorySource, MemoryVoiceConsentActor, ReturnReceipt, VoiceSwitchActor,
)
from src.memory.privacy_services import voice_proposal_settings
from src.memory.voice_attribution import (
    AttributionReason, ProfileScore, ScoringSnapshot, VoiceAttributionCapability, VoiceConsentState,
    VoiceProposalSettings, VoiceVerificationOutcome, bucket, verification_outcome,
)
from src.users.voice_ledger import VoiceLedger
from tests.mocks.memory_sources import memory_source

NOW = 2_000_000_000.0
DAY = 86400.0
ON = VoiceProposalSettings(enabled=True)
OPERATOR = MemoryAssigner("synthetic-operator")
REVIEWER = MemoryReviewer("synthetic-reviewer")
SWITCH = VoiceSwitchActor("synthetic-operator")
CAP = VoiceAttributionCapability("synthetic-test")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
PERSON, OTHER = "user_" + "a" * 32, "user_" + "b" * 32
P1, P2, P3 = ("00000000-0000-4000-8000-%012x" % index for index in (1, 2, 3))
ROOT = Path(__file__).resolve().parents[3]


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


@pytest.fixture
def clock():
    return Clock()


def _open(path, clock, settings=ON):
    ledger = FactLedger(str(path), clock=clock, voice_settings=settings)
    ledger.initialize()
    return ledger


@pytest.fixture
def ledger(tmp_path, clock):
    value = _open(tmp_path / "privacy.db", clock)
    value.set_voice_switch(True, 0, SWITCH)
    try:
        yield value
    finally:
        value.close()


def add_profile(ledger, profile, person=None, *, revision=1, link_revision=1):
    """A live confirmed profile, optionally linked to a person."""
    source = memory_source(now=ledger.clock() - 10)
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_profiles VALUES (?,?,'enc_synthetic',2,?,?,?,?,?,NULL)",
                   (profile, "Synthetic " + profile[-1], b"\0" * 8, json.dumps([source.to_dict()], sort_keys=True),
                    ledger.clock() - 5, "synthetic-reviewer", revision))
        if person is not None:
            db.execute("INSERT INTO voice_profile_people VALUES (?,?,?,?,'synthetic-reviewer')",
                       (profile, person, link_revision, ledger.clock()))


def household(ledger):
    """P1 linked to PERSON, P2 unlinked, P3 linked to OTHER."""
    add_profile(ledger, P1, PERSON)
    add_profile(ledger, P2)
    add_profile(ledger, P3, OTHER)


def me(person=PERSON):
    return MemoryVoiceConsentActor("synthetic-home", person)


def snap(*scores, speech=3.0, complete=True, revisions=None):
    revisions = revisions or {}
    return ScoringSnapshot("enc_synthetic", tuple(ProfileScore(profile, revisions.get(profile, 1), score)
                                                  for profile, score in scores), complete, speech)


def full(top=P1, score=.95, rest=.30, **kwargs):
    return snap(*((profile, score if profile == top else rest) for profile in (P1, P2, P3)), **kwargs)


def scorer(snapshot):
    return lambda db: snapshot


def verify(ledger, snapshot, person=PERSON, **kwargs):
    consent = ledger.voice_consent(me(person))
    binding = ledger.open_voice_verification(me(person), consent.revision)
    return ledger.verify_voice_consent(binding, scorer(snapshot), CAP, **{**ALLOW, **kwargs})


def activate(ledger, person=PERSON, top=P1):
    ledger.request_voice_consent(me(person), ledger.voice_consent(me(person)).revision)
    result = verify(ledger, full(top), person)
    assert result.outcome is VoiceVerificationOutcome.PASSED
    return result


def proposal(ledger, content="Synthetic voice proposal", origin=MemoryOrigin.VOICE, sources=1):
    made = tuple(MemorySource(str(uuid4()), str(uuid4()), ledger.clock() - 1, ledger.clock() + 47 * 3600, origin)
                 for _ in range(sources))
    return ledger.store(content, "fact", 1.0, made, proposed=True)


def attribute(ledger, fact, snapshot=None, **kwargs):
    return ledger.attribute_voice_proposal(fact.id, scorer(snapshot or full()), CAP, **{**ALLOW, **kwargs})


def owned_voice(ledger):
    """A household with PERSON verified on P1 and one voice proposal attributed to PERSON."""
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    assert attribute(ledger, fact).owner
    return fact


def rows(ledger, table, where="1", parameters=()):
    return [tuple(row) for row in ledger._db.execute("SELECT * FROM " + table + " WHERE " + where, parameters)]


def file_bytes(ledger):
    return Path(ledger.path).read_bytes()


def refused(ledger, error, code, call):
    before = file_bytes(ledger)
    with pytest.raises(error, match=code):
        call()
    assert file_bytes(ledger) == before


# --- G2: settings are read, never written ---------------------------------------------------------


def _digest(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("config, enabled", [
    ({}, False), ({"memory": {}}, False), ({"memory": {"voice_proposals": None}}, False),
    ({"memory": {"voice_proposals": {}}}, False), ({"memory": {"voice_proposals": {"enabled": False}}}, False),
    ({"memory": {"voice_proposals": {"enabled": "true"}}}, False), ({"memory": {"voice_proposals": {"enabled": 1}}}, False),
    ({"memory": {"voice_proposals": {"enabled": True}}}, True),
])
def test_missing_or_non_bool_settings_mean_off_and_the_config_is_untouched(config, enabled):
    before = copy.deepcopy(config)
    settings = VoiceProposalSettings.from_config(config)
    assert settings.enabled is enabled
    assert settings == VoiceProposalSettings(enabled=enabled)
    assert config == before and _digest(config) == _digest(before)


@pytest.mark.parametrize("values", [
    {"min_score": "0.8"}, {"min_score": math.nan}, {"min_score": 1.5}, {"min_margin": -0.1},
    {"min_scoring_profiles": 1}, {"min_scoring_profiles": 2.0}, {"verification_max_failures": True},
    {"not_mine_window_days": math.inf}, {"verification_lock_window_s": 1.0}, "not a mapping",
])
def test_invalid_present_values_leave_the_voice_stage_off_without_touching_the_config(values, caplog):
    config = {"memory": {"voice_proposals": values if type(values) is str else {"enabled": True, **values}}}
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="voice_proposals_config_invalid"):
        VoiceProposalSettings.from_config(config)
    with caplog.at_level(logging.WARNING):
        assert voice_proposal_settings(config) == VoiceProposalSettings()
    assert "voice_proposals_config_invalid" not in caplog.text  # type-only log
    assert config == before


def test_valid_thresholds_are_copied():
    config = {"memory": {"voice_proposals": {"enabled": True, "min_score": 1, "not_mine_suspend_count": 3}}}
    settings = VoiceProposalSettings.from_config(config)
    assert settings.min_score == 1.0 and type(settings.min_score) is float and settings.not_mine_suspend_count == 3
    assert config == {"memory": {"voice_proposals": {"enabled": True, "min_score": 1, "not_mine_suspend_count": 3}}}


def test_settings_off_keeps_the_stage_off_even_with_the_switch_row_on(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    ledger.close()
    ledger = _open(tmp_path / "privacy.db", clock, VoiceProposalSettings())
    try:
        state = ledger.voice_switch()
        assert (state.configured, state.enabled, state.revision, state.effective) == (False, True, 1, False)
        household(ledger)
        refused(ledger, ValueError, "memory_voice_disabled", lambda: ledger.request_voice_consent(me(), 0))
    finally:
        ledger.close()


# --- G1: the switch ------------------------------------------------------------------------------


def test_switch_missing_row_reads_off_and_writes_are_cas(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    try:
        state = ledger.voice_switch()
        assert (state.enabled, state.revision, state.effective) == (False, 0, False)
        before = file_bytes(ledger)
        assert ledger.set_voice_switch(False, 0, SWITCH).revision == 0  # off over a missing row: no write
        assert file_bytes(ledger) == before and rows(ledger, "voice_attribution_control") == []
        refused(ledger, ValueError, "memory_voice_switch_conflict", lambda: ledger.set_voice_switch(True, 1, SWITCH))
        assert ledger.set_voice_switch(True, 0, SWITCH).effective
        assert rows(ledger, "voice_attribution_control") == [(1, 1, 1, NOW, "synthetic-operator")]
        refused(ledger, ValueError, "memory_voice_switch_conflict", lambda: ledger.set_voice_switch(False, 0, SWITCH))
        off = ledger.set_voice_switch(False, 1, SWITCH)
        assert (off.enabled, off.revision, off.effective) == (False, 2, False)
        refused(ledger, PermissionError, "memory_voice_switch_denied",
                lambda: ledger.set_voice_switch(False, 2, OPERATOR))
    finally:
        ledger.close()


def test_switch_on_needs_the_settings_flag(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock, VoiceProposalSettings())
    try:
        refused(ledger, ValueError, "memory_voice_not_configured", lambda: ledger.set_voice_switch(True, 0, SWITCH))
        assert ledger.set_voice_switch(False, 0, SWITCH).revision == 0
    finally:
        ledger.close()


def _principal(role, permissions, user_id=None):
    from src.dashboard.auth import DashboardPrincipal
    from src.dashboard.authority import DashboardRole
    return DashboardPrincipal("synthetic-subject", frozenset(permissions), DashboardRole(role), user_id)


def test_only_an_operator_with_admin_and_assign_mints_the_switch_actor():
    full_set = {"admin:manage", "memory:assign"}
    assert VoiceSwitchActor.from_dashboard(_principal("operator", full_set)).subject == "synthetic-subject"
    for principal in (_principal("home", full_set, PERSON), _principal("evaluator", full_set),
                      _principal("operator", {"admin:manage"}), _principal("operator", {"memory:assign"}), object()):
        with pytest.raises(PermissionError, match="memory_voice_switch_denied"):
            VoiceSwitchActor.from_dashboard(principal)


def test_only_a_person_with_the_voice_grant_mints_the_consent_actor():
    actor = MemoryVoiceConsentActor.from_dashboard(_principal("home", {"memory:own:voice"}, PERSON))
    assert actor.person_id == PERSON
    for principal in (_principal("home", {"memory:own:write", "memory:own:read"}, PERSON),
                      _principal("home", {"memory:own:voice"}), _principal("operator", {"memory:own:voice"}, PERSON)):
        with pytest.raises(PermissionError, match="memory_voice_actor_denied"):
            MemoryVoiceConsentActor.from_dashboard(principal)


def test_switching_off_between_the_two_attribution_steps_writes_nothing(ledger):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    original = ledger._attribution_read

    def flip(db, fact_id, score, retired, active, expected):
        if expected is not None:
            db.execute("UPDATE voice_attribution_control SET enabled=0, revision=revision+1")
        return original(db, fact_id, score, retired, active, expected)

    ledger._attribution_read = flip
    assert attribute(ledger, fact).reason is AttributionReason.DISABLED
    assert rows(ledger, "fact_owners") == [] and rows(ledger, "fact_owner_revisions") == []


# --- consent state machine ------------------------------------------------------------------------


def test_consent_lifecycle_and_refusals(ledger):
    household(ledger)
    assert ledger.voice_consent(me()).state is None
    refused(ledger, ValueError, "memory_voice_consent_conflict", lambda: ledger.request_voice_consent(me(), 1))
    pending = ledger.request_voice_consent(me(), 0)
    assert (pending.state, pending.revision, pending.verified_profile_id) == (
        VoiceConsentState.PENDING_VERIFICATION, 1, None)
    refused(ledger, ValueError, "memory_voice_consent_state", lambda: ledger.request_voice_consent(me(), 1))
    revoked = ledger.revoke_voice_consent(me(), 1)
    assert (revoked.state, revoked.revision) == (VoiceConsentState.REVOKED, 2)
    refused(ledger, ValueError, "memory_voice_consent_state", lambda: ledger.revoke_voice_consent(me(), 2))
    assert ledger.request_voice_consent(me(), 2).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full()).outcome is VoiceVerificationOutcome.PASSED
    active = ledger.voice_consent(me())
    assert (active.state, active.verified_profile_id) == (VoiceConsentState.ACTIVE, P1)
    assert rows(ledger, "voice_attribution_consents") == [
        (PERSON, "active", 4, P1, 1, 1, NOW, "system:voice-verification")]
    refused(ledger, ValueError, "memory_voice_consent_state", lambda: ledger.request_voice_consent(me(), 4))
    refused(ledger, ValueError, "memory_voice_consent_state",
            lambda: ledger.reopen_voice_consent(PERSON, 4, OPERATOR))
    assert ledger.revoke_voice_consent(me(), 4).state is VoiceConsentState.REVOKED
    with pytest.raises(PermissionError):
        ledger.request_voice_consent(MemoryOwnerActor("synthetic-home", PERSON), 5)


def test_consent_requires_exactly_one_live_linked_profile(ledger):
    add_profile(ledger, P2)
    refused(ledger, ValueError, "memory_voice_no_profile", lambda: ledger.request_voice_consent(me(), 0))
    add_profile(ledger, P1, PERSON)
    add_profile(ledger, P3, PERSON)
    refused(ledger, ValueError, "memory_voice_profile_ambiguous", lambda: ledger.request_voice_consent(me(), 0))


def test_revocation_and_status_work_with_the_switch_off(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    ledger.set_voice_switch(False, 1, SWITCH)
    assert ledger.voice_consent(me()).state is VoiceConsentState.PENDING_VERIFICATION
    refused(ledger, ValueError, "memory_voice_disabled", lambda: ledger.open_voice_verification(me(), 1))
    assert ledger.revoke_voice_consent(me(), 1).state is VoiceConsentState.REVOKED
    refused(ledger, ValueError, "memory_voice_disabled", lambda: ledger.request_voice_consent(me(), 2))


def test_only_an_operator_reopens_a_locked_consent(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    for _ in range(3):
        result = verify(ledger, full(score=.5))
    assert result.state is VoiceConsentState.VERIFICATION_LOCKED
    locked = ledger.voice_consent(me())
    refused(ledger, ValueError, "memory_voice_consent_locked", lambda: ledger.request_voice_consent(me(), locked.revision))
    refused(ledger, ValueError, "memory_voice_consent_state",
            lambda: ledger.open_voice_verification(me(), locked.revision))
    refused(ledger, ValueError, "memory_voice_consent_state",
            lambda: ledger.reopen_voice_consent(PERSON, locked.revision + 1, OPERATOR))
    refused(ledger, ValueError, "memory_voice_consent_state", lambda: ledger.reopen_voice_consent(OTHER, 0, OPERATOR))
    with pytest.raises(PermissionError):
        ledger.reopen_voice_consent(PERSON, locked.revision, me())
    assert ledger.reopen_voice_consent(PERSON, locked.revision, OPERATOR).state is VoiceConsentState.PENDING_VERIFICATION


# --- verification -----------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", list(AttributionReason))
def test_classification_matches_r8(reason):
    expected = {"owner": "passed", "top_not_grantable": "failed", "below_score": "failed",
                "below_margin": "failed", "short_speech": "failed", "tie": "failed",
                "top_foreign": "failed_foreign"}.get(reason.value, "unavailable")
    assert verification_outcome(reason).value == expected


def _stale(ledger):
    consent = ledger.voice_consent(me())
    ledger.revoke_voice_consent(me(), consent.revision)
    ledger.request_voice_consent(me(), consent.revision + 1)


CASES = {
    "owner": (full(), None), "below_score": (full(score=.7), None), "below_margin": (full(rest=.86, score=.9), None),
    "short_speech": (full(speech=1.0), None), "tie": (full(score=.9, rest=.9), None),
    "top_not_grantable": (full(top=P2), None), "top_foreign": (full(top=P3), None),
    "incomplete_scoring_set": (full(complete=False), None),
    "incomplete_scoring_set/missing_profile": (snap((P1, .95), (P2, .3)), None),
    "incomplete_scoring_set/stale_revision": (full(revisions={P2: 2}), None),
    "error": (None, None), "disabled": (full(), "switch_off"), "consent_stale": (full(), "stale"),
    "suspended": (full(), "suspend"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_verification_reasons_with_injected_scores(ledger, case):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    snapshot, action = CASES[case]
    binding = ledger.open_voice_verification(me(), 1)
    if action == "switch_off":
        ledger.set_voice_switch(False, 1, SWITCH)
    elif action == "stale":
        _stale(ledger)
    elif action == "suspend":
        with ledger._transaction() as db:
            db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')", (P1, NOW))

    def score(db):
        if snapshot is None:
            raise RuntimeError("synthetic scorer failure")
        return snapshot

    result = ledger.verify_voice_consent(binding, score, CAP, **ALLOW)
    reason = AttributionReason(case.split("/")[0])
    assert result.reason is reason and result.outcome is verification_outcome(reason)
    attempts = [row[2] for row in rows(ledger, "voice_verification_attempts", "person_id=? ORDER BY attempted_at",
                                       (PERSON,))]
    assert attempts == ["window_opened", result.outcome.value]
    consent = ledger.voice_consent(me())
    assert (consent.state is VoiceConsentState.ACTIVE) is (reason is AttributionReason.OWNER)
    if reason is AttributionReason.OWNER:
        assert rows(ledger, "voice_attribution_consents")[0][3:6] == (P1, 1, 1)


def test_an_unavailable_person_writes_no_attempt(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    binding = ledger.open_voice_verification(me(), 1)
    for retired, active in ((lambda ids: True, ALLOW["active"]), (ALLOW["retired"], lambda person: None),
                            (lambda ids: 1 / 0, ALLOW["active"])):
        result = ledger.verify_voice_consent(binding, scorer(full()), CAP, retired=retired, active=active)
        assert result.reason is AttributionReason.PERSON_UNAVAILABLE
    assert len(rows(ledger, "voice_verification_attempts")) == 1


def test_verification_and_attribution_require_the_capability(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    binding = ledger.open_voice_verification(me(), 1)
    fact = proposal(ledger)
    for capability in (None, OPERATOR, "synthetic"):
        refused(ledger, PermissionError, "memory_voice_capability_denied",
                lambda: ledger.verify_voice_consent(binding, scorer(full()), capability, **ALLOW))
        refused(ledger, PermissionError, "memory_voice_capability_denied",
                lambda: ledger.attribute_voice_proposal(fact.id, scorer(full()), capability, **ALLOW))


def test_lock_arithmetic_counts_only_failed_within_the_window(ledger, clock):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    for _ in range(3):
        assert verify(ledger, full(top=P3)).outcome is VoiceVerificationOutcome.FAILED_FOREIGN
    assert ledger.voice_consent(me()).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    clock.value += DAY + 10  # both counted failures (and the openings) are now past their windows, not pruned
    assert len(rows(ledger, "voice_verification_attempts", "outcome='failed'")) == 2
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.VERIFICATION_LOCKED


def test_window_preconditions(ledger):
    add_profile(ledger, P1, PERSON)
    ledger.request_voice_consent(me(), 0)
    refused(ledger, ValueError, "memory_voice_insufficient_profiles", lambda: ledger.open_voice_verification(me(), 1))
    add_profile(ledger, P2)
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')", (P1, NOW))
    refused(ledger, ValueError, "memory_voice_profile_suspended", lambda: ledger.open_voice_verification(me(), 1))
    refused(ledger, ValueError, "memory_voice_consent_conflict", lambda: ledger.open_voice_verification(me(), 0))


def test_window_rate_limit_is_durable_and_time_filtered(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    for _ in range(6):
        ledger.open_voice_verification(me(), 1)
    refused(ledger, PermissionError, "memory_voice_rate_limited", lambda: ledger.open_voice_verification(me(), 1))
    ledger.close()
    ledger = _open(tmp_path / "privacy.db", clock)  # a restart does not reset it
    try:
        refused(ledger, PermissionError, "memory_voice_rate_limited", lambda: ledger.open_voice_verification(me(), 1))
        clock.value += DAY + 1  # past the window; no prune ran, the rows are still there but do not count
        assert len(rows(ledger, "voice_verification_attempts")) == 6
        assert ledger.open_voice_verification(me(), 1).opened_at == clock.value
    finally:
        ledger.close()


def test_attempt_key_collisions_bump_by_the_smallest_step(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    first = ledger.open_voice_verification(me(), 1)
    second = ledger.open_voice_verification(me(), 1)
    assert second.opened_at == math.nextafter(first.opened_at, math.inf)


# --- attribution ------------------------------------------------------------------------------------


def test_owner_attribution_writes_the_voice_owned_proposal(ledger):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    decision = attribute(ledger, fact)
    assert (decision.reason, decision.profile_id, decision.score_bucket, decision.margin_bucket) == (
        AttributionReason.OWNER, P1, bucket(.95), bucket(.95 - .30))
    owner = rows(ledger, "fact_owners")
    assert owner == [(fact.id, PERSON, "voice_proposal", 1, NOW, "system:voice-attribution", json.dumps(
        {"consent_revision": 2, "link_revision": 1, "margin_bucket": bucket(.95 - .30), "profile_id": P1,
         "profile_revision": 1, "score_bucket": bucket(.95), "v": 1}, sort_keys=True, separators=(",", ":")))]
    assert rows(ledger, "fact_owner_revisions") == [(fact.id, 1, "voice_proposed", 0, NOW)]
    stored = ledger.get(fact.id, review=True)
    assert (stored.state, stored.expires_at, stored.revision) == (FactState.PROPOSED, fact.expires_at, 1)
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    assert attribute(ledger, fact).reason is AttributionReason.FACT_CHANGED  # already owned


def _refusal_world(ledger, case):
    household(ledger)
    if case != "consent_not_active":
        activate(ledger)
    if case == "consent_stale":
        with ledger._transaction() as db:
            db.execute("UPDATE voice_attribution_consents SET verified_link_revision=2")
    if case == "suspended":
        with ledger._transaction() as db:
            db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')", (P1, NOW))
    if case == "top_not_grantable/own_other_profile":
        add_profile(ledger, "00000000-0000-4000-8000-000000000004", PERSON)


REFUSALS = {
    "top_not_grantable": {"snapshot": full(top=P2)},
    "top_not_grantable/own_other_profile": {"snapshot": snap(
        (P1, .3), (P2, .3), (P3, .3), ("00000000-0000-4000-8000-000000000004", .95))},
    "consent_not_active": {}, "consent_not_active/foreign_without_consent": {"snapshot": full(top=P3)},
    "consent_stale": {}, "suspended": {}, "person_unavailable": {"retired": lambda ids: None},
    "incomplete_scoring_set/missing_profile": {"snapshot": snap((P1, .95), (P2, .3))},
    "multi_source": {"sources": 2}, "origin_not_voice": {"origin": MemoryOrigin.AMBIENT},
    "fact_changed": {"confirm": True}, "incomplete_scoring_set": {"snapshot": full(complete=False)},
    "tie": {"snapshot": full(score=.9, rest=.9)}, "below_score": {"snapshot": full(score=.7)},
    "below_margin": {"snapshot": full(score=.9, rest=.85)}, "short_speech": {"snapshot": full(speech=.5)},
    "disabled": {"switch_off": True},
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_every_refusal_leaves_the_proposal_unowned_and_the_proof_published(ledger, case):
    options = REFUSALS[case]
    _refusal_world(ledger, case)
    fact = proposal(ledger, origin=options.get("origin", MemoryOrigin.VOICE), sources=options.get("sources", 1))
    if options.get("confirm"):
        ledger.confirm(fact.id, REVIEWER)
    if options.get("switch_off"):
        ledger.set_voice_switch(False, 1, SWITCH)
    proof = ledger._fact_owner_proof.snapshot
    before = file_bytes(ledger)
    decision = attribute(ledger, fact, options.get("snapshot"),
                         **({"retired": options["retired"]} if "retired" in options else {}))
    assert decision.reason is AttributionReason(case.split("/")[0]) and decision.profile_id is None
    assert file_bytes(ledger) == before
    assert rows(ledger, "fact_owners") == [] and rows(ledger, "fact_owner_revisions") == []
    assert ledger._fact_owner_proof.snapshot.available and ledger._fact_owner_proof.snapshot.users == proof.users


@pytest.mark.parametrize("change", ["profile_revision", "link_revision", "consent", "new_profile", "fact", "triple"])
def test_a_change_between_the_steps_is_fact_changed_and_writes_nothing(ledger, change):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    original = ledger._attribution_read

    def race(db, fact_id, score, retired, active, expected):
        if expected is not None:
            db.execute({"profile_revision": "UPDATE voice_profiles SET revision=2 WHERE id='" + P2 + "'",
                        "link_revision": "UPDATE voice_profile_people SET revision=2 WHERE profile_id='" + P1 + "'",
                        "consent": "UPDATE voice_attribution_consents SET revision=revision+1",
                        "new_profile": "UPDATE voice_profiles SET deleted_at=1 WHERE id='" + P3 + "'",
                        "fact": "UPDATE facts SET revision=2",
                        # Review G: the triple rewritten at the same consent revision (defence in depth).
                        "triple": "UPDATE voice_attribution_consents SET verified_profile_revision=5"}[change])
        return original(db, fact_id, score, retired, active, expected)

    ledger._attribution_read = race
    assert attribute(ledger, fact).reason is AttributionReason.FACT_CHANGED
    assert rows(ledger, "fact_owners") == [] and rows(ledger, "fact_owner_revisions") == []


def test_a_retired_person_between_the_steps_gets_nothing(ledger):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    answers = iter([False, True])
    decision = attribute(ledger, fact, retired=lambda ids: next(answers))
    assert decision.reason is AttributionReason.PERSON_UNAVAILABLE and rows(ledger, "fact_owners") == []


def test_expiry_in_run_deletes_an_owned_voice_proposal(ledger, clock):
    fact = owned_voice(ledger)
    clock.value = fact.expires_at + 1
    assert ledger.expire() == 1
    assert rows(ledger, "facts", "id=?", (fact.id,))[0][1] is None
    assert rows(ledger, "fact_owners") == [] and rows(ledger, "fact_owner_revisions") == []


# --- operator confirmation (Q1 A, R5) ----------------------------------------------------------


def test_operator_confirms_the_same_person_and_clears_the_evidence(ledger, clock):
    fact = owned_voice(ledger)
    refused(ledger, ValueError, "memory_owner_confirm_required", lambda: ledger.confirm(fact.id, REVIEWER))
    clock.value += 60
    receipt = ledger.assign_owner(fact.id, PERSON, 1, 1, OPERATOR, **ALLOW)
    assert (receipt.fact_revision, receipt.owner_revision) == (2, 2)
    stored = ledger.get(fact.id)
    assert (stored.state, stored.expires_at, stored.confirmed_by) == (FactState.CONFIRMED, None, "synthetic-operator")
    assert rows(ledger, "fact_owners") == [(fact.id, PERSON, "voice_proposal", 2, NOW + 60, "synthetic-operator", "{}")]
    assert rows(ledger, "fact_owner_revisions") == [(fact.id, 2, "operator_confirmed", 1, NOW + 60)]
    before = file_bytes(ledger)
    assert ledger.assign_owner(fact.id, PERSON, 2, 2, OPERATOR, **ALLOW).owner_revision == 2  # confirmed: no-op
    # Review H: confirming an already-confirmed owned fact stays an idempotent no-op (writes nothing).
    assert ledger.confirm(fact.id, REVIEWER).revision == 2
    assert file_bytes(ledger) == before


def test_operator_reassignment_of_a_voice_proposal_is_an_ordinary_assignment(ledger):
    fact = owned_voice(ledger)
    ledger.assign_owner(fact.id, OTHER, 1, 1, OPERATOR, **ALLOW)
    assert rows(ledger, "fact_owners")[0][1:3] + rows(ledger, "fact_owners")[0][6:] == (
        OTHER, "operator_assignment", "{}")
    assert rows(ledger, "fact_owner_revisions")[0][2:4] == ("reassigned", 1)


# --- Q5 projection, Q6 Descartar, N5 return counter, R5 ---------------------------------------


READER = MemoryOwnerReader("synthetic-home", PERSON)
MINE = MemoryOwnerActor("synthetic-home", PERSON)


def test_pending_voice_proposal_is_content_free_until_confirmed(ledger):
    fact = owned_voice(ledger)
    [item] = ledger.owned_page(READER).items
    assert item == {"id": fact.id, "kind": "voice_proposal_pending",
                    "created_at": math.floor(NOW / 3600) * 3600.0,
                    "expires_at": math.floor(fact.expires_at / 3600) * 3600.0}
    ledger.assign_owner(fact.id, PERSON, 1, 1, OPERATOR, **ALLOW)
    [item] = ledger.owned_page(READER).items
    assert item["content"] == "Synthetic voice proposal" and item["owner_method"] == "voice_proposal"


def test_descartar_deletes_the_pending_proposal_without_counting(ledger):
    fact = owned_voice(ledger)
    assert ledger.delete_owned(fact.id, MINE) is True
    assert rows(ledger, "facts", "id=?", (fact.id,))[0][1:5] == (None, "fact", 1.0, "deleted")
    assert rows(ledger, "fact_owners") == [] and rows(ledger, "fact_owner_revisions") == []
    assert rows(ledger, "voice_attribution_events") == [] and rows(ledger, "voice_attribution_suspensions") == []
    assert ledger.returned_page(OPERATOR).items == ()


def test_no_es_mio_on_a_pending_proposal_counts_and_goes_to_devueltos(ledger, clock):
    fact = owned_voice(ledger)
    receipt = ledger.return_owned(fact.id, PERSON, MINE)
    assert receipt == ReturnReceipt(True, True) and bool(receipt)
    [event] = rows(ledger, "voice_attribution_events")
    assert event[1:] == (P1, "not_mine", bucket(.95), NOW) and event[0].startswith("vae_")
    [returned] = ledger.returned_page(OPERATOR).items
    assert returned["id"] == fact.id and returned["state"] == "proposed" and returned["expires_at"] == fact.expires_at
    assert ledger.owned_page(READER).items == ()


def test_returning_a_confirmed_voice_fact_does_not_count(ledger):
    fact = owned_voice(ledger)
    ledger.assign_owner(fact.id, PERSON, 1, 1, OPERATOR, **ALLOW)
    assert ledger.return_owned(fact.id, PERSON, MINE) == ReturnReceipt(True, False)
    assert rows(ledger, "voice_attribution_events") == []


def test_ordinary_and_foreign_returns(ledger):
    fact = ledger.confirm(proposal(ledger, "Synthetic ordinary").id, REVIEWER)
    ledger.assign_owner(fact.id, PERSON, fact.revision, 0, OPERATOR, **ALLOW)
    assert ledger.return_owned(fact.id, OTHER, MemoryOwnerActor("h", OTHER)) == ReturnReceipt(False, False)
    assert not ledger.return_owned("fact_" + "0" * 64, PERSON, MINE)
    assert ledger.return_owned(fact.id, PERSON, MINE) == ReturnReceipt(True, False)
    assert rows(ledger, "voice_attribution_events") == []


def _owned_again(ledger):
    fact = proposal(ledger, "Synthetic voice proposal " + uuid4().hex)
    assert attribute(ledger, fact).owner
    return fact


def test_suspension_is_idempotent_and_the_event_insert_is_capped(ledger, clock):
    first = owned_voice(ledger)
    second, third = _owned_again(ledger), _owned_again(ledger)
    ledger.return_owned(first.id, PERSON, MINE)
    assert rows(ledger, "voice_attribution_suspensions") == []
    clock.value += 1
    ledger.return_owned(second.id, PERSON, MINE)
    assert rows(ledger, "voice_attribution_suspensions") == [(P1, 1, NOW + 1, "not_mine_threshold")]
    events = rows(ledger, "voice_attribution_events")
    clock.value += 1
    assert ledger.return_owned(third.id, PERSON, MINE) == ReturnReceipt(True, True)
    assert rows(ledger, "voice_attribution_events") == events and len(events) == 2
    assert rows(ledger, "voice_attribution_suspensions") == [(P1, 1, NOW + 1, "not_mine_threshold")]
    assert attribute(ledger, proposal(ledger, "Synthetic later")).reason is AttributionReason.SUSPENDED
    assert ledger.voice_suspension(P1) == 1
    refused(ledger, ValueError, "memory_voice_suspension_conflict",
            lambda: ledger.resume_voice_attribution(P1, 2, OPERATOR))
    with pytest.raises(PermissionError):
        ledger.resume_voice_attribution(P1, 1, MINE)
    assert ledger.resume_voice_attribution(P1, 1, OPERATOR) is True
    assert rows(ledger, "voice_attribution_suspensions") == [] and rows(ledger, "voice_attribution_events") == []
    assert ledger.voice_suspension(P1) == 0
    assert _owned_again(ledger)


def test_events_past_the_window_do_not_count_before_any_prune(ledger, clock):
    first = owned_voice(ledger)
    ledger.return_owned(first.id, PERSON, MINE)
    clock.value += 31 * DAY
    activate_again = proposal(ledger, "Synthetic late")
    assert attribute(ledger, activate_again).owner
    ledger.return_owned(activate_again.id, PERSON, MINE)
    assert len(rows(ledger, "voice_attribution_events")) == 2
    assert rows(ledger, "voice_attribution_suspensions") == []


@pytest.mark.parametrize("evidence", ["{}", '{"v":1}', "not json",
                                      '{"consent_revision":3,"link_revision":1,"margin_bucket":13,"profile_id":"p",'
                                      '"profile_revision":1,"score_bucket":99,"v":1}'])
def test_malformed_evidence_fails_closed_at_open(tmp_path, clock, evidence):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    fact = owned_voice(ledger)
    ledger._db.execute("UPDATE fact_owners SET evidence=?", (evidence,))
    ledger.close()
    with pytest.raises(RuntimeError, match="fact_owner_value_unknown"):
        _open(tmp_path / "privacy.db", clock)
    assert fact


def test_evidence_on_any_other_owner_change_fails_closed(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    fact = owned_voice(ledger)
    evidence = rows(ledger, "fact_owners")[0][6]
    ledger.assign_owner(fact.id, PERSON, 1, 1, OPERATOR, **ALLOW)
    ledger._db.execute("UPDATE fact_owners SET evidence=?", (evidence,))
    ledger.close()
    with pytest.raises(RuntimeError, match="fact_owner_value_unknown"):
        _open(tmp_path / "privacy.db", clock)


# --- proof (R4), N3 retirement, N4 pruning ------------------------------------------------------


def test_consent_and_attempt_rows_are_owner_references(ledger):
    household(ledger)
    seen = []
    original = ledger._write_consent

    def spy(db, person, *args, **kwargs):
        seen.append((ledger._fact_owner_proof.snapshot.available, ledger.owner_refs_cleared(frozenset({person}))))
        return original(db, person, *args, **kwargs)

    ledger._write_consent = spy
    ledger.request_voice_consent(me(), 0)
    assert seen == [(False, False)]  # withdrawn while the reference is written
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    assert ledger.owner_refs_cleared(frozenset({OTHER}))


def test_a_retired_person_with_only_an_attempt_row_is_pruned(ledger):
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    ledger.open_voice_verification(me(), 1)
    with ledger._transaction() as db:
        db.execute("DELETE FROM voice_attribution_consents")
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    assert ledger.prune_retired_owners(lambda ids: ids == frozenset({PERSON})) == 0
    assert rows(ledger, "voice_verification_attempts") == []
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


def test_n3_cascade_deletes_pending_proposal_consent_and_attempts(ledger):
    fact = owned_voice(ledger)
    assert ledger.clear_user_facts(frozenset({PERSON}), lambda ids: True) == 1
    assert rows(ledger, "facts", "id=?", (fact.id,))[0][1] is None
    for table in ("fact_owners", "fact_owner_revisions", "voice_attribution_consents", "voice_verification_attempts"):
        assert rows(ledger, table) == [], table
    assert ledger.owner_refs_cleared(frozenset({PERSON}))
    voices = VoiceLedger(ledger)
    before = file_bytes(ledger)
    with ledger._transaction() as db:
        assert ledger._retire_voice_people(db, frozenset({PERSON})) == 0  # the clear_user_links hook finds nothing
    assert file_bytes(ledger) == before
    assert voices.people.clear_user_links(frozenset({PERSON}), lambda ids: True) == 1  # only the link remains


def test_clear_user_links_alone_still_deletes_a_pending_voice_proposal(ledger):
    fact = owned_voice(ledger)
    voices = VoiceLedger(ledger)
    assert voices.people.clear_user_links(frozenset({PERSON}), lambda ids: True) == 1
    assert rows(ledger, "facts", "id=?", (fact.id,))[0][4] == "deleted"
    assert rows(ledger, "voice_attribution_consents") == [] and rows(ledger, "voice_verification_attempts") == []
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


def test_the_prune_runs_only_in_run_and_republishes_the_proof(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    fact = owned_voice(ledger)
    ledger.return_owned(fact.id, PERSON, MINE)
    with ledger._transaction() as db:
        db.execute("DELETE FROM voice_attribution_consents")
    ledger.close()
    clock.value += 40 * DAY
    before = (tmp_path / "privacy.db").read_bytes()
    ledger = _open(tmp_path / "privacy.db", clock)  # open never prunes
    try:
        assert (tmp_path / "privacy.db").read_bytes() == before
        assert not ledger.owner_refs_cleared(frozenset({PERSON}))  # only aged attempt rows name the person
        assert ledger.prune_voice_attribution() == 3  # window_opened, passed, not_mine
        assert rows(ledger, "voice_verification_attempts") == [] and rows(ledger, "voice_attribution_events") == []
        assert ledger.owner_refs_cleared(frozenset({PERSON}))
        assert ledger.prune_voice_attribution() == 0
    finally:
        ledger.close()


def test_prune_keeps_rows_inside_their_windows(ledger, clock):
    fact = owned_voice(ledger)
    ledger.return_owned(fact.id, PERSON, MINE)
    clock.value += DAY - 1
    assert ledger.prune_voice_attribution() == 0
    clock.value += 2
    assert ledger.prune_voice_attribution() == 2  # attempts only; the event lives 30 days
    assert len(rows(ledger, "voice_attribution_events")) == 1


# --- invalidation hooks -------------------------------------------------------------------------


def _invalidated_world(ledger):
    fact = owned_voice(ledger)
    confirmed = _owned_again(ledger)
    ledger.assign_owner(confirmed.id, PERSON, 1, 1, OPERATOR, **ALLOW)
    return fact, confirmed


def _assert_invalidated(ledger, fact, confirmed, label):
    [consent] = rows(ledger, "voice_attribution_consents")
    assert consent[1:3] == ("invalidated", 3) and consent[7] == label and consent[3:6] == (None, None, None)
    assert rows(ledger, "fact_owners", "fact_id=?", (fact.id,)) == []
    assert rows(ledger, "fact_owner_revisions", "fact_id=?", (fact.id,))[0][2] == "operator_unassigned"
    assert rows(ledger, "fact_owners", "fact_id=?", (confirmed.id,))[0][1] == PERSON
    assert ledger.returned_page(OPERATOR).items == ()
    assert fact.id not in ledger.eligible_ids(scope="household") and [
        item["id"] for item in ledger.review_page()["proposals"]] == [fact.id]
    assert [item["id"] for item in ledger.owned_page(READER).items] == [confirmed.id]


def test_unlink_invalidates_and_unassigns(ledger):
    fact, confirmed = _invalidated_world(ledger)
    voices = VoiceLedger(ledger)
    with ledger._transaction() as db:
        assert voices.people.unlink_profile(db, P1) == 1
    ledger.republish_owner_proof()
    _assert_invalidated(ledger, fact, confirmed, "system:retired-profile")


@pytest.mark.parametrize("path", ["delete_profile", "prune"])
def test_profile_deletion_invalidates_with_its_label(ledger, path):
    fact, confirmed = _invalidated_world(ledger)
    voices = VoiceLedger(ledger)
    if path == "delete_profile":
        from src.users.voice_ledger import VoiceReviewer
        assert voices.delete_profile(P1, 1, VoiceReviewer("synthetic-voice-operator")) is True
        label = "synthetic-voice-operator"
    else:
        source = json.loads(rows(ledger, "voice_profiles", "id=?", (P1,))[0][5])[0]
        ledger.delete_conversation(source["conversation_id"], REVIEWER)
        voices.prune()
        label = "system:pruned-profile"
    _assert_invalidated(ledger, fact, confirmed, label)
    assert ledger._owner_proof_dirty is False


def test_only_active_consents_on_that_profile_are_invalidated(ledger):
    household(ledger)
    activate(ledger)
    ledger.request_voice_consent(me(OTHER), 0)
    with ledger._transaction() as db:
        assert ledger._invalidate_voice_consents(db, P3, "system:retired-profile") == 0
        assert ledger._invalidate_voice_consents(db, P2, "system:retired-profile") == 0
    assert [row[1] for row in rows(ledger, "voice_attribution_consents", "1 ORDER BY person_id")] == [
        "active", "pending_verification"]


# --- Q7 flag-off exposure and production pins ---------------------------------------------------


def test_flag_off_with_seeded_voice_rows_keeps_every_exposure_closed(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock)
    ledger.set_voice_switch(True, 0, SWITCH)
    fact = owned_voice(ledger)
    ledger.close()
    ledger = _open(tmp_path / "privacy.db", clock, VoiceProposalSettings())
    try:
        assert not ledger.voice_switch().effective
        assert ledger.owned_page(READER).items[0]["kind"] == "voice_proposal_pending"
        refused(ledger, ValueError, "memory_owner_confirm_required", lambda: ledger.confirm(fact.id, REVIEWER))
        assert attribute(ledger, proposal(ledger, "Synthetic off")).reason is AttributionReason.DISABLED
        assert ledger.return_owned(fact.id, PERSON, MINE) == ReturnReceipt(True, True)
    finally:
        ledger.close()


_DARK = {"attribute_voice_proposal", "verify_voice_consent", "open_voice_verification", "request_voice_consent",
         "revoke_voice_consent", "reopen_voice_consent", "set_voice_switch", "resume_voice_attribution",
         "voice_consent", "voice_switch", "voice_suspension", "VoiceAttributionCapability",
         "MemoryVoiceConsentActor", "VoiceSwitchActor", "memory:own:voice", "open_voice_window"}
_ALLOWED = {"src/memory/fact_ledger.py", "src/memory/voice_attribution.py"}


def test_the_voice_stage_has_exactly_the_4a_2_production_callers():
    """4a-2 wires routes only: no attributor, verifier or capability construction until 4b.

    Pinned per name as a count of referencing lines (4a-2 review P), so a second call site of an
    already-listed writer fails too.
    """
    from collections import Counter
    from tests.unit.memory.test_fact_owners import _references
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        if relative in _ALLOWED:
            continue
        uses = _references(path.read_text(encoding="utf-8"), _DARK)
        if uses:
            found[relative] = dict(Counter(use.split(":", 1)[1] for use in uses))
    assert found == {
        "src/dashboard/authority.py": {"'memory:own:voice'": 1},
        "src/dashboard/memory_voice_api.py": {
            "'memory:own:voice'": 1, "MemoryVoiceConsentActor": 6, "VoiceSwitchActor": 1,
            "import MemoryVoiceConsentActor": 1, "import VoiceSwitchActor": 1, "open_voice_window": 1,
            "reopen_voice_consent": 1, "request_voice_consent": 1, "resume_voice_attribution": 1,
            "revoke_voice_consent": 1, "set_voice_switch": 1, "voice_switch": 1},
    }


def test_the_hooks_and_the_prune_have_exactly_the_wired_callers():
    from tests.unit.memory.test_fact_owners import _references
    names = {"_invalidate_voice_consents", "_retire_voice_people", "republish_owner_proof",
             "prune_voice_attribution", "VoiceProposalSettings"}
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        if relative in _ALLOWED:
            continue
        uses = sorted({use.split(":", 1)[1] for use in _references(path.read_text(encoding="utf-8"), names)})
        if uses:
            found[relative] = uses
    assert found == {
        "src/memory/privacy_services.py": ["VoiceProposalSettings", "import VoiceProposalSettings",
                                           "prune_voice_attribution"],
        "src/users/voice_ledger.py": ["republish_owner_proof"],
        "src/users/voice_people.py": ["_invalidate_voice_consents", "_retire_voice_people", "republish_owner_proof"],
    }


def test_the_gate_never_uses_the_identification_path():
    for relative in _ALLOWED:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names & {"identify", "compute_similarity", "SpeakerIdentifier"}, relative


def test_owner_rows_keep_their_closed_shape(ledger):
    fact = owned_voice(ledger)
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT method, assigned_by FROM fact_owners WHERE fact_id=?", (fact.id,)).fetchone() == (
            "voice_proposal", "system:voice-attribution")


def test_only_the_writing_attribution_step_withdraws_the_owner_proof(ledger):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    seen, original = [], ledger._attribution_read

    def spy(db, *args):
        seen.append(ledger._fact_owner_proof.snapshot.available)
        return original(db, *args)

    ledger._attribution_read = spy
    assert attribute(ledger, fact).owner
    assert seen == [True, False]
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))


def test_the_verified_snapshot_carries_the_actual_revisions(ledger):
    """The consent stores the verified profile and link revisions it was checked at, not defaults."""
    add_profile(ledger, P1, PERSON, revision=3, link_revision=2)
    add_profile(ledger, P2)
    ledger.request_voice_consent(me(), 0)
    binding = ledger.open_voice_verification(me(), 1)
    assert (binding.profile_revision, binding.link_revision) == (3, 2)
    snapshot = snap((P1, .95), (P2, .3), revisions={P1: 3})
    assert ledger.verify_voice_consent(binding, scorer(snapshot), CAP, **ALLOW).outcome is VoiceVerificationOutcome.PASSED
    assert rows(ledger, "voice_attribution_consents")[0][3:6] == (P1, 3, 2)
    fact = proposal(ledger)
    assert attribute(ledger, fact, snapshot).owner
    evidence = json.loads(rows(ledger, "fact_owners")[0][6])
    assert (evidence["profile_revision"], evidence["link_revision"], evidence["consent_revision"]) == (3, 2, 2)


# --- 4a-1 independent review fixes ------------------------------------------------------------


def _lock(ledger):
    for _ in range(3):
        result = verify(ledger, full(score=.5))
    assert result.state is VoiceConsentState.VERIFICATION_LOCKED
    return ledger.voice_consent(me())


def test_a_reopen_restores_a_full_set_of_attempts(ledger):
    """Review A: only failures after the last operator reopen count."""
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    locked = _lock(ledger)
    reopened = ledger.reopen_voice_consent(PERSON, locked.revision, OPERATOR)
    assert reopened.state is VoiceConsentState.PENDING_VERIFICATION
    assert rows(ledger, "voice_verification_attempts", "outcome='failed'") == []
    assert len(rows(ledger, "voice_verification_attempts", "outcome='window_opened'")) == 3  # rate limit kept
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full(score=.5)).state is VoiceConsentState.VERIFICATION_LOCKED


def test_revoke_and_request_cannot_lift_a_lock(ledger):
    """Review B: locked -> revoke -> request -> good voice stays refused; only an operator lifts it."""
    household(ledger)
    ledger.request_voice_consent(me(), 0)
    locked = _lock(ledger)
    revoked = ledger.revoke_voice_consent(me(), locked.revision)
    assert (revoked.state, revoked.revision) == (VoiceConsentState.VERIFICATION_LOCKED, locked.revision + 1)
    assert rows(ledger, "voice_attribution_consents")[0][7] == "synthetic-home"  # the revocation is recorded
    refused(ledger, ValueError, "memory_voice_consent_locked",
            lambda: ledger.request_voice_consent(me(), revoked.revision))
    refused(ledger, ValueError, "memory_voice_consent_state",
            lambda: ledger.open_voice_verification(me(), revoked.revision))
    with pytest.raises(ValueError):
        verify(ledger, full())
    assert ledger.voice_consent(me()).state is VoiceConsentState.VERIFICATION_LOCKED
    # The operator reopen lifts it into 'revoked' (the person's last choice), never into pending.
    assert ledger.reopen_voice_consent(PERSON, revoked.revision, OPERATOR).state is VoiceConsentState.REVOKED
    assert ledger.request_voice_consent(me(), revoked.revision + 1).state is VoiceConsentState.PENDING_VERIFICATION
    assert verify(ledger, full()).outcome is VoiceVerificationOutcome.PASSED


def test_a_live_profile_without_an_embedding_makes_the_set_incomplete(ledger):
    """Review D: the profile stays in the scoring set and every decision is refused, never narrowed."""
    household(ledger)
    activate(ledger)
    ledger.request_voice_consent(me(OTHER), 0)
    fact = proposal(ledger)
    with ledger._transaction() as db:
        db.execute("UPDATE voice_profiles SET embedding=NULL WHERE id=?", (P2,))
    assert P2 in ledger._live_voice_profiles(ledger._db)
    assert attribute(ledger, fact).reason is AttributionReason.INCOMPLETE_SCORING_SET
    without = ScoringSnapshot("enc_synthetic", tuple(item for item in full().scores if item.profile_id != P2), True, 3.0)
    assert attribute(ledger, fact, without).reason is AttributionReason.INCOMPLETE_SCORING_SET
    assert rows(ledger, "fact_owners") == []
    refused(ledger, ValueError, "memory_voice_scoring_incomplete", lambda: ledger.open_voice_verification(me(OTHER), 1))


def test_an_embedding_dropped_between_the_steps_writes_nothing(ledger):
    household(ledger)
    activate(ledger)
    fact = proposal(ledger)
    original = ledger._attribution_read

    def race(db, fact_id, score, retired, active, expected):
        if expected is not None:
            db.execute("UPDATE voice_profiles SET embedding=NULL WHERE id=?", (P2,))
        return original(db, fact_id, score, retired, active, expected)

    ledger._attribution_read = race
    assert attribute(ledger, fact).reason is AttributionReason.FACT_CHANGED
    assert rows(ledger, "fact_owners") == []


@pytest.mark.parametrize("top, second, expected", [
    (.75, .50, AttributionReason.OWNER), (.75 - 2 ** -20, .25, AttributionReason.BELOW_SCORE),
    (1.0, .75, AttributionReason.OWNER), (1.0, .75 + 2 ** -20, AttributionReason.BELOW_MARGIN),
])
def test_gate_boundaries_are_inclusive_at_the_thresholds(top, second, expected):
    """Review E: acceptance is score >= min_score and margin >= min_margin (exactly representable values)."""
    from src.memory.voice_attribution import evaluate_gate
    settings = VoiceProposalSettings(enabled=True, min_score=.75, min_margin=.25)
    decision = evaluate_gate(snap((P1, top), (P2, second)), settings, lambda profile: None)
    assert decision.reason is expected


def test_the_ledger_applies_the_same_inclusive_boundaries(tmp_path, clock):
    ledger = _open(tmp_path / "privacy.db", clock, VoiceProposalSettings(enabled=True, min_score=.75, min_margin=.25))
    try:
        ledger.set_voice_switch(True, 0, SWITCH)
        household(ledger)
        ledger.request_voice_consent(me(), 0)
        assert verify(ledger, full(score=.75, rest=.5)).outcome is VoiceVerificationOutcome.PASSED
        assert attribute(ledger, proposal(ledger), full(score=.75, rest=.5)).owner
    finally:
        ledger.close()


def test_profile_liveness_is_one_definition_with_the_voice_ledger(ledger):
    """Review F: the scoring set and VoiceLedger._live(confirmed=True) agree, including a revoked source."""
    household(ledger)
    variants = {
        "revoked_source": None, "deleted": "UPDATE voice_profiles SET deleted_at=1 WHERE id=?",
        "unconfirmed": "UPDATE voice_profiles SET confirmed_by='' WHERE id=?",
        "future": "UPDATE voice_profiles SET confirmed_at=confirmed_at+999 WHERE id=?",
        "bad_sources": "UPDATE voice_profiles SET sources_json='[]' WHERE id=?",
        "no_embedding": "UPDATE voice_profiles SET embedding=NULL WHERE id=?",
    }
    extra = {}
    for index, (name, sql) in enumerate(variants.items()):
        profile = "00000000-0000-4000-8000-0000000001%02x" % index
        add_profile(ledger, profile)
        extra[name] = profile
        if sql is not None:
            with ledger._transaction() as db:
                db.execute(sql, (profile,))
    source = json.loads(rows(ledger, "voice_profiles", "id=?", (extra["revoked_source"],))[0][5])[0]
    ledger.delete_conversation(source["conversation_id"], REVIEWER)
    voices = VoiceLedger(ledger)
    with ledger._transaction() as db:
        live = ledger._live_voice_profiles(db)
        everything = db.execute("SELECT * FROM voice_profiles").fetchall()
        assert {row["id"] for row in everything if voices._live(row, confirmed=True)} == set(live)
    assert set(live) == {P1, P2, P3, extra["no_embedding"]}


def test_a_malformed_subject_is_refused_not_relabelled(ledger):
    """Review I: the real subject is stored or the write is refused."""
    household(ledger)
    for subject in ("x" * 129, "bad\nsubject"):
        refused(ledger, PermissionError, "memory_voice_subject_invalid",
                lambda subject=subject: ledger.set_voice_switch(False, 1, VoiceSwitchActor(subject)))
        refused(ledger, PermissionError, "memory_voice_subject_invalid",
                lambda subject=subject: ledger.request_voice_consent(MemoryVoiceConsentActor(subject, PERSON), 0))
    assert ledger.request_voice_consent(MemoryVoiceConsentActor("s" * 128, PERSON), 0).revision == 1
    assert rows(ledger, "voice_attribution_consents")[0][7] == "s" * 128
