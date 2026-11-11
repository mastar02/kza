"""U3b-4a-2 review T/W/L/C: verification windows, their binding, and the operator listing.

Real FactLedger files with injected scores, reusing the 4a-1 fixtures.
"""

from types import SimpleNamespace

import pytest

from src.memory import fact_ledger as fact_ledger_module
from src.memory.voice_attribution import VerificationWindow, VoiceConsentState, VoiceVerificationWindows
from tests.unit.memory.test_voice_attribution_ledger import (
    ALLOW, CAP, OPERATOR, OTHER, P1, P2, P3, PERSON, SWITCH, Clock, _open, add_profile, full, household, me, scorer,
)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(tmp_path, clock):
    value = _open(tmp_path / "privacy.db", clock)
    value.set_voice_switch(True, 0, SWITCH)
    try:
        yield value
    finally:
        value.close()



def available(room_id):
    return room_id == "living"


def attempts(ledger, person=PERSON):
    return ledger._db.execute("SELECT COUNT(*) FROM voice_verification_attempts WHERE person_id=?",
                              (person,)).fetchone()[0]


def pending(ledger, person=PERSON):
    ledger.request_voice_consent(me(person), 0)
    return ledger.voice_consent(me(person)).revision


# --- the registry ------------------------------------------------------------------------------


def binding(person=PERSON, profile=P1):
    return SimpleNamespace(person_id=person, profile_id=profile)


def test_window_lasts_exactly_sixty_seconds(clock):
    windows = VoiceVerificationWindows(clock)
    assert VoiceVerificationWindows.WINDOW_S == 60.0
    window = windows.open(binding(), "living")
    assert window == VerificationWindow(window.binding, "living", clock.value + 60.0, None)
    clock.value += 59.999
    assert windows.current(PERSON) is window
    clock.value += 0.001
    assert windows.current(PERSON) is None
    clock.value -= 1  # an expired window is dropped, not hidden
    assert windows.current(PERSON) is None


def test_one_window_per_person_replace_close_profile_and_clear(clock):
    windows = VoiceVerificationWindows(clock)
    first = windows.open(binding(), "living")
    second = windows.open(binding(), "kitchen")
    assert windows.current(PERSON) is second and second is not first
    windows.open(binding(OTHER, P3), "living")
    windows.close_profile(P1)
    assert windows.current(PERSON) is None and windows.current(OTHER) is not None
    windows.close(OTHER)
    assert windows.current(OTHER) is None
    windows.open(binding(), "living")
    windows.open(binding(OTHER, P3), "living")
    windows.clear()
    assert windows.current(PERSON) is None and windows.current(OTHER) is None


# --- the ledger window layer (review R, W, T) ---------------------------------------------------


def test_open_voice_window_registers_a_bound_window(ledger):
    household(ledger)
    revision = pending(ledger)
    window = ledger.open_voice_window(me(), revision, "living", available)
    assert window.room_id == "living" and window.binding.profile_id == P1
    assert ledger.voice_status(me())["window"] is window and attempts(ledger) == 1


@pytest.mark.parametrize("room", [None, lambda room_id: False, lambda room_id: "yes", lambda room_id: 1,
                                  lambda room_id: 1 / 0, "not callable"])
def test_a_room_refusal_counts_and_leaves_no_window(ledger, room):
    household(ledger)
    revision = pending(ledger)
    assert ledger.open_voice_window(me(), revision, "living", room) is None
    assert attempts(ledger) == 1 and ledger.voice_status(me())["window"] is None


def test_room_refusals_consume_the_daily_limit(ledger):
    household(ledger)
    revision = pending(ledger)
    for _ in range(6):
        assert ledger.open_voice_window(me(), revision, "living", lambda room_id: False) is None
    with pytest.raises(PermissionError, match="memory_voice_rate_limited"):
        ledger.open_voice_window(me(), revision, "living", available)
    assert ledger.voice_status(me())["window"] is None


@pytest.mark.parametrize("setup,expected", [
    ("none", "memory_voice_consent_state"), ("active", "memory_voice_consent_state"),
    ("locked", "memory_voice_consent_state"), ("stale", "memory_voice_consent_conflict"),
    ("off", "memory_voice_disabled")])
def test_the_room_is_never_read_before_the_consent_checks(ledger, setup, expected):
    household(ledger)
    revision = 0
    if setup != "none":
        revision = pending(ledger)
    if setup == "active":
        binding_ = ledger.open_voice_verification(me(), revision)
        assert ledger.verify_voice_consent(binding_, scorer(full()), CAP, **ALLOW).state is VoiceConsentState.ACTIVE
        revision = ledger.voice_consent(me()).revision
    if setup == "locked":
        for _ in range(3):
            binding_ = ledger.open_voice_verification(me(), revision)
            ledger.verify_voice_consent(binding_, scorer(full(top=P2)), CAP, **ALLOW)
        assert ledger.voice_consent(me()).state is VoiceConsentState.VERIFICATION_LOCKED
        revision = ledger.voice_consent(me()).revision
    if setup == "stale":
        revision -= 1
    if setup == "off":
        ledger.set_voice_switch(False, 1, SWITCH)
    before = attempts(ledger)
    reads = []
    for room in (lambda room_id: reads.append(room_id) or True, lambda room_id: reads.append(room_id) or False):
        with pytest.raises(ValueError, match=expected):
            ledger.open_voice_window(me(), revision, "living", room)
    assert reads == [] and attempts(ledger) == before


def test_a_ledger_refusal_registers_no_window(ledger):
    household(ledger)
    revision = pending(ledger)
    with pytest.raises(ValueError, match="memory_voice_consent_conflict"):
        ledger.open_voice_window(me(), revision + 1, "living", available)
    assert ledger.voice_windows.current(PERSON) is None


def test_a_failure_after_registration_rolls_back_and_closes_the_window(ledger, monkeypatch):
    household(ledger)
    revision = pending(ledger)
    original = ledger.voice_windows.open

    def register_then_fail(binding_, room_id):
        original(binding_, room_id)
        raise RuntimeError("synthetic failure after registration")
    monkeypatch.setattr(ledger.voice_windows, "open", register_then_fail)
    with pytest.raises(RuntimeError):
        ledger.open_voice_window(me(), revision, "living", available)
    assert ledger.voice_windows.current(PERSON) is None and attempts(ledger) == 0


def test_revocation_after_opening_closes_the_window(ledger):
    household(ledger)
    revision = pending(ledger)
    ledger.open_voice_window(me(), revision, "living", available)
    ledger.revoke_voice_consent(me(), revision)
    assert ledger.voice_windows.current(PERSON) is None


def test_status_never_shows_a_window_whose_binding_no_longer_holds(ledger):
    """Probe W2 order: binding, revoke, then a registration; status closes it."""
    household(ledger)
    revision = pending(ledger)
    stale = ledger.open_voice_verification(me(), revision)
    ledger.revoke_voice_consent(me(), revision)
    ledger.voice_windows.open(stale, "living")
    assert ledger.voice_status(me())["window"] is None
    assert ledger.voice_windows.current(PERSON) is None


@pytest.mark.parametrize("change", ["unlink", "relink_other", "profile_revision", "suspend", "switch_off",
                                    "delete_profile", "retire"])
def test_a_pending_window_closes_when_its_binding_changes(ledger, change):
    """Probe W1: invalidating writes close (or status hides) a pending person's window."""
    household(ledger)
    revision = pending(ledger)
    ledger.open_voice_window(me(), revision, "living", available)
    with ledger._transaction() as db:
        if change in ("unlink", "relink_other"):
            ledger._invalidate_voice_consents(db, P1, "synthetic-operator")
            db.execute("UPDATE voice_profile_people SET user_id=?, revision=revision+1 WHERE profile_id=?",
                       (None if change == "unlink" else OTHER, P1))
        elif change == "profile_revision":
            db.execute("UPDATE voice_profiles SET revision=revision+1 WHERE id=?", (P1,))
        elif change == "suspend":
            db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')",
                       (P1, ledger.clock()))
        elif change == "delete_profile":
            db.execute("UPDATE voice_profiles SET deleted_at=? WHERE id=?", (ledger.clock(), P1))
        elif change == "retire":
            ledger._retire_voice_people(db, frozenset({PERSON}))
    if change == "switch_off":
        ledger.set_voice_switch(False, 1, SWITCH)
    if change in ("unlink", "relink_other", "retire", "switch_off"):
        assert ledger.voice_windows.current(PERSON) is None  # closed by the writer itself
    if change != "retire":
        assert ledger.voice_status(me())["window"] is None
    assert ledger.voice_windows.current(PERSON) is None


def test_a_window_from_an_earlier_pending_consent_is_never_shown(ledger):
    """Revoke and request again: same profile and link, new consent revision; the old binding is stale."""
    household(ledger)
    revision = pending(ledger)
    stale = ledger.open_voice_verification(me(), revision)
    ledger.revoke_voice_consent(me(), revision)
    ledger.request_voice_consent(me(), revision + 1)
    assert ledger.voice_consent(me()).state is VoiceConsentState.PENDING_VERIFICATION
    ledger.voice_windows.open(stale, "living")
    assert ledger.voice_status(me())["window"] is None and ledger.voice_windows.current(PERSON) is None


def test_a_window_is_not_shown_while_the_settings_flag_is_off(ledger):
    """The switch row stays on; only the settings flag is off (no switch write to clear the registry)."""
    from src.memory.voice_attribution import VoiceProposalSettings
    household(ledger)
    revision = pending(ledger)
    ledger.open_voice_window(me(), revision, "living", available)
    ledger.voice_settings = VoiceProposalSettings()
    assert ledger.voice_status(me())["window"] is None and ledger.voice_windows.current(PERSON) is None


def test_another_persons_change_keeps_my_window(ledger):
    household(ledger)
    revision = pending(ledger)
    window = ledger.open_voice_window(me(), revision, "living", available)
    with ledger._transaction() as db:
        ledger._invalidate_voice_consents(db, P3, "synthetic-operator")
    assert ledger.voice_status(me())["window"] is window


def test_the_route_module_never_touches_the_registry():
    """W: windows are opened only under the ledger transaction (open_voice_window)."""
    from tests.unit.memory.test_voice_attribution_ledger import ROOT
    assert "voice_windows" not in (ROOT / "src/dashboard/memory_voice_api.py").read_text(encoding="utf-8")


# --- the operator listing (review L, C) ---------------------------------------------------------


def test_listing_counts_only_live_linked_profiles_and_hides_deleted_names(ledger):
    household(ledger)
    binding_ = ledger.open_voice_verification(me(), pending(ledger))
    ledger.verify_voice_consent(binding_, scorer(full()), CAP, **ALLOW)
    unreviewed = "00000000-0000-4000-8000-0000000000aa"
    add_profile(ledger, unreviewed, PERSON)
    with ledger._transaction() as db:
        db.execute("UPDATE voice_profiles SET confirmed_at=? WHERE id=?", (ledger.clock() + 3600, unreviewed))
    listing = ledger.voice_consents(OPERATOR)
    assert [(c["person_id"], c["linked_live_profiles"], c["verified_profile"]) for c in listing["consents"]] == [
        (PERSON, 1, {"id": P1, "name": "Synthetic 1"})]
    assert listing["multiple_profiles"] == [] and listing["truncated"] is False
    with ledger._transaction() as db:
        db.execute("UPDATE voice_profiles SET deleted_at=? WHERE id=?", (ledger.clock(), P1))
    listing = ledger.voice_consents(OPERATOR)
    assert listing["consents"][0]["verified_profile"] == {"id": P1, "name": None}
    assert listing["consents"][0]["linked_live_profiles"] == 0


def test_listing_is_bounded(ledger, monkeypatch):
    household(ledger)
    pending(ledger)
    pending(ledger, OTHER)
    monkeypatch.setattr(fact_ledger_module, "VOICE_CONSENT_LISTING_MAX", 1)
    listing = ledger.voice_consents(OPERATOR)
    assert [c["person_id"] for c in listing["consents"]] == [PERSON] and listing["truncated"] is True
    monkeypatch.setattr(fact_ledger_module, "VOICE_CONSENT_LISTING_MAX", 2)
    assert ledger.voice_consents(OPERATOR)["truncated"] is False
