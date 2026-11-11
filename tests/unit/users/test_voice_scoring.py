"""U3b-4b-1: the real scorer reads stored embeddings inside the ledger transaction (synthetic only).

Profiles come from VoiceLedger.confirm over three synthetic fingerprints
each; vectors are 192-dimension float32 unit vectors built from an
orthonormal basis, so every cosine is exact up to float32 rounding. No model,
no audio, no GPU. Person links are written as the operator CAS row would be.
"""

import json
from uuid import uuid4

import numpy as np
import pytest

from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource, MemoryVoiceConsentActor, VoiceSwitchActor
from src.memory.voice_attribution import (
    AttributionReason, ProfileScore, ScoringSnapshot, VoiceAttributionCapability, VoiceConsentState,
    VoiceProposalSettings, VoiceVerificationOutcome, evaluate_gate,
)
from src.users.voice_ledger import VoiceLedger, VoiceReviewer, fingerprint_identity
from src.users.voice_scoring import FingerprintScorer, UnavailableScorer

NOW = 2_000_000_000.0
MODEL = "enc_synthetic-scoring"
CAP = VoiceAttributionCapability("synthetic-scoring")
ACTOR = VoiceReviewer("synthetic-admin")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
PERSON, OTHER = "user_" + "a" * 32, "user_" + "b" * 32
ON = VoiceProposalSettings(enabled=True)


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


def unit(**weights) -> np.ndarray:
    """sum(w_i e_i) plus the remainder on e_191, so the result is a unit vector with exact cosines w_i."""
    vector = np.zeros(192, dtype=np.float64)
    for name, weight in weights.items():
        vector[int(name[1:])] = weight
    rest = 1.0 - float(np.dot(vector, vector))
    assert rest >= -1e-12
    vector[191] = max(0.0, rest) ** 0.5
    return vector.astype(np.float32)


def source(clock, origin=MemoryOrigin.VOICE) -> MemorySource:
    return MemorySource(str(uuid4()), str(uuid4()), clock() - 1, clock() + 47 * 3600, origin)


class House:
    def __init__(self, tmp_path, settings=ON):
        self.clock = Clock()
        self.ledger = FactLedger(str(tmp_path / "privacy.db"), clock=self.clock, voice_settings=settings)
        self.ledger.initialize()
        self.ledger.set_voice_switch(True, 0, VoiceSwitchActor("synthetic-operator"))
        self.voices = VoiceLedger(self.ledger)

    def profile(self, vector, name, person=None, *, model=MODEL):
        identities = tuple(self.voices.put(model, vector.copy(), 2.0, (source(self.clock),)) for _ in range(3))
        profile = self.voices.confirm(identities, name, str(uuid4()), ACTOR)["id"]
        if person is not None:
            with self.ledger._transaction() as db:
                db.execute("INSERT INTO voice_profile_people VALUES (?,?,1,?,'synthetic-reviewer')",
                           (profile, person, self.clock()))
        return profile

    def utterance(self, vector, duration=2.0, origin=MemoryOrigin.VOICE):
        capture = source(self.clock, origin)
        return self.voices.put(MODEL, vector.copy(), duration, (capture,)), capture

    def scorer(self, vector, duration=2.0):
        identity, capture = self.utterance(vector, duration)
        return FingerprintScorer(self.ledger, identity, (capture,)), capture

    def activate(self, person, vector):
        actor = MemoryVoiceConsentActor("synthetic-home", person)
        self.ledger.request_voice_consent(actor, self.ledger.voice_consent(actor).revision)
        result = self.verify(person, vector)
        assert result.outcome is VoiceVerificationOutcome.PASSED, result
        return result

    def verify(self, person, vector, duration=2.0):
        actor = MemoryVoiceConsentActor("synthetic-home", person)
        binding = self.ledger.open_voice_verification(actor, self.ledger.voice_consent(actor).revision)
        scorer, _ = self.scorer(vector, duration)
        return self.ledger.verify_voice_consent(binding, scorer, CAP, **ALLOW, review_fingerprint=scorer.fingerprint_id)

    def attribute(self, vector, duration=2.0):
        scorer, capture = self.scorer(vector, duration)
        fact = self.ledger.store("Propuesta sintética", "fact", 1.0, (capture,), proposed=True)
        return self.ledger.attribute_voice_proposal(fact.id, scorer, CAP, **ALLOW), fact

    def owner(self, fact):
        with self.ledger._transaction() as db:
            row = db.execute("SELECT person_id FROM fact_owners WHERE fact_id=?", (fact.id,)).fetchone()
        return None if row is None else row[0]


@pytest.fixture
def house(tmp_path):
    value = House(tmp_path)
    try:
        yield value
    finally:
        value.ledger.close()


@pytest.fixture
def household(house):
    """A (PERSON, active, e0), B (OTHER, linked, not enrolled, e1), X (unlinked, 0.8 e0 + 0.6 e2: close to A)."""
    house.a = house.profile(unit(e0=1.0), "Voz A", PERSON)
    house.b = house.profile(unit(e1=1.0), "Voz B", OTHER)
    house.x = house.profile(unit(e0=0.8, e2=0.6), "Voz X")
    house.activate(PERSON, unit(e0=0.95, e1=0.2))
    return house


def test_the_snapshot_is_the_raw_cosine_of_stored_unit_vectors_against_every_live_profile(household):
    scorer, _ = household.scorer(unit(e0=0.9, e1=0.3, e2=-0.2), duration=2.5)  # X = 0.72 - 0.12
    with household.ledger._transaction() as db:
        snapshot = scorer(db)
    assert snapshot.model_key == MODEL and snapshot.complete and snapshot.speech_s == 2.5
    assert [item.profile_id for item in snapshot.scores] == sorted([household.a, household.b, household.x])
    scores = {item.profile_id: item.score for item in snapshot.scores}
    assert scores[household.a] == pytest.approx(0.9, abs=1e-6)
    assert scores[household.b] == pytest.approx(0.3, abs=1e-6)
    assert scores[household.x] == pytest.approx(0.6, abs=1e-6)
    assert all(item.revision == 1 for item in snapshot.scores)


@pytest.mark.parametrize(("vector", "duration", "reason"), [
    (unit(e0=0.9, e1=0.3), 2.0, AttributionReason.OWNER),
    (unit(e0=0.79, e1=0.3), 2.0, AttributionReason.BELOW_SCORE),
    (unit(e0=0.95, e2=0.25), 2.0, AttributionReason.BELOW_MARGIN),  # A .95, X .91
    (unit(e0=0.6, e1=0.6), 2.0, AttributionReason.TIE),
    (unit(e2=0.95, e0=0.2), 2.0, AttributionReason.TOP_NOT_GRANTABLE),  # X .73 on top
    (unit(e1=0.95, e0=0.2), 2.0, AttributionReason.CONSENT_NOT_ACTIVE),
    (unit(e0=0.9, e1=0.3), 1.0, AttributionReason.SHORT_SPEECH),
])
def test_attribution_matrix_never_names_another_person(household, vector, duration, reason):
    decision, fact = household.attribute(vector, duration)
    assert decision.reason is reason
    assert household.owner(fact) == (PERSON if reason is AttributionReason.OWNER else None)


def test_the_persons_own_second_profile_as_runner_up_below_the_margin_is_no_owner(household):
    """R14: the margin is checked against every other profile, the person's own included."""
    household.profile(unit(e0=0.9, e3=0.43), "Voz A dos", PERSON)
    decision, fact = household.attribute(unit(e0=0.97, e3=0.2))  # A .97, A dos .96
    assert decision.reason is AttributionReason.BELOW_MARGIN and household.owner(fact) is None


@pytest.mark.parametrize(("score", "reason"), [(0.81, AttributionReason.OWNER), (0.79, AttributionReason.BELOW_SCORE)])
def test_with_two_profiles_a_guest_like_vector_is_bounded_by_min_score_alone(house, score, reason):
    """A guest has no profile: far from both, so the margin passes and only min_score decides."""
    house.a = house.profile(unit(e0=1.0), "Voz A", PERSON)
    house.x = house.profile(unit(e2=1.0), "Voz X")
    house.activate(PERSON, unit(e0=0.95))
    decision, fact = house.attribute(unit(e0=score, e5=0.4))
    assert decision.reason is reason


def test_another_model_key_or_dimension_makes_the_set_incomplete(household):
    with household.ledger._transaction() as db:
        db.execute("UPDATE voice_profiles SET model_key='enc_other' WHERE id=?", (household.x,))
    decision, fact = household.attribute(unit(e0=0.9))
    assert decision.reason is AttributionReason.INCOMPLETE_SCORING_SET and household.owner(fact) is None
    scorer, _ = household.scorer(unit(e0=0.9))
    with household.ledger._transaction() as db:
        snapshot = scorer(db)
    assert snapshot.complete is False and household.x not in {item.profile_id for item in snapshot.scores}


def test_an_invalid_stored_embedding_makes_the_set_incomplete(household):
    with household.ledger._transaction() as db:
        db.execute("UPDATE voice_profiles SET embedding=? WHERE id=?", (np.full(192, 3.0, dtype="<f4").tobytes(), household.x))
    decision, _ = household.attribute(unit(e0=0.9))
    assert decision.reason is AttributionReason.INCOMPLETE_SCORING_SET


def test_a_missing_foreign_or_expired_fingerprint_is_no_fingerprint(household):
    identity, capture = household.utterance(unit(e0=0.9))
    other = source(household.clock)
    fact = household.ledger.store("Propuesta sintética", "fact", 1.0, (capture,), proposed=True)
    for scorer in (FingerprintScorer(household.ledger, "vf_" + "0" * 64, (capture,)),
                   FingerprintScorer(household.ledger, identity, (other,)),  # another capture's fingerprint
                   UnavailableScorer()):
        decision = household.ledger.attribute_voice_proposal(fact.id, scorer, CAP, **ALLOW)
        assert decision.reason is AttributionReason.NO_FINGERPRINT
    assert identity == fingerprint_identity(MODEL, (capture,))
    household.clock.value += 48 * 3600
    with household.ledger._transaction() as db:
        with pytest.raises(LookupError, match="voice_fingerprint_unavailable"):
            FingerprintScorer(household.ledger, identity, (capture,))(db)


def test_a_dead_utterance_source_or_an_expired_fingerprint_alone_is_no_fingerprint(household):
    """Review W10/W11: each condition on its own, with the other still holding."""
    dead, dead_capture = household.utterance(unit(e0=0.9))
    expired, expired_capture = household.utterance(unit(e0=0.9))
    with household.ledger._transaction() as db:
        db.execute("INSERT INTO conversation_tombstones VALUES (?,?,?)",
                   (dead_capture.conversation_id, household.clock(), "synthetic-erasure"))
        db.execute("UPDATE voice_fingerprints SET expires_at=? WHERE id=?", (household.clock() - 1, expired))
        assert tuple(db.execute("SELECT expires_at > ? FROM voice_fingerprints WHERE id=?",
                          (household.clock(), dead)).fetchone()) == (1,), "the dead source's fingerprint is unexpired"
        assert household.ledger._sources_live(db, (expired_capture,)), "the expired fingerprint's source is live"
        for identity, capture in ((dead, dead_capture), (expired, expired_capture)):
            with pytest.raises(LookupError, match="voice_fingerprint_unavailable"):
                FingerprintScorer(household.ledger, identity, (capture,))(db)
    for identity, capture in ((dead, dead_capture), (expired, expired_capture)):
        fact = household.ledger.store("Propuesta sintética", "fact", 1.0, (household.utterance(unit(e0=0.9))[1],),
                                      proposed=True)
        decision = household.ledger.attribute_voice_proposal(
            fact.id, FingerprintScorer(household.ledger, identity, (capture,)), CAP, **ALLOW)
        assert decision.reason is AttributionReason.NO_FINGERPRINT


def test_only_the_unavailable_lookup_is_no_fingerprint_any_other_error_is_error(household):
    """Review W16: a LookupError of another kind is a failure, not a missing fingerprint."""
    _identity, capture = household.utterance(unit(e0=0.9))
    fact = household.ledger.store("Propuesta sintética", "fact", 1.0, (capture,), proposed=True)
    for error, reason in ((LookupError("voice_fingerprint_unavailable"), AttributionReason.NO_FINGERPRINT),
                          (LookupError("something_else"), AttributionReason.ERROR), (KeyError("x"), AttributionReason.ERROR),
                          (RuntimeError("voice_fingerprint_unavailable"), AttributionReason.ERROR)):
        def broken(db, _error=error):
            raise _error
        assert household.ledger.attribute_voice_proposal(fact.id, broken, CAP, **ALLOW).reason is reason, error
    assert household.owner(fact) is None


def test_decisions_do_not_depend_on_row_order(household):
    scorer, _ = household.scorer(unit(e0=0.9, e1=0.3, e2=0.1))
    with household.ledger._transaction() as db:
        snapshot = scorer(db)
    reversed_snapshot = ScoringSnapshot(snapshot.model_key, tuple(reversed(snapshot.scores)), True, snapshot.speech_s)

    def grantable(top):
        return None if top == household.a else AttributionReason.TOP_NOT_GRANTABLE

    assert evaluate_gate(snapshot, ON, grantable) == evaluate_gate(reversed_snapshot, ON, grantable)
    with household.ledger._transaction() as db:
        assert scorer(db) == snapshot, "the same rows give the same snapshot"


@pytest.mark.parametrize(("vector", "outcome", "reason"), [
    (unit(e0=0.9, e1=0.3), VoiceVerificationOutcome.PASSED, AttributionReason.OWNER),
    (unit(e2=0.9, e0=0.3), VoiceVerificationOutcome.FAILED, AttributionReason.TOP_NOT_GRANTABLE),
    (unit(e1=0.9, e0=0.3), VoiceVerificationOutcome.FAILED_FOREIGN, AttributionReason.TOP_FOREIGN),
    (unit(e0=0.7), VoiceVerificationOutcome.FAILED, AttributionReason.BELOW_SCORE),
])
def test_verification_with_the_real_scorer(house, vector, outcome, reason):
    house.a = house.profile(unit(e0=1.0), "Voz A", PERSON)
    house.b = house.profile(unit(e1=1.0), "Voz B", OTHER)
    house.x = house.profile(unit(e2=1.0), "Voz X")
    actor = MemoryVoiceConsentActor("synthetic-home", PERSON)
    house.ledger.request_voice_consent(actor, 0)
    result = house.verify(PERSON, vector)
    assert (result.outcome, result.reason) == (outcome, reason)
    expected = VoiceConsentState.ACTIVE if outcome is VoiceVerificationOutcome.PASSED else VoiceConsentState.PENDING_VERIFICATION
    assert house.ledger.voice_consent(actor).state is expected
    if outcome is VoiceVerificationOutcome.PASSED:
        assert house.ledger.voice_consent(actor).verified_profile_id == house.a


def test_verification_without_a_fingerprint_is_recorded_as_unavailable_and_never_counts(house):
    house.a = house.profile(unit(e0=1.0), "Voz A", PERSON)
    house.x = house.profile(unit(e2=1.0), "Voz X")
    actor = MemoryVoiceConsentActor("synthetic-home", PERSON)
    house.ledger.request_voice_consent(actor, 0)
    for _ in range(4):
        binding = house.ledger.open_voice_verification(actor, house.ledger.voice_consent(actor).revision)
        result = house.ledger.verify_voice_consent(binding, UnavailableScorer(), CAP, **ALLOW)
        assert (result.outcome, result.reason) == (VoiceVerificationOutcome.UNAVAILABLE, AttributionReason.NO_FINGERPRINT)
    assert house.ledger.voice_consent(actor).state is VoiceConsentState.PENDING_VERIFICATION


def test_the_review_clip_of_the_phrase_is_deleted_in_the_evaluation_transaction(house, monkeypatch):
    house.a = house.profile(unit(e0=1.0), "Voz A", PERSON)
    house.x = house.profile(unit(e2=1.0), "Voz X")
    actor = MemoryVoiceConsentActor("synthetic-home", PERSON)
    house.ledger.request_voice_consent(actor, 0)
    identity, capture = house.utterance(unit(e0=0.9))
    with house.ledger._transaction() as db:
        db.execute("INSERT INTO voice_capture_reviews (fingerprint_id,audio_wav,transcript,room_id,duration_s,clip_duration_s) "
                   "VALUES (?,?,?,?,?,?)", (identity, b"\0" * 25644, "gato perro barco nube", "living", 2.0, 0.8))
    binding = house.ledger.open_voice_verification(actor, 1)
    result = house.ledger.verify_voice_consent(binding, FingerprintScorer(house.ledger, identity, (capture,)), CAP,
                                               **ALLOW, review_fingerprint=identity)
    assert result.outcome is VoiceVerificationOutcome.PASSED
    with house.ledger._transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM voice_capture_reviews").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM voice_fingerprints WHERE id=?", (identity,)).fetchone()[0] == 1
    assert house.ledger.discard_voice_review(identity, CAP) is False
    with pytest.raises(PermissionError):
        house.ledger.discard_voice_review(identity, object())
    with pytest.raises(ValueError):
        house.ledger.verify_voice_consent(binding, UnavailableScorer(), CAP, **ALLOW, review_fingerprint="vf_x")


def test_the_scorer_reads_inside_the_callers_transaction(household):
    """No second connection: the scorer is handed the transaction's own connection."""
    seen = []
    scorer, capture = household.scorer(unit(e0=0.9))
    fact = household.ledger.store("Propuesta sintética", "fact", 1.0, (capture,), proposed=True)

    def spy(db):
        seen.append(db is household.ledger._db)
        return scorer(db)

    assert household.ledger.attribute_voice_proposal(fact.id, spy, CAP, **ALLOW).owner
    assert seen == [True]


def test_no_score_or_snapshot_detail_is_stored(household):
    decision, fact = household.attribute(unit(e0=0.9, e1=0.3))
    assert decision.owner
    with household.ledger._transaction() as db:
        evidence = json.loads(db.execute("SELECT evidence FROM fact_owners WHERE fact_id=?", (fact.id,)).fetchone()[0])
    assert set(evidence) >= {"score_bucket", "margin_bucket"} and "score" not in evidence
    assert ProfileScore  # the snapshot type is not persisted
