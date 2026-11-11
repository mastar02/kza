"""U3b-4a-2 voice opt-in routes: denial matrix and flows through real logins and real routes.

Matrix rows are refusals or reads: each asserts the exact status and body, that
every privacy and voice table is unchanged, and that no seeded content, other
person's id or profile id leaks. Rows of one group must be byte-identical to
the group's reference row. Check order: auth (401) -> permission (403) ->
settings flag (409 memory_voice_disabled) -> body (422) -> state (409).
The flows test the writing paths (positive rows) on a fresh graph.
"""

import ast
from contextlib import AsyncExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import pytest

from src.audio.microphone_status import CaptureState
from src.dashboard import authority as authority_module
from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardPermission, DashboardRole
from src.dashboard.memory_api import RETURN_NOTICE, VOICE_RETURN_NOTICE
from src.dashboard.memory_voice_api import MULTI_PROFILE_NOTICE, REENROLMENT_NOTICE
from src.memory.fact_ledger import (
    MemoryOwnerActor, MemoryVoiceConsentActor, VoiceSwitchActor,
)
from src.memory.voice_attribution import (
    ProfileScore, ScoringSnapshot, VoiceAttributionCapability, VoiceProposalSettings,
)
from tests.mocks.memory_owner_graph import PERSON_A, PERSON_B, PERSON_C, SEED_ASSIGNER, owner_graph

ROOT = Path(__file__).resolve().parents[2]
PERSON_D, PERSON_E, PERSON_F = ("user_" + c * 32 for c in "def")
P_A, P_B, P_C1, P_C2, P_X, P_F = ("00000000-0000-4000-8000-%012x" % index
                                  for index in (0xA, 0xB, 0xC1, 0xC2, 0xF, 0xF1))
P_MISSING = "00000000-0000-4000-8000-0000000000ff"
ON = VoiceProposalSettings(enabled=True)
CAP = VoiceAttributionCapability("synthetic-matrix")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
TABLES = ("facts", "fact_sources", "fact_owners", "fact_owner_revisions", "voice_attribution_consents",
          "voice_verification_attempts", "voice_attribution_events", "voice_attribution_suspensions",
          "voice_attribution_control", "voice_profile_people", "voice_profiles")
DENIED = {"detail": "dashboard_permission_denied"}
UNAUTHENTICATED = {"detail": "auth required"}
ORIGIN = {"detail": "origin denied"}
DISABLED = {"detail": "memory_voice_disabled"}
INVALID = {"detail": "memory_voice_invalid_request"}
CONFLICT = {"detail": "memory_voice_conflict"}
STATE = {"detail": "memory_voice_consent_state"}
NOT_ALLOWED = {"detail": "Method Not Allowed"}
FOREIGN_ORIGIN = {"origin": "http://evil.example"}


@dataclass(frozen=True)
class Row:
    """One request and its exact expected answer."""

    actor: str
    method: str
    path: str
    status: int
    body: object
    payload: object = None
    group: str | None = None
    headers: tuple = ()
    phase: str = "on"  # 'off': the settings flag is false for this row
    room: str = "none"  # the microphone/capture state the dashboard reports for this row (ROOMS)

    @property
    def name(self) -> str:
        return f"[{self.phase}/{self.room}] {self.actor} {self.method} {self.path} {json.dumps(self.payload)}"


def _mics(*mics, stale=False):
    return SimpleNamespace(stale=stale, microphones=tuple(
        SimpleNamespace(room_id=room, capture=SimpleNamespace(state=state)) for room, state in mics))


def _raises():
    raise RuntimeError("synthetic telemetry failure")


_LIVING = _mics(("living", CaptureState.RECEIVING))
# room state -> (microphone snapshot provider, capture control) on the real DashboardAPI.
ROOMS = {
    "none": (None, None),
    "stale": (lambda: _mics(("living", CaptureState.RECEIVING), stale=True), None),
    "paused": (lambda: _LIVING, SimpleNamespace(capture_allowed=False)),
    "closed": (lambda: _mics(("living", CaptureState.CLOSED)), None),
    "other_room": (lambda: _mics(("kitchen", CaptureState.RECEIVING)), None),
    "raises": (_raises, None),
    "waiting": (lambda: _mics(("living", CaptureState.WAITING)), SimpleNamespace(capture_allowed=True)),
    "receiving": (lambda: _LIVING, None),
}
UNAVAILABLE_ROOMS = ("none", "stale", "paused", "closed", "other_room", "raises")


def _room(graph, name):
    graph.api.microphone_snapshot_provider, graph.api.capture_control = ROOMS[name]


PERSON_ROUTES = [("GET", "/api/memory/mine/voice", None),
                 ("POST", "/api/memory/mine/voice/consent", {"expected_revision": 0}),
                 ("DELETE", "/api/memory/mine/voice/consent", {"expected_revision": 0}),
                 ("POST", "/api/memory/mine/voice/verification", {"room_id": "living", "expected_revision": 1})]
OPERATOR_ROUTES = [("GET", "/api/memory/voice/consents", None),
                   ("POST", "/api/memory/voice/consents/%s/reopen" % PERSON_B, {"expected_revision": 1}),
                   ("GET", "/api/memory/voice/switch", None),
                   ("POST", "/api/memory/voice/switch", {"enabled": False, "expected_revision": 1}),
                   ("POST", "/api/voices/profiles/%s/attribution/resume" % P_X, {"expected_revision": 1})]
WRITES = [(method, path, payload) for method, path, payload in PERSON_ROUTES + OPERATOR_ROUTES if method != "GET"]


def _rows() -> list[Row]:
    rows = []
    # D1 / D19: unauthenticated, retired account, re-pointed credential (B's credential revised to A).
    for actor in ("anonymous", "home_d_retired", "home_revised_b_to_a"):
        rows += [Row(actor, m, p, 401, UNAUTHENTICATED, q) for m, p, q in PERSON_ROUTES + OPERATOR_ROUTES]
    # D2 / D4: the legacy administrative bearer and operators hold no person binding.
    for actor in ("legacy_bearer", "operator"):
        rows += [Row(actor, m, p, 403, DENIED, q) for m, p, q in PERSON_ROUTES]
    # D5 / D6: evaluator everywhere; a household account without a person on the person routes.
    rows += [Row("evaluator", m, p, 403, DENIED, q) for m, p, q in PERSON_ROUTES + OPERATOR_ROUTES]
    rows += [Row("home_none", m, p, 403, DENIED, q) for m, p, q in PERSON_ROUTES + OPERATOR_ROUTES]
    # D8: a person on every operator route and on resume.
    for actor in ("home_a", "home_b"):
        rows += [Row(actor, m, p, 403, DENIED, q) for m, p, q in OPERATOR_ROUTES]
    # D3: a foreign Origin on every write, for the actor that would otherwise pass.
    for method, path, payload in WRITES:
        actor = "home_b" if path.startswith("/api/memory/mine") else "operator"
        rows.append(Row(actor, method, path, 403, ORIGIN, payload, headers=tuple(FOREIGN_ORIGIN.items())))
    # D11: strict bodies (no person or profile id, no extra keys, no bool or float revisions).
    bad = [None, [], "1", {}, {"expected_revision": True}, {"expected_revision": 1.0}, {"expected_revision": -1},
           {"expected_revision": 2 ** 53}, {"expected_revision": "1"}, {"expected_revision": 1, "person_id": PERSON_A},
           {"expected_revision": 1, "profile_id": P_A}, {"expected_revision": 1, "extra": 1}]
    for payload in bad:
        rows.append(Row("home_b", "POST", "/api/memory/mine/voice/consent", 422, INVALID, payload))
        rows.append(Row("home_b", "DELETE", "/api/memory/mine/voice/consent", 422, INVALID, payload))
        rows.append(Row("operator", "POST", "/api/memory/voice/consents/%s/reopen" % PERSON_B, 422, INVALID, payload))
        rows.append(Row("operator", "POST", "/api/voices/profiles/%s/attribution/resume" % P_X, 422, INVALID, payload))
    rows.append(Row("operator", "POST", "/api/voices/profiles/%s/attribution/resume" % P_X, 422, INVALID,
                    {"expected_revision": 0}))  # a suspension row always has revision >= 1
    for payload in ({"expected_revision": 1}, {"room_id": "living"}, {"room_id": "Living Room", "expected_revision": 1},
                    {"room_id": "", "expected_revision": 1}, {"room_id": 7, "expected_revision": 1},
                    {"room_id": "living room", "expected_revision": 1},
                    {"room_id": "living", "expected_revision": False},
                    {"room_id": "living", "expected_revision": 1, "person_id": PERSON_A}):
        rows.append(Row("home_b", "POST", "/api/memory/mine/voice/verification", 422, INVALID, payload))
    for payload in ({"enabled": 1, "expected_revision": 1}, {"enabled": "false", "expected_revision": 1},
                    {"enabled": False}, {"enabled": False, "expected_revision": 1.0},
                    {"enabled": False, "expected_revision": 1, "actor": "x"}):
        rows.append(Row("operator", "POST", "/api/memory/voice/switch", 422, INVALID, payload))
    rows += [Row("home_b", "GET", "/api/memory/mine/voice?person_id=" + PERSON_A, 422, INVALID),
             Row("operator", "GET", "/api/memory/voice/consents?person_id=" + PERSON_A, 422, INVALID),
             Row("operator", "POST", "/api/voices/profiles/not-a-profile/attribution/resume", 422, INVALID,
                 {"expected_revision": 1}),
             Row("operator", "POST", "/api/voices/profiles/%s0/attribution/resume" % P_X, 422, INVALID,
                 {"expected_revision": 1}),
             Row("operator", "POST", "/api/memory/voice/consents/bad%20person/reopen", 422, INVALID,
                 {"expected_revision": 1})]
    # D12: stale revisions and wrong states.
    rows += [Row("home_b", "POST", "/api/memory/mine/voice/consent", 409, CONFLICT, {"expected_revision": 0}),
             Row("home_b", "DELETE", "/api/memory/mine/voice/consent", 409, CONFLICT, {"expected_revision": 0}),
             Row("home_a", "POST", "/api/memory/mine/voice/consent", 409, STATE, {"expected_revision": 2}),
             Row("home_e", "DELETE", "/api/memory/mine/voice/consent", 409, STATE, {"expected_revision": 0}),
             Row("operator", "POST", "/api/memory/voice/switch", 409, CONFLICT, {"enabled": False, "expected_revision": 0}),
             Row("operator", "POST", "/api/voices/profiles/%s/attribution/resume" % P_X, 409, CONFLICT,
                 {"expected_revision": 2}),
             Row("operator", "POST", "/api/voices/profiles/%s/attribution/resume" % P_MISSING, 409, CONFLICT,
                 {"expected_revision": 1})]
    # D29: the operator reopen of a non-locked, active or missing consent is one body.
    for person, revision in ((PERSON_B, 1), (PERSON_A, 2), (PERSON_E, 0), (PERSON_D, 0), ("user_unknown", 0)):
        rows.append(Row("operator", "POST", "/api/memory/voice/consents/%s/reopen" % person, 409, STATE,
                        {"expected_revision": revision}, group="operator:reopen"))
    # D13: no linked profile; two linked profiles (with the re-enrolment notice).
    rows += [Row("home_e", "POST", "/api/memory/mine/voice/consent", 409, {"detail": "memory_voice_no_profile"},
                 {"expected_revision": 0}),
             Row("home_c", "POST", "/api/memory/mine/voice/consent", 409, {"detail": {
                 "code": "memory_voice_profile_ambiguous", "notice": REENROLMENT_NOTICE}}, {"expected_revision": 0})]
    # D15 / review R: a person who could not open a window never learns the room state. Non-pending
    # persons (active A, none E, locked F) get one byte-identical body in every room state; a stale
    # revision likewise. (A pending person's room answer is counted: test_room_states_for_a_pending_person.)
    for room in ROOMS:
        for actor, revision in (("home_a", 2), ("home_e", 0), ("home_f", 2)):
            rows.append(Row(actor, "POST", "/api/memory/mine/voice/verification", 409, STATE,
                            {"room_id": "living", "expected_revision": revision}, group="room:not_pending",
                            room=room))
        rows.append(Row("home_b", "POST", "/api/memory/mine/voice/verification", 409, CONFLICT,
                        {"room_id": "living", "expected_revision": 0}, group="room:stale", room=room))
    # D25 / D27: the review route never confirms a voice-owned proposal; a person has no confirm route.
    rows += [Row("operator", "POST", "/api/memory/proposals/{voice_pending}/confirm", 409,
                 {"detail": "memory_owner_confirm_required"}),
             Row("home_a", "POST", "/api/memory/mine/facts/{voice_pending}/confirm", 405, NOT_ALLOWED)]
    # D28 / D32: another person's voice proposal answers exactly like a missing id.
    for name in ("voice_pending", "voice_confirmed", "missing"):
        rows.append(Row("home_b", "DELETE", "/api/memory/mine/facts/{%s}" % name, 200, {"deleted": False},
                        group="home_b:delete"))
        rows.append(Row("home_b", "POST", "/api/memory/mine/facts/{%s}/return" % name, 200, {"returned": False},
                        group="home_b:return"))
    # D10: the settings flag off refuses every route except status, revoke and the G1 switch,
    # before any body validation (valid and invalid bodies get the same answer).
    for actor, method, path in (("home_b", "POST", "/api/memory/mine/voice/consent"),
                                ("home_b", "POST", "/api/memory/mine/voice/verification"),
                                ("operator", "POST", "/api/memory/voice/consents/%s/reopen" % PERSON_B),
                                ("operator", "POST", "/api/voices/profiles/%s/attribution/resume" % P_X)):
        for payload in ({"expected_revision": 1, "room_id": "living"} if "verification" in path
                        else {"expected_revision": 1}, {"bad": True}, None):
            rows.append(Row(actor, method, path, 409, DISABLED, payload, phase="off"))
    rows += [Row("operator", "GET", "/api/memory/voice/consents", 409, DISABLED, phase="off"),
             Row("operator", "GET", "/api/memory/voice/consents?x=1", 409, DISABLED, phase="off"),
             # Permission still comes first with the flag off.
             Row("home_b", "GET", "/api/memory/voice/consents", 403, DENIED, phase="off"),
             Row("anonymous", "POST", "/api/memory/mine/voice/consent", 401, UNAUTHENTICATED,
                 {"expected_revision": 0}, phase="off"),
             # D31: status stays available and content-minimal.
             Row("home_b", "GET", "/api/memory/mine/voice", 200,
                 {"state": "pending_verification", "revision": 1, "enabled": False, "window": None}, phase="off"),
             Row("home_e", "GET", "/api/memory/mine/voice", 200,
                 {"state": None, "revision": 0, "enabled": False, "window": None}, phase="off"),
             # Revocation is not flag-gated: a body error is still a body error.
             Row("home_b", "DELETE", "/api/memory/mine/voice/consent", 422, INVALID, {"bad": True}, phase="off"),
             # G1 off path stays reachable: switching on without the settings flag is refused.
             Row("operator", "POST", "/api/memory/voice/switch", 409, {"detail": "memory_voice_not_configured"},
                 {"enabled": True, "expected_revision": 1}, phase="off"),
             Row("operator", "GET", "/api/memory/voice/switch", 200,
                 {"configured": False, "enabled": True, "revision": 1, "effective": False}, phase="off")]
    return rows


ROWS = _rows()


def test_matrix_size_and_distinct_rows():
    groups = {"D1", "D2", "D3", "D4", "D5", "D6", "D8", "D10", "D11", "D12", "D13", "D15/R", "D19", "D19b", "D25",
              "D27", "D28", "D29", "D31", "D32"}
    assert len(ROWS) >= 200 and len(groups) >= 20
    assert len({(r.phase, r.room, r.actor, r.method, r.path, json.dumps(r.payload), r.headers)
                for r in ROWS}) == len(ROWS)
    print(f"U3B4A2_MATRIX_ROWS={len(ROWS)}")


def _profile(ledger, profile, person, clock):
    from tests.mocks.memory_sources import memory_source
    source = memory_source(now=clock.now - 10)
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_profiles VALUES (?,?,'enc_synthetic',2,?,?,?,'synthetic-reviewer',1,NULL)",
                   (profile, "Synthetic voice " + profile[-2:], b"\0" * 8,
                    json.dumps([source.to_dict()], sort_keys=True), clock.now - 5))
        if person is not None:
            db.execute("INSERT INTO voice_profile_people VALUES (?,?,1,?,'synthetic-reviewer')",
                       (profile, person, clock.now))


def scores(top):
    return lambda db: ScoringSnapshot("enc_synthetic", tuple(
        ProfileScore(row[0], row[1], .95 if row[0] == top else .3) for row in db.execute(
            "SELECT id, revision FROM voice_profiles WHERE deleted_at IS NULL")), True, 3.0)


async def _seed(graph):
    ledger = graph.memory.ledger
    ledger.voice_settings = ON
    graph.accounts |= {PERSON_D, PERSON_E, PERSON_F}

    def seed():
        ledger.set_voice_switch(True, 0, VoiceSwitchActor("synthetic-seed-operator"))
        for profile, person in ((P_A, PERSON_A), (P_B, PERSON_B), (P_C1, PERSON_C), (P_C2, PERSON_C), (P_X, None),
                                (P_F, PERSON_F)):
            _profile(ledger, profile, person, graph.clock)
        home_a = MemoryVoiceConsentActor("synthetic-seed-home-a", PERSON_A)
        ledger.request_voice_consent(home_a, 0)
        binding = ledger.open_voice_verification(home_a, 1)
        assert ledger.verify_voice_consent(binding, scores(P_A), CAP, **ALLOW).outcome.value == "passed"
        ledger.request_voice_consent(MemoryVoiceConsentActor("synthetic-seed-home-b", PERSON_B), 0)
        home_f = MemoryVoiceConsentActor("synthetic-seed-home-f", PERSON_F)
        ledger.request_voice_consent(home_f, 0)
        for _ in range(3):  # F: three foreign-profile failures lock the consent (revision 2)
            ledger.verify_voice_consent(ledger.open_voice_verification(home_f, 1), scores(P_X), CAP, **ALLOW)
        assert ledger.voice_status(home_f)["state"] == "verification_locked"
        assert ledger.voice_status(home_f)["revision"] == 2
        with ledger._transaction() as db:
            db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')",
                       (P_X, graph.clock.now))
    await graph.call(seed)
    for name in ("voice_pending", "voice_confirmed"):
        await graph.store(name, state="proposed")
        decision = await graph.call(ledger.attribute_voice_proposal, graph.facts[name], scores(P_A), CAP, **ALLOW)
        assert decision.owner, decision
    await graph.call(ledger.assign_owner, graph.facts["voice_confirmed"], PERSON_A, 1, 1, SEED_ASSIGNER, **ALLOW)
    graph.facts["missing"], graph.contents["missing"] = "fact_" + "0" * 64, "fact_" + "0" * 64


async def _clients(graph, stack, dashboard_auth):
    clients = {"anonymous": await stack.enter_async_context(graph.client())}
    for actor, role, subject, person in (
            ("home_none", DashboardRole.HOME, "synthetic-home-none", None),
            ("evaluator", DashboardRole.EVALUATOR, "synthetic-evaluator", None),
            ("home_a", DashboardRole.HOME, "synthetic-home-a", PERSON_A),
            ("home_b", DashboardRole.HOME, "synthetic-home-b", PERSON_B),
            ("home_c", DashboardRole.HOME, "synthetic-home-c", PERSON_C),
            ("home_e", DashboardRole.HOME, "synthetic-home-e", PERSON_E),
            ("home_f", DashboardRole.HOME, "synthetic-home-f", PERSON_F),
            ("operator", DashboardRole.OPERATOR, "synthetic-operator", None),
            ("home_d_retired", DashboardRole.HOME, "synthetic-home-d", PERSON_D),
            ("home_revised_b_to_a", DashboardRole.HOME, "synthetic-home-b-revised", PERSON_B)):
        clients[actor] = await stack.enter_async_context(graph.client(role, subject, person))
    clients["legacy_bearer"] = await stack.enter_async_context(graph.client(bearer=dashboard_auth["Authorization"]))
    await graph.authority.retire_user(graph.root, PERSON_D)
    revised = clients["home_revised_b_to_a"].grant
    assert await graph.authority.revise_credential(graph.root, revised.identity, role=DashboardRole.HOME,
                                                   expires_at=graph.clock.now + 7200, user_id=PERSON_A)
    return clients


def _dump(graph):
    with sqlite3.connect(graph.path) as db:
        return {table: sorted(map(tuple, db.execute("SELECT * FROM " + table))) for table in TABLES}


async def _send(graph, clients, row):
    path = re.sub(r"\{(\w+)\}", lambda m: graph.facts[m.group(1)], row.path)
    kwargs = {} if row.payload is None else {"content": json.dumps(row.payload),
                                             "headers": {"content-type": "application/json"}}
    if row.headers:
        kwargs["headers"] = {**kwargs.get("headers", {}), **dict(row.headers)}
    response = await clients[row.actor].request(row.method, path, **kwargs)
    return response.status_code, response.content


def _secrets(graph):
    """Strings no refusal may contain: contents, fact ids and every profile id."""
    values = [graph.contents[name] for name in graph.facts if name != "missing"]
    values += [graph.facts[name] for name in graph.facts if name != "missing"]
    return values + [P_A, P_C1, P_C2]


async def test_voice_denial_matrix(tmp_path, dashboard_auth):
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        await _seed(graph)
        clients = await _clients(graph, stack, dashboard_auth)
        ledger = graph.memory.ledger
        failures, groups = [], {}
        for phase in ("on", "off"):
            ledger.voice_settings = ON if phase == "on" else VoiceProposalSettings()
            for row in (r for r in ROWS if r.phase == phase):
                _room(graph, row.room)
                before = _dump(graph)
                status, content = await _send(graph, clients, row)
                text = content.decode("utf-8", "replace")
                body = json.loads(content) if content else None
                if (status, body) != (row.status, row.body):
                    failures.append(f"{row.name}: got {status} {text[:200]!r}, expected {row.status} {row.body}")
                if _dump(graph) != before:
                    failures.append(f"{row.name}: tables changed")
                leaked = [value for value in _secrets(graph) if value in text]
                if leaked and row.path != "/api/memory/mine/voice":
                    failures.append(f"{row.name}: leaked {leaked}")
                if row.group is not None:
                    groups.setdefault(row.group, []).append((row.name, status, content))
        for group, members in groups.items():
            reference = members[-1]
            for name, status, content in members:
                if (status, content) != reference[1:]:
                    failures.append(f"{name}: differs from {reference[0]} in {group}")
        assert failures == [], "\n".join(failures)
        # Positive controls on the same graph: the denials above are not vacuous.
        ledger.voice_settings = ON
        _room(graph, "none")
        listing = (await clients["legacy_bearer"].get("/api/memory/voice/consents")).json()
        assert [(c["person_id"], c["state"], c["verified_profile"]) for c in listing["consents"]] == [
            (PERSON_A, "active", {"id": P_A, "name": "Synthetic voice 0a"}), (PERSON_B, "pending_verification", None),
            (PERSON_F, "verification_locked", None)]
        assert listing["truncated"] is False
        assert listing["multiple_profiles"] == [{"person_id": PERSON_C, "linked_live_profiles": 2}]
        assert listing["multiple_profiles_notice"] == MULTI_PROFILE_NOTICE
        assert listing["suspended_profiles"] == [{"profile_id": P_X, "revision": 1}]
        status = (await clients["home_a"].get("/api/memory/mine/voice")).json()
        assert {key: status[key] for key in ("state", "revision", "enabled", "window", "failures_left")} == {
            "state": "active", "revision": 2, "enabled": True, "window": None, "failures_left": None}
        assert P_A not in json.dumps(status) and PERSON_B not in json.dumps(status)
        mine = (await clients["home_a"].get("/api/memory/mine")).json()["facts"]  # D20 / D34
        pending = next(item for item in mine if item["id"] == graph.facts["voice_pending"])
        assert set(pending) == {"id", "kind", "created_at", "expires_at"} and pending["kind"] == "voice_proposal_pending"
        assert pending["created_at"] % 3600 == 0 and pending["expires_at"] % 3600 == 0
        assert graph.contents["voice_pending"] not in json.dumps(mine)


@pytest.fixture
async def flows(tmp_path, dashboard_auth):
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        await _seed(graph)
        clients = await _clients(graph, stack, dashboard_auth)
        graph.api._voice_room_available = lambda room_id: room_id == "living"
        yield graph, clients


async def test_window_flow_rate_limit_and_isolation(flows):
    """D16, D17, D18: windows, the durable 6-per-day limit and no trace of another person."""
    graph, clients = flows
    ledger = graph.memory.ledger

    def unlink_c2():
        with ledger._transaction() as db:
            db.execute("DELETE FROM voice_profile_people WHERE profile_id=?", (P_C2,))
    await graph.call(unlink_c2)
    assert (await clients["home_c"].post("/api/memory/mine/voice/consent", json={"expected_revision": 0})).json() == {
        "state": "pending_verification", "revision": 1}
    stale = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                         json={"room_id": "living", "expected_revision": 0})
    assert (stale.status_code, stale.json()) == (409, {"detail": "memory_voice_conflict"})
    first = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                         json={"room_id": "living", "expected_revision": 1})
    assert first.status_code == 200 and first.json()["window"]["room_id"] == "living"
    before_c = (await clients["home_c"].get("/api/memory/mine/voice")).content
    other = await clients["home_c"].post("/api/memory/mine/voice/verification",
                                         json={"room_id": "living", "expected_revision": 1})
    assert other.status_code == 200
    status_b = (await clients["home_b"].get("/api/memory/mine/voice")).json()
    assert status_b["window"]["room_id"] == "living" and PERSON_C not in json.dumps(status_b)
    assert before_c != (await clients["home_c"].get("/api/memory/mine/voice")).content  # C's own window shows
    for _ in range(5):
        assert (await clients["home_b"].post("/api/memory/mine/voice/verification",
                                             json={"room_id": "living", "expected_revision": 1})).status_code == 200
    seventh = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                           json={"room_id": "living", "expected_revision": 1})
    assert (seventh.status_code, seventh.json()) == (429, {"detail": "memory_voice_rate_limited"})
    ledger.voice_windows = type(ledger.voice_windows)(ledger.clock)  # a restart closes windows, not the limit
    again = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                         json={"room_id": "living", "expected_revision": 1})
    assert again.status_code == 429
    a_before = (await clients["home_a"].get("/api/memory/mine/voice")).content
    with ledger._transaction() as db:  # D18: B's consent changes; A's answer is byte-identical
        db.execute("UPDATE voice_attribution_consents SET state='revoked', revision=revision+1 WHERE person_id=?",
                   (PERSON_B,))
    assert (await clients["home_a"].get("/api/memory/mine/voice")).content == a_before


async def test_revocation_closes_the_window_and_works_with_the_flag_off(flows):
    """D31: DELETE consent with the flag off revokes; a later request needs the flag again."""
    graph, clients = flows
    ledger = graph.memory.ledger
    assert (await clients["home_b"].post("/api/memory/mine/voice/verification",
                                         json={"room_id": "living", "expected_revision": 1})).status_code == 200
    ledger.voice_settings = VoiceProposalSettings()
    revoked = await clients["home_b"].request("DELETE", "/api/memory/mine/voice/consent",
                                              json={"expected_revision": 1})
    assert revoked.json() == {"state": "revoked", "revision": 2}
    ledger.voice_settings = ON
    assert (await clients["home_b"].get("/api/memory/mine/voice")).json()["window"] is None
    assert (await clients["home_b"].post("/api/memory/mine/voice/consent",
                                         json={"expected_revision": 2})).json()["state"] == "pending_verification"


async def test_insufficient_and_suspended_profiles(flows):
    """D14: scoring set below 2, and a suspended verified profile."""
    graph, clients = flows
    ledger = graph.memory.ledger
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')",
                   (P_B, graph.clock.now))
    suspended = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                             json={"room_id": "living", "expected_revision": 1})
    assert suspended.json() == {"detail": "memory_voice_profile_suspended"}
    ledger.voice_settings = VoiceProposalSettings(enabled=True, min_scoring_profiles=10)
    with ledger._transaction() as db:
        db.execute("DELETE FROM voice_attribution_suspensions WHERE profile_id=?", (P_B,))
    few = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                       json={"room_id": "living", "expected_revision": 1})
    assert few.status_code == 409 and few.json()["detail"]["code"] == "memory_voice_insufficient_profiles"


async def test_descartar_and_no_es_mio_on_voice_proposals(flows):
    """D21, D22, D23, D24, D33: notices, counting and Devueltos."""
    graph, clients = flows
    ledger = graph.memory.ledger
    extra = []
    for index in range(4):
        name = "voice_extra_%d" % index
        await graph.store(name, state="proposed")
        assert (await graph.call(ledger.attribute_voice_proposal, graph.facts[name], scores(P_A), CAP, **ALLOW)).owner
        extra.append(name)
    for name in extra[:2]:  # D33: two discards count nothing
        assert (await clients["home_a"].delete("/api/memory/mine/facts/" + graph.facts[name])).json() == {
            "deleted": True}
    events = lambda: ledger._db.execute("SELECT COUNT(*) FROM voice_attribution_events").fetchone()[0]  # noqa: E731
    assert events() == 0
    returned = await clients["home_a"].post("/api/memory/mine/facts/%s/return" % graph.facts["voice_pending"])
    assert returned.json() == {"returned": True, "notice": VOICE_RETURN_NOTICE} and events() == 1
    returned = await clients["home_a"].post("/api/memory/mine/facts/%s/return" % graph.facts[extra[2]])
    assert returned.json()["notice"] == VOICE_RETURN_NOTICE and events() == 2
    suspensions = ledger._db.execute("SELECT * FROM voice_attribution_suspensions WHERE profile_id=?",
                                     (P_A,)).fetchall()
    assert len(suspensions) == 1
    third = await clients["home_a"].post("/api/memory/mine/facts/%s/return" % graph.facts[extra[3]])
    assert third.json()["returned"] is True and events() == 2  # D23: no new event, one suspension
    assert ledger._db.execute("SELECT * FROM voice_attribution_suspensions WHERE profile_id=?",
                              (P_A,)).fetchall() == suspensions
    confirmed = await clients["home_a"].post("/api/memory/mine/facts/%s/return" % graph.facts["voice_confirmed"])
    assert confirmed.json() == {"returned": True, "notice": RETURN_NOTICE} and events() == 2  # D24
    devueltos = (await clients["operator"].get("/api/memory/returned")).json()["facts"]
    assert graph.facts["voice_pending"] in {item["id"] for item in devueltos}
    assert not {graph.facts[name] for name in extra[:2]} & {item["id"] for item in devueltos}
    deleted = ledger._db.execute("SELECT content, state FROM facts WHERE id=?", (graph.facts[extra[0]],)).fetchone()
    assert tuple(deleted) == (None, "deleted")


async def test_operator_reopen_switch_and_resume(flows):
    graph, clients = flows
    ledger = graph.memory.ledger
    for _ in range(3):
        binding = await graph.call(ledger.open_voice_verification, MemoryVoiceConsentActor("s", PERSON_B), 1)
        await graph.call(ledger.verify_voice_consent, binding, scores(P_X), CAP, **ALLOW)
    status = (await clients["home_b"].get("/api/memory/mine/voice")).json()
    assert status["state"] == "verification_locked" and "administración" in status["message"]
    reopened = await clients["operator"].post("/api/memory/voice/consents/%s/reopen" % PERSON_B,
                                              json={"expected_revision": 2})
    assert reopened.json() == {"state": "pending_verification", "revision": 3}
    resumed = await clients["operator"].post("/api/voices/profiles/%s/attribution/resume" % P_X,
                                             json={"expected_revision": 1})
    assert resumed.json() == {"resumed": True}
    off = await clients["operator"].post("/api/memory/voice/switch", json={"enabled": False, "expected_revision": 1})
    assert off.json() == {"configured": True, "enabled": False, "revision": 2, "effective": False}
    refused = await clients["home_e"].post("/api/memory/mine/voice/consent", json={"expected_revision": 0})
    assert refused.json() == {"detail": "memory_voice_disabled"}  # the switch row gates the person routes too
    on = await clients["legacy_bearer"].post("/api/memory/voice/switch", json={"enabled": True, "expected_revision": 2})
    assert on.json()["effective"] is True


async def test_a_pre_4a2_session_is_reminted_with_the_voice_grant(tmp_path, dashboard_auth):
    """D7: a household session minted before memory:own:voice existed never acts with stale permissions."""
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        graph.memory.ledger.voice_settings = ON
        original = authority_module._permissions

        def pre_4a2(role, user_id):
            return frozenset(original(role, user_id) - {"memory:own:voice"})

        authority_module._permissions = pre_4a2
        try:
            client = await stack.enter_async_context(graph.client(DashboardRole.HOME, "synthetic-old-a", PERSON_A))
        finally:
            authority_module._permissions = original
        answer = await client.get("/api/memory/mine/voice")
        assert answer.status_code == 200 and answer.json()["state"] is None


def test_route_permissions_for_the_new_surfaces():
    """D9 and the auth.py branches: exact permission sets per route and method."""
    from types import SimpleNamespace

    def permissions(method, path):
        return _route_permissions(SimpleNamespace(url=SimpleNamespace(path=path), scope={"method": method},
                                                  query_params={}))
    voice, assign, read = (DashboardPermission.MEMORY_OWN_VOICE, DashboardPermission.MEMORY_ASSIGN,
                           DashboardPermission.MEMORY_READ)
    admin, review = DashboardPermission.ADMIN_MANAGE, DashboardPermission.VOICES_REVIEW
    assert permissions("GET", "/api/memory/mine/voice") == {voice}
    assert permissions("POST", "/api/memory/mine/voice/consent") == {voice}
    assert permissions("DELETE", "/api/memory/mine/voice/anything/else") == {voice}
    assert permissions("GET", "/api/memory/voice/consents") == {assign, read}
    assert permissions("POST", "/api/memory/voice/consents/x/reopen") == {assign}
    assert permissions("GET", "/api/memory/voice/switch") == {assign, read}
    assert permissions("POST", "/api/memory/voice/switch") == {assign, admin}
    assert permissions("POST", "/api/voices/profiles/%s/attribution/resume" % P_X) == {review, admin, assign}
    assert permissions("GET", "/api/memory/mine") == {DashboardPermission.MEMORY_OWN_READ}


def test_only_household_accounts_with_a_person_hold_the_voice_grant():
    grant = DashboardPermission.MEMORY_OWN_VOICE.value
    assert grant in authority_module._permissions(DashboardRole.HOME, PERSON_A)
    assert grant not in authority_module._permissions(DashboardRole.HOME, None)
    assert grant not in authority_module._permissions(DashboardRole.OPERATOR, None)
    assert grant not in authority_module._permissions(DashboardRole.EVALUATOR, None)
    lease = authority_module.legacy_lease("Bearer x", expected=lambda: "x")
    assert grant not in lease.principal.permissions


def test_no_route_creates_or_activates_a_consent_outside_its_own_writer():
    """D30: exact call-site counts; verify_voice_consent and attribute_voice_proposal are called only by the 4b-2 attributor."""
    from collections import Counter
    names = {"request_voice_consent", "verify_voice_consent", "attribute_voice_proposal", "reopen_voice_consent",
             "set_voice_switch", "resume_voice_attribution", "open_voice_window", "open_voice_verification",
             "revoke_voice_consent"}
    calls = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name in ("fact_ledger.py",):
            continue
        found = Counter(node.attr for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                        if isinstance(node, ast.Attribute) and node.attr in names)
        if found:
            calls[str(path.relative_to(ROOT))] = dict(found)
    assert calls == {"src/dashboard/memory_voice_api.py": {
        "request_voice_consent": 1, "revoke_voice_consent": 1, "open_voice_window": 1, "reopen_voice_consent": 1,
        "set_voice_switch": 1, "resume_voice_attribution": 1},
        "src/users/voice_attributor.py": {"attribute_voice_proposal": 1, "verify_voice_consent": 1}}


def test_visible_texts_are_spanish():
    from src.dashboard import memory_voice_api as module
    for name in ("OPT_IN_DISCLOSURE", "VOICES_DISCLOSURE", "REENROLMENT_NOTICE", "MULTI_PROFILE_NOTICE",
                 "INSUFFICIENT_PROFILES_MESSAGE"):
        assert any(word in getattr(module, name) for word in ("voz", "voces", "administración")), name
    assert "vence sola en menos de 48 horas" in VOICE_RETURN_NOTICE
    assert MemoryOwnerActor  # imported for the flows' shared vocabulary


async def test_each_route_refuses_on_its_own_when_the_route_permission_layer_is_bypassed(tmp_path, dashboard_auth,
                                                                                         monkeypatch):
    """Defence in depth: with _route_permissions emptied, the minted actors still refuse every wrong role."""
    from src.dashboard import auth as auth_module
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        await _seed(graph)
        clients = await _clients(graph, stack, dashboard_auth)
        graph.api._voice_room_available = lambda room_id: True
        before = _dump(graph)
        monkeypatch.setattr(auth_module, "_route_permissions", lambda request: frozenset())
        person = {"detail": "memory_person_required"}
        cases = [(actor, method, path, payload, person) for actor in ("operator", "legacy_bearer", "home_none",
                                                                      "evaluator")
                 for method, path, payload in PERSON_ROUTES]
        cases += [(actor, method, path, payload,
                   {"detail": "memory_voice_switch_denied" if path.endswith("/switch") and method == "POST"
                    else "memory_assign_denied"})
                  for actor in ("home_a", "home_none", "evaluator") for method, path, payload in OPERATOR_ROUTES]
        failures = []
        for actor, method, path, payload, expected in cases:
            kwargs = {} if payload is None else {"json": payload}
            response = await clients[actor].request(method, path, **kwargs)
            if (response.status_code, response.json()) != (403, expected):
                failures.append(f"{actor} {method} {path}: {response.status_code} {response.text[:120]}")
        # Resume's own actor also requires voices:review and admin:manage (route layer still bypassed).
        original = authority_module._permissions
        for missing in ("voices:review", "admin:manage"):
            monkeypatch.setattr(authority_module, "_permissions",
                                lambda role, user_id, gone=missing: original(role, user_id) - {gone})
            response = await clients["operator"].post("/api/voices/profiles/%s/attribution/resume" % P_X,
                                                      json={"expected_revision": 1})
            if (response.status_code, response.json()) != (403, {"detail": "memory_assign_denied"}):
                failures.append(f"resume without {missing}: {response.status_code} {response.text[:120]}")
        monkeypatch.setattr(authority_module, "_permissions", original)
        assert failures == [], "\n".join(failures)
        assert _dump(graph) == before


async def test_room_states_for_a_pending_person(tmp_path, dashboard_auth):
    """Review R/T over HTTP with the real room check: each unavailable state is refused, counted, leaves no
    window; the seventh opening of the day is refused before the room is read; capturing rooms open."""
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        await _seed(graph)
        clients = await _clients(graph, stack, dashboard_auth)
        ledger = graph.memory.ledger
        count = lambda: ledger._db.execute(  # noqa: E731
            "SELECT COUNT(*) FROM voice_verification_attempts WHERE person_id=?", (PERSON_B,)).fetchone()[0]
        verify = lambda revision=1: clients["home_b"].post(  # noqa: E731
            "/api/memory/mine/voice/verification", json={"room_id": "living", "expected_revision": revision})
        window = lambda: clients["home_b"].get("/api/memory/mine/voice")  # noqa: E731
        _room(graph, "receiving")
        stale = await verify(0)  # Y14: a ledger refusal registers nothing and counts nothing
        assert (stale.status_code, stale.json()) == (409, {"detail": "memory_voice_conflict"})
        assert (await window()).json()["window"] is None and count() == 0
        for index, room in enumerate(UNAVAILABLE_ROOMS, 1):
            _room(graph, room)
            refused = await verify()
            assert (refused.status_code, refused.json()) == (409, {"detail": "memory_voice_room_unavailable"}), room
            assert count() == index and (await window()).json()["window"] is None, room
        reads = []
        graph.api.microphone_snapshot_provider = lambda: reads.append(1) or _LIVING
        limited = await verify()
        assert (limited.status_code, limited.json()) == (429, {"detail": "memory_voice_rate_limited"})
        assert reads == [] and count() == 6 and (await window()).json()["window"] is None
        with ledger._transaction() as db:  # a day later (moving the shared clock would expire the sessions)
            db.execute("UPDATE voice_verification_attempts SET attempted_at=attempted_at-86401 WHERE person_id=?",
                       (PERSON_B,))
        for room in ("waiting", "receiving"):
            _room(graph, room)
            opened = await verify()
            assert opened.status_code == 200 and opened.json()["window"]["room_id"] == "living", room
            assert (await window()).json()["window"] == opened.json()["window"]


async def test_unlinking_the_profile_closes_a_pending_window_over_http(flows):
    """Review W over HTTP: the real link writer closes B's window; status never shows it again."""
    graph, clients = flows
    ledger = graph.memory.ledger
    opened = await clients["home_b"].post("/api/memory/mine/voice/verification",
                                          json={"room_id": "living", "expected_revision": 1})
    assert opened.status_code == 200 and (await clients["home_b"].get("/api/memory/mine/voice")).json()["window"]
    from src.users.voice_people import VoicePersonLinks

    def unlink():
        with ledger._transaction() as db:
            VoicePersonLinks.unlink_profile(SimpleNamespace(ledger=ledger), db, P_B)
    await graph.call(unlink)
    assert ledger.voice_windows.current(PERSON_B) is None
    assert (await clients["home_b"].get("/api/memory/mine/voice")).json()["window"] is None
