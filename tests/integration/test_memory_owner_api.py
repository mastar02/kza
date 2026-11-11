"""U3b-2: 'Mis recuerdos' and operator owner routes through real logins and routes.

The person comes only from the household credential's account binding.
Refusals are checked against the committed privacy tables (nothing written).
"""

import ast
import asyncio
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import pytest

from src.dashboard import memory_api
from src.dashboard.authority import DashboardRole
from src.memory.fact_ledger import MEMORY_PAGE_LIMIT
from tests.mocks.memory_owner_graph import PERSON_A, PERSON_B, PERSON_C, owner_graph

ROOT = Path(__file__).resolve().parents[2]
OWN = {"memory:own:read", "memory:own:write"}


@pytest.fixture
async def graph(tmp_path, dashboard_auth):
    async with owner_graph(tmp_path, dashboard_auth) as value:
        yield value


def leaked(graph, response, allowed=()):
    """Names of seeded facts whose content or id appears in a response body."""
    return sorted(name for name in graph.facts if name not in allowed
                  and (graph.contents[name] in response.text or graph.facts[name] in response.text))


async def owner_body(graph, name, person, **extra):
    fact = await graph.call(graph.memory.ledger.get, graph.facts[name], review=True)
    current = await graph.call(lambda: graph.memory.ledger._db.execute(
        "SELECT revision FROM fact_owner_revisions WHERE fact_id=?", (fact.id,)).fetchone())
    return {"person_id": person, "expected_fact_revision": fact.revision,
            "expected_owner_revision": 0 if current is None else current[0], **extra}


# --- the person's own view -------------------------------------------------------------------


async def test_each_person_sees_only_their_own_memories(graph):
    for name, person in (("a1", PERSON_A), ("a2", PERSON_A), ("b1", PERSON_B), ("c1", PERSON_C)):
        graph.clock.now += 1
        await graph.store(name)
        await graph.assign(name, person)
    await graph.store("household")
    await graph.store("auto", state="automatic")
    for person, expected in ((PERSON_A, ["a2", "a1"]), (PERSON_B, ["b1"]), (PERSON_C, ["c1"])):
        async with graph.client(DashboardRole.HOME, "synthetic-home-" + person[-1], person) as client:
            whoami = (await client.get("/api/admin/auth/whoami")).json()
            assert OWN <= set(whoami["permissions"]) and "memory:assign" not in whoami["permissions"]
            response = await client.get("/api/memory/mine")
            assert response.status_code == 200
            body = response.json()
            assert [item["id"] for item in body["facts"]] == [graph.facts[name] for name in expected]
            assert body["next"] is None and body["limit"] == MEMORY_PAGE_LIMIT
            assert set(body) == {"facts", "next", "limit"}
            assert leaked(graph, response, expected) == []
            for item in body["facts"]:
                assert item["owner_method"] == "operator_assignment" and item["state"] == "confirmed"
                assert not {"sources", "confirmed_by", "confidence", "revision", "pending_index"} & set(item)
            assert "synthetic-seed" not in response.text and "status" not in body


async def test_person_deletes_own_memory_with_its_derived_data(graph):
    await graph.store("mine")
    await graph.store("child", state="automatic", parent="mine")
    await graph.assign("mine", PERSON_A)
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A) as client:
        listed = (await client.get("/api/memory/mine")).json()["facts"]
        assert listed[0]["has_hidden_dependents"] is True and listed[0]["owned_dependents"] == []
        first = await client.delete("/api/memory/mine/facts/" + graph.facts["mine"])
        again = await client.delete("/api/memory/mine/facts/" + graph.facts["mine"])
    assert (first.status_code, first.json(), again.json()) == (200, {"deleted": True}, {"deleted": False})
    for name in ("mine", "child"):
        assert await graph.call(graph.memory.ledger.get, graph.facts[name], review=True) is None
    assert graph.dump()["fact_owners"] == []


async def test_no_es_mio_returns_to_devueltos_and_operator_reassigns_or_deletes(graph):
    for name in ("x", "y"):
        await graph.store(name)
        await graph.assign(name, PERSON_A)
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A) as client:
        for name in ("x", "y"):
            response = await client.post(f"/api/memory/mine/facts/{graph.facts[name]}/return")
            assert response.status_code == 200 and response.json()["returned"] is True
            assert "No se borró" in response.json()["notice"]
        assert (await client.get("/api/memory/mine")).json()["facts"] == []
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        returned = (await client.get("/api/memory/returned")).json()
        assert {item["id"] for item in returned["facts"]} == {graph.facts["x"], graph.facts["y"]}
        assert all(item["owner"] is None and item["owner_change"] == "owner_returned" for item in returned["facts"])
        counts = (await client.get("/api/memory/owners/counts")).json()
        assert (counts["returned"], counts["currently_owned"]) == (2, 0)
        reassigned = await client.post(f"/api/memory/facts/{graph.facts['x']}/owner",
                                       json=await owner_body(graph, "x", PERSON_B))
        assert reassigned.status_code == 200, reassigned.text
        assert reassigned.json()["owner_revision"] == 3 and reassigned.json()["confirmed_now"] is False
        deleted = await client.delete("/api/memory/facts/" + graph.facts["y"])
        assert deleted.json()["deleted"] is True
        assert (await client.get("/api/memory/returned")).json()["facts"] == []
    async with graph.client(DashboardRole.HOME, "synthetic-home-b", PERSON_B) as client:
        assert [item["id"] for item in (await client.get("/api/memory/mine")).json()["facts"]] == [graph.facts["x"]]


# --- operator assignment ----------------------------------------------------------------------


async def test_assigning_an_automatic_memory_confirms_it_and_discloses_retired_receipts(graph):
    await graph.store("auto", state="automatic")
    await graph.store("held")
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        snapshot = (await client.get("/api/memory")).json()
        assert "registros derivados" in snapshot["assign_notice"]
        item = next(f for f in snapshot["facts"] if f["id"] == graph.facts["auto"])
        assert (item["owner"], item["owner_revision"], item["owner_change"]) == (None, 0, None)
        body = await owner_body(graph, "auto", PERSON_A)
        response = await client.post(f"/api/memory/facts/{graph.facts['auto']}/owner", json=body)
        assert response.status_code == 200, response.text
        receipt = response.json()
        assert set(receipt) == {"fact_revision", "owner_revision", "deleted_dependents", "deleted_dependent_ids",
                                "confirmed_now", "derived_records_retired", "notice"}
        assert receipt["deleted_dependents"] == 0 and receipt["deleted_dependent_ids"] == []
        assert receipt["fact_revision"] == body["expected_fact_revision"] + 1 and receipt["owner_revision"] == 1
        assert receipt["confirmed_now"] is receipt["derived_records_retired"] is True
        assert "dejan de estar disponibles" in receipt["notice"]
        already = await client.post(f"/api/memory/facts/{graph.facts['held']}/owner",
                                    json=await owner_body(graph, "held", PERSON_A))
        assert already.json()["derived_records_retired"] is False and already.json()["notice"] is None
        after = (await client.get("/api/memory")).json()
        item = next(f for f in after["facts"] if f["id"] == graph.facts["auto"])
        assert item["state"] == "confirmed" and item["expires_at"] is None
        assert item["owner"]["person_id"] == PERSON_A and item["owner"]["method"] == "operator_assignment"
        # Same person, matching revisions: a no-op that changes nothing.
        before = graph.dump()
        again = await client.post(f"/api/memory/facts/{graph.facts['auto']}/owner",
                                  json=await owner_body(graph, "auto", PERSON_A))
        assert again.status_code == 200 and again.json()["owner_revision"] == 1 and graph.dump() == before


async def test_dependents_are_listed_with_states_and_deleted_only_with_the_flag(graph):
    await graph.store("parent", state="automatic")
    await graph.store("auto_child", state="automatic", parent="parent")
    await graph.store("confirmed_child", state="automatic", parent="parent")

    def confirm_by_sql():  # No API confirms a derived fact; the 1a W10 row builds it the same way.
        graph.memory.ledger._db.execute("UPDATE facts SET state='confirmed', expires_at=NULL WHERE id=?",
                                        (graph.facts["confirmed_child"],))
        graph.memory.ledger._db.commit()
    await graph.call(confirm_by_sql)
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        body = await owner_body(graph, "parent", PERSON_A)
        before = graph.dump()
        refused = await client.post(f"/api/memory/facts/{graph.facts['parent']}/owner", json=body)
        assert refused.status_code == 409 and graph.dump() == before
        detail = refused.json()["detail"]
        assert detail["code"] == "memory_owner_has_dependents" and "eliminarlos" in detail["notice"]
        assert detail["dependents"] == sorted(
            [{"id": graph.facts["auto_child"], "state": "automatic"},
             {"id": graph.facts["confirmed_child"], "state": "confirmed"}], key=lambda d: d["id"])
        accepted = await client.post(f"/api/memory/facts/{graph.facts['parent']}/owner",
                                     json={**body, "delete_dependents": True})
        assert accepted.status_code == 200 and accepted.json()["deleted_dependents"] == 2
        # The receipt names the same ids the 409 listed to this assigner, nothing else (gap R1).
        assert accepted.json()["deleted_dependent_ids"] == [entry["id"] for entry in detail["dependents"]]
    for name in ("auto_child", "confirmed_child"):
        assert await graph.call(graph.memory.ledger.get, graph.facts[name], review=True) is None


@pytest.mark.parametrize("case,status,detail", [
    ("missing", 404, "memory_fact_unavailable"),
    ("deleted", 404, "memory_fact_unavailable"),
    ("fact_conflict", 409, "memory_owner_conflict"),
    ("owner_conflict", 409, "memory_owner_conflict"),
    ("derived", 409, "memory_owner_derived_fact"),
    ("flag_on_confirmed", 422, "memory_owner_dependents_not_applicable"),
    ("invalid_person", 422, "memory_owner_invalid_person"),
    ("unknown_person", 422, "memory_owner_inactive"),
    ("retired_person", 422, "memory_owner_retired"),
    ("authority_missing", 503, "memory_owner_authority_missing"),
    ("authority_unknown", 503, "memory_owner_authority_unavailable"),
    # Review R2: a LookupError that is not the ledger's refusal is a 503, never 'fact missing'.
    ("lookup_defect", 503, "memory_unavailable"),
    # The ledger guard refuses (memory_assign_denied) and the delivery guard then
    # replaces any answer to a retired session with its fixed denial.
    ("session_retired", 403, "dashboard_session_retired"),
])
async def test_assignment_refusals_map_contracts_1a_and_write_nothing(graph, monkeypatch, case, status, detail):
    await graph.store("target")
    await graph.store("parent", state="automatic")
    await graph.store("derived", state="automatic", parent="parent")
    await graph.store("gone")
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        name, body = "target", await owner_body(graph, "target", PERSON_A)
        if case == "missing":
            graph.facts["missing"], name = "fact_" + "0" * 64, "missing"
        elif case == "deleted":
            await client.delete("/api/memory/facts/" + graph.facts["gone"])
            name, body = "gone", {**body}
        elif case == "fact_conflict":
            body["expected_fact_revision"] += 1
        elif case == "owner_conflict":
            body["expected_owner_revision"] = 3
        elif case == "derived":
            name, body = "derived", await owner_body(graph, "derived", PERSON_A)
        elif case == "flag_on_confirmed":
            body["delete_dependents"] = True
        elif case == "invalid_person":
            body["person_id"] = "user\u0007bell"
        elif case == "unknown_person":
            body["person_id"] = "user_" + "f" * 32
        elif case == "retired_person":
            await graph.authority.retire_user(graph.root, PERSON_C)
            body["person_id"] = PERSON_C
        elif case == "authority_missing":
            monkeypatch.setattr(memory_api, "dashboard_authority", lambda request: None)
        elif case == "authority_unknown":
            monkeypatch.setattr(graph.authority, "memory_owner_active", lambda person: None)
        elif case == "lookup_defect":
            def defect(*args, **kwargs):
                raise KeyError("synthetic defect")
            monkeypatch.setattr(graph.memory.ledger, "assign_owner", defect)
        elif case == "session_retired":
            original, current, ended = graph.authority.memory_owner_retired, graph.authority._current, []

            def revoke_then_check(identities):
                # The operator's session ends between the route and the commit guard.
                ended.append(True)
                return original(identities)
            monkeypatch.setattr(graph.authority, "memory_owner_retired", revoke_then_check)
            monkeypatch.setattr(graph.authority, "_current", lambda *a, **k: not ended and current(*a, **k))
        before = graph.dump()
        response = await client.post(f"/api/memory/facts/{graph.facts[name]}/owner", json=body)
    assert (response.status_code, response.json()["detail"]) == (status, detail)
    assert graph.dump() == before


@pytest.mark.parametrize("body", [
    None, [], "x", {}, {"person_id": PERSON_A}, {"person_id": PERSON_A, "expected_fact_revision": 1},
    {"person_id": PERSON_A, "expected_fact_revision": "1", "expected_owner_revision": 0},
    {"person_id": PERSON_A, "expected_fact_revision": 1.0, "expected_owner_revision": 0},
    {"person_id": PERSON_A, "expected_fact_revision": True, "expected_owner_revision": 0},
    {"person_id": PERSON_A, "expected_fact_revision": 0, "expected_owner_revision": 0},
    {"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": -1},
    {"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": 0, "delete_dependents": 1},
    {"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": 0, "extra": True},
    {"person_id": 7, "expected_fact_revision": 2, "expected_owner_revision": 0},
    {"person_id": None}, {"person_id": None, "expected_owner_revision": "0"},
    {"person_id": None, "expected_owner_revision": 0, "expected_fact_revision": 2},
])
async def test_malformed_owner_bodies_are_one_422(graph, body):
    await graph.store("target")
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        before = graph.dump()
        response = await client.post(f"/api/memory/facts/{graph.facts['target']}/owner",
                                     content=json.dumps(body), headers={"Content-Type": "application/json"})
        garbage = await client.post(f"/api/memory/facts/{graph.facts['target']}/owner", content=b"{",
                                    headers={"Content-Type": "application/json"})
    assert (response.status_code, response.json()) == (422, {"detail": "memory_owner_invalid_request"})
    assert (garbage.status_code, garbage.json()) == (422, {"detail": "memory_owner_invalid_request"})
    assert graph.dump() == before


async def test_operator_unassign_does_not_enter_devueltos(graph):
    await graph.store("target")
    await graph.assign("target", PERSON_A)
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        stale = await client.post(f"/api/memory/facts/{graph.facts['target']}/owner",
                                  json={"person_id": None, "expected_owner_revision": 0})
        assert (stale.status_code, stale.json()["detail"]) == (409, "memory_owner_conflict")
        response = await client.post(f"/api/memory/facts/{graph.facts['target']}/owner",
                                     json={"person_id": None, "expected_owner_revision": 1})
        assert response.status_code == 200 and response.json()["owner_revision"] == 2
        assert (await client.get("/api/memory/returned")).json()["facts"] == []
        counts = (await client.get("/api/memory/owners/counts")).json()
        assert (counts["currently_owned"], counts["returned"], counts["ever_assigned_live"]) == (0, 0, 1)
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A) as client:
        assert (await client.get("/api/memory/mine")).json()["facts"] == []


async def test_counts_expose_the_retired_owner_prune_health_signal(graph):
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        assert (await client.get("/api/memory/owners/counts")).json()["retired_owner_prune_pending"] is False
        graph.memory.ledger.retired_owner_prune_pending = True
        counts = (await client.get("/api/memory/owners/counts")).json()
    assert counts["retired_owner_prune_pending"] is True
    assert set(counts) == {"currently_owned", "returned", "ever_assigned_live", "owned_proposals",
                           "retired_owner_prune_pending"}


# --- keyset pagination over HTTP --------------------------------------------------------------


async def test_operator_list_keyset_reaches_every_memory_and_keeps_the_u3_first_page(graph):
    await graph.store("old")
    await graph.assign("old", PERSON_A)
    for index in range(MEMORY_PAGE_LIMIT + 3):
        graph.clock.now += 1
        await graph.store(f"n{index}", state="automatic")
    async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
        first = (await client.get("/api/memory")).json()
        legacy = await graph.call(graph.memory.ledger.snapshot)
        assert [f["id"] for f in first["facts"]] == [f["id"] for f in legacy["facts"]]
        assert len(first["facts"]) == MEMORY_PAGE_LIMIT and first["facts_next"] and first["proposals_next"] is None
        seen, cursor = [f["id"] for f in first["facts"]], first["facts_next"]
        while cursor:
            page = (await client.get("/api/memory", params={"facts_after": cursor, "limit": "2"})).json()
            assert not {f["id"] for f in page["facts"]} & set(seen), "a page repeated an item"
            seen += [f["id"] for f in page["facts"]]
            cursor = page["facts_next"]
        for query in ({"facts_after": "x"}, {"limit": "0"}, {"limit": "101"}, {"limit": "01"},
                      {"proposals_after": first["facts_next"][:-1]}, [("limit", "1"), ("limit", "2")]):
            bad = await client.get("/api/memory", params=query)
            assert (bad.status_code, bad.json()) == (422, {"detail": "memory_page_invalid"}), query
    assert len(seen) == len(set(seen)) == MEMORY_PAGE_LIMIT + 4 and seen[-1] == graph.facts["old"]


async def test_mine_keyset_boundary_at_equal_time_and_tampered_cursor(graph):
    for index in range(5):
        await graph.store(f"same{index}")  # one instant: ordering falls back to the id
        await graph.assign(f"same{index}", PERSON_A)
    expected = sorted(graph.facts[f"same{index}"] for index in range(5))
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A) as client:
        seen, cursor = [], None
        while True:
            page = (await client.get("/api/memory/mine", params={"limit": "2", **({"after": cursor} if cursor else {})})).json()
            assert not {item["id"] for item in page["facts"]} & set(seen), "a page repeated an item"
            seen += [item["id"] for item in page["facts"]]
            cursor = page["next"]
            if cursor is None:
                break
        assert seen == expected
        first = (await client.get("/api/memory/mine", params={"limit": "2"})).json()["next"]
        for tampered in (first.replace("k1.", "k2."), first[:-1] + "Z", first + "0", "k1." + "0" * 16 + ".fact_",
                         first.upper()):
            response = await client.get("/api/memory/mine", params={"after": tampered})
            assert (response.status_code, response.json()) == (422, {"detail": "memory_page_invalid"})
        for extra in ({"person_id": PERSON_B}, {"scope": "all"}, {"user_id": PERSON_B}):
            response = await client.get("/api/memory/mine", params=extra)
            assert (response.status_code, response.json()) == (422, {"detail": "memory_page_invalid"})


# --- sessions ---------------------------------------------------------------------------------


async def test_pre_upgrade_home_session_is_reminted_not_denied(graph):
    await graph.store("mine")
    await graph.assign("mine", PERSON_A)
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A, legacy_permissions=True) as client:
        whoami = (await client.get("/api/admin/auth/whoami")).json()
        assert OWN <= set(whoami["permissions"])  # recomputed from role + user id, never stored
        response = await client.get("/api/memory/mine")
        assert response.status_code == 200 and [i["id"] for i in response.json()["facts"]] == [graph.facts["mine"]]
    # Nothing about the memory permissions is persisted; they are derived on every resolve.
    with sqlite3.connect(graph.authority.path) as db:
        assert "memory:own" not in json.dumps(db.execute("SELECT * FROM sessions").fetchall())
        assert "memory:own" not in json.dumps(db.execute("SELECT * FROM credentials").fetchall())


BINDING = re.compile(r"[0-9a-f]{32}")


async def test_whoami_binding_tells_grants_and_persons_apart_without_naming_either(graph):
    """Gap R3: an opaque per-grant tag, only for personal sessions; never the person or credential id."""
    shared = "synthetic-home-shared"
    first, other_person, same_person = (await graph.credential(DashboardRole.HOME, shared, PERSON_A),
                                        await graph.credential(DashboardRole.HOME, shared, PERSON_B),
                                        await graph.credential(DashboardRole.HOME, shared, PERSON_A))
    seen = {}
    for name, grant in (("first", first), ("again", first), ("other_person", other_person), ("same_person", same_person)):
        async with graph.client(grant=grant) as client:
            response = await client.get("/api/admin/auth/whoami")
            body = response.json()
            assert body["subject"] == shared and body["personal_scope"] is True
            assert BINDING.fullmatch(body["account_binding"]), body
            # Neither the person nor the credential is named, in the value or anywhere in the answer.
            for secret in (PERSON_A, PERSON_B, first.identity, other_person.identity, same_person.identity,
                           first.token, other_person.token, same_person.token):
                assert secret not in response.text
            seen[name] = body["account_binding"]
    assert seen["first"] == seen["again"]  # stable across logins of the same grant
    assert len({seen["first"], seen["other_person"], seen["same_person"]}) == 3
    # Rebinding the same credential to another person changes the tag.
    revised = await graph.authority.revise_credential(graph.root, first.identity, role=DashboardRole.HOME,
                                                      expires_at=graph.clock.now + 7200, user_id=PERSON_B)
    assert revised is True
    async with graph.client(grant=first) as client:
        rebound = (await client.get("/api/admin/auth/whoami")).json()["account_binding"]
    assert BINDING.fullmatch(rebound) and rebound not in seen.values()
    # No person, no tag: operators and household logins without a person.
    for role, person in ((DashboardRole.OPERATOR, None), (DashboardRole.HOME, None)):
        async with graph.client(role, shared) as client:
            body = (await client.get("/api/admin/auth/whoami")).json()
            assert body["account_binding"] is None and body["personal_scope"] is False


def _binding_uses(path: Path) -> list[tuple[str, str]]:
    """Every mention of account_binding in a module, as (kind, enclosing class.function)."""
    uses = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = scope + (child.name,)
                if child.name == "account_binding":
                    uses.append(("def", ".".join(scope) or "<module>"))
            if isinstance(child, ast.Name) and child.id == "account_binding":
                uses.append(("name", ".".join(scope)))
            elif isinstance(child, ast.Attribute) and child.attr == "account_binding":
                uses.append(("attribute", ".".join(scope)))
            elif isinstance(child, ast.keyword) and child.arg == "account_binding":
                uses.append(("keyword", ".".join(scope)))
            elif isinstance(child, ast.Constant) and child.value == "account_binding":
                uses.append(("string", ".".join(scope)))
            visit(child, inner)

    visit(ast.parse(path.read_text(encoding="utf-8")), ())
    return sorted(uses)


def test_account_binding_is_metadata_only():
    """Review P: nothing compares, stores or authorizes on the tag; every use is allow-listed by AST.

    authority.py may only define it (the helper and the DashboardPrincipal field) and compute it once
    in resolve(); admin.py may only project principal.account_binding in whoami. Nothing else reads it.
    """
    users = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.py")
                   if "account_binding" in path.read_text(encoding="utf-8"))
    assert users == ["src/dashboard/admin.py", "src/dashboard/authority.py"]
    assert _binding_uses(ROOT / "src/dashboard/authority.py") == [
        ("def", "<module>"),                              # def account_binding(...)
        ("name", "DashboardAuthority.resolve"),           # the one call that sets it
        ("name", "DashboardPrincipal"),                   # the dataclass field
    ]
    assert _binding_uses(ROOT / "src/dashboard/admin.py") == [
        ("attribute", "register_admin_routes.whoami"), ("string", "register_admin_routes.whoami")]


async def test_operator_who_is_also_household_has_two_independent_sessions(graph):
    await graph.store("c_fact")
    await graph.assign("c_fact", PERSON_C)
    await graph.store("household")
    async with graph.client(DashboardRole.OPERATOR, "synthetic-person-c-admin") as operator, \
            graph.client(DashboardRole.HOME, "synthetic-person-c-home", PERSON_C) as home:
        denied = await operator.get("/api/memory/mine")
        assert (denied.status_code, denied.json()["detail"]) == (403, "dashboard_permission_denied")
        assert not OWN & set((await operator.get("/api/admin/auth/whoami")).json()["permissions"])
        mine = await home.get("/api/memory/mine")
        assert [i["id"] for i in mine.json()["facts"]] == [graph.facts["c_fact"]]
        assert leaked(graph, mine, ["c_fact"]) == []
        for path in ("/api/memory", "/api/memory/returned", "/api/memory/owners/counts"):
            response = await home.get(path)
            assert (response.status_code, response.json()["detail"]) == (403, "dashboard_permission_denied")
        assert (await operator.get("/api/memory")).status_code == 200


async def test_retired_account_session_is_cut_before_any_read(graph):
    await graph.store("mine")
    await graph.assign("mine", PERSON_A)
    async with graph.client(DashboardRole.HOME, "synthetic-home-a", PERSON_A) as client:
        assert (await client.get("/api/memory/mine")).status_code == 200
        graph.accounts.discard(PERSON_A)
        response = await client.get("/api/memory/mine")
        assert (response.status_code, response.json()["detail"]) == (401, "auth required")
        assert leaked(graph, response) == []


# --- the binding comes only from the authenticated account ------------------------------------


def test_person_binding_comes_only_from_the_credential_account():
    """Voice/persona links never grant memory permissions or select the person."""
    import inspect
    from src.dashboard import authority
    assert list(inspect.signature(authority._permissions).parameters) == ["role", "user_id"]
    tree = ast.parse((ROOT / "src/dashboard/authority.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {module for module in imported if module and ("voice" in module or "speaker" in module)}
    source = (ROOT / "src/memory/fact_ledger.py").read_text(encoding="utf-8")
    for name in ("MemoryOwnerActor", "MemoryOwnerReader"):
        cls = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == name)
        mint = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_dashboard")
        assert "principal.user_id" in ast.unparse(mint)
    api = ast.parse((ROOT / "src/dashboard/memory_api.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(api) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(api) if isinstance(n, ast.Attribute)}
    # U3b-4a-2 (N5): the return route only chooses its notice from the ledger receipt; no voice name binds a person.
    assert {name for name in names if "voice" in name.lower() or "speaker" in name.lower()} == {
        "VOICE_RETURN_NOTICE", "voice_proposal"}
    # The person is read from the minted actor only, never from the request.
    person_reads = [ast.unparse(n) for n in ast.walk(api) if isinstance(n, ast.Attribute) and n.attr == "person_id"]
    assert person_reads == ["actor.person_id"]


async def test_household_without_person_cannot_name_one(graph):
    await graph.store("mine")
    await graph.assign("mine", PERSON_A)
    async with graph.client(DashboardRole.HOME, "synthetic-home-none") as client:
        for request in (client.get("/api/memory/mine", params={"person_id": PERSON_A}),
                        client.get("/api/memory/mine", headers={"X-KZA-Person": PERSON_A}),
                        client.post(f"/api/memory/mine/facts/{graph.facts['mine']}/return",
                                    json={"person_id": PERSON_A})):
            response = await request
            assert (response.status_code, response.json()["detail"]) == (403, "dashboard_permission_denied")
            assert leaked(graph, response) == []


# --- live channels ----------------------------------------------------------------------------


async def _socket(graph, path, token):
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    scope = {"type": "websocket", "path": path, "raw_path": path.encode(), "root_path": "", "scheme": "ws",
             "query_string": b"", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
             "headers": [(b"authorization", ("Bearer " + token).encode())], "subprotocols": [], "state": {}}
    await incoming.put({"type": "websocket.connect"})
    task = asyncio.create_task(graph.api.app(scope, incoming.get, outgoing.put))
    return incoming, outgoing, task


async def test_live_channels_carry_no_memory(graph):
    """/ws admits a person (home:read) but memory changes publish nothing; /ws/live is operator-only."""
    await graph.store("a_fact")
    grant = await graph.credential(DashboardRole.HOME, "synthetic-home-a", PERSON_A)
    incoming, outgoing, task = await _socket(graph, "/ws", grant.token)
    try:
        assert (await asyncio.wait_for(outgoing.get(), 3))["type"] == "websocket.accept"
        await graph.assign("a_fact", PERSON_A)
        async with graph.client(DashboardRole.HOME, "synthetic-home-a2", PERSON_A) as client:
            await client.post(f"/api/memory/mine/facts/{graph.facts['a_fact']}/return")
        async with graph.client(DashboardRole.OPERATOR, "synthetic-operator") as client:
            await client.delete("/api/memory/facts/" + graph.facts["a_fact"])
        await incoming.put({"type": "websocket.receive", "text": json.dumps({"type": "ping"})})
        frame = await asyncio.wait_for(outgoing.get(), 3)
        assert json.loads(frame["text"]) == {"type": "pong"} and outgoing.empty()
    finally:
        await incoming.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(task, 3)
    # /ws/live and /ws/voices are registered only with the event bus (not in this graph);
    # their route permissions are operator-only, which a person-bound session never holds.
    from src.dashboard.auth import _route_permissions
    from src.dashboard.authority import _permissions
    home = _permissions(DashboardRole.HOME, PERSON_A)
    for path in ("/ws/live", "/ws/voices"):
        scope = SimpleNamespace(url=SimpleNamespace(path=path), scope={"method": "GET"},
                                query_params={})
        assert not _route_permissions(scope) <= home
    tree = ast.parse((ROOT / "src/dashboard/memory_api.py").read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and "broadcast" in n.attr]
