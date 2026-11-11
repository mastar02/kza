"""U3b-2 server-side denial matrix (>= 200 rows) through real logins and real routes.

Every row is a refusal or a no-op. For each row the exact status and body are
asserted, the committed privacy tables must be unchanged, and no seeded fact
content or id may appear. Rows on another person's fact must be byte-identical
to the missing-id row of the same actor and route, run the same SQL
statements, and change nothing.
"""

from contextlib import AsyncExitStack
from dataclasses import dataclass
import json
import re

from src.dashboard.authority import DashboardRole
from tests.mocks.memory_owner_graph import PERSON_A, PERSON_B, PERSON_C, owner_graph

PERSON_D = "user_" + "d" * 32
MISSING = "fact_" + "0" * 64
DENIED = {"detail": "dashboard_permission_denied"}
UNAUTHENTICATED = {"detail": "auth required"}
NOT_FOUND = {"detail": "Not Found"}
NOT_ALLOWED = {"detail": "Method Not Allowed"}
INVALID_ID = {"detail": "memory_fact_invalid"}
INVALID_PAGE = {"detail": "memory_page_invalid"}
UNAVAILABLE = {"detail": "memory_fact_unavailable"}
# Named facts and who owns them after seeding (None: nobody).
OWNED = {"own_a": PERSON_A, "own_a_auto": PERSON_A, "own_b": PERSON_B, "own_c": PERSON_C}
OTHER_FACTS = ["household", "automatic", "proposal", "returned", "unassigned_by_operator", "deleted",
               "tombstoned", "expired", "derived", "missing"]
PERSON_ACTORS = {"home_a": PERSON_A, "home_b": PERSON_B, "home_c": PERSON_C, "home_a_pre_upgrade": PERSON_A}


@dataclass(frozen=True)
class Row:
    """One request and its exact expected answer."""

    actor: str
    method: str
    path: str  # '{name}' placeholders are replaced by seeded fact ids
    status: int
    body: dict | None
    raw_path: str | None = None
    payload: dict | None = None
    group: str | None = None  # rows of one group must be indistinguishable

    @property
    def name(self) -> str:
        return f"{self.actor} {self.method} {self.raw_path or self.path}"


def _person_rows() -> list[Row]:
    rows = []
    for actor, person in PERSON_ACTORS.items():
        foreign = [name for name, owner in OWNED.items() if owner != person] + OTHER_FACTS
        for name in foreign:
            rows.append(Row(actor, "DELETE", "/api/memory/mine/facts/{%s}" % name, 200, {"deleted": False},
                            group=f"{actor}:delete"))
            rows.append(Row(actor, "POST", "/api/memory/mine/facts/{%s}/return" % name, 200, {"returned": False},
                            group=f"{actor}:return"))
    return rows


PERSON_ROUTES = [("GET", "/api/memory/mine", None), ("DELETE", "/api/memory/mine/facts/{own_a}", None),
                 ("POST", "/api/memory/mine/facts/{own_a}/return", None)]
OPERATOR_ROUTES = [
    ("GET", "/api/memory", None), ("GET", "/api/memory/returned", None), ("GET", "/api/memory/owners/counts", None),
    ("POST", "/api/memory/facts/{own_a}/owner", {"person_id": PERSON_B, "expected_fact_revision": 2,
                                                  "expected_owner_revision": 1}),
    ("POST", "/api/memory/facts/{household}/owner", {"person_id": None, "expected_owner_revision": 0}),
    ("GET", "/api/memory/facts", None), ("GET", "/api/memory/proposals", None),
    ("POST", "/api/memory/proposals/{proposal}/confirm", None), ("POST", "/api/memory/proposals/{proposal}/reject", None),
    ("DELETE", "/api/memory/facts/{own_a}", None),
]


def _role_rows() -> list[Row]:
    rows = []
    for actor, body, status in (("anonymous", UNAUTHENTICATED, 401), ("home_none", DENIED, 403),
                                ("evaluator", DENIED, 403), ("home_d_retired", UNAUTHENTICATED, 401),
                                ("home_revised_b_to_a", UNAUTHENTICATED, 401)):
        rows += [Row(actor, method, path, status, body, payload=payload)
                 for method, path, payload in PERSON_ROUTES + OPERATOR_ROUTES]
    for actor in ("operator", "operator_c", "legacy_bearer"):
        rows += [Row(actor, method, path, 403, DENIED) for method, path, _ in PERSON_ROUTES]
    for actor in PERSON_ACTORS:
        rows += [Row(actor, method, path, 403, DENIED, payload=payload) for method, path, payload in OPERATOR_ROUTES]
    return rows


INVALID_IDS = ["fact_xyz", "FACT_" + "a" * 64, "fact_" + "A" * 64, "fact_" + "a" * 63, "fact_" + "a" * 65,
               "fact_" + "g" * 64, "fact_" + "a" * 64 + "x", "fact_" + "a" * 64 + "%20"]


def _tricky_rows() -> list[Row]:
    # Paths no API route matches fall to the static frontend mount at '/': no handler,
    # no permission branch, no data. It answers 404 to GET/HEAD and 405 otherwise, the
    # same for every actor (anonymous rows included), so nothing reaches an operator route.
    rows = []
    for actor in ("home_a", "operator", "anonymous"):
        rows += [
            Row(actor, "GET", "/api/memory/minefoo", 404, NOT_FOUND),
            Row(actor, "GET", "/api/memory/MINE", 404, NOT_FOUND),
            Row(actor, "GET", "/api/memory/mine/unknown", 404, NOT_FOUND),
            Row(actor, "GET", "/api/memory/mine/../facts", 404, NOT_FOUND, raw_path="/api/memory/mine/../facts"),
            Row(actor, "DELETE", "/api/memory/mine/../facts/{own_b}", 405, NOT_ALLOWED,
                raw_path="/api/memory/mine/../facts/{own_b}"),
            Row(actor, "POST", "/api/memory/mine/../proposals/{proposal}/confirm", 405, NOT_ALLOWED,
                raw_path="/api/memory/mine/../proposals/{proposal}/confirm"),
            Row(actor, "POST", "/api/memory/mine/facts/{own_a}/owner", 405, NOT_ALLOWED,
                payload={"person_id": PERSON_B, "expected_fact_revision": 2, "expected_owner_revision": 1}),
            Row(actor, "GET", "/api/memory/mine/facts/{own_a}", 404, NOT_FOUND),
        ]
    rows += [
        Row("home_a", "GET", "/api/memory/mine/facts", 404, NOT_FOUND),
        # An encoded slash or letter is decoded by the server before routing and permission:
        # it behaves exactly as the plain path (the person's own branch).
        Row("home_a", "DELETE", "/api/memory/mine/facts/{own_b}", 200, {"deleted": False},
            raw_path="/api/memory/mine%2Ffacts/{own_b}"),
        Row("home_a", "DELETE", "/api/memory/mine/facts/{own_b}", 200, {"deleted": False},
            raw_path="/api/memory/%6Dine/facts/{own_b}"),
        Row("operator", "DELETE", "/api/memory/mine/facts/{own_b}", 403, DENIED,
            raw_path="/api/memory/mine%2Ffacts/{own_b}"),
        Row("home_a", "POST", "/api/memory/mine/facts/{own_a}/delete", 405, NOT_ALLOWED),
        Row("home_a", "GET", "/api/memory/mine/facts/{own_a}/return", 404, NOT_FOUND),
        Row("home_a", "PUT", "/api/memory/mine/facts/{own_a}", 405, NOT_ALLOWED),
        Row("home_a", "PATCH", "/api/memory/mine/facts/{own_a}", 405, NOT_ALLOWED),
        Row("home_a", "HEAD", "/api/memory/mine", 404, None),
    ]
    rows += [Row("home_a", method, "/api/memory/mine", 405, NOT_ALLOWED)
             for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS")]
    for value in INVALID_IDS:
        rows.append(Row("home_a", "DELETE", "/api/memory/mine/facts/" + value, 422, INVALID_ID))
        rows.append(Row("home_a", "POST", f"/api/memory/mine/facts/{value}/return", 422, INVALID_ID))
        rows.append(Row("operator", "POST", f"/api/memory/facts/{value}/owner", 422, INVALID_ID,
                        payload={"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": 0}))
    for query in ("after=bad", "after=k1.41ddcd6500000000.fact_" + "0" * 63, "limit=0", "limit=101", "limit=abc",
                  "limit=-1", "limit=1.5", "limit=1&limit=2", "scope=all", "person_id=" + PERSON_B,
                  "user_id=" + PERSON_B, "after=&limit=1"):
        rows.append(Row("home_a", "GET", "/api/memory/mine?" + query, 422, INVALID_PAGE))
        rows.append(Row("operator", "GET", "/api/memory/returned?" + query, 422, INVALID_PAGE))
    rows += [
        Row("operator", "GET", "/api/memory/facts/{own_a}/owner", 404, NOT_FOUND),
        Row("operator", "DELETE", "/api/memory/facts/{own_a}/owner", 405, NOT_ALLOWED),
        Row("operator", "PUT", "/api/memory/facts/{own_a}/owner", 405, NOT_ALLOWED),
        Row("home_a", "POST", "/api/memory/facts/{own_b}/owner", 403, DENIED,
            payload={"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": 1}),
    ]
    # Expired, tombstoned and deleted facts answer exactly like a missing id to the operator too.
    for name in ("missing", "deleted", "tombstoned", "expired"):
        rows.append(Row("operator", "POST", "/api/memory/facts/{%s}/owner" % name, 404, UNAVAILABLE,
                        payload={"person_id": PERSON_A, "expected_fact_revision": 2, "expected_owner_revision": 0},
                        group="operator:assign-unavailable"))
        rows.append(Row("operator", "POST", "/api/memory/facts/{%s}/owner" % name, 404, UNAVAILABLE,
                        payload={"person_id": None, "expected_owner_revision": 0},
                        group="operator:unassign-unavailable"))
    return rows


ROWS = _person_rows() + _role_rows() + _tricky_rows()


def test_matrix_has_at_least_200_distinct_rows():
    assert len(ROWS) >= 200
    assert len({(r.actor, r.method, r.raw_path or r.path, json.dumps(r.payload)) for r in ROWS}) == len(ROWS)
    print(f"U3B2_MATRIX_ROWS={len(ROWS)}")


async def _seed(graph):
    graph.accounts.add(PERSON_D)
    expired = await graph.store("expired", state="automatic")
    graph.clock.now += 31 * 86400  # past the 30-day automatic retention of "expired"
    for name, person in OWNED.items():
        graph.clock.now += 1
        await graph.store(name, state="automatic" if name == "own_a_auto" else "confirmed")
        await graph.assign(name, person)
    await graph.store("derived", state="automatic", parent="own_a")
    await graph.store("household")
    await graph.store("automatic", state="automatic")
    await graph.store("proposal", state="proposed")
    for name in ("returned", "unassigned_by_operator", "deleted"):
        await graph.store(name)
        await graph.assign(name, PERSON_B if name == "returned" else PERSON_A)
    ledger = graph.memory.ledger
    from src.memory.fact_ledger import MemoryOwnerActor
    assert await graph.call(ledger.return_owned, graph.facts["returned"], PERSON_B,
                            MemoryOwnerActor("synthetic-seed-home-b", PERSON_B))
    from tests.mocks.memory_owner_graph import SEED_ASSIGNER, SEED_REVIEWER
    await graph.call(ledger.unassign_owner, graph.facts["unassigned_by_operator"], 1, SEED_ASSIGNER)
    await graph.call(ledger.delete_fact, graph.facts["deleted"], SEED_REVIEWER)
    conversation = "6f9619ff-8b86-4011-b42d-00cf4fc964ff"
    await graph.store("tombstoned", conversation=conversation)
    await graph.assign("tombstoned", PERSON_A)
    await graph.call(ledger.delete_conversation, conversation, SEED_REVIEWER)
    graph.facts["missing"], graph.contents["missing"] = MISSING, MISSING
    assert await graph.call(ledger.get, expired, review=True) is None


async def _clients(graph, stack, dashboard_auth):
    clients = {"anonymous": await stack.enter_async_context(graph.client())}
    for actor, role, subject, person in (
            ("home_none", DashboardRole.HOME, "synthetic-home-none", None),
            ("evaluator", DashboardRole.EVALUATOR, "synthetic-evaluator", None),
            ("home_a", DashboardRole.HOME, "synthetic-home-a", PERSON_A),
            ("home_b", DashboardRole.HOME, "synthetic-home-b", PERSON_B),
            ("home_c", DashboardRole.HOME, "synthetic-person-c-home", PERSON_C),
            ("operator", DashboardRole.OPERATOR, "synthetic-operator", None),
            ("operator_c", DashboardRole.OPERATOR, "synthetic-person-c-admin", None),
            ("home_d_retired", DashboardRole.HOME, "synthetic-home-d", PERSON_D),
            ("home_revised_b_to_a", DashboardRole.HOME, "synthetic-home-b-revised", PERSON_B)):
        clients[actor] = await stack.enter_async_context(graph.client(role, subject, person))
    clients["home_a_pre_upgrade"] = await stack.enter_async_context(
        graph.client(DashboardRole.HOME, "synthetic-home-a-old", PERSON_A, legacy_permissions=True))
    clients["legacy_bearer"] = await stack.enter_async_context(graph.client(bearer=dashboard_auth["Authorization"]))
    await graph.authority.retire_user(graph.root, PERSON_D)
    revised = clients["home_revised_b_to_a"].grant
    assert await graph.authority.revise_credential(graph.root, revised.identity, role=DashboardRole.HOME,
                                                   expires_at=graph.clock.now + 7200, user_id=PERSON_A)
    return clients


async def _raw(graph, client, method, path, raw_path, payload):
    """Send one request without client-side URL normalisation (dot segments, encodings)."""
    headers = [(b"host", b"testserver")]
    if "Authorization" in client.headers:
        headers.append((b"authorization", client.headers["Authorization"].encode()))
    cookie = "; ".join(f"{name}={value}" for name, value in client.cookies.items())
    if cookie:
        headers.append((b"cookie", cookie.encode()))
    body = b"" if payload is None else json.dumps(payload).encode()
    if payload is not None:
        headers.append((b"content-type", b"application/json"))
    target, _, query = raw_path.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path.partition("?")[0], "raw_path": target.encode(), "query_string": query.encode(),
             "root_path": "", "headers": headers, "server": ("testserver", 80), "client": ("127.0.0.1", 1)}
    sent, delivered = [], False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)
    await graph.api.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = sorted((k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in start.get("headers", []))
    return start["status"], b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"), headers


async def _send(graph, clients, row):
    def fill(text):
        return re.sub(r"\{(\w+)\}", lambda m: graph.facts[m.group(1)], text)
    client = clients[row.actor]
    if row.raw_path is not None:
        return await _raw(graph, client, row.method, fill(row.path), fill(row.raw_path), row.payload)
    kwargs = {} if row.payload is None else {"json": row.payload}
    response = await client.request(row.method, fill(row.path), **kwargs)
    return response.status_code, response.content, sorted((k.lower(), v) for k, v in response.headers.items())


async def test_denial_matrix(tmp_path, dashboard_auth):
    async with owner_graph(tmp_path, dashboard_auth) as graph, AsyncExitStack() as stack:
        await _seed(graph)
        clients = await _clients(graph, stack, dashboard_auth)
        failures, groups = [], {}
        ledger = graph.memory.ledger
        for row in ROWS:
            before = graph.dump()
            statements = []
            await graph.call(ledger._db.set_trace_callback, statements.append)
            try:
                status, content, headers = await _send(graph, clients, row)
            finally:
                await graph.call(ledger._db.set_trace_callback, None)
            text = content.decode("utf-8", "replace")
            body = json.loads(content) if content else None
            leaked = [name for name in graph.facts if name != "missing"
                      and (graph.contents[name] in text or graph.facts[name] in text)]
            if (status, body) != (row.status, row.body):
                failures.append(f"{row.name}: got {status} {text[:160]!r}, expected {row.status} {row.body}")
            if graph.dump() != before:
                failures.append(f"{row.name}: the privacy tables changed")
            if leaked:
                failures.append(f"{row.name}: leaked {leaked}")
            if row.group is not None:
                normalised = [re.sub(r"fact_id='[^']*'", "fact_id='<id>'", re.sub(r"fact_[0-9a-f]{64}", "<id>", s))
                              for s in statements]
                groups.setdefault(row.group, []).append((row.name, status, content, normalised, headers))
        for group, members in groups.items():
            _, status, content, statements, headers = next(m for m in members if "{missing}" in m[0])
            assert ("cache-control", "no-store") in headers, group
            for name, other_status, other_content, other_statements, other_headers in members:
                if (other_status, other_content) != (status, content):
                    failures.append(f"{name}: body differs from the missing id in {group}")
                if other_headers != headers:  # review R4: every header, including length and cache policy
                    failures.append(f"{name}: headers differ from the missing id in {group}")
                if group.startswith("home") and other_statements != statements:
                    failures.append(f"{name}: SQL differs from the missing id in {group}")
        assert failures == [], "\n".join(failures)
        # Positive controls on the same graph: the denials above are not vacuous.
        mine = (await clients["home_a"].get("/api/memory/mine")).json()["facts"]
        assert {item["id"] for item in mine} == {graph.facts["own_a"], graph.facts["own_a_auto"]}
        pre = (await clients["home_a_pre_upgrade"].get("/api/memory/mine")).json()["facts"]
        assert {item["id"] for item in pre} == {graph.facts["own_a"], graph.facts["own_a_auto"]}
        assert (await clients["operator"].get("/api/memory/owners/counts")).status_code == 200
        assert (await clients["legacy_bearer"].get("/api/memory/returned")).json()["facts"][0]["id"] == \
            graph.facts["returned"]
        deleted = await clients["home_b"].delete("/api/memory/mine/facts/" + graph.facts["own_b"])
        assert deleted.json() == {"deleted": True}
