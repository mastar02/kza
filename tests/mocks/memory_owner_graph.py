"""Real dashboard routes, authority and memory services for U3b-2 owner tests.

Three synthetic accounts (A, B, C). C is also the person behind an operator
credential, to exercise the operator-who-is-household case. Facts are seeded
through the real ledger; assignments use the authority's wired callables.
"""

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import sqlite3
from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from src.core.account_metadata import ActiveAccounts
from src.dashboard import authority as authority_module
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from src.memory.fact_ledger import FactDependency, MemoryAssigner, MemoryOrigin, MemoryReviewer, MemorySource
from src.memory.services import build_memory_services
from tests.mocks.memory_graph import Clock, MemoryClient

PERSON_A, PERSON_B, PERSON_C = ("user_" + c * 32 for c in "abc")
SEED_ASSIGNER = MemoryAssigner("synthetic-seed-operator")
SEED_REVIEWER = MemoryReviewer("synthetic-seed-reviewer")
LEDGER_TABLES = ("facts", "fact_sources", "fact_dependencies", "fact_owners", "fact_owner_revisions",
                 "index_jobs", "policy_index_deletions", "conversation_tombstones")


_U3B2_PERMISSIONS = authority_module._permissions


def pre_u3b2_permissions(role, user_id):
    """The U3b-1b permission function, before memory:assign / memory:own:* existed."""
    granted = _U3B2_PERMISSIONS(role, user_id)
    return frozenset(granted - {"memory:assign", "memory:own:read", "memory:own:write"})


@dataclass
class OwnerGraph:
    """Handles for one graph; ``facts`` maps a name to a fact id."""

    authority: DashboardAuthority
    api: DashboardAPI
    memory: object
    clock: Clock
    root: object
    accounts: set
    path: str
    facts: dict = field(default_factory=dict)
    contents: dict = field(default_factory=dict)
    owners: dict = field(default_factory=dict)

    def source(self, ttl: float = 24 * 3600, conversation: str | None = None, dependencies=()) -> MemorySource:
        now = self.clock.now
        return MemorySource(str(uuid4()), conversation or str(uuid4()), now - 1, now + ttl, MemoryOrigin.VOICE,
                            tuple(dependencies))

    async def call(self, function, *args, **kwargs):
        return await self.memory.acall(function, *args, **kwargs)

    async def store(self, name: str, *, state: str = "confirmed", conversation: str | None = None,
                    parent: str | None = None) -> str:
        """Store one synthetic fact whose content names it, so leaks are searchable."""
        ledger = self.memory.ledger
        content = f"Synthetic private memory {name} {uuid4().hex[:8]}"
        dependencies = ()
        if parent is not None:
            fact = await self.call(ledger.get, self.facts[parent], review=True)
            dependencies = (FactDependency(fact.id, fact.revision, fact.expires_at),)
        source = self.source(conversation=conversation, dependencies=dependencies)

        def create():
            stored = ledger.store(content, "fact", .9 if state == "automatic" else 1.0, (source,),
                                  proposed=state in ("proposed", "confirmed"))
            return ledger.confirm(stored.id, SEED_REVIEWER) if state == "confirmed" else stored
        fact = await self.call(create)
        self.facts[name], self.contents[name] = fact.id, content
        return fact.id

    async def assign(self, name: str, person: str, **kwargs) -> object:
        ledger = self.memory.ledger
        fact = await self.call(ledger.get, self.facts[name], review=True)
        current = await self.call(lambda: ledger._db.execute(
            "SELECT revision FROM fact_owner_revisions WHERE fact_id=?", (fact.id,)).fetchone())
        receipt = await self.call(ledger.assign_owner, fact.id, person, fact.revision,
                                  0 if current is None else current[0], SEED_ASSIGNER,
                                  retired=self.authority.memory_owner_retired,
                                  active=self.authority.memory_owner_active, **kwargs)
        self.owners[name] = person
        return receipt

    def dump(self) -> dict:
        """Every privacy table row, read from committed SQL by a separate connection."""
        with sqlite3.connect(self.path) as db:
            return {table: sorted(map(tuple, db.execute("SELECT * FROM " + table))) for table in LEDGER_TABLES}

    async def credential(self, role: DashboardRole, subject: str, user_id: str | None = None):
        now = self.clock.now
        samples = (EvaluationSample("vf_" + "e" * 64, frozenset({"f" * 64}),
                                    frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 3600),) \
            if role is DashboardRole.EVALUATOR else ()
        return await self.authority.issue_credential(self.root, subject=subject, role=role, expires_at=now + 7200,
                                                     samples=samples, user_id=user_id)

    @asynccontextmanager
    async def client(self, role: DashboardRole | None = None, subject: str = "", user_id: str | None = None, *,
                     bearer: str | None = None, legacy_permissions: bool = False, grant=None):
        """A logged-in browser-like client (cookie session), a bearer client, or anonymous."""
        async with AsyncClient(transport=ASGITransport(app=self.api.app), base_url="http://testserver") as client:
            if bearer is not None:
                client.headers["Authorization"] = bearer
            elif role is not None or grant is not None:
                grant = grant or await self.credential(role, subject, user_id)
                original = authority_module._permissions
                if legacy_permissions:
                    authority_module._permissions = pre_u3b2_permissions
                try:
                    response = await client.post("/api/admin/auth/login", json={"token": grant.token})
                finally:
                    authority_module._permissions = original
                assert response.status_code == 200, response.text
                client.grant = grant
            yield client


@asynccontextmanager
async def owner_graph(tmp_path, dashboard_auth):
    """Build the graph; the authority and memory share one synthetic clock."""
    clock = Clock()
    accounts = {PERSON_A, PERSON_B, PERSON_C}
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / "auth.db", clock=clock, session_ttl_s=3600,
                                       account_provider=lambda: ActiveAccounts(True, frozenset(accounts)))
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        path = str(tmp_path / "privacy.db")
        services = await build_memory_services({"memory": {"privacy_db_path": path,
                                                           "preferences_path": str(tmp_path / "preferences.json")}},
                                               shutdown=stack, chroma_client=MemoryClient(tmp_path / "index"),
                                               clock=clock)
        api = DashboardAPI(authority=authority, memory_manager=services.manager)
        stack.push_async_callback(api.stop)
        memory = services.manager.long_term
        await memory.acall(memory.initialize)
        yield OwnerGraph(authority, api, memory, clock, authority.resolve(dashboard_auth["Authorization"]),
                         accounts, memory.ledger.path)
