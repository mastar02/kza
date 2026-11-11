"""Private dashboard authority, independent of HTTP and acoustic identities."""

from __future__ import annotations

from src.memory import output_registration_v2 as output_registration
import asyncio
from contextlib import closing
from contextvars import Context
from dataclasses import dataclass, field
from enum import StrEnum
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping
from uuid import uuid4

from src.core.owned_work import OwnedBlockingWork
from src.core.account_metadata import ActiveAccounts
from src.core.reader_binding import ReaderBinding

if TYPE_CHECKING:
    from src.dashboard.evaluation_grants import EvaluationGrantCatalog


class DashboardRole(StrEnum):
    """Closed server-selected roles; a voice label never selects one."""

    OPERATOR = "operator"
    HOME = "home"
    EVALUATOR = "evaluator"


class DashboardPermission(StrEnum):
    """Separate domestic authority, private content, diagnosis and review."""

    HOME_READ = "home:read"
    HOME_CONTROL = "home:control"
    TASKS_READ = "tasks:read"
    TASKS_WRITE = "tasks:write"
    PERSONAL_TASKS = "tasks:personal"
    ROUTINES_EXECUTE = "routines:execute"
    ROUTINES_MANAGE = "routines:manage"
    DIAGNOSTICS_READ = "diagnostics:read"
    CONVERSATIONS_READ = "conversations:read"
    MEMORY_READ = "memory:read"
    MEMORY_REVIEW = "memory:review"
    MEMORY_ASSIGN = "memory:assign"
    MEMORY_OWN_READ = "memory:own:read"
    MEMORY_OWN_WRITE = "memory:own:write"
    MEMORY_OWN_VOICE = "memory:own:voice"
    VOICES_READ = "voices:read"
    VOICES_REVIEW = "voices:review"
    EVALUATION_READ = "evaluation:read"
    AMBIENT_CONTROL = "ambient:control"
    ADMIN_MANAGE = "admin:manage"
    AUTH_MANAGE = "auth:manage"


class EvaluationModality(StrEnum):
    """Approval for a preview does not authorize its full segment transcript."""

    AUDIO_PREVIEW = "audio_preview"
    TRANSCRIPT_FULL = "transcript_full"


_HOME_PERMISSIONS = frozenset({
    DashboardPermission.HOME_READ, DashboardPermission.HOME_CONTROL,
    DashboardPermission.TASKS_READ, DashboardPermission.TASKS_WRITE,
    DashboardPermission.ROUTINES_EXECUTE,
})
# 'Mis recuerdos' (U3b-2): only a household credential bound to a person holds
# these. Operator credentials can never carry a user_id, so they do not get them.
# U3b-4a-2: memory:own:voice is the separate biometric opt-in grant, same rule.
_PERSON_MEMORY_PERMISSIONS = frozenset({
    DashboardPermission.MEMORY_OWN_READ, DashboardPermission.MEMORY_OWN_WRITE, DashboardPermission.MEMORY_OWN_VOICE,
})
MAX_CREDENTIAL_TTL_S = 30 * 86400
DEFAULT_SESSION_TTL_S = 30 * 86400
MAX_CREDENTIALS = 256
MAX_SESSIONS = 4096


def _identifier(value: str, *, maximum: int = 128) -> None:
    if (type(value) is not str or not 1 <= len(value) <= maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("invalid_dashboard_identity")


def _timestamp(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_dashboard_time")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_digest(source: Mapping) -> str:
    """Bind an evaluation grant to the complete original source declaration."""
    return _hash(json.dumps(dict(source), sort_keys=True, separators=(",", ":"), allow_nan=False))


@dataclass(frozen=True)
class EvaluationSample:
    """One exact fingerprint, original source proofs, modes and expiry."""

    fingerprint_id: str = field(repr=False)
    source_digests: frozenset[str] = field(repr=False)
    modalities: frozenset[EvaluationModality]
    expires_at: float

    def __post_init__(self) -> None:
        if (type(self.fingerprint_id) is not str or len(self.fingerprint_id) != 67
                or not self.fingerprint_id.startswith("vf_")
                or any(c not in "0123456789abcdef" for c in self.fingerprint_id[3:])):
            raise ValueError("invalid_evaluation_fingerprint")
        if (type(self.source_digests) is not frozenset or not 1 <= len(self.source_digests) <= 256
                or any(type(d) is not str or len(d) != 64 or any(c not in "0123456789abcdef" for c in d)
                       for d in self.source_digests)):
            raise ValueError("invalid_evaluation_sources")
        if (type(self.modalities) is not frozenset or not self.modalities
                or any(type(m) is not EvaluationModality for m in self.modalities)):
            raise ValueError("invalid_evaluation_modalities")
        _timestamp(self.expires_at)

    def to_dict(self) -> dict:
        """Serialize only bounded authority metadata, never source content."""
        return {"fingerprint_id": self.fingerprint_id, "source_digests": sorted(self.source_digests),
                "modalities": sorted(self.modalities), "expires_at": self.expires_at}

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationSample:
        """Reject incomplete persisted scopes rather than expanding them."""
        return cls(data["fingerprint_id"], frozenset(data["source_digests"]),
                   frozenset(EvaluationModality(m) for m in data["modalities"]), data["expires_at"])


@dataclass(frozen=True)
class DashboardPrincipal:
    """Server-owned identity; permissions and personal binding are not claims."""

    subject: str
    permissions: frozenset[str]
    role: DashboardRole = DashboardRole.OPERATOR
    user_id: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    revision: int = 0
    samples: tuple[EvaluationSample, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class _Credential:
    identity: str
    digest: str = field(repr=False)
    subject: str
    role: DashboardRole
    user_id: str | None
    issued_at: float
    expires_at: float
    revision: int
    samples: tuple[EvaluationSample, ...] = field(repr=False)


@dataclass(frozen=True)
class _Session:
    digest: str = field(repr=False)
    credential: str
    legacy_digest: str | None = field(repr=False)
    issued_at: float
    expires_at: float
    revision: int


@dataclass(frozen=True)
class DashboardLease:
    """Original reader binding, revalidated without retaining HTTP or tokens."""

    principal: DashboardPrincipal
    binding: str = field(repr=False)
    session: bool
    _check: Callable[[], bool] = field(repr=False, compare=False)
    _clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)
    _authority: object | None = field(default=None, repr=False, compare=False)

    def current(self, permissions: frozenset[str] = frozenset()) -> bool:
        """Recheck the original authority; never resolve a replacement actor."""
        return permissions <= self.principal.permissions and self._check() is True

    def reader_binding(self, permissions: frozenset[str] = frozenset()) -> ReaderBinding | None:
        """Bind transport ownership to this server identity and grant revision."""
        if type(permissions) is not frozenset:
            raise ValueError('invalid_reader_permissions')
        if type(self._authority) is not DashboardAuthority or not self.current(permissions):
            return None
        namespace = _hash(json.dumps(('kza-reader-v1', self._authority._reader_realm, self.binding,
                                      self.principal.revision), separators=(',', ':')))
        return ReaderBinding(namespace, lambda: self.current(permissions))


    def remaining_ttl(self) -> int:
        """Round the cookie hint; server expiration retains its exact deadline."""
        if self.principal.expires_at is None or not self.current():
            return 0
        return max(0, math.ceil(self.principal.expires_at - self._clock()))

    def sample(self, identity: str, sources: frozenset[str] | None = None) -> EvaluationSample | None:
        """Check exact sample/source authority before reading any private bytes."""
        if not self.current(frozenset({DashboardPermission.EVALUATION_READ})):
            return None
        return next((s for s in self.principal.samples if s.fingerprint_id == identity
                     and s.expires_at > self._clock() and (sources is None or sources == s.source_digests)), None)


@dataclass(frozen=True)
class IssuedCredential:
    """Return the generated secret once; its representation never prints it."""

    identity: str
    token: str = field(repr=False)
    principal: DashboardPrincipal


@dataclass(frozen=True)
class IssuedSession:
    """An individual opaque cookie, published only after persistence."""

    token: str = field(repr=False)
    lease: DashboardLease


def _expected_token() -> str | None:
    value = os.environ.get("KZA_DASHBOARD_TOKEN", "").strip()
    return value if value and not value.startswith("${") else None


def _permissions(role: DashboardRole, user_id: str | None) -> frozenset[str]:
    if role is DashboardRole.OPERATOR:
        values = frozenset(DashboardPermission) - _PERSON_MEMORY_PERMISSIONS
    elif role is DashboardRole.HOME:
        values = _HOME_PERMISSIONS | (frozenset({DashboardPermission.PERSONAL_TASKS}) | _PERSON_MEMORY_PERMISSIONS
                                      if user_id else frozenset())
    else:
        values = frozenset({DashboardPermission.EVALUATION_READ})
    return frozenset(value.value for value in values)


def legacy_lease(authorization: str | None, *, expected: Callable[[], str | None] = _expected_token) -> DashboardLease | None:
    """Preserve the administrative bearer, without legacy secret cookies."""
    if type(authorization) is not str or len(authorization) > 4096:
        return None
    token = authorization
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    configured = expected()
    if not configured or not token or not secrets.compare_digest(_hash(token), _hash(configured)):
        return None
    digest, retired = _hash(configured), False
    lock = threading.Lock()
    def current() -> bool:
        nonlocal retired
        with lock:
            value = expected()
            retired = retired or not (value and secrets.compare_digest(_hash(value), digest))
            return not retired
    return DashboardLease(DashboardPrincipal("dashboard-admin", _permissions(DashboardRole.OPERATOR, None)),
                          "legacy:" + digest, False, current)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS credentials (
 id TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, subject TEXT NOT NULL, role TEXT NOT NULL,
 user_id TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL,
 samples TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
 digest TEXT PRIMARY KEY, credential TEXT NOT NULL, legacy_digest TEXT,
 issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS retired_users (
 user_id TEXT PRIMARY KEY, retired_at REAL NOT NULL);
"""


class AuthorityRejected(ValueError):
    """A valid request exceeded a bounded authority capacity."""


class DashboardAuthority:
    """One private writer and immutable live authority through cancellation."""

    def __init__(self, db_path: str | Path, *, session_ttl_s: float = DEFAULT_SESSION_TTL_S,
                 clock: Callable[[], float] = time.time,
                 expected_token: Callable[[], str | None] = _expected_token,
                 work: OwnedBlockingWork | None = None,
                 account_provider: Callable[[], ActiveAccounts | None] | None = None):
        _timestamp(session_ttl_s)
        if not 1 <= session_ttl_s <= DEFAULT_SESSION_TTL_S:
            raise ValueError("invalid_dashboard_session_ttl")
        self.path = Path(db_path)
        self._reader_realm = os.path.abspath(os.fspath(self.path))
        self.session_ttl_s = session_ttl_s
        self._clock = clock
        self._expected = expected_token
        self._account_provider = account_provider
        self._work = work or OwnedBlockingWork(capacity=1)
        self._serial = asyncio.Lock()
        self._state_lock = threading.RLock()
        self._credentials = MappingProxyType({})
        self._sessions = MappingProxyType({})
        self._revoking: set[str] = set()
        self._retiring_users: set[str] = set()
        self._retired_users: frozenset[str] = frozenset()
        self._revision_admission: dict[str, int] = {}
        self._credential_reservations: dict[str, dict] = {}
        self._credential_revoking: set[str] = set()
        self._task_mutations: dict[asyncio.Task, tuple] = {}
        self._committed_tasks: set[asyncio.Task] = set()
        self._active_commit_witness = None
        self._attempts = MappingProxyType({})
        self._last_now = 0.0
        self._persisted_floor = 0.0
        self._ready = self._closed = self._failed = False
        self._lease_fd: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._initializing: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._observed_digest: str | None = None
        self._observed_epoch = self._persisted_epoch = 0
        self._persisted_digest: str | None = None
        from src.dashboard.credential_commands import NativeCredentialCommands
        self.commands = NativeCredentialCommands(self)

    def _now(self) -> float:
        value = self._clock()
        _timestamp(value)
        with self._state_lock:
            self._last_now = max(self._last_now, value)
            return self._last_now

    def _observe_legacy(self) -> None:
        value = self._expected()
        digest = _hash(value) if value else None
        with self._state_lock:
            if self._ready and digest != self._observed_digest:
                self._observed_digest = digest
                self._observed_epoch += 1
                self._revoking.update(d for d, s in self._sessions.items() if s.legacy_digest is not None)

    def _legacy_synced(self) -> bool:
        return (self._observed_epoch == self._persisted_epoch
                and self._observed_digest == self._persisted_digest)

    @property
    def configured(self) -> bool:
        """An unavailable authority never authenticates even a legacy bearer."""
        self._observe_legacy()
        with self._state_lock:
            return (self._usable() and bool(self._expected() or any(
                c.expires_at > self._now() for c in self._credentials.values())))


    @property
    def available(self) -> bool:
        """Report initialized usable storage without creating any resource."""
        with self._state_lock:
            return self._usable()

    async def initialize(self) -> None:
        """Acquire the writer lease and load only committed authority."""
        if self._initializing is None:
            self._initializing = self._admit("initialize", ())
        await asyncio.shield(self._initializing)

    async def synchronize(self) -> None:
        """Persist rotations and observed expirations before authenticating HTTP."""
        await self.initialize()
        for _ in range(4):
            self._observe_legacy()
            with self._state_lock:
                synced = self._legacy_synced()
                expired = any(s.expires_at <= self._now() for s in self._sessions.values()) or any(
                    c.expires_at <= self._now() for c in self._credentials.values())
                expired = expired or any(self._persisted_floor < s.expires_at <= self._now()
                                         for c in self._credentials.values() for s in c.samples)
                if synced and not expired:
                    return
                args = self._observed_digest, self._observed_epoch
            await asyncio.shield(self._admit("refresh" if synced else "rotate", args))
        raise RuntimeError("dashboard_credential_changing")

    def resolve(self, authorization: str | None, cookie: str | None = None) -> DashboardLease | None:
        """An explicit header selects one identity and never falls back to cookie."""
        if not self.configured:
            return None
        value = authorization if authorization is not None else cookie
        if type(value) is not str or not value or len(value) > 4096:
            return None
        if authorization is not None:
            if value.lower().startswith("bearer "):
                value = value.split(" ", 1)[1].strip()
            legacy = legacy_lease(value, expected=self._expected)
            if legacy is not None:
                with self._state_lock:
                    epoch, digest = self._observed_epoch, self._observed_digest
                principal = DashboardPrincipal("dashboard-admin", legacy.principal.permissions, revision=epoch)
                lease = DashboardLease(principal, f"legacy:{epoch}:{digest}", False,
                                       lambda: self._legacy_current(epoch, digest), self._now, self)
                return lease if lease.current() else None
        elif not value.startswith("kzas_"):
            return None
        digest = _hash(value)
        with self._state_lock:
            session = self._sessions.get(digest)
            credential = self._credentials.get(digest) if session is None else None
            if session is not None:
                credential = next((c for c in self._credentials.values() if c.identity == session.credential), None)
            if session is not None and session.legacy_digest is not None:
                principal = DashboardPrincipal("dashboard-admin", _permissions(DashboardRole.OPERATOR, None),
                                               expires_at=session.expires_at, revision=session.revision)
            elif credential is not None:
                expiry = min(credential.expires_at, session.expires_at) if session else credential.expires_at
                principal = DashboardPrincipal(credential.subject, _permissions(credential.role, credential.user_id),
                                               credential.role, credential.user_id, expiry, credential.revision, credential.samples)
            else:
                return None
        lease = DashboardLease(principal, ("session:" if session else "credential:") + digest, session is not None,
                               lambda: self._current(digest, principal, session is not None), self._now, self)
        return lease if lease.current() else None

    def evaluation_grants(self, actor: DashboardLease) -> EvaluationGrantCatalog:
        """Project only live choices from this original evaluator lease."""
        if (type(actor) is not DashboardLease or actor._authority is not self
                or actor.principal.role is not DashboardRole.EVALUATOR
                or not actor.current(frozenset({DashboardPermission.EVALUATION_READ}))):
            raise PermissionError('evaluation_catalog_denied')
        from src.dashboard.evaluation_grants import EvaluationGrantCatalog, EvaluationGrantMetadata
        with self._state_lock:
            now = self._now()
            if len(actor.principal.samples)>64:
                raise ValueError('invalid_evaluation_catalog')
            samples = tuple(sample for sample in actor.principal.samples if sample.expires_at>now)
            grants = tuple(EvaluationGrantMetadata(sample.fingerprint_id,sample.modalities,
                min(sample.expires_at,actor.principal.expires_at)) for sample in samples)
            catalog = EvaluationGrantCatalog(actor,now,
                min([actor.principal.expires_at,*[grant.expires_at for grant in grants]]),grants,samples)
        if not catalog.current():
            raise PermissionError('evaluation_catalog_retired')
        return catalog

    def _usable(self) -> bool:
        return self._ready and not self._closed and not self._failed

    def _legacy_current(self, epoch: int, digest: str | None) -> bool:
        self._observe_legacy()
        with self._state_lock:
            return bool(self._usable() and self._legacy_synced() and digest
                        and epoch == self._observed_epoch and digest == self._observed_digest)

    def _account_current(self, user_id: str | None) -> bool:
        """Check only an immutable canonical snapshot; providers must never acquire IO/locks."""
        if user_id is None:
            return True
        with self._state_lock:
            if user_id in self._retiring_users or user_id in self._retired_users or self._account_provider is None:
                return False
            try:
                snapshot = self._account_provider()
            except Exception:
                return False
            return type(snapshot) is ActiveAccounts and snapshot.available and user_id in snapshot.active_ids

    def users_retired(self, identities: frozenset[str]) -> bool:
        """Prove a durable published retirement without SQL or account locks."""
        with self._state_lock:
            return bool(type(identities) is frozenset and self._usable()
                        and identities <= self._retired_users)


    def memory_owner_retired(self, identities: frozenset[str]) -> bool | None:
        """Owner-assignment check (U3b-1b): a retirement in progress already counts.

        Unlike ``users_retired`` (a durable proof that is False when the
        authority is unusable), this returns None when the answer is unknown,
        which the fact ledger refuses as ``memory_owner_authority_unavailable``.
        Reads only published in-memory state: no SQL, no account or ledger lock.
        """
        with self._state_lock:
            if type(identities) is not frozenset or not identities or not self._usable():
                return None
            return bool(identities & (self._retiring_users | self._retired_users))

    def memory_owner_active(self, user_id: str) -> bool | None:
        """Owner-assignment check (U3b-1b) with ``_account_current`` semantics.

        True only for a canonical active account that is neither retiring nor
        retired; None (unknown) when the authority is unusable or the id is
        not a string. Reads only published in-memory state.
        """
        if type(user_id) is not str:
            return None
        with self._state_lock:
            if not self._usable():
                return None
            return self._account_current(user_id)

    def users_linkable(self, identities: frozenset[str]) -> bool:
        """Check a bounded set against one canonical snapshot, without account IO/locks."""
        if type(identities) is not frozenset or not 1 <= len(identities) <= 256:
            return False
        try:
            for identity in identities:
                _identifier(identity)
        except ValueError:
            return False
        with self._state_lock:
            if (not self._usable() or self._account_provider is None
                    or identities & (self._retiring_users | self._retired_users)):
                return False
            try:
                snapshot = self._account_provider()
            except Exception:
                return False
            return (type(snapshot) is ActiveAccounts and snapshot.available
                    and identities <= snapshot.active_ids)

    def _current(self, digest: str, original: DashboardPrincipal, session: bool, *, allow_revoking: str | None = None) -> bool:
        self._observe_legacy()
        with self._state_lock:
            if not self._usable() or digest in self._revoking and digest != allow_revoking:
                return False
            now = self._now()
            entry = self._sessions.get(digest) if session else None
            if session and (entry is None or entry.expires_at <= now or entry.revision != original.revision):
                return False
            if entry is not None and entry.legacy_digest is not None:
                return self._legacy_current(entry.revision, entry.legacy_digest)
            credential = (next((c for c in self._credentials.values() if c.identity == entry.credential), None)
                          if entry else self._credentials.get(digest))
            return bool(credential and self._account_current(credential.user_id)
                        and credential.expires_at > now and credential.revision == original.revision
                        and (credential.identity not in (self._revoking|self._credential_revoking) or credential.identity == allow_revoking)
                        and (credential.revision >= self._revision_admission.get(credential.identity, 0)
                             or credential.identity == allow_revoking)
                        and credential.subject == original.subject and credential.role is original.role
                        and credential.user_id == original.user_id and credential.samples == original.samples
                        and _permissions(credential.role, credential.user_id) == original.permissions)

    def _reserve_credential_mutation_locked(self,identity,kind,revision=None,*,cas=False):
        """Own only this mutation's admission through commit and publication."""
        if self._closed or not cas and len(self._tasks)>=32:
            raise RuntimeError('dashboard_authority_closed_or_busy')
        writers = self._credential_reservations.get(identity,{})
        if writers and (cas or any(value[2] for value in writers.values())):
            raise AuthorityRejected('dashboard_credential_writer_busy')
        owner = object()
        self._credential_reservations.setdefault(identity,{})[owner] = kind,revision,cas
        self._recompute_credential_admission_locked(identity)
        return identity,owner

    def _recompute_credential_admission_locked(self,identity):
        writers = self._credential_reservations.get(identity,{})
        revisions = [value[1] for value in writers.values() if value[1] is not None]
        published = next((entry.revision for entry in self._credentials.values() if entry.identity==identity),0)
        floor = max([published,*revisions])
        if revisions:
            self._revision_admission[identity] = floor
        else:
            self._revision_admission.pop(identity,None)
        if any(value[0]=='revoke' for value in writers.values()):
            self._credential_revoking.add(identity)
        else:
            self._credential_revoking.discard(identity)

    def _release_credential_mutation_locked(self,reservation):
        if reservation is None:
            return
        identity,owner = reservation
        writers = self._credential_reservations.get(identity)
        if writers is not None:
            writers.pop(owner,None)
            if not writers:
                self._credential_reservations.pop(identity,None)
        self._recompute_credential_admission_locked(identity)

    def _release_task_mutation_locked(self,task):
        self._release_credential_mutation_locked(self._task_mutations.pop(task,None))

    async def _run_transaction(self,action,args,guard):
        """Retain uncertainty once native commit starts, including publication failures."""
        witness = threading.Event()
        self._active_commit_witness = witness
        try:
            return await self._work.run(self._transaction,action,args,guard)
        finally:
            if witness.is_set():
                self._committed_tasks.add(asyncio.current_task())
            self._active_commit_witness = None

    def _publish_state(self,state):
        result,credentials,sessions,floor,digest,epoch,retired_users,attempts = state
        with self._state_lock:
            self._credentials = MappingProxyType(credentials)
            self._sessions = MappingProxyType(sessions)
            self._attempts = MappingProxyType(attempts)
            self._retired_users = retired_users
            self._retiring_users.difference_update(retired_users)
            self._last_now = max(self._last_now,floor)
            self._persisted_floor = max(self._persisted_floor,floor)
            if not self._ready:
                self._observed_digest,self._observed_epoch = digest,epoch
            self._persisted_digest,self._persisted_epoch = digest,epoch
            self._revoking.intersection_update(set(credentials)|set(sessions)|{entry.identity for entry in credentials.values()})
            self._ready = True
            self._observe_legacy()
        return result

    def _admit(self, action: str, args: tuple, guard: Callable[[], bool] | None = None, *, reservation=None) -> asyncio.Task:
        if self._closed or len(self._tasks) >= 32:
            with self._state_lock:
                self._release_credential_mutation_locked(reservation)
            raise RuntimeError("dashboard_authority_closed_or_busy")
        try:
            task = asyncio.create_task(self._operate(action, args, guard), context=Context())
        except BaseException:
            with self._state_lock:
                self._release_credential_mutation_locked(reservation)
            raise
        self._tasks.add(task)
        if reservation is not None:
            self._task_mutations[task] = reservation
        def finished(done):
            self._tasks.discard(done)
            if done.cancelled():
                self._failed = True
            else:
                done.exception()
        task.add_done_callback(finished, context=Context())
        return task

    async def _operate(self, action: str, args: tuple, guard: Callable[[], bool] | None) -> object:
        async with self._serial:
            try:
                if action=='credential_command_run':
                    result = await self.commands._run_locked(args[0])
                else:
                    state = await self._run_transaction(action,args,guard)
                    if action == "initialize":
                        await output_registration.async_ready(self)
                    result = self._publish_state(state)
                with self._state_lock:
                    self._release_task_mutation_locked(asyncio.current_task())
                return result
            except (PermissionError,AuthorityRejected):
                if asyncio.current_task() in self._committed_tasks:
                    self._failed = True
                    raise RuntimeError('dashboard_authority_outcome_unknown') from None
                with self._state_lock:
                    self._release_task_mutation_locked(asyncio.current_task())
                raise
            except BaseException:
                self._failed = True
                raise
            finally:
                self._committed_tasks.discard(asyncio.current_task())
                args = guard = None

    def _open(self) -> None:
        output_registration.before_native(self, self.path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lease = self.path.with_name(self.path.name + ".lease")
        fd = os.open(lease, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("unsafe_dashboard_authority_lease")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                info = self.path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise ValueError("unsafe_dashboard_authority_file")
            else:
                file_fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                os.close(file_fd)
        except BaseException:
            os.close(fd)
            raise
        self._lease_fd = fd

    def _transaction(self, action: str, args: tuple, guard: Callable[[], bool] | None):
        output_registration.dashboard_operation(self, action, args, guard)
        if action == "initialize":
            self._open()
        if self._lease_fd is None:
            raise RuntimeError("dashboard_authority_not_initialized")
        command_action = action == "credential_command"
        if not command_action and guard is not None and guard() is not True:
            raise PermissionError("dashboard_authority_revoked")
        from src.dashboard.credential_journal import migrate_schema, read_attempts, read_metadata, validate_history
        now = self._now()
        with output_registration.native_connection(self, self.path, sqlite3.connect, self.path, timeout=1) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            if action == "initialize":
                migrate_schema(db,_SCHEMA)
            persisted_floor,epoch,digest = read_metadata(db)
            floor = max(now,persisted_floor)
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (floor,))
            db.execute("DELETE FROM credentials WHERE expires_at<=?", (floor,))
            result = None
            if action in {"initialize", "rotate"}:
                value = self._expected() if action == "initialize" else None
                next_digest, next_epoch = ((_hash(value) if value else None), epoch) if action == "initialize" else args
                next_epoch = max(epoch + (digest != next_digest), next_epoch)
                if next_epoch != epoch or next_digest != digest:
                    db.execute("DELETE FROM sessions WHERE legacy_digest IS NOT NULL")
                digest, epoch = next_digest, next_epoch
                db.executemany("INSERT OR REPLACE INTO auth_meta VALUES (?,?)", (
                    ("legacy_digest", digest or ""), ("legacy_epoch", str(epoch))))
            elif action == "issue":
                if db.execute("SELECT 1 FROM credential_attempts WHERE credential_id=? AND kind='revoke' AND outcome='committed'",(args[0],)).fetchone():
                    raise AuthorityRejected("dashboard_credential_retired")
                if db.execute("SELECT count(*) FROM credentials").fetchone()[0] >= MAX_CREDENTIALS:
                    raise AuthorityRejected("dashboard_credential_capacity")
                db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)", args)
            elif action == "login":
                values, previous = args
                if previous is not None:
                    if db.execute("DELETE FROM sessions WHERE digest=?", (previous,)).rowcount != 1:
                        raise PermissionError("dashboard_session_replaced")
                if db.execute("SELECT count(*) FROM sessions").fetchone()[0] >= MAX_SESSIONS:
                    raise AuthorityRejected("dashboard_session_capacity")
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", values)
            elif action == "revise":
                result = db.execute("""UPDATE credentials SET role=?,user_id=?,issued_at=?,expires_at=?,
                                    revision=?,samples=? WHERE id=?""", args).rowcount == 1
                if not result:
                    raise AuthorityRejected("dashboard_credential_missing")
                db.execute("DELETE FROM sessions WHERE credential=?", (args[-1],))
            elif action == "retire_user":
                user_id, admitted = args
                targets = admitted | {row[0] for row in db.execute("SELECT id FROM credentials WHERE user_id=?", (user_id,))}
                db.execute("INSERT OR IGNORE INTO retired_users VALUES (?,?)", (user_id, floor))
                for identity in targets:
                    db.execute("DELETE FROM sessions WHERE credential=?", (identity,))
                    db.execute("DELETE FROM credentials WHERE id=?", (identity,))
                result = True
            elif action == "revoke_credential":
                result = db.execute("DELETE FROM credentials WHERE id=?", args).rowcount == 1
                db.execute("DELETE FROM sessions WHERE credential=?", args)
            elif action == "logout":
                result = db.execute("DELETE FROM sessions WHERE digest=?", args).rowcount == 1
            elif action == "credential_command":
                result = self.commands._apply_sql(db,args[0],guard,floor)
            elif action == "refresh":
                pass
            else:
                raise ValueError("invalid_dashboard_authority_operation")
            db.execute("DELETE FROM credentials WHERE user_id IN (SELECT user_id FROM retired_users)")
            db.execute("DELETE FROM sessions WHERE credential!='legacy' AND credential NOT IN (SELECT id FROM credentials)")
            db.execute("INSERT OR REPLACE INTO auth_meta VALUES ('clock_floor',?)", (str(floor),))
            credentials, sessions = self._read(db)
            retired_users = set()
            for row in db.execute("SELECT user_id,retired_at FROM retired_users"):
                _identifier(row["user_id"])
                _timestamp(row["retired_at"])
                retired_users.add(row["user_id"])
            attempts = read_attempts(db)
            validate_history(credentials,attempts,floor)
            if command_action:
                floor = max(floor,self._now())
                result = self.commands._completed_sql(db,result,floor)
                db.execute("INSERT OR REPLACE INTO auth_meta VALUES ('clock_floor',?)",(str(floor),))
                attempts = read_attempts(db)
                validate_history(credentials,attempts,floor)
                if result.applied and guard() is not True:
                    result = self.commands._retired_sql(db,result)
                    credentials,sessions = self._read(db)
                    floor = max(floor,self._now())
                    result = self.commands._completed_sql(db,result,floor)
                    db.execute("INSERT OR REPLACE INTO auth_meta VALUES ('clock_floor',?)",(str(floor),))
                    attempts = read_attempts(db)
                    validate_history(credentials,attempts,floor)
            elif guard is not None and guard() is not True:
                raise PermissionError("dashboard_authority_revoked")
            if self._active_commit_witness is not None:
                self._active_commit_witness.set()
            db.commit()
        return result,credentials,sessions,floor,digest,epoch,frozenset(retired_users),attempts

    @staticmethod
    def _read(db: sqlite3.Connection) -> tuple[dict, dict]:
        credentials, sessions = {}, {}
        for row in db.execute("SELECT * FROM credentials"):
            role = DashboardRole(row["role"])
            samples = tuple(EvaluationSample.from_dict(s) for s in json.loads(row["samples"]))
            _identifier(row["id"])
            _identifier(row["subject"])
            if row["user_id"] is not None:
                _identifier(row["user_id"])
            if (row["user_id"] is not None and role is not DashboardRole.HOME
                    or (role is DashboardRole.EVALUATOR) != bool(samples) or len(samples) > 64
                    or len({s.fingerprint_id for s in samples}) != len(samples)
                    or sum(len(s.source_digests) for s in samples) > 1024):
                raise ValueError("invalid_dashboard_stored_scope")
            for value in (row["issued_at"], row["expires_at"]):
                _timestamp(value)
            if (not 0 < row["expires_at"] - row["issued_at"] <= MAX_CREDENTIAL_TTL_S
                    or type(row["revision"]) is not int or row["revision"] < 1):
                raise ValueError("invalid_dashboard_stored_credential")
            entry = _Credential(row["id"], row["digest"], row["subject"], role, row["user_id"],
                                row["issued_at"], row["expires_at"], row["revision"], samples)
            credentials[entry.digest] = entry
        for row in db.execute("SELECT * FROM sessions"):
            for value in (row["issued_at"], row["expires_at"]):
                _timestamp(value)
            if (not 0 < row["expires_at"] - row["issued_at"] <= DEFAULT_SESSION_TTL_S
                    or type(row["revision"]) is not int or row["revision"] < 0
                    or (row["credential"] == "legacy") != (row["legacy_digest"] is not None)):
                raise ValueError("invalid_dashboard_stored_session")
            entry = _Session(row["digest"], row["credential"], row["legacy_digest"],
                             row["issued_at"], row["expires_at"], row["revision"])
            sessions[entry.digest] = entry
        if len(credentials) > MAX_CREDENTIALS or len(sessions) > MAX_SESSIONS:
            raise ValueError("dashboard_authority_capacity")
        for digest in (*credentials, *sessions, *(s.legacy_digest for s in sessions.values() if s.legacy_digest is not None)):
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid_dashboard_stored_digest")
        return credentials, sessions

    def _validate_grant(self, subject: str, role: DashboardRole, expires_at: float,
                        user_id: str | None, samples: tuple[EvaluationSample, ...]) -> float:
        _identifier(subject)
        _timestamp(expires_at)
        now = self._now()
        if type(role) is not DashboardRole or not now < expires_at <= now + MAX_CREDENTIAL_TTL_S:
            raise ValueError("invalid_dashboard_grant")
        if user_id is not None:
            _identifier(user_id)
            if role is not DashboardRole.HOME:
                raise ValueError("invalid_dashboard_personal_binding")
            if not self._account_current(user_id):
                raise ValueError("dashboard_personal_binding_unavailable")
        if (type(samples) is not tuple or len(samples) > 64
                or any(type(s) is not EvaluationSample or not now < s.expires_at for s in samples)
                or len({s.fingerprint_id for s in samples}) != len(samples)
                or sum(len(s.source_digests) for s in samples) > 1024
                or (role is DashboardRole.EVALUATOR) != bool(samples)):
            raise ValueError("invalid_dashboard_evaluation_scope")
        return now

    def _prepared_evaluation_guard(self,actor,role,expires_at,user_id,samples,prepared_scope):
        """Bind a server proof without borrowing another login or issuer guard."""
        if prepared_scope is None:
            return lambda:True
        from src.dashboard.evaluation_grants import PreparedEvaluationScope
        if (type(prepared_scope) is not PreparedEvaluationScope
                or not prepared_scope.matches(actor,role,expires_at,user_id,samples)):
            raise PermissionError('dashboard_grant_scope_mismatch')
        return lambda:prepared_scope.current(self._now())

    async def issue_credential(self, actor: DashboardLease, *, subject: str, role: DashboardRole,
                               expires_at: float, user_id: str | None = None,
                               samples: tuple[EvaluationSample, ...] = (),
                               prepared_scope: object | None = None) -> IssuedCredential:
        """Issue a bounded role grant; HTTP resolves user/source bindings first."""
        if (type(actor) is not DashboardLease or actor._authority is not self
                or not actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))):
            raise PermissionError("dashboard_grant_denied")
        now = self._validate_grant(subject, role, expires_at, user_id, samples)
        scope_current = self._prepared_evaluation_guard(actor,role,expires_at,user_id,samples,prepared_scope)
        if scope_current() is not True:
            raise PermissionError('dashboard_grant_scope_retired')
        identity, token = str(uuid4()), "kza_" + secrets.token_urlsafe(32)
        try:
            args = (identity, _hash(token), subject, role.value, user_id, now, expires_at, 1,
                    json.dumps([s.to_dict() for s in samples], separators=(",", ":")))
            await asyncio.shield(self._admit("issue", args, lambda: actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))
                                            and self._account_current(user_id) and scope_current()))
            lease = self.resolve(token)
            if lease is None:
                raise PermissionError("dashboard_grant_expired")
            return IssuedCredential(identity, token, lease.principal)
        finally:
            token = None


    async def revise_credential(self, actor: DashboardLease, identity: str, *, role: DashboardRole,
                                expires_at: float, user_id: str | None = None,
                                samples: tuple[EvaluationSample, ...] = (),
                                prepared_scope: object | None = None) -> bool:
        """Keep the secret but retire every lease and session of the old grant."""
        if (type(actor) is not DashboardLease or actor._authority is not self
                or not actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))):
            raise PermissionError("dashboard_grant_denied")
        with self._state_lock:
            original = next((c for c in self._credentials.values() if c.identity == identity), None)
            if original is None:
                return False
            original_current = self._revocation_guard(actor, identity)
            now = self._validate_grant(original.subject, role, expires_at, user_id, samples)
            scope_current = self._prepared_evaluation_guard(actor,role,expires_at,user_id,samples,prepared_scope)
            if scope_current() is not True:
                raise PermissionError('dashboard_grant_scope_retired')
            if original_current() is not True:
                raise PermissionError('dashboard_grant_denied')
            revision = max(original.revision, self._revision_admission.get(identity, 0)) + 1
            encoded = json.dumps([sample.to_dict() for sample in samples],separators=(',',':'))
            reservation = self._reserve_credential_mutation_locked(identity,'revise',revision)
        args = (role.value,user_id,now,expires_at,revision,encoded,identity)
        return await asyncio.shield(self._admit("revise",args,
            lambda:original_current() and self._account_current(user_id) and scope_current(),reservation=reservation))

    async def login(self, token: str, *, replacing: DashboardLease | None = None) -> IssuedSession:
        """Issue a cookie, optionally replacing this browser's current session atomically."""
        await self.synchronize()
        actor = self.resolve(token)
        if actor is None or actor.session:
            raise PermissionError("dashboard_login_denied")
        if replacing is not None and (type(replacing) is not DashboardLease
                or replacing._authority is not self or not replacing.session or not replacing.current()):
            raise PermissionError("dashboard_session_replaced")
        now = self._now()
        expiry = min(now + self.session_ttl_s, actor.principal.expires_at or math.inf)
        cookie = "kzas_" + secrets.token_urlsafe(32)
        digest = actor.binding.rsplit(":", 1)[1]
        legacy = digest if actor.binding.startswith("legacy:") else None
        credential = "legacy" if legacy else self._credentials[digest].identity
        try:
            values = (_hash(cookie), credential, legacy, now, expiry, actor.principal.revision)
            previous = replacing.binding.split(":", 1)[1] if replacing is not None else None
            def guard() -> bool:
                return actor.current() and (replacing is None or replacing.current())
            await asyncio.shield(self._admit("login", (values, previous), guard))
            lease = self.resolve(None, cookie)
            if lease is None:
                raise PermissionError("dashboard_session_expired")
            return IssuedSession(cookie, lease)
        finally:
            token = cookie = None

    async def retire_user(self, actor: DashboardLease, user_id: str) -> bool:
        """Seal all current bindings, then publish a durable canonical-user retirement."""
        if (type(actor) is not DashboardLease or actor._authority is not self
                or not actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))):
            raise PermissionError("dashboard_grant_denied")
        _identifier(user_id)
        with self._state_lock:
            admitted = frozenset(c.identity for c in self._credentials.values() if c.user_id == user_id)
            self._retiring_users.add(user_id)
            self._revoking.update(admitted)
        return await asyncio.shield(self._admit("retire_user", (user_id, admitted),
            lambda: actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))))

    async def revoke_credential(self, actor: DashboardLease, identity: str) -> bool:
        """Close admission immediately and confirm only a committed revocation."""
        if (type(actor) is not DashboardLease or actor._authority is not self
                or not actor.current(frozenset({DashboardPermission.AUTH_MANAGE}))):
            raise PermissionError("dashboard_grant_denied")
        with self._state_lock:
            if not any(c.identity == identity for c in self._credentials.values()):
                return False
            current = self._revocation_guard(actor,identity)
            if current() is not True:
                raise PermissionError('dashboard_grant_denied')
            reservation = self._reserve_credential_mutation_locked(identity,'revoke')
        return await asyncio.shield(self._admit('revoke_credential',(identity,),current,reservation=reservation))

    def _revocation_guard(self, actor: DashboardLease, target: str) -> Callable[[], bool]:
        if type(actor.binding) is not str:
            raise PermissionError('invalid_dashboard_binding')
        if actor.binding.startswith("legacy:"):
            return actor.current
        prefix,separator,digest = actor.binding.partition(':')
        if (not separator or prefix!=('session' if actor.session else 'credential')
                or len(digest)!=64 or any(char not in '0123456789abcdef' for char in digest)):
            raise PermissionError('invalid_dashboard_binding')
        return lambda: self._current(digest, actor.principal, actor.session, allow_revoking=target)

    async def logout(self, actor: DashboardLease) -> bool:
        """Revoke this session; a shared administrative bearer is not revoked."""
        if type(actor) is not DashboardLease or actor._authority is not self or not actor.current():
            raise PermissionError("dashboard_logout_denied")
        if not actor.session:
            return False
        digest = actor.binding.split(":", 1)[1]
        with self._state_lock:
            self._revoking.add(digest)
        return await asyncio.shield(self._admit("logout", (digest,), self._revocation_guard(actor, digest)))

    async def aclose(self) -> None:
        """Reject new authority and drain actual transactions before the lease."""
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(self._close(), context=Context())
            self._closing.add_done_callback(lambda t: None if t.cancelled() else t.exception(), context=Context())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        try:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        finally:
            # An operation observer may have been cancelled after dispatch.
            # Keep the writer lease until the actual native drain completes.
            drain = self._work.begin_close()
            cancelled = False
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    cancelled = True
            drain.result()
            await self._work.aclose()
            try:
                if self._ready and self._lease_fd is not None:
                    # Native guards can observe rotation/expiry without another
                    # HTTP request. Persist those fences on a clean shutdown.
                    self._observe_legacy()
                    args = self._observed_digest, self._observed_epoch
                    final = OwnedBlockingWork(capacity=1)
                    output_registration.final_worker(self, final, args)
                    try:
                        await output_registration.run_final_rotation(self, final, args)
                    finally:
                        finish = final.begin_close()
                        while not finish.done():
                            try:
                                await asyncio.shield(finish)
                            except asyncio.CancelledError:
                                cancelled = True
                        finish.result()
                        await final.aclose()
            finally:
                if self._lease_fd is not None:
                    fd, self._lease_fd = self._lease_fd, None
                    os.close(fd)
            await output_registration.owner_closed(self)
            if cancelled:
                raise asyncio.CancelledError
