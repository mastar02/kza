"""Process-local review of sourced household voice routine proposals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.core.owned_work import OwnedBlockingWork
from src.core.sqlite_work import AuthorityGuard
from src.routines.models import (
    RoutineConflict, RoutineNotFound, RoutineSpec, RoutineValidationError,
    parse_routine_spec, spec_to_dict,
)
from src.routines.routine_store import HOUSEHOLD_VOICE

if TYPE_CHECKING:
    from src.memory.fact_ledger import FactLedger
    from src.routines.routine_scheduler import RoutineScheduler

MAX_LIFETIME_SECONDS = 48 * 60 * 60


class ProposalOperation(StrEnum):
    """Supported changes to the routine definition store."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ENABLE = "enable"
    DISABLE = "disable"


class ProposalNotFound(LookupError):
    """A proposal is absent, expired or inaccessible from its source."""


class ProposalUnavailable(RuntimeError):
    """The tray cannot currently admit a request."""


class ProposalDecision(StrEnum):
    """Terminal review decisions retained for retries until expiry or erasure."""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ProposalReceipt:
    """Minimal replayable result of a review decision."""

    proposal_id: str
    status: ProposalDecision
    routine_id: str | None = None
    revision: int | None = None


@dataclass
class _Proposal:
    proposal_id: str
    operation: ProposalOperation
    conversation_id: str = field(repr=False)
    fingerprint: bytes = field(repr=False)
    created_at: float
    expires_at: float
    routine_id: str | None
    expected_revision: int | None
    spec: RoutineSpec | None = field(repr=False)
    receipt: ProposalReceipt | None = None

    def summary(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "operation": self.operation.value,
            "source": HOUSEHOLD_VOICE,
            "created_at": datetime.fromtimestamp(self.created_at, UTC).isoformat(),
            "expires_at": datetime.fromtimestamp(self.expires_at, UTC).isoformat(),
            "routine_id": self.routine_id,
            "expected_revision": self.expected_revision,
        }


class RoutineProposalTray:
    """Hold pending reviews on the event loop, with native commit-time source checks.

    Args:
        source_policy: Shared authority for expiry and conversation revocations.
    """

    def __init__(self, source_policy: FactLedger) -> None:
        self.source_policy = source_policy
        self.retention_notify: Callable[[], None] | None = None
        self._entries: dict[str, _Proposal] = {}
        self._confirming: dict[str, str] = {}
        self._work = OwnedBlockingWork(isolate_context=True)

    def submit(
        self, operation: ProposalOperation, *, conversation_id: str, spec: RoutineSpec | None = None,
        routine_id: str | None = None, expected_revision: int | None = None,
        expires_at: float | None = None,
    ) -> str:
        """Queue a normalized change internally and return its opaque proposal ID.

        Args:
            operation: Requested change.
            conversation_id: Private UUID of the source conversation.
            spec: Complete editable definition for create or update.
            routine_id: Existing routine for any change other than create.
            expected_revision: Revision observed when the change was proposed.
            expires_at: Optional source deadline as UTC epoch seconds. The tray
                enforces its own maximum even if the source lives longer.

        Returns:
            The same ID for a replay of an equivalent change from this source.
        """
        operation = ProposalOperation(operation)
        conversation_id = str(UUID(conversation_id))
        if operation is ProposalOperation.CREATE:
            if routine_id is not None or expected_revision is not None:
                raise RoutineValidationError("unexpected_proposal_target")
        elif (not isinstance(routine_id, str) or not routine_id.strip()
              or type(expected_revision) is not int or expected_revision < 1):
            raise RoutineValidationError("invalid_proposal_target")
        if (spec is not None) != (operation in {ProposalOperation.CREATE, ProposalOperation.UPDATE}):
            raise RoutineValidationError("invalid_proposal_definition")
        if spec is not None:
            spec = parse_routine_spec(spec_to_dict(spec))

        self._require_available()
        now = self.source_policy.clock()
        deadline = now + MAX_LIFETIME_SECONDS
        if expires_at is not None:
            if not math.isfinite(expires_at):
                raise RoutineValidationError("invalid_proposal_expiry")
            deadline = min(deadline, expires_at)
        if deadline <= now or self.source_policy.conversation_revoked(conversation_id):
            raise PermissionError("proposal_source_unavailable")
        self.prune_retained()
        payload = [operation.value, routine_id, expected_revision, spec_to_dict(spec) if spec else None]
        fingerprint = sha256(json.dumps(payload, sort_keys=True).encode()).digest()
        for entry in self._entries.values():
            if entry.conversation_id == conversation_id and entry.fingerprint == fingerprint:
                if deadline < entry.expires_at:
                    entry.expires_at = deadline
                    if self.retention_notify is not None:
                        self.retention_notify()
                return entry.proposal_id
        proposal_id = uuid4().hex
        self._entries[proposal_id] = _Proposal(
            proposal_id, operation, conversation_id, fingerprint, now, deadline,
            routine_id, expected_revision, spec,
        )
        if self.retention_notify is not None:
            self.retention_notify()
        return proposal_id

    def list_pending(self) -> list[dict]:
        """Return only reviewable metadata, without private source identifiers."""
        self._require_available()
        self.prune_retained()
        return [
            entry.summary() for entry in self._entries.values()
            if entry.receipt is None and self._source_current(entry)
        ]

    async def preview(self, proposal_id: str, scheduler: RoutineScheduler) -> dict:
        """Project the proposal and current definition without mutating either."""
        entry = self._get(proposal_id)
        if entry.receipt is not None:
            raise ProposalNotFound
        current = await scheduler.get_routine(entry.routine_id) if entry.routine_id is not None else None
        if not self._source_current(entry) or entry.receipt is not None:
            raise ProposalNotFound
        before = spec_to_dict(current.spec) if current is not None else None
        after = spec_to_dict(entry.spec) if entry.spec is not None else None
        if before is not None and entry.operation in {ProposalOperation.ENABLE, ProposalOperation.DISABLE}:
            after = {**before, "enabled": entry.operation is ProposalOperation.ENABLE}
        revision = current.revision if current is not None else None
        return {
            **entry.summary(),
            "current_revision": revision,
            "stale": revision != entry.expected_revision,
            "diff": {"before": before, "after": after},
        }

    async def confirm(
        self, proposal_id: str, scheduler: RoutineScheduler, *, actor: str, guard: AuthorityGuard,
    ) -> ProposalReceipt:
        """Retain the commit through a cancelled observer and replay its receipt."""
        self._require_available()
        work = self._work.status()
        if work["operations"] >= work["capacity"]:
            raise ProposalUnavailable
        return await self._work.run_async(self._confirm, proposal_id, scheduler, actor, guard)

    async def _confirm(
        self, proposal_id: str, scheduler: RoutineScheduler, actor: str, guard: AuthorityGuard,
    ) -> ProposalReceipt:
        entry = self._get(proposal_id)

        def commit_allowed() -> bool:
            return guard() and self._source_current(entry)

        if not commit_allowed():
            raise PermissionError("proposal_write_denied")
        if entry.receipt is not None:
            if entry.receipt.status is ProposalDecision.REJECTED:
                raise RoutineConflict("proposal_rejected")
            return entry.receipt
        if proposal_id in self._confirming:
            raise RoutineConflict("proposal_confirming")
        self._confirming[proposal_id] = entry.conversation_id
        try:
            receipt = await self._apply(entry, scheduler, actor, commit_allowed)
            entry.receipt = receipt
            entry.spec = None
        finally:
            del self._confirming[proposal_id]
        # A completed commit wins even if authority is lost before publication.
        # A retry can see the receipt only while the source remains current.
        if not self._source_current(entry):
            raise ProposalNotFound
        return receipt

    @staticmethod
    async def _apply(
        entry: _Proposal, scheduler: RoutineScheduler, actor: str, guard: AuthorityGuard,
    ) -> ProposalReceipt:
        if entry.operation is ProposalOperation.CREATE:
            assert entry.spec is not None
            routine = await scheduler.register_routine(entry.spec, actor=actor, guard=guard)
        else:
            routine_id, revision = entry.routine_id, entry.expected_revision
            assert routine_id is not None and revision is not None
            try:
                if entry.operation is ProposalOperation.UPDATE:
                    assert entry.spec is not None
                    routine = await scheduler.update_routine(
                        routine_id, entry.spec, expected_revision=revision, actor=actor, guard=guard,
                    )
                elif entry.operation is ProposalOperation.DELETE:
                    await scheduler.unregister_routine(routine_id, expected_revision=revision, guard=guard)
                    return ProposalReceipt(entry.proposal_id, ProposalDecision.CONFIRMED, routine_id)
                else:
                    toggle = (scheduler.enable_routine if entry.operation is ProposalOperation.ENABLE
                              else scheduler.disable_routine)
                    routine = await toggle(routine_id, expected_revision=revision, actor=actor, guard=guard)
            except RoutineNotFound:
                raise RoutineConflict("revision_conflict") from None
        return ProposalReceipt(entry.proposal_id, ProposalDecision.CONFIRMED, routine.routine_id, routine.revision)

    def reject(self, proposal_id: str) -> ProposalReceipt:
        """Discard a pending definition and retain a minimal refusal for retries."""
        entry = self._get(proposal_id)
        if proposal_id in self._confirming:
            raise RoutineConflict("proposal_confirming")
        if entry.receipt is None:
            entry.receipt = ProposalReceipt(proposal_id, ProposalDecision.REJECTED)
            entry.spec = None
        elif entry.receipt.status is ProposalDecision.CONFIRMED:
            raise RoutineConflict("proposal_confirmed")
        return entry.receipt

    def prune_retained(self) -> dict:
        """Remove expired and committed-revoked rows during the context sweep."""
        now = self.source_policy.clock()
        removed = self._drop(
            lambda entry: entry.expires_at <= now
            or self.source_policy.conversation_committed_revoked(entry.conversation_id)
        )
        return {
            "next_at": min((entry.expires_at for entry in self._entries.values()), default=now + 30),
            "pruned": removed, "unknown": 0, "pending_operations": len(self._confirming),
        }

    def erase_conversation(self, conversation_id: str) -> int:
        """Remove a committed source and count any still-retained confirmations."""
        if not self.source_policy.conversation_committed_revoked(conversation_id):
            raise RuntimeError("proposal_erasure_authority_unavailable")
        self._drop(lambda entry: entry.conversation_id == conversation_id)
        return sum(source == conversation_id for source in self._confirming.values())

    async def aclose(self) -> None:
        """Seal the tray before the scheduler closes and drain admitted commits."""
        self.begin_close()
        await self._work.aclose()

    def begin_close(self) -> asyncio.Task:
        """Retain the actual drain before the source authority closes."""
        self._entries.clear()
        return self._work.begin_close()

    def _drop(self, predicate: Callable[[_Proposal], bool]) -> int:
        removed = [key for key, entry in self._entries.items() if predicate(entry)]
        for key in removed:
            del self._entries[key]
        return len(removed)

    def _require_available(self) -> None:
        if self._work.status()["closed"] or not self.source_policy.authority_available:
            raise ProposalUnavailable

    def _source_current(self, entry: _Proposal) -> bool:
        # Also called from the routines SQLite worker, using the ledger's
        # in-memory revocation fence (including pending deletion).
        return (
            self.source_policy.authority_available
            and self.source_policy.clock() < entry.expires_at
            and not self.source_policy.conversation_revoked(entry.conversation_id)
        )

    def _get(self, proposal_id: str) -> _Proposal:
        self._require_available()
        self.prune_retained()
        entry = self._entries.get(proposal_id)
        if entry is None or not self._source_current(entry):
            raise ProposalNotFound
        return entry
