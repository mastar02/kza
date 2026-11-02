"""Exact, metadata-only authority for one selected evaluation modality."""

from dataclasses import dataclass, field
import math

from src.dashboard.authority import (
    DashboardAuthority, DashboardLease, DashboardRole, EvaluationModality,
    EvaluationSample,
)


def evaluation_digests(value: frozenset[str]) -> None:
    """Accept only a complete bounded set of actual source declaration hashes."""
    if (type(value) is not frozenset or not 1 <= len(value) <= 256
            or any(type(item) is not str or len(item) != 64
                or any(char not in '0123456789abcdef' for char in item) for item in value)):
        raise ValueError('invalid_evaluation_sources')


@dataclass(frozen=True, repr=False)
class EvaluationReadActor:
    """Keep the original server lease; never resolve another login by subject."""

    lease: DashboardLease = field(repr=False)
    fingerprint_id: str = field(repr=False)
    modality: EvaluationModality
    _selected: EvaluationSample = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (type(self.lease) is not DashboardLease
                or type(self.lease._authority) is not DashboardAuthority
                or self.lease.principal.role is not DashboardRole.EVALUATOR
                or type(self.modality) is not EvaluationModality
                or type(self.fingerprint_id) is not str or len(self.fingerprint_id) != 67
                or not self.fingerprint_id.startswith('vf_')
                or any(char not in '0123456789abcdef' for char in self.fingerprint_id[3:])
                or type(self.lease.principal.expires_at) not in (int,float)
                or not math.isfinite(self.lease.principal.expires_at)):
            raise PermissionError('evaluation_scope_unavailable')
        sample = self.lease.sample(self.fingerprint_id)
        if sample is None or self.modality not in sample.modalities:
            raise PermissionError('evaluation_scope_unavailable')
        object.__setattr__(self,'_selected',sample)

    def admit_metadata(self) -> EvaluationSample:
        """Grant lookup permits proof lookup only, never payload selection."""
        sample = self.lease.sample(self.fingerprint_id)
        if sample != self._selected or self.modality not in sample.modalities:
            raise PermissionError('evaluation_scope_unavailable')
        return sample

    def require_payload(self, actual_digests: frozenset[str]) -> EvaluationSample:
        """Revalidate the complete proof read from storage before private bytes."""
        evaluation_digests(actual_digests)
        sample = self.lease.sample(self.fingerprint_id,actual_digests)
        if sample != self._selected or self.modality not in sample.modalities:
            raise PermissionError('evaluation_scope_unavailable')
        return sample

    def now(self) -> float:
        """Use the authority's clamped clock, not its rounded cookie TTL hint."""
        return self.lease._clock()

    def grant_deadline(self) -> float:
        """Keep the exact original credential/session and selected-sample bound."""
        return min(self.lease.principal.expires_at,self._selected.expires_at)

    def current(self, actual_digests: frozenset[str], deadline: float) -> bool:
        try:
            self.require_payload(actual_digests)
            return (type(deadline) in (int,float) and math.isfinite(deadline)
                and deadline <= self.grant_deadline() and self.now() < deadline)
        except Exception:
            return False
