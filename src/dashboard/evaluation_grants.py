"""Immutable metadata proofs for server-prepared temporary evaluation grants."""

from dataclasses import dataclass, field
import math
from weakref import ReferenceType

from src.dashboard.authority import (
    DashboardAuthority, DashboardLease, DashboardRole, EvaluationModality, EvaluationSample,
)
from src.memory.fact_ledger import FactLedger, MemorySource


@dataclass(frozen=True,repr=False)
class EvaluationGrantSelection:
    fingerprint_id: str
    modalities: frozenset[EvaluationModality]
    expires_at: float

    def __post_init__(self):
        if (type(self.fingerprint_id) is not str or len(self.fingerprint_id)!=67
                or not self.fingerprint_id.startswith('vf_')
                or any(char not in '0123456789abcdef' for char in self.fingerprint_id[3:])
                or type(self.modalities) is not frozenset or not self.modalities
                or any(type(mode) is not EvaluationModality for mode in self.modalities)
                or type(self.expires_at) not in (int,float) or not math.isfinite(self.expires_at)):
            raise ValueError('invalid_evaluation_grant_selection')


@dataclass(frozen=True,repr=False)
class EvaluationGrantRecord:
    selection: EvaluationGrantSelection
    sample: EvaluationSample
    sources: tuple[MemorySource,...]
    fingerprint_expires_at: float

    def __post_init__(self):
        if (type(self.selection) is not EvaluationGrantSelection or type(self.sample) is not EvaluationSample
                or type(self.sources) is not tuple or not 1<=len(self.sources)<=256
                or any(type(source) is not MemorySource for source in self.sources)
                or sum(len(source.fact_dependencies) for source in self.sources)>256
                or type(self.fingerprint_expires_at) not in (int,float)
                or not math.isfinite(self.fingerprint_expires_at)
                or self.sample.fingerprint_id!=self.selection.fingerprint_id
                or self.sample.modalities!=self.selection.modalities
                or self.sample.expires_at>min(self.selection.expires_at,self.fingerprint_expires_at,
                    *[source.expires_at for source in self.sources])):
            raise ValueError('invalid_evaluation_grant_record')


@dataclass(frozen=True,repr=False)
class PreparedEvaluationScope:
    """Keep real source proof for exactly one opening and original issuer."""

    issuer: DashboardLease
    policy: ReferenceType
    generation: int
    credential_expires_at: float
    records: tuple[EvaluationGrantRecord,...]
    samples: tuple[EvaluationSample,...] = field(init=False)

    def __post_init__(self):
        if (type(self.issuer) is not DashboardLease or type(self.issuer._authority) is not DashboardAuthority
                or self.issuer.principal.role is not DashboardRole.OPERATOR
                or 'auth:manage' not in self.issuer.principal.permissions
                or type(self.policy) is not ReferenceType or type(self.policy()) is not FactLedger
                or type(self.generation) is not int or self.generation<1
                or type(self.credential_expires_at) not in (int,float)
                or not math.isfinite(self.credential_expires_at)
                or type(self.records) is not tuple or not 1<=len(self.records)<=64
                or any(type(record) is not EvaluationGrantRecord for record in self.records)
                or len({record.sample.fingerprint_id for record in self.records})!=len(self.records)
                or sum(len(record.sample.source_digests) for record in self.records)>1024):
            raise ValueError('invalid_prepared_evaluation_scope')
        for record in self.records:
            deadline = min(self.credential_expires_at,record.selection.expires_at,
                record.fingerprint_expires_at,*[source.expires_at for source in record.sources],
                *[dep.expires_at for source in record.sources for dep in source.fact_dependencies
                    if dep.expires_at is not None])
            if record.sample.expires_at!=deadline:
                raise ValueError('invalid_prepared_evaluation_deadline')
        object.__setattr__(self,'samples',tuple(record.sample for record in self.records))

    def matches(self,issuer,role,expires_at,user_id,samples) -> bool:
        return (issuer is self.issuer and role is DashboardRole.EVALUATOR and user_id is None
            and type(expires_at) in (int,float) and expires_at==self.credential_expires_at
            and type(samples) is tuple and samples==self.samples)

    def current(self,now: float) -> bool:
        """Check only source lifetime; Authority owns the original issuer guard."""
        try:
            policy = self.policy()
            if (type(policy) is not FactLedger or policy.lifecycle_generation!=self.generation
                    or type(now) not in (int,float) or not math.isfinite(now)):
                return False
            clocks = now,policy.clock(),self.issuer._clock()
            if any(type(value) not in (int,float) or not math.isfinite(value) for value in clocks):
                return False
            observed = max(clocks)
            if not all(observed<record.sample.expires_at and policy.sources_available(record.sources)
                    for record in self.records):
                return False
            clocks = now,policy.clock(),self.issuer._clock()
            return (all(type(value) in (int,float) and math.isfinite(value) for value in clocks)
                and all(max(clocks)<record.sample.expires_at for record in self.records)
                and policy.lifecycle_generation==self.generation)
        except Exception:
            return False
