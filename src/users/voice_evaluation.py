"""Read one approved evaluation modality without operator voice projections."""

from base64 import b64encode
from dataclasses import dataclass, field
import json
import math
import re
import sqlite3

from src.dashboard.authority import EvaluationModality, source_digest
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.live_retention_types import EvaluationScopeFence, RevisionSet
from src.memory.fact_ledger import FactLedger, MemorySource
from src.users.voice_capture import validate_voice_preview


MAX_SOURCE_JSON_BYTES = 262144


def _source_proof(encoded: str) -> tuple[tuple[MemorySource, ...], frozenset[str]]:
    if type(encoded) is not str:
        raise ValueError('invalid_evaluation_sources')
    def unique_object(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('ambiguous_evaluation_sources')
            result[key] = value
        return result
    declarations = json.loads(encoded,object_pairs_hook=unique_object)
    if (type(declarations) is not list or not 1 <= len(declarations) <= 256
            or any(type(item) is not dict for item in declarations)):
        raise ValueError('invalid_evaluation_sources')
    # Hash the complete original declarations. A normalized MemorySource can
    # omit unknown fields or add optional defaults and is not that proof.
    digests = frozenset(source_digest(item) for item in declarations)
    sources = tuple(MemorySource.from_dict(item) for item in declarations)
    if sum(len(source.fact_dependencies) for source in sources) > 256:
        raise ValueError('invalid_evaluation_sources')
    return sources,digests


@dataclass(repr=False)
class EvaluationProjection:
    """Own one bounded row until its actual response consumer has finished."""

    rows: list[dict] = field(repr=False)
    sources: tuple[MemorySource, ...] = field(repr=False)
    source_digests: frozenset[str] = field(repr=False)
    observed_at: float
    expires_at: float
    actor: EvaluationReadActor = field(repr=False)
    channel: str = field(default='evaluation:sample',init=False)
    revision: RevisionSet = field(init=False)

    def __post_init__(self) -> None:
        if (type(self.actor) is not EvaluationReadActor or type(self.rows) is not list
                or len(self.rows) != 1 or type(self.sources) is not tuple or not 1<=len(self.sources)<=256
                or any(type(source) is not MemorySource for source in self.sources)
                or sum(len(source.fact_dependencies) for source in self.sources)>256
                or type(self.observed_at) not in (int,float) or not math.isfinite(self.observed_at)
                or type(self.expires_at) not in (int,float) or not math.isfinite(self.expires_at)
                or not self.observed_at < self.expires_at
                or self.expires_at > min(self.actor.grant_deadline(),*[source.expires_at for source in self.sources])):
            raise ValueError('invalid_evaluation_projection')
        row = self.rows[0]
        expected = ({'id','modality','audio_mime','audio_base64','clip_duration_s'}
            if self.actor.modality is EvaluationModality.AUDIO_PREVIEW
            else {'id','modality','transcript_scope','transcript'})
        if (type(row) is not dict or set(row)!=expected or row['id']!=self.actor.fingerprint_id
                or row['modality']!=self.actor.modality.value):
            raise ValueError('invalid_evaluation_payload')
        if self.actor.modality is EvaluationModality.AUDIO_PREVIEW:
            if (row['audio_mime']!='audio/wav' or type(row['clip_duration_s']) not in (int,float)
                    or not .8<=row['clip_duration_s']<=6. or type(row['audio_base64']) is not str
                    or not 1<=len(row['audio_base64'])<=256060 or len(row['audio_base64'])%4
                    or re.fullmatch(r'[A-Za-z0-9+/]+={0,2}',row['audio_base64']) is None):
                raise ValueError('invalid_evaluation_payload')
        elif (row['transcript_scope']!='full_segment' or type(row['transcript']) is not str
                or len(row['transcript'])>8192):
            raise ValueError('invalid_evaluation_payload')
        self.actor.require_payload(self.source_digests)
        if hasattr(self,'revision'):
            if type(self.revision) is not RevisionSet or len(self.revision.stamps)!=1:
                raise ValueError('invalid_evaluation_fence')
            fence = self.revision.stamps[0]
            if (type(fence) is not EvaluationScopeFence or fence.actor is not self.actor
                    or fence.sources!=self.sources or fence.source_digests!=self.source_digests
                    or fence.expires_at!=self.expires_at):
                raise ValueError('invalid_evaluation_fence')
        else:
            self.revision = RevisionSet((EvaluationScopeFence(self.actor,self.source_digests,self.expires_at,self.sources),))


class VoiceEvaluationReader:
    """Share the native ledger owner, with no model or group/profile lookup."""

    def __init__(self, ledger: FactLedger):
        if type(ledger) is not FactLedger:
            raise ValueError('invalid_evaluation_ledger')
        self.ledger = ledger

    def read(self, actor: EvaluationReadActor) -> EvaluationProjection:
        if type(actor) is not EvaluationReadActor:
            raise PermissionError('evaluation_scope_unavailable')
        proof = metadata = payload = data = sources = digests = result = transcript = None
        invalid = denied = False

        def guard():
            actor.admit_metadata()
            if proof is not None:
                actual_sources,actual_digests,deadline = proof
                actor.require_payload(actual_digests)
                if (max(actor.now(),self.ledger.clock()) >= deadline
                        or not self.ledger.sources_available(actual_sources)):
                    raise LookupError('evaluation_sample_unavailable')

        try:
            with self.ledger._transaction(guard):
                db = self.ledger._db
                metadata = db.execute('''SELECT sources_json,expires_at FROM voice_fingerprints
                    WHERE id=? AND length(CAST(sources_json AS BLOB))<=?''',
                    (actor.fingerprint_id,MAX_SOURCE_JSON_BYTES)).fetchone()
                if metadata is None:
                    raise LookupError('evaluation_sample_unavailable')
                sources,digests = _source_proof(metadata['sources_json'])
                sample = actor.require_payload(digests)
                expiry = metadata['expires_at']
                if type(expiry) not in (int,float) or not math.isfinite(expiry):
                    raise LookupError('evaluation_sample_unavailable')
                deadline = min(expiry,sample.expires_at,actor.lease.principal.expires_at,
                    *[source.expires_at for source in sources],
                    *[dep.expires_at for source in sources for dep in source.fact_dependencies
                        if dep.expires_at is not None])
                proof = sources,digests,deadline
                guard()
                if actor.modality is EvaluationModality.AUDIO_PREVIEW:
                    payload = db.execute('''SELECT audio_wav,clip_duration_s FROM voice_capture_reviews
                        WHERE fingerprint_id=? AND length(CAST(audio_wav AS BLOB)) BETWEEN 45 AND 192044''',
                        (actor.fingerprint_id,)).fetchone()
                else:
                    payload = db.execute('''SELECT transcript FROM voice_capture_reviews
                        WHERE fingerprint_id=? AND length(CAST(transcript AS BLOB))<=32768''',
                        (actor.fingerprint_id,)).fetchone()
                if payload is None:
                    raise LookupError('evaluation_sample_unavailable')
                guard()
            # Payload validation and encoding never hold the ledger lock.
            guard()
            if actor.modality is EvaluationModality.AUDIO_PREVIEW:
                validate_voice_preview(payload['audio_wav'],payload['clip_duration_s'])
                guard()
                data = {'id':actor.fingerprint_id,'modality':actor.modality.value,
                    'audio_mime':'audio/wav','clip_duration_s':payload['clip_duration_s'],
                    'audio_base64':b64encode(payload['audio_wav']).decode('ascii')}
            else:
                transcript = payload['transcript']
                if type(transcript) is not str or len(transcript) > 8192:
                    raise LookupError('evaluation_sample_unavailable')
                data = {'id':actor.fingerprint_id,'modality':actor.modality.value,
                    'transcript_scope':'full_segment','transcript':transcript}
                transcript = None
            guard()
            observed = max(actor.now(),self.ledger.clock())
            if observed >= deadline:
                raise LookupError('evaluation_sample_unavailable')
            result = EvaluationProjection([data],sources,digests,observed,deadline,actor)
            return result
        except PermissionError:
            denied = True
        except (LookupError,ValueError,TypeError,OverflowError,RecursionError,sqlite3.Error):
            invalid = True
        finally:
            proof = metadata = payload = data = sources = digests = result = transcript = None
        # Raise after the handlers to drop validators' and constructors' private
        # tracebacks; `from None` alone would keep them through __context__.
        if denied:
            raise PermissionError('evaluation_scope_unavailable')
        if invalid:
            raise LookupError('evaluation_sample_unavailable')
