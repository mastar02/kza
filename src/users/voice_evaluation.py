"""Read one approved evaluation modality without operator voice projections."""

from base64 import b64decode, b64encode
from dataclasses import dataclass, field
import json
import math
import re
import sqlite3
from weakref import ref

from src.dashboard.authority import (
    DashboardAuthority, DashboardLease, DashboardRole, EvaluationModality, EvaluationSample,
    MAX_CREDENTIAL_TTL_S, source_digest,
)
from src.dashboard.evaluation_grants import EvaluationGrantRecord, EvaluationGrantSelection, PreparedEvaluationScope
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.live_retention_types import EvaluationScopeFence, RevisionSet
from src.memory.fact_ledger import FactLedger, MemorySource
from src.users.voice_capture import validate_voice_preview
from src.users.voice_capture_metadata import (
    MAX_METADATA_BYTES, decode_audio_metadata, decode_transcript_metadata,
    encode_audio_metadata, encode_transcript_metadata,
)


MAX_SOURCE_JSON_BYTES = 262144


def _metadata_tree(value, *, depth=0, budget=None):
    """Bound native mutable DTOs before serializing them through the codec."""
    budget = [64] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError('invalid_evaluation_metadata')
    if value is None or type(value) is bool:
        return value
    if type(value) is int and -(1 << 63) <= value <= (1 << 63) - 1:
        return value
    if type(value) is str and len(value) <= 128:
        return value
    if type(value) is dict and depth <= 2 and len(value) <= 12:
        if any(type(key) is not str or len(key)>32 for key in value):
            raise ValueError('invalid_evaluation_metadata')
        return {key:_metadata_tree(item,depth=depth+1,budget=budget) for key,item in value.items()}
    raise ValueError('invalid_evaluation_metadata')


def _selected_metadata(raw, *, transcript):
    """Decode bounded selected SQL bytes, returning a fresh canonical object."""
    document = encoded = provenance = span = truncated = None
    invalid = False
    try:
        if raw is not None:
            if type(raw) is not bytes or not 1<=len(raw)<=MAX_METADATA_BYTES:
                raise ValueError('invalid_evaluation_metadata')
            encoded = raw.decode('utf-8')
        if transcript:
            provenance,span,truncated = decode_transcript_metadata(encoded)
            encoded = encode_transcript_metadata(provenance,span,truncated)
        else:
            provenance,span = decode_audio_metadata(encoded)
            encoded = encode_audio_metadata(provenance,span)
        document = None if encoded is None else json.loads(encoded)
        return document,provenance,span,truncated
    except (ValueError,TypeError,AttributeError,KeyError,OverflowError,RecursionError):
        invalid = True
    finally:
        raw = encoded = document = provenance = span = truncated = None
    if invalid:
        raise ValueError('invalid_evaluation_metadata')


def _metadata_object(value, *, transcript):
    """Revalidate public/native objects without trusting their mutable aliases."""
    document = raw = None
    invalid = False
    try:
        if value is None:
            return _selected_metadata(None,transcript=transcript)
        if type(value) is not dict:
            raise ValueError('invalid_evaluation_metadata')
        document = _metadata_tree(value)
        raw = json.dumps(document,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        return _selected_metadata(raw,transcript=transcript)
    except (ValueError,TypeError,AttributeError,KeyError,OverflowError,RecursionError):
        invalid = True
    finally:
        value = document = raw = None
    if invalid:
        raise ValueError('invalid_evaluation_metadata')


def _check_audio_metadata(audio_wav, clip_duration_s, provenance, span, *, duration_s=None):
    """Validate only audio and its descriptor; never read text to do so."""
    try:
        validate_voice_preview(audio_wav,clip_duration_s)
        if provenance is not None and (provenance.sample_rate_hz!=16000
                or not 12800<=provenance.frame_count<=480000
                or span.start_frame!=0 or span.end_frame!=min(provenance.frame_count,96000)
                or clip_duration_s!=(span.end_frame-span.start_frame)/16000
                or duration_s is not None and duration_s!=provenance.frame_count/16000):
            raise ValueError('invalid_evaluation_audio_metadata')
    finally:
        audio_wav = provenance = span = None


def _audio_projection_metadata(encoded, clip_duration_s, value):
    preview = document = provenance = span = _truncated = None
    invalid = False
    try:
        document,provenance,span,_truncated = _metadata_object(value,transcript=False)
        preview = b64decode(encoded,validate=True)
        _check_audio_metadata(preview,clip_duration_s,provenance,span)
        return document
    except (ValueError,TypeError,AttributeError,KeyError,OverflowError,RecursionError):
        invalid = True
    finally:
        encoded = value = preview = document = provenance = span = _truncated = None
    if invalid:
        raise ValueError('invalid_evaluation_payload')


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
        expected = ({'id','modality','audio_mime','audio_base64','clip_duration_s','audio_metadata'}
            if self.actor.modality is EvaluationModality.AUDIO_PREVIEW
            else {'id','modality','transcript_scope','transcript','transcript_metadata'})
        if (type(row) is not dict or set(row)!=expected or row['id']!=self.actor.fingerprint_id
                or type(row['id']) is not str or type(row['modality']) is not str
                or row['modality']!=self.actor.modality.value):
            raise ValueError('invalid_evaluation_payload')
        if self.actor.modality is EvaluationModality.AUDIO_PREVIEW:
            if (type(row['audio_mime']) is not str or row['audio_mime']!='audio/wav' or type(row['clip_duration_s']) not in (int,float)
                    or not .8<=row['clip_duration_s']<=6. or type(row['audio_base64']) is not str
                    or not 1<=len(row['audio_base64'])<=256060 or len(row['audio_base64'])%4
                    or re.fullmatch(r'[A-Za-z0-9+/]+={0,2}',row['audio_base64']) is None):
                raise ValueError('invalid_evaluation_payload')
            row['audio_metadata'] = _audio_projection_metadata(
                row['audio_base64'],row['clip_duration_s'],row['audio_metadata'])
        else:
            if (type(row['transcript_scope']) is not str or row['transcript_scope']!='full_segment'
                    or type(row['transcript']) is not str or len(row['transcript'])>8192):
                raise ValueError('invalid_evaluation_payload')
            document,provenance,span,truncated = _metadata_object(row['transcript_metadata'],transcript=True)
            if provenance is not None and (span.start_frame!=0 or span.end_frame!=provenance.frame_count):
                raise ValueError('invalid_evaluation_payload')
            row['transcript_metadata'] = document
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

    def prepare_grant(self,issuer: DashboardLease,selections: tuple[EvaluationGrantSelection,...],
                      credential_expires_at: float) -> PreparedEvaluationScope:
        """Prepare real metadata only; the grant issuer never reads private bytes."""
        if (type(issuer) is not DashboardLease or type(issuer._authority) is not DashboardAuthority
                or issuer.principal.role is not DashboardRole.OPERATOR
                or not issuer.current(frozenset({'auth:manage'}))):
            raise PermissionError('evaluation_grant_denied')
        if (type(selections) is not tuple or not 1<=len(selections)<=64
                or any(type(selection) is not EvaluationGrantSelection for selection in selections)
                or len({selection.fingerprint_id for selection in selections})!=len(selections)
                or type(credential_expires_at) not in (int,float) or not math.isfinite(credential_expires_at)):
            raise ValueError('invalid_evaluation_grant_selection')
        observed = issuer._clock()
        if (not observed<credential_expires_at<=observed+MAX_CREDENTIAL_TTL_S
                or any(selection.expires_at<=observed for selection in selections)):
            raise ValueError('invalid_evaluation_grant_deadline')
        records = []
        generation = None
        metadata = sources = digests = sample = result = None
        denied = invalid = False

        def guard():
            if not issuer.current(frozenset({'auth:manage'})):
                raise PermissionError('evaluation_grant_denied')
            if generation is not None and self.ledger.lifecycle_generation!=generation:
                raise LookupError('evaluation_grant_unavailable')
            now = max(issuer._clock(),self.ledger.clock())
            if any(now>=record.sample.expires_at or not self.ledger.sources_available(record.sources)
                    for record in records):
                raise LookupError('evaluation_grant_unavailable')

        try:
            with self.ledger._transaction(guard) as db:
                generation = self.ledger.lifecycle_generation
                for selection in selections:
                    guard()
                    metadata = db.execute('''SELECT CAST(f.sources_json AS BLOB) AS sources_json,expires_at
                        FROM voice_fingerprints AS f WHERE f.id=? AND typeof(f.sources_json)='text'
                        AND length(CAST(f.sources_json AS BLOB)) BETWEEN 1 AND ?
                        AND typeof(expires_at) IN ('integer','real') AND expires_at>=0
                        AND expires_at<=1.7976931348623157e308
                        AND EXISTS(SELECT 1 FROM voice_capture_reviews AS r WHERE r.fingerprint_id=f.id)''',
                        (selection.fingerprint_id,MAX_SOURCE_JSON_BYTES)).fetchone()
                    if metadata is None:
                        raise LookupError('evaluation_grant_unavailable')
                    sources,digests = _source_proof(metadata['sources_json'].decode('utf-8'))
                    if sum(len(record.sample.source_digests) for record in records)+len(digests)>1024:
                        raise LookupError('evaluation_grant_unavailable')
                    expiry = metadata['expires_at']
                    if type(expiry) not in (int,float) or not math.isfinite(expiry):
                        raise LookupError('evaluation_grant_unavailable')
                    deadline = min(credential_expires_at,selection.expires_at,expiry,
                        *[source.expires_at for source in sources],
                        *[dep.expires_at for source in sources for dep in source.fact_dependencies
                            if dep.expires_at is not None])
                    sample = EvaluationSample(selection.fingerprint_id,digests,selection.modalities,deadline)
                    records.append(EvaluationGrantRecord(selection,sample,sources,expiry))
                    metadata = sources = digests = sample = None
                    guard()
            guard()
            result = PreparedEvaluationScope(issuer,ref(self.ledger),generation,credential_expires_at,tuple(records))
            if not result.current(issuer._clock()):
                raise LookupError('evaluation_grant_unavailable')
            guard()
            return result
        except PermissionError:
            denied = True
        except (LookupError,ValueError,TypeError,OverflowError,RecursionError,sqlite3.Error):
            invalid = True
        finally:
            metadata = sources = digests = sample = result = None
            records.clear()
        if denied:
            raise PermissionError('evaluation_grant_denied')
        if invalid:
            raise LookupError('evaluation_grant_unavailable')

    def read(self, actor: EvaluationReadActor) -> EvaluationProjection:
        if type(actor) is not EvaluationReadActor:
            raise PermissionError('evaluation_scope_unavailable')
        proof = metadata = payload = data = sources = digests = result = transcript = None
        selected_document = selected_provenance = selected_span = _selected_truncated = None
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
                metadata = db.execute('''SELECT CAST(sources_json AS BLOB) AS sources_json,expires_at
                    FROM voice_fingerprints WHERE id=? AND typeof(sources_json)='text'
                    AND length(CAST(sources_json AS BLOB)) BETWEEN 1 AND ?
                    AND typeof(expires_at) IN ('integer','real') AND expires_at>=0
                    AND expires_at<=1.7976931348623157e308''',
                    (actor.fingerprint_id,MAX_SOURCE_JSON_BYTES)).fetchone()
                if metadata is None:
                    raise LookupError('evaluation_sample_unavailable')
                sources,digests = _source_proof(metadata['sources_json'].decode('utf-8'))
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
                    payload = db.execute('''SELECT audio_wav,duration_s,clip_duration_s,
                        CAST(audio_metadata_json AS BLOB) AS audio_metadata_json FROM voice_capture_reviews
                        WHERE fingerprint_id=? AND typeof(audio_wav)='blob'
                        AND length(audio_wav) BETWEEN 25644 AND 192044
                        AND typeof(duration_s) IN ('integer','real') AND duration_s BETWEEN 0.8 AND 30.0
                        AND typeof(clip_duration_s) IN ('integer','real')
                        AND clip_duration_s BETWEEN 0.8 AND 6.0 AND clip_duration_s<=duration_s
                        AND (audio_metadata_json IS NULL OR (typeof(audio_metadata_json)='text'
                            AND length(CAST(audio_metadata_json AS BLOB)) BETWEEN 1 AND ?))''',
                        (actor.fingerprint_id,MAX_METADATA_BYTES)).fetchone()
                else:
                    payload = db.execute('''SELECT CAST(transcript AS BLOB) AS transcript,
                        CAST(transcript_metadata_json AS BLOB) AS transcript_metadata_json FROM voice_capture_reviews
                        WHERE fingerprint_id=? AND typeof(transcript)='text'
                        AND length(CAST(transcript AS BLOB))<=32768
                        AND (transcript_metadata_json IS NULL OR (typeof(transcript_metadata_json)='text'
                            AND length(CAST(transcript_metadata_json AS BLOB)) BETWEEN 1 AND ?))''',
                        (actor.fingerprint_id,MAX_METADATA_BYTES)).fetchone()
                if payload is None:
                    raise LookupError('evaluation_sample_unavailable')
                guard()
            # Payload validation and encoding never hold the ledger lock.
            guard()
            if actor.modality is EvaluationModality.AUDIO_PREVIEW:
                selected_document,selected_provenance,selected_span,_selected_truncated = _selected_metadata(
                    payload['audio_metadata_json'],transcript=False)
                _check_audio_metadata(payload['audio_wav'],payload['clip_duration_s'],
                    selected_provenance,selected_span,duration_s=payload['duration_s'])
                guard()
                data = {'id':actor.fingerprint_id,'modality':actor.modality.value,
                    'audio_mime':'audio/wav','clip_duration_s':payload['clip_duration_s'],
                    'audio_base64':b64encode(payload['audio_wav']).decode('ascii'),
                    'audio_metadata':selected_document}
            else:
                transcript = payload['transcript'].decode('utf-8')
                if type(transcript) is not str or len(transcript) > 8192:
                    raise LookupError('evaluation_sample_unavailable')
                selected_document,selected_provenance,selected_span,_selected_truncated = _selected_metadata(
                    payload['transcript_metadata_json'],transcript=True)
                if selected_provenance is not None and (selected_span.start_frame!=0
                        or selected_span.end_frame!=selected_provenance.frame_count):
                    raise ValueError('invalid_evaluation_metadata')
                data = {'id':actor.fingerprint_id,'modality':actor.modality.value,
                    'transcript_scope':'full_segment','transcript':transcript,
                    'transcript_metadata':selected_document}
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
            selected_document = selected_provenance = selected_span = _selected_truncated = None
        # Raise after the handlers to drop validators' and constructors' private
        # tracebacks; `from None` alone would keep them through __context__.
        if denied:
            raise PermissionError('evaluation_scope_unavailable')
        if invalid:
            raise LookupError('evaluation_sample_unavailable')
