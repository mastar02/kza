"""Source-bound acoustic fingerprints and explicit permanent profile review."""

from collections.abc import Callable
import base64
from dataclasses import dataclass,field
import hashlib
import json
import math
import struct
from uuid import UUID,uuid4
from weakref import ref

import numpy as np

from src.dashboard.auth import DashboardPrincipal
from src.memory.fact_ledger import MemorySource
from src.users.voice_capture import VoiceCaptureReview
from src.users.voice_groups import VoiceGroups
from src.users.voice_projection import VoiceProjection
from src.dashboard.live_retention_types import RevisionSet,RevisionStamp


@dataclass(frozen=True)
class VoiceReviewer:
    """Dashboard review authority; acoustic labels never construct it."""

    subject: str
    _current: Callable[[],bool] | None = field(default=None,repr=False,compare=False)

    @classmethod
    def from_dashboard(cls, principal: DashboardPrincipal, current: Callable[[],bool] | None = None) -> 'VoiceReviewer':
        """Require server-owned review permission, independently of voice matches."""
        if type(principal) is not DashboardPrincipal or 'voices:review' not in principal.permissions:
            raise PermissionError('voice_review_denied')
        return cls(principal.subject,current)

    def require_current(self) -> None:
        """A queued native mutation cannot outlive credential revocation."""
        if self._current is not None and self._current() is not True:
            raise PermissionError('voice_review_denied')


def vector_bytes(vector: np.ndarray) -> tuple[bytes,int]:
    """Validate and detach one finite float32 vector without object coercion."""
    if (type(vector) is not np.ndarray or vector.dtype!=np.float32 or vector.ndim!=1
            or not 16<=vector.size<=1024 or not np.isfinite(vector).all()):
        raise ValueError('invalid_voice_embedding')
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm<1e-8:
        raise ValueError('invalid_voice_embedding')
    return (vector/norm).astype('<f4').tobytes(),int(vector.size)


class VoiceLedger:
    """Use the leased authority transaction for every read, review and mutation."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.groups = VoiceGroups(self)

    @property
    def projection_revision(self) -> int:
        """All wrappers over the sole authority share one invalidation fence."""
        return self.ledger._voice_projection_revision

    @staticmethod
    def _sources(encoded):
        decoded = json.loads(encoded)
        if type(decoded) is not list or not 1<=len(decoded)<=256:
            raise ValueError('invalid_voice_sources')
        return tuple(MemorySource.from_dict(source) for source in decoded)

    @staticmethod
    def _model(model_key):
        if type(model_key) is not str or not 1<=len(model_key)<=256:
            raise ValueError('invalid_voice_encoder')

    @staticmethod
    def _capture_cursor(row) -> str:
        """Retain the sort position even when its capture expires between pages."""
        return 'vc1_' + struct.pack('>d', row['created_at']).hex() + '_' + row['id'][3:]

    @staticmethod
    def _capture_position(cursor):
        if cursor is None:
            return None
        if type(cursor) is not str:
            raise ValueError('invalid_voice_cursor')
        if (len(cursor) == 67 and cursor.startswith('vf_')
                and all(char in '0123456789abcdef' for char in cursor[3:])):
            return cursor  # Accept legacy ID cursors while their anchor exists.
        if (len(cursor) != 85 or not cursor.startswith('vc1_') or cursor[20] != '_'
                or any(char not in '0123456789abcdef' for char in cursor[4:20] + cursor[21:])):
            raise ValueError('invalid_voice_cursor')
        created_at = struct.unpack('>d', bytes.fromhex(cursor[4:20]))[0]
        if not math.isfinite(created_at) or created_at < 0:
            raise ValueError('invalid_voice_cursor')
        return created_at, 'vf_' + cursor[21:]

    def _live(self, row, *, confirmed=False):
        try:
            sources = self._sources(row['sources_json'])
            if confirmed:
                return (row['deleted_at'] is None and type(row['confirmed_by']) is str and bool(row['confirmed_by'])
                    and type(row['confirmed_at']) in (int,float) and math.isfinite(row['confirmed_at'])
                    and row['confirmed_at']<=self.ledger.clock() and self.ledger.activity_sources_available(sources))
            return self.ledger.clock()<row['expires_at'] and self.ledger.sources_available(sources)
        except (ValueError,KeyError,TypeError):
            return False

    def put(self, model_key: str, embedding: np.ndarray, duration_s: float, sources: tuple[MemorySource,...],
            review: VoiceCaptureReview | None = None) -> str | None:
        """Commit once per actual source set; retries cannot renew its deadline."""
        self._model(model_key)
        if (type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not .8<=duration_s<=30
                or type(sources) is not tuple or not 1<=len(sources)<=256
                or any(type(source) is not MemorySource for source in sources)
                or sum(len(source.fact_dependencies) for source in sources)>256):
            raise ValueError('invalid_voice_capture')
        if review is not None and (type(review) is not VoiceCaptureReview or abs(review.duration_s-duration_s)>1/16000):
            raise ValueError('invalid_voice_review_capture')
        encoded = json.dumps([source.to_dict() for source in sorted(sources,key=lambda source:source.capture_id)],sort_keys=True,separators=(',',':'))
        identity = 'vf_'+hashlib.sha256((model_key+encoded).encode()).hexdigest()
        raw,dimensions = vector_bytes(embedding)
        try:
            with self.ledger._transaction() as db:
                self.groups.initialize(db)
                if not self.ledger._sources_live(db,sources):
                    return None
                if db.execute('SELECT 1 FROM voice_fingerprints WHERE id=?',(identity,)).fetchone():
                    return identity
                if db.execute('SELECT COUNT(*) FROM voice_fingerprints').fetchone()[0]>=4096:
                    raise RuntimeError('voice_fingerprint_capacity')
                suggested,revision,similarity = None,None,None
                for row in db.execute('SELECT * FROM voice_profiles WHERE deleted_at IS NULL AND model_key=? AND dimensions=?',(model_key,dimensions)):
                    if not self._live(row,confirmed=True):
                        continue
                    profile = np.frombuffer(row['embedding'],dtype='<f4')
                    score = float(np.dot(np.frombuffer(raw,dtype='<f4'),profile))
                    if math.isfinite(score) and (similarity is None or score>similarity):
                        suggested,revision,similarity = row['id'],row['revision'],max(-1.,min(1.,score))
                if not self.ledger._sources_live(db,sources):
                    return None
                db.execute('INSERT INTO voice_fingerprints VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (identity,model_key,dimensions,raw,duration_s,self.ledger.clock(),min(source.expires_at for source in sources),encoded,suggested,revision,similarity))
                db.executemany('INSERT INTO voice_fingerprint_sources VALUES (?,?)',
                    [(identity,conversation) for conversation in sorted({source.conversation_id for source in sources})])
                if review is not None:
                    db.execute('INSERT INTO voice_capture_reviews VALUES (?,?,?,?,?,?)',
                        (identity,review.audio_wav,review.transcript,review.room_id,review.duration_s,review.clip_duration_s))
                self.groups.add(db,identity,model_key,dimensions,raw,duration_s,sources)
                grouped = self.groups.metadata(db,identity)
                if grouped is None or not self.ledger._sources_live(db,grouped[1]):
                    raise LookupError('voice_capture_unavailable')
                return identity
        finally:
            embedding = raw = review = None

    def snapshot(self, cursor: str | None = None) -> dict:
        """Return newest-first bounded metadata, never vectors, transcripts or audio."""
        position = self._capture_position(cursor)
        with self.ledger._transaction() as db:
            if isinstance(position, str):
                anchor = db.execute('SELECT created_at FROM voice_fingerprints WHERE id=?', (position,)).fetchone()
                if anchor is None:
                    raise ValueError('invalid_voice_cursor')
                position = anchor['created_at'], position
            profiles = {row['id']:row for row in db.execute('SELECT id,name,revision,sources_json,confirmed_at,confirmed_by,deleted_at FROM voice_profiles WHERE deleted_at IS NULL')
                if self._live(row,confirmed=True)}
            captures = []
            group_cache = {}
            query = '''SELECT v.id,v.model_key,v.created_at,v.expires_at,v.duration_s,v.sources_json,
                v.suggested_profile,v.suggested_revision,v.similarity,
                EXISTS(SELECT 1 FROM voice_capture_reviews r WHERE r.fingerprint_id=v.id) AS review_available
                FROM voice_fingerprints v'''
            params = ()
            if position is not None:
                query += ' WHERE v.created_at<? OR (v.created_at=? AND v.id<?)'
                params = position[0], position[0], position[1]
            page = db.execute(query + ' ORDER BY v.created_at DESC,v.id DESC LIMIT 101', params).fetchall()
            for row in page[:100]:
                if not self._live(row):
                    continue
                group = self.groups.metadata(db,row['id'],cache=group_cache)
                if group is None:
                    continue
                suggestion = profiles.get(row['suggested_profile'])
                if suggestion is not None and suggestion['revision']!=row['suggested_revision']:
                    suggestion = None
                captures.append({'id':row['id'],**group[0],'created_at':row['created_at'],
                    'expires_at':row['expires_at'],'duration_s':row['duration_s'],'model_key':row['model_key'],
                    'capture_ids':[source.capture_id for source in self._sources(row['sources_json'])],
                    'review_available':bool(row['review_available']),
                    'suggested_profile':suggestion['id'] if suggestion is not None else None,
                    'similarity':row['similarity'] if suggestion is not None else None,
                    'association':'suggested' if suggestion is not None else 'unknown'})
            return {'observed_at':self.ledger.clock(),'fingerprints':captures,
                'profiles':[self._profile(row) for row in profiles.values()],
                'automatic_grouping_accepted':False,'legacy_policy':'excluded_pending_inventory',
                'recognition_grants_permissions':False,'similarity_metric':'cosine_not_identity_probability',
                'next_cursor':self._capture_cursor(page[99]) if len(page)>100 else None}

    def review_capture(self, identity: str, actor: VoiceReviewer) -> dict:
        """Return one listening sample only while its original raw sources live."""
        if type(actor) is not VoiceReviewer or not actor.subject:
            raise PermissionError('voice_review_denied')
        if (type(identity) is not str or len(identity)!=67 or not identity.startswith('vf_')
                or any(char not in '0123456789abcdef' for char in identity[3:])):
            raise ValueError('invalid_voice_capture_id')
        data = row = capture = None
        try:
            actor.require_current()
            with self.ledger._transaction() as db:
                row = db.execute('''SELECT v.sources_json,v.expires_at,r.* FROM voice_capture_reviews r
                    JOIN voice_fingerprints v ON v.id=r.fingerprint_id WHERE v.id=?
                    AND length(r.audio_wav)<=192044 AND length(r.transcript)<=8192''',(identity,)).fetchone()
                if row is None or not self._live(row):
                    raise LookupError('voice_capture_unavailable')
            # SQLite has already detached this bounded row. Preview validation
            # and encoding must not hold the authority's writer transaction.
            valid = True
            try:
                capture = VoiceCaptureReview(row['audio_wav'],row['transcript'],row['room_id'],row['duration_s'],row['clip_duration_s'])
            except ValueError:
                valid = False
            if not valid:
                raise LookupError('voice_capture_unavailable')
            actor.require_current()
            data = {'id':identity,'observed_at':self.ledger.clock(),'expires_at':row['expires_at'],
                'transcript':capture.transcript,'room_id':capture.room_id,
                'duration_s':capture.duration_s,'clip_duration_s':capture.clip_duration_s,
                'clip_truncated':capture.clip_duration_s<capture.duration_s,
                'transcript_scope':'full_segment','audio_mime':'audio/wav',
                'audio_base64':base64.b64encode(capture.audio_wav).decode('ascii')}
            actor.require_current()
            if not self._live(row):
                raise LookupError('voice_capture_unavailable')
            return data
        finally:
            row = data = capture = None

    @staticmethod
    def _profile(row):
        return {'id':row['id'],'name':row['name'],'revision':row['revision'],
            'confirmed_at':row['confirmed_at'],'confirmed_by':row['confirmed_by'],'expires_at':None}

    def project(self, kind: str, cursor: str | None, actor: VoiceReviewer) -> VoiceProjection:
        """Separate raw capture lists from source-linked permanent profile labels."""
        if type(actor) is not VoiceReviewer:
            raise PermissionError('voice_review_denied')
        actor.require_current()
        rows = sources = snapshot = data = result = page = row = original = None
        try:
            if kind=='review':
                data = self.review_capture(cursor,actor)
                with self.ledger._transaction() as db:
                    source_row = db.execute('SELECT sources_json FROM voice_fingerprints WHERE id=?',(cursor,)).fetchone()
                    if source_row is None:
                        raise LookupError('voice_capture_unavailable')
                    sources = self._sources(source_row['sources_json'])
                actor.require_current()
                if not self.ledger.sources_available(sources):
                    raise LookupError('voice_capture_unavailable')
                return VoiceProjection([data],sources,data['observed_at'],data['expires_at'],'voices:review')
            if kind not in ('captures','profiles'):
                raise ValueError('invalid_voice_projection')
            snapshot = self.snapshot(cursor) if kind=='captures' else None
            with self.ledger._transaction() as db:
                if kind=='profiles':
                    if cursor is not None and (type(cursor) is not str or str(UUID(cursor))!=cursor):
                        raise ValueError('invalid_voice_cursor')
                    page = db.execute('SELECT id,name,revision,sources_json,confirmed_at,confirmed_by,deleted_at FROM voice_profiles WHERE deleted_at IS NULL AND id>? ORDER BY id LIMIT 51',(cursor or '',)).fetchall()
                    rows,sources,refs = [],set(),[]
                    next_cursor = None
                    for index,row in enumerate(page):
                        if index==50:
                            next_cursor = page[49]['id']
                            break
                        if not self._live(row,confirmed=True):
                            continue
                        original = self._sources(row['sources_json'])
                        candidate = sources | set(original)
                        if len(candidate)>256 or sum(len(source.fact_dependencies) for source in candidate)>256:
                            if not rows:
                                raise ValueError('voice_profile_source_capacity')
                            next_cursor = page[index-1]['id'] if index else None
                            break
                        sources.update(original)
                        rows.append(self._profile(row))
                        refs.append((row['id'],row['revision']))
                    observed = self.ledger.clock()
                    expiry = min([observed+300,*[dep.expires_at for source in sources
                        for dep in source.fact_dependencies if dep.expires_at is not None]])
                    result = VoiceProjection(rows,tuple(sources),observed,expiry,'voices:profiles',next_cursor,tuple(refs),
                        RevisionStamp(ref(self),self.projection_revision))
                else:
                    rows,sources,extra,associated = [],set(),set(),set()
                    group_cache = {}
                    next_cursor = snapshot['next_cursor']
                    for index,item in enumerate(snapshot['fingerprints']):
                        if index==50:
                            next_cursor = self._capture_cursor(rows[-1])
                            break
                        original = self._sources(db.execute('SELECT sources_json FROM voice_fingerprints WHERE id=?',(item['id'],)).fetchone()[0])
                        parent = None
                        related = ()
                        if item['suggested_profile']:
                            parent = db.execute('SELECT id,revision,sources_json,confirmed_at,confirmed_by,deleted_at FROM voice_profiles WHERE id=?',(item['suggested_profile'],)).fetchone()
                            if parent is not None and self._live(parent,confirmed=True):
                                related = self._sources(parent['sources_json'])
                            else:
                                item = dict(item,suggested_profile=None,similarity=None,association='unknown')
                        group = self.groups.metadata(db,item['id'],cache=group_cache)
                        if (group is None or group[0]['group_id']!=item['group_id']
                                or group[0]['group_revision']!=item['group_revision']):
                            raise LookupError('voice_group_changed')
                        original = tuple(set(original) | set(group[1]))
                        candidate = sources | extra | set(original) | set(related)
                        if len(candidate)>256 or sum(len(source.fact_dependencies) for source in candidate)>256:
                            if rows:
                                next_cursor = self._capture_cursor(rows[-1])
                                break
                            # An unrepresentable hint is an abstention, never
                            # an association with silently dropped parents.
                            item = dict(item,suggested_profile=None,similarity=None,association='unknown')
                            related = ()
                            if len(set(original))>256 or sum(len(source.fact_dependencies) for source in original)>256:
                                raise ValueError('voice_capture_source_capacity')
                        sources.update(original)
                        extra.update(related)
                        if related:
                            associated.add((parent['id'],parent['revision']))
                        rows.append(item)
                    observed = self.ledger.clock()
                    expiry = min([*[source.expires_at for source in sources],*[dep.expires_at for source in extra
                        for dep in source.fact_dependencies if dep.expires_at is not None]],default=observed)
                    result = VoiceProjection(rows,tuple(sources),observed,expiry,'voices:captures',next_cursor,
                        revision=RevisionSet((RevisionStamp(ref(self.groups),self.groups.projection_revision),
                            *([RevisionStamp(ref(self),self.projection_revision)] if associated else []))),
                        associated_profiles=tuple(sorted(associated)),activity_sources=tuple(extra))
                actor.require_current()
                return result
        finally:
            rows = sources = snapshot = data = result = page = row = original = None

    def confirm(self, identities: tuple[str,...], name: str, request_id: str, actor: VoiceReviewer) -> dict:
        """Permanently retain a reviewed centroid; no account or permission changes."""
        if type(actor) is not VoiceReviewer or not actor.subject:
            raise PermissionError('voice_review_denied')
        actor.require_current()
        if type(name) is not str or not 1<=len(name.strip())<=64 or any(ord(char)<32 for char in name):
            raise ValueError('invalid_voice_profile_name')
        if type(identities) is not tuple or not 3<=len(identities)<=32 or len(set(identities))!=len(identities):
            raise ValueError('voice_profile_requires_distinct_captures')
        if type(request_id) is not str or str(UUID(request_id))!=request_id:
            raise ValueError('invalid_voice_review_request')
        digest = hashlib.sha256(json.dumps([sorted(identities),name.strip()]).encode()).hexdigest()
        with self.ledger._transaction() as db:
            actor.require_current()
            prior = db.execute('SELECT * FROM voice_review_requests WHERE actor=? AND request_id=?',(actor.subject,request_id)).fetchone()
            if prior is not None:
                profile = db.execute('SELECT * FROM voice_profiles WHERE id=?',(prior['profile_id'],)).fetchone()
                if profile is None or not self._live(profile,confirmed=True):
                    raise LookupError('voice_profile_unavailable')
                if prior['digest']!=digest:
                    raise ValueError('voice_review_request_conflict')
                return self._profile(profile)
            rows = [db.execute('SELECT * FROM voice_fingerprints WHERE id=?',(identity,)).fetchone() for identity in identities]
            if any(row is None or not self._live(row) for row in rows):
                raise LookupError('voice_capture_unavailable')
            if len({(row['model_key'],row['dimensions']) for row in rows})!=1:
                raise ValueError('incompatible_voice_encoders')
            sources = tuple({source for row in rows for source in self._sources(row['sources_json'])})
            if len(sources)>256 or len({source.capture_id for source in sources})<3:
                raise ValueError('voice_profile_requires_distinct_captures')
            if sum(len(source.fact_dependencies) for source in sources)>256:
                raise ValueError('voice_profile_source_capacity')
            if db.execute('SELECT COUNT(*) FROM voice_profiles WHERE deleted_at IS NULL').fetchone()[0]>=256:
                raise RuntimeError('voice_profile_capacity')
            vector = np.mean([np.frombuffer(row['embedding'],dtype='<f4') for row in rows],axis=0).astype(np.float32)
            raw,dimensions = vector_bytes(vector)
            if not self.ledger._sources_live(db,sources):
                raise LookupError('voice_capture_unavailable')
            actor.require_current()
            identity = str(uuid4())
            self.ledger._voice_projection_revision+=1
            encoded = json.dumps([source.to_dict() for source in sorted(sources,key=lambda source:source.capture_id)],sort_keys=True)
            db.execute('INSERT INTO voice_profiles VALUES (?,?,?,?,?,?,?,?,?,NULL)',
                (identity,name.strip(),rows[0]['model_key'],dimensions,raw,encoded,self.ledger.clock(),actor.subject,1))
            db.executemany('INSERT INTO voice_profile_sources VALUES (?,?)',[(identity,conversation) for conversation in sorted({source.conversation_id for source in sources})])
            db.execute('INSERT INTO voice_review_requests VALUES (?,?,?,?)',(actor.subject,request_id,digest,identity))
            return self._profile(db.execute('SELECT * FROM voice_profiles WHERE id=?',(identity,)).fetchone())

    def delete_profile(self, identity: str, revision: int, actor: VoiceReviewer) -> bool:
        """Erase vector and name while preserving a minimal stale-write fence."""
        if type(actor) is not VoiceReviewer:
            raise PermissionError('voice_review_denied')
        with self.ledger._transaction() as db:
            actor.require_current()
            row = db.execute('SELECT * FROM voice_profiles WHERE id=?',(identity,)).fetchone()
            if row is None or row['deleted_at'] is not None:
                return False
            if type(revision) is not int or revision!=row['revision']:
                raise ValueError('voice_profile_revision_conflict')
            self._delete(db,identity)
            return True

    def _delete(self, db, identity):
        self.ledger._voice_projection_revision+=1
        db.execute('UPDATE voice_profiles SET name=NULL,embedding=NULL,deleted_at=?,revision=revision+1 WHERE id=? AND deleted_at IS NULL',
            (self.ledger.clock(),identity))
        db.execute('UPDATE voice_fingerprints SET suggested_profile=NULL,suggested_revision=NULL,similarity=NULL WHERE suggested_profile=?',(identity,))
        db.execute("UPDATE voice_review_requests SET digest='deleted' WHERE profile_id=?",(identity,))

    def prune(self) -> None:
        """Physically remove expired captures and committed-invalid profiles."""
        with self.ledger._transaction() as db:
            self.groups.prune(db)
            for row in db.execute('SELECT id,sources_json,expires_at FROM voice_fingerprints').fetchall():
                sources = self._sources(row['sources_json'])
                if self.ledger.clock()>=row['expires_at'] or self.ledger.sources_committed_unavailable(sources):
                    db.execute('DELETE FROM voice_fingerprints WHERE id=?',(row['id'],))
            for row in db.execute('SELECT * FROM voice_profiles WHERE deleted_at IS NULL').fetchall():
                if self.ledger.sources_committed_unavailable(self._sources(row['sources_json'])):
                    self._delete(db,row['id'])

    def pending(self, conversation: str) -> int:
        """Count actual retained vectors linked to a committed deletion."""
        with self.ledger._transaction() as db:
            pending = 0
            for query in ('SELECT sources_json FROM voice_fingerprints',
                    'SELECT sources_json FROM voice_profiles WHERE embedding IS NOT NULL',
                    'SELECT sources_json FROM voice_groups', 'SELECT sources_json FROM voice_group_requests'):
                for row in db.execute(query):
                    try:
                        sources = self._sources(row['sources_json'])
                        pending+=any(source.conversation_id==conversation for source in sources)
                    except (ValueError,KeyError,TypeError):
                        pending+=1
            # A damaged partial restore cannot certify unindexed or orphaned PCM.
            pending+=db.execute('''SELECT COUNT(*) FROM voice_capture_reviews r LEFT JOIN voice_fingerprints v
                ON v.id=r.fingerprint_id WHERE v.id IS NULL''').fetchone()[0]
            pending+=db.execute('''SELECT COUNT(*) FROM voice_group_members m LEFT JOIN voice_groups g ON g.id=m.group_id
                LEFT JOIN voice_fingerprints v ON v.id=m.fingerprint_id WHERE g.id IS NULL OR v.id IS NULL''').fetchone()[0]
            pending+=db.execute('''SELECT COUNT(*) FROM voice_fingerprints v LEFT JOIN voice_group_members m
                ON v.id=m.fingerprint_id WHERE m.fingerprint_id IS NULL''').fetchone()[0]
            pending+=db.execute('''SELECT COUNT(*) FROM voice_group_lineage p LEFT JOIN voice_groups g
                ON g.id=p.group_id WHERE g.id IS NULL''').fetchone()[0]
            pending+=db.execute('''SELECT COUNT(*) FROM voice_group_request_lineage p LEFT JOIN voice_group_requests r
                ON r.actor=p.actor AND r.request_id=p.request_id WHERE r.request_id IS NULL''').fetchone()[0]
            pending+=sum(not self.groups._request_complete(db,row) for row in db.execute('SELECT * FROM voice_group_requests'))
            for group in db.execute('''SELECT g.*,(SELECT m.fingerprint_id FROM voice_group_members m
                    WHERE m.group_id=g.id LIMIT 1) AS fingerprint_id FROM voice_groups g'''):
                # Valid revocation/expiry has known lineage; incomplete live
                # groups remain unknown without scanning all members per row.
                if not self.groups._proof_complete(db,group) or (self._live(group) and (group['fingerprint_id'] is None
                        or self.groups.metadata(db,group['fingerprint_id']) is None)):
                    pending+=1
            return pending
