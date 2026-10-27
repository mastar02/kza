"""Bounded raw voice groups and source-bound manual corrections."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import TYPE_CHECKING
from uuid import UUID,uuid4

import numpy as np

from src.memory.fact_ledger import MemorySource

if TYPE_CHECKING:
    from src.users.voice_ledger import VoiceLedger,VoiceReviewer

MIN_GROUP_DURATION = 2.
MIN_GROUP_COSINE = .88
GROUP_MARGIN = .08
MAX_GROUP_MEMBERS = 32


class VoiceGroups:
    """Persist candidate membership; manual edits never become identity authority."""

    def __init__(self, voices: VoiceLedger):
        self.voices = voices
        self.ledger = voices.ledger

    @property
    def projection_revision(self) -> int:
        """All group wrappers use the same authority fence."""
        return self.ledger._voice_group_revision

    def _changed(self):
        self.ledger._voice_group_revision+=1

    @staticmethod
    def _source_digest(sources):
        canonical = sorted(json.dumps(source.to_dict(),sort_keys=True,separators=(',',':')) for source in set(sources))
        return hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def _member_digest(identities):
        return hashlib.sha256(json.dumps(sorted(identities),separators=(',',':')).encode()).hexdigest()

    def _seal(self, db, identity):
        # Only a writer that has already verified its parents may commit a new
        # proof. Startup/read/cleanup never reconstruct a lost proof as truth.
        group = db.execute('SELECT sources_json FROM voice_groups WHERE id=?',(identity,)).fetchone()
        members = [row[0] for row in db.execute('SELECT fingerprint_id FROM voice_group_members WHERE group_id=?',(identity,))]
        db.execute('INSERT OR REPLACE INTO voice_group_lineage VALUES (?,?,?)',
            (identity,self._source_digest(self.voices._sources(group['sources_json'])),self._member_digest(members)))

    def _request_complete(self, db, row):
        proof = db.execute('SELECT source_digest FROM voice_group_request_lineage WHERE actor=? AND request_id=?',
            (row['actor'],row['request_id'])).fetchone()
        try:
            return proof is not None and proof[0]==self._source_digest(self.voices._sources(row['sources_json']))
        except (ValueError,KeyError,TypeError):
            return False

    def _proof_complete(self, db, row):
        proof = db.execute('SELECT source_digest,member_digest FROM voice_group_lineage WHERE group_id=?',(row['id'],)).fetchone()
        try:
            members = [member[0] for member in db.execute('SELECT fingerprint_id FROM voice_group_members WHERE group_id=?',(row['id'],))]
            return (proof is not None and proof['source_digest']==self._source_digest(self.voices._sources(row['sources_json']))
                and proof['member_digest']==self._member_digest(members))
        except (ValueError,KeyError,TypeError):
            return False

    def _create(self, db, identities, sources, state, reason, *, score=None):
        sources = tuple(set(sources))
        if len(sources)>256 or sum(len(source.fact_dependencies) for source in sources)>256:
            raise ValueError('voice_group_source_capacity')
        if not identities or len(identities)>MAX_GROUP_MEMBERS:
            raise ValueError('voice_group_member_capacity')
        if db.execute('SELECT COUNT(*) FROM voice_groups').fetchone()[0]>=4096:
            raise RuntimeError('voice_group_capacity')
        identity = str(uuid4())
        encoded = json.dumps([source.to_dict() for source in sorted(sources,key=lambda source:source.capture_id)],sort_keys=True)
        db.execute('INSERT INTO voice_groups VALUES (?,?,?,?,?,?,?,?)',
            (identity,1,state,reason,score,self.ledger.clock(),min(source.expires_at for source in sources),encoded))
        db.executemany('INSERT INTO voice_group_members VALUES (?,?)',[(item,identity) for item in identities])
        self._seal(db,identity)
        self._changed()
        return identity

    def initialize(self, db: sqlite3.Connection) -> None:
        """Backfill old captures once as unknown; never invent acoustic evidence."""
        if db.execute("SELECT 1 FROM voice_group_schema WHERE version=1").fetchone():
            return
        for row in db.execute('SELECT id,sources_json,expires_at FROM voice_fingerprints').fetchall():
            if self.voices._live(row):
                self._create(db,(row['id'],),self.voices._sources(row['sources_json']),'unknown','unreviewed')
        db.execute('INSERT INTO voice_group_schema VALUES (1)')

    def _live(self, row):
        return self.voices._live(row)

    def _members(self, db, identity, *, vectors=False):
        if vectors:
            return db.execute('''SELECT v.* FROM voice_group_members m JOIN voice_fingerprints v
                ON v.id=m.fingerprint_id WHERE m.group_id=? ORDER BY v.id''',(identity,)).fetchall()
        return db.execute('''SELECT v.id,v.model_key,v.dimensions,v.duration_s,v.created_at,v.expires_at,v.sources_json
            FROM voice_group_members m JOIN voice_fingerprints v
            ON v.id=m.fingerprint_id WHERE m.group_id=? ORDER BY v.id''',(identity,)).fetchall()

    def add(self, db: sqlite3.Connection, identity: str, model_key: str, dimensions: int,
            raw: bytes, duration_s: float, sources: tuple[MemorySource,...]) -> str:
        """Compare all group members off-loop, never chaining two existing groups."""
        if duration_s<MIN_GROUP_DURATION:
            return self._create(db,(identity,),sources,'unknown','short_audio')
        vector = np.frombuffer(raw,dtype='<f4')
        candidates = []
        for group in db.execute("SELECT * FROM voice_groups WHERE state='suggested' ORDER BY id"):
            if not self._live(group):
                continue
            members = self._members(db,group['id'],vectors=True)
            if not members or len(members)>=MAX_GROUP_MEMBERS or any(not self.voices._live(row)
                    or row['model_key']!=model_key or row['dimensions']!=dimensions
                    or row['duration_s']<MIN_GROUP_DURATION for row in members):
                continue
            if self.metadata(db,members[0]['id']) is None:
                continue
            combined = set(sources) | set(self.voices._sources(group['sources_json']))
            if len(combined)>256 or sum(len(source.fact_dependencies) for source in combined)>256:
                continue
            score = min(float(np.dot(vector,np.frombuffer(row['embedding'],dtype='<f4'))) for row in members)
            if math.isfinite(score):
                candidates.append((max(-1.,min(1.,score)),group))
        candidates.sort(key=lambda item:(-item[0],item[1]['id']))
        if not candidates or candidates[0][0]<MIN_GROUP_COSINE:
            return self._create(db,(identity,),sources,'suggested','new_voice')
        if len(candidates)>1 and candidates[0][0]-candidates[1][0]<GROUP_MARGIN:
            return self._create(db,(identity,),sources,'unknown','ambiguous')
        score,group = candidates[0]
        combined = set(sources) | set(self.voices._sources(group['sources_json']))
        encoded = json.dumps([source.to_dict() for source in sorted(combined,key=lambda source:source.capture_id)],sort_keys=True)
        db.execute('INSERT INTO voice_group_members VALUES (?,?)',(identity,group['id']))
        db.execute('''UPDATE voice_groups SET revision=revision+1,reason='pairwise_similarity',similarity=?,
            expires_at=?,sources_json=? WHERE id=?''',
            (min(score,group['similarity']) if group['similarity'] is not None else score,
             min(source.expires_at for source in combined),encoded,group['id']))
        self._seal(db,group['id'])
        self._changed()
        return group['id']

    def metadata(self, db: sqlite3.Connection, identity: str, *, cache: dict | None = None) -> tuple[dict,tuple] | None:
        """Return current raw membership metadata plus its complete lineage."""
        row = db.execute('''SELECT g.* FROM voice_groups g JOIN voice_group_members m ON g.id=m.group_id
            WHERE m.fingerprint_id=?''',(identity,)).fetchone()
        if row is None or not self._live(row):
            return None
        if cache is not None and row['id'] in cache:
            return cache[row['id']]
        size = db.execute('SELECT COUNT(*) FROM voice_group_members WHERE group_id=?',(row['id'],)).fetchone()[0]
        members = self._members(db,row['id'])
        sources = self.voices._sources(row['sources_json'])
        proof = db.execute('SELECT source_digest,member_digest FROM voice_group_lineage WHERE group_id=?',(row['id'],)).fetchone()
        if (proof is None or proof['source_digest']!=self._source_digest(sources)
                or proof['member_digest']!=self._member_digest(member['id'] for member in members)
                or not 1<=size<=32 or len(members)!=size or row['state'] not in ('suggested','manual','unknown')
                or len({(member['model_key'],member['dimensions']) for member in members})!=1
                or any(not self.voices._live(member) or not set(self.voices._sources(member['sources_json']))<=set(sources)
                    for member in members)):
            if cache is not None:
                cache[row['id']] = None
            return None
        result = ({'group_id':row['id'],'group_revision':row['revision'],'group_state':row['state'],
            'group_reason':row['reason'],'group_size':size,'group_similarity':row['similarity'],
            'group_expires_at':row['expires_at']},sources)
        if cache is not None:
            cache[row['id']] = result
        return result

    def correct(self, identities: tuple[str,...], revisions: dict[str,int], action: str,
            request_id: str, actor: VoiceReviewer) -> bool:
        """Apply one idempotent merge, split or unknown edit against exact groups."""
        from src.users.voice_ledger import VoiceReviewer
        if type(actor) is not VoiceReviewer or not actor.subject:
            raise PermissionError('voice_review_denied')
        actor.require_current()
        if (type(identities) is not tuple or not 1<=len(identities)<=32
                or any(type(identity) is not str or len(identity)!=67 or not identity.startswith('vf_')
                    or any(char not in '0123456789abcdef' for char in identity[3:]) for identity in identities)
                or len(set(identities))!=len(identities)
                or action not in ('merge','split','unknown') or type(revisions) is not dict or not 1<=len(revisions)<=32
                or any(type(key) is not str or str(UUID(key))!=key or type(value) is not int or value<1 for key,value in revisions.items())
                or type(request_id) is not str or str(UUID(request_id))!=request_id):
            raise ValueError('invalid_voice_group_correction')
        digest = hashlib.sha256(json.dumps([sorted(identities),sorted(revisions.items()),action]).encode()).hexdigest()
        with self.ledger._transaction() as db:
            actor.require_current()
            prior = db.execute('SELECT * FROM voice_group_requests WHERE actor=? AND request_id=?',(actor.subject,request_id)).fetchone()
            if prior is not None:
                if prior['digest']!=digest:
                    raise ValueError('voice_group_request_conflict')
                if not self._request_complete(db,prior) or not self._live(prior):
                    raise LookupError('voice_group_unavailable')
                return True
            if db.execute('SELECT COUNT(*) FROM voice_group_requests').fetchone()[0]>=4096:
                raise RuntimeError('voice_group_request_capacity')
            rows = [db.execute('SELECT id,model_key,dimensions,sources_json,expires_at FROM voice_fingerprints WHERE id=?',(identity,)).fetchone() for identity in identities]
            if any(row is None or not self.voices._live(row) for row in rows):
                raise LookupError('voice_capture_unavailable')
            if len({(row['model_key'],row['dimensions']) for row in rows})!=1:
                raise ValueError('incompatible_voice_encoders')
            groups = {}
            for row in rows:
                group = db.execute('''SELECT g.* FROM voice_groups g JOIN voice_group_members m ON g.id=m.group_id
                    WHERE m.fingerprint_id=?''',(row['id'],)).fetchone()
                if group is None or not self._live(group):
                    raise LookupError('voice_group_unavailable')
                if self.metadata(db,row['id']) is None:
                    raise LookupError('voice_group_unavailable')
                groups[group['id']] = group
            if set(groups)!=set(revisions) or any(group['revision']!=revisions[key] for key,group in groups.items()):
                raise ValueError('voice_group_revision_conflict')
            sources = tuple({source for group in groups.values() for source in self.voices._sources(group['sources_json'])})
            if len(sources)>256 or sum(len(source.fact_dependencies) for source in sources)>256:
                raise ValueError('voice_group_source_capacity')
            members = {key:self._members(db,key) for key in groups}
            if any(not self.voices._live(row) for page in members.values() for row in page):
                raise LookupError('voice_group_unavailable')
            chosen = set(identities)
            if action=='merge':
                all_rows = [row for page in members.values() for row in page]
                if len(groups)<2 or len(all_rows)>32:
                    raise ValueError('voice_group_merge_capacity')
                if len({(row['model_key'],row['dimensions']) for row in all_rows})!=1:
                    raise ValueError('incompatible_voice_encoders')
                retained = min(groups)
                db.executemany('DELETE FROM voice_group_members WHERE group_id=?',[(key,) for key in groups])
                db.executemany('DELETE FROM voice_groups WHERE id=?',[(key,) for key in groups if key!=retained])
                encoded = json.dumps([source.to_dict() for source in sources],sort_keys=True)
                db.execute("UPDATE voice_groups SET revision=revision+1,state='manual',reason='manual',similarity=NULL,expires_at=?,sources_json=? WHERE id=?",
                    (min(source.expires_at for source in sources),encoded,retained))
                db.executemany('INSERT INTO voice_group_members VALUES (?,?)',[(row['id'],retained) for row in all_rows])
                self._changed()
            else:
                if action=='split' and (len(groups)!=1 or len(chosen)>=len(next(iter(members.values())))):
                    raise ValueError('voice_group_split_requires_remainder')
                db.executemany('DELETE FROM voice_group_members WHERE fingerprint_id=?',[(identity,) for identity in identities])
                for key,page in members.items():
                    remaining = [row['id'] for row in page if row['id'] not in chosen]
                    if remaining:
                        db.execute("UPDATE voice_groups SET revision=revision+1,state='manual',reason='manual',similarity=NULL WHERE id=?",(key,))
                    else:
                        db.execute('DELETE FROM voice_groups WHERE id=?',(key,))
                if action=='split':
                    self._create(db,identities,sources,'manual' if len(identities)>1 else 'unknown',
                        'manual' if len(identities)>1 else 'manual_unknown')
                else:
                    for identity in identities:
                        # Unknown is an explicit refusal, still derived from the
                        # reviewed group until that group's original deadline.
                        self._create(db,(identity,),sources,'unknown','manual_unknown')
                self._changed()
            actor.require_current()
            if not self.ledger.sources_available(sources) or not self.ledger._sources_live(db,sources):
                raise LookupError('voice_group_unavailable')
            for key in groups:
                if db.execute('SELECT 1 FROM voice_groups WHERE id=?',(key,)).fetchone():
                    self._seal(db,key)
            db.execute('INSERT INTO voice_group_requests VALUES (?,?,?,?,?)',
                (actor.subject,request_id,digest,min(source.expires_at for source in sources),
                 json.dumps([source.to_dict() for source in sources],sort_keys=True)))
            db.execute('INSERT INTO voice_group_request_lineage VALUES (?,?,?)',
                (actor.subject,request_id,self._source_digest(sources)))
            actor.require_current()
            if not self.ledger.sources_available(sources):
                raise LookupError('voice_group_unavailable')
            return True

    def prune(self, db: sqlite3.Connection) -> None:
        """Retire a derived group at its earliest source; raw samples keep their TTL."""
        self.initialize(db)
        for row in db.execute('SELECT * FROM voice_groups').fetchall():
            sources = self.voices._sources(row['sources_json'])
            if self.ledger.clock()<min(row['expires_at'],*(source.expires_at for source in sources)) and not self.ledger.sources_committed_unavailable(sources):
                continue
            survivors = self._members(db,row['id'])
            db.execute('DELETE FROM voice_groups WHERE id=?',(row['id'],))
            self._changed()
            for member in survivors:
                if self.voices._live(member):
                    self._create(db,(member['id'],),self.voices._sources(member['sources_json']),'unknown','prior_group_expired')
        for row in db.execute('SELECT * FROM voice_group_requests').fetchall():
            sources = self.voices._sources(row['sources_json'])
            if self.ledger.clock()>=min(row['expires_at'],*(source.expires_at for source in sources)) or self.ledger.sources_committed_unavailable(sources):
                db.execute('DELETE FROM voice_group_requests WHERE actor=? AND request_id=?',(row['actor'],row['request_id']))
