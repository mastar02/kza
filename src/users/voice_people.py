"""Manual profile-person metadata; it never participates in acoustic identity."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
import sqlite3
from typing import TYPE_CHECKING
from uuid import UUID
from weakref import ref

from src.dashboard.live_retention_types import VoicePersonFence

if TYPE_CHECKING:
    from src.users.voice_ledger import VoiceLedger, VoiceReviewer


@dataclass(frozen=True, repr=False)
class VoicePersonReceipt:
    """A committed metadata revision, without a person or account payload."""

    profile_id: str
    link_revision: int


def _identity(value: str) -> None:
    if (type(value) is not str or not 1 <= len(value) <= 128
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('invalid_voice_person_identity')


def _profile(value: str) -> None:
    try:
        valid = type(value) is str and str(UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('invalid_voice_person_profile')


class VoicePersonLinks:
    """Keep native writes and a conservative nonblocking account-removal proof."""

    def __init__(self, voices: VoiceLedger):
        self.voices = voices
        self.ledger = voices.ledger
        self._proof = self.ledger._voice_person_proof

    def user_links_cleared(self, identities: frozenset[str]) -> bool:
        """Read one immutable proof; never acquire ledger or account locks here."""
        snapshot = self._proof.snapshot
        if type(identities) is not frozenset or len(identities) > 256:
            return False
        try:
            for identity in identities:
                _identity(identity)
        except ValueError:
            return False
        return self.ledger.authority_available and snapshot.available and not identities & snapshot.users

    def _begin(self, *, withdraw: bool = True) -> None:
        self._proof.begin(withdraw=withdraw)

    def _finish(self, committed: bool, users: frozenset[str] | None) -> None:
        self._proof.finish(committed, users)

    @staticmethod
    def _metadata(row: sqlite3.Row | None) -> dict:
        if row is None:
            return {'user_id': None, 'revision': 0}
        _profile(row['profile_id'])
        if row['user_id'] is not None:
            _identity(row['user_id'])
        if (type(row['revision']) is not int or row['revision'] < 1
                or type(row['updated_at']) not in (int, float) or not math.isfinite(row['updated_at'])
                or row['updated_at'] < 0 or type(row['updated_by']) is not str
                or not 1 <= len(row['updated_by']) <= 256
                or any(ord(char) < 32 or ord(char) == 127 for char in row['updated_by'])):
            raise ValueError('invalid_voice_person_metadata')
        return {'user_id': row['user_id'], 'revision': row['revision']}

    def _users(self, db: sqlite3.Connection) -> frozenset[str]:
        rows = db.execute('SELECT * FROM voice_profile_people WHERE user_id IS NOT NULL LIMIT 257').fetchall()
        if len(rows) > 256:
            raise ValueError('voice_person_link_capacity')
        return frozenset(self._metadata(row)['user_id'] for row in rows)

    @contextmanager
    def _transaction(self, guard: Callable[[], None] | None = None, *, withdraw: bool = True) -> Iterator[sqlite3.Connection]:
        self._begin(withdraw=withdraw)
        # Keep publication ordered with the actual commits. A later writer
        # must never publish and then be overwritten by an older SQL snapshot.
        with self.ledger._lock:
            committed, users = False, None
            try:
                with self.ledger._transaction(guard=guard) as db:
                    yield db
                    users = self._users(db)
                committed = True
            finally:
                self._finish(committed, users)

    def reconcile(self) -> None:
        """Recover the proof from actual SQL, including an uncertain earlier commit."""
        # Read-only maintenance cannot introduce references. An already valid
        # proof stays usable, while every actual writer still withdraws it.
        with self._transaction(withdraw=False):
            pass

    @staticmethod
    def _actor(actor: VoiceReviewer) -> VoiceReviewer:
        from src.users.voice_ledger import VoiceReviewer

        if type(actor) is not VoiceReviewer:
            raise PermissionError('voice_person_review_denied')
        return actor

    def set_person(self, profile_id: str, user_id: str | None, expected_profile_revision: int,
                   expected_link_revision: int, actor: VoiceReviewer) -> VoicePersonReceipt:
        """CAS only this relationship; preserve the acoustic profile and its lifetime."""
        _profile(profile_id)
        if user_id is not None:
            _identity(user_id)
        if (type(expected_profile_revision) is not int or expected_profile_revision < 1
                or type(expected_link_revision) is not int or expected_link_revision < 0):
            raise ValueError('invalid_voice_person_revision')
        actor = self._actor(actor)
        targets = frozenset({user_id}) if user_id is not None else frozenset()
        profile = None
        def current() -> None:
            actor.require_people(targets)
            if profile is not None and not self.voices._live(profile, confirmed=True):
                raise LookupError('voice_profile_unavailable')
        current()
        with self._transaction(current) as db:
            profile = db.execute('''SELECT id,revision,sources_json,confirmed_at,confirmed_by,deleted_at
                FROM voice_profiles WHERE id=?''', (profile_id,)).fetchone()
            if profile is None or not self.voices._live(profile, confirmed=True):
                raise LookupError('voice_profile_unavailable')
            if profile['revision'] != expected_profile_revision:
                raise ValueError('voice_person_profile_conflict')
            previous = self._metadata(db.execute('SELECT * FROM voice_profile_people WHERE profile_id=?', (profile_id,)).fetchone())
            if previous['revision'] != expected_link_revision:
                raise ValueError('voice_person_link_conflict')
            revision = previous['revision']
            if previous['user_id'] != user_id:
                revision += 1
                self.ledger._voice_projection_revision += 1
                db.execute('''INSERT INTO voice_profile_people VALUES (?,?,?,?,?)
                    ON CONFLICT(profile_id) DO UPDATE SET user_id=excluded.user_id,revision=excluded.revision,
                    updated_at=excluded.updated_at,updated_by=excluded.updated_by''',
                    (profile_id, user_id, revision, self.ledger.clock(), actor.subject))
        return VoicePersonReceipt(profile_id, revision)

    def clear_user_links(self, identities: frozenset[str], retired: Callable[[frozenset[str]], bool]) -> int:
        """Reconcile a durable account retirement; this never creates a relationship."""
        if type(identities) is not frozenset or not 1 <= len(identities) <= 256:
            raise ValueError('invalid_voice_person_targets')
        if not callable(retired):
            raise PermissionError('voice_person_retirement_required')
        for identity in identities:
            _identity(identity)
        def current() -> None:
            if retired(identities) is not True:
                raise PermissionError('voice_person_retirement_required')
        current()
        with self._transaction(current) as db:
            placeholders = ','.join('?' for _ in identities)
            changed = db.execute(f'''UPDATE voice_profile_people SET user_id=NULL,revision=revision+1,
                updated_at=?,updated_by=? WHERE user_id IN ({placeholders})''',
                (self.ledger.clock(), 'system:retired-account', *sorted(identities))).rowcount
            if changed:
                self.ledger._voice_projection_revision += 1
        return changed

    def unlink_profile(self, db: sqlite3.Connection, profile_id: str) -> int:
        """Only remove metadata; an older proof may conservatively retain the former ID."""
        return db.execute('''UPDATE voice_profile_people SET user_id=NULL,revision=revision+1,
            updated_at=?,updated_by=? WHERE profile_id=? AND user_id IS NOT NULL''',
            (self.ledger.clock(), 'system:retired-profile', profile_id)).rowcount

    def project_rows(self, db: sqlite3.Connection, profile_ids: tuple[str, ...], actor: VoiceReviewer,
                     person_id: str | None = None) -> tuple[list[dict], VoicePersonFence | None]:
        """Resolve IDs only, under the caller's transaction and one canonical batch fence."""
        actor = self._actor(actor)
        actor.require_people()
        if type(profile_ids) is not tuple or len(profile_ids) > 50 or len(set(profile_ids)) != len(profile_ids):
            raise ValueError('invalid_voice_person_projection')
        rows = []
        for identity in profile_ids:
            _profile(identity)
            rows.append(self._metadata(db.execute('SELECT * FROM voice_profile_people WHERE profile_id=?', (identity,)).fetchone()))
        identities = frozenset(row['user_id'] for row in rows if row['user_id'] is not None)
        if person_id is not None:
            _identity(person_id)
            if any(row['user_id'] != person_id for row in rows):
                raise LookupError('voice_person_changed')
            identities = identities | frozenset({person_id})
        authority = actor.require_people(identities)
        fence = VoicePersonFence(ref(authority), tuple(sorted(identities))) if identities else None
        return rows, fence
