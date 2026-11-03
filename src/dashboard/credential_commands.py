"""Credential-only commands using Authority's existing writer and ownership."""

import asyncio
from contextvars import Context
from dataclasses import dataclass
import json
import secrets
import sqlite3

from src.dashboard import authority as authority_types
from src.dashboard.authority import AuthorityRejected, DashboardLease, DashboardRole, _hash
from src.dashboard.credential_journal import (
    CredentialAction, CredentialCommand, CredentialCommandResult, CredentialOutcome,
    CredentialReceipt, MAX_CREDENTIAL_ATTEMPTS, canonical_attempt, insert_receipt,
)
from src.dashboard.evaluation_grants import PreparedEvaluationScope

AUTH = frozenset({'auth:manage'})


class _SecretSlot:
    """Only the initial observer can consume this non-persistent value."""

    def __init__(self):
        self._value = None
        self._closed = False

    def publish(self,value):
        if not self._closed:
            self._value = value

    def take(self):
        value,self._value = self._value,None
        return value

    def close(self):
        self._closed = True
        self._value = None


@dataclass(repr=False)
class _Admission:
    actor: DashboardLease
    command: CredentialCommand
    actor_key: str
    slot: _SecretSlot
    prepared_scope: object | None = None
    prepare_scope: object | None = None
    task: asyncio.Task | None = None


@dataclass(frozen=True,repr=False)
class _CommandPlan:
    command: CredentialCommand
    actor_key: str
    digest: str | None
    revision: int | None
    subject: str | None
    samples: tuple
    reason: str | None


@dataclass(frozen=True,repr=False)
class _Effect:
    receipt: CredentialReceipt
    applied: bool
    savepoint: bool = False


class NativeCredentialCommands:
    def __init__(self,authority):
        self.authority = authority
        self._inflight = {}

    def _actor(self,actor):
        if (type(actor) is not DashboardLease or actor._authority is not self.authority
                or not actor.current(AUTH)):
            raise PermissionError('dashboard_grant_denied')
        reader = actor.reader_binding(AUTH)
        if reader is None:
            raise PermissionError('dashboard_grant_denied')
        return reader.namespace

    async def execute(self,actor,command,*,prepared_scope=None,prepare_scope=None):
        key = self._actor(actor)
        if type(command) is not CredentialCommand:
            raise ValueError('invalid_credential_command')
        if prepare_scope is not None and not callable(prepare_scope):
            raise ValueError('invalid_credential_scope_provider')
        admission = self._inflight.get(command.request_id)
        first = admission is None
        if first:
            admission = _Admission(actor,command,key,_SecretSlot(),prepared_scope,prepare_scope)
            task = self.authority._admit('credential_command_run',(admission,),lambda:actor.current(AUTH))
            admission.task = task
            self._inflight[command.request_id] = admission
            def finished(done):
                if self._inflight.get(command.request_id) is admission:
                    self._inflight.pop(command.request_id,None)
            task.add_done_callback(finished,context=Context())
        elif admission.actor_key!=key or admission.command.digest!=command.digest:
            raise AuthorityRejected('credential_attempt_conflict')
        try:
            receipt,applied = await asyncio.shield(admission.task)
            return CredentialCommandResult(receipt,applied if first else False,
                admission.slot.take() if first and applied else None)
        finally:
            if first:
                admission.slot.close()
            actor = prepared_scope = prepare_scope = None

    def attempt(self,actor,request_id):
        self._actor(actor)
        canonical_attempt(request_id)
        with self.authority._state_lock:
            receipt = self.authority._attempts.get(request_id)
            if receipt is not None:
                value = {'state':'terminal','receipt':receipt.public()}
            elif request_id in self._inflight:
                value = {'state':'pending','receipt':None}
            else:
                value = {'state':'unconfirmed','receipt':None}
        self._actor(actor)
        return value

    def catalog(self,actor,*,after=None,limit=50):
        self._actor(actor)
        if type(limit) is not int or not 1<=limit<=50 or after is not None and (type(after) is not str or not 1<=len(after)<=256):
            raise ValueError('invalid_credential_page')
        authority = self.authority
        with authority._state_lock:
            now = authority._now()
            entries = sorted((entry for entry in authority._credentials.values()
                if after is None or entry.identity>after),key=lambda entry:entry.identity)[:limit+1]
            rows = []
            for entry in entries[:limit]:
                active = (entry.expires_at>now and authority._account_current(entry.user_id)
                    and entry.identity not in (authority._revoking|authority._credential_revoking)
                    and entry.revision>=authority._revision_admission.get(entry.identity,0))
                rows.append({'id':entry.identity,'subject':entry.subject,'role':entry.role.value,
                    'user_id':entry.user_id,'issued_at':entry.issued_at,'expires_at':entry.expires_at,
                    'revision':entry.revision,'active':active,
                    'pending':bool(authority._credential_reservations.get(entry.identity)),
                    'samples':[{'fingerprint_id':sample.fingerprint_id,
                        'modalities':sorted(mode.value for mode in sample.modalities),
                        'expires_at':sample.expires_at} for sample in entry.samples]})
            value = {'rows':rows,'observed_at':now,
                'next_cursor':entries[limit-1].identity if len(entries)>limit else None}
        self._actor(actor)
        return value

    async def _run_locked(self,admission):
        authority,command,actor = self.authority,admission.command,admission.actor
        token = scope = samples = guard = plan = None
        try:
            actor_live = actor.current(AUTH)
            with authority._state_lock:
                receipt = authority._attempts.get(command.request_id)
            if receipt is not None:
                if not actor_live:
                    raise PermissionError('dashboard_grant_denied')
                if receipt.actor_key!=admission.actor_key or receipt.command_digest!=command.digest:
                    raise AuthorityRejected('credential_attempt_conflict')
                return receipt,False
            if len(authority._attempts)>=MAX_CREDENTIAL_ATTEMPTS:
                raise AuthorityRejected('credential_attempt_capacity')
            reason = None if actor_live else 'actor_retired'
            subject = command.grant.subject if command.grant is not None else None
            revision = 1 if command.action is CredentialAction.ISSUE else None
            def current():
                return actor.current(AUTH)
            if reason is None and command.action is not CredentialAction.ISSUE:
                with authority._state_lock:
                    original = next((value for value in authority._credentials.values() if value.identity==command.target),None)
                    if original is None:
                        reason = 'target_missing'
                    elif authority._credential_reservations.get(command.target):
                        reason = 'writer_busy'
                    elif original.revision!=command.expected_revision:
                        reason = 'revision_conflict'
                    elif command.target in (authority._revoking|authority._credential_revoking) or not authority._account_current(original.user_id):
                        reason = 'account_unavailable'
                    else:
                        current = authority._revocation_guard(actor,command.target)
                        subject = original.subject
                        revision = (max(original.revision,authority._revision_admission.get(command.target,0))+1
                            if command.action is CredentialAction.REVISE else original.revision)
                        reservation = authority._reserve_credential_mutation_locked(command.target,
                            'revoke' if command.action is CredentialAction.REVOKE else 'prepare',cas=True)
                        authority._task_mutations[asyncio.current_task()] = reservation
            samples = ()
            def scope_current():
                return True
            if reason is None and command.grant is not None:
                grant = command.grant
                if grant.role is DashboardRole.EVALUATOR:
                    scope = admission.prepared_scope
                    if scope is None and admission.prepare_scope is not None:
                        try:
                            scope = await admission.prepare_scope(actor,grant)
                        except (LookupError,ValueError,PermissionError,RuntimeError,OSError,sqlite3.Error):
                            reason = 'scope_unavailable'
                    if reason is None:
                        if (type(scope) is not PreparedEvaluationScope
                                or tuple(record.selection for record in scope.records)!=grant.selections):
                            reason = 'scope_unavailable'
                        else:
                            samples = scope.samples
                if reason is None:
                    try:
                        authority._validate_grant(subject,grant.role,grant.expires_at,grant.user_id,samples)
                        scope_current = authority._prepared_evaluation_guard(actor,grant.role,grant.expires_at,
                            grant.user_id,samples,scope)
                        if not scope_current():
                            reason = 'scope_unavailable'
                    except (ValueError,PermissionError):
                        reason = 'grant_invalid'
            if reason is None and current() is not True:
                reason = 'actor_retired'
            if reason is None and command.action is CredentialAction.REVISE:
                with authority._state_lock:
                    reservation = authority._task_mutations[asyncio.current_task()]
                    authority._credential_reservations[reservation[0]][reservation[1]] = 'revise',revision,True
                    authority._recompute_credential_admission_locked(command.target)
            if reason is None and command.action is CredentialAction.ISSUE:
                token = 'kza_'+secrets.token_urlsafe(32)
            def guard():
                return current() and (command.grant is None or
                    authority._now()<command.grant.expires_at
                    and authority._account_current(command.grant.user_id) and scope_current())
            plan = _CommandPlan(command,admission.actor_key,None if token is None else _hash(token),revision,subject,samples,reason)
            state = await authority._run_transaction('credential_command',(plan,),guard)
            effect = authority._publish_state(state)
            if effect.applied and command.action is CredentialAction.ISSUE:
                admission.slot.publish(token)
            return effect.receipt,effect.applied
        finally:
            token = scope = samples = guard = plan = None
            admission.prepared_scope = admission.prepare_scope = None

    def _apply_sql(self,db,plan,guard,floor):
        command = plan.command
        existing = db.execute('SELECT request_id FROM credential_attempts WHERE request_id=?',(command.request_id,)).fetchone()
        if existing is not None:
            raise AuthorityRejected('credential_attempt_conflict')
        if db.execute('SELECT COUNT(*) FROM credential_attempts').fetchone()[0]>=MAX_CREDENTIAL_ATTEMPTS:
            raise AuthorityRejected('credential_attempt_capacity')
        db.execute('SAVEPOINT credential_effect')
        reason = plan.reason
        if reason is None and guard() is not True:
            reason = 'actor_retired'
        if reason is None and command.action is CredentialAction.ISSUE:
            if db.execute('SELECT COUNT(*) FROM credentials').fetchone()[0]>=authority_types.MAX_CREDENTIALS:
                reason = 'capacity'
            elif (db.execute('SELECT 1 FROM credentials WHERE id=?',(command.target,)).fetchone()
                    or db.execute("SELECT 1 FROM credential_attempts WHERE credential_id=? AND kind='revoke' AND outcome='committed'",(command.target,)).fetchone()):
                reason = 'revision_conflict'
            else:
                grant = command.grant
                db.execute('INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)',(
                    command.target,plan.digest,plan.subject,grant.role.value,grant.user_id,floor,grant.expires_at,
                    1,json.dumps([sample.to_dict() for sample in plan.samples],separators=(',',':'))))
        elif reason is None:
            original = db.execute('SELECT revision FROM credentials WHERE id=?',(command.target,)).fetchone()
            if original is None:
                reason = 'target_missing'
            elif original[0]!=command.expected_revision:
                reason = 'revision_conflict'
            elif command.action is CredentialAction.REVISE:
                grant = command.grant
                db.execute('UPDATE credentials SET role=?,user_id=?,issued_at=?,expires_at=?,revision=?,samples=? WHERE id=? AND revision=?',
                    (grant.role.value,grant.user_id,floor,grant.expires_at,plan.revision,
                    json.dumps([sample.to_dict() for sample in plan.samples],separators=(',',':')),
                    command.target,command.expected_revision))
                db.execute('DELETE FROM sessions WHERE credential=?',(command.target,))
            else:
                db.execute('DELETE FROM credentials WHERE id=? AND revision=?',(command.target,command.expected_revision))
                db.execute('DELETE FROM sessions WHERE credential=?',(command.target,))
        receipt = CredentialReceipt(command.request_id,plan.actor_key,command.digest,command.action,
            command.target,command.expected_revision,CredentialOutcome.COMMITTED if reason is None else CredentialOutcome.NO_EFFECT,
            plan.revision if reason is None else None,'applied' if reason is None else reason,floor)
        insert_receipt(db,receipt)
        return _Effect(receipt,reason is None,True)

    @staticmethod
    def _completed_sql(db,effect,now):
        receipt = effect.receipt
        value = CredentialReceipt(receipt.request_id,receipt.actor_key,receipt.command_digest,receipt.action,
            receipt.credential_id,receipt.expected_revision,receipt.outcome,receipt.result_revision,receipt.reason,now)
        db.execute('UPDATE credential_attempts SET completed_at=? WHERE request_id=?',(now,receipt.request_id))
        return _Effect(value,effect.applied)

    @staticmethod
    def _retired_sql(db,effect):
        db.execute('ROLLBACK TO credential_effect')
        db.execute('RELEASE credential_effect')
        receipt = effect.receipt
        value = CredentialReceipt(receipt.request_id,receipt.actor_key,receipt.command_digest,receipt.action,
            receipt.credential_id,receipt.expected_revision,CredentialOutcome.NO_EFFECT,None,'actor_retired',receipt.completed_at)
        insert_receipt(db,value)
        return _Effect(value,False)
