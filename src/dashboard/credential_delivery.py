"""Retained delivery of credential metadata and one-time issuance secrets."""

from dataclasses import dataclass
import asyncio
import json

from starlette.responses import Response

from src.core.owned_work import OwnedBlockingWork
from src.dashboard.authority import AuthorityRejected, DashboardAuthority, DashboardLease
from src.dashboard.credential_journal import CredentialCommand, CredentialCommandResult
from src.dashboard.native_work import run_admin_sequence
from src.dashboard.rest_delivery import RestDeliveryOwner, RestOperation, RestResult
from src.users.voice_evaluation import VoiceEvaluationReader

AUTH = frozenset({'auth:manage'})


@dataclass(frozen=True)
class CredentialCatalog:
    """Detached request for a bounded page of safe metadata."""

    after: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class CredentialAttempt:
    """Detached request for a terminal receipt, without a secret."""

    request_id: str


class _CredentialGuard:
    def __init__(self, issuer: DashboardLease, owner: RestDeliveryOwner):
        self._issuer,self._owner = issuer,owner
        self._issued = None
        self._binding_started = self._binding_failed = False

    def bind(self, issued: DashboardLease, revision: int) -> None:
        repeated = self._binding_started
        self._binding_started = True
        if (repeated or type(issued) is not DashboardLease
                or issued._authority is not self._issuer._authority
                or issued.principal.revision != revision or not issued.current()):
            self._binding_failed = True
            raise PermissionError('credential_delivery_retired')
        self._issued = issued

    def __call__(self) -> bool:
        return (not self._binding_failed and not self._owner.retention._closed
            and self._issuer.current(AUTH) and (self._issued is None or self._issued.current()))


async def _error(send, status: int) -> None:
    await send({'type':'http.response.start','status':status,'headers':[
        (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
    await send({'type':'http.response.body','body':b'{"detail":"credential_request_unavailable"}'})


async def _deliver(owner, authority, voices, issuer, request, guard, operation, send) -> RestResult:
    result = payload = body = issued = failure_send = None
    started = False
    try:
        if not guard() or owner.retention._closed:
            raise PermissionError('credential_delivery_retired')
        if type(request) is CredentialCommand:
            async def prepare(actor, grant):
                if (voices is None or type(voices.evaluation) is not VoiceEvaluationReader
                        or voices.evaluation.ledger is not owner.retention.policy
                        or voices.privacy.ledger is not owner.retention.policy):
                    raise RuntimeError('credential_scope_unavailable')
                return await voices.acall(voices.evaluation.prepare_grant,actor,grant.selections,grant.expires_at)
            result = await authority.commands.execute(issuer,request,prepare_scope=prepare)
            if type(result) is not CredentialCommandResult:
                raise ValueError('invalid_credential_command_result')
            payload = {'receipt':result.receipt.public(),'applied':result.applied}
            if result.token is not None:
                issued = authority.resolve(result.token)
                guard.bind(issued,result.receipt.result_revision)
                payload['token'] = result.token
        elif type(request) is CredentialCatalog:
            payload = authority.commands.catalog(issuer,after=request.after,limit=request.limit)
        elif type(request) is CredentialAttempt:
            payload = authority.commands.attempt(issuer,request.request_id)
        else:
            raise ValueError('invalid_credential_request')
        if not guard() or owner.retention._closed:
            raise PermissionError('credential_delivery_retired')
        body = json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        payload = result = issued = None
        started = True
        await send({'type':'http.response.start','status':200,'headers':[
            (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
        if not guard() or owner.retention._closed:
            raise PermissionError('credential_delivery_retired')
        await send({'type':'http.response.body','body':body})
        return RestResult.COMPLETE
    except (Exception,asyncio.CancelledError) as error:
        status = (403 if isinstance(error,PermissionError) else
            409 if isinstance(error,AuthorityRejected) and str(error)=='credential_attempt_conflict' else 503)
        if not started:
            failure_send = send
    finally:
        result = payload = body = issued = issuer = request = send = voices = None
    if failure_send is not None:
        try:
            await _error(failure_send,status)
        except (Exception,asyncio.CancelledError):
            pass
        finally:
            failure_send = None
    return RestResult.FAILED


class CredentialResponse(Response):
    """Deliver one native result while retaining the original authority guards."""

    def __init__(self, owner: RestDeliveryOwner, work: OwnedBlockingWork,
                 authority: DashboardAuthority, issuer: DashboardLease, request,
                 *, voices=None) -> None:
        super().__init__(content=b'')
        self.owner,self.work,self.authority,self.issuer = owner,work,authority,issuer
        self.request,self.voices = request,voices

    async def __call__(self, scope, receive, send) -> None:
        """The retained task returns only delivery status, never the secret."""
        del receive
        operation = scope.get('kza.rest_operation')
        guard = _CredentialGuard(self.issuer,self.owner)
        try:
            if type(operation) is not RestOperation:
                await _error(send,503)
                return
            operation.sources = ()
            operation.channel = 'credentials'
            operation.private_envelope = False
            operation.guard = guard
            scope['kza.dashboard_delivery_guard'] = guard
            await run_admin_sequence(self.work,guard,_deliver,self.owner,self.authority,self.voices,
                self.issuer,self.request,guard,operation,send)
        except Exception:
            try:
                await _error(send,503)
            except (Exception,asyncio.CancelledError):
                pass
        finally:
            self.owner = self.work = self.authority = self.issuer = self.request = self.voices = None
            scope = send = operation = guard = None
