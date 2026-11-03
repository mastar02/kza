"""Closed HTTP inputs for native credential administration."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request

from src.core.owned_work import OwnedBlockingWork
from src.dashboard.auth import current_lease, dashboard_authority, require_dashboard_session
from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.credential_delivery import AUTH, CredentialAttempt, CredentialCatalog, CredentialResponse
from src.dashboard.credential_journal import (
    CredentialAction, CredentialCommand, CredentialGrantSpec, canonical_attempt,
)
from src.dashboard.evaluation_grants import EvaluationGrantSelection
from src.dashboard.rest_delivery import RestDeliveryOwner, RestOperation

MAX_BODY = 65536


def _object(pairs):
    value = {}
    for key,item in pairs:
        if key in value:
            raise ValueError('duplicate_credential_field')
        value[key] = item
    return value


def _constant(value):
    raise ValueError('invalid_credential_number')


def _text(value):
    if type(value) is not str:
        raise ValueError('invalid_credential_text')
    value.encode('utf-8')
    return value


def _fields(value, required, optional=frozenset()):
    if type(value) is not dict or not required<=value.keys() or value.keys()-required-optional:
        raise ValueError('invalid_credential_fields')


def _command(value, action, identity=None):
    required = {'request_id','expected_revision'} if action is CredentialAction.REVOKE else (
        {'request_id','subject','role','expires_at'} if action is CredentialAction.ISSUE else
        {'request_id','expected_revision','role','expires_at'})
    _fields(value,required,{'user_id','selections'} if action is not CredentialAction.REVOKE else set())
    grant = None
    if action is not CredentialAction.REVOKE:
        selections = value.get('selections',[])
        if type(selections) is not list or len(selections)>64:
            raise ValueError('invalid_credential_selections')
        selected = []
        for item in selections:
            _fields(item,{'fingerprint_id','modalities','expires_at'})
            modes = item['modalities']
            if (type(modes) is not list or not 1<=len(modes)<=2
                    or any(type(mode) is not str for mode in modes) or len(set(modes))!=len(modes)):
                raise ValueError('invalid_credential_modalities')
            selected.append(EvaluationGrantSelection(item['fingerprint_id'],
                frozenset(EvaluationModality(mode) for mode in modes),item['expires_at']))
        user = value.get('user_id')
        subject = _text(value['subject']) if action is CredentialAction.ISSUE else None
        grant = CredentialGrantSpec(DashboardRole(value['role']),value['expires_at'],
            None if user is None else _text(user),tuple(selected),subject)
    return CredentialCommand(_text(value['request_id']),action,
        None if identity is None else _text(identity),value.get('expected_revision'),grant)


async def _body(request):
    raw = bytearray()
    chunk = value = None
    failed = False
    try:
        async for chunk in request.stream():
            if len(raw)+len(chunk)>MAX_BODY:
                raise ValueError('credential_body_capacity')
            raw.extend(chunk)
        value = json.loads(raw,object_pairs_hook=_object,parse_constant=_constant)
    except (ValueError,TypeError,RecursionError,OverflowError):
        failed = True
    finally:
        raw.clear()
        chunk = None
    if failed:
        raise HTTPException(422,detail='invalid_credential_request')
    return value


def _context(request, work):
    authority = dashboard_authority(request,required=True)
    issuer = current_lease(request)
    owner = getattr(request.app.state,'rest_delivery_owner',None)
    operation = request.scope.get('kza.rest_operation')
    if issuer is None or issuer._authority is not authority or not issuer.current(AUTH):
        raise HTTPException(403,detail='dashboard_permission_denied')
    if (not isinstance(owner,RestDeliveryOwner) or type(operation) is not RestOperation
            or not isinstance(work,OwnedBlockingWork)):
        raise HTTPException(503,detail='credential_delivery_unavailable')
    operation.sources = ()
    return owner,work,authority,issuer


def register_credential_routes(app, *, work: OwnedBlockingWork, voices=None) -> None:
    """Register safe metadata and durable commands with required real authority."""
    router = APIRouter(prefix='/api/admin/auth',dependencies=[Depends(require_dashboard_session)])

    @router.get('/credentials')
    async def catalog(request: Request):
        context = _context(request,work)
        value = None
        invalid = False
        try:
            query = request.query_params
            if set(query)-{'after','limit'} or any(len(query.getlist(key))!=1 for key in query):
                raise ValueError('invalid_credential_page')
            limit = query.get('limit','50')
            if not limit.isascii() or not limit.isdigit() or not 1<=int(limit)<=50:
                raise ValueError('invalid_credential_page')
            after = query.get('after')
            if after is not None and not 1<=len(_text(after))<=256:
                raise ValueError('invalid_credential_page')
            value = CredentialCatalog(after,int(limit))
        except (ValueError,TypeError,OverflowError):
            invalid = True
        if invalid:
            raise HTTPException(422,detail='invalid_credential_request')
        return CredentialResponse(*context,value,voices=voices)

    @router.get('/credential-attempts/{request_id}')
    async def attempt(request: Request,request_id: str):
        context = _context(request,work)
        invalid = False
        try:
            canonical_attempt(request_id)
            if request.query_params:
                raise ValueError('invalid_credential_query')
        except (ValueError,TypeError):
            invalid = True
        if invalid:
            raise HTTPException(422,detail='invalid_credential_request')
        return CredentialResponse(*context,CredentialAttempt(request_id),voices=voices)

    async def mutation(request,action,identity=None):
        context = _context(request,work)
        if request.query_params:
            raise HTTPException(422,detail='invalid_credential_request')
        value = await _body(request)
        command = None
        invalid = False
        try:
            command = _command(value,action,identity)
        except (ValueError,TypeError,OverflowError,RecursionError):
            invalid = True
        finally:
            value = None
        if invalid:
            raise HTTPException(422,detail='invalid_credential_request')
        return CredentialResponse(*context,command,voices=voices)

    @router.post('/credentials')
    async def issue(request: Request):
        return await mutation(request,CredentialAction.ISSUE)

    @router.put('/credentials/{credential_id}')
    async def revise(request: Request,credential_id: str):
        return await mutation(request,CredentialAction.REVISE,credential_id)

    @router.delete('/credentials/{credential_id}')
    async def revoke(request: Request,credential_id: str):
        return await mutation(request,CredentialAction.REVOKE,credential_id)

    app.include_router(router)
