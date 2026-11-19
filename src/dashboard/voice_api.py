"""Explicit dashboard-only voice review; vectors never cross the HTTP boundary."""

import json
import math
import re
import sqlite3
from typing import Literal

from fastapi import APIRouter,Depends,HTTPException,Query,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field,ValidationError

from src.dashboard.auth import (
    DashboardPrincipal, current_lease, require_dashboard_session, session_guard, session_reader,
    voice_person_scope_requested, web_session_requested,
)
from src.dashboard.authority import DashboardAuthority
from src.users.voice_ledger import (
    QUERY_CLOCK_SKEW_S, QUERY_TEXT_MAX_CHARS, QUERY_TEXT_MAX_TERMS, VoiceCaptureFilter, VoiceReviewer,
    capture_query_position, normalize_voice_text,
)
from src.users.voice_people import VoicePersonReceipt
from src.dashboard.voice_delivery import VoiceResponse

QUERY_BODY_MAX_BYTES = 4096
VOICE_CAPTURE_ID = re.compile(r'^vf_[0-9a-f]{64}$')
INVALID_QUERY = 'invalid_voice_query'


class ConfirmVoiceBody(BaseModel):
    """A manual centroid review contains no account, permission or editable TTL."""

    model_config = ConfigDict(extra='forbid',strict=True)
    fingerprint_ids: list[str] = Field(min_length=3,max_length=32)
    name: str = Field(min_length=1,max_length=64)
    request_id: str = Field(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


class DeleteVoiceBody(BaseModel):
    """An exact current revision prevents stale review deletion."""

    model_config = ConfigDict(extra='forbid',strict=True)
    expected_revision: int = Field(ge=1)


class SetVoicePersonBody(BaseModel):
    """Compare both metadata revisions; the body cannot choose roles or consent."""

    model_config = ConfigDict(extra='forbid',strict=True)
    user_id: str | None = Field(max_length=128)
    expected_profile_revision: int = Field(ge=1)
    expected_link_revision: int = Field(ge=0)


class CorrectVoiceGroupsBody(BaseModel):
    """Exact group revisions make a human merge/split atomic and retryable."""

    model_config = ConfigDict(extra='forbid',strict=True)
    fingerprint_ids: list[str] = Field(min_length=1,max_length=32)
    expected_revisions: dict[str,int] = Field(min_length=1,max_length=32)
    action: str = Field(pattern=r'^(merge|split|unknown)$')
    request_id: str = Field(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


class VoiceRoomQuery(BaseModel):
    """A recorded room id (same bound as capture)."""

    model_config = ConfigDict(extra='forbid',strict=True)
    kind: Literal['room']
    id: str = Field(min_length=1,max_length=128,pattern=r'^[^\x00-\x1f\x7f]+$')


class VoiceUnrecordedRoomQuery(BaseModel):
    """Samples whose review row recorded no room."""

    model_config = ConfigDict(extra='forbid',strict=True)
    kind: Literal['unrecorded']


class VoiceWindowQuery(BaseModel):
    """Creation-time window [since, until) inside the 48 h retention plus skew."""

    model_config = ConfigDict(extra='forbid',strict=True)
    since: float = Field(allow_inf_nan=False)
    until: float = Field(allow_inf_nan=False)


class VoiceCaptureQuery(BaseModel):
    """T044-U1 closed body: every key is required on every request (R3-5).

    ``text`` null is a filters-only request; a non-null ``text`` is the
    admin-only transcript search (Q27). Values never travel in the URL.
    """

    model_config = ConfigDict(extra='forbid',strict=True)
    room: VoiceRoomQuery | VoiceUnrecordedRoomQuery | None
    window: VoiceWindowQuery | None
    group_state: Literal['suggested','manual','unknown'] | None
    association: Literal['suggested','unknown'] | None
    listenable_only: bool
    text: str | None = Field(max_length=1024)
    cursor: str | None = Field(min_length=118,max_length=118)


def _unique_object(pairs):
    keys = [key for key,_ in pairs]
    if len(set(keys))!=len(keys):
        raise ValueError(INVALID_QUERY)  # Every field is single-valued.
    return dict(pairs)


def _invalid_query() -> JSONResponse:
    """Fixed, no-store 422: the body (filters or query) is never echoed."""
    return JSONResponse(status_code=422,content={'detail':INVALID_QUERY},headers={'Cache-Control':'no-store'})


def register_voice_routes(app, *, voices=None) -> None:
    """Register source-retained, authenticated voice metadata and review actions."""
    router = APIRouter(prefix='/api/voices')
    async def reviewer(request: Request,principal: DashboardPrincipal = Depends(require_dashboard_session)):
        try:
            person = voice_person_scope_requested(request)
            authority = getattr(app.state,'dashboard_authority',None) if person else None
            if person and type(authority) is not DashboardAuthority:
                raise PermissionError('voice_person_authority_required')
            permissions = {'voices:review'}
            if person:
                permissions.add('admin:manage')
                if request.method in {'GET','HEAD','OPTIONS'}:
                    permissions.add('voices:read')
            return VoiceReviewer.from_dashboard(principal,
                session_guard(request,frozenset(permissions),principal),person_authority=authority)
        except PermissionError:
            raise HTTPException(403,detail='voice_review_denied') from None
    def service():
        if voices is None:
            raise HTTPException(503,detail='voices_unavailable')
        return voices
    async def call(function,*args):
        try:
            return await service().acall(function,*args)
        except PermissionError:
            raise HTTPException(403,detail='voice_review_denied') from None
        except LookupError:
            raise HTTPException(404,detail='voice_data_unavailable') from None
        except ValueError as error:
            raise HTTPException(409,detail=str(error)) from None
        except (RuntimeError,OSError,sqlite3.Error):
            raise HTTPException(503,detail='voices_unavailable') from None
    @router.get('')
    async def snapshot(request: Request,cursor: str | None = Query(None,max_length=85),actor: VoiceReviewer = Depends(reviewer)):
        return response(request,'captures',cursor,actor)
    @router.get('/profiles')
    async def profiles(request: Request,cursor: str | None = Query(None,max_length=36),
                       include_person: bool = Query(False),
                       person_id: str | None = Query(None,min_length=1,max_length=128,pattern=r'^[^\x00-\x1f\x7f]+$'),
                       actor: VoiceReviewer = Depends(reviewer)):
        if person_id is not None and not include_person:
            raise HTTPException(422,detail='person_filter_requires_enriched_profiles')
        return response(request,'profiles',cursor,actor,include_person=include_person,person_id=person_id)
    def response(request,kind,cursor,actor,*,include_person=False,person_id=None,variant='audio'):
        owner = getattr(app.state,'rest_delivery_owner',None)
        if owner is None:
            raise HTTPException(503,detail='voice_delivery_unavailable')
        permissions = {'voices:read','voices:review'}
        if include_person:
            permissions.add('admin:manage')
        return VoiceResponse(owner,service(),request.headers.get('x-kza-live-connection'),kind,cursor,actor,
            reader=session_reader(request,frozenset(permissions)),include_person=include_person,person_id=person_id,
            variant=variant)
    @router.post('/profiles')
    async def confirm(body: ConfirmVoiceBody,actor: VoiceReviewer = Depends(reviewer)):
        profile = await call(service().ledger.confirm,tuple(body.fingerprint_ids),body.name,body.request_id,actor)
        return {'profile_id':profile['id'],'revision':profile['revision']}


    @router.put('/profiles/{profile_id}/person')
    async def set_person(profile_id: str,body: SetVoicePersonBody,actor: VoiceReviewer = Depends(reviewer)):
        receipt = await call(service().ledger.people.set_person,profile_id,body.user_id,
            body.expected_profile_revision,body.expected_link_revision,actor)
        if type(receipt) is not VoicePersonReceipt:
            raise HTTPException(503,detail='voice_person_receipt_unavailable')
        return {'ok':True,'profile_id':receipt.profile_id,'link_revision':receipt.link_revision}
    @router.post('/groups/corrections')
    async def correct_groups(body: CorrectVoiceGroupsBody,actor: VoiceReviewer = Depends(reviewer)):
        return {'updated':await call(service().ledger.groups.correct,tuple(body.fingerprint_ids),
            body.expected_revisions,body.action,body.request_id,actor)}
    @router.get('/captures/{fingerprint_id}/review')
    async def review(request: Request,fingerprint_id: str,actor: VoiceReviewer = Depends(reviewer)):
        return response(request,'review',fingerprint_id,actor)
    @router.get('/captures/{fingerprint_id}/transcript')
    async def transcript(request: Request,fingerprint_id: str,actor: VoiceReviewer = Depends(reviewer)):
        """QB: the transcript and room of one sample, as the review's text-only variant."""
        if request.url.query or not VOICE_CAPTURE_ID.fullmatch(fingerprint_id):
            return JSONResponse(status_code=422,content={'detail':'invalid_voice_capture_id'},
                headers={'Cache-Control':'no-store'})
        return response(request,'review',fingerprint_id,actor,variant='text')
    @router.post('/captures/query')
    async def query(request: Request,principal: DashboardPrincipal = Depends(require_dashboard_session)):
        """Filtered captures page with a live count; a non-null ``text`` is the admin search.

        Nothing about the body (filters, query or result size) is logged here.
        """
        raw = body = filters = bound = text = window = None
        try:
            if request.url.query:
                return _invalid_query()
            raw = bytearray()
            async for part in request.stream():
                raw += part
                if len(raw)>QUERY_BODY_MAX_BYTES:
                    return _invalid_query()
            try:
                body = VoiceCaptureQuery.model_validate(json.loads(bytes(raw),object_pairs_hook=_unique_object))
            except (ValueError,TypeError,ValidationError,RecursionError):
                return _invalid_query()
            text = None
            if body.text is not None:
                text = normalize_voice_text(body.text)
                if not 2<=len(text)<=QUERY_TEXT_MAX_CHARS or len(text.split())>QUERY_TEXT_MAX_TERMS:
                    return _invalid_query()
            window = body.window
            if window is not None:
                now = service().ledger.ledger.clock()
                if not (math.isfinite(now) and 0<=window.since<window.until<=now+QUERY_CLOCK_SKEW_S):
                    return _invalid_query()
            try:
                filters = VoiceCaptureFilter(body.room.kind if body.room else None,
                    body.room.id if type(body.room) is VoiceRoomQuery else None,
                    window.since if window else None,window.until if window else None,
                    body.group_state,body.association,body.listenable_only,text)
                bound = capture_query_position(body.cursor)[2] if body.cursor is not None else None
            except ValueError:
                return _invalid_query()
            permissions = {'voices:read','voices:review'}
            if filters.search:
                # Q27b/F30: the body is invisible to the route permission, so the
                # search's web-session rule and admin authority are enforced here.
                if not web_session_requested(request):
                    raise HTTPException(403,detail='voice_search_requires_session')
                lease = current_lease(request)
                if lease is None or lease.principal is not principal or not lease.current(frozenset({'admin:manage'})):
                    raise HTTPException(403,detail='voice_search_denied')
                permissions.add('admin:manage')
            if bound is not None and bound!=filters.digest():
                # A cursor of another filter, query or process; checked here because
                # delivery errors surface as 503.
                return JSONResponse(status_code=409,content={'detail':'voice_cursor_filter_mismatch'},
                    headers={'Cache-Control':'no-store'})
            try:
                actor = VoiceReviewer.from_dashboard(principal,session_guard(request,frozenset(permissions),principal))
            except PermissionError:
                raise HTTPException(403,detail='voice_review_denied') from None
            owner = getattr(app.state,'rest_delivery_owner',None)
            if owner is None:
                raise HTTPException(503,detail='voice_delivery_unavailable')
            return VoiceResponse(owner,service(),request.headers.get('x-kza-live-connection'),'captures',body.cursor,actor,
                reader=session_reader(request,frozenset(permissions)),query=filters)
        finally:
            raw = body = filters = bound = text = window = None
    @router.delete('/profiles/{profile_id}')
    async def delete(profile_id: str,body: DeleteVoiceBody,actor: VoiceReviewer = Depends(reviewer)):
        return {'deleted':await call(service().ledger.delete_profile,profile_id,body.expected_revision,actor)}
    app.include_router(router)
