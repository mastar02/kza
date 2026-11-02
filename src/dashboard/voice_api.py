"""Explicit dashboard-only voice review; vectors never cross the HTTP boundary."""

import sqlite3

from fastapi import APIRouter,Depends,HTTPException,Query,Request
from pydantic import BaseModel,ConfigDict,Field

from src.dashboard.auth import DashboardPrincipal, require_dashboard_session, session_guard, session_reader, voice_person_scope_requested
from src.dashboard.authority import DashboardAuthority
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_people import VoicePersonReceipt
from src.dashboard.voice_delivery import VoiceResponse


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
                       include_person: bool = Query(False),actor: VoiceReviewer = Depends(reviewer)):
        return response(request,'profiles',cursor,actor,include_person=include_person)
    def response(request,kind,cursor,actor,*,include_person=False):
        owner = getattr(app.state,'rest_delivery_owner',None)
        if owner is None:
            raise HTTPException(503,detail='voice_delivery_unavailable')
        permissions = {'voices:read','voices:review'}
        if include_person:
            permissions.add('admin:manage')
        return VoiceResponse(owner,service(),request.headers.get('x-kza-live-connection'),kind,cursor,actor,
            reader=session_reader(request,frozenset(permissions)),include_person=include_person)
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
    @router.delete('/profiles/{profile_id}')
    async def delete(profile_id: str,body: DeleteVoiceBody,actor: VoiceReviewer = Depends(reviewer)):
        return {'deleted':await call(service().ledger.delete_profile,profile_id,body.expected_revision,actor)}
    app.include_router(router)
