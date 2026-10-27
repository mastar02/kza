"""Explicit dashboard-only voice review; vectors never cross the HTTP boundary."""

from fastapi import APIRouter,Depends,HTTPException,Query,Request
from pydantic import BaseModel,ConfigDict,Field

from src.dashboard.auth import COOKIE_NAME,DashboardPrincipal,check_token,require_dashboard_session
from src.users.voice_ledger import VoiceReviewer
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
        header,cookie = request.headers.get('authorization'),request.cookies.get(COOKIE_NAME)
        try:
            return VoiceReviewer.from_dashboard(principal,lambda:check_token(header,cookie))
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
        except RuntimeError:
            raise HTTPException(503,detail='voices_unavailable') from None
    @router.get('')
    async def snapshot(request: Request,cursor: str | None = Query(None,max_length=67),actor: VoiceReviewer = Depends(reviewer)):
        return response(request,'captures',cursor,actor)
    @router.get('/profiles')
    async def profiles(request: Request,cursor: str | None = Query(None,max_length=36),actor: VoiceReviewer = Depends(reviewer)):
        return response(request,'profiles',cursor,actor)
    def response(request,kind,cursor,actor):
        owner = getattr(app.state,'rest_delivery_owner',None)
        if owner is None:
            raise HTTPException(503,detail='voice_delivery_unavailable')
        return VoiceResponse(owner,service(),request.headers.get('x-kza-live-connection'),kind,cursor,actor)
    @router.post('/profiles')
    async def confirm(body: ConfirmVoiceBody,actor: VoiceReviewer = Depends(reviewer)):
        profile = await call(service().ledger.confirm,tuple(body.fingerprint_ids),body.name,body.request_id,actor)
        return {'profile_id':profile['id'],'revision':profile['revision']}
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
