"""Exact temporary evaluation reads, without catalogue or voice-review powers."""

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from src.dashboard.auth import current_lease, dashboard_authority, require_dashboard_session
from src.dashboard.authority import DashboardPrincipal, DashboardRole, EvaluationModality
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.evaluation_delivery import EvaluationGrantsResponse, EvaluationResponse
from src.dashboard.rest_delivery import RestDeliveryOwner


def register_evaluation_routes(app, *, voices=None) -> None:
    router = APIRouter(prefix='/api/evaluation')

    @router.get('/grants')
    async def grants(request: Request,
                     _principal: DashboardPrincipal = Depends(require_dashboard_session)):
        authority = dashboard_authority(request,required=True)
        actor = current_lease(request)
        if (actor is None or actor._authority is not authority
                or actor.principal.role is not DashboardRole.EVALUATOR
                or not actor.current(frozenset({'evaluation:read'}))):
            raise HTTPException(403,detail='evaluation_scope_unavailable')
        if request.query_params:
            raise HTTPException(422,detail='invalid_evaluation_query')
        owner = getattr(app.state,'rest_delivery_owner',None)
        if not isinstance(owner,RestDeliveryOwner):
            raise HTTPException(503,detail='evaluation_delivery_unavailable')
        return EvaluationGrantsResponse(owner,authority,actor)

    @router.get('/samples/{fingerprint_id}')
    async def sample(request: Request,
                     fingerprint_id: str = Path(min_length=67,max_length=67,pattern=r'^vf_[0-9a-f]{64}$'),
                     modality: EvaluationModality = Query(...),
                     _principal: DashboardPrincipal = Depends(require_dashboard_session)):
        if set(request.query_params)!={'modality'} or len(request.query_params.getlist('modality'))!=1:
            raise HTTPException(422,detail='invalid_evaluation_query')
        try:
            lease = current_lease(request)
            actor = EvaluationReadActor(lease,fingerprint_id,modality)
            reader = lease.reader_binding(frozenset({'evaluation:read'}))
            if reader is None:
                raise PermissionError('evaluation_scope_unavailable')
        except PermissionError:
            raise HTTPException(403,detail='evaluation_scope_unavailable') from None
        owner = getattr(app.state,'rest_delivery_owner',None)
        if voices is None or owner is None:
            raise HTTPException(503,detail='evaluation_delivery_unavailable')
        connections = request.headers.getlist('x-kza-live-connection')
        if len(connections)!=1:
            raise HTTPException(428,detail='evaluation_connection_required')
        return EvaluationResponse(owner,voices,connections[0],actor,reader=reader)

    app.include_router(router)
