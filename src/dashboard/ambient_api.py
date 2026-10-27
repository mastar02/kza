"""Dashboard-only pause/reactivation, independent of audio input or identities."""

from fastapi import APIRouter,Depends,HTTPException,Request,Response
from pydantic import BaseModel,ConfigDict,Field

from src.ambient.control import AmbientController
from src.dashboard.auth import COOKIE_NAME,DashboardPrincipal,check_token,require_dashboard_session


class AmbientPauseBody(BaseModel):
    """An exact revision prevents stale tabs from reactivating collection."""

    model_config = ConfigDict(extra='forbid',strict=True)
    paused: bool
    expected_revision: int = Field(ge=0)


def register_ambient_routes(app, *, control=None) -> None:
    """Expose real status and one bounded, server-owned transition at a time."""
    router = APIRouter(prefix='/api/ambient/control')
    async def controller(request: Request, principal: DashboardPrincipal = Depends(require_dashboard_session)):
        header,cookie = request.headers.get('authorization'),request.cookies.get(COOKIE_NAME)
        try:
            return AmbientController.from_dashboard(principal,lambda:check_token(header,cookie))
        except PermissionError:
            raise HTTPException(403,detail='ambient_control_denied') from None
    def service():
        if control is None:
            raise HTTPException(503,detail='ambient_control_unavailable')
        return control
    @router.get('')
    async def status(response: Response,actor: AmbientController = Depends(controller)):
        try:
            actor.require_current()
        except PermissionError:
            raise HTTPException(403,detail='ambient_control_denied') from None
        response.headers['Cache-Control'] = 'no-store'
        return service().snapshot()
    @router.post('',status_code=202)
    async def pause(body: AmbientPauseBody,response: Response,actor: AmbientController = Depends(controller)):
        try:
            response.headers['Cache-Control'] = 'no-store'
            return service().request(body.paused,body.expected_revision,actor)
        except PermissionError:
            raise HTTPException(403,detail='ambient_control_denied') from None
        except ValueError:
            raise HTTPException(409,detail='ambient_control_revision_or_transition_conflict') from None
        except RuntimeError:
            raise HTTPException(503,detail='ambient_control_unavailable') from None
    app.include_router(router)
