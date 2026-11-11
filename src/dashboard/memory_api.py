"""Authenticated fact/proposal review, owner assignment and 'Mis recuerdos'.

Operator routes (U3/U3b-2) see every fact with its owner state. Person routes
under ``/api/memory/mine`` (U3b-2) act only on facts the authenticated
household account owns: the person comes from the credential's account
binding, never from a voice label, and another person's fact answers exactly
like a missing one.
"""

import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from src.dashboard.auth import DashboardPrincipal, dashboard_authority, require_dashboard_session, session_guard
from src.memory.fact_ledger import (
    MEMORY_PAGE_LIMIT,
    FactPageCursor,
    MemoryAssigner,
    MemoryOwnerActor,
    MemoryOwnerDependents,
    MemoryOwnerReader,
    MemoryReviewer,
    MemoryScope,
)

logger = logging.getLogger(__name__)

FactId = Annotated[str, Path(pattern=r"^fact_[a-f0-9]{64}$")]
_FACT_ID = re.compile(r"fact_[a-f0-9]{64}")
_LIMIT = re.compile(r"[1-9][0-9]{0,2}")

# A16 (U3b-1a R5): assigning an automatic memory (or a proposal) confirms it with
# a new revision, so receipts derived from the previous revision stop counting.
ASSIGN_NOTICE = ("Asignar un recuerdo lo deja confirmado y sin vencimiento. Si era automático o una propuesta, "
                 "los registros derivados de su versión anterior (eventos, clips de activación, correcciones "
                 "y turnos de conversación) dejan de estar disponibles.")
DEPENDENTS_NOTICE = ("Este recuerdo automático tiene recuerdos derivados. Para asignarlo hay que eliminarlos "
                     "en la misma operación.")
RETURN_NOTICE = ("Lo devolviste a la administración. No se borró: sigue guardado hasta que la administración "
                 "lo reasigne o lo elimine. Si querés que se borre, usá Eliminar antes de devolverlo.")

# contracts_1a.exceptions -> HTTP (design v5), plus memory_owner_actor_denied (A10).
_OWNER_ERRORS = {
    "memory_owner_conflict": 409,
    "memory_owner_derived_fact": 409,
    "memory_owner_dependents_not_applicable": 422,
    "memory_owner_invalid_person": 422,
    "memory_owner_retired": 422,
    "memory_owner_inactive": 422,
    "memory_owner_authority_missing": 503,
    "memory_owner_authority_unavailable": 503,
    "memory_assign_denied": 403,
}


def _invalid(detail: str) -> HTTPException:
    return HTTPException(422, detail=detail)


def _page_query(request: Request, *cursors: str) -> tuple[int, dict[str, FactPageCursor | None]]:
    """Parse ``limit`` and named cursors; every malformed value is the same 422."""
    params = request.query_params
    try:
        for name in ("limit", *cursors):
            if len(params.getlist(name)) > 1:
                raise ValueError("memory_page_invalid")
        text = params.get("limit")
        if text is not None and (_LIMIT.fullmatch(text) is None or int(text) > MEMORY_PAGE_LIMIT):
            raise ValueError("memory_page_invalid")
        limit = MEMORY_PAGE_LIMIT if text is None else int(text)
        return limit, {name: None if params.get(name) is None else FactPageCursor.decode(params.get(name))
                       for name in cursors}
    except ValueError as error:
        raise _invalid("memory_page_invalid") from error


def _strict_query(request: Request, *allowed: str) -> None:
    if any(key not in allowed for key in request.query_params.keys()):
        raise _invalid("memory_page_invalid")


def _fact_id(value: str) -> str:
    if _FACT_ID.fullmatch(value) is None:
        raise _invalid("memory_fact_invalid")
    return value


async def _owner_body(request: Request) -> dict:
    """Parse the owner change body strictly (no coercion, no extra keys)."""
    try:
        body = await request.json()
    except Exception:
        raise _invalid("memory_owner_invalid_request") from None

    def revision(value: object, minimum: int) -> bool:
        return type(value) is int and minimum <= value < 2 ** 53

    if type(body) is not dict:
        raise _invalid("memory_owner_invalid_request")
    if body.get("person_id") is None and "person_id" in body:
        if set(body) != {"person_id", "expected_owner_revision"} or not revision(body["expected_owner_revision"], 0):
            raise _invalid("memory_owner_invalid_request")
        return body
    if (not {"person_id", "expected_fact_revision", "expected_owner_revision"} <= set(body)
            or not set(body) <= {"person_id", "expected_fact_revision", "expected_owner_revision", "delete_dependents"}
            or type(body["person_id"]) is not str or not revision(body["expected_fact_revision"], 1)
            or not revision(body["expected_owner_revision"], 0)
            or type(body.get("delete_dependents", False)) is not bool):
        raise _invalid("memory_owner_invalid_request")
    return body


def register_memory_routes(app, *, manager=None) -> None:
    """Register session-bound reads, review, owner assignment and the person's own view."""
    router = APIRouter(prefix="/api/memory")

    async def review(request: Request, principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryReviewer:
        try:
            return MemoryReviewer.from_dashboard(principal,
                session_guard(request, frozenset({'memory:review'}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_review_denied") from error

    async def assigner(request: Request, principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryAssigner:
        try:
            return MemoryAssigner.from_dashboard(principal,
                session_guard(request, frozenset({'memory:assign'}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_assign_denied") from error

    async def owner_reader(request: Request,
                           principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryOwnerReader:
        try:
            return MemoryOwnerReader.from_dashboard(principal,
                session_guard(request, frozenset({'memory:own:read'}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_person_required") from error

    async def owner_actor(request: Request,
                          principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryOwnerActor:
        try:
            return MemoryOwnerActor.from_dashboard(principal,
                session_guard(request, frozenset({'memory:own:write'}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_person_required") from error

    def memory():
        if manager is None:
            raise HTTPException(503, detail="memory_unavailable")
        return manager.long_term

    async def run(function, *args, **kwargs):
        """Run one ledger call; an integrity or availability failure is a content-free 503."""
        service = memory()
        try:
            return await service.acall(function, *args, **kwargs)
        except RuntimeError as error:
            logger.warning("[MemoryAPI] memory call unavailable: %s", type(error).__name__)
            raise HTTPException(503, detail="memory_unavailable") from error

    # --- operator ---------------------------------------------------------------------------

    @router.get("")
    async def snapshot(request: Request, _principal: DashboardPrincipal = Depends(require_dashboard_session)):
        limit, cursors = _page_query(request, "facts_after", "proposals_after")
        service = memory()
        page = await run(service.ledger.review_page, facts_after=cursors["facts_after"],
                         proposals_after=cursors["proposals_after"], limit=limit)
        return {**page, "assign_notice": ASSIGN_NOTICE}

    @router.get("/facts")
    async def facts(_principal: DashboardPrincipal = Depends(require_dashboard_session)):
        service = memory()
        rows = await service.acall(service.ledger.list, scope=MemoryScope.ALL, limit=100)
        return {"facts": [fact.to_dict() for fact in rows], "status": await service.acall(service.get_stats)}

    @router.get("/proposals")
    async def proposals(_principal: DashboardPrincipal = Depends(require_dashboard_session)):
        service = memory()
        return {"proposals": await service.acall(service.proposals)}

    @router.get("/returned")
    async def returned(request: Request, actor: MemoryAssigner = Depends(assigner)):
        _strict_query(request, "after", "limit")
        limit, cursors = _page_query(request, "after")
        service = memory()
        try:
            page = await run(service.ledger.returned_page, actor, after=cursors["after"], limit=limit)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_assign_denied") from error
        return {"facts": list(page.items), "next": page.next(), "limit": limit}

    @router.get("/owners/counts")
    async def owner_counts(request: Request, actor: MemoryAssigner = Depends(assigner)):
        _strict_query(request)
        service = memory()
        try:
            counts = await run(service.ledger.owned_counts, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_assign_denied") from error
        return {**counts.to_dict(), "retired_owner_prune_pending": service.ledger.retired_owner_prune_pending is True}

    @router.post("/facts/{fact_id}/owner")
    async def change_owner(fact_id: str, request: Request, actor: MemoryAssigner = Depends(assigner)):
        _fact_id(fact_id)
        body = await _owner_body(request)
        service = memory()
        authority = dashboard_authority(request)
        try:
            if body["person_id"] is None:
                receipt = await run(service.ledger.unassign_owner, fact_id, body["expected_owner_revision"], actor)
                return {"fact_revision": receipt.fact_revision, "owner_revision": receipt.owner_revision}
            if authority is None:
                raise PermissionError("memory_owner_authority_missing")
            receipt = await run(service.ledger.assign_owner, fact_id, body["person_id"],
                                body["expected_fact_revision"], body["expected_owner_revision"], actor,
                                retired=authority.memory_owner_retired, active=authority.memory_owner_active,
                                delete_dependents=body.get("delete_dependents", False))
        except MemoryOwnerDependents as error:
            raise HTTPException(409, detail={
                "code": error.code, "notice": DEPENDENTS_NOTICE,
                "dependents": [{"id": identity, "state": state} for identity, state in error.dependents]}) from error
        except LookupError as error:
            # Only the ledger's own missing/ineligible-fact refusal is a 404 (review R2);
            # any other lookup failure is a defect, never presented as 'fact missing'.
            if str(error) != "memory_fact_unavailable":
                logger.warning("[MemoryAPI] unexpected lookup failure: %s", type(error).__name__)
                raise HTTPException(503, detail="memory_unavailable") from error
            raise HTTPException(404, detail="memory_fact_unavailable") from error
        except (ValueError, PermissionError) as error:
            code = str(error)
            if code not in _OWNER_ERRORS:
                logger.warning("[MemoryAPI] unmapped owner refusal: %s", type(error).__name__)
                raise HTTPException(503, detail="memory_unavailable") from error
            raise HTTPException(_OWNER_ERRORS[code], detail=code) from error
        confirmed = receipt.fact_revision != body["expected_fact_revision"]
        if confirmed:
            try:
                await service.acall(service.owner_index_changed)
            except Exception as error:  # The assignment is durable; maintenance retries indexing.
                logger.warning("[MemoryAPI] index update deferred after assignment: %s", type(error).__name__)
        return {"fact_revision": receipt.fact_revision, "owner_revision": receipt.owner_revision,
                "deleted_dependents": receipt.deleted_dependents, "confirmed_now": confirmed,
                "derived_records_retired": confirmed, "notice": ASSIGN_NOTICE if confirmed else None}

    @router.post("/proposals/{fact_id}/confirm")
    async def confirm(fact_id: FactId, actor: MemoryReviewer = Depends(review)):
        service = memory()
        try:
            fact = await service.acall(service.confirm, fact_id, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_review_denied") from error
        except LookupError as error:
            raise HTTPException(404, detail="memory_proposal_unavailable") from error
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error
        return {"fact": fact, "status": await service.acall(service.get_stats)}

    @router.post("/proposals/{fact_id}/reject")
    async def reject(fact_id: FactId, actor: MemoryReviewer = Depends(review)):
        service = memory()
        try:
            deleted = await service.acall(service.reject_proposal, fact_id, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_review_denied") from error
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error
        return {"deleted": deleted, "status": await service.acall(service.get_stats)}

    @router.delete("/facts/{fact_id}")
    async def remove(fact_id: FactId, actor: MemoryReviewer = Depends(review)):
        service = memory()
        try:
            deleted = await service.acall(service.delete_fact, fact_id, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_review_denied") from error
        return {"deleted": deleted, "status": await service.acall(service.get_stats)}

    # --- the person's own memories ('Mis recuerdos') ----------------------------------------
    # No household counters, sources, operator subject or other people's ids.

    @router.get("/mine")
    async def mine(request: Request, actor: MemoryOwnerReader = Depends(owner_reader)):
        _strict_query(request, "after", "limit")
        limit, cursors = _page_query(request, "after")
        service = memory()
        try:
            page = await run(service.ledger.owned_page, actor, after=cursors["after"], limit=limit)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_owner_actor_denied") from error
        return {"facts": list(page.items), "next": page.next(), "limit": limit}

    @router.delete("/mine/facts/{fact_id}")
    async def delete_mine(fact_id: str, actor: MemoryOwnerActor = Depends(owner_actor)):
        _fact_id(fact_id)
        service = memory()
        try:
            deleted = await run(service.delete_owned, fact_id, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_owner_actor_denied") from error
        return {"deleted": deleted}

    @router.post("/mine/facts/{fact_id}/return")
    async def return_mine(fact_id: str, actor: MemoryOwnerActor = Depends(owner_actor)):
        _fact_id(fact_id)
        service = memory()
        try:
            returned = await run(service.ledger.return_owned, fact_id, actor.person_id, actor)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_owner_actor_denied") from error
        return {"returned": True, "notice": RETURN_NOTICE} if returned else {"returned": False}

    app.include_router(router)
