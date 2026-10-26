"""Authenticated fact/proposal review, separate from voice suggestions."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from src.dashboard.auth import DashboardPrincipal, require_dashboard_session
from src.memory.fact_ledger import MemoryReviewer

FactId = Annotated[str, Path(pattern=r"^fact_[a-f0-9]{64}$")]


def register_memory_routes(app, *, manager=None) -> None:
    """Register session-bound reads and explicit confirmation/rejection."""
    router = APIRouter(prefix="/api/memory")

    async def review(principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryReviewer:
        try:
            return MemoryReviewer.from_dashboard(principal)
        except PermissionError as error:
            raise HTTPException(403, detail="memory_review_denied") from error

    def memory():
        if manager is None:
            raise HTTPException(503, detail="memory_unavailable")
        return manager.long_term

    @router.get("")
    async def snapshot(_reviewer: MemoryReviewer = Depends(review)):
        service = memory()
        return await service.acall(service.ledger.snapshot)

    @router.get("/facts")
    async def facts(_reviewer: MemoryReviewer = Depends(review)):
        service = memory()
        rows = await service.acall(service.ledger.list, limit=100)
        return {"facts": [fact.to_dict() for fact in rows], "status": await service.acall(service.get_stats)}

    @router.get("/proposals")
    async def proposals(_reviewer: MemoryReviewer = Depends(review)):
        service = memory()
        return {"proposals": await service.acall(service.proposals)}

    @router.post("/proposals/{fact_id}/confirm")
    async def confirm(fact_id: FactId, actor: MemoryReviewer = Depends(review)):
        service = memory()
        try:
            fact = await service.acall(service.confirm, fact_id, actor)
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
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error
        return {"deleted": deleted, "status": await service.acall(service.get_stats)}

    @router.delete("/facts/{fact_id}")
    async def remove(fact_id: FactId, actor: MemoryReviewer = Depends(review)):
        service = memory()
        deleted = await service.acall(service.delete_fact, fact_id, actor)
        return {"deleted": deleted, "status": await service.acall(service.get_stats)}

    app.include_router(router)
