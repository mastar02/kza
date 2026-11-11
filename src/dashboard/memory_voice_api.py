"""Voice opt-in routes for 'Mis recuerdos' and their administration (U3b-4a-2).

Check order on every route: authentication (401), then route permission
(403, in src/dashboard/auth.py), then the settings flag (409
``memory_voice_disabled``), then body validation (422), then state (409).
With ``memory.voice_proposals.enabled`` false only the person's status and
revocation answer, so revocation is always possible; the G1 switch routes also
stay reachable (they are the off path and report ``configured``). The person
comes from the credential's account binding, never from a body or a voice
label. Responses never name another person's state, window, room or profile.
"""

import logging
import re
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request

from src.dashboard.auth import (
    DashboardPrincipal, require_dashboard_session, session_guard,
)
from src.memory.fact_ledger import MemoryAssigner, MemoryVoiceConsentActor, VoiceSwitchActor

logger = logging.getLogger(__name__)

# Visible text is Spanish (design v4 disclosure_copy).
OPT_IN_DISCLOSURE = (
    "Si te inscribís, lo que pidas recordar con tu voz ('acordate que…') puede quedar como propuesta tuya. "
    "Hasta que la administración la confirme vas a ver sólo un aviso sin el texto: podés descartarla, o decir "
    "'No es mío' si no fuiste vos; si nadie la confirma, vence sola. Para inscribirte tenés que decir en voz alta, "
    "en una habitación, una frase que te mostramos. Como cualquier cosa que se dice en casa, esa frase puede "
    "quedar unos segundos en el registro de voces que revisa la administración o un evaluador autorizado, hasta "
    "48 horas en la transcripción ambiental de esa habitación, y el audio de las primeras palabras puede quedar "
    "en los clips de activación ('Nexa') que se guardan para mejorar el reconocimiento. KZA compara cada pedido "
    "con todas las voces registradas en el hogar sólo para descartar confusiones: si no está seguro, la "
    "propuesta queda para la administración. Tu voz no te da permisos ni acceso. Podés darte de baja en "
    "cualquier momento.")
VOICES_DISCLOSURE = (
    "Para evitar confusiones, KZA compara cada '¿acordate que…?' con todas las voces registradas en el hogar, "
    "incluidas las de personas no inscriptas, sólo para descartar coincidencias. Nunca asigna un recuerdo a "
    "alguien que no se haya inscripto y verificado. Vincular una voz a una persona no le da permisos ni la "
    "inscribe.")
REENROLMENT_NOTICE = ("Tenés más de una voz registrada a tu nombre. Para inscribirte, pedile a la administración "
                      "que deje una sola.")
MULTI_PROFILE_NOTICE = ("Esta persona tiene más de una voz vinculada: no puede verificarse ni recibir propuestas "
                        "por voz hasta que quede una sola")
STATE_MESSAGES = {
    "active": "Listo: verificamos tu voz. Desde ahora, lo que pidas recordar con tu voz puede quedar como "
              "propuesta tuya.",
    "verification_locked": "Hubo varios intentos sin éxito (también cuenta si otra persona dijo tu frase). "
                           "Pedile a la administración que habilite uno nuevo.",
}
INSUFFICIENT_PROFILES_MESSAGE = ("Todavía no se puede verificar tu voz en este hogar. La administración te va a "
                                 "avisar cuando sea posible.")

_INVALID = "memory_voice_invalid_request"
_ROOM = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_PERSON = re.compile(r"[A-Za-z0-9_.:@-]{1,128}")
_PROFILE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Ledger refusal -> (status, public code). Anything else is a content-free 503.
_ERRORS = {
    "memory_voice_disabled": (409, "memory_voice_disabled"),
    "memory_voice_consent_conflict": (409, "memory_voice_conflict"),
    "memory_voice_switch_conflict": (409, "memory_voice_conflict"),
    "memory_voice_suspension_conflict": (409, "memory_voice_conflict"),
    "memory_voice_consent_state": (409, "memory_voice_consent_state"),
    "memory_voice_consent_locked": (409, "memory_voice_consent_locked"),
    "memory_voice_no_profile": (409, "memory_voice_no_profile"),
    "memory_voice_profile_ambiguous": (409, "memory_voice_profile_ambiguous"),
    "memory_voice_insufficient_profiles": (409, "memory_voice_insufficient_profiles"),
    "memory_voice_profile_suspended": (409, "memory_voice_profile_suspended"),
    "memory_voice_scoring_incomplete": (409, "memory_voice_scoring_incomplete"),
    "memory_voice_not_configured": (409, "memory_voice_not_configured"),
    "memory_voice_rate_limited": (429, "memory_voice_rate_limited"),
    "memory_voice_actor_denied": (403, "memory_voice_actor_denied"),
    "memory_voice_switch_denied": (403, "memory_voice_switch_denied"),
    "memory_voice_subject_invalid": (403, "memory_voice_subject_invalid"),
    "memory_assign_denied": (403, "memory_assign_denied"),
}


def _revision(minimum: int) -> Callable[[object], bool]:
    return lambda value: type(value) is int and minimum <= value < 2 ** 53


async def _body(request: Request, **checks: Callable[[object], bool]) -> dict:
    """Exactly these keys, no coercion: any other shape is the same 422."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, detail=_INVALID) from None
    if type(body) is not dict or set(body) != set(checks) or not all(check(body[key]) for key, check in checks.items()):
        raise HTTPException(422, detail=_INVALID)
    return body


def _refusal(error: Exception) -> HTTPException:
    status, code = _ERRORS.get(str(error), (None, None))
    if status is None:
        logger.warning("[MemoryVoiceAPI] unmapped refusal: %s", type(error).__name__)
        return HTTPException(503, detail="memory_unavailable")
    if code == "memory_voice_profile_ambiguous":
        return HTTPException(status, detail={"code": code, "notice": REENROLMENT_NOTICE})
    if code == "memory_voice_insufficient_profiles":
        return HTTPException(status, detail={"code": code, "notice": INSUFFICIENT_PROFILES_MESSAGE})
    return HTTPException(status, detail=code)


def register_memory_voice_routes(app, *, manager=None,
                                 room_available: Callable[[str], bool] | None = None) -> None:
    """Register the person's voice opt-in routes and the operator's consent, switch and resume routes."""
    # Origin is checked by require_dashboard_session after authentication, as on every memory route.
    router = APIRouter()

    def ledger():
        if manager is None:
            raise HTTPException(503, detail="memory_unavailable")
        return manager.long_term.ledger

    def flag() -> None:
        if ledger().voice_settings.enabled is not True:
            raise HTTPException(409, detail="memory_voice_disabled")

    async def run(function, *args, **kwargs):
        service = manager.long_term
        try:
            return await service.acall(function, *args, **kwargs)
        except RuntimeError as error:
            logger.warning("[MemoryVoiceAPI] memory call unavailable: %s", type(error).__name__)
            raise HTTPException(503, detail="memory_unavailable") from error
        except (ValueError, PermissionError) as error:
            raise _refusal(error) from error

    async def person(request: Request,
                     principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryVoiceConsentActor:
        try:
            return MemoryVoiceConsentActor.from_dashboard(
                principal, session_guard(request, frozenset({"memory:own:voice"}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_person_required") from error

    async def assigner(request: Request,
                       principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryAssigner:
        try:
            return MemoryAssigner.from_dashboard(principal,
                                                 session_guard(request, frozenset({"memory:assign"}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_assign_denied") from error

    # --- the person ------------------------------------------------------------------------

    @router.get("/api/memory/mine/voice")
    async def status(request: Request, actor: MemoryVoiceConsentActor = Depends(person)):
        if request.query_params:
            raise HTTPException(422, detail=_INVALID)
        current = await run(ledger().voice_status, actor)
        if ledger().voice_settings.enabled is not True:  # D31: status stays available, content-minimal
            return {"state": current["state"], "revision": current["revision"], "enabled": False, "window": None}
        window = current["window"]  # validated against its binding by the ledger
        return {"state": current["state"], "revision": current["revision"], "enabled": current["effective"],
                "window": None if window is None else {"room_id": window.room_id, "phrase": window.phrase,
                                                       "expires_at": window.expires_at},
                "failures_left": current["failures_left"], "message": STATE_MESSAGES.get(current["state"]),
                "disclosure": OPT_IN_DISCLOSURE}

    @router.post("/api/memory/mine/voice/consent")
    async def request_consent(request: Request, actor: MemoryVoiceConsentActor = Depends(person)):
        flag()
        body = await _body(request, expected_revision=_revision(0))
        consent = await run(ledger().request_voice_consent, actor, body["expected_revision"])
        return {"state": consent.state.value, "revision": consent.revision}

    @router.delete("/api/memory/mine/voice/consent")
    async def revoke_consent(request: Request, actor: MemoryVoiceConsentActor = Depends(person)):
        body = await _body(request, expected_revision=_revision(0))  # works with the flag off (revocation)
        consent = await run(ledger().revoke_voice_consent, actor, body["expected_revision"])
        return {"state": consent.state.value, "revision": consent.revision}

    @router.post("/api/memory/mine/voice/verification")
    async def open_window(request: Request, actor: MemoryVoiceConsentActor = Depends(person)):
        flag()
        body = await _body(request, room_id=lambda value: type(value) is str and _ROOM.fullmatch(value) is not None,
                           expected_revision=_revision(0))
        # Review R/W: consent, profile and rate-limit checks run before the room is read, the
        # room answer consumes a window opening, and the window is registered under the ledger.
        window = await run(ledger().open_voice_window, actor, body["expected_revision"], body["room_id"],
                           room_available)
        if window is None:
            raise HTTPException(409, detail="memory_voice_room_unavailable")
        return {"window": {"room_id": window.room_id, "phrase": window.phrase, "expires_at": window.expires_at}}

    # --- administration ----------------------------------------------------------------------

    @router.get("/api/memory/voice/consents")
    async def consents(request: Request, actor: MemoryAssigner = Depends(assigner)):
        flag()
        if request.query_params:
            raise HTTPException(422, detail=_INVALID)
        listing = await run(ledger().voice_consents, actor)
        return {**listing, "multiple_profiles_notice": MULTI_PROFILE_NOTICE, "disclosure": VOICES_DISCLOSURE}

    @router.post("/api/memory/voice/consents/{person_id}/reopen")
    async def reopen(person_id: str, request: Request, actor: MemoryAssigner = Depends(assigner)):
        flag()
        body = await _body(request, expected_revision=_revision(0))
        if _PERSON.fullmatch(person_id) is None:
            raise HTTPException(422, detail=_INVALID)
        consent = await run(ledger().reopen_voice_consent, person_id, body["expected_revision"], actor)
        return {"state": consent.state.value, "revision": consent.revision}

    @router.get("/api/memory/voice/switch")
    async def switch(request: Request, _actor: MemoryAssigner = Depends(assigner)):
        if request.query_params:
            raise HTTPException(422, detail=_INVALID)
        state = await run(ledger().voice_switch)
        return {"configured": state.configured, "enabled": state.enabled, "revision": state.revision,
                "effective": state.effective}

    @router.post("/api/memory/voice/switch")
    async def set_switch(request: Request, principal: DashboardPrincipal = Depends(require_dashboard_session)):
        try:
            actor = VoiceSwitchActor.from_dashboard(
                principal, session_guard(request, frozenset({"admin:manage", "memory:assign"}), principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_voice_switch_denied") from error
        body = await _body(request, enabled=lambda value: type(value) is bool, expected_revision=_revision(0))
        state = await run(ledger().set_voice_switch, body["enabled"], body["expected_revision"], actor)
        return {"configured": state.configured, "enabled": state.enabled, "revision": state.revision,
                "effective": state.effective}

    async def resume_actor(request: Request,
                           principal: DashboardPrincipal = Depends(require_dashboard_session)) -> MemoryAssigner:
        # Defence in depth: the route permission layer already requires these three.
        needed = frozenset({"memory:assign", "voices:review", "admin:manage"})
        try:
            if not needed <= principal.permissions:
                raise PermissionError("memory_assign_denied")
            return MemoryAssigner.from_dashboard(principal, session_guard(request, needed, principal))
        except PermissionError as error:
            raise HTTPException(403, detail="memory_assign_denied") from error

    @router.post("/api/voices/profiles/{profile_id}/attribution/resume")
    async def resume(profile_id: str, request: Request, actor: MemoryAssigner = Depends(resume_actor)):
        flag()
        body = await _body(request, expected_revision=_revision(1))
        if _PROFILE.fullmatch(profile_id) is None:
            raise HTTPException(422, detail=_INVALID)
        await run(ledger().resume_voice_attribution, profile_id, body["expected_revision"], actor)
        return {"resumed": True}

    app.include_router(router)
