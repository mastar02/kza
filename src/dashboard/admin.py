"""
Endpoints mutativos del dashboard: users CRUD + voice enrollment, alerts
ack/dismiss, service restart. Todos protegidos por `require_admin`.

Mantiene la separación con `observability.py`: ese es read-only, este es write.
"""

import asyncio
import io
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import partial

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.core.owned_work import OwnedBlockingWork
from src.dashboard.native_work import NativeAuthorityRetired, ensure_admin_work, require_current, run_admin_work
from src.memory.producer_context import model_admission

from src.dashboard.auth import (
    DashboardPrincipal,
    auth_configured,
    current_lease,
    dashboard_authority,
    login_response,
    logout_response,
    require_admin,
    require_dashboard_origin,
    require_dashboard_session,
    session_guard,
)

logger = logging.getLogger(__name__)


# Servicios que un admin puede reiniciar via API. Allowlist explícita —
# nunca acepta input arbitrario en systemctl.
# Alineada con el registry de src/dashboard/system_monitor.py:_service_probes.
# Tenía "vllm-shared" (servicio que dejó de existir en el reorg de endpoints
# 2026-05-07, 269ed4e) y le faltaba "kza-llm-fast": la API podía reiniciar un
# fantasma pero no el fast router real. Corregido 2026-08-02.
RESTARTABLE_SERVICES = frozenset({
    "kza-voice", "kza-llm-ik", "kza-llm-fast", "chromadb",
})


_MAX_LEGACY_SAMPLES = 16
_MAX_AUDIO_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class LegacyEnrollmentReceipt:
    """Keep raw audio and vectors out of the returned administrative result."""

    user_id: str
    samples: int
    embedding_dim: int


class LoginBody(BaseModel):
    token: str


class CreateUserBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    permission_level: str = Field(default="adult")  # guest|child|teen|adult|admin


class UpdatePermsBody(BaseModel):
    permission_level: str


def register_admin_routes(
    app: FastAPI,
    *,
    user_manager=None,
    speaker_identifier=None,
    alert_manager=None,
    voice_registry=None,
    admin_work: OwnedBlockingWork | None = None,
    model_lock=None,
) -> None:
    """Registrar /api/admin/*. Llamar antes del mount de StaticFiles."""
    work = ensure_admin_work(app, admin_work)
    model_lock = model_lock if model_lock is not None else threading.Lock()

    if not auth_configured(getattr(app.state, "dashboard_authority", None)):
        logger.warning(
            "Dashboard credential is not configured; protected routes return 503"
        )

    router = APIRouter(prefix="/api/admin", tags=["admin"])

    # ---------------- Auth helpers (sin require_admin) ----------------

    @router.post("/auth/login", dependencies=[Depends(require_dashboard_origin)])
    async def login(body: LoginBody, request: Request):
        authority = dashboard_authority(request, required=True)
        return await login_response(body.token, request, authority=authority, secure=request.url.scheme == "https")

    @router.post("/auth/logout")
    async def logout(request: Request, _principal: DashboardPrincipal = Depends(require_dashboard_session)):
        lease = current_lease(request)
        if lease is None:
            raise HTTPException(status_code=403, detail="dashboard_session_required")
        return await logout_response(lease, authority=dashboard_authority(request))

    @router.get("/auth/whoami")
    async def whoami(request: Request, principal: DashboardPrincipal = Depends(require_dashboard_session)):
        return JSONResponse(content={"ok": True,
            "auth_configured": auth_configured(dashboard_authority(request)),
            "subject": principal.subject, "role": principal.role.value,
            "permissions": sorted(principal.permissions), "expires_at": principal.expires_at,
            "personal_scope": principal.user_id is not None},
            headers={"Cache-Control": "no-store"})

    # ---------------- Users ----------------

    @router.post("/users", dependencies=[Depends(require_admin)])
    async def create_user(body: CreateUserBody, request: Request):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        from src.users.user_manager import PermissionLevel
        try:
            level = PermissionLevel[body.permission_level.upper()]
        except KeyError:
            raise HTTPException(400, f"permission_level inválido: {body.permission_level}")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            user, message = await run_admin_work(work, current, user_manager.add_user,
                name=body.name, permission_level=level, before_save=partial(require_current, current))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "user_catalog_unavailable") from error
        if user is None:
            raise HTTPException(409, message)
        return {"ok": True, "user_id": user.user_id, "name": user.name, "message": message}

    @router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
    async def delete_user(user_id: str, request: Request):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            deleted, message = await run_admin_work(work, current, user_manager.remove_user, user_id,
                before_save=partial(require_current, current))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "user_catalog_unavailable") from error
        if not deleted:
            raise HTTPException(404, message)
        return {"ok": True, "message": message}

    @router.put("/users/{user_id}/permissions", dependencies=[Depends(require_admin)])
    async def update_user_permissions(user_id: str, body: UpdatePermsBody, request: Request):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        from src.users.user_manager import PermissionLevel
        try:
            level = PermissionLevel[body.permission_level.upper()]
        except KeyError:
            raise HTTPException(400, f"permission_level inválido: {body.permission_level}")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            updated, message = await run_admin_work(work, current, user_manager.update_user, user_id,
                permission_level=level, before_save=partial(require_current, current))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(503, "user_catalog_unavailable") from error
        if not updated:
            raise HTTPException(404, message)
        return {"ok": True, "permission_level": level.name}

    if voice_registry is not None:
        @router.post("/users/{user_id}/enroll", dependencies=[Depends(require_admin)])
        async def review_voice_profile(user_id: str):
            raise HTTPException(409,detail='voice_profile_review_required')
    else:
        @router.post("/users/{user_id}/enroll", dependencies=[Depends(require_admin)])
        async def enroll_voice(
            user_id: str, request: Request,
            samples: list[UploadFile] = File(..., description="WAV mono PCM, entre 3 y 16 muestras"),
        ):
            current = session_guard(request, frozenset({'admin:manage'}))
            if user_manager is None or speaker_identifier is None:
                raise HTTPException(503, "user_manager o speaker_identifier no inyectados")
            if getattr(user_manager, "voice_review_managed", False) is True:
                raise HTTPException(409, "voice_profile_review_required")
            if not 3 <= len(samples) <= _MAX_LEGACY_SAMPLES:
                raise HTTPException(400, "se requieren entre 3 y 16 muestras de audio")
            payloads = []
            try:
                for sample in samples:
                    require_current(current)
                    payload = await sample.read(_MAX_AUDIO_BYTES + 1)
                    if len(payload) > _MAX_AUDIO_BYTES:
                        raise HTTPException(413, "audio_sample_too_large")
                    payloads.append(payload)
                receipt = await run_admin_work(work, current, _enroll_native,
                    user_manager, speaker_identifier, model_lock, user_id, tuple(payloads), current)
            except NativeAuthorityRetired as error:
                raise HTTPException(403, "dashboard_authority_retired") from error
            except ValueError as error:
                raise HTTPException(400, "audio_enrollment_invalid") from error
            except (OSError, RuntimeError) as error:
                raise HTTPException(503, "enrollment_unavailable") from error
            finally:
                payloads.clear()
                payloads = payload = None
            if receipt is None:
                raise HTTPException(404, "user_not_found")
            return {"ok": True, "user_id": receipt.user_id, "samples": receipt.samples,
                    "embedding_dim": receipt.embedding_dim}

    # ---------------- Alerts ----------------

    @router.post("/alerts/{alert_id}/ack", dependencies=[Depends(require_admin)])
    async def ack_alert(alert_id: str, request: Request):
        if alert_manager is None:
            raise HTTPException(503, "alert_manager no inyectado")
        from datetime import datetime, timezone
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            found = await alert_manager.mark_processed(alert_id,
                before_publish=partial(require_current, current),
                timestamp_factory=lambda: datetime.now(timezone.utc))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        if not found:
            raise HTTPException(404, "alert_not_found")
        return {"ok": True, "alert_id": alert_id}

    @router.delete("/alerts/{alert_id}", dependencies=[Depends(require_admin)])
    async def dismiss_alert(alert_id: str, request: Request):
        if alert_manager is None:
            raise HTTPException(503, "alert_manager no inyectado")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            found = await alert_manager.mark_processed(alert_id, update_timestamp=False,
                before_publish=partial(require_current, current))
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        if not found:
            raise HTTPException(404, "alert_not_found")
        return {"ok": True, "alert_id": alert_id, "dismissed": True}

    # ---------------- Services ----------------

    @router.post("/services/{name}/restart", dependencies=[Depends(require_admin)])
    async def restart_service(name: str, request: Request):
        if name not in RESTARTABLE_SERVICES:
            raise HTTPException(403, f"servicio '{name}' no está en allowlist")
        if not shutil.which("systemctl"):
            raise HTTPException(503, "systemctl no disponible (no estás en server)")
        current = session_guard(request, frozenset({'admin:manage'}))
        try:
            result = await run_admin_work(work, current, subprocess.run,
                ["systemctl", "--user", "restart", name], capture_output=True, text=True, timeout=15)
        except NativeAuthorityRetired as error:
            raise HTTPException(403, "dashboard_authority_retired") from error
        except subprocess.TimeoutExpired:
            raise HTTPException(504, detail={"reason": "restart_timeout", "effect": "uncertain"})
        except OSError as error:
            raise HTTPException(503, detail={"reason": "restart_process_unavailable", "effect": "uncertain"}) from error
        except RuntimeError as error:
            raise HTTPException(503, "administration_unavailable") from error
        if result.returncode != 0:
            raise HTTPException(500, detail={"reason": "restart_command_failed", "effect": "uncertain",
                                           "returncode": result.returncode})
        return {"ok": True, "service": name, "command_completed": True}

    app.include_router(router)


# ---------------- Helpers ----------------

async def _decode_wav(upload: UploadFile) -> np.ndarray:
    """Decode a bounded legacy sample outside the event loop."""
    raw = await upload.read(_MAX_AUDIO_BYTES + 1)
    if len(raw) > _MAX_AUDIO_BYTES:
        raise ValueError("audio_sample_too_large")
    try:
        return await asyncio.to_thread(_decode_wav_bytes, raw)
    finally:
        raw = None


def _decode_wav_bytes(raw: bytes) -> np.ndarray:
    """Validate WAV dimensions before bounded allocation and independent mono conversion."""
    data = None
    try:
        if not raw or len(raw) > _MAX_AUDIO_BYTES:
            raise ValueError("audio_sample_invalid")
        import soundfile as sf
        try:
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                rate = audio.samplerate
                if (audio.format not in {"WAV", "WAVEX", "RF64"}
                        or not 8000 <= rate <= 192000 or audio.channels not in (1, 2)
                        or not 0 < audio.frames <= rate * 30):
                    raise ValueError("audio_sample_invalid")
                data = audio.read(frames=rate * 30 + 1, dtype="float32", always_2d=False)
        except Exception:
            raise ValueError("audio_sample_invalid") from None
        if not data.size or len(data) > rate * 30 or not np.isfinite(data).all():
            raise ValueError("audio_sample_invalid")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != 16000:
            count = int(len(data) * 16000 / rate)
            indices = np.linspace(0, len(data) - 1, count).astype(np.int64)
            data = data[indices]
        return data.astype(np.float32, copy=True)
    finally:
        raw = data = audio = None


def _enroll_native(user_manager, speaker, model_lock, user_id: str,
                   payloads: tuple[bytes, ...], current) -> LegacyEnrollmentReceipt | None:
    """Own decode, model admission and guarded legacy publication as one job."""
    arrays, embedding = [], None
    validator = partial(require_current, current)
    try:
        validator()
        if user_manager.get_user(user_id, before_read=validator) is None:
            return None
        for payload in payloads:
            validator()
            arrays.append(_decode_wav_bytes(payload))
        with model_admission(model_lock, validator):
            embedding = np.array(speaker.create_enrollment_embedding(arrays), dtype=np.float32, copy=True)
        if embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all():
            raise ValueError("voice_embedding_invalid")
        if not user_manager.set_voice_embedding(user_id, embedding, before_save=validator):
            return None
        return LegacyEnrollmentReceipt(user_id, len(payloads), int(embedding.shape[0]))
    finally:
        arrays.clear()
        arrays = embedding = payloads = payload = validator = current = None
