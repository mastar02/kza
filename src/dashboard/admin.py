"""
Endpoints mutativos del dashboard: users CRUD + voice enrollment, alerts
ack/dismiss, service restart. Todos protegidos por `require_admin`.

Mantiene la separación con `observability.py`: ese es read-only, este es write.
"""

import asyncio
import io
import logging
import re
import shutil
import subprocess
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.dashboard.auth import (
    auth_configured,
    login_response,
    logout_response,
    require_admin,
)

logger = logging.getLogger(__name__)


# Servicios que un admin puede reiniciar via API. Allowlist explícita —
# nunca acepta input arbitrario en systemctl.
RESTARTABLE_SERVICES = frozenset({
    "kza-voice", "kza-llm-ik", "vllm-shared", "chromadb",
})


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
) -> None:
    """Registrar /api/admin/*. Llamar antes del mount de StaticFiles."""

    if not auth_configured():
        logger.warning(
            "KZA_DASHBOARD_TOKEN no seteado — endpoints /api/admin/* "
            "ESTÁN ABIERTOS sin auth. Setealo en /home/kza/secrets/.env"
        )

    router = APIRouter(prefix="/api/admin", tags=["admin"])

    # ---------------- Auth helpers (sin require_admin) ----------------

    @router.post("/auth/login")
    async def login(body: LoginBody):
        return login_response(body.token)

    @router.post("/auth/logout", dependencies=[Depends(require_admin)])
    async def logout():
        return logout_response()

    @router.get("/auth/whoami", dependencies=[Depends(require_admin)])
    async def whoami():
        return {"ok": True, "auth_configured": auth_configured()}

    # ---------------- Users ----------------

    @router.post("/users", dependencies=[Depends(require_admin)])
    async def create_user(body: CreateUserBody):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        from src.users.user_manager import PermissionLevel
        try:
            level = PermissionLevel[body.permission_level.upper()]
        except KeyError:
            raise HTTPException(400, f"permission_level inválido: {body.permission_level}")
        user, msg = user_manager.add_user(name=body.name, permission_level=level)
        if user is None:
            raise HTTPException(409, msg)
        return {"ok": True, "user_id": user.user_id, "name": user.name, "message": msg}

    @router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
    async def delete_user(user_id: str):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        ok, msg = user_manager.remove_user(user_id)
        if not ok:
            raise HTTPException(404, msg)
        return {"ok": True, "message": msg}

    @router.put("/users/{user_id}/permissions", dependencies=[Depends(require_admin)])
    async def update_user_permissions(user_id: str, body: UpdatePermsBody):
        if user_manager is None:
            raise HTTPException(503, "user_manager no inyectado")
        from src.users.user_manager import PermissionLevel
        user = user_manager.get_user(user_id)
        if user is None:
            raise HTTPException(404, f"user {user_id} no encontrado")
        try:
            level = PermissionLevel[body.permission_level.upper()]
        except KeyError:
            raise HTTPException(400, f"permission_level inválido: {body.permission_level}")
        user.permission_level = level
        user_manager._save()
        return {"ok": True, "permission_level": level.name}

    @router.post("/users/{user_id}/enroll", dependencies=[Depends(require_admin)])
    async def enroll_voice(
        user_id: str,
        samples: list[UploadFile] = File(..., description="WAV mono PCM, ≥3 muestras"),
    ):
        if user_manager is None or speaker_identifier is None:
            raise HTTPException(503, "user_manager o speaker_identifier no inyectados")
        if len(samples) < 3:
            raise HTTPException(400, "se requieren ≥3 muestras de audio")
        user = user_manager.get_user(user_id)
        if user is None:
            raise HTTPException(404, f"user {user_id} no encontrado")
        try:
            audio_arrays = [await _decode_wav(s) for s in samples]
        except ValueError as e:
            raise HTTPException(400, f"audio inválido: {e}")
        try:
            embedding = await asyncio.to_thread(
                speaker_identifier.create_enrollment_embedding, audio_arrays
            )
        except Exception as e:
            logger.error(f"create_enrollment_embedding falló: {e}")
            raise HTTPException(500, f"enrollment falló: {e}")
        user.voice_embedding = embedding
        user_manager._save()
        return {
            "ok": True, "user_id": user_id,
            "samples": len(samples),
            "embedding_dim": int(embedding.shape[0]),
        }

    # ---------------- Alerts ----------------

    @router.post("/alerts/{alert_id}/ack", dependencies=[Depends(require_admin)])
    async def ack_alert(alert_id: str):
        if alert_manager is None:
            raise HTTPException(503, "alert_manager no inyectado")
        alert = alert_manager.get_alert(alert_id)
        if alert is None:
            raise HTTPException(404, f"alert {alert_id} no encontrada")
        from datetime import datetime, timezone
        alert.processed = True
        alert.processed_at = datetime.now(timezone.utc)
        return {"ok": True, "alert_id": alert_id}

    @router.delete("/alerts/{alert_id}", dependencies=[Depends(require_admin)])
    async def dismiss_alert(alert_id: str):
        if alert_manager is None:
            raise HTTPException(503, "alert_manager no inyectado")
        alert = alert_manager.get_alert(alert_id)
        if alert is None:
            raise HTTPException(404, f"alert {alert_id} no encontrada")
        # AlertManager mantiene historial inmutable; "dismiss" = mark processed
        alert.processed = True
        return {"ok": True, "alert_id": alert_id, "dismissed": True}

    # ---------------- Services ----------------

    @router.post("/services/{name}/restart", dependencies=[Depends(require_admin)])
    async def restart_service(name: str):
        if name not in RESTARTABLE_SERVICES:
            raise HTTPException(403, f"servicio '{name}' no está en allowlist")
        if not shutil.which("systemctl"):
            raise HTTPException(503, "systemctl no disponible (no estás en server)")
        try:
            r = await asyncio.to_thread(
                subprocess.run,
                ["systemctl", "--user", "restart", name],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(504, f"systemctl restart {name} timeout")
        if r.returncode != 0:
            raise HTTPException(500, f"restart falló rc={r.returncode}: {r.stderr.strip()}")
        return {"ok": True, "service": name}

    app.include_router(router)


# ---------------- Helpers ----------------

async def _decode_wav(upload: UploadFile) -> np.ndarray:
    """Decodifica un WAV upload → numpy float32 mono 16kHz.

    El frontend envía PCM 16kHz mono, pero validamos por las dudas. Soporta
    16-bit int (lo más común desde MediaRecorder/AudioContext).
    """
    raw = await upload.read()
    if not raw:
        raise ValueError("archivo vacío")
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise ValueError(f"no es un WAV válido: {e}") from e
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix a mono
    if sr != 16000:
        # Resample lineal simple — para enrollment queremos exacto 16kHz que
        # es lo que ECAPA espera; si el browser entregó 48kHz lo bajamos.
        ratio = 16000 / sr
        n_out = int(len(data) * ratio)
        idx = np.linspace(0, len(data) - 1, n_out).astype(np.int64)
        data = data[idx]
    return data.astype(np.float32, copy=False)
