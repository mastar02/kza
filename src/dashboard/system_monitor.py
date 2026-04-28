"""
Snapshots de GPUs y servicios systemd. Detecta plataforma:
- Linux con nvidia-smi → datos reales
- Resto → datos vacíos (caller decide fallback a mocks)
"""

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Servicios systemd --user que monitoreamos (ver MEMORY.md)
KZA_SERVICES = ["kza-voice", "kza-llm-ik", "vllm-shared", "chromadb",
                "home-assistant", "ma1260-bridge"]

# Mapping de procesos GPU → rol legible (ver mocks)
GPU_ROLES = {
    0: "STT • TTS • SpeakerID • Emotion • Embeddings",
    1: "vLLM Qwen 7B AWQ (compartido)",
}


def _have_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def gpu_snapshot() -> Optional[list[dict]]:
    """Parsea nvidia-smi. Devuelve None si no disponible."""
    if not _have_nvidia_smi():
        return None
    try:
        # Una línea por GPU: index, name, util%, vram_used MiB, vram_total MiB, temp, power_W
        q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.5, check=True,
        ).stdout.strip()
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            idx = int(parts[0])
            gpu = {
                "id": idx,
                "name": f"{parts[1]} (cuda:{idx})",
                "role": GPU_ROLES.get(idx, "—"),
                "util": int(float(parts[2])),
                "vramUsed": round(float(parts[3]) / 1024, 1),
                "vramTotal": round(float(parts[4]) / 1024, 1),
                "temp": int(float(parts[5])),
                "power": int(float(parts[6])),
                "procs": _gpu_procs(idx),
            }
            gpus.append(gpu)
        return gpus
    except Exception as e:
        logger.debug(f"nvidia-smi failed: {e}")
        return None


def _gpu_procs(gpu_index: int) -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        ).stdout.strip()
        procs = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            procs.append({"name": parts[0], "vram": round(float(parts[1]) / 1024, 1)})
        return procs
    except Exception:
        return []


def services_snapshot() -> Optional[list[dict]]:
    """systemctl --user status para servicios KZA. None si no disponible."""
    if not shutil.which("systemctl"):
        return None
    out = []
    for svc in KZA_SERVICES:
        info = _systemctl_show(svc)
        if info is None:
            # Si ningún servicio responde, asumimos que estamos fuera del server.
            if not out:
                return None
            continue
        out.append(info)
    return out or None


def _systemctl_show(svc: str) -> Optional[dict]:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", svc,
             "--property=ActiveState,MainPID,ExecMainStartTimestamp,MemoryCurrent"],
            capture_output=True, text=True, timeout=1.5,
        )
        if r.returncode != 0:
            return None
        props = {}
        for line in r.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                props[k] = v
        active = props.get("ActiveState", "unknown")
        mem_bytes = int(props.get("MemoryCurrent", "0") or 0)
        return {
            "name": svc,
            "status": active,
            "uptime": props.get("ExecMainStartTimestamp", "—") or "—",
            "mem": _fmt_bytes(mem_bytes),
            "cpu": "—",
            "pid": int(props.get("MainPID", "0") or 0),
        }
    except Exception:
        return None


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
