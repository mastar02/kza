"""
Snapshots de GPUs y servicios systemd. Detecta plataforma:
- Linux con nvidia-smi → datos reales
- Resto → datos vacíos (caller decide fallback a mocks)

Distingue tipos de fallo:
- "no instalado" / "no estamos en el server" → INFO/None silencioso
- "tool present pero falló" (timeout, exit≠0, parseo) → ERROR con causa
"""

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Servicios systemd --user que monitoreamos. Ver MEMORY KZA — sub-rango 9500-9599.
KZA_SERVICES = ["kza-voice", "kza-llm-ik", "vllm-shared", "chromadb",
                "home-assistant", "ma1260-bridge"]

# cuda:0 = STT/TTS/SpeakerID/Emotion/Embeddings; cuda:1 = vLLM 7B compartido.
# Idx >= 2 intencionalmente "—" — el server actual tiene 2 GPUs.
GPU_ROLES = {
    0: "STT • TTS • SpeakerID • Emotion • Embeddings",
    1: "vLLM Qwen 7B AWQ (compartido)",
}


def _have_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _have_systemctl() -> bool:
    return shutil.which("systemctl") is not None


def gpu_snapshot() -> Optional[list[dict]]:
    """Parsea nvidia-smi. None si no instalado o falla irrecuperable.

    Distingue: no-instalado (None silencioso), timeout (ERROR — driver wedged),
    rc≠0 (ERROR con stderr), parse error (ERROR — version mismatch?).
    """
    if not _have_nvidia_smi():
        return None
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.5, check=True,
        )
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi timeout 2.5s — driver wedged?")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi rc={e.returncode}: {e.stderr.strip()!r}")
        return None
    except OSError as e:
        logger.error(f"nvidia-smi launch failed: {e}")
        return None

    try:
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                logger.warning(f"nvidia-smi malformed line skipped: {line!r}")
                continue
            idx = int(parts[0])
            gpus.append({
                "id": idx,
                "name": f"{parts[1]} (cuda:{idx})",
                "role": GPU_ROLES.get(idx, "—"),
                "util": int(float(parts[2])),
                "vramUsed": round(float(parts[3]) / 1024, 1),
                "vramTotal": round(float(parts[4]) / 1024, 1),
                "temp": int(float(parts[5])),
                "power": int(float(parts[6])),
                "procs": _gpu_procs(idx),
            })
        return gpus
    except (ValueError, IndexError) as e:
        logger.error(f"nvidia-smi parse error (version mismatch?): {e}")
        return None


def _gpu_procs(gpu_index: int) -> list[dict]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"nvidia-smi --query-compute-apps id={gpu_index} timeout")
        return []
    except subprocess.CalledProcessError as e:
        logger.warning(f"nvidia-smi compute-apps rc={e.returncode}")
        return []
    except OSError:
        return []
    procs = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            procs.append({"name": parts[0], "vram": round(float(parts[1]) / 1024, 1)})
        except ValueError:
            continue
    return procs


def services_snapshot() -> Optional[list[dict]]:
    """systemctl --user show de cada servicio KZA.

    Detecta "fuera del server" via `systemctl --user list-units` ejecutado UNA
    vez (no via primer servicio). Servicios faltantes individuales no abortan
    el snapshot — devuelven entries con status=missing.
    """
    if not _have_systemctl():
        return None
    if not _systemctl_user_works():
        return None
    out = []
    for svc in KZA_SERVICES:
        info = _systemctl_show(svc)
        if info is None:
            out.append({
                "name": svc, "status": "missing", "uptime": "—",
                "mem": "—", "cpu": "—", "pid": 0,
            })
        else:
            out.append(info)
    return out


def _systemctl_user_works() -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "list-units", "--no-pager", "--no-legend"],
            capture_output=True, text=True, timeout=1.5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _systemctl_show(svc: str) -> Optional[dict]:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", svc,
             "--property=ActiveState,MainPID,ExecMainStartTimestamp,MemoryCurrent"],
            capture_output=True, text=True, timeout=1.5,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"systemctl show {svc} failed: {e}")
        return None
    if r.returncode != 0:
        return None
    props = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v
    if not props.get("ActiveState"):
        return None
    mem_raw = props.get("MemoryCurrent", "").strip()
    try:
        mem_bytes = int(mem_raw) if mem_raw else 0
    except ValueError:
        mem_bytes = 0
    pid_raw = props.get("MainPID", "").strip()
    try:
        pid = int(pid_raw) if pid_raw else 0
    except ValueError:
        pid = 0
    return {
        "name": svc,
        "status": props["ActiveState"],
        "uptime": props.get("ExecMainStartTimestamp", "—") or "—",
        "mem": _fmt_bytes(mem_bytes),
        "cpu": "—",
        "pid": pid,
    }


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "—"
    n_f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n_f < 1024:
            return f"{n_f:.1f} {unit}"
        n_f /= 1024
    return f"{n_f:.1f} TB"
