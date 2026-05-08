"""
Snapshots de GPUs y servicios para el dashboard.

Servicios: registry mixto. Cada servicio define su método de probe:
- "systemctl_user": `systemctl --user show <name>` (servicios bajo `kza`)
- "http": GET a una URL → 200 == active. Para servicios fuera de la sesión
  systemd del user (HA root, vLLM bajo usuario infra, containers).
- "in_process": siempre activo si el código del dashboard está corriendo
  (chromadb se carga in-process en kza-voice).

GPUs: nvidia-smi parsing con narrow excepts (timeout = driver wedged, etc).
"""

import logging
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# cuda:0 = STT/TTS/SpeakerID/Emotion/Embeddings; cuda:1 = vLLM 7B compartido.
GPU_ROLES = {
    0: "STT • TTS • SpeakerID • Emotion • Embeddings",
    1: "vLLM Qwen 7B AWQ (compartido)",
}


# Service probe registry. Cada entry: (probe_kind, probe_args)
# Las URLs pueden venir de env vars en runtime para reflejar el deploy real.
def _service_probes() -> list[dict]:
    ha_url = os.environ.get("HOME_ASSISTANT_URL", "http://192.168.1.100:8123")
    return [
        {"name": "kza-voice", "kind": "systemctl_user"},
        {"name": "kza-llm-ik", "kind": "systemctl_user",
         "http_probe": "http://127.0.0.1:8200/v1/models"},
        {"name": "kza-llm-fast", "kind": "systemctl_user",
         "http_probe": "http://127.0.0.1:8101/v1/models",
         "role": "ik_llama Qwen 7B Q4 — fast router (:8101)"},
        {"name": "home-assistant", "kind": "http",
         "url": f"{ha_url.rstrip('/')}/api/",
         "role": "Home Assistant — control de domótica"},
        {"name": "chromadb", "kind": "in_process",
         "role": "ChromaDB — vector store (in-process en kza-voice)"},
        {"name": "ma1260-bridge", "kind": "systemctl_user",
         "role": "Amplificador MA1260 (serial bridge)"},
    ]


def gpu_snapshot() -> Optional[list[dict]]:
    """Parsea nvidia-smi. None si no instalado o falla irrecuperable."""
    if not shutil.which("nvidia-smi"):
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
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
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
    """Probe registry mixto. None si no estamos en server (no systemctl ni HTTP)."""
    have_systemctl = bool(shutil.which("systemctl"))
    out = []
    for probe in _service_probes():
        info = _probe_service(probe, have_systemctl)
        out.append(info)
    # Si NINGÚN probe externo (systemctl/HTTP) responde activo, asumimos dev box.
    # `in_process` no cuenta — siempre da activo y no aporta signal.
    external = [s for s in out if s.get("mem") != "in-process"]
    if external and all(s["status"] in ("missing", "unreachable") for s in external):
        return None
    return out


def _probe_service(probe: dict, have_systemctl: bool) -> dict:
    name = probe["name"]
    kind = probe["kind"]
    role = probe.get("role", "")

    if kind == "in_process":
        return {"name": name, "status": "active", "uptime": "—",
                "mem": "in-process", "cpu": "—", "pid": 0, "role": role}

    if kind == "systemctl_user" and have_systemctl:
        info = _systemctl_show(name)
        if info is not None:
            info["role"] = role or info.get("role", "")
            # Probe HTTP secundario si está definido (ej. kza-llm-ik :8200)
            http = probe.get("http_probe")
            if http and info["status"] == "active":
                info["http_ok"] = _http_ok(http)
            return info
        # systemctl no encontró el servicio → puede ser que no esté en este host
        return {"name": name, "status": "missing", "uptime": "—",
                "mem": "—", "cpu": "—", "pid": 0, "role": role}

    if kind == "http":
        ok = _http_ok(probe["url"])
        return {
            "name": name,
            "status": "active" if ok else "unreachable",
            "uptime": "—", "mem": "—", "cpu": "—", "pid": 0,
            "role": role, "http_ok": ok, "url": probe["url"],
        }

    return {"name": name, "status": "missing", "uptime": "—",
            "mem": "—", "cpu": "—", "pid": 0, "role": role}


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # 2xx + algunos 4xx (HA devuelve 401 sin token, está vivo)
            return r.status < 500
    except urllib.error.HTTPError as e:
        # 401/403 = el servicio responde, está vivo
        return 400 <= e.code < 500
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
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
