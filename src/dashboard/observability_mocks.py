"""
Mock data para el dashboard de observability.

Mismas shapes que `frontend/obs/src/mocks.jsx`. Sirve cuando `USE_MOCKS=True` o
cuando un servicio real no está disponible (fallback).
"""

ZONES = [
    {"id": "sala", "name": "Sala", "icon": "◰",
     "user": {"id": "u_marco", "name": "Marco", "present": True, "ble_rssi": -52},
     "lastUtterance": {"text": "subí dos grados la calefacción", "ts": "14:32:08", "user": "Marco"},
     "audioState": "speaking", "volume": 42, "ma1260_zone": "A",
     "spotify": {"playing": True, "song": "Sun Setters", "artist": "Khruangbin", "progress": 142, "total": 218}},
    {"id": "dormitorio", "name": "Dormitorio", "icon": "▢",
     "user": {"id": "u_lu", "name": "Lucía", "present": True, "ble_rssi": -64},
     "lastUtterance": {"text": "apagá la luz del techo", "ts": "14:28:55", "user": "Lucía"},
     "audioState": "idle", "volume": 18, "ma1260_zone": "B", "spotify": {"playing": False}},
    {"id": "cocina", "name": "Cocina", "icon": "◐", "user": None,
     "lastUtterance": {"text": "temporizador de quince minutos", "ts": "13:55:12", "user": "Marco"},
     "audioState": "idle", "volume": 0, "ma1260_zone": "C", "spotify": {"playing": False}},
    {"id": "estudio", "name": "Estudio", "icon": "◧",
     "user": {"id": "u_marco", "name": "Marco", "present": False, "ble_rssi": None},
     "lastUtterance": {"text": "cuántos pull requests abiertos tengo", "ts": "12:11:41", "user": "Marco"},
     "audioState": "listening", "volume": 30, "ma1260_zone": "D",
     "spotify": {"playing": True, "song": "Lost in the Dream", "artist": "The War on Drugs", "progress": 88, "total": 354}},
    {"id": "patio", "name": "Patio", "icon": "◔", "user": None, "lastUtterance": None,
     "audioState": "idle", "volume": 0, "ma1260_zone": "E", "spotify": {"playing": False}},
]

LLM_ENDPOINTS = [
    {"id": "vllm-7b", "name": "vLLM Qwen 7B AWQ", "url": "http://localhost:8100",
     "priority": 1, "role": "fast / routing", "state": "healthy",
     "tps": 82, "ttft_ms": 142, "last_check": "14:32:31",
     "failures_7d": {"timeout": 1, "billing": 0, "rate_limit": 0, "idle": 2},
     "cooldown_ends": None},
    {"id": "ik-30b-cpu", "name": "ik_llama Qwen3-30B-A3B", "url": "http://localhost:8200",
     "priority": 2, "role": "slow / reasoning", "state": "cooldown",
     "tps": 63, "ttft_ms": 1840, "last_check": "14:30:02",
     "failures_7d": {"timeout": 4, "billing": 0, "rate_limit": 0, "idle": 0},
     "cooldown_ends": "14:35:02", "cooldown_step": "5m"},
    {"id": "embed-bge", "name": "BGE-M3 (cuda:0)", "url": "local",
     "priority": 0, "role": "embeddings", "state": "healthy",
     "tps": None, "ttft_ms": 8, "last_check": "14:32:30",
     "failures_7d": {"timeout": 0, "billing": 0, "rate_limit": 0, "idle": 0},
     "cooldown_ends": None},
]

CONVERSATIONS = [
    {"id": "turn_8821", "ts": "14:32:08", "user": "Marco", "zone": "sala", "path": "fast",
     "stt": "subí dos grados la calefacción", "intent": "climate.set_temperature",
     "target": "climate.living_room", "args": {"temperature": 22.5},
     "tts": "Subí la sala a veintidós con cinco.", "latency_ms": 192, "success": True},
    {"id": "turn_8820", "ts": "14:28:55", "user": "Lucía", "zone": "dormitorio", "path": "fast",
     "stt": "apagá la luz del techo", "intent": "light.turn_off",
     "target": "light.bedroom_ceiling", "args": {}, "tts": "Listo.",
     "latency_ms": 178, "success": True},
    {"id": "turn_8819", "ts": "14:21:02", "user": "Marco", "zone": "estudio", "path": "slow",
     "stt": "cuántos pull requests abiertos tengo en el repo de kza",
     "intent": "reasoning.dev_query", "target": None,
     "args": {"query": "github_open_prs", "repo": "kza"},
     "tts": "Tenés ocho pull requests abiertos.", "latency_ms": 12480, "success": True},
]

HA_ENTITIES = [
    {"id": "light.living_room_ceiling", "domain": "light", "name": "Luz techo sala",
     "state": "on", "attrs": {"brightness": 180}, "score": 0.94, "lastSeen": "14:32"},
    {"id": "climate.living_room", "domain": "climate", "name": "Calefacción sala",
     "state": "heat", "attrs": {"current": 22.4, "target": 22.5}, "score": 0.97, "lastSeen": "14:32"},
    {"id": "media_player.living_room", "domain": "media", "name": "Sonos sala",
     "state": "playing", "attrs": {"volume": 42}, "score": 0.95, "lastSeen": "14:30"},
]

HA_ACTIONS = [
    {"id": "act_4421", "ts": "14:32:09", "idem": "idm_8821_climate", "user": "Marco",
     "service": "climate.set_temperature", "target": "climate.living_room",
     "args": "{temperature:22.5}", "ok": True, "lat_ms": 22},
    {"id": "act_4420", "ts": "14:28:56", "idem": "idm_8820_light", "user": "Lucía",
     "service": "light.turn_off", "target": "light.bedroom_ceiling",
     "args": "{}", "ok": True, "lat_ms": 14},
]

USERS = [
    {"id": "u_marco", "name": "Marco", "samples": 24, "lastEnroll": "2026-03-12",
     "emotions": {"neutral": 62, "happy": 18, "focused": 14, "frustrated": 4, "tired": 2},
     "topCommands": [{"cmd": "climate.set_temperature", "n": 412}],
     "permissions": {"climate": True, "lights": True, "security": True, "music": True, "scenes": True},
     "pca": [[-0.4, 0.2], [-0.42, 0.18]]},
    {"id": "u_lu", "name": "Lucía", "samples": 18, "lastEnroll": "2026-04-02",
     "emotions": {"neutral": 58, "happy": 22, "calm": 14, "tired": 4, "frustrated": 2},
     "topCommands": [{"cmd": "light.turn_off", "n": 245}],
     "permissions": {"climate": True, "lights": True, "security": False, "music": True, "scenes": True},
     "pca": [[0.32, -0.18], [0.34, -0.16]]},
]

ALERTS = [
    {"id": "al_201", "ts": "14:31:55", "priority": "warn", "type": "pattern", "zone": "sala",
     "title": "Temperatura objetivo cambiada 4 veces en 2min",
     "body": "Marco modificó climate.living_room múltiples veces.", "acked": False},
    {"id": "al_200", "ts": "14:18:21", "priority": "critical", "type": "security", "zone": "patio",
     "title": "Movimiento detectado con todos ausentes",
     "body": "binary_sensor.motion_patio activo.", "acked": False},
]

GPUS = [
    {"id": 0, "name": "RTX 3070 (cuda:0)",
     "role": "STT • TTS • SpeakerID • Emotion • Embeddings",
     "util": 64, "vramUsed": 7.4, "vramTotal": 8, "temp": 71, "power": 168,
     "procs": [{"name": "whisper-v3-turbo", "vram": 2.8},
               {"name": "kokoro-tts", "vram": 1.4},
               {"name": "bge-m3-embeddings", "vram": 1.2}]},
    {"id": 1, "name": "RTX 3070 (cuda:1)", "role": "vLLM Qwen 7B AWQ (compartido)",
     "util": 38, "vramUsed": 6.1, "vramTotal": 8, "temp": 64, "power": 142,
     "procs": [{"name": "vllm-qwen-7b-awq", "vram": 6.1}]},
]

SERVICES = [
    {"name": "kza-voice", "status": "active", "uptime": "6d 14h", "mem": "2.1 GB", "cpu": "8%", "pid": 18221},
    {"name": "kza-llm-ik", "status": "active", "uptime": "6d 14h", "mem": "38.4 GB", "cpu": "64%", "pid": 18244},
    {"name": "vllm-shared", "status": "active", "uptime": "12d 02h", "mem": "6.1 GB", "cpu": "12%", "pid": 9802},
    {"name": "home-assistant", "status": "active", "uptime": "21d 08h", "mem": "684 MB", "cpu": "3%", "pid": 4112},
]
