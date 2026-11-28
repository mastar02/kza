#!/usr/bin/env bash
set -eu

once=0
interval=5
host=nexa
usage() {
    echo "Uso: $0 [--once] [--interval segundos] [--host alias-ssh]" >&2
    exit 2
}
while (($#)); do
    case $1 in
        --once) once=1 ;;
        --interval) shift; interval=${1:-} ;;
        --host) shift; host=${1:-} ;;
        *) usage ;;
    esac
    (($#)) || usage
    shift
done
[[ $interval =~ ^[1-9][0-9]*$ && $host =~ ^[a-zA-Z0-9][a-zA-Z0-9.@_-]*$ ]] || usage

probe() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" /opt/nexa/ml-venv/bin/python - <<'PY'
import json
import re
import struct
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import grpc

def command(*args):
    return subprocess.check_output(args, text=True, timeout=10, stderr=subprocess.DEVNULL).strip()

def section(name, fn):
    try:
        fn()
        return True
    except (OSError, ValueError, KeyError, IndexError, struct.error, subprocess.SubprocessError, grpc.RpcError):
        print(name, "sin datos")
        return False

def release():
    manifest = json.loads(Path("/opt/nexa/current/release-manifest.json").read_bytes())
    version, sha = manifest["version"], manifest["git_sha"]
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", version) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError()
    print("Release", version, sha)
    print(command("df", "-h", "/"))

def soak():
    files = Path("/var/lib/nexa/soak").glob("*/summary.json")
    path = max(files, key=lambda p: p.stat().st_mtime, default=None)
    if path is None:
        print("Soak sin resumen")
        return
    data = json.loads(path.read_bytes())
    print(f"Soak {int(data['samples_collected'])}/{int(data['expected_samples'])} muestras",
          "estado", data["status"] if data["status"] in ("running", "completed", "failed") else "?")

def units():
    for name in ("nexa.target", "nexa-memoryd", "nexa-llm-fast", "nexa-audiod", "nexa-ml",
                 "nexa-brain", "nexa-web", "nexa-dashboard-proxy", "nexa-audiod-soak-20h"):
        state = command("systemctl", "--user", "show", "-p", "ActiveState", "--value", name)
        print(name, state)

def process():
    pid = int(command("systemctl", "--user", "show", "-p", "MainPID", "--value", "nexa-audiod"))
    proc = Path("/proc") / str(pid)
    rss = re.search(r"^VmRSS:\s+(\d+ kB)", (proc / "status").read_text(), re.M)
    if rss is None:
        raise ValueError()
    names = {"alsa-capture", "libusb_event", "nexa-audiod", "tokio-rt-worker",
             "tokio-runtime-w", "tokio-blocking"}
    threads = Counter()
    for path in (proc / "task").glob("*/comm"):
        try:
            name = path.read_text().strip()
        except FileNotFoundError:
            continue
        threads[name if name in names else "otros"] += 1
    cpu = command("ps", "-p", str(pid), "-o", "%cpu=")
    print(f"PID={pid} Tasks={sum(threads.values())} VmRSS={rss[1]} FDs={len(list((proc/'fd').iterdir()))} CPU={cpu}%")
    print("Hilos  cantidad")
    for name, count in sorted(threads.items()):
        print(name, count)

# Wire types del contrato nexa.v1. Se preservan las entradas repeated.
def varint(buf, pos):
    value = 0
    for shift in range(0, 70, 7):
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError()

def fields(buf):
    pos = 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        wire = tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        else:
            if wire == 2:
                size, pos = varint(buf, pos)
            elif wire in (1, 5):
                size = 8 if wire == 1 else 4
            else:
                raise ValueError()
            value = buf[pos:pos + size]
            if len(value) != size:
                raise ValueError()
            pos += size
        yield tag >> 3, value

def rpc(method):
    return fields(channel.unary_unary("/nexa.v1.AudioService/" + method)(b"", timeout=3))

def health():
    h = dict(rpc("GetHealth"))
    print("GetHealth state", {1:"ok", 2:"degraded", 3:"unavailable"}.get(h.get(3), "?"),
          "uptime_ms", h.get(4, 0))
    counters = dict(re.findall(rb"(\w+)=(\d+)(?=;|\s|$)", h.get(5, b"")))
    for key in ("capture_xruns", "ring_overruns", "watchdog_reboots", "utterances_published",
                "utterances_dropped", "utterances_lagged", "playbacks_failed"):
        print(key, counters.get(key.encode(), b"?").decode())

def snapshot():
    rooms = {}
    observed = 0
    for tag, value in rpc("GetMicrophones"):
        if tag == 1:
            observed = value
        elif tag == 3:
            row = dict(fields(value))
            rooms[row[1]] = row
    return observed, rooms

def microphones():
    cards = {}
    for card in Path("/sys/class/sound").glob("card[0-9]*"):
        for part in reversed(card.resolve().parts):
            if re.fullmatch(r"\d+-\d+(?:\.\d+)*", part):
                cards[part] = card.name
                break
    t0, first = snapshot()
    time.sleep(1)
    t1, last = snapshot()
    print("room_id usb_port ALSA state last_rms frames_captured fps_est recoveries frame_age_ms")
    for room in ("living", "hall", "cocina", "escritorio"):
        key = room.encode()
        if key not in last:
            print(room, "sin datos")
            continue
        row = last[key]
        port = row.get(3, b"").decode("ascii")
        if not re.fullmatch(r"\d+-\d+(?:\.\d+)*", port):
            port = "?"
        count = row.get(6, 0)
        delta = count - first.get(key, {}).get(6, 0)
        fps = f"{1000*delta/(t1-t0):.1f}" if key in first and t1 > t0 and delta >= 0 else "?"
        rms, = struct.unpack("<f", row.get(5, bytes(4)))
        state = {1:"waiting", 2:"receiving", 3:"recovering", 4:"muted", 5:"offline"}
        print(room, port, cards.get(port, "?"), state.get(row.get(4), "?"), f"{rms:.4f}",
              count, fps, row.get(7, 0), t1-row[8] if row.get(8) else "?")

def journal():
    # Emitir solo timestamp + categoría constante. WARN/ERROR pueden contener texto privado.
    labels = ("room capture", "XVF3800", "overrun", "ring ", "watchdog", "reboot", "playback ",
              "wake word detected", "Ambient segment", "utterance published", "WARN ", "ERROR ")
    pattern = "|".join(map(re.escape, labels))
    lines = command("journalctl", "--user", "-u", "nexa-audiod.service", "--no-pager",
                    "--grep", pattern, "--case-sensitive=yes", "-n", "12", "-o", "json",
                    "--output-fields=MESSAGE")
    print("Journal audiod UTC")
    for line in lines.splitlines():
        entry = json.loads(line)
        msg = entry.get("MESSAGE")
        # journald codifica MESSAGE como bytes JSON si hay escapes de terminal.
        if isinstance(msg, list):
            msg = bytes(msg).decode("utf-8", errors="replace")
        if isinstance(msg, str):
            match = re.search(pattern, msg)
            if match:
                stamp = datetime.fromtimestamp(int(entry["__REALTIME_TIMESTAMP"])/1e6, timezone.utc)
                print(stamp.isoformat(timespec="seconds"), match[0].strip())

print(datetime.now(timezone.utc).isoformat(timespec="seconds"))
with grpc.insecure_channel("unix:///run/user/1022/nexa/audio.sock",
                           options=[("grpc.default_authority", "localhost")]) as channel:
    checks = [section(fn.__name__, fn) for fn in (release, soak, units, process, health, microphones, journal)]
raise SystemExit(0 if all(checks) else 1)
PY
}

while :; do
    if [[ -t 1 && $once == 0 ]]; then printf '\033[H\033[2J'; fi
    rc=0
    probe || rc=$?
    if ((once)); then exit "$rc"; fi
    sleep "$interval"
done
