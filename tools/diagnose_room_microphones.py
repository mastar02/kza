#!/usr/bin/env python3
"""Read-only USB/ALSA and log diagnosis for every configured room microphone.

Rooms and their USB ports come from ``rooms`` in config/settings.yaml: a room is
a mapping with ``mic_usb_port`` (T025). Nothing opens an audio device, restarts
a service or exports ordinary application logs: only kernel lines about the
configured ports or the USB host controller, and ``[audio-watchdog]`` lines.

Usage (as ``kza``, from the compiled slot)::

    cd /home/kza/releases/current
    /home/kza/kza/.venv/bin/python -B tools/diagnose_room_microphones.pyc
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / 'config' / 'settings.yaml'
SYS_ROOT = Path('/sys')
PROC_ROOT = Path('/proc')
# Managed launcher console first, then the legacy per-slot console (DEPLOYMENT.md).
MANAGED_CONSOLE_LOG = Path('/home/kza/releases/logs/console.log')
CURRENT_RELEASE = Path('/home/kza/releases/current')
MAX_LOG_BYTES = 1_000_000
MAX_LINES = 60
MAX_ROOM_LINES = 20
SAMPLE_INTERVAL_S = 3.0
# bus-port[.port...] as the kernel names it; anything else never reaches sysfs or a regex.
USB_PORT = re.compile(r'\d+-\d+(?:\.\d+)*')
WATCHDOG_TAG = '[audio-watchdog]'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
HW_PTR = re.compile(r'^hw_ptr\s*:\s*(\d+)\s*$', re.M)


def unavailable(exc):
    """Report a failure by its type only; the message may carry private paths."""
    return f'no disponible ({type(exc).__name__})'


def read_text(path):
    try:
        return Path(path).read_text(errors='replace').strip()
    except OSError as exc:
        return unavailable(exc)


def configured_rooms(config_path):
    """Return (rooms with a microphone port, rooms without one) from settings.yaml.

    Raises OSError or a YAML error when the configuration cannot be read.
    """
    import yaml  # third-party; imported here so a missing package is reported, not fatal

    with open(config_path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    rooms = config.get('rooms') if isinstance(config, dict) else None
    if not isinstance(rooms, dict):
        return {}, {}
    with_mic, without_mic = {}, {}
    for room_id, room in rooms.items():
        if not isinstance(room, dict):
            continue
        port = room.get('mic_usb_port')
        if port:
            with_mic[str(room_id)] = {'name': room.get('name', room_id), 'usb_port': str(port)}
        elif 'name' in room:
            without_mic[str(room_id)] = {'name': room['name']}
    return with_mic, without_mic


def capture_state(statuses):
    """Summarise ALSA capture substreams: running, closed, other or no_capture."""
    if not statuses:
        return 'no_capture'
    if any(status.startswith('state: RUNNING') for status in statuses):
        return 'running'
    if all(status == 'closed' for status in statuses):
        return 'closed'
    return 'other'


def hw_pointer(status):
    match = HW_PTR.search(status)
    return int(match.group(1)) if match else None


def inspect_room(port, sys_root=SYS_ROOT, proc_root=PROC_ROOT):
    info = {'usb_port': port}
    if not USB_PORT.fullmatch(port):
        info['present'] = False
        info['error'] = 'puerto USB inválido en la configuración'
        return info
    usb = Path(sys_root) / 'bus/usb/devices' / port
    info['present'] = usb.exists()
    if not info['present']:
        return info
    for name in ('product', 'speed', 'authorized', 'power/runtime_status'):
        info[name] = read_text(usb / name)
    usb_path = usb.resolve()
    cards, statuses = [], []
    for card in sorted((Path(sys_root) / 'class/sound').glob('card[0-9]*')):
        if usb_path not in (card / 'device').resolve().parents:
            continue
        number = card.name[4:]
        captures = []
        for status_path in sorted((Path(proc_root) / f'asound/card{number}').glob('pcm*c/sub*/status')):
            status = read_text(status_path)
            statuses.append(status)
            captures.append({'path': str(status_path), 'status': status, 'hw_ptr': hw_pointer(status),
                             'hw_params': read_text(status_path.with_name('hw_params'))})
        cards.append({'card': number, 'captures': captures})
    info['alsa_cards'] = cards
    info['capture_state'] = capture_state(statuses)
    return info


def capture_progress(first, second):
    """Whether each room's running capture pointer moved between the two samples.

    None means not measurable (absent device, no running capture in both samples).
    A moving pointer does not prove real audio: a hung XVF3800 delivers zeros
    at the right rate (runbook 2026-07-25); the audio heartbeat checks that.
    """
    progress = {}
    for room_id, before in first.items():
        after = second.get(room_id, {})
        pointers = {}
        for sample, index in ((before, 0), (after, 1)):
            for card in sample.get('alsa_cards', []):
                for capture in card['captures']:
                    if capture['status'].startswith('state: RUNNING') and capture['hw_ptr'] is not None:
                        pointers.setdefault(capture['path'], [None, None])[index] = capture['hw_ptr']
        moved = [b != a for a, b in pointers.values() if a is not None and b is not None]
        progress[room_id] = {'hw_ptr_advanced': any(moved) if moved else None}
    return progress


def kernel_pattern(ports):
    """Lines about a configured port (not a neighbour such as 17-5.3 or 7-5.31) or the host."""
    alternatives = [r'(?<![\d.-])' + re.escape(port) + r'(?!\d|\.\d)'
                    for port in sorted(ports) if USB_PORT.fullmatch(port)]
    # Generic xHCI/URB errors may belong to another USB device; do not attribute them.
    alternatives += [r'capture urb', r'xhci.*(?:error|failed|timeout)']
    return re.compile('|'.join(alternatives), re.I)


def kernel_messages(ports, run=subprocess.run):
    try:
        result = run(
            ['journalctl', '-k', '--since=-2 hours', '-n', '10000', '--no-pager',
             '--output=short-iso'], capture_output=True, text=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': unavailable(exc)}
    pattern = kernel_pattern(ports)
    return {'exit_code': result.returncode,
            'messages': [line[:500] for line in result.stdout.splitlines()
                         if pattern.search(line)][-MAX_LINES:],
            'stderr': result.stderr[:500]}


def default_console_logs():
    candidates = [MANAGED_CONSOLE_LOG]
    try:
        candidates.append(CURRENT_RELEASE.resolve(strict=True).parent / 'state/logs/console.log')
    except OSError:
        pass
    return candidates


def newest_readable(candidates):
    best = None
    for path in candidates:
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            continue
        if os.access(path, os.R_OK) and (best is None or mtime > best[0]):
            best = (mtime, Path(path))
    return None if best is None else best[1]


def watchdog_messages(room_ids, candidates):
    """Only ``[audio-watchdog]`` lines, bounded per room so one flooding room hides no other."""
    path = newest_readable(candidates)
    if path is None:
        return {'error': unavailable(FileNotFoundError())}
    try:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - MAX_LOG_BYTES))
            tail = handle.read(MAX_LOG_BYTES).decode('utf-8', errors='replace')
    except OSError as exc:
        return {'error': unavailable(exc)}
    # Never export ordinary application logs, conversations or credentials.
    per_room = {room_id: [] for room_id in room_ids}
    other = []
    for line in tail.splitlines():
        if WATCHDOG_TAG not in line:
            continue
        line = ANSI.sub('', line)[:500]
        room = next((room_id for room_id in room_ids if f'{WATCHDOG_TAG} {room_id}:' in line), None)
        (per_room[room] if room is not None else other).append(line)
    return {'path': str(path), 'last_bytes_examined': MAX_LOG_BYTES,
            'rooms': {room_id: lines[-MAX_ROOM_LINES:] for room_id, lines in per_room.items()},
            'other': other[-MAX_LINES:]}


def collect(config_path=DEFAULT_CONFIG, *, sys_root=SYS_ROOT, proc_root=PROC_ROOT, console_logs=None,
            run=subprocess.run, sleep=time.sleep, interval=SAMPLE_INTERVAL_S):
    """Build the diagnosis; ``config_error`` is set when no room could be read."""
    result = {'checked_at': dt.datetime.now().astimezone().isoformat(),
              'effective_uid': os.geteuid(),
              'note': 'Solo lectura; no se abren dispositivos ni se reinician servicios.',
              'config_path': str(config_path)}
    try:
        rooms, without_mic = configured_rooms(config_path)
    except Exception as exc:  # any unreadable or malformed configuration is reported, never raised
        result['config_error'] = unavailable(exc)
        return result
    result['rooms'] = rooms
    result['rooms_without_microphone'] = without_mic
    usb_root = Path(sys_root) / 'bus/usb/devices'
    result['usb_sysfs'] = 'disponible' if usb_root.is_dir() else 'no disponible'
    ports = {room_id: room['usb_port'] for room_id, room in rooms.items()}
    result['first'] = {room_id: inspect_room(port, sys_root, proc_root) for room_id, port in ports.items()}
    sleep(interval)
    result['second'] = {room_id: inspect_room(port, sys_root, proc_root) for room_id, port in ports.items()}
    result['capture_progress'] = capture_progress(result['first'], result['second'])
    result['kernel_last_2h'] = kernel_messages(ports.values(), run)
    result['voice_watchdog_log_tail'] = watchdog_messages(
        list(rooms), default_console_logs() if console_logs is None else console_logs)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--console-log', type=Path, action='append', dest='console_logs',
                        help='consola de KZA a revisar (repetible; por defecto la gestionada y la del slot)')
    parser.add_argument('--sys-root', type=Path, default=SYS_ROOT, help='raíz de sysfs (pruebas)')
    parser.add_argument('--proc-root', type=Path, default=PROC_ROOT, help='raíz de procfs (pruebas)')
    parser.add_argument('--interval', type=float, default=SAMPLE_INTERVAL_S)
    args = parser.parse_args(argv)
    result = collect(args.config, sys_root=args.sys_root, proc_root=args.proc_root,
                     console_logs=args.console_logs, interval=args.interval)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if 'config_error' in result else 0


if __name__ == '__main__':
    sys.exit(main())
