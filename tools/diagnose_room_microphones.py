#!/usr/bin/env python3
"""Read-only USB/ALSA and log diagnosis for the living and office microphones."""

import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOMS = {'living': '1-2.3', 'escritorio': '3-1.4'}
MAX_LOG_BYTES = 1_000_000
MAX_LINES = 60


def read_text(path):
    try:
        return Path(path).read_text(errors='replace').strip()
    except OSError as exc:
        return f'no disponible ({type(exc).__name__})'


def inspect_room(port):
    usb = Path('/sys/bus/usb/devices') / port
    info = {'usb_port': port, 'present': usb.exists()}
    if not usb.exists():
        return info
    for name in ('product', 'speed', 'authorized', 'power/runtime_status'):
        info[name] = read_text(usb / name)
    usb_path = usb.resolve()
    cards = []
    for card in sorted(Path('/sys/class/sound').glob('card[0-9]*')):
        if usb_path not in (card / 'device').resolve().parents:
            continue
        number = card.name[4:]
        captures = []
        for status in sorted(glob.glob(f'/proc/asound/card{number}/pcm*c/sub*/status')):
            captures.append({'path': status, 'status': read_text(status),
                             'hw_params': read_text(Path(status).with_name('hw_params'))})
        cards.append({'card': number, 'captures': captures})
    info['alsa_cards'] = cards
    return info


def kernel_messages():
    try:
        result = subprocess.run(
            ['journalctl', '-k', '--since=-2 hours', '-n', '10000', '--no-pager',
             '--output=short-iso'], capture_output=True, text=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': type(exc).__name__}
    # Generic xHCI/URB errors may belong to another USB device; do not attribute them.
    pattern = re.compile(r'1-2\.3|3-1\.4|capture urb|xhci.*(?:error|failed|timeout)', re.I)
    return {'exit_code': result.returncode,
            'messages': [line[:500] for line in result.stdout.splitlines()
                         if pattern.search(line)][-MAX_LINES:],
            'stderr': result.stderr[:500]}


def watchdog_messages():
    try:
        runtime = Path('/home/kza/releases/current').resolve(strict=True)
        log = runtime.parent / 'state/logs/console.log'
        with log.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - MAX_LOG_BYTES))
            tail = handle.read(MAX_LOG_BYTES).decode('utf-8', errors='replace')
    except OSError as exc:
        return {'error': type(exc).__name__}
    # Never export ordinary application logs, conversations or credentials.
    lines = [re.sub(r'\x1b\[[0-9;]*m', '', line)[:500] for line in tail.splitlines()
             if '[audio-watchdog]' in line and 'cocina: device' not in line]
    return {'last_bytes_examined': MAX_LOG_BYTES, 'messages': lines[-MAX_LINES:]}


def main():
    result = {'checked_at': dt.datetime.now().astimezone().isoformat(),
              'effective_uid': os.geteuid(),
              'note': 'Solo lectura; no se abren dispositivos ni se reinician servicios.',
              'first': {room: inspect_room(port) for room, port in ROOMS.items()}}
    time.sleep(3)
    result['second'] = {room: inspect_room(port) for room, port in ROOMS.items()}
    result['kernel_last_2h'] = kernel_messages()
    result['voice_watchdog_log_tail'] = watchdog_messages()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
