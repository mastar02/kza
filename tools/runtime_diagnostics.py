"""Fixed read-only runtime snapshot for a restricted local SSH key.

No command-line arguments or SSH_ORIGINAL_COMMAND are interpreted. This tool
does not expose credentials, process environments or arbitrary files, execute
caller commands, open audio streams, or mutate service state.
"""

from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

ROOT = Path('/home/kza/releases')
MAX_BYTES = 1024 * 1024
UNIT_PROPERTIES = 'ActiveState,SubState,MainPID,NRestarts,Result,ExecMainStartTimestamp'
SENSITIVE_KEY = re.compile(r'token|secret|password|authorization|cookie|api.?key', re.I)


def redact(value: object, secrets: tuple[str, ...] = ()) -> object:
    """Remove credential fields and credential values from projected output."""
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()
                if not SENSITIVE_KEY.search(str(key))}
    if isinstance(value, (tuple, list)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[redacted]')
        value = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [redacted]', value)
        value = re.sub(r'(?i)((?:token|api_key|password)=)[^&\s]+', r'\1[redacted]', value)
    return value


def _json_file(path: Path) -> object:
    with path.open('rb') as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError('diagnostic_input_too_large')
    return json.loads(content)


def _command(arguments: list[str], *, user_bus: bool = False) -> str:
    environment = dict(os.environ)
    if user_bus:
        environment.update(XDG_RUNTIME_DIR='/run/user/1000',
                           DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
    result = subprocess.run(arguments, env=environment, capture_output=True,
                            text=True, timeout=4, check=True)  # nosec B603 -- fixed internal commands only
    return result.stdout


def _unit(name: str, *, user: bool) -> dict:
    command = ['/usr/bin/systemctl', *(['--user'] if user else []),
               'show', name, '--property=' + UNIT_PROPERTIES]
    text = _command(command, user_bus=user)
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def _get_endpoint(path: str, credential: str) -> dict:
    if path not in ('/api/health', '/api/home/microphones', '/api/alerts'):
        raise ValueError('unsupported_diagnostic_endpoint')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request('http://127.0.0.1:9500' + path,
                                     headers={'Authorization': 'Bearer ' + credential})
    try:
        with opener.open(request, timeout=4) as response:
            content = response.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError('diagnostic_response_too_large')
            data = json.loads(content)
            if path == '/api/alerts':
                data = {'count': len(data), 'items': [
                    {key: item.get(key) for key in ('id', 'priority', 'type', 'acked')}
                    for item in data[:100]]}
            return {'http_status': response.status,
                    'source': response.headers.get('X-KZA-Source'), 'data': data}
    except urllib.error.HTTPError as error:
        return {'http_status': error.code}


def _process(pid: int) -> dict:
    if pid <= 0:
        return {'available': False}
    root = Path('/proc') / str(pid)
    status = {}
    for line in (root / 'status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Name', 'State', 'Threads', 'VmRSS', 'VmSwap'):
            status[key] = value.strip()
    waits = Counter()
    for task in (root / 'task').iterdir():
        try:
            waits[(task / 'wchan').read_text().strip()] += 1
        except OSError:
            continue
    return {'pid': pid, 'status': status, 'io': (root / 'io').read_text(),
            'main_wait': (root / 'wchan').read_text().strip(),
            'thread_wait_counts': dict(waits),
            'stdout': os.readlink(root / 'fd/1')}


def _capture_settings(config: dict) -> dict:
    rooms = config.get('rooms', {})
    allowed = ('mic_usb_port', 'mic_device_index', 'capture_channel', 'enabled')
    return {'rooms': {name: {key: room[key] for key in allowed if key in room}
                      for name, room in rooms.items()
                      if isinstance(room, dict) and 'mic_usb_port' in room},
            'watchdog': {key: rooms.get('stream_watchdog', {}).get(key) for key in
                         ('enabled', 'check_interval_s', 'no_frames_timeout_s',
                          'first_frame_grace_s', 'reopen_backoff_min_s', 'reopen_backoff_max_s')}}


def collect_snapshot() -> dict:
    """Observe fixed runtime resources without modifying their state."""
    from dotenv import dotenv_values
    import yaml

    result = {'checked_at': datetime.now(UTC).isoformat(), 'effective_uid': os.getuid()}
    credentials = dotenv_values('/home/kza/secrets/.env')
    secrets = tuple(str(value) for key, value in credentials.items()
                    if value and SENSITIVE_KEY.search(key))

    def observe(name, function):
        try:
            result[name] = function()
        except Exception as error:
            # Exception strings may contain private URLs, file bodies or tokens.
            result[name] = {'available': False, 'error_type': type(error).__name__}

    runtime = (ROOT / 'current').resolve(strict=True)
    if runtime.parent.parent != ROOT / 'slots' or runtime.name != 'runtime':
        raise ValueError('unexpected_runtime_location')
    state = runtime.parent / 'state'
    result['release_id'] = runtime.parent.name
    observe('source_commit', lambda: _json_file(runtime / 'release.json')['git_commit'])
    observe('capture_settings', lambda: _capture_settings(
        yaml.safe_load((state / 'config/settings.yaml').read_text())))
    observe('audio_health', lambda: {'file_mtime': (state / 'data/audio_health.json').stat().st_mtime,
                                     'snapshot': _json_file(state / 'data/audio_health.json')})
    for unit in ('kza-voice.service', 'kza-dashboard-proxy.service', 'kza-console-rotate.timer'):
        observe(unit, lambda unit=unit: _unit(unit, user=True))
    observe('journald', lambda: _unit('systemd-journald.service', user=False))
    voice = result.get('kza-voice.service', {})
    observe('voice_process', lambda: _process(int(voice.get('MainPID', 0))))
    observe('journal_service_messages', lambda: [
        {'timestamp_us': item.get('__REALTIME_TIMESTAMP'), 'message': item.get('MESSAGE')}
        for line in _command(['/usr/bin/journalctl', '--quiet', '--no-pager',
                              '--unit=systemd-journald.service', '--since=-1hour',
                              '-n', '40', '-o', 'json']).splitlines()
        if (item := json.loads(line))])
    credential = credentials.get('KZA_DASHBOARD_TOKEN')
    if credential:
        for name, path in (('health', '/api/health'), ('microphones', '/api/home/microphones'),
                           ('alerts_summary', '/api/alerts')):
            observe(name, lambda path=path: _get_endpoint(path, credential))
    return redact(result, secrets)


def main() -> None:
    """Print one snapshot; caller commands and arguments never select actions."""
    try:
        result = collect_snapshot()
    except Exception as error:
        result = {'available': False, 'error_type': type(error).__name__}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
