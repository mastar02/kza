"""Publish the verified voice-order artifact using the existing private state.

Build and test as yo; execute the compiled bundle as the runtime operator.
Only kza-voice is stopped/restarted. The proxy, secrets, database paths, models
and log targets stay in place. An activation failure restores the previous code
and service override, without restoring old databases over newer user data.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from uuid import uuid4

import aiohttp

try:
    from scripts.stage_runtime import _read_archive
    from tools import xvf_noise_trial as noise
except ModuleNotFoundError:
    from stage_runtime import _read_archive
    import xvf_noise_trial as noise


COMMIT = '63ea8d319b50506cc37ba73df1e41c87a41054cf'
PREVIOUS_COMMIT = 'b0866e8db01fd934ff3e6c483cca07d94168a9dc'
ARCHIVE_SHA256 = 'ea81e3c07d787cf92a307bdcb1e28246c889f5eb117e7c0e80ab7629c44ac19c'
CERTIFICATE_SHA256 = 'b05e58f5c3b577fa22393de9e3a67866bf3c972a70ac9ee7d44a0190083deaa0'
APP_CHANGES = {'src/users/voice_ledger.pyc', 'src/dashboard/voice_api.pyc',
               'src/dashboard/frontend/obs/src/voice-view.js'}
RESOURCES = ('config', 'data', 'models', 'logs', '.env')
BASE = 'http://127.0.0.1:9500'


class ReleaseError(RuntimeError):
    """Expose a fixed error code instead of private exception contents."""


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _record(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_bytes(path, (json.dumps(document, allow_nan=False) + '\n').encode())


def verify_runtime(runtime: Path) -> dict:
    """Verify every code/resource hash without following code-directory links."""
    manifest = json.loads((runtime / 'release.json').read_text())
    for name, expected in manifest['files'].items():
        path = runtime / name
        if (not name.startswith(('src/', 'scripts/', 'tools/')) or path.resolve() != path
                or path.suffix == '.py' or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ReleaseError('runtime_integrity_mismatch')
    actual = {str(path.relative_to(runtime)) for path in runtime.rglob('*')
              if path.is_file() or path.is_symlink()}
    if actual != set(manifest['files']) | {'release.json', 'config', 'data', '.env'}:
        raise ReleaseError('unexpected_runtime_files')
    return manifest


def prepare_code_slot(archive: Path, root: Path, previous: Path, name: str,
                      expected_commit: str, expected_sha: str) -> tuple[Path, dict, dict]:
    """Stage immutable code, linking state resources to their existing canonical paths."""
    if previous.name != 'runtime' or previous.parent.parent != root / 'slots' or previous.resolve() != previous:
        raise ReleaseError('unexpected_previous_slot')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ReleaseError('archive_hash_mismatch')
    manifest, files = _read_archive(archive)
    if manifest['git_commit'] != expected_commit:
        raise ReleaseError('unexpected_artifact_commit')
    old = verify_runtime(previous)
    old_app = {name: value for name, value in old['files'].items() if name.startswith('src/')}
    new_app = {name: value for name, value in manifest['files'].items() if name.startswith('src/')}
    if old_app.keys() != new_app.keys() or {name for name in old_app if old_app[name] != new_app[name]} != APP_CHANGES:
        raise ReleaseError('application_diff_exceeds_voice_order_change')
    if not 1 <= len(name) <= 80 or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in name):
        raise ReleaseError('invalid_slot_name')
    state = previous.parent / 'state'
    resources = {resource: (state / resource).resolve(strict=True) for resource in RESOURCES}
    if not all(resources[key].is_dir() for key in ('config', 'data', 'models', 'logs')) or not resources['.env'].is_file():
        raise ReleaseError('missing_existing_state_resources')
    target = root / 'slots' / name
    if target.exists() or target.is_symlink():
        raise ReleaseError('slot_already_exists')
    with tempfile.TemporaryDirectory(prefix='.voice-order-', dir=root / 'slots') as temporary:
        staged = Path(temporary)
        runtime = staged / 'runtime'
        runtime.mkdir(mode=0o700)
        for filename, content in files.items():
            path = runtime / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (runtime / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
        container = staged / 'state'
        container.mkdir(mode=0o700)
        for resource, canonical in resources.items():
            (container / resource).symlink_to(canonical)
        for resource in ('config', 'data', '.env'):
            (runtime / resource).symlink_to(Path('..') / 'state' / resource)
        verify_runtime(runtime)
        os.rename(staged, target)
    return target, manifest, resources


def service_override(original: bytes, previous_state: Path, new_state: Path) -> bytes:
    """Change only WorkingDirectory; preserve all code selection and log settings."""
    lines = original.decode().splitlines(keepends=True)
    indices = [index for index, line in enumerate(lines) if line.startswith('WorkingDirectory=')]
    if len(indices) != 1 or lines[indices[0]].split('=', 1)[1].strip().strip('"') != str(previous_state):
        raise ReleaseError('unexpected_service_working_directory')
    lines[indices[0]] = 'WorkingDirectory=' + str(new_state) + '\n'
    if 'PYTHONPATH=/home/kza/releases/current' not in original.decode():
        raise ReleaseError('unexpected_code_selector')
    return ''.join(lines).encode()


def backup_databases(data: Path, destination: Path) -> int:
    """Keep consistent SQLite snapshots without modifying the original databases."""
    destination.mkdir(mode=0o700)
    count = 0
    for path in sorted(data.rglob('*')):
        if path.suffix not in ('.db', '.sqlite3') or path.is_symlink() or not path.is_file():
            continue
        target = destination / path.relative_to(data)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as source:
            if source.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ReleaseError('source_database_check_failed')
            with closing(sqlite3.connect(target)) as copied:
                source.backup(copied)
                if copied.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    raise ReleaseError('database_backup_check_failed')
        target.chmod(0o600)
        count += 1
    return count


def _select(root: Path, runtime: Path) -> None:
    if runtime.resolve() != runtime or runtime.parent.parent != root / 'slots' or runtime.name != 'runtime':
        raise ReleaseError('unsafe_code_selection')
    temporary = root / ('.current-' + str(uuid4()))
    try:
        temporary.symlink_to(runtime.relative_to(root))
        os.replace(temporary, root / 'current')
    finally:
        temporary.unlink(missing_ok=True)


def _control(action: str) -> None:
    command = ['/usr/bin/systemctl', '--user', action]
    if action != 'daemon-reload':
        command.append('kza-voice.service')
    result = subprocess.run(command, capture_output=True, timeout=130)
    if result.returncode:
        raise ReleaseError('voice_control_' + action + '_failed')
    if action == 'stop' and int(noise._unit().get('MainPID', 0)):
        raise ReleaseError('voice_process_did_not_stop')


def _get(path: str, token: str) -> dict:
    request = urllib.request.Request(BASE + path, headers={'Authorization': 'Bearer ' + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), noise._NoRedirect())
    with opener.open(request, timeout=4) as response:
        content = response.read(1_048_577)
        if len(content) > 1_048_576:
            raise ReleaseError('runtime_response_too_large')
        return json.loads(content)


def _receiving(snapshot: dict) -> dict:
    now = time.time()
    return {mic['room_id']: mic['capture']['frames_received'] for mic in snapshot.get('microphones', [])
            if mic.get('capture', {}).get('state') == 'receiving'
            and 0 <= now - (mic['capture'].get('last_callback_at') or 0) < 6
            and 0 <= now - (mic['capture'].get('last_nonzero_at') or 0) < 6}


def _wait_ready(token: str, required: set[str], seconds: float = 300) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            health = _get('/api/health', token)
            first = _receiving(_get('/api/home/microphones', token))
            time.sleep(.3)
            second = _receiving(_get('/api/home/microphones', token))
            if required and all(room in first and second.get(room, 0) > first[room] for room in required):
                return {'health_status': health.get('status'), 'receiving_rooms': sorted(second)}
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise ReleaseError('runtime_readiness_timed_out')


async def check_dashboard(token: str, asset_sha: str, *, base_url: str = BASE,
                          tls_context=True) -> dict:
    """Verify the served asset and two authenticated chronological REST pages."""
    headers = {'Authorization': 'Bearer ' + token}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), trust_env=False) as session:
        async with session.get(base_url + '/api/health', ssl=tls_context, allow_redirects=False) as response:
            await response.read()
            if response.status != 401:
                raise ReleaseError('private_api_access_boundary_failed')
        async with session.get(base_url + '/obs/src/voice-view.js', headers=headers, ssl=tls_context,
                               allow_redirects=False) as response:
            content = await response.read()
            if response.status != 200 or hashlib.sha256(content).hexdigest() != asset_sha:
                raise ReleaseError('served_frontend_hash_mismatch')
        async with session.ws_connect(base_url + '/ws/live', headers=headers, ssl=tls_context,
                                      max_msg_size=1_048_576) as socket:
            async def consume(frame):
                kind = frame.get('type')
                if kind == 'live_reset':
                    payload = frame['payload']
                    await socket.send_json({'type': 'live_reset_ack', 'id': payload['id'], 'nonce': payload['nonce']})
                receipts = ([frame['receipt']] if frame.get('receipt') else [])
                receipts.extend({'ticket': item['ticket'], 'nonce': item['nonce']}
                                for item in frame.get('payload', {}).get('receipts', []))
                identity = frame.get('connection_id') if kind == 'rest_ready' else None
                frame = None
                for offset in range(0, len(receipts), 16):
                    await socket.send_json({'type': 'live_ack', 'receipts': receipts[offset:offset + 16]})
                return identity

            await socket.send_json({'type': 'live_hello', 'version': 3, 'client_id': str(uuid4())})
            connection = None
            while connection is None:
                connection = await consume(await socket.receive_json(timeout=20))
            cursor, previous, rows_checked, pages = None, None, 0, 0
            for _ in range(2):
                receipt = None
                try:
                    async with session.get(base_url + '/api/voices', params={'cursor': cursor} if cursor else {},
                            headers={**headers, 'X-KZA-Live-Connection': connection}, ssl=tls_context,
                            allow_redirects=False) as response:
                        if response.status != 200:
                            raise ReleaseError('voice_page_http_failed')
                        body = await response.json()
                        batch = body.get('snapshot')
                        if batch is None:
                            raise ReleaseError('voice_page_unavailable')
                        receipt = batch['receipt']
                        for row in batch['rows']:
                            position = row['created_at'], row['id']
                            if previous is not None and position >= previous:
                                raise ReleaseError('voice_rows_not_descending')
                            previous = position
                            rows_checked += 1
                        cursor = batch.get('next_cursor')
                finally:
                    row = body = batch = None
                    response = None
                    if receipt is not None:
                        await socket.send_json({'type': 'live_ack', 'receipts': [receipt]})
                pages += 1
                if cursor is not None and (not cursor.startswith('vc1_') or len(cursor) != 85):
                    raise ReleaseError('legacy_pagination_still_served')
                # Drain live/retire frames only after releasing the HTTP body.
                while True:
                    try:
                        await consume(await socket.receive_json(timeout=.05))
                    except asyncio.TimeoutError:
                        break
                if not cursor:
                    break
            if not rows_checked:
                raise ReleaseError('no_voice_rows_to_verify')
    return {'rows_checked': rows_checked, 'pages_checked': pages, 'descending': True,
            'frontend_hash_matches': True, 'unauthenticated_private_api_status': 401}


def _hall_profile(runtime: Path) -> tuple[dict, dict]:
    import yaml

    state = runtime.parent / 'state'
    config_path = state / 'config/settings.yaml'
    config = yaml.safe_load(config_path.read_text())
    unit = noise._unit()
    if Path('/proc/' + unit['MainPID'] + '/cwd').resolve() != state:
        raise ReleaseError('runtime_working_directory_mismatch')
    snapshot = noise._microphones()
    identity = noise.verify_preflight(config, snapshot, unit, config_path.stat().st_mtime,
                                      time.time(), time.monotonic())
    controller = noise.XvfController(usb_port=identity['port'], read_timeout_ms=100, max_retries=20)
    if not controller.open() or controller._dev.serial_number != identity['serial']:
        raise ReleaseError('hall_usb_identity_mismatch')
    profile = noise.read_static_params(controller)
    for name, value in config['rooms']['wake_word']['xvf_tuning']['params'].items():
        if name != noise.PARAMETER and (name not in profile or not noise._same(profile[name], value)):
            raise ReleaseError('restart_would_change_other_dsp_parameters')
    if not any(noise._same(profile[noise.PARAMETER], (value,)) for value in (noise.BASELINE, noise.TRIAL)):
        raise ReleaseError('unexpected_hall_trial_setting')
    return profile, snapshot


def _preserve_hall(expected: dict, runtime: Path) -> None:
    actual, _ = _hall_profile(runtime)
    if not noise._same(expected[noise.PARAMETER], actual[noise.PARAMETER]):
        restore = noise._same(expected[noise.PARAMETER], (noise.BASELINE,))
        # The temporary trial's own preflight and retained backup are reused.
        noise.run_trial(restore)
        actual, _ = _hall_profile(runtime)
    if any(not noise._same(expected[name], actual[name]) for name in noise.STATIC_PARAMS):
        raise ReleaseError('hall_profile_not_preserved')


def _https_check(bundle: Path, token: str, manifest: dict) -> dict:
    certificate = bundle / 'dashboard-public.crt'
    der = ssl.PEM_cert_to_DER_cert(certificate.read_text())
    if hashlib.sha256(der).hexdigest() != CERTIFICATE_SHA256:
        raise ReleaseError('tls_certificate_mismatch')
    context = ssl.create_default_context(cafile=str(certificate))
    return asyncio.run(check_dashboard(token, manifest['files']['src/dashboard/frontend/obs/src/voice-view.js'],
        base_url='https://192.168.1.2:9540', tls_context=context))


def publish(bundle: Path, rollback: bool = False) -> dict:
    """Perform the operator's code-only activation, with automatic code rollback."""
    from dotenv import dotenv_values

    if os.getuid() != 1000:
        raise ReleaseError('runtime_operator_required')
    root = Path('/home/kza/releases')
    override = Path('/home/kza/.config/systemd/user/kza-voice.service.d/release.conf')
    journal = root / 'deployments/voice-order-63ea8d3.json'
    token = dotenv_values('/home/kza/secrets/.env').get('KZA_DASHBOARD_TOKEN')
    if not token:
        raise ReleaseError('dashboard_credential_missing')
    with (root / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = (root / 'current').resolve(strict=True)
        manifest = verify_runtime(previous)
        if rollback:
            record = json.loads(journal.read_text())
            old_runtime = Path(record['previous_runtime'])
            if old_runtime.parent.parent != root / 'slots' or verify_runtime(old_runtime)['git_commit'] != PREVIOUS_COMMIT:
                raise ReleaseError('rollback_target_mismatch')
            if previous == old_runtime:
                ready = _wait_ready(token, set(record['required_rooms']))
                return {'status': 'already_rolled_back', 'source_commit': PREVIOUS_COMMIT, **ready}
            _control('stop')
            _select(root, old_runtime)
            _atomic_bytes(override, Path(record['backup']).joinpath('release.conf').read_bytes())
            _control('daemon-reload')
            _control('start')
            ready = _wait_ready(token, set(record['required_rooms']))
            _preserve_hall(record['hall_profile'], old_runtime)
            result = {'status': 'rolled_back', 'source_commit': PREVIOUS_COMMIT, **ready}
            _record(journal, {**record, 'status': 'rolled_back', 'result': result})
            return result
        if manifest['git_commit'] == COMMIT:
            record = json.loads(journal.read_text())
            if record.get('status') != 'active':
                raise ReleaseError('unfinished_activation_requires_rollback')
            _hall_profile(previous)
            ready = _wait_ready(token, set(record['required_rooms']))
            checks = _https_check(bundle, token, manifest)
            return {'status': 'already_active', 'source_commit': COMMIT,
                    'release_id': previous.parent.name, **ready, **checks}
        if manifest['git_commit'] != PREVIOUS_COMMIT:
            raise ReleaseError('unexpected_active_release')
        profile, snapshot = _hall_profile(previous)
        required = set(_receiving(snapshot))
        name = 'release_' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ') + '_63ea8d3'
        target, candidate, resources = prepare_code_slot(bundle / 'kza-runtime.tar.gz', root, previous,
                                                        name, COMMIT, ARCHIVE_SHA256)
        original = override.read_bytes()
        replacement = service_override(original, previous.parent / 'state', target / 'state')
        backup = root / 'backups' / name
        backup.mkdir(parents=True, mode=0o700)
        _atomic_bytes(backup / 'release.conf', original)
        shutil.copytree(resources['config'], backup / 'config', symlinks=True)
        shutil.copyfile(resources['.env'], backup / '.env')
        (backup / '.env').chmod(0o600)
        record = {'slot': name, 'previous_runtime': str(previous), 'backup': str(backup),
                  'required_rooms': sorted(required), 'hall_profile': profile,
                  'source_commit': COMMIT, 'status': 'prepared'}
        _record(journal, record)
        stopped = False
        try:
            print('[release] Stopping kza-voice; the existing stop timeout may take 90 seconds.', file=sys.stderr, flush=True)
            _control('stop')
            stopped = True
            databases = backup_databases(resources['data'], backup / 'databases')
            _select(root, target / 'runtime')
            _atomic_bytes(override, replacement)
            _control('daemon-reload')
            _control('start')
            print('[release] Waiting for authenticated API and the previously receiving microphones.', file=sys.stderr, flush=True)
            ready = _wait_ready(token, required)
            _preserve_hall(profile, target / 'runtime')
            _atomic_bytes(journal, (json.dumps({**record, 'status': 'verifying'}) + '\n').encode())
            checks = _https_check(bundle, token, candidate)
            # Retain the same narrow console access under the new slot's state container.
            subprocess.run(['/usr/bin/setfacl', '-m', 'u:yo:--x', str(target), str(target / 'state')],
                           check=True, capture_output=True, timeout=5)
            result = {'status': 'active', 'source_commit': COMMIT, 'release_id': name,
                      'previous_release_id': previous.parent.name, 'database_backups': databases,
                      'canonical_private_resources_preserved': True, 'hall_PP_MIN_NN': profile[noise.PARAMETER][0],
                      'other_hall_static_parameters_preserved': True, 'proxy_restarted': False,
                      'dependencies_installed': False, 'manifest_files': len(candidate['files']),
                      'compiled_modules': sum(name.endswith('.pyc') for name in candidate['files']), **ready, **checks}
            _record(journal, {**record, 'status': 'active', 'result': result})
            return result
        except Exception as error:
            failure = str(error) if isinstance(error, ReleaseError) else type(error).__name__
            rollback_error = None
            if stopped or not int(noise._unit().get('MainPID', 0)):
                try:
                    _control('stop')
                    _select(root, previous)
                    _atomic_bytes(override, original)
                    _control('daemon-reload')
                    _control('start')
                    _wait_ready(token, required)
                    _preserve_hall(profile, previous)
                except Exception as problem:
                    rollback_error = type(problem).__name__
            result = {'status': 'rollback_needs_attention' if rollback_error else 'previous_release_restored',
                      'error_code': failure, 'rollback_error_type': rollback_error, 'published': False}
            _record(journal, {**record, 'status': result['status'], 'result': result})
            return result


def main() -> None:
    """Run only the prepared release or its recorded code rollback, emitting JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--apply', action='store_true')
    action.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = publish(Path(__file__).resolve().parent, args.rollback)
    except Exception as error:
        result = {'status': 'not_published', 'error_code': str(error) if isinstance(error, ReleaseError) else type(error).__name__}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    if result['status'] not in ('active', 'already_active', 'rolled_back', 'already_rolled_back'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
