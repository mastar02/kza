"""Temporarily A/B only Hall's PP_MIN_NN, without restarting the running app.

Build and test as yo. The runtime operator executes a compiled bundle because
the preflight must read the actual private startup config and authenticated
capture state. Refuse live writes if SPENERGY or direction polling is enabled.
No application code/config, USB routing, flash or service state is changed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

try:
    from src.audio.xvf_controller import XvfController
except ModuleNotFoundError:
    from xvf_controller import XvfController


ROOM = 'hall'
PARAMETER = 'PP_MIN_NN'
BASELINE = .4
TRIAL = .51
STATIC_PARAMS = (PARAMETER, 'PP_MIN_NS', 'PP_AGCONOFF', 'PP_AGCMAXGAIN',
                 'PP_ATTNS_MODE', 'PP_ATTNS_SLOPE', 'AUDIO_MGR_OP_L',
                 'AUDIO_MGR_OP_R', 'AEC_ASROUTONOFF', 'AEC_ASROUTGAIN', 'AUDIO_MGR_MIC_GAIN')


def _same(left, right) -> bool:
    return len(left) == len(right) and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                                         for a, b in zip(left, right))


def verify_preflight(config: dict, snapshot: dict, unit: dict, config_mtime: float,
                     now: float, monotonic: float, *, require_capture: bool = True) -> dict:
    """Require actual startup flags, identity binding and a fresh receiving Hall."""
    if unit.get('ActiveState') != 'active' or int(unit.get('MainPID', 0)) <= 0:
        raise ValueError('voice_service_not_active')
    started = int(unit.get('ExecMainStartTimestampMonotonic', 0)) / 1_000_000
    if not started or config_mtime > now - monotonic + started + 1:
        raise ValueError('config_changed_since_service_start')
    rooms = config.get('rooms', {})
    if rooms.get('wake_word', {}).get('spenergy_gate', {}).get('enabled') is not False:
        raise ValueError('spenergy_poller_not_verified_disabled')
    if rooms.get('direction_observation', {}).get('enabled') is not False:
        raise ValueError('direction_poller_not_verified_disabled')
    if snapshot.get('stale') is not False:
        raise ValueError('stale_microphone_snapshot')
    matches = [mic for mic in snapshot.get('microphones', []) if mic.get('room_id') == ROOM]
    if len(matches) != 1:
        raise ValueError('hall_identity_not_unique')
    mic = matches[0]
    port = rooms.get(ROOM, {}).get('mic_usb_port')
    capture = mic.get('capture', {})
    allowed_bindings = ('matched',) if require_capture else ('matched', 'unknown')
    if (not port or mic.get('connection') != 'connected' or mic.get('binding') not in allowed_bindings
            or mic.get('usb_port') != port or mic.get('configured_usb_port') != port
            or not mic.get('serial') or mic.get('direction', {}).get('state') != 'disabled'):
        raise ValueError('hall_identity_or_polling_unverified')
    callback = capture.get('last_callback_at')
    nonzero = capture.get('last_nonzero_at')
    if require_capture and (capture.get('state') != 'receiving' or capture.get('nonzero_seen') is not True
            or capture.get('sample_rate_hz') != 16000 or capture.get('channels') != 2
            or not isinstance(callback, (int, float)) or not -.5 <= now - callback <= 6
            or not isinstance(nonzero, (int, float)) or not -.5 <= now - nonzero <= 6):
        raise ValueError('hall_capture_not_fresh')
    return {'port': port, 'serial': mic['serial'], 'generation': capture.get('generation'),
            'pid': int(unit['MainPID'])}


def read_static_params(controller) -> dict:
    """Require a complete baseline instead of treating a busy read as a value."""
    values = {name: controller.read_param(name) for name in STATIC_PARAMS}
    if any(value is None for value in values.values()):
        raise ValueError('incomplete_dsp_snapshot')
    if (not _same(values['AUDIO_MGR_OP_L'], (8, 0))
            or not _same(values['AUDIO_MGR_OP_R'], (7, 3))
            or not _same(values['AEC_ASROUTONOFF'], (1,))):
        raise ValueError('unexpected_output_routing')
    return values


def change_verified_parameter(controller, before: dict, expected: float, target: float) -> dict:
    """Write only the trial parameter and verify that other static values stay put."""
    if not _same(before[PARAMETER], (expected,)):
        raise ValueError('unexpected_current_noise_setting')
    if expected == target:
        return before
    if not controller.write_param(PARAMETER, [target]):
        raise ValueError('dsp_write_not_confirmed')
    after = read_static_params(controller)
    if not _same(after[PARAMETER], (target,)):
        raise ValueError('dsp_readback_mismatch')
    if any(not _same(before[name], after[name]) for name in STATIC_PARAMS if name != PARAMETER):
        raise ValueError('another_static_parameter_changed')
    return after


def _unit() -> dict:
    result = subprocess.run(['/usr/bin/systemctl', '--user', 'show', 'kza-voice.service',
                             '--property=ActiveState,MainPID,ExecMainStartTimestampMonotonic'],
                            check=True, capture_output=True, text=True, timeout=5)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def _microphones() -> dict:
    from dotenv import dotenv_values
    token = dotenv_values('/home/kza/secrets/.env').get('KZA_DASHBOARD_TOKEN')
    if not token:
        raise ValueError('dashboard_credential_unavailable')
    request = urllib.request.Request('http://127.0.0.1:9500/api/home/microphones',
                                     headers={'Authorization': 'Bearer ' + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=5) as response:
        body = response.read(1_048_577)
        if len(body) > 1_048_576:
            raise ValueError('microphone_response_too_large')
        return json.loads(body)


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(document, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_trial(restore: bool) -> dict:
    """Check runtime readers, retain a private baseline, then change Hall's RAM."""
    import yaml

    if os.getuid() != 1000:
        raise ValueError('runtime_operator_required_for_private_preflight')
    runtime = Path('/home/kza/releases/current').resolve(strict=True)
    if runtime.name != 'runtime' or runtime.parent.parent != Path('/home/kza/releases/slots'):
        raise ValueError('unexpected_release_path')
    config_path = runtime.parent / 'state/config/settings.yaml'
    unit = _unit()
    pid = int(unit.get('MainPID', 0))
    cwd = Path(f'/proc/{pid}/cwd').resolve(strict=True)
    if cwd != runtime.parent / 'state':
        raise ValueError('service_working_directory_mismatch')
    # Only CONFIG_PATH is selected from the owning process's startup
    # environment. No environment body or other variable is exported.
    selector = None
    for field in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        if field.startswith(b'CONFIG_PATH='):
            selector = os.fsdecode(field.split(b'=', 1)[1])
            break
    field = None
    if selector is None:
        from dotenv import dotenv_values
        selector = dotenv_values('/home/kza/secrets/.env').get('CONFIG_PATH', 'config/settings.yaml')
    effective_path = Path(selector)
    if not effective_path.is_absolute():
        effective_path = cwd / effective_path
    if effective_path.resolve(strict=True) != config_path.resolve(strict=True):
        raise ValueError('effective_startup_config_mismatch')
    config = yaml.safe_load(config_path.read_text())
    initial_snapshot = _microphones()
    identity = verify_preflight(config, initial_snapshot, unit, config_path.stat().st_mtime,
                                time.time(), time.monotonic(), require_capture=not restore)
    frames_before = next(mic['capture']['frames_received'] for mic in initial_snapshot['microphones']
                         if mic.get('room_id') == ROOM)
    controller = XvfController(usb_port=identity['port'], read_timeout_ms=100, max_retries=20)
    if not controller.open() or controller._dev.serial_number != identity['serial']:
        raise ValueError('usb_identity_mismatch')
    before = read_static_params(controller)
    path = Path('/home/kza/.local/state/kza-audio-quality/hall-pp-min-nn.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get('serial') != identity['serial'] or saved.get('port') != identity['port']
                or not _same(saved.get('original', []), (BASELINE,))):
            raise ValueError('saved_baseline_identity_mismatch')
    elif restore:
        raise ValueError('no_saved_baseline')
    else:
        if not _same(before[PARAMETER], (BASELINE,)):
            raise ValueError('unexpected_initial_noise_setting')
        saved = {**identity, 'original': list(before[PARAMETER]), 'trial': TRIAL,
                 'before': before, 'release': runtime.parent.name}
        _save(path, saved)
    target = BASELINE if restore else TRIAL
    current = float(before[PARAMETER][0])
    if not any(math.isclose(current, allowed, abs_tol=1e-6) for allowed in (BASELINE, TRIAL)):
        raise ValueError('unrecognized_trial_state')
    # Capture the attempted operation before the device transfer. Even a failed
    # readback leaves the original value available for an explicit rollback.
    saved.update(operation='restore' if restore else 'apply', verified=False,
                 attempted_at=datetime.now(UTC).isoformat())
    _save(path, saved)
    after = change_verified_parameter(controller, before, current, target)
    changed_at = datetime.now(UTC).isoformat()
    time.sleep(.2)
    final_snapshot = _microphones()
    final = verify_preflight(config, final_snapshot, _unit(), config_path.stat().st_mtime,
                             time.time(), time.monotonic(), require_capture=not restore)
    keys = ('port', 'serial', 'pid') if restore else tuple(identity)
    if (any(final[key] != identity[key] for key in keys)
            or Path('/home/kza/releases/current').resolve() != runtime):
        raise ValueError('runtime_or_capture_changed_during_trial')
    frames_after = next(mic['capture']['frames_received'] for mic in final_snapshot['microphones']
                        if mic.get('room_id') == ROOM)
    capture_continues = frames_after > frames_before
    if not restore and not capture_continues:
        raise ValueError('capture_did_not_advance')
    saved.update(verified=True, verified_at=changed_at, after=after)
    _save(path, saved)
    return {'status': 'restored' if restore else 'applied', 'room': ROOM, 'parameter': PARAMETER,
            'before': before[PARAMETER][0], 'after': after[PARAMETER][0], 'verified_at': changed_at,
            'other_static_parameters_unchanged': True, 'capture_continues': capture_continues,
            'frames_advanced': frames_after - frames_before,
            'service_restarted': False, 'persistent_config_changed': False, 'flash_written': False,
            'note': 'Wait 35 seconds quietly, then record a fresh, identifiable phrase in Hall. Old previews do not change.'}


def main() -> None:
    """Apply the single-value trial or restore its saved baseline, emitting JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--apply', action='store_true')
    action.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = run_trial(args.restore)
    except Exception as error:
        result = {'status': 'not_verified', 'error_type': type(error).__name__,
                  'reason': str(error) if type(error) is ValueError else 'private_runtime_error',
                  'note': 'Do not assume the change succeeded; a retained baseline allows explicit restore.'}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
