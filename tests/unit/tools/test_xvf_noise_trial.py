"""The controlled trial refuses active pollers and changes only its one knob."""

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from tools.xvf_noise_trial import (BASELINE, PARAMETER, STATIC_PARAMS, TRIAL,
                                   change_verified_parameter, read_static_params, verify_preflight)


def runtime():
    config = {'rooms': {'wake_word': {'spenergy_gate': {'enabled': False}},
                        'direction_observation': {'enabled': False},
                        'hall': {'mic_usb_port': '5-1.2'}}}
    capture = {'state': 'receiving', 'nonzero_seen': True, 'last_callback_at': 999.,
               'last_nonzero_at': 999., 'sample_rate_hz': 16000., 'channels': 2,
               'frames_received': 16000, 'generation': 1}
    mic = {'room_id': 'hall', 'connection': 'connected', 'binding': 'matched',
           'usb_port': '5-1.2', 'configured_usb_port': '5-1.2', 'serial': 'synthetic-serial',
           'direction': {'state': 'disabled'}, 'capture': capture}
    snapshot = {'stale': False, 'microphones': [mic]}
    unit = {'ActiveState': 'active', 'MainPID': '42', 'ExecMainStartTimestampMonotonic': '50000000'}
    return config, snapshot, unit


def test_preflight_binds_only_fresh_hall_identity():
    config, snapshot, unit = runtime()
    target = verify_preflight(config, snapshot, unit, 900, 1000, 100)
    assert target == {'port': '5-1.2', 'serial': 'synthetic-serial', 'pid': 42, 'generation': 1}


def test_restore_can_verify_identity_even_when_capture_has_stopped():
    config, snapshot, unit = runtime()
    mic = snapshot['microphones'][0]
    mic['binding'] = 'unknown'
    mic['capture'].update(state='closed', last_callback_at=None, last_nonzero_at=None)
    assert verify_preflight(config, snapshot, unit, 900, 1000, 100,
                            require_capture=False)['serial'] == 'synthetic-serial'
    with pytest.raises(ValueError):
        verify_preflight(config, snapshot, unit, 900, 1000, 100)


@pytest.mark.parametrize('case', ['spenergy', 'direction', 'stale', 'binding', 'no_frames',
                                 'no_signal', 'changed_config', 'duplicate_room'])
def test_preflight_refuses_unverified_or_competing_runtime(case):
    config, snapshot, unit = runtime()
    config_mtime = 900
    mic = snapshot['microphones'][0]
    if case == 'spenergy':
        config['rooms']['wake_word']['spenergy_gate']['enabled'] = True
    elif case == 'direction':
        config['rooms']['direction_observation']['enabled'] = True
    elif case == 'stale':
        snapshot['stale'] = True
    elif case == 'binding':
        mic['usb_port'] = '5-1.3'
    elif case == 'no_frames':
        mic['capture']['last_callback_at'] = 900
    elif case == 'no_signal':
        mic['capture']['last_nonzero_at'] = 900
    elif case == 'changed_config':
        config_mtime = 999
    else:
        snapshot['microphones'].append(deepcopy(mic))
    with pytest.raises(ValueError):
        verify_preflight(config, snapshot, unit, config_mtime, 1000, 100)


class Controller:
    def __init__(self):
        self.values = {name: (1.,) for name in STATIC_PARAMS}
        self.values.update({PARAMETER: (BASELINE,), 'AUDIO_MGR_OP_L': (8, 0),
                            'AUDIO_MGR_OP_R': (7, 3), 'AEC_ASROUTONOFF': (1,)})
        self.writes = []
        self.fail = False

    def read_param(self, name):
        return self.values[name]

    def write_param(self, name, values):
        self.writes.append((name, list(values)))
        if self.fail:
            return False
        self.values[name] = tuple(values)
        return True


def test_apply_and_restore_touch_only_noise_parameter():
    controller = Controller()
    before = read_static_params(controller)
    after = change_verified_parameter(controller, before, BASELINE, TRIAL)
    assert after[PARAMETER] == (TRIAL,)
    restored = change_verified_parameter(controller, after, TRIAL, BASELINE)
    assert restored == before
    assert controller.writes == [(PARAMETER, [TRIAL]), (PARAMETER, [BASELINE])]


def test_wrong_baseline_is_rejected_before_any_write():
    controller = Controller()
    with pytest.raises(ValueError, match='unexpected_current'):
        change_verified_parameter(controller, read_static_params(controller), .3, TRIAL)
    assert controller.writes == []


def test_incomplete_read_or_unexpected_routing_is_not_a_baseline():
    controller = Controller()
    controller.values['PP_MIN_NS'] = None
    with pytest.raises(ValueError, match='incomplete_dsp'):
        read_static_params(controller)
    controller.values['PP_MIN_NS'] = (.15,)
    controller.values['AUDIO_MGR_OP_L'] = (1, 0)
    with pytest.raises(ValueError, match='unexpected_output'):
        read_static_params(controller)
    assert controller.writes == []


def test_failed_write_is_not_reported_as_applied():
    controller = Controller()
    controller.fail = True
    with pytest.raises(ValueError, match='write_not_confirmed'):
        change_verified_parameter(controller, read_static_params(controller), BASELINE, TRIAL)
    assert controller.values[PARAMETER] == (BASELINE,)


def test_unrelated_static_change_is_reported():
    controller = Controller()
    original = controller.write_param
    def changing(name, values):
        ok = original(name, values)
        controller.values['PP_AGCMAXGAIN'] = (64.,)
        return ok
    controller.write_param = changing
    with pytest.raises(ValueError, match='another_static_parameter'):
        change_verified_parameter(controller, read_static_params(controller), BASELINE, TRIAL)


def test_operator_flow_saves_baseline_before_write_and_restores_it(tmp_path, monkeypatch):
    import yaml

    import tools.xvf_noise_trial as trial

    config, snapshot, unit = runtime()
    runtime_path = tmp_path / 'releases/slots/synthetic-release/runtime'
    state = runtime_path.parent / 'state'
    runtime_path.mkdir(parents=True)
    (state / 'config').mkdir(parents=True)
    config_path = state / 'config/settings.yaml'
    config_path.write_text(yaml.safe_dump(config))
    os.utime(config_path, (time.time() - 120, time.time() - 120))
    (tmp_path / 'releases/current').symlink_to(runtime_path)
    proc = tmp_path / 'proc/42'
    proc.mkdir(parents=True)
    (proc / 'cwd').symlink_to(state)
    (proc / 'environ').write_bytes(b'CONFIG_PATH=config/settings.yaml\0')
    unit['ExecMainStartTimestampMonotonic'] = str(int((time.monotonic() - 60) * 1_000_000))
    original_path = Path
    def mapped_path(value):
        value = str(value)
        if value.startswith('/home/kza/'):
            return tmp_path / value.removeprefix('/home/kza/')
        if value.startswith('/proc/42/'):
            return tmp_path / value.lstrip('/')
        return original_path(value)
    monkeypatch.setattr(trial, 'Path', mapped_path)
    monkeypatch.setattr(trial.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(trial, '_unit', lambda: dict(unit))
    calls = 0
    def microphones():
        nonlocal calls
        calls += 1
        result = deepcopy(snapshot)
        result['microphones'][0]['capture'].update(last_callback_at=time.time(),
            last_nonzero_at=time.time(), frames_received=16000 * calls)
        return result
    monkeypatch.setattr(trial, '_microphones', microphones)
    controller = Controller()
    controller.open = lambda: True
    controller._dev = SimpleNamespace(serial_number='synthetic-serial')
    saved_path = tmp_path / '.local/state/kza-audio-quality/hall-pp-min-nn.json'
    original_write = controller.write_param
    def checked_write(name, values):
        assert saved_path.exists()
        assert json.loads(saved_path.read_text())['original'] == [BASELINE]
        return original_write(name, values)
    controller.write_param = checked_write
    monkeypatch.setattr(trial, 'XvfController', lambda **kwargs: controller)
    applied = trial.run_trial(False)
    assert applied['status'] == 'applied' and applied['after'] == TRIAL
    assert applied['capture_continues'] and applied['frames_advanced'] > 0
    assert saved_path.stat().st_mode & 0o777 == 0o600
    restored = trial.run_trial(True)
    assert restored['status'] == 'restored' and restored['after'] == BASELINE
    assert controller.writes == [(PARAMETER, [TRIAL]), (PARAMETER, [BASELINE])]
    assert json.loads(saved_path.read_text())['verified'] is True
