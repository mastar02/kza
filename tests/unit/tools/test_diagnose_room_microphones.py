"""T025: the room microphone diagnosis reads every configured room, Cocina included.

A fake sysfs/procfs tree stands in for the server; nothing here opens a device.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import diagnose_room_microphones as tool

REPOSITORY = Path(__file__).resolve().parents[3]

CONFIG = {
    'rooms': {
        'enabled': True,
        'fallback_room': 'living',
        'telemetry': {'interval_s': 2},
        'stream_watchdog': {'enabled': True, 'health_path': './data/audio_health.json'},
        'living': {'name': 'Living', 'mic_usb_port': '1-2.3', 'capture_channel': 1},
        'cocina': {'name': 'Cocina', 'mic_usb_port': '7-5.3', 'capture_channel': 1},
        'escritorio': {'name': 'Escritorio', 'mic_usb_port': '3-1.4'},
        'bano': {'name': 'Baño', 'mic_usb_port': None},
    },
}

RUNNING = 'state: RUNNING\nowner_pid   : 4242\n-----\nhw_ptr      : {ptr}\nappl_ptr    : {ptr}'


def _config(tmp_path, data=CONFIG):
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
    return path


class FakeHost:
    """Minimal /sys and /proc: USB devices, their sound cards and capture substreams."""

    def __init__(self, base):
        self.sys = base / 'sys'
        self.proc = base / 'proc'
        (self.sys / 'bus/usb/devices').mkdir(parents=True)
        (self.sys / 'class/sound').mkdir(parents=True)
        (self.proc / 'asound').mkdir(parents=True)

    def device(self, port, product='XVF3800 Voice Processor'):
        real = self.sys / 'devices/pci0000:00/usb' / port
        real.mkdir(parents=True)
        (real / 'product').write_text(product + '\n')
        (real / 'speed').write_text('480\n')
        (real / 'authorized').write_text('1\n')
        (real / 'power').mkdir()
        (real / 'power/runtime_status').write_text('active\n')
        (self.sys / 'bus/usb/devices' / port).symlink_to(real)
        return real

    def card(self, number, port, status):
        interface = self.sys / 'devices/pci0000:00/usb' / port / f'{port}:1.0'
        interface.mkdir(parents=True, exist_ok=True)
        card = self.sys / 'class/sound' / f'card{number}'
        card.mkdir()
        (card / 'device').symlink_to(interface)
        self.capture(number, status)

    def capture(self, number, status):
        sub = self.proc / f'asound/card{number}/pcm0c/sub0'
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'status').write_text(status + '\n')
        (sub / 'hw_params').write_text('format: S16_LE\nchannels: 2\nrate: 16000\n')


def _no_journal(*args, **kwargs):
    raise FileNotFoundError('journalctl')


def _collect(host, config, **kwargs):
    kwargs.setdefault('run', _no_journal)
    kwargs.setdefault('console_logs', [])
    kwargs.setdefault('sleep', lambda seconds: None)
    return tool.collect(config, sys_root=host.sys, proc_root=host.proc, **kwargs)


# --- configured rooms -------------------------------------------------------------------------

def test_rooms_come_from_the_configuration_not_a_fixed_list(tmp_path):
    rooms, without_mic = tool.configured_rooms(_config(tmp_path))
    assert rooms == {'living': {'name': 'Living', 'usb_port': '1-2.3'},
                     'cocina': {'name': 'Cocina', 'usb_port': '7-5.3'},
                     'escritorio': {'name': 'Escritorio', 'usb_port': '3-1.4'}}
    assert without_mic == {'bano': {'name': 'Baño'}}


def test_repository_configuration_reports_cocina_with_its_configured_port():
    config = REPOSITORY / 'config' / 'settings.yaml'
    expected = yaml.safe_load(config.read_text(encoding='utf-8'))['rooms']['cocina']['mic_usb_port']
    rooms, _ = tool.configured_rooms(config)
    assert rooms['cocina'] == {'name': 'Cocina', 'usb_port': expected}
    assert {'living', 'escritorio', 'cocina'} <= set(rooms)


def test_unreadable_configuration_is_reported_without_touching_devices(tmp_path, capsys):
    code = tool.main(['--config', str(tmp_path / 'missing.yaml'), '--sys-root', str(tmp_path),
                      '--proc-root', str(tmp_path), '--interval', '0'])
    output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert output['config_error'] == 'no disponible (FileNotFoundError)'
    assert 'first' not in output and 'rooms' not in output


def test_rooms_without_a_mapping_section_yield_no_rooms(tmp_path):
    assert tool.configured_rooms(_config(tmp_path, {'rooms': ['living']})) == ({}, {})
    assert tool.configured_rooms(_config(tmp_path, {'audio': {}})) == ({}, {})


# --- sysfs / procfs ---------------------------------------------------------------------------

def test_present_device_with_running_capture(tmp_path):
    host = FakeHost(tmp_path)
    host.device('7-5.3')
    host.card(2, '7-5.3', RUNNING.format(ptr=1000))
    info = tool.inspect_room('7-5.3', host.sys, host.proc)
    assert info['present'] is True
    assert info['product'] == 'XVF3800 Voice Processor'
    assert info['power/runtime_status'] == 'active'
    assert info['capture_state'] == 'running'
    [card] = info['alsa_cards']
    assert card['card'] == '2'
    [capture] = card['captures']
    assert capture['hw_ptr'] == 1000
    assert capture['hw_params'].startswith('format: S16_LE')


def test_absent_device_reports_only_its_absence(tmp_path):
    host = FakeHost(tmp_path)
    assert tool.inspect_room('7-5.3', host.sys, host.proc) == {'usb_port': '7-5.3', 'present': False}


def test_closed_capture_is_reported_as_closed(tmp_path):
    host = FakeHost(tmp_path)
    host.device('1-2.3')
    host.card(1, '1-2.3', 'closed')
    info = tool.inspect_room('1-2.3', host.sys, host.proc)
    assert info['capture_state'] == 'closed'
    assert info['alsa_cards'][0]['captures'][0]['hw_ptr'] is None


def test_capture_state_summary():
    assert tool.capture_state([]) == 'no_capture'
    assert tool.capture_state(['closed', 'closed']) == 'closed'
    assert tool.capture_state(['closed', RUNNING.format(ptr=1)]) == 'running'
    assert tool.capture_state(['closed', 'state: XRUN']) == 'other'
    assert tool.capture_state(['no disponible (PermissionError)']) == 'other'


def test_present_device_without_sound_card(tmp_path):
    host = FakeHost(tmp_path)
    host.device('7-5.3')
    info = tool.inspect_room('7-5.3', host.sys, host.proc)
    assert info['alsa_cards'] == [] and info['capture_state'] == 'no_capture'


def test_sound_card_of_another_usb_device_is_not_attributed(tmp_path):
    host = FakeHost(tmp_path)
    host.device('7-5.3')
    host.device('7-5.1', product='UGREEN BT')
    host.card(3, '7-5.1', RUNNING.format(ptr=5))
    info = tool.inspect_room('7-5.3', host.sys, host.proc)
    assert info['alsa_cards'] == [] and info['capture_state'] == 'no_capture'


@pytest.mark.parametrize('port', ['../../../etc', '7-5.3/..', '', '7-5.3 ', 'usb7'])
def test_invalid_configured_port_never_reaches_the_filesystem(tmp_path, port):
    host = FakeHost(tmp_path)
    info = tool.inspect_room(port, host.sys, host.proc)
    assert info == {'usb_port': port, 'present': False, 'error': 'puerto USB inválido en la configuración'}


def test_capture_progress_compares_the_two_samples(tmp_path):
    def sample(ptr):
        return {'alsa_cards': [{'card': '2', 'captures': [
            {'path': '/proc/asound/card2/pcm0c/sub0/status', 'status': RUNNING.format(ptr=ptr), 'hw_ptr': ptr}]}]}

    closed = {'alsa_cards': [{'card': '1', 'captures': [
        {'path': '/proc/asound/card1/pcm0c/sub0/status', 'status': 'closed', 'hw_ptr': None}]}]}
    first = {'cocina': sample(100), 'living': sample(100), 'escritorio': closed, 'hall': {'present': False}}
    second = {'cocina': sample(4900), 'living': sample(100), 'escritorio': closed, 'hall': {'present': False}}
    assert tool.capture_progress(first, second) == {
        'cocina': {'hw_ptr_advanced': True}, 'living': {'hw_ptr_advanced': False},
        'escritorio': {'hw_ptr_advanced': None}, 'hall': {'hw_ptr_advanced': None}}


def test_capture_progress_needs_the_same_substream_running_in_both_samples():
    def sample(path, ptr):
        return {'alsa_cards': [{'card': '2', 'captures': [
            {'path': path, 'status': RUNNING.format(ptr=ptr), 'hw_ptr': ptr}]}]}

    first = {'cocina': sample('/a/status', 100)}
    second = {'cocina': sample('/b/status', 900)}
    assert tool.capture_progress(first, second) == {'cocina': {'hw_ptr_advanced': None}}
    assert tool.capture_progress(first, {}) == {'cocina': {'hw_ptr_advanced': None}}


# --- kernel lines -----------------------------------------------------------------------------

KERNEL = [
    '2026-11-16T10:00:00-0300 kza kernel: usb 7-5.3: new high-speed USB device number 9 using xhci_hcd',
    '2026-11-16T10:00:01-0300 kza kernel: usb 7-5.3: USB disconnect, device number 9',
    '2026-11-16T10:00:02-0300 kza kernel: usb 17-5.3: new full-speed USB device',
    '2026-11-16T10:00:03-0300 kza kernel: usb 7-5.31: new full-speed USB device',
    '2026-11-16T10:00:04-0300 kza kernel: usb 7-5.3.1: new full-speed USB device',
    '2026-11-16T10:00:05-0300 kza kernel: usb 1-2.3: reset high-speed USB device',
    '2026-11-16T10:00:06-0300 kza kernel: xhci_hcd 0000:05:00.0: WARN: transfer error on endpoint',
    '2026-11-16T10:00:07-0300 kza kernel: cannot submit capture urb (err = -19)',
    '2026-11-16T10:00:08-0300 kza kernel: audit: type=1400 apparmor="ALLOWED" name="/home/kza/secrets/.env"',
    '2026-11-16T10:00:09-0300 kza kernel: usb 7-5.3:1.0: interface 0 claimed',
]


def _journal(stdout, returncode=0, seen=None):
    def run(command, **kwargs):
        if seen is not None:
            seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr='')
    return run


def test_kernel_lines_are_only_configured_ports_and_host_controller_errors():
    seen = []
    result = tool.kernel_messages(['7-5.3', '1-2.3'], _journal('\n'.join(KERNEL) + '\n', seen=seen))
    assert result['exit_code'] == 0
    assert result['messages'] == [KERNEL[0], KERNEL[1], KERNEL[5], KERNEL[6], KERNEL[7], KERNEL[9]]
    [(command, kwargs)] = seen
    assert command[:2] == ['journalctl', '-k'] and kwargs['timeout'] == 15


def test_kernel_lines_skip_an_invalid_port_and_unconfigured_rooms():
    result = tool.kernel_messages(['1-2.3', '.*'], _journal('\n'.join(KERNEL)))
    assert result['messages'] == [KERNEL[5], KERNEL[6], KERNEL[7]]


def test_kernel_lines_are_bounded():
    lines = [f'kernel: usb 7-5.3: line {index}' for index in range(200)]
    result = tool.kernel_messages(['7-5.3'], _journal('\n'.join(lines)))
    assert result['messages'] == lines[-tool.MAX_LINES:]


@pytest.mark.parametrize('error', [FileNotFoundError('journalctl'), subprocess.TimeoutExpired('journalctl', 15)])
def test_unavailable_journal_is_reported_by_type(error):
    def run(*args, **kwargs):
        raise error
    assert tool.kernel_messages(['7-5.3'], run) == {'error': f'no disponible ({type(error).__name__})'}


# --- watchdog lines ---------------------------------------------------------------------------

def _console(path, lines):
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def test_only_watchdog_lines_leave_the_console_log(tmp_path):
    log = _console(tmp_path / 'console.log', [
        '10:00 INFO [stt] transcripción: "apagá la luz de la cocina"',
        '10:00 INFO [audio-watchdog] living: recovered (device=4)',
        '10:01 INFO [dashboard] login token=abc123',
        '10:01 WARNING \x1b[33m[audio-watchdog] cocina: device 7-5.3 absent, retry in 2.0s\x1b[0m',
        '10:02 ERROR [audio-watchdog] streams [\'cocina\'] stopped delivering audio; recovering [\'cocina\']',
        '10:03 INFO [audio-watchdog] hall: recovered (device=6)',
    ])
    result = tool.watchdog_messages(['living', 'cocina', 'escritorio'], [log])
    assert result['path'] == str(log)
    assert result['rooms'] == {
        'living': ['10:00 INFO [audio-watchdog] living: recovered (device=4)'],
        'cocina': ['10:01 WARNING [audio-watchdog] cocina: device 7-5.3 absent, retry in 2.0s'],
        'escritorio': []}
    assert result['other'] == [
        '10:02 ERROR [audio-watchdog] streams [\'cocina\'] stopped delivering audio; recovering [\'cocina\']',
        '10:03 INFO [audio-watchdog] hall: recovered (device=6)']
    exported = json.dumps(result, ensure_ascii=False)
    assert 'transcripción' not in exported and 'token' not in exported and '\x1b' not in exported


def test_a_flooding_room_does_not_hide_the_others(tmp_path):
    flood = [f'[audio-watchdog] cocina: device 7-5.3 absent, retry {index}' for index in range(500)]
    log = _console(tmp_path / 'console.log', ['[audio-watchdog] living: recovered (device=4)', *flood])
    result = tool.watchdog_messages(['living', 'cocina'], [log])
    assert result['rooms']['living'] == ['[audio-watchdog] living: recovered (device=4)']
    assert result['rooms']['cocina'] == flood[-tool.MAX_ROOM_LINES:]


def test_other_watchdog_lines_and_long_lines_are_bounded(tmp_path):
    other = [f'[audio-watchdog] streams [{index}] stopped ' + 'x' * 600 for index in range(100)]
    log = _console(tmp_path / 'console.log', other)
    result = tool.watchdog_messages(['living'], [log])
    assert result['other'] == [line[:500] for line in other[-tool.MAX_LINES:]]


def test_only_the_last_megabyte_is_examined(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, 'MAX_LOG_BYTES', 64)
    log = _console(tmp_path / 'console.log', ['[audio-watchdog] living: old', 'y' * 200,
                                              '[audio-watchdog] living: new'])
    result = tool.watchdog_messages(['living'], [log])
    assert result['rooms']['living'] == ['[audio-watchdog] living: new']
    assert result['last_bytes_examined'] == 64


def test_room_prefix_needs_the_exact_room_id(tmp_path):
    log = _console(tmp_path / 'console.log', ['[audio-watchdog] cocina2: device absent'])
    result = tool.watchdog_messages(['cocina'], [log])
    assert result['rooms'] == {'cocina': []}
    assert result['other'] == ['[audio-watchdog] cocina2: device absent']


def test_newest_readable_console_log_is_used(tmp_path):
    old = _console(tmp_path / 'old.log', ['[audio-watchdog] living: old console'])
    new = _console(tmp_path / 'new.log', ['[audio-watchdog] living: new console'])
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    result = tool.watchdog_messages(['living'], [tmp_path / 'missing.log', new, old])
    assert result['path'] == str(new)
    os.utime(old, (3_000, 3_000))
    assert tool.watchdog_messages(['living'], [new, old])['path'] == str(old)


def test_missing_console_log_is_reported_not_raised(tmp_path):
    assert tool.watchdog_messages(['living'], [tmp_path / 'missing.log']) == {
        'error': 'no disponible (FileNotFoundError)'}
    assert tool.watchdog_messages(['living'], []) == {'error': 'no disponible (FileNotFoundError)'}


# --- whole run ----------------------------------------------------------------------------------

def test_whole_run_reports_every_configured_room_including_cocina(tmp_path):
    host = FakeHost(tmp_path)
    host.device('7-5.3')
    host.card(2, '7-5.3', RUNNING.format(ptr=100))
    host.device('1-2.3')
    host.card(1, '1-2.3', 'closed')
    log = _console(tmp_path / 'console.log', ['[audio-watchdog] cocina: recovered (device=5)',
                                              'INFO [stt] texto privado'])
    def advance(seconds):
        assert seconds == 0.5
        host.capture(2, RUNNING.format(ptr=4900))

    result = _collect(host, _config(tmp_path), console_logs=[log], interval=0.5, sleep=advance,
                      run=_journal(KERNEL[0] + '\n' + KERNEL[8] + '\n'))
    assert set(result['first']) == set(result['second']) == {'living', 'cocina', 'escritorio'}
    assert result['first']['cocina']['present'] is True
    assert result['first']['escritorio'] == {'usb_port': '3-1.4', 'present': False}
    assert result['capture_progress'] == {'living': {'hw_ptr_advanced': None},
                                          'cocina': {'hw_ptr_advanced': True},
                                          'escritorio': {'hw_ptr_advanced': None}}
    assert result['rooms_without_microphone'] == {'bano': {'name': 'Baño'}}
    assert result['usb_sysfs'] == 'disponible'
    assert result['kernel_last_2h']['messages'] == [KERNEL[0]]
    assert result['voice_watchdog_log_tail']['rooms']['cocina'] == ['[audio-watchdog] cocina: recovered (device=5)']
    assert 'texto privado' not in json.dumps(result, ensure_ascii=False)


def test_host_without_sysfs_reports_no_disponible_without_crashing(tmp_path, capsys):
    empty = tmp_path / 'empty'
    empty.mkdir()
    code = tool.main(['--config', str(_config(tmp_path)), '--sys-root', str(empty), '--proc-root', str(empty),
                      '--console-log', str(tmp_path / 'missing.log'), '--interval', '0'])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output['usb_sysfs'] == 'no disponible'
    assert all(info == {'usb_port': info['usb_port'], 'present': False} for info in output['second'].values())
    assert output['voice_watchdog_log_tail'] == {'error': 'no disponible (FileNotFoundError)'}
    assert 'cocina' in output['first']
