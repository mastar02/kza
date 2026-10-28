"""Security boundaries of the fixed runtime diagnostic projection."""

import json
import urllib.error
import urllib.request

import pytest

from tools import runtime_diagnostics as diagnostics


def test_projection_does_not_expose_secret_fields_or_embedded_credential_values():
    credential = 'private-runtime-credential'
    data = {'healthy': True, 'nested': [{'Authorization': credential, 'api_key': credential,
             'detail': 'failed using ' + credential, 'other': 'Bearer unknown-token'}],
            'password': 'hidden', 'url': 'http://example/?token=hidden&ok=yes'}
    text = json.dumps(diagnostics.redact(data, (credential,)))
    assert credential not in text
    assert 'hidden' not in text
    assert 'unknown-token' not in text
    assert json.loads(text)['healthy'] is True


def test_capture_projection_only_returns_diagnostic_settings():
    config = {'rooms': {'room-a': {'mic_usb_port': '1-1', 'capture_channel': 1,
                                  'password': 'hidden', 'occupant': 'private'},
                        'stream_watchdog': {'enabled': True, 'secret': 'hidden'}},
              'other_private_data': 'hidden'}
    result = diagnostics._capture_settings(config)
    assert result['rooms'] == {'room-a': {'mic_usb_port': '1-1', 'capture_channel': 1}}
    assert 'hidden' not in json.dumps(result)
    assert 'private' not in json.dumps(result)


def test_ssh_requested_commands_and_cli_arguments_are_never_executed(monkeypatch, capsys):
    monkeypatch.setenv('SSH_ORIGINAL_COMMAND', 'touch /tmp/forbidden-diagnostic-write')
    monkeypatch.setattr('sys.argv', ['tool.pyc', '--file', '/home/kza/secrets/.env'])
    monkeypatch.setattr(diagnostics, 'collect_snapshot', lambda: {'fixed_snapshot': True})
    monkeypatch.setattr(diagnostics.subprocess, 'run', lambda *a, **k: pytest.fail('caller command executed'))
    diagnostics.main()
    assert json.loads(capsys.readouterr().out) == {'fixed_snapshot': True}


def test_failed_diagnostic_does_not_print_exception_secrets(monkeypatch, capsys):
    def fail():
        raise RuntimeError('private-password-in-error-message')
    monkeypatch.setattr(diagnostics, 'collect_snapshot', fail)
    diagnostics.main()
    assert json.loads(capsys.readouterr().out) == {'available': False, 'error_type': 'RuntimeError'}


def test_redirect_cannot_forward_runtime_bearer_to_another_server():
    request = urllib.request.Request('http://127.0.0.1:9500/api/health',
                                     headers={'Authorization': 'Bearer private'})
    assert diagnostics._NoRedirect().redirect_request(
        request, None, 302, 'redirect', {}, 'http://example.com/') is None


def test_endpoint_path_is_closed_before_any_network_call(monkeypatch):
    monkeypatch.setattr(diagnostics.urllib.request, 'build_opener',
                        lambda *args: pytest.fail('unexpected network call'))
    with pytest.raises(ValueError, match='unsupported_diagnostic_endpoint'):
        diagnostics._get_endpoint('/api/admin/arbitrary', 'private')
