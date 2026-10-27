"""CLI and direct LoRA entrypoints cannot bypass private training admission."""

import asyncio
import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.training.conversation_collector import LoRATrainer
from src.training.nightly_trainer import NightlyTrainer, main as nightly_main
from tests.mocks.conversations import graph

pytestmark = pytest.mark.asyncio
ROOT = Path(__file__).resolve().parents[2]


def forbid_training_io(patch):
    """Reject filesystem, training imports, scheduler and process boundaries."""
    forbidden = Mock(side_effect=AssertionError('unadmitted training boundary'))
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.split('.')[0] in {'transformers', 'peft', 'datasets', 'torch'}:
            forbidden()
        return original_import(name, *args, **kwargs)

    patch.setattr(builtins, '__import__', importing)
    patch.setattr(builtins, 'open', forbidden)
    for name in ('open', 'exists', 'mkdir', 'iterdir', 'resolve'):
        patch.setattr(Path, name, forbidden)
    patch.setattr('threading.Thread.start', forbidden)
    patch.setattr('asyncio.create_subprocess_exec', forbidden)
    return forbidden


@pytest.mark.parametrize('kind', ['canonical', 'copied', 'symlink', 'hardlink'])
async def test_lora_rejects_actual_managed_export_at_any_path_before_io(tmp_path, monkeypatch, kind):
    async with graph(tmp_path) as g:
        assert await g.add()
        view, = await g.collector.get_turns()
        assert await g.collector.mark_response(view, 'good', sources=(g.base.source(),))
        export = await g.collector.export_for_training(str(tmp_path/'exports'))
        original = Path(export.output_path)
        body = original.read_bytes()
        candidate = tmp_path/'renamed.jsonl'
        if kind == 'canonical':
            candidate = original
        elif kind == 'copied':
            candidate.write_bytes(body)
        elif kind == 'symlink':
            candidate.symlink_to(original)
        else:
            os.link(original, candidate)
        with monkeypatch.context() as patch:
            forbidden = forbid_training_io(patch)
            trainer = LoRATrainer(output_dir=str(tmp_path/'models'))
            assert trainer.training_blocked
            with pytest.raises(RuntimeError, match='^private_training_session_unavailable$'):
                trainer.train(str(candidate))
            forbidden.assert_not_called()
        assert candidate.read_bytes() == original.read_bytes() == body
        assert not (tmp_path/'models').exists()


@pytest.mark.parametrize('name', ['missing.jsonl', 'arbitrary.jsonl'])
async def test_lora_does_not_read_or_infer_authorization_for_untracked_data(tmp_path, monkeypatch, name):
    candidate = tmp_path/name
    if name == 'arbitrary.jsonl':
        candidate.write_text('{"instruction":"synthetic legacy", "output":"synthetic legacy"}\n')
    with monkeypatch.context() as patch:
        forbidden = forbid_training_io(patch)
        trainer = LoRATrainer(output_dir=str(tmp_path/'models'))
        with pytest.raises(RuntimeError, match='^private_training_session_unavailable$'):
            trainer.train(str(candidate))
        with pytest.raises(RuntimeError, match='^private_training_session_unavailable$'):
            trainer.list_adapters()
        forbidden.assert_not_called()


@pytest.mark.parametrize('action', ['--status', '--collect', '--run'])
async def test_nightly_cli_reports_fixed_state_without_constructing_an_unmanaged_trainer(
    monkeypatch, capsys, action,
):
    with monkeypatch.context() as patch:
        forbidden = forbid_training_io(patch)
        patch.setattr(NightlyTrainer, '__init__', forbidden)
        code = await nightly_main([action, '--gpus', 'private-device-marker'])
        forbidden.assert_not_called()
    captured = capsys.readouterr()
    payload = json.loads(captured.out if action == '--status' else captured.err)
    assert code == (0 if action == '--status' else 1)
    assert payload['status'] == 'blocked' and payload['training_available'] is False
    assert payload['blocked_reason'] == 'private_training_session_unavailable'
    assert payload['runtime_status_checked'] is False
    assert 'private-device-marker' not in captured.out+captured.err
    assert not (captured.err if action == '--status' else captured.out)


@pytest.mark.parametrize('argv', [[], ['--help'], ['--status', '--collect']])
async def test_nightly_usage_and_invalid_actions_do_not_open_legacy_data(monkeypatch, capsys, argv):
    with monkeypatch.context() as patch:
        forbidden = forbid_training_io(patch)
        patch.setattr(NightlyTrainer, '__init__', forbidden)
        if not argv:
            assert await nightly_main(argv) == 0
        else:
            with pytest.raises(SystemExit) as exit_info:
                await nightly_main(argv)
            assert exit_info.value.code == (0 if argv == ['--help'] else 2)
        forbidden.assert_not_called()
    captured = capsys.readouterr()
    assert 'usage:' in captured.out+captured.err


@pytest.mark.parametrize('action', ['train', 'list'])
async def test_assistant_lora_command_rejects_before_importing_or_echoing_private_arguments(
    tmp_path, monkeypatch, capsys, action,
):
    spec = importlib.util.spec_from_file_location('synthetic_training_cli', ROOT/'scripts/train_assistant.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = SimpleNamespace(action=action, data=str(tmp_path/'private-data-marker'),
        output_dir=str(tmp_path/'private-model-marker'), base_model='private-model-name-marker')
    with monkeypatch.context() as patch:
        forbidden = forbid_training_io(patch)
        assert cli.cmd_lora(args) == 1
        forbidden.assert_not_called()
    captured = capsys.readouterr()
    assert not captured.out and 'private_training_session_unavailable' in captured.err
    assert not any(value in captured.err for value in (args.data, args.output_dir, args.base_model))
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize('entrypoint,action,code', [
    ('nightly', '--status', 0), ('nightly', '--collect', 1), ('nightly', '--run', 1),
    ('assistant', 'train', 1), ('assistant', 'list', 1),
])
async def test_actual_cli_process_preserves_legacy_data_and_propagates_unavailable_exit_code(
    tmp_path, entrypoint, action, code,
):
    nightly = tmp_path/'data/nightly_training'
    nightly.mkdir(parents=True)
    history = nightly/'training_history.json'
    history.write_text(json.dumps({'sessions':[{'id':'private-history-marker','started_at':1,
        'status':'completed','adapter_path':'private-adapter-marker'}]}))
    contexts = tmp_path/'data/contexts'
    contexts.mkdir()
    (contexts/'user_synthetic.json').write_text(json.dumps({'user_id':'synthetic',
        'conversation_history':[{'timestamp':9999999999,'content':'private-content-marker'}]}))
    before = {path.relative_to(tmp_path):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    if entrypoint == 'nightly':
        command = [sys.executable, '-m', 'src.training.nightly_trainer', action]
    else:
        command = [sys.executable, str(ROOT/'scripts/train_assistant.py'), 'lora', '--action', action,
            '--data', str(tmp_path/'private-source-marker'), '--output-dir', str(tmp_path/'models'),
            '--base-model', 'private-model-marker']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    result = await asyncio.to_thread(subprocess.run, command, cwd=tmp_path, env=environment,
                                    capture_output=True, text=True, timeout=20)
    assert result.returncode == code
    assert 'private_training_session_unavailable' in result.stdout+result.stderr
    assert not any(marker in result.stdout+result.stderr for marker in (
        'private-history-marker', 'private-adapter-marker', 'private-content-marker',
        'private-source-marker', 'private-model-marker',
    ))
    after = {path.relative_to(tmp_path):path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert after == before and not (tmp_path/'models').exists()
    if action == '--status':
        assert json.loads(result.stdout)['runtime_status_checked'] is False
