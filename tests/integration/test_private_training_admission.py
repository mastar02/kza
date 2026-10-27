"""Private runtime inputs never enter the legacy training lifecycle."""

import builtins
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.analytics.event_logger import EventLogger
from src.orchestrator.context_manager import ContextManager
from src.training.habit_dataset_generator import HabitDatasetGenerator, HabitExample
from src.training.nightly_trainer import NightlyConfig, NightlyTrainer, TrainingStatus
from tests.mocks.conversations import graph
from tests.mocks.stt_corrections import graph as stt_graph

pytestmark = pytest.mark.asyncio


def forbid_filesystem(patch):
    """Fail at the filesystem boundary, including before a file is opened."""
    forbidden = Mock(side_effect=AssertionError('private training touched disk'))
    patch.setattr(builtins, 'open', forbidden)
    patch.setattr(Path, 'mkdir', forbidden)
    patch.setattr(Path, 'exists', forbidden)
    patch.setattr(Path, 'iterdir', forbidden)
    return forbidden


@pytest.mark.parametrize('admission', ['policy', 'conversation', 'events'])
async def test_private_habit_constructor_does_not_load_legacy_profiles_or_create_roots(
    tmp_path, monkeypatch, admission,
):
    async with graph(tmp_path) as g:
        root = tmp_path/'habits'
        root.mkdir()
        legacy = root/'user_profiles.json'
        original = json.dumps({'synthetic-owner': {'user_name': 'private-profile-marker'}}).encode()
        legacy.write_bytes(original)
        dependencies = {}
        event_logger = None
        if admission == 'policy':
            dependencies['source_policy'] = g.privacy.ledger
        elif admission == 'conversation':
            dependencies['conversation_collector'] = g.collector
        else:
            event_logger = EventLogger(str(tmp_path/'events.db'), source_policy=g.privacy.ledger)
            dependencies['event_logger'] = event_logger
        try:
            with monkeypatch.context() as patch:
                io = forbid_filesystem(patch)
                generator = HabitDatasetGenerator(data_dir=str(root), **dependencies)
                assert generator.training_blocked
                assert generator._user_profiles == {} and generator._examples == []
                io.assert_not_called()
            assert legacy.read_bytes() == original
        finally:
            if event_logger is not None:
                await event_logger.aclose()


@pytest.mark.parametrize('operation', [
    'build_user_profiles', 'generate_dataset', 'export_jsonl',
    'export_with_metadata', 'prepare_for_nightly_training',
])
async def test_private_habit_entrypoints_reject_before_any_io(tmp_path, monkeypatch, operation):
    async with graph(tmp_path) as g:
        generator = HabitDatasetGenerator(
            data_dir=str(tmp_path/'new-habits'), source_policy=g.privacy.ledger,
            event_logger=Mock(), pattern_learner=Mock(),
        )
        # A cached example must not bypass admission in either export method.
        generator._examples.append(HabitExample('private', 'private', 'private', 'preference', 'event', 1.))
        with monkeypatch.context() as patch:
            io = forbid_filesystem(patch)
            with pytest.raises(RuntimeError, match='^private_training_session_unavailable$'):
                getattr(generator, operation)()
            io.assert_not_called()
        generator.event_logger.query_recent.assert_not_called()
        assert not (tmp_path/'new-habits').exists()


async def test_private_nightly_start_and_manual_run_are_inert_with_legacy_history(
    tmp_path, monkeypatch, caplog,
):
    async with graph(tmp_path) as g:
        root = tmp_path/'nightly'
        root.mkdir()
        history = root/'training_history.json'
        original = json.dumps({'sessions': [{
            'id': 'private-history-marker', 'started_at': 1, 'status': 'completed',
            'adapter_path': 'private-adapter-marker',
        }]}).encode()
        history.write_bytes(original)
        config = NightlyConfig(data_dir=str(root), output_dir=str(tmp_path/'models'))
        callbacks = [Mock() for _ in range(3)]
        with monkeypatch.context() as patch:
            io = forbid_filesystem(patch)
            trainer = NightlyTrainer(
                config, source_policy=g.privacy.ledger, alert_callback=callbacks[0],
                unload_callback=callbacks[1], reload_callback=callbacks[2],
            )
            assert trainer.training_blocked and trainer.training_history == []
            assert trainer.start_scheduler() is False
            assert trainer._scheduler_thread is None and not trainer._scheduler_running
            session = await trainer.run_training()
            assert session.status == TrainingStatus.BLOCKED
            assert session.error_message == 'private_training_session_unavailable'
            assert trainer.training_history == []
            status = trainer.get_status()
            assert status['training_available'] is False
            assert status['blocked_reason'] == 'private_training_session_unavailable'
            assert status['current_session']['status'] == 'blocked'
            assert status['last_training'] is None
            assert trainer.get_latest_adapter() is None and trainer.list_adapters() == []
            io.assert_not_called()
        for callback in callbacks:
            callback.assert_not_called()
        assert history.read_bytes() == original
        assert not (tmp_path/'models').exists()
        assert 'private-history-marker' not in caplog.text
        assert 'private-adapter-marker' not in caplog.text


@pytest.mark.parametrize('operation', ['collect', 'dataset', 'script', 'report'])
async def test_private_nightly_direct_entrypoints_cannot_bypass_admission(
    tmp_path, monkeypatch, operation,
):
    async with graph(tmp_path) as g:
        trainer = NightlyTrainer(
            NightlyConfig(data_dir=str(tmp_path/'nightly'), output_dir=str(tmp_path/'models')),
            source_policy=g.privacy.ledger,
        )
        with monkeypatch.context() as patch:
            io = forbid_filesystem(patch)
            spawning = Mock(side_effect=AssertionError('private training spawned a process'))
            patch.setattr('asyncio.create_subprocess_exec', spawning)
            with pytest.raises(RuntimeError, match='^private_training_session_unavailable$'):
                if operation == 'collect':
                    trainer.collect_daily_data()
                elif operation == 'dataset':
                    trainer.prepare_dataset([{'instruction': 'private', 'output': 'private'}])
                elif operation == 'script':
                    await trainer._run_training_script('private.jsonl', [1])
                else:
                    trainer._generate_report()
            io.assert_not_called()
            spawning.assert_not_called()


async def test_managed_conversation_dependency_blocks_entire_legacy_nightly_lifecycle(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        with monkeypatch.context() as patch:
            io = forbid_filesystem(patch)
            generator = HabitDatasetGenerator(
                data_dir=str(tmp_path/'habits'), conversation_collector=g.collector,
            )
            trainer = NightlyTrainer(
                NightlyConfig(data_dir=str(tmp_path/'nightly'), output_dir=str(tmp_path/'models')),
                habit_generator=generator,
            )
            assert trainer.training_blocked and not trainer.start_scheduler()
            assert (await trainer.run_training()).status == TrainingStatus.BLOCKED
            io.assert_not_called()


async def test_managed_stt_dependency_blocks_nightly_before_other_legacy_reads(tmp_path, monkeypatch):
    async with stt_graph(tmp_path) as g:
        with monkeypatch.context() as patch:
            io = forbid_filesystem(patch)
            trainer = NightlyTrainer(
                NightlyConfig(data_dir=str(tmp_path/'nightly'), output_dir=str(tmp_path/'models')),
                stt_correction_collector=g.collector,
            )
            assert trainer.training_blocked and not trainer.start_scheduler()
            assert (await trainer.run_training()).status == TrainingStatus.BLOCKED
            with pytest.raises(RuntimeError, match='private_training_session_unavailable'):
                trainer.collect_daily_data()
            io.assert_not_called()


async def test_managed_context_dependency_blocks_legacy_history_and_training(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        context = ContextManager(source_policy=g.privacy.ledger)
        try:
            with monkeypatch.context() as patch:
                io = forbid_filesystem(patch)
                trainer = NightlyTrainer(
                    NightlyConfig(data_dir=str(tmp_path/'nightly'), output_dir=str(tmp_path/'models')),
                    context_manager=context,
                )
                assert trainer.training_blocked and not trainer.start_scheduler()
                assert (await trainer.run_training()).status == TrainingStatus.BLOCKED
                io.assert_not_called()
        finally:
            await context.aclose()
