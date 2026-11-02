"""Household list selection cannot acquire ownership from an acoustic identity."""

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.application import ApplicationFactories
from src.core.task_outcome import PersistenceState, TaskStatus
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, OwnerType
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake, offline_config
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport


@pytest.mark.parametrize('user_id', ['unknown', 'known-profile'])
async def test_shared_selection_excludes_private_homonyms_before_projection(tmp_path, monkeypatch, user_id):
    async with AsyncExitStack() as stack:
        store = ListStore(str(tmp_path / 'lists.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        private = await store.create_list('compras', OwnerType.USER, user_id)
        private_item = await store.add_item(private.id, 'PRIVATE_ITEM_638')
        shared = await store.create_list('compras', OwnerType.SHARED, None)
        shared_item = await store.add_item(shared.id, 'Artículo compartido.')
        lists, items = [], []
        convert_list, convert_item = store._row_to_list, store._row_to_item

        def project_list(row):
            assert row['id'] != private.id
            lists.append(row['id'])
            return convert_list(row)

        def project_item(row):
            assert row['list_id'] != private.id
            items.append(row['id'])
            return convert_item(row)

        with monkeypatch.context() as patch:
            patch.setattr(store, '_row_to_list', project_list)
            patch.setattr(store, '_row_to_item', project_item)
            assert [x.id for x in await store.get_lists_for_user(user_id, shared_only=True)] == [shared.id]
            assert (await store.find_list_by_name('compras', user_id, shared_only=True)).id == shared.id
            assert await store.get_items(private.id, shared_only=True) == []
            assert [x.id for x in await store.get_items(shared.id, shared_only=True)] == [shared_item.id]
            assert await store.add_item(private.id, 'denied', shared_only=True) is None
            assert await store.remove_item(private_item.id, list_id=private.id, shared_only=True) is False
            assert await store.complete_item(private_item.id, list_id=private.id, shared_only=True) is None
            assert await store.clear_list(private.id, shared_only=True) is False
            assert await store.delete_list(private.id, shared_only=True) is False
        assert lists and items == [shared_item.id]
        assert [x.text for x in await store.get_items(private.id)] == ['PRIVATE_ITEM_638']
        assert (await store.get_items(private.id))[0].completed is False


async def test_missing_shared_list_never_creates_private_default_or_replicates(tmp_path):
    async with AsyncExitStack() as stack:
        store = ListStore(str(tmp_path / 'lists.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        private = await store.create_list('compras', OwnerType.USER, 'unknown')
        manager = ListManager(store)
        manager._ha_sync = MagicMock(sync_add_item=AsyncMock(), sync_remove_item=AsyncMock())
        result = await manager.add_item_with_outcome('unknown', 'pan', shared_only=True)
        assert result.item is None and result.outcome.status is TaskStatus.NEEDS_INPUT
        assert result.outcome.persistence is PersistenceState.NOT_WRITTEN
        assert await manager.get_list('unknown', shared_only=True) is None
        assert await manager.remove_item('unknown', 'pan', shared_only=True) is False
        manager._ha_sync.sync_add_item.assert_not_awaited()
        manager._ha_sync.sync_remove_item.assert_not_awaited()
        assert [x.id for x in await manager.get_all_lists('unknown')] == [private.id]
        with pytest.raises(ValueError, match='shared_list_required'):
            await manager.create_list('unknown', 'denied', shared_only=True)
        shared = await manager.create_list('known-profile', 'compras', shared=True, shared_only=True)
        item = await manager.add_item('known-profile', 'arroz', shared_only=True)
        assert shared.owner_type is OwnerType.SHARED and shared.owner_id is None
        assert item.list_id == shared.id and item.added_by is None
        assert [x.text for x in await store.get_items(private.id)] == []


@pytest.mark.parametrize('operation', ['add', 'remove', 'clear', 'delete'])
async def test_parent_scope_is_rechecked_before_mutation_and_replication(tmp_path, monkeypatch, operation):
    async with AsyncExitStack() as stack:
        store = ListStore(str(tmp_path / 'lists.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        shared = await store.create_list('compras', OwnerType.SHARED, None)
        original_item = await store.add_item(shared.id, 'pan')
        manager = ListManager(store)
        manager._ha_sync = MagicMock(sync_add_item=AsyncMock(), sync_remove_item=AsyncMock())
        name = {'add': 'add_item', 'remove': 'remove_item', 'clear': 'clear_list', 'delete': 'delete_list'}[operation]
        original = getattr(store, name)

        async def change_scope(*args, **kwargs):
            await store._work.write(lambda db: db.execute(
                "UPDATE lists SET owner_type='user', owner_id='unknown' WHERE id=?", (shared.id,)).rowcount)
            return await original(*args, **kwargs)

        monkeypatch.setattr(store, name, change_scope)
        if operation in {'add', 'remove'}:
            method = manager.add_item_with_outcome if operation == 'add' else manager.remove_item_with_outcome
            result = await method('unknown', 'pan', shared_only=True)
            assert result.item is None and result.outcome.status is TaskStatus.NEEDS_INPUT
        else:
            method = manager.clear_list if operation == 'clear' else manager.delete_list
            assert await method('unknown', 'compras', shared_only=True) is False
        assert [x.id for x in await store.get_items(shared.id)] == [original_item.id]
        assert (await store.find_list_by_name('compras', 'unknown')).owner_type is OwnerType.USER
        manager._ha_sync.sync_add_item.assert_not_awaited()
        manager._ha_sync.sync_remove_item.assert_not_awaited()


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_actual_factory_voice_uses_shared_lists_without_private_homonym_fallback(tmp_path, monkeypatch, orchestrated):
    import src.tts.piper_tts as output_module
    import src.vectordb.chroma_sync as index_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, 'SentenceTransformer', lambda *a, **kw: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(output_module, 'sd', backend)
    stt, tts, amp = OfflineSTT(), OfflineTTS(), MagicMock()
    amp.audio_output_device = 7
    factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None,
        wake_detector=OfflineWake, amplifier=lambda **kw: amp)
    transport = TaskTransport()
    async with transport.serve() as url:
        config = offline_config(tmp_path, url, rooms=True)
        config['rooms']['living'] = {'name': 'Living', 'mic_device_index': 0,
            'capture_channel': 1, 'ma1260_zone': 1}
        config['presence'] = {'enabled': False}
        config['tts']['streaming']['default_output_zone'] = 'living'
        config['reminders']['output_zone'] = 'living'
        config['orchestrator']['enabled'] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            store = graph.list_manager._store
            private = await store.create_list('compras', OwnerType.USER, 'unknown')
            private_other = await store.create_list('pendientes', OwnerType.USER, 'known-profile')
            await store.add_item(private.id, 'PRIVATE_UNKNOWN_ITEM_742')
            await store.add_item(private_other.id, 'PRIVATE_KNOWN_ITEM_921')
            shared = await store.create_list('compras', OwnerType.SHARED, None)
            await store.add_item(shared.id, 'Artículo del hogar.')
            if orchestrated:
                await graph.pipeline.request_router._orchestrator.start()

            async def command(phrase):
                stt.text = phrase
                return await graph.pipeline.process_command(CommandEvent(
                    np.full(16000, .1, dtype=np.float32), room_id='living'))

            result = await command('Nexa lee la lista de compras')
            if orchestrated:
                assert result['success'] and 'Artículo del hogar.' in result['response']
                assert len(backend.heard()) > 0
                amp.select_zone.assert_called_with(1)
            denied = await command('Nexa crea una lista de secretos')
            assert not denied['success']
            assert await store.find_list_by_name('secretos', 'unknown') is None
            await command('Nexa lee la lista de pendientes')
            if orchestrated:
                added = await command('Nexa agrega pan a la lista de compras')
                assert added['success']
                assert [x.text for x in await store.get_items(shared.id)] == ['Artículo del hogar.', 'pan']
                assert (await command('Nexa vacía la lista de compras'))['success']
                assert await store.get_items(shared.id) == []
                assert (await command('Nexa borra la lista de compras'))['success']
                assert not (await command('Nexa agrega pan a la lista de compras'))['success']
                assert await store.find_list_by_name('compras', None, shared_only=True) is None
                created = await command('Nexa crea una lista compartida de viajes')
                assert created['success']
                assert (await store.find_list_by_name('viajes', None, shared_only=True)).owner_id is None
            else:
                await command('Nexa vacía la lista de compras')
                await command('Nexa borra la lista de compras')
                assert [x.text for x in await store.get_items(shared.id)] == ['Artículo del hogar.']
            assert [x.text for x in await store.get_items(private.id)] == ['PRIVATE_UNKNOWN_ITEM_742']
            assert [x.text for x in await store.get_items(private_other.id)] == ['PRIVATE_KNOWN_ITEM_921']
            spoken = ' '.join(tts.texts)
            assert 'PRIVATE_UNKNOWN_ITEM_742' not in spoken and 'PRIVATE_KNOWN_ITEM_921' not in spoken
            assert transport.sends == []
