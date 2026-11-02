"""Catalog commits cannot publish rejected state or mutable aliases."""

import json
import re

import numpy as np
import pytest

import src.users.user_manager as catalog_module
from src.users.user_manager import PermissionLevel, UserManager


@pytest.mark.parametrize('operation', ['create', 'remove', 'update', 'embedding', 'default_admin'])
@pytest.mark.parametrize('raising', [False, True])
def test_retirement_after_temporary_write_preserves_file_ram_and_version(tmp_path, monkeypatch, operation, raising):
    manager = UserManager(str(tmp_path / 'users.json'))
    user, _ = manager.add_user('Original', PermissionLevel.ADULT)
    before, version = manager.data_path.read_bytes(), manager._version
    snapshot = manager.get_user(user.user_id).to_dict()
    active = [True]
    fsync = catalog_module.os.fsync

    def flush(descriptor):
        fsync(descriptor)
        active[0] = False

    def current():
        if raising and not active[0]:
            raise PermissionError('retired')
        return active[0]

    monkeypatch.setattr(catalog_module.os, 'fsync', flush)
    with pytest.raises(PermissionError):
        if operation == 'create':
            manager.add_user('Denied', PermissionLevel.ADULT, before_save=current)
        elif operation == 'remove':
            manager.remove_user(user.user_id, before_save=current)
        elif operation == 'update':
            manager.update_user(user.user_id, name='Denied', before_save=current)
        elif operation == 'embedding':
            manager.set_voice_embedding(user.user_id, np.ones(4, dtype=np.float32), before_save=current)
        else:
            manager.create_default_admin(before_save=current)
    assert manager.data_path.read_bytes() == before and manager._version == version
    assert manager.get_user(user.user_id).to_dict() == snapshot
    assert [row.user_id for row in manager.get_all_users()] == [user.user_id]
    assert list(tmp_path.glob('.accounts-*')) == []


def test_serialization_failure_does_not_publish_or_truncate(tmp_path):
    manager = UserManager(str(tmp_path / 'users.json'))
    user, _ = manager.add_user('Original', PermissionLevel.ADULT)
    before, version = manager.data_path.read_bytes(), manager._version
    with pytest.raises(TypeError):
        manager.update_user(user.user_id, preferences={'not_json': object()})
    assert manager.data_path.read_bytes() == before and manager._version == version
    assert manager.get_user(user.user_id).preferences == {}
    assert list(tmp_path.glob('.accounts-*')) == []


def test_approved_replace_completes_publication_without_late_authority_check(tmp_path, monkeypatch):
    manager = UserManager(str(tmp_path / 'users.json'))
    user, _ = manager.add_user('Original', PermissionLevel.ADULT)
    version, active = manager._version, [True]
    replace = catalog_module.os.replace

    def commit(source, target):
        replace(source, target)
        active[0] = False

    monkeypatch.setattr(catalog_module.os, 'replace', commit)
    assert manager.update_user(user.user_id, name='Committed', before_save=lambda: active[0])[0]
    assert not active[0] and manager._version == version + 1
    assert manager.get_user(user.user_id).name == 'Committed'
    assert json.loads(manager.data_path.read_text())['users'][0]['name'] == 'Committed'


@pytest.mark.parametrize('retirement', ['downgrade', 'inactive', 'delete'])
@pytest.mark.parametrize('operation', ['create', 'remove'])
def test_detached_requester_cannot_keep_retired_catalog_permissions(tmp_path, retirement, operation):
    manager = UserManager(str(tmp_path / 'users.json'))
    requester, _ = manager.add_user('Requester', PermissionLevel.ADMIN)
    manager.add_user('Other admin', PermissionLevel.ADMIN)
    target, _ = manager.add_user('Target', PermissionLevel.ADULT)
    if retirement == 'downgrade':
        manager.update_user(requester.user_id, permission_level=PermissionLevel.CHILD)
    elif retirement == 'inactive':
        manager.update_user(requester.user_id, is_active=False)
    else:
        assert manager.remove_user(requester.user_id)[0]
    before, version = manager.data_path.read_bytes(), manager._version
    if operation == 'create':
        assert manager.add_user('Denied', PermissionLevel.ADULT, requesting_user=requester)[0] is None
    else:
        assert manager.remove_user(target.user_id, requesting_user=requester)[0] is False
    assert manager.data_path.read_bytes() == before and manager._version == version
    assert manager.get_user(target.user_id) is not None


def test_new_identity_is_not_reused_and_never_contains_name_path_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog_module.time, 'time', lambda: 123.)
    manager = UserManager(str(tmp_path / 'users.json'))
    first, _ = manager.add_user('A/B', PermissionLevel.ADULT)
    assert manager.remove_user(first.user_id)[0]
    second, _ = manager.add_user('A/B', PermissionLevel.ADULT)
    assert first.user_id != second.user_id
    assert re.fullmatch(r'user_[a-f0-9]{32}', second.user_id)


def test_catalog_inputs_outputs_and_serialization_have_no_mutable_aliases(tmp_path):
    manager = UserManager(str(tmp_path / 'users.json'))
    embedding = np.arange(4, dtype=np.float32)
    returned, _ = manager.add_user('Original', PermissionLevel.ADULT, voice_embedding=embedding)
    preferences, allowed = {'nested': {'items': ['original']}}, ['light.original']
    manager.update_user(returned.user_id, preferences=preferences, allowed_entities=allowed)
    before = manager.data_path.read_bytes()
    embedding[:] = 99
    returned.name = 'Changed'
    returned.voice_embedding[:] = 99
    preferences['nested']['items'].append('external')
    allowed.append('light.external')
    for row in [manager.get_user(returned.user_id), manager.get_user_by_name('Original'),
                manager.get_all_users()[0]]:
        serialized = row.to_dict()
        serialized['preferences']['nested']['items'].append('serialized')
        serialized['allowed_entities'].append('light.serialized')
        assert row.preferences['nested']['items'] == ['original']
        row.preferences['nested']['items'].append('dto')
        row.allowed_entities.append('light.dto')
        row.voice_embedding[:] = 12
    manager.get_all_embeddings()[returned.user_id][:] = 13
    current = manager.get_user(returned.user_id)
    assert current.name == 'Original' and current.preferences == {'nested': {'items': ['original']}}
    assert current.allowed_entities == ['light.original']
    np.testing.assert_array_equal(current.voice_embedding, np.arange(4, dtype=np.float32))
    assert manager.data_path.read_bytes() == before


@pytest.mark.parametrize('managed', [False, True])
def test_invalid_later_row_cannot_publish_or_overwrite_a_partial_catalog(tmp_path, managed):
    path = tmp_path / 'users.json'
    path.write_text(json.dumps({'users': [
        {'user_id': 'old', 'name': 'Original', 'permission_level': PermissionLevel.ADULT.value},
        {'user_id': 'invalid', 'name': 'Invalid', 'permission_level': 999}]}))
    before = path.read_bytes()
    manager = UserManager(str(path), voice_registry=object() if managed else None)
    with pytest.raises(RuntimeError, match='unverified_account_catalog'):
        manager.get_all_users()
    with pytest.raises(RuntimeError, match='unverified_account_catalog'):
        manager.add_user('Denied', PermissionLevel.ADULT)
    assert path.read_bytes() == before and manager._version == 0
