"""Canonical account snapshots and minimum administrative projections."""

from dataclasses import FrozenInstanceError
import json
import threading

import numpy as np
import pytest

from src.users.user_manager import PermissionLevel, UserManager


def create(manager, name='Synthetic'):
    user, _ = manager.add_user(name, PermissionLevel.ADULT, voice_embedding=np.ones(8, dtype=np.float32))
    assert user is not None
    return user


def test_active_membership_has_no_catalog_lock_and_never_exposes_private_material(tmp_path):
    manager = UserManager(str(tmp_path / 'users.json'))
    assert manager.active_accounts().available
    user = create(manager)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    result = []

    def hold():
        with manager._lock:
            entered.set()
            assert release.wait(3)

    def read():
        result.append(manager.active_accounts())
        completed.set()

    blocker = threading.Thread(target=hold)
    reader = threading.Thread(target=read)
    blocker.start()
    try:
        assert entered.wait(2)
        reader.start()
        assert completed.wait(0.5), 'canonical guard must not acquire UserManager lock'
    finally:
        release.set()
        blocker.join(3)
        if reader.ident is not None:
            reader.join(3)
    [snapshot] = result
    assert snapshot.active_ids == frozenset({user.user_id})
    assert not hasattr(snapshot, 'users') and not hasattr(snapshot, 'name')
    with pytest.raises(FrozenInstanceError):
        snapshot.available = False


def test_projection_never_copies_voice_or_preferences_and_is_detached(tmp_path, monkeypatch):
    import src.users.user_manager as module

    manager = UserManager(str(tmp_path / 'users.json'))
    user = create(manager)
    manager.update_user(user.user_id, preferences={'synthetic-private': ['never-project']})

    def forbidden(*args, **kwargs):
        raise AssertionError('full user material was touched')

    monkeypatch.setattr(module, 'deepcopy', forbidden)
    monkeypatch.setattr(manager, 'get_all_users', forbidden)
    [account] = manager.project_accounts()
    assert vars(account) == {'id': user.user_id, 'name': 'Synthetic', 'permission_level': 'adult', 'is_active': True}
    with pytest.raises(FrozenInstanceError):
        account.name = 'Changed'
    assert manager._users[user.user_id].name == 'Synthetic'


@pytest.mark.parametrize('change', ['delete_active', 'delete_inactive', 'deactivate'])
@pytest.mark.parametrize('proof', ['absent', 'false', 'failure'])
def test_destructive_catalog_write_requires_durable_retirement(tmp_path, change, proof):
    path = tmp_path / 'users.json'
    manager = UserManager(str(path))
    user = create(manager)
    if change == 'delete_inactive':
        data = json.loads(path.read_text())
        data['users'][0]['is_active'] = False
        path.write_text(json.dumps(data))

    def fail(_):
        raise RuntimeError('synthetic proof unavailable')

    verifier = None if proof == 'absent' else fail if proof == 'failure' else lambda identities: False
    manager = UserManager(str(path), retirement_proof=verifier)
    previous = path.read_bytes(), manager._version, manager.active_accounts()
    with pytest.raises((PermissionError, RuntimeError)):
        if change == 'deactivate':
            manager.update_user(user.user_id, is_active=False)
        else:
            manager.remove_user(user.user_id)
    assert path.read_bytes() == previous[0]
    assert manager._version == previous[1]
    assert manager.active_accounts() is previous[2]
    assert manager.get_user(user.user_id) is not None
    assert not list(tmp_path.glob('.accounts-*'))


def test_approved_retirement_publishes_new_snapshot_and_keeps_old_snapshot_immutable(tmp_path):
    proven = set()
    manager = UserManager(str(tmp_path / 'users.json'), retirement_proof=lambda identities: identities <= proven)
    user = create(manager)
    old = manager.active_accounts()
    proven.add(user.user_id)
    assert manager.remove_user(user.user_id)[0]
    new = manager.active_accounts()
    assert new.available and new.active_ids == frozenset()
    assert new.revision == old.revision + 1
    assert old.active_ids == frozenset({user.user_id})


def test_non_destructive_metadata_changes_need_no_retirement_and_advance_revision(tmp_path):
    def forbidden(_):
        raise AssertionError('no identity is being retired')

    manager = UserManager(str(tmp_path / 'users.json'), retirement_proof=forbidden)
    user = create(manager)
    old = manager.active_accounts()
    assert manager.update_user(user.user_id, name='Renamed', permission_level=PermissionLevel.TEEN)[0]
    assert manager.active_accounts().active_ids == old.active_ids
    assert manager.active_accounts().revision == old.revision + 1


def test_serialization_failure_cannot_publish_candidate_membership(tmp_path):
    manager = UserManager(str(tmp_path / 'users.json'))
    user = create(manager)
    old, content = manager.active_accounts(), manager.data_path.read_bytes()
    with pytest.raises(TypeError):
        manager.update_user(user.user_id, is_active=False, preferences={'unserializable': object()})
    assert manager.active_accounts() is old
    assert manager.data_path.read_bytes() == content
    assert manager.get_user(user.user_id).is_active is True


def test_last_seen_keeps_the_same_membership_snapshot_and_persisted_catalog(tmp_path):
    manager = UserManager(str(tmp_path / 'users.json'))
    user = create(manager)
    old, content, version = manager.active_accounts(), manager.data_path.read_bytes(), manager._version
    manager.update_last_seen(user.user_id)
    assert manager.active_accounts() is old
    assert manager._version == version
    assert manager.data_path.read_bytes() == content


@pytest.mark.parametrize('value', ['false', 1, None])
@pytest.mark.parametrize('managed', [False, True])
def test_non_boolean_activity_invalidates_whole_catalog(tmp_path, value, managed):
    legacy = tmp_path / 'users.json'
    path = tmp_path / 'users_accounts_v1.json' if managed else legacy
    payload = {'users': [{'user_id': 'valid', 'name': 'Valid', 'permission_level': 3, 'is_active': True},
                         {'user_id': 'bad', 'name': 'Bad', 'permission_level': 3, 'is_active': value}]}
    if managed:
        payload['schema'] = 'kza-users-accounts-v1'
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    manager = UserManager(str(legacy), voice_registry=object() if managed else None)
    assert manager.active_accounts().available is False
    assert manager.active_accounts().active_ids == frozenset()
    with pytest.raises(RuntimeError, match='unverified_account_catalog'):
        manager.project_accounts()
    assert path.read_bytes() == original
