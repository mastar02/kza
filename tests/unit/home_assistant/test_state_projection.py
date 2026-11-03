"""Pure protocol vectors; synthetic states only, with no network or models."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
import gc
import json
from types import MappingProxyType
import weakref

import pytest

import src.home_assistant.state_projection as projection
from src.home_assistant.state_projection import (
    DEFAULT_LIMITS, MAX_ENTITIES, MAX_FRAME_BYTES, REST_SOURCE, REST_SNAPSHOT_SOURCE, STATE_TTL_S, WS_SOURCE,
    EntityObservation, ProjectionLimits, StateFrame,
    apply_delta, apply_initial, apply_rest, apply_rest_failure, apply_rest_snapshot,
    decode_message, decode_states_message, empty_frame, entity_observation, legacy_frame, observation_view,
)


A, B = 'light.synthetic_a', 'sensor.synthetic_b'


def compressed(state='off', *, attributes=None, context='context-a', lc=1., **extra):
    return {'s': state, 'a': {} if attributes is None else attributes,
            'c': context, 'lc': lc, **extra}


def initial(values=None, *, observed_at=100., generation=1, limits=DEFAULT_LIMITS):
    return apply_initial(empty_frame(), {'a': {A: compressed()} if values is None else values},
                         generation=generation, observed_at=observed_at, limits=limits)


def state(frame, identity=A, *, now=100.):
    return entity_observation(frame, identity, now=now).state


def rest_point(identity=A, *, value='off', attributes=None, updated=3., changed=None):
    return {'entity_id': identity, 'state': value,
        'attributes': {} if attributes is None else attributes,
        'context': {'id': 'context-a', 'parent_id': None, 'user_id': None},
        'last_changed': datetime.fromtimestamp(updated if changed is None else changed, UTC).isoformat(),
        'last_updated': datetime.fromtimestamp(updated, UTC).isoformat()}


def test_explicit_empty_initial_is_authoritative_and_does_not_store_arbitrary_absence_ids():
    before = initial()
    frame = apply_initial(before, {'a': {}}, generation=2, observed_at=102.)
    assert frame.initialized and not frame.stream_states and not frame.entries
    assert frame.removed_ids == (A,) and frame.changed_ids == ()
    for identity in (A, B, 'sensor.never_seen'):
        observation = entity_observation(frame, identity, now=103.)
        assert observation.exists is False and observation.current(103.)
        assert observation.source == WS_SOURCE and observation.reason is None
        assert observation.observed_at == 102. and observation.expires_at == 108.
    assert not frame.entries
    expired = entity_observation(frame, A, now=108.)
    assert expired.exists is None and expired.reason == 'observation_expired'
    assert expired.observed_at == 102. and expired.expires_at == 108.
    assert not frame.baseline_current(108.)


def test_initial_decodes_context_and_lu_without_conflating_local_observation_time():
    frame = initial({A: compressed(context='new-context'), B: compressed('23', lu=2.,
        context={'id': 'id-b', 'parent_id': 'parent-b', 'user_id': None})})
    first = state(frame)
    assert first == {'entity_id': A, 'state': 'off', 'attributes': {},
        'context': {'id': 'new-context', 'parent_id': None, 'user_id': None},
        'last_changed': '1970-01-01T00:00:01+00:00',
        'last_updated': '1970-01-01T00:00:01+00:00'}
    assert state(frame, B)['last_updated'] == '1970-01-01T00:00:02+00:00'
    assert frame.entries[A].observed_at == 100. and frame.entries[A].expires_at == 106.
    assert frame.entry_revision(A) == frame.revision
    assert frame.entry_revision('sensor.absent') is None
    assert frame.entries[A].attempted_at is None


def test_delta_preserves_omitted_values_shallow_attributes_and_explicit_null_context():
    frame = initial({A: compressed(attributes={'keep': {'nested': 1}, 'delete': 2, 'null': 0},
        context={'id': 'old', 'parent_id': 'parent', 'user_id': 'user'})})
    changed = apply_delta(frame, {'c': {A: {'+': {'a': {'null': None, 'new': [1, None]},
        'c': {'parent_id': None}}, '-': {'a': ['delete']}}}}, observed_at=101.)
    value = state(changed, now=101.)
    assert value['state'] == 'off'
    assert value['attributes'] == {'keep': {'nested': 1}, 'null': None, 'new': [1, None]}
    assert value['context'] == {'id': 'old', 'parent_id': None, 'user_id': 'user'}
    assert value['last_changed'] == state(frame)['last_changed']
    again = apply_delta(changed, {'c': {A: {'+': {'c': 'next'}}}}, observed_at=102.)
    assert state(again, now=102.)['context'] == {'id': 'next', 'parent_id': None, 'user_id': 'user'}
    assert state(frame)['attributes']['delete'] == 2
    assert changed.changed_ids == (A,) and not changed.removed_ids


def test_lu_changes_only_updated_and_lc_changes_both_with_lc_precedence():
    frame = initial()
    updated = apply_delta(frame, {'c': {A: {'+': {'lu': 2.}}}}, observed_at=101.)
    assert state(updated, now=101.)['last_changed'] == '1970-01-01T00:00:01+00:00'
    assert state(updated, now=101.)['last_updated'] == '1970-01-01T00:00:02+00:00'
    changed = apply_delta(updated, {'c': {A: {'+': {'lc': 3., 'lu': 4.}}}}, observed_at=102.)
    assert state(changed, now=102.)['last_changed'] == '1970-01-01T00:00:03+00:00'
    assert state(changed, now=102.)['last_updated'] == '1970-01-01T00:00:03+00:00'


def test_add_delete_and_empty_change_observe_only_the_entities_in_the_delta():
    frame = initial()
    added = apply_delta(frame, {'a': {B: compressed('on')}}, observed_at=103.)
    assert added.changed_ids == (B,)
    assert added.entries[A].observed_at == 100. and added.entries[B].observed_at == 103.
    assert added.observed_at == frame.observed_at == 100.
    no_value_change = apply_delta(added, {'c': {A: {'+': {}}}}, observed_at=104.)
    assert not no_value_change.changed_ids
    assert no_value_change.entries[A].observed_at == 104.
    removed = apply_delta(no_value_change, {'r': [A]}, observed_at=105.)
    observation = entity_observation(removed, A, now=105.)
    assert observation.exists is False and observation.current(105.) and observation.state is None
    assert removed.removed_ids == (A,) and A not in removed.stream_states
    assert entity_observation(removed, A, now=111.).exists is None


def test_all_payload_and_metadata_are_detached_and_replace_preserves_history():
    wire = {'a': {A: compressed(attributes={'nested': [1, {'x': 2}]})}}
    frame = apply_initial(empty_frame(), wire, generation=1, observed_at=100.)
    wire['a'][A]['a']['nested'][1]['x'] = 9
    observation = frame.entries[A]
    exposed = observation.state
    exposed['attributes']['nested'][1]['x'] = 10
    assert observation.state['attributes']['nested'] == [1, {'x': 2}]
    with pytest.raises(TypeError):
        frame.stream_states[A]['attributes']['other'] = 1
    with pytest.raises(TypeError):
        frame.entries[B] = observation
    with pytest.raises(FrozenInstanceError):
        observation.exists = False
    failed = replace(observation, exists=None, reason='source_unavailable', attempted_at=103.)
    assert failed.state == observation.state and not failed.current(103.)
    assert failed.observed_at == 100. and failed.expires_at == 106.
    assert failed.attempted_at == 103. and observation.current(103.)


def test_frame_constructor_detaches_mappingproxy_backing_dictionaries():
    original = initial()
    stream, entries = dict(original.stream_states), dict(original.entries)
    frame = replace(original, stream_states=MappingProxyType(stream), entries=MappingProxyType(entries))
    stream.clear()
    entries.clear()
    assert A in frame.stream_states and frame.entries[A].state['state'] == 'off'


@pytest.mark.parametrize('now, reason', [(99., 'clock_before_observation'), (106., 'observation_expired')])
def test_expiry_and_clock_skew_preserve_original_times_and_last_known_state(now, reason):
    frame = initial()
    observation = entity_observation(frame, A, now=now)
    assert observation.exists is None and observation.reason == reason
    assert observation.state['state'] == 'off' and observation.to_state(now) is None
    assert observation.observed_at == 100. and observation.expires_at == 106.
    assert frame.entries[A].exists is True


def test_rest_projection_is_separate_from_stream_baseline_and_cas_blocks_resurrection():
    frame = initial({A: compressed(attributes={'from_stream': 1})})
    rest = apply_rest(frame, A, {'entity_id': A, 'state': 'rest', 'attributes': {'rest_only': 2},
        'last_changed': '1970-01-01T00:00:02+00:00', 'last_updated': '1970-01-01T00:00:02+00:00'},
        observed_at=101., expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    assert rest.entries[A].source == REST_SOURCE and rest.entries[A].state['state'] == 'rest'
    assert rest.stream_states[A]['state'] == 'off'
    delta = apply_delta(rest, {'c': {A: {'+': {'a': {'next': 3}, 'lu': 1.5}}}}, observed_at=102.)
    assert dict(delta.stream_states[A]['attributes']) == {'from_stream': 1, 'next': 3}
    assert delta.entries[A].state['attributes'] == {'rest_only': 2}
    assert delta.entries[A].source == REST_SOURCE and delta.entries[A].observed_at == 101.
    deleted = apply_delta(delta, {'r': [A]}, observed_at=103.)
    assert apply_rest(deleted, A, {'entity_id': A, 'state': 'old'}, observed_at=104.,
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A)) is deleted
    rebuilt = apply_initial(deleted, {'a': {}}, generation=2, observed_at=104.)
    assert apply_rest(rebuilt, A, {'entity_id': A, 'state': 'old'}, observed_at=105.,
        expected_generation=1, expected_revision=None) is rebuilt


def test_actual_rest_missing_is_fresh_negative_and_unknown_legacy_is_not_promoted():
    before = empty_frame(generation=3)
    missing = apply_rest(before, A, None, observed_at=100., attempted_at=99.,
        expected_generation=3, expected_revision=None)
    observation = entity_observation(missing, A, now=101.)
    assert observation.exists is False and observation.current(101.) and observation.reason is None
    assert observation.observed_at == 100. and observation.attempted_at == 99.
    assert not missing.initialized and missing.stream_states == {}
    raw = {A: {'entity_id': A, 'state': 'on', 'attributes': {}}}
    legacy = legacy_frame(raw, generation=3, revision=7)
    raw[A]['state'] = 'mutated'
    unknown = entity_observation(legacy, A, now=100.)
    assert unknown.state['state'] == 'on' and unknown.exists is None
    assert unknown.source is unknown.observed_at is unknown.expires_at is None
    assert unknown.reason == 'observation_time_unknown' and unknown.to_state(100.) is None
    with pytest.raises(ValueError):
        apply_rest(before, A, None, observed_at=100., expected_generation=3,
                   expected_revision=None, source=WS_SOURCE)


@pytest.mark.parametrize('rest_overlay', [False, True])
def test_failed_rest_preserves_history_source_and_original_dates(rest_overlay):
    frame = initial({A: compressed(), B: compressed('unchanged')})
    if rest_overlay:
        frame = apply_rest(frame, A, {'entity_id': A, 'state': 'rest', 'attributes': {'nested': [1]}},
            observed_at=101., attempted_at=100.5, expected_generation=frame.generation,
            expected_revision=frame.entry_revision(A))
    before = frame.entries[A]
    failed = apply_rest_failure(frame, A, attempted_at=103., reason='source_denied',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    value = failed.entries[A]
    assert value.exists is None and value.reason == 'source_denied'
    assert value.source == before.source and value.state == before.state
    assert (value.observed_at, value.expires_at) == (before.observed_at, before.expires_at)
    assert value.attempted_at == 103. and value.revision == failed.revision == frame.revision + 1
    assert value.generation == frame.generation and not value.current(103.) and value.to_state(103.) is None
    assert failed.changed_ids == failed.removed_ids == ()
    assert failed.stream_states == frame.stream_states
    assert failed.entries[B].current(103.) and failed.entries[B].revision == frame.entries[B].revision
    assert frame.entries[A].current(103.) and frame.entries[A].attempted_at == before.attempted_at
    if rest_overlay:
        detached = value.state
        detached['attributes']['nested'].append(2)
        assert value.state['attributes']['nested'] == [1]


@pytest.mark.parametrize('explicit_missing', [False, True])
def test_failed_negative_read_preserves_original_deadline(explicit_missing):
    frame = initial({})
    if explicit_missing:
        frame = apply_rest(frame, A, None, observed_at=101., attempted_at=100.5,
            expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    before = entity_observation(frame, A, now=102.)
    assert before.exists is False and before.current(102.)
    failed = apply_rest_failure(frame, A, attempted_at=103., reason='source_unavailable',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    value = entity_observation(failed, A, now=103.)
    assert value.exists is None and value.reason == 'source_unavailable' and value.state is None
    assert value.source == before.source
    assert (value.observed_at, value.expires_at) == (before.observed_at, before.expires_at)
    assert value.attempted_at == 103. and not value.current(103.)
    assert not value.current(value.expires_at + 1.)
    assert failed.changed_ids == failed.removed_ids == ()
    assert entity_observation(frame, A, now=103.).exists is False


def test_first_failed_read_does_not_invent_dates_or_legacy_history_provenance():
    frame = empty_frame(generation=3)
    failed = apply_rest_failure(frame, A, attempted_at=100., reason='source_invalid',
        expected_generation=3, expected_revision=None)
    value = entity_observation(failed, A, now=100.)
    assert value.exists is None and value.reason == 'source_invalid' and value.state is None
    assert value.source == REST_SOURCE and value.attempted_at == 100.
    assert value.observed_at is value.expires_at is None and not value.current(100.)
    assert not frame.entries
    legacy = legacy_frame({A: {'entity_id': A, 'state': 'legacy'}}, generation=3, revision=7)
    failed_legacy = apply_rest_failure(legacy, A, attempted_at=101., reason='source_unavailable',
        expected_generation=3, expected_revision=legacy.entry_revision(A))
    assert failed_legacy.entries[A].state['state'] == 'legacy'
    assert failed_legacy.entries[A].source is None
    assert failed_legacy.entries[A].observed_at is failed_legacy.entries[A].expires_at is None
    assert failed_legacy.entries[A].attempted_at == 101.


@pytest.mark.parametrize('successor', ['delta', 'delete', 'rebuild', 'failure'])
def test_late_failure_cannot_replace_newer_evidence_or_a_new_baseline(successor):
    frame = initial()
    if successor == 'delta':
        newer = apply_delta(frame, {'c': {A: {'+': {'s': 'new'}}}}, observed_at=101.)
    elif successor == 'delete':
        newer = apply_delta(frame, {'r': [A]}, observed_at=101.)
    elif successor == 'rebuild':
        newer = apply_initial(frame, {'a': {A: compressed('new')}}, generation=2, observed_at=101.)
    else:
        newer = apply_rest_failure(frame, A, attempted_at=101., reason='source_denied',
            expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    assert apply_rest_failure(newer, A, attempted_at=102., reason='source_unavailable',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A)) is newer


def test_recovery_requires_failure_revision_and_only_real_success_renews_evidence():
    frame = initial()
    failed = apply_rest_failure(frame, A, attempted_at=101., reason='source_unavailable',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    payload = {'entity_id': A, 'state': 'recovered'}
    assert apply_rest(failed, A, payload, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A)) is failed
    recovered = apply_rest(failed, A, payload, observed_at=102., attempted_at=101.5,
        expected_generation=failed.generation, expected_revision=failed.entry_revision(A))
    value = recovered.entries[A]
    assert value.exists is True and value.current(102.) and value.reason is None
    assert value.source == REST_SOURCE and value.state['state'] == 'recovered'
    assert value.observed_at == 102. and value.expires_at == 108. and value.attempted_at == 101.5
    assert failed.entries[A].observed_at == 100. and not failed.entries[A].current(102.)


def test_capacity_failure_fences_missing_reads_without_invalidating_unrelated_entities():
    limits = replace(DEFAULT_LIMITS, max_entities=2)
    frame = initial({A: compressed(), B: compressed()}, limits=limits)
    missing = 'sensor.absent'
    captured_revision = frame.entry_revision(missing)
    assert captured_revision is None
    failed = apply_rest_failure(frame, missing, attempted_at=101., reason='source_denied',
        expected_generation=frame.generation, expected_revision=captured_revision, limits=limits)
    assert failed.generation == frame.generation and failed.revision == frame.revision + 1
    assert not failed.initialized and not failed.baseline_current(101.)
    assert failed.absence_revision == failed.entry_revision(missing) == failed.revision
    assert len(failed.entries) == 2 and missing not in failed.entries
    assert failed.stream_states == frame.stream_states
    unknown = entity_observation(failed, missing, now=101.)
    assert unknown.exists is None and not unknown.current(101.) and unknown.revision == failed.absence_revision
    for identity in (A, B):
        assert failed.entries[identity].current(101.)
        assert failed.entries[identity].revision == frame.entries[identity].revision
        assert failed.entries[identity].observed_at == frame.entries[identity].observed_at
    assert failed.changed_ids == failed.removed_ids == ()
    assert apply_rest(failed, missing, None, observed_at=102., expected_generation=frame.generation,
        expected_revision=captured_revision, limits=limits) is failed
    assert apply_rest_failure(failed, missing, attempted_at=102., reason='source_invalid',
        expected_generation=frame.generation, expected_revision=captured_revision, limits=limits) is failed
    updated = apply_rest(failed, A, {'entity_id': A, 'state': 'new'}, observed_at=102.,
        expected_generation=failed.generation, expected_revision=failed.entry_revision(A), limits=limits)
    assert updated.absence_revision == failed.absence_revision
    assert updated.entries[B].current(102.) and updated.entries[B].revision == frame.entries[B].revision
    another_id = 'sensor.another_absent'
    again = apply_rest_failure(updated, another_id, attempted_at=103., reason='source_unavailable',
        expected_generation=updated.generation, expected_revision=updated.entry_revision(another_id), limits=limits)
    assert len(again.entries) == 2 and again.absence_revision == again.revision
    assert again.absence_revision != updated.absence_revision
    assert apply_rest(again, missing, None, observed_at=103., expected_generation=updated.generation,
        expected_revision=updated.entry_revision(missing), limits=limits) is again
    rebuilt = apply_initial(again, {'a': {A: compressed()}}, generation=2, observed_at=104., limits=limits)
    assert rebuilt.initialized and rebuilt.absence_revision is None and rebuilt.entry_revision(missing) is None
    assert entity_observation(rebuilt, missing, now=104.).exists is False


def test_capacity_still_allows_failure_publication_for_an_existing_entry():
    limits = replace(DEFAULT_LIMITS, max_entities=1)
    frame = initial(limits=limits)
    failed = apply_rest_failure(frame, A, attempted_at=101., reason='source_denied',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A), limits=limits)
    assert len(failed.entries) == 1 and failed.initialized and failed.absence_revision is None
    assert failed.entries[A].exists is None and failed.entries[A].state['state'] == 'off'
    assert failed.entries[A].observed_at == 100. and failed.entries[A].expires_at == 106.


@pytest.mark.parametrize('overrides', [
    {'attempted_at': True}, {'attempted_at': -1.}, {'attempted_at': float('nan')},
    {'attempted_at': float('inf')}, {'reason': None}, {'reason': True},
    {'reason': 'source_closed'}, {'reason': 'arbitrary_exception'},
    {'expected_generation': True}, {'expected_revision': True},
])
def test_failed_read_rejects_invalid_native_metadata_atomically(overrides):
    frame = initial()
    values = dict(attempted_at=101., reason='source_unavailable',
                  expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    values.update(overrides)
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$') as caught:
        apply_rest_failure(frame, A, **values)
    assert caught.value.__cause__ is caught.value.__context__ is None
    assert frame.entries[A].current(101.) and frame.entries[A].observed_at == 100.


@pytest.mark.parametrize('invalid', [True, -1, 1.5, float('nan'), '1', 1 << 63])
def test_absence_fence_requires_an_exact_bounded_integer(invalid):
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        replace(empty_frame(), absence_revision=invalid)


@pytest.mark.parametrize('source', [REST_SOURCE, REST_SNAPSHOT_SOURCE])
def test_older_ws_advances_baseline_preserving_newer_rest_then_newer_ws_can_publish(source):
    before = initial({A: compressed(attributes={'brightness': 10, 'anchor': 'baseline'})})
    point_two = rest_point(attributes={'brightness': 30, 'rest_only': 'point2'})
    rest = apply_rest(before, A, point_two, observed_at=102., source=source,
        expected_generation=before.generation, expected_revision=before.entry_revision(A))
    stale = apply_delta(rest, {'c': {A: {'+': {'s': 'on', 'a': {'brightness': 20}, 'lc': 2.},
        '-': {'a': ['anchor']}}}}, observed_at=103.)
    assert stale.stream_states[A]['state'] == 'on'
    assert dict(stale.stream_states[A]['attributes']) == {'brightness': 20}
    assert stale.stream_states[A]['last_updated'] == '1970-01-01T00:00:02+00:00'
    selected = stale.entries[A]
    assert selected.state == point_two and selected.source == source and selected.current(103.)
    assert (selected.observed_at, selected.expires_at) == (102., 108.)
    assert selected.revision == stale.revision and stale.entry_revision(A) != rest.entry_revision(A)
    assert stale.changed_ids == stale.removed_ids == ()
    # The actual equal point must not renew the REST deadline either.
    equal = apply_delta(stale, {'c': {A: {'+': {'s': 'off', 'lc': 3.,
        'a': {'brightness': 30, 'rest_only': 'point2'}}}}}, observed_at=104.)
    assert equal.entries[A].state == point_two and equal.entries[A].source == source
    assert (equal.entries[A].observed_at, equal.entries[A].expires_at) == (102., 108.)
    assert equal.changed_ids == equal.removed_ids == ()
    # T3 omits state; only the correctly advanced WS baseline can reconstruct it.
    latest = apply_delta(equal, {'c': {A: {'+': {'a': {'brightness': 31}, 'lu': 4.}}}}, observed_at=105.)
    current = latest.entries[A]
    assert current.source == WS_SOURCE and current.current(105.)
    assert current.state['state'] == 'off'
    assert current.state['attributes'] == {'brightness': 31, 'rest_only': 'point2'}
    assert (current.observed_at, current.expires_at) == (105., 111.)
    assert latest.changed_ids == (A,) and not latest.removed_ids
    assert apply_rest(latest, A, point_two, observed_at=106., expected_generation=equal.generation,
        expected_revision=equal.entry_revision(A)) is latest


@pytest.mark.parametrize('conflict', ['state', 'attributes', 'context', 'last_changed'])
def test_equal_remote_version_with_conflicting_payload_is_unknown_rest_history(conflict):
    frame = initial()
    point = rest_point(attributes={'value': 1}, changed=1.)
    rest = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    change = {'s': 'off', 'a': {'value': 1}, 'lu': 3.}
    if conflict == 'state':
        change['s'] = 'on'
    elif conflict == 'attributes':
        change['a'] = {'value': 2}
    elif conflict == 'context':
        change['c'] = 'different-context'
    else:
        change['lc'] = 3.
    reduced = apply_delta(rest, {'c': {A: {'+': change}}}, observed_at=103.)
    selected = reduced.entries[A]
    assert selected.state == point and selected.source == REST_SOURCE
    assert selected.exists is None and selected.reason == 'source_changed' and not selected.current(103.)
    assert (selected.observed_at, selected.expires_at) == (102., 108.)
    assert selected.revision == reduced.revision and reduced.changed_ids == reduced.removed_ids == ()
    assert reduced.stream_states[A]['last_updated'] == point['last_updated']


def test_equal_point_uses_aware_timestamps_and_ignores_rest_only_last_reported():
    frame = initial()
    point = rest_point(changed=1.)
    point['last_updated'] = '1970-01-01T01:00:03+01:00'
    point['last_changed'] = '1970-01-01T01:00:01+01:00'
    point['last_reported'] = '1970-01-01T00:00:05+00:00'
    rest = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    equal = apply_delta(rest, {'c': {A: {'+': {'lu': 3.}}}}, observed_at=103.)
    assert equal.entries[A].state == point and equal.entries[A].current(103.)
    assert equal.entries[A].source == REST_SOURCE and equal.entries[A].expires_at == 108.
    assert equal.changed_ids == equal.removed_ids == ()


@pytest.mark.parametrize('case', ['missing_version', 'rest_missing', 'ws_removed'])
def test_incomparable_versions_or_negative_observations_preserve_unknown_rest_history(case):
    frame = initial()
    point = None if case == 'rest_missing' else rest_point()
    if case == 'missing_version':
        point.pop('last_updated')
    rest = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    event = {'r': [A]} if case == 'ws_removed' else {'c': {A: {'+': {'s': 'on', 'lc': 4.}}}}
    reduced = apply_delta(rest, event, observed_at=103.)
    selected = reduced.entries[A]
    assert selected.state == point and selected.source == REST_SOURCE
    assert selected.exists is None and selected.reason == 'source_changed' and not selected.current(103.)
    assert (selected.observed_at, selected.expires_at) == (102., 108.)
    assert selected.revision == reduced.revision and reduced.changed_ids == reduced.removed_ids == ()
    assert (A not in reduced.stream_states) == (case == 'ws_removed')


def test_an_older_ws_point_never_renews_expired_rest_or_emits_a_current_callback():
    frame = initial()
    point = rest_point()
    rest = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    reduced = apply_delta(rest, {'c': {A: {'+': {'s': 'on', 'lc': 2.}}}}, observed_at=109.)
    selected = entity_observation(reduced, A, now=109.)
    assert selected.state == point and selected.exists is None and selected.reason == 'observation_expired'
    assert selected.source == REST_SOURCE and selected.observed_at == 102. and selected.expires_at == 108.
    assert reduced.changed_ids == reduced.removed_ids == ()


def test_unknown_undated_first_failure_cannot_block_actual_ws_evidence():
    frame = empty_frame(generation=1)
    failed = apply_rest_failure(frame, A, attempted_at=100., reason='source_unavailable',
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    baseline = apply_initial(failed, {'a': {A: compressed('on')}}, generation=2,
        observed_at=101., admitted_revision=0)
    assert baseline.entries[A].source == WS_SOURCE and baseline.entries[A].current(101.)
    first_read = empty_frame(generation=baseline.generation)
    failed_delta = apply_rest_failure(first_read, B, attempted_at=102., reason='source_denied',
        expected_generation=first_read.generation, expected_revision=first_read.entry_revision(B))
    undated = replace(baseline, entries=MappingProxyType({A: baseline.entries[A],
                                                        B: failed_delta.entries[B]}))
    added = apply_delta(undated, {'a': {B: compressed('new')}}, observed_at=103.)
    assert added.entries[B].source == WS_SOURCE and added.entries[B].current(103.)
    assert added.entries[B].state['state'] == 'new'


def test_old_ws_preserves_failed_dated_rest_history_without_restoring_currentness():
    frame = initial()
    point = rest_point()
    rest = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    failed = apply_rest_failure(rest, A, attempted_at=103., reason='source_denied',
        expected_generation=rest.generation, expected_revision=rest.entry_revision(A))
    old = apply_delta(failed, {'c': {A: {'+': {'s': 'on', 'lc': 2.}}}}, observed_at=104.)
    value = old.entries[A]
    assert value.state == point and value.source == REST_SOURCE and value.exists is None
    assert value.reason == 'source_denied' and value.attempted_at == 103. and not value.current(104.)
    assert (value.observed_at, value.expires_at) == (102., 108.)
    assert value.revision == old.revision and old.changed_ids == old.removed_ids == ()


@pytest.mark.parametrize('negative', [False, True])
def test_new_initial_clears_prior_rest_but_preserves_concurrent_missing_history(negative):
    frame = empty_frame()
    prior = apply_rest(frame, A, rest_point(), observed_at=100.,
        expected_generation=frame.generation, expected_revision=frame.entry_revision(A))
    admitted_revision = prior.revision
    point = None if negative else rest_point(B, value='concurrent')
    concurrent = apply_rest(prior, B, point, observed_at=101., expected_generation=prior.generation,
        expected_revision=prior.entry_revision(B))
    rebuilt = apply_initial(concurrent, {'a': {}}, generation=1, observed_at=102.,
        admitted_revision=admitted_revision)
    assert rebuilt.initialized and not rebuilt.stream_states
    assert A not in rebuilt.entries and entity_observation(rebuilt, A, now=102.).exists is False
    history = entity_observation(rebuilt, B, now=102.)
    assert history.state == point and history.source == REST_SOURCE
    assert history.exists is None and history.reason == 'source_changed' and not history.current(102.)
    assert history.generation == concurrent.entries[B].generation
    assert history.observed_at == 101. and history.expires_at == 107. and history.revision == rebuilt.revision
    assert rebuilt.removed_ids == (A,) and rebuilt.changed_ids == ()
    # The backward-compatible omitted token treats all current entries as prior.
    default_token = apply_initial(concurrent, {'a': {}}, generation=1, observed_at=102.)
    assert not default_token.entries and entity_observation(default_token, B, now=102.).exists is False


def test_new_generation_never_uses_ha_timestamp_order_to_promote_concurrent_rest_history():
    frame = initial()
    token = frame.revision
    point = rest_point(value='rest', updated=3.)
    concurrent = apply_rest(frame, A, point, observed_at=102., expected_generation=frame.generation,
        expected_revision=frame.entry_revision(A))
    rebuilt = apply_initial(concurrent, {'a': {A: compressed('stream', lc=4.)}}, generation=2,
        observed_at=103., admitted_revision=token)
    history = rebuilt.entries[A]
    assert history.state == point and history.exists is None and history.reason == 'source_changed'
    assert history.generation == frame.generation != rebuilt.generation
    assert history.source == REST_SOURCE and history.observed_at == 102. and history.expires_at == 108.
    assert rebuilt.stream_states[A]['state'] == 'stream' and rebuilt.changed_ids == ()
    # A failed read cannot silently grant the old point the new stream epoch.
    failed = apply_rest_failure(rebuilt, A, attempted_at=103.5, reason='source_unavailable',
        expected_generation=rebuilt.generation, expected_revision=rebuilt.entry_revision(A))
    assert failed.entries[A].generation == history.generation
    newer = apply_delta(failed, {'c': {A: {'+': {'s': 'later', 'lc': 5.}}}}, observed_at=104.)
    assert newer.entries[A].state == point and not newer.entries[A].current(104.)
    assert newer.entries[A].reason == 'source_changed' and newer.entries[A].revision == newer.revision
    assert newer.changed_ids == newer.removed_ids == ()
    # Only actual new-epoch REST evidence replaces this retained old history.
    fresh = apply_rest(newer, A, rest_point(value='validated', updated=6.), observed_at=105.,
        expected_generation=newer.generation, expected_revision=newer.entry_revision(A))
    assert fresh.entries[A].generation == rebuilt.generation and fresh.entries[A].current(105.)


@pytest.mark.parametrize('token', [True, -1, 1.5, float('nan'), 1 << 63, 2])
def test_initial_rejects_invalid_or_future_admission_revision(token):
    frame = initial()
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        apply_initial(frame, {'a': {}}, generation=2, observed_at=102., admitted_revision=token)
    assert frame.initialized and frame.entries[A].current(102.)


def test_initial_concurrent_history_and_complete_map_overflow_are_rejected_atomically():
    limits = replace(DEFAULT_LIMITS, max_entities=1)
    frame = empty_frame()
    concurrent = apply_rest(frame, B, rest_point(B), observed_at=101.,
        expected_generation=frame.generation, expected_revision=None, limits=limits)
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        apply_initial(concurrent, {'a': {A: compressed()}}, generation=1, observed_at=102.,
            admitted_revision=frame.revision, limits=limits)
    assert set(concurrent.entries) == {B} and concurrent.entries[B].current(102.)
    assert concurrent.entries[B].source == REST_SOURCE and not concurrent.stream_states


def test_states_list_decoder_is_canonical_detached_and_uses_utf8_byte_boundary():
    row = {'entity_id': A, 'state': 'off'}
    assert decode_states_message(json.dumps([row])) == ({**row, 'attributes': {}},)
    payload = [rest_point(attributes={'unicode': '😀'})]
    wire = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    exact = len(wire)
    parsed = decode_states_message(wire, limits=replace(DEFAULT_LIMITS, max_frame_bytes=exact))
    assert type(parsed) is tuple and parsed == tuple(payload)
    payload[0]['attributes']['unicode'] = 'changed'
    assert parsed[0]['attributes']['unicode'] == '😀'
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        decode_states_message(wire, limits=replace(DEFAULT_LIMITS, max_frame_bytes=exact - 1))
    assert decode_states_message('[]') == ()


@pytest.mark.parametrize('wire', [
    '{}', 'null', '[null]', '[{"entity_id":"light.synthetic_a","state":"off","state":"on"}]',
    '[{"entity_id":"light.synthetic_a","state":"off","attributes":{"x":NaN}}]',
    '[{"entity_id":"light.synthetic_a","state":"off","extra":"private"}]',
    '[{"entity_id":"light.synthetic_a","state":false}]', b'\xff',
])
def test_states_list_decoder_rejects_malformed_native_shapes_with_fixed_error(wire):
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$') as caught:
        decode_states_message(wire)
    assert caught.value.__cause__ is caught.value.__context__ is None


def test_states_list_rejects_duplicate_entities_and_canonical_node_or_entity_overflow():
    with pytest.raises(ValueError):
        decode_states_message(json.dumps([rest_point(), rest_point()]))
    with pytest.raises(ValueError):
        decode_states_message(json.dumps([rest_point(), rest_point(B)]),
            limits=replace(DEFAULT_LIMITS, max_entities=1))
    with pytest.raises(ValueError):
        decode_states_message('[{"entity_id":"light.synthetic_a","state":"off"}]',
            limits=replace(DEFAULT_LIMITS, max_nodes=4))


def test_rest_snapshot_batch_uses_one_revision_original_dates_and_no_inferred_absence():
    frame = initial({A: compressed(), B: compressed('old-b')})
    third = 'sensor.synthetic_c'
    rows = [rest_point(value='rest-a'), rest_point(third, value='rest-c')]
    expected = {identity: frame.entry_revision(identity) for identity in (A, third)}
    batch = apply_rest_snapshot(frame, rows, observed_at=102.25, expires_at=105.75, attempted_at=101.5,
        expected_generation=frame.generation, expected_revisions=MappingProxyType(expected))
    assert batch.revision == frame.revision + 1 and batch.initialized == frame.initialized
    assert batch.observed_at == frame.observed_at and batch.expires_at == frame.expires_at
    assert batch.stream_states == frame.stream_states and batch.removed_ids == ()
    for identity in (A, third):
        value = batch.entries[identity]
        assert value.source == REST_SNAPSHOT_SOURCE and value.exists is True and value.current(102.25)
        assert value.observed_at == 102.25 and value.expires_at == 105.75 and value.attempted_at == 101.5
        assert value.revision == batch.revision
    assert batch.entries[B].state == frame.entries[B].state and batch.entries[B].observed_at == 100.
    assert batch.entries[B].revision == frame.entries[B].revision
    rows[0]['attributes']['changed-later'] = True
    assert batch.entries[A].state['attributes'] == {}
    assert apply_rest_snapshot(batch, (), observed_at=104., expires_at=110., attempted_at=103.,
        expected_generation=batch.generation, expected_revisions={}) is batch


def test_batch_per_entity_cas_keeps_newer_delta_and_applies_only_unchanged_entities():
    frame = initial({A: compressed(), B: compressed('old-b')})
    expected = {identity: frame.entry_revision(identity) for identity in (A, B)}
    newer = apply_delta(frame, {'r': [A]}, observed_at=101.)
    rows = (rest_point(value='stale-a'), rest_point(B, value='new-b'))
    batch = apply_rest_snapshot(newer, rows, observed_at=102., expires_at=108., attempted_at=100.5,
        expected_generation=frame.generation, expected_revisions=expected)
    assert batch.entries[A].exists is False and batch.entries[A].source == WS_SOURCE
    assert batch.entries[A].revision == newer.entries[A].revision
    assert batch.entries[B].state['state'] == 'new-b' and batch.entries[B].source == REST_SNAPSHOT_SOURCE
    assert batch.changed_ids == (B,) and not batch.removed_ids
    assert apply_rest_snapshot(batch, rows, observed_at=103., expires_at=109., attempted_at=102.5,
        expected_generation=frame.generation, expected_revisions=expected) is batch
    assert apply_rest_snapshot(batch, rows, observed_at=103., expires_at=109., attempted_at=102.5,
        expected_generation=frame.generation + 1,
        expected_revisions={identity: batch.entry_revision(identity) for identity in (A, B)}) is batch


@pytest.mark.parametrize('failure', ['missing_token', 'duplicate', 'bad_row', 'bad_revision', 'bad_window'])
def test_rest_batch_rejects_entire_native_input_without_partial_frame_publication(failure):
    frame = initial({A: compressed(), B: compressed()})
    rows = [rest_point(), rest_point(B)]
    expected = {identity: frame.entry_revision(identity) for identity in (A, B)}
    expires_at = 108.
    if failure == 'missing_token':
        expected.pop(B)
    elif failure == 'duplicate':
        rows[1] = rest_point()
    elif failure == 'bad_row':
        rows[1]['attributes']['invalid'] = float('nan')
    elif failure == 'bad_revision':
        expected[B] = True
    else:
        expires_at = 100.
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$') as caught:
        apply_rest_snapshot(frame, rows, observed_at=102., expires_at=expires_at, attempted_at=101.,
            expected_generation=frame.generation, expected_revisions=expected)
    assert caught.value.__cause__ is caught.value.__context__ is None
    assert frame.entries[A].state['state'] == 'off' and frame.entries[B].state['state'] == 'off'
    assert frame.revision == 1 and frame.entries[A].observed_at == 100.


def test_batch_uses_absence_fence_and_enforces_full_cache_cap_without_partial_insert():
    limits = replace(DEFAULT_LIMITS, max_entities=1)
    frame = initial(limits=limits)
    failed = apply_rest_failure(frame, B, attempted_at=101., reason='source_denied',
        expected_generation=frame.generation, expected_revision=None, limits=limits)
    assert failed.absence_revision is not None
    assert apply_rest_snapshot(failed, (rest_point(B),), observed_at=102., expires_at=108., attempted_at=101.5,
        expected_generation=failed.generation, expected_revisions={B: None}, limits=limits) is failed
    with pytest.raises(ValueError):
        apply_rest_snapshot(failed, (rest_point(B),), observed_at=102., expires_at=108., attempted_at=101.5,
            expected_generation=failed.generation, expected_revisions={B: failed.entry_revision(B)}, limits=limits)
    assert set(failed.entries) == {A} and failed.entries[A].current(102.)


def test_rest_snapshot_batch_constructs_one_frame_for_many_entities(monkeypatch):
    rows = tuple(rest_point(f'sensor.batch_{index}') for index in range(64))
    frame = empty_frame()
    calls = []
    original = projection._frame

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(projection, '_frame', counted)
    batch = apply_rest_snapshot(frame, rows, observed_at=100., expires_at=106., attempted_at=99.,
        expected_generation=0, expected_revisions={row['entity_id']: None for row in rows})
    assert len(calls) == 1 and len(batch.entries) == 64 and batch.revision == 1
    assert all(value.revision == batch.revision for value in batch.entries.values())


def test_metadata_views_and_expiry_share_sealed_payload_without_reparsing_it(monkeypatch):
    original = initial().entries[A]
    calls = []
    validate = projection._rest_state

    def counted(*args, **kwargs):
        calls.append(args[0])
        return validate(*args, **kwargs)

    monkeypatch.setattr(projection, '_rest_state', counted)
    expired = original.as_of(106.)
    closed = observation_view(original, exists=None, reason='source_closed', attempted_at=105.)
    assert expired._value is original._value and closed._value is original._value
    assert expired.reason == 'observation_expired' and closed.reason == 'source_closed'
    assert expired.observed_at == closed.observed_at == 100. and expired.expires_at == closed.expires_at == 106.
    assert closed.attempted_at == 105. and not expired.current(106.) and not closed.current(105.)
    assert not calls
    # Public native construction/replacement deliberately remain full validation.
    copied = replace(original)
    assert calls and copied._value is not original._value and copied.state == original.state


@pytest.mark.parametrize('field, invalid', [
    ('exists', 1), ('source', 'invented'), ('reason', 'invented'), ('observed_at', True),
    ('expires_at', 99.), ('attempted_at', float('nan')), ('generation', True), ('revision', True),
    ('entity_id', B), ('_value', {'entity_id': A, 'state': 'off', 'attributes': {'invalid': float('nan')}}),
])
def test_native_dto_mutation_is_revalidated_before_fast_view_or_internal_frame_share(field, invalid):
    frame = initial({A: compressed(), B: compressed()})
    value = frame.entries[A]
    object.__setattr__(value, field, invalid)
    for read in (lambda: value.as_of(200.), lambda: value.current(101.),
                 lambda: observation_view(value, exists=None, reason='source_closed'),
                 lambda: apply_delta(frame, {'c': {B: {'+': {'s': 'new', 'lc': 2.}}}}, observed_at=101.)):
        with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
            read()
    assert frame.entries[B].state['state'] == 'off' and frame.entries[B].observed_at == 100.


def test_native_value_swap_takes_full_validation_and_detaches_aliased_data(monkeypatch):
    value = initial().entries[A]
    native = rest_point(attributes={'nested': [1]})
    object.__setattr__(value, '_value', native)
    seen = []
    validate = projection._rest_state

    def counted(*args, **kwargs):
        seen.append(args[0])
        return validate(*args, **kwargs)

    monkeypatch.setattr(projection, '_rest_state', counted)
    view = observation_view(value, exists=None, reason='source_unavailable')
    assert seen and view._value is not native
    native['attributes']['nested'].append(2)
    assert view.state['attributes']['nested'] == [1]
    # Swapping in a different previously sealed payload also loses this DTO's seal.
    other = initial({A: compressed('other')}).entries[A]
    object.__setattr__(value, '_value', other._value)
    seen.clear()
    second = observation_view(value, exists=None, reason='source_closed')
    assert seen and second._value is not other._value and second.state['state'] == 'other'


def test_internal_delta_only_revalidates_touched_payload_and_public_frame_still_detaches(monkeypatch):
    frame = initial({A: compressed(), **{f'sensor.untouched_{index}': compressed() for index in range(64)}})
    seen = []
    validate = projection._rest_state

    def counted(identity, value, limits):
        seen.append(identity)
        return validate(identity, value, limits)

    monkeypatch.setattr(projection, '_rest_state', counted)
    changed = apply_delta(frame, {'c': {A: {'+': {'s': 'on', 'lc': 2.}}}}, observed_at=101.)
    assert set(seen) == {A} and len(seen) <= 6
    untouched = 'sensor.untouched_0'
    assert changed.entries[untouched] is frame.entries[untouched]
    assert changed.stream_states[untouched] is frame.stream_states[untouched]
    seen.clear()
    public_copy = replace(changed)
    assert untouched in seen and public_copy.entries[untouched] is not changed.entries[untouched]
    assert public_copy.stream_states[untouched] is not changed.stream_states[untouched]


def test_validation_seals_are_weak_and_do_not_retain_old_dto_or_payload():
    value = EntityObservation(A, rest_point())
    value_id, payload_id = id(value), id(value._value)
    value_ref, payload_ref = weakref.ref(value), weakref.ref(value._value)
    view = observation_view(value, exists=None, reason='source_closed')
    del value
    gc.collect()
    assert value_ref() is None and value_id not in projection._OBSERVATION_SEALS
    assert payload_ref() is not None  # The live view deliberately owns this tree.
    del view
    gc.collect()
    assert payload_ref() is None and payload_id not in projection._STATE_SEALS


def test_identical_rebuild_renews_actual_observation_without_inventing_changed_callback():
    before = initial()
    frame = apply_initial(before, {'a': {A: compressed()}}, generation=2, observed_at=102.)
    assert frame.changed_ids == frame.removed_ids == ()
    assert frame.entries[A].observed_at == 102. and frame.entries[A].expires_at == 108.
    assert before.entries[A].observed_at == 100.


@pytest.mark.parametrize('event', [
    {}, {'c': {}}, {'a': {}, 'r': []}, {'a': []}, {'a': None}, {'a': {}, 'extra': None},
])
def test_initial_requires_only_explicit_added_map(event):
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        apply_initial(empty_frame(), event, generation=1, observed_at=100.)


@pytest.mark.parametrize('event', [
    {'c': {B: {'+': {'s': 'on'}}}},
    {'a': {B: compressed()}, 'c': {A: {'+': {'s': None}}}},
    {'c': {A: {'+': {'a': None}}}}, {'c': {A: {'+': {'c': None}}}},
    {'c': {A: {'+': {'c': {'id': None}}}}}, {'c': {A: {'+': {'unknown': 1}}}},
    {'c': {A: {'-': {'c': ['id']}}}}, {'c': {A: {'-': {'a': ['x', 'x']}}}},
    {'a': {A: compressed()}, 'r': [A]}, {'r': [A, A]}, {'r': (A,)},
    {'a': {'not-an-entity': compressed()}}, {'c': {A: {}}}, {'unknown': {}},
])
def test_malformed_delta_is_atomic_even_after_valid_earlier_changes(event):
    frame = initial()
    before = frame.entries[A].state
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        apply_delta(frame, event, observed_at=101.)
    assert frame.entries[A].state == before and frame.revision == 1
    assert set(frame.stream_states) == {A} and frame.entries[A].observed_at == 100.


@pytest.mark.parametrize('field, value', [
    ('lc', True), ('lc', None), ('lc', float('nan')), ('lc', float('inf')),
    ('lc', -1), ('lu', '2'), ('s', 2), ('a', []), ('c', {'id': 'x'}),
])
def test_malformed_full_state_fails_without_exposing_partial_initial(field, value):
    full = compressed()
    full[field] = value
    with pytest.raises(ValueError):
        initial({A: compressed(), B: full})


@pytest.mark.parametrize('count', [MAX_ENTITIES, MAX_ENTITIES + 1])
def test_entity_count_exact_boundary(count):
    assert MAX_ENTITIES == 4096 and MAX_FRAME_BYTES == 4 * 1024 * 1024
    event = {'a': {f'sensor.e{index}': compressed() for index in range(count)}}
    if count == MAX_ENTITIES:
        frame = apply_initial(empty_frame(), event, generation=1, observed_at=100.)
        assert len(frame.entries) == len(frame.stream_states) == count
    else:
        with pytest.raises(ValueError):
            apply_initial(empty_frame(), event, generation=1, observed_at=100.)


def test_overflow_and_retained_tombstones_reject_complete_delta_without_eviction():
    limits = replace(DEFAULT_LIMITS, max_entities=2, max_attributes=2)
    frame = initial({A: compressed(attributes={'one': 1, 'two': 2}), B: compressed()}, limits=limits)
    with pytest.raises(ValueError):
        apply_delta(frame, {'c': {A: {'+': {'a': {'three': 3}}}}}, observed_at=101., limits=limits)
    deleted = apply_delta(frame, {'r': [B]}, observed_at=101., limits=limits)
    with pytest.raises(ValueError):
        apply_delta(deleted, {'a': {'sensor.third': compressed()}}, observed_at=102., limits=limits)
    assert deleted.entries[B].exists is False and set(deleted.stream_states) == {A}
    assert frame.entries[A].state['attributes'] == {'one': 1, 'two': 2}


def test_utf8_frame_and_cache_byte_caps_and_recursive_limits_are_explicit():
    event = {'a': {A: compressed(attributes={'unicode': '😀'})}}
    encoded = json.dumps(event, ensure_ascii=False, separators=(',', ':'))
    exact = len(encoded.encode('utf-8'))
    assert decode_message(encoded, limits=replace(DEFAULT_LIMITS, max_frame_bytes=exact)) == event
    with pytest.raises(ValueError):
        decode_message(encoded, limits=replace(DEFAULT_LIMITS, max_frame_bytes=exact - 1))
    with pytest.raises(ValueError):
        initial(event['a'], limits=replace(DEFAULT_LIMITS, max_cache_bytes=1))
    nested = None
    for _ in range(20):
        nested = [nested]
    with pytest.raises(ValueError):
        initial({A: compressed(attributes={'deep': nested})})
    with pytest.raises(ValueError):
        initial({A: compressed(attributes={'x': 'y' * 65537})})


@pytest.mark.parametrize('raw', [
    b'\xff', '[]', '{"a":{},"a":{}}', '{"a":{"x":NaN}}',
    '{"a":{"x":Infinity}}', '{"a":',
])
def test_strict_json_rejects_malformed_duplicate_nonfinite_and_wrong_root(raw):
    with pytest.raises(ValueError, match='^ha_entity_projection_invalid$'):
        decode_message(raw)


def test_fixed_errors_do_not_retain_raw_parser_context_or_private_frame_locals():
    marker = 'SYNTHETIC_PRIVATE_STATE_CANARY'
    with pytest.raises(ValueError) as caught:
        decode_message('{"private":"' + marker + '",')
    error = caught.value
    assert error.__context__ is error.__cause__ is None
    assert str(error) == 'ha_entity_projection_invalid'
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_globals.get('__name__') == 'src.home_assistant.state_projection':
            assert marker not in repr(trace.tb_frame.f_locals)
        trace = trace.tb_next


@pytest.mark.parametrize('observed_at, ttl_s', [
    (True, 6.), (-1., 6.), (float('nan'), 6.), (100., False), (100., 0.), (100., float('inf')),
])
def test_observation_window_never_coerces_or_accepts_nonfinite_values(observed_at, ttl_s):
    with pytest.raises(ValueError):
        apply_initial(empty_frame(), {'a': {}}, generation=1, observed_at=observed_at, ttl_s=ttl_s)


def test_public_observation_rejects_unknown_reason_or_source():
    with pytest.raises(ValueError):
        EntityObservation(A, reason='arbitrary_exception')
    with pytest.raises(ValueError):
        EntityObservation(A, source='invented')
    assert ProjectionLimits().max_depth == 16 and STATE_TTL_S == 6.
    assert isinstance(empty_frame(), StateFrame)
