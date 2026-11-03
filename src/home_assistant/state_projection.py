"""Bounded pure projection of Home Assistant subscribe_entities messages.

Protocol sources: contributors.home-assistant.io/core/internals/websocket-api/
and the official core websocket_api/{commands,messages}.py, core.py and
home-assistant-js-websocket/lib/states.ts. These describe upstream semantics,
not the installed HA version. Observations are never action authority.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
import json
import math
import re
from threading import RLock
from types import MappingProxyType
import weakref


MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ENTITIES = 4096
STATE_TTL_S = 6.0
WS_SOURCE = 'ha-ws-entities'
REST_SOURCE = 'ha-rest-entity'
REST_SNAPSHOT_SOURCE = 'ha-rest-states'
LEGACY_WS_SOURCE = 'ha-ws-state-changed'
_SOURCES = {WS_SOURCE, REST_SOURCE, REST_SNAPSHOT_SOURCE, LEGACY_WS_SOURCE}
_REASONS = {None, 'source_unavailable', 'source_denied', 'source_invalid',
    'observation_expired', 'source_closed', 'clock_before_observation',
    'source_changed', 'observation_time_unknown'}
_ERROR = 'ha_entity_projection_invalid'
_ENTITY_ID = re.compile(r'^(?!.+__)(?!_)[a-z0-9_]+(?<!_)\.(?!_)[a-z0-9_]+(?<!_)$')
_MAX_COUNTER = (1 << 63) - 1
_MAX_EPOCH = 253402300799
_KEEP = object()
_FAILURE_REASONS = {'source_unavailable', 'source_denied', 'source_invalid'}


@dataclass(frozen=True, slots=True)
class ProjectionLimits:
    max_frame_bytes: int = MAX_FRAME_BYTES
    max_entities: int = MAX_ENTITIES
    max_depth: int = 16
    max_nodes: int = 262144
    max_container_items: int = 4096
    max_attributes: int = 1024
    max_string_bytes: int = 65536
    max_attribute_key_bytes: int = 256
    max_cache_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(_ERROR)


DEFAULT_LIMITS = ProjectionLimits()
_SEAL_LOCK = RLock()
_STATE_SEALS: dict[int, tuple] = {}
_OBSERVATION_SEALS: dict[int, tuple] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class _OwnedState(Mapping):
    """Private immutable owned tree; external native input never confers a seal."""
    _data: Mapping = field(repr=False)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def _remember(registry, value, *metadata):
    identity = id(value)

    def collected(reference):
        with _SEAL_LOCK:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

    reference = weakref.ref(value, collected)
    with _SEAL_LOCK:
        registry[identity] = (reference, *metadata)


def _state_seal(value):
    if type(value) is not _OwnedState:
        return None
    with _SEAL_LOCK:
        record = _STATE_SEALS.get(id(value))
        return record if record is not None and record[0]() is value else None


def _owned_state(value):
    owned = _OwnedState(value)
    _remember(_STATE_SEALS, owned, _encoded_size(_thaw(value)))
    return owned


def _remember_observation(value):
    payload = None if value._value is None else weakref.ref(value._value)
    _remember(_OBSERVATION_SEALS, value, payload)


def _observation_sealed(value):
    with _SEAL_LOCK:
        record = _OBSERVATION_SEALS.get(id(value))
        return (record is not None and record[0]() is value
                and (value._value is None if record[1] is None else
                     value._value is not None and record[1]() is value._value)
                and (value._value is None or _state_seal(value._value) is not None))


def _safe(function):
    """Discard raw frames and parser errors before raising a fixed code."""
    @wraps(function)
    def call(*args, **kwargs):
        failed = False
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            failed = True
        finally:
            args, kwargs = (), {}
        if failed:
            raise ValueError(_ERROR)
    return call


def _counter(value):
    if type(value) is not int or not 0 <= value <= _MAX_COUNTER:
        raise ValueError(_ERROR)


def _instant(value):
    if type(value) not in (int, float) or not 0 <= value <= _MAX_EPOCH or not math.isfinite(value):
        raise ValueError(_ERROR)
    return float(value)


def _window(observed_at, ttl_s):
    observed_at = _instant(observed_at)
    if type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or ttl_s <= 0:
        raise ValueError(_ERROR)
    expiry = observed_at + ttl_s
    if not math.isfinite(expiry) or not observed_at < expiry <= _MAX_EPOCH:
        raise ValueError(_ERROR)
    return observed_at, expiry


def _text(value, maximum, *, empty=True):
    if type(value) is not str or (not empty and not value) or len(value.encode('utf-8')) > maximum:
        raise ValueError(_ERROR)
    return value


def _entity_id(value):
    _text(value, 255, empty=False)
    if _ENTITY_ID.fullmatch(value) is None:
        raise ValueError(_ERROR)
    return value


def _tree(value, limits, *, depth=0, budget=None, active=None, freeze=False):
    budget = [limits.max_nodes] if budget is None else budget
    active = set() if active is None else active
    budget[0] -= 1
    if budget[0] < 0 or depth > limits.max_depth:
        raise ValueError(_ERROR)
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -_MAX_COUNTER - 1 <= value <= _MAX_COUNTER:
            raise ValueError(_ERROR)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(_ERROR)
        return value
    if type(value) is str:
        return _text(value, limits.max_string_bytes)
    if type(value) not in (dict, list):
        raise ValueError(_ERROR)
    if len(value) > limits.max_container_items or id(value) in active:
        raise ValueError(_ERROR)
    active.add(id(value))
    try:
        if type(value) is dict:
            copied = {}
            for key, item in value.items():
                _text(key, limits.max_string_bytes)
                copied[key] = _tree(item, limits, depth=depth + 1, budget=budget,
                                    active=active, freeze=freeze)
            return MappingProxyType(copied) if freeze else copied
        copied = [_tree(item, limits, depth=depth + 1, budget=budget,
                        active=active, freeze=freeze) for item in value]
        return tuple(copied) if freeze else copied
    finally:
        active.remove(id(value))


def _thaw(value):
    if type(value) in (MappingProxyType, _OwnedState):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8'))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _constant(_value):
    raise ValueError(_ERROR)


def _decode_document(raw, limits):
    if type(limits) is not ProjectionLimits or type(raw) not in (str, bytes):
        raise ValueError(_ERROR)
    if type(raw) is bytes:
        if not 1 <= len(raw) <= limits.max_frame_bytes:
            raise ValueError(_ERROR)
        raw = raw.decode('utf-8')
    if not raw or len(raw.encode('utf-8')) > limits.max_frame_bytes:
        raise ValueError(_ERROR)
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
    return _tree(value, limits)


@_safe
def decode_message(raw: str | bytes, *, limits: ProjectionLimits = DEFAULT_LIMITS) -> dict:
    """Strict bounded loader for one object-shaped protocol message."""
    value = _decode_document(raw, limits)
    if type(value) is not dict:
        raise ValueError(_ERROR)
    return value


def _snapshot_states(states, limits):
    if type(limits) is not ProjectionLimits or type(states) not in (tuple, list):
        raise ValueError(_ERROR)
    if len(states) > limits.max_entities:
        raise ValueError(_ERROR)
    states = _tree(list(states), limits)
    if _encoded_size(states) > limits.max_frame_bytes:
        raise ValueError(_ERROR)
    result, identities = [], set()
    for value in states:
        if type(value) is not dict or 'entity_id' not in value:
            raise ValueError(_ERROR)
        identity = _entity_id(value['entity_id'])
        if identity in identities:
            raise ValueError(_ERROR)
        identities.add(identity)
        result.append(_rest_state(identity, value, limits))
    if _encoded_size(result) > limits.max_frame_bytes:
        raise ValueError(_ERROR)
    return tuple(_tree(result, limits))


@_safe
def decode_states_message(raw: str | bytes, *, limits: ProjectionLimits = DEFAULT_LIMITS) -> tuple[dict, ...]:
    """Decode a bounded REST states list into detached canonical state rows."""
    value = _decode_document(raw, limits)
    if type(value) is not list:
        raise ValueError(_ERROR)
    return _snapshot_states(value, limits)


def _event(event, limits, *, initial):
    if type(limits) is not ProjectionLimits or type(event) is not dict:
        raise ValueError(_ERROR)
    copied = _tree(event, limits)
    if _encoded_size(copied) > limits.max_frame_bytes:
        raise ValueError(_ERROR)
    if initial:
        if set(copied) != {'a'} or type(copied['a']) is not dict:
            raise ValueError(_ERROR)
    elif not copied or not set(copied) <= {'a', 'c', 'r'}:
        raise ValueError(_ERROR)
    for name in ('a', 'c'):
        if name in copied and type(copied[name]) is not dict:
            raise ValueError(_ERROR)
    removed = copied.get('r', [])
    if type(removed) is not list or len(removed) != len(set(removed)):
        raise ValueError(_ERROR)
    groups = [set(copied.get(name, {})) for name in ('a', 'c')] + [set(removed)]
    if any(groups[index] & groups[other] for index in range(3) for other in range(index)):
        raise ValueError(_ERROR)
    for group in groups:
        for identity in group:
            _entity_id(identity)
    return copied


def _timestamp(value):
    return datetime.fromtimestamp(_instant(value), UTC).isoformat()


def _context(value, *, previous=None):
    if type(value) is str:
        identity = _text(value, 128, empty=False)
        result = dict(previous) if previous is not None else {'parent_id': None, 'user_id': None}
        result['id'] = identity
        return result
    if type(value) is not dict or not set(value) <= {'id', 'parent_id', 'user_id'}:
        raise ValueError(_ERROR)
    if previous is None and set(value) != {'id', 'parent_id', 'user_id'}:
        raise ValueError(_ERROR)
    result = dict(previous) if previous is not None else {}
    for key, item in value.items():
        if item is None and key != 'id':
            result[key] = None
        else:
            result[key] = _text(item, 128, empty=False)
    return result


def _attributes(value, limits):
    if type(value) is not dict or len(value) > limits.max_attributes:
        raise ValueError(_ERROR)
    for key in value:
        _text(key, limits.max_attribute_key_bytes)
    return _tree(value, limits)


def _full(identity, value, limits):
    if (type(value) is not dict or not {'s', 'a', 'c', 'lc'} <= set(value)
            or not set(value) <= {'s', 'a', 'c', 'lc', 'lu'}):
        raise ValueError(_ERROR)
    state = _text(value['s'], 1024)
    if len(state) > 255:
        raise ValueError(_ERROR)
    last_changed = _timestamp(value['lc'])
    return {'entity_id': identity, 'state': state, 'attributes': _attributes(value['a'], limits),
        'context': _context(value['c']), 'last_changed': last_changed,
        'last_updated': _timestamp(value['lu']) if 'lu' in value else last_changed}


def _change(previous, diff, limits):
    if type(diff) is not dict or not diff or not set(diff) <= {'+', '-'}:
        raise ValueError(_ERROR)
    added, removed = diff.get('+', {}), diff.get('-', {})
    if (type(added) is not dict or not set(added) <= {'s', 'a', 'c', 'lc', 'lu'}
            or type(removed) is not dict or not set(removed) <= {'a'}):
        raise ValueError(_ERROR)
    state = _thaw(previous)
    if 's' in added:
        state['state'] = _text(added['s'], 1024)
        if len(state['state']) > 255:
            raise ValueError(_ERROR)
    if 'c' in added:
        state['context'] = _context(added['c'], previous=state['context'])
    if 'lc' in added:
        state['last_changed'] = state['last_updated'] = _timestamp(added['lc'])
        if 'lu' in added:
            _timestamp(added['lu'])  # lc dominates lu in the official reducer.
    elif 'lu' in added:
        state['last_updated'] = _timestamp(added['lu'])
    if 'a' in added:
        state['attributes'].update(_attributes(added['a'], limits))
    if 'a' in removed:
        keys = removed['a']
        if type(keys) is not list or len(keys) != len(set(keys)):
            raise ValueError(_ERROR)
        for key in keys:
            _text(key, limits.max_attribute_key_bytes)
            state['attributes'].pop(key, None)
    if len(state['attributes']) > limits.max_attributes:
        raise ValueError(_ERROR)
    return state


def _rest_state(identity, value, limits):
    if type(value) is not dict or value.get('entity_id') != identity or 'state' not in value:
        raise ValueError(_ERROR)
    if not set(value) <= {'entity_id', 'state', 'attributes', 'context',
                          'last_changed', 'last_updated', 'last_reported'}:
        raise ValueError(_ERROR)
    state = _tree(value, limits)
    _text(state['state'], 1024)
    if len(state['state']) > 255:
        raise ValueError(_ERROR)
    state['attributes'] = _attributes(state.get('attributes', {}), limits)
    if 'context' in state:
        state['context'] = _context(state['context'])
    for key in ('last_changed', 'last_updated', 'last_reported'):
        if key in state:
            instant = datetime.fromisoformat(_text(state[key], 128))
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError(_ERROR)
    if _encoded_size(state) > limits.max_frame_bytes:
        raise ValueError(_ERROR)
    return state


def _observation_metadata(value):
    _entity_id(value.entity_id)
    _counter(value.generation)
    if value.revision is not None:
        _counter(value.revision)
    if value.exists is not None and type(value.exists) is not bool:
        raise ValueError(_ERROR)
    if (value.source is not None and type(value.source) is not str and not isinstance(value.source, StrEnum)
            or value.reason is not None and type(value.reason) is not str and not isinstance(value.reason, StrEnum)):
        raise ValueError(_ERROR)
    source = None if value.source is None else str(value.source)
    reason = None if value.reason is None else str(value.reason)
    if (source is not None and source not in _SOURCES) or reason not in _REASONS:
        raise ValueError(_ERROR)
    if (value.observed_at is None) != (value.expires_at is None):
        raise ValueError(_ERROR)
    if value.observed_at is not None:
        _instant(value.observed_at)
        _instant(value.expires_at)
        if value.expires_at <= value.observed_at:
            raise ValueError(_ERROR)
    if value.attempted_at is not None:
        _instant(value.attempted_at)
    if value.exists is not None and (source is None or value.observed_at is None or reason is not None):
        raise ValueError(_ERROR)
    if (value.exists is True and value._value is None) or (value.exists is False and value._value is not None):
        raise ValueError(_ERROR)
    if _state_seal(value._value) is not None and value._value['entity_id'] != value.entity_id:
        raise ValueError(_ERROR)
    return source, reason


def _validated_observation(value):
    if type(value) is not EntityObservation:
        raise ValueError(_ERROR)
    _observation_metadata(value)
    return value if _observation_sealed(value) else replace(value)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class EntityObservation:
    entity_id: str
    _value: object = field(default=None, repr=False)
    exists: bool | None = None
    source: str | None = None
    observed_at: float | None = None
    expires_at: float | None = None
    attempted_at: float | None = None
    reason: str | None = 'observation_time_unknown'
    generation: int = 0
    revision: int | None = None

    def __post_init__(self):
        source, reason = _observation_metadata(self)
        object.__setattr__(self, 'source', source)
        object.__setattr__(self, 'reason', reason)
        if self._value is not None:
            value = _rest_state(self.entity_id, _thaw(self._value), DEFAULT_LIMITS)
            object.__setattr__(self, '_value', _owned_state(_tree(value, DEFAULT_LIMITS, freeze=True)))
        _remember_observation(self)

    @property
    @_safe
    def state(self) -> dict | None:
        """Return a fresh detached legacy-shaped dictionary, including history."""
        value = _validated_observation(self)
        return None if value._value is None else _thaw(value._value)

    @_safe
    def current(self, now: float) -> bool:
        """Whether positive or negative evidence is valid at this epoch."""
        value = _validated_observation(self)
        now = _instant(now)
        return (value.exists is not None and value.reason is None and value.observed_at is not None
                and value.observed_at <= now < value.expires_at)

    @_safe
    def as_of(self, now: float) -> 'EntityObservation':
        """Return a dated view without renewing evidence or discarding history."""
        value = _validated_observation(self)
        now = _instant(now)
        if value.current(now) or value.exists is None:
            return value
        reason = ('observation_time_unknown' if value.observed_at is None else
                  'clock_before_observation' if now < value.observed_at else 'observation_expired')
        return observation_view(value, exists=None, reason=reason)

    def to_state(self, now: float) -> dict | None:
        """Return detached positive current state, otherwise no current value."""
        return self.state if self.current(now) and self.exists else None


@_safe
def observation_view(value: EntityObservation, *, exists: object = _KEEP, source: object = _KEEP,
                     observed_at: object = _KEEP, expires_at: object = _KEEP,
                     attempted_at: object = _KEEP, reason: object = _KEEP,
                     generation: object = _KEEP, revision: object = _KEEP) -> EntityObservation:
    """Validate a metadata-only view while sharing only a sealed immutable payload.

    Public constructors and dataclasses.replace still validate/copy full native
    payloads. A mutated DTO loses its seal and takes that normal validation path.
    Dates are changed only when explicitly supplied; no deadline is renewed.
    """
    value = _validated_observation(value)
    fields = {name: getattr(value, name) for name in EntityObservation.__dataclass_fields__}
    updates = dict(exists=exists, source=source, observed_at=observed_at, expires_at=expires_at,
                   attempted_at=attempted_at, reason=reason, generation=generation, revision=revision)
    fields.update({name: item for name, item in updates.items() if item is not _KEEP})
    result = object.__new__(EntityObservation)
    for name, item in fields.items():
        object.__setattr__(result, name, item)
    canonical_source, canonical_reason = _observation_metadata(result)
    object.__setattr__(result, 'source', canonical_source)
    object.__setattr__(result, 'reason', canonical_reason)
    _remember_observation(result)
    return result


def _frame_metadata(value):
    _counter(value.generation)
    _counter(value.revision)
    if value.absence_revision is not None:
        _counter(value.absence_revision)
    if (type(value.initialized) is not bool or type(value.stream_states) is not MappingProxyType
            or type(value.entries) is not MappingProxyType):
        raise ValueError(_ERROR)
    if len(value.stream_states) > MAX_ENTITIES or len(value.entries) > MAX_ENTITIES:
        raise ValueError(_ERROR)
    if (value.observed_at is None) != (value.expires_at is None):
        raise ValueError(_ERROR)
    if value.observed_at is not None:
        _instant(value.observed_at)
        _instant(value.expires_at)
        if value.expires_at <= value.observed_at:
            raise ValueError(_ERROR)
    if value.initialized and value.observed_at is None:
        raise ValueError(_ERROR)
    for identities in (value.changed_ids, value.removed_ids):
        if type(identities) is not tuple or len(identities) != len(set(identities)):
            raise ValueError(_ERROR)
        for identity in identities:
            _entity_id(identity)


@dataclass(frozen=True, slots=True)
class StateFrame:
    generation: int = 0
    revision: int = 0
    initialized: bool = False
    stream_states: Mapping = field(default_factory=lambda: MappingProxyType({}), repr=False)
    entries: Mapping = field(default_factory=lambda: MappingProxyType({}), repr=False)
    observed_at: float | None = None
    expires_at: float | None = None
    changed_ids: tuple[str, ...] = ()
    removed_ids: tuple[str, ...] = ()
    absence_revision: int | None = None

    def __post_init__(self):
        _frame_metadata(self)
        for key, value in self.entries.items():
            if type(value) is not EntityObservation or key != value.entity_id:
                raise ValueError(_ERROR)
        stream = {}
        for key, value in self.stream_states.items():
            _entity_id(key)
            state = _rest_state(key, _thaw(value), DEFAULT_LIMITS)
            stream[key] = _owned_state(_tree(state, DEFAULT_LIMITS, freeze=True))
        # MappingProxyType alone does not detach somebody else's backing dict.
        # Own each admitted state/observation rather than relying on that alias.
        entries = {key: replace(value) for key, value in self.entries.items()}
        if _cache_size(stream, entries) > DEFAULT_LIMITS.max_cache_bytes:
            raise ValueError(_ERROR)
        object.__setattr__(self, 'stream_states', MappingProxyType(stream))
        object.__setattr__(self, 'entries', MappingProxyType(entries))

    def entry_revision(self, entity_id: str) -> int | None:
        """Read one entity token, or the bounded implicit-absence fence."""
        _entity_id(entity_id)
        value = self.entries.get(entity_id)
        return self.absence_revision if value is None else value.revision

    def baseline_current(self, now: float) -> bool:
        """Whether original complete membership still proves implicit absence."""
        now = _instant(now)
        return self.initialized and self.observed_at <= now < self.expires_at


def _payload_size(value):
    if value is None:
        return 4
    seal = _state_seal(value)
    if seal is None:
        raise ValueError(_ERROR)
    return seal[1]


def _cache_size(stream, entries):
    stream_size = 2 + max(0, len(stream) - 1)
    stream_size += sum(_encoded_size(key) + 1 + _payload_size(value) for key, value in stream.items())
    return stream_size + sum(_payload_size(value._value) for value in entries.values())


def _frame(previous, stream, entries, *, limits, generation=None, initialized=None,
           observed_at=None, expires_at=None, changed=(), removed=(), absence_revision=_KEEP):
    if len(stream) > limits.max_entities or len(entries) > limits.max_entities:
        raise ValueError(_ERROR)
    owned_stream, owned_entries = {}, {}
    for key, value in stream.items():
        _entity_id(key)
        if _state_seal(value) is None or value['entity_id'] != key:
            state = _rest_state(key, _thaw(value), DEFAULT_LIMITS)
            value = _owned_state(_tree(state, DEFAULT_LIMITS, freeze=True))
        owned_stream[key] = value
    for key, value in entries.items():
        value = _validated_observation(value)
        if key != value.entity_id:
            raise ValueError(_ERROR)
        owned_entries[key] = value
    if _cache_size(owned_stream, owned_entries) > min(limits.max_cache_bytes, DEFAULT_LIMITS.max_cache_bytes):
        raise ValueError(_ERROR)
    result = object.__new__(StateFrame)
    fields = dict(generation=previous.generation if generation is None else generation,
        revision=previous.revision + 1, initialized=previous.initialized if initialized is None else initialized,
        stream_states=MappingProxyType(owned_stream), entries=MappingProxyType(owned_entries),
        observed_at=previous.observed_at if observed_at is None else observed_at,
        expires_at=previous.expires_at if expires_at is None else expires_at,
        changed_ids=tuple(sorted(changed)), removed_ids=tuple(sorted(removed)),
        absence_revision=previous.absence_revision if absence_revision is _KEEP else absence_revision)
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    _frame_metadata(result)
    return result


def _rest_overlay(value):
    return (value is not None and value.source in {REST_SOURCE, REST_SNAPSHOT_SOURCE}
            and value.observed_at is not None and value.expires_at is not None)


def _state_version(value):
    if value is None or 'last_updated' not in value:
        return None
    return datetime.fromisoformat(value['last_updated'])


def _same_state_point(left, right):
    # last_reported is absent from the compact protocol and conveys no state
    # difference. Do not require equivalent timestamp spellings to match.
    for key in ('entity_id', 'state', 'attributes', 'context'):
        if left.get(key) != right.get(key):
            return False
    left_changed, right_changed = left.get('last_changed'), right.get('last_changed')
    if left_changed is None or right_changed is None:
        return left_changed is right_changed
    return datetime.fromisoformat(left_changed) == datetime.fromisoformat(right_changed)


def _select_stream(previous, incoming, *, revision):
    if previous is not None:
        previous = _validated_observation(previous)
    if not _rest_overlay(previous):
        return incoming
    # Reconnection is not proof of an unchanged HA clock/epoch. Keep the
    # history's original generation so later deltas cannot silently confer it.
    left, right = _state_version(previous._value), _state_version(incoming._value)
    comparable = previous.generation == incoming.generation and left is not None and right is not None
    if comparable and (right < left or right == left and _same_state_point(previous._value, incoming._value)):
        return observation_view(previous, revision=revision)
    if comparable and right > left:
        return incoming
    return observation_view(previous, exists=None, reason='source_changed', revision=revision)


def _selection_changes(previous, selected, *, now):
    if not selected.current(now):
        return False, False
    if selected.exists is False:
        return False, True
    return previous is None or previous._value != selected._value, False


def empty_frame(generation: int = 0, revision: int = 0) -> StateFrame:
    """Create an empty unknown frame without inventing observation times."""
    return StateFrame(generation=generation, revision=revision)


@_safe
def legacy_frame(states: dict[str, dict], *, generation: int = 0, revision: int = 0,
                 limits: ProjectionLimits = DEFAULT_LIMITS) -> StateFrame:
    """Own legacy history as unknown; raw assignment cannot confer freshness."""
    if type(states) is not dict or len(states) > limits.max_entities:
        raise ValueError(_ERROR)
    states = _tree(states, limits)
    if _encoded_size(states) > limits.max_cache_bytes:
        raise ValueError(_ERROR)
    entries = {}
    for identity, state in states.items():
        _entity_id(identity)
        value = _rest_state(identity, state, limits)
        entries[identity] = EntityObservation(identity, _tree(value, limits, freeze=True),
            generation=generation, revision=revision)
    return StateFrame(generation=generation, revision=revision, entries=MappingProxyType(entries))


@_safe
def apply_initial(frame: StateFrame, event: dict, *, generation: int, observed_at: float,
                  ttl_s: float = STATE_TTL_S, limits: ProjectionLimits = DEFAULT_LIMITS,
                  admitted_revision: int | None = None) -> StateFrame:
    """Replace membership, retaining REST reads concurrent with dispatch as history.

    The optional token must be captured immediately before sending the
    subscription. The default treats existing entries as prior observations.
    """
    if type(frame) is not StateFrame:
        raise ValueError(_ERROR)
    _counter(generation)
    if generation < frame.generation or (frame.initialized and generation == frame.generation):
        raise ValueError(_ERROR)
    if admitted_revision is None:
        admitted_revision = frame.revision
    _counter(admitted_revision)
    if admitted_revision > frame.revision:
        raise ValueError(_ERROR)
    observed, expiry = _window(observed_at, ttl_s)
    event = _event(event, limits, initial=True)
    stream, entries, changed = {}, {}, set()
    concurrent = {key: value for key, value in frame.entries.items()
                  if _rest_overlay(value) and value.revision is not None
                  and value.revision > admitted_revision}
    for identity, value in event['a'].items():
        state = _full(identity, value, limits)
        stream[identity] = _tree(state, limits, freeze=True)
        old = frame.entries.get(identity)
        incoming = EntityObservation(identity, stream[identity], True, WS_SOURCE,
            observed, expiry, None, None, generation, frame.revision + 1)
        selected = _select_stream(concurrent.get(identity), incoming, revision=frame.revision + 1)
        entries[identity] = selected
        if _selection_changes(old, selected, now=observed)[0]:
            changed.add(identity)
    for identity, value in concurrent.items():
        if identity not in stream:
            # Missing membership cannot order a concurrent REST point. Retain
            # explicit unknown history so it shadows the implicit negative.
            entries[identity] = observation_view(value, exists=None, reason='source_changed',
                                                 revision=frame.revision + 1)
    old_present = set(frame.stream_states) | {key for key, value in frame.entries.items() if value.exists is True}
    return _frame(frame, stream, entries, limits=limits, generation=generation, initialized=True,
        observed_at=observed, expires_at=expiry, changed=changed,
        removed=old_present - set(stream) - set(concurrent), absence_revision=None)


@_safe
def apply_delta(frame: StateFrame, event: dict, *, observed_at: float,
                ttl_s: float = STATE_TTL_S, limits: ProjectionLimits = DEFAULT_LIMITS) -> StateFrame:
    """Reduce a complete delta over its WS baseline, publishing nothing partial."""
    if type(frame) is not StateFrame or not frame.initialized:
        raise ValueError(_ERROR)
    observed, expiry = _window(observed_at, ttl_s)
    event = _event(event, limits, initial=False)
    stream, entries = dict(frame.stream_states), dict(frame.entries)
    changed, removed = set(), set()
    for identity, value in event.get('a', {}).items():
        state = _tree(_full(identity, value, limits), limits, freeze=True)
        stream[identity] = state
        incoming = EntityObservation(identity, state, True, WS_SOURCE,
            observed, expiry, None, None, frame.generation, frame.revision + 1)
        entries[identity] = _select_stream(frame.entries.get(identity), incoming,
                                          revision=frame.revision + 1)
        if _selection_changes(frame.entries.get(identity), entries[identity], now=observed)[0]:
            changed.add(identity)
    for identity, diff in event.get('c', {}).items():
        if identity not in stream:
            raise ValueError(_ERROR)
        state = _tree(_change(stream[identity], diff, limits), limits, freeze=True)
        stream[identity] = state
        incoming = EntityObservation(identity, state, True, WS_SOURCE,
            observed, expiry, None, None, frame.generation, frame.revision + 1)
        entries[identity] = _select_stream(frame.entries.get(identity), incoming,
                                          revision=frame.revision + 1)
        if _selection_changes(frame.entries.get(identity), entries[identity], now=observed)[0]:
            changed.add(identity)
    for identity in event.get('r', []):
        stream.pop(identity, None)
        incoming = EntityObservation(identity, None, False, WS_SOURCE,
            observed, expiry, None, None, frame.generation, frame.revision + 1)
        entries[identity] = _select_stream(frame.entries.get(identity), incoming,
                                          revision=frame.revision + 1)
        if _selection_changes(frame.entries.get(identity), entries[identity], now=observed)[1]:
            removed.add(identity)
    return _frame(frame, stream, entries, limits=limits, changed=changed, removed=removed)


@_safe
def apply_rest(frame: StateFrame, entity_id: str, state: dict | None, *, observed_at: float,
               ttl_s: float = STATE_TTL_S, expected_generation: int,
               expected_revision: int | None, source: str = REST_SOURCE,
               attempted_at: float | None = None, limits: ProjectionLimits = DEFAULT_LIMITS) -> StateFrame:
    """Publish actual REST evidence under CAS, or return the identical frame."""
    if type(frame) is not StateFrame:
        raise ValueError(_ERROR)
    _entity_id(entity_id)
    _counter(expected_generation)
    if expected_revision is not None:
        _counter(expected_revision)
    if frame.generation != expected_generation or frame.entry_revision(entity_id) != expected_revision:
        return frame
    if (type(source) is not str and not isinstance(source, StrEnum)) or source not in {
            REST_SOURCE, REST_SNAPSHOT_SOURCE, LEGACY_WS_SOURCE}:
        raise ValueError(_ERROR)
    source = str(source)
    if attempted_at is not None:
        _instant(attempted_at)
    observed, expiry = _window(observed_at, ttl_s)
    value = None if state is None else _tree(_rest_state(entity_id, state, limits), limits, freeze=True)
    entries = dict(frame.entries)
    previous = entries.get(entity_id)
    entries[entity_id] = EntityObservation(entity_id, value, state is not None, source,
        observed, expiry, attempted_at, None, frame.generation, frame.revision + 1)
    changed = (entity_id,) if state is not None and (previous is None or previous._value != value) else ()
    return _frame(frame, frame.stream_states, entries, limits=limits, changed=changed,
                  removed=(entity_id,) if state is None else ())


@_safe
def apply_rest_snapshot(frame: StateFrame, snapshotstates: tuple[dict, ...] | list[dict], *,
                        observed_at: float, expires_at: float, attempted_at: float | None,
                        expected_generation: int, expected_revisions: Mapping[str, int | None],
                        limits: ProjectionLimits = DEFAULT_LIMITS) -> StateFrame:
    """Atomically project one REST batch with original dates and per-entity CAS.

    Omitted entities and an empty batch do not prove absence. Every incoming
    ID needs an explicit admission token, including the captured absence fence.
    The complete batch is validated before one frame revision is published.
    """
    if type(frame) is not StateFrame or type(expected_revisions) not in (dict, MappingProxyType):
        raise ValueError(_ERROR)
    if type(limits) is not ProjectionLimits:
        raise ValueError(_ERROR)
    _counter(expected_generation)
    observed, expiry = _instant(observed_at), _instant(expires_at)
    if expiry <= observed:
        raise ValueError(_ERROR)
    attempted = None if attempted_at is None else _instant(attempted_at)
    if len(expected_revisions) > limits.max_entities:
        raise ValueError(_ERROR)
    expected = dict(expected_revisions)
    for identity, revision in expected.items():
        _entity_id(identity)
        if revision is not None:
            _counter(revision)
    states = _snapshot_states(snapshotstates, limits)
    if any(value['entity_id'] not in expected for value in states):
        raise ValueError(_ERROR)
    if frame.generation != expected_generation or not states:
        return frame
    entries, changed, admitted = dict(frame.entries), set(), False
    for state in states:
        identity = state['entity_id']
        if frame.entry_revision(identity) != expected[identity]:
            continue
        previous = frame.entries.get(identity)
        incoming = EntityObservation(identity, state, True, REST_SNAPSHOT_SOURCE,
            observed, expiry, attempted, None, frame.generation, frame.revision + 1)
        entries[identity] = incoming
        admitted = True
        if previous is None or previous._value != incoming._value:
            changed.add(identity)
    if not admitted:
        return frame
    return _frame(frame, frame.stream_states, entries, limits=limits, changed=changed)


@_safe
def apply_rest_failure(frame: StateFrame, entity_id: str, *, attempted_at: float,
                       reason: str, expected_generation: int, expected_revision: int | None,
                       limits: ProjectionLimits = DEFAULT_LIMITS) -> StateFrame:
    """CAS-publish a failed read without renewing or discarding prior evidence.

    At the entry cap, invalidate only implicit inventory completeness and its
    bounded absence token. Existing entity evidence keeps its own dates/token.
    """
    if type(frame) is not StateFrame:
        raise ValueError(_ERROR)
    _entity_id(entity_id)
    _counter(expected_generation)
    if expected_revision is not None:
        _counter(expected_revision)
    if frame.generation != expected_generation or frame.entry_revision(entity_id) != expected_revision:
        return frame
    attempted = _instant(attempted_at)
    if (type(reason) is not str and not isinstance(reason, StrEnum)) or reason not in _FAILURE_REASONS:
        raise ValueError(_ERROR)
    reason = str(reason)
    if entity_id not in frame.entries and len(frame.entries) >= limits.max_entities:
        return _frame(frame, frame.stream_states, frame.entries, limits=limits,
                      initialized=False, absence_revision=frame.revision + 1)
    previous = entity_observation(frame, entity_id, now=attempted)
    source = REST_SOURCE if previous.source is None and previous._value is None else previous.source
    entries = dict(frame.entries)
    entries[entity_id] = observation_view(previous, exists=None, source=source,
        attempted_at=attempted, reason=reason, revision=frame.revision + 1)
    return _frame(frame, frame.stream_states, entries, limits=limits)


@_safe
def entity_observation(frame: StateFrame, entity_id: str, *, now: float) -> EntityObservation:
    """Read dated evidence/history without storing arbitrary absent IDs."""
    if type(frame) is not StateFrame:
        raise ValueError(_ERROR)
    _entity_id(entity_id)
    now = _instant(now)
    if entity_id in frame.entries:
        value = frame.entries[entity_id]
        if value.generation != frame.generation:
            return observation_view(value, exists=None, reason='source_changed')
        return value.as_of(now)
    if frame.initialized:
        value = EntityObservation(entity_id, None, False, WS_SOURCE,
            frame.observed_at, frame.expires_at, None, None, frame.generation,
            frame.entry_revision(entity_id))
        return value.as_of(now)
    return EntityObservation(entity_id, generation=frame.generation,
                             revision=frame.entry_revision(entity_id))
