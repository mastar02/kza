"""Closed V4 plans and lifecycle observations for the T024 audio family."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from src.core.household_clock import ROOM_SECTION_KEYS
from src.core.runtime_admission import _config_digest
from src.memory.output_publisher import GrammarSpec, PublisherPhase, publisher_grammar, publisher_plan
from src.memory.output_registration import _require

if TYPE_CHECKING:
    from src.memory.output_publisher import GenericOutputPublisher
    from src.memory.output_registration_v2 import OutputRegistrationV2


@dataclass(frozen=True, slots=True)
class _Role:
    name: str
    nominal_type: str
    minimum: int = 0
    maximum: int = 1
    receipt: str | None = None
    enabled_setting: tuple[str, ...] = ()
    enabled_default: bool = False


@dataclass(frozen=True, slots=True)
class _Destination:
    name: str
    settings: tuple[str, ...]
    relative_path: str


@dataclass(frozen=True, slots=True)
class _Row:
    kind: str
    roles: tuple[_Role, ...]
    destinations: tuple[_Destination, ...] = ()
    references: tuple[tuple[str, str], ...] = ()


_ROWS = {
    "audio_input": _Row("resource_only", (
        _Role("audio_manager", "src.pipeline.audio_manager.AudioManager", 1),
        _Role("audio_loop", "src.pipeline.audio_loop.AudioLoop", 1, receipt="_closing"),
        _Role("multi_room_audio_loop", "src.pipeline.multi_room_audio_loop.MultiRoomAudioLoop", receipt="_closing"),
        _Role("room_stream", "src.pipeline.multi_room_audio_loop.RoomStream", maximum=256),
        _Role("input_group", "src.audio.input_group.InputGroup", receipt="_closing"),
        _Role("owned_input_stream", "src.audio.owned_input.OwnedInputStream", 1, receipt="_closing"),
    )),
    "audio_health": _Row("atomic_json", (
        _Role("multi_room_audio_loop", "src.pipeline.multi_room_audio_loop.MultiRoomAudioLoop", 1, receipt="_closing"),
    ), (
        _Destination("health_file", ("rooms", "stream_watchdog", "health_path"), "data/audio_health.json"),
    )),
    "audio_output": _Row("resource_only", (
        _Role("voice_services", "src.pipeline.voice_services.VoiceServices", 1),
        _Role("response_handler", "src.pipeline.response_handler.ResponseHandler", 1, receipt="_shutdown_task"),
        _Role("tts_model_owner", "src.application.TTSModelOwner", 1, receipt="_closing"),
        _Role("response_cache", "src.tts.response_cache.ResponseCache", receipt="_closing"),
        _Role("local_voice_delivery", "src.core.voice_delivery.LocalVoiceDelivery"),
    )),
    "presence_hardware": _Row("resource_only", (
        _Role("presence_detector", "src.presence.presence_detector.PresenceDetector"),
        _Role("room_context_manager", "src.rooms.room_context.RoomContextManager"),
        _Role("zone_manager", "src.audio.zone_manager.ZoneManager"),
        _Role("ma1260", "src.audio.ma1260_controller.MA1260Controller"),
        _Role("xvf_controller", "src.audio.xvf_controller.XvfController", maximum=256),
    )),
    "capture": _Row("file_tree", (
        _Role("capture_eraser", "src.ambient.erasure.CaptureEraser", 1, receipt="_closing"),
        _Role("capture_control", "src.memory.capture_control.CaptureControl", 1, receipt="_closing"),
        _Role("ambient_store", "src.ambient.store.AmbientStore", enabled_setting=("ambient", "enabled")),
        _Role("audio_archiver", "src.ambient.audio_archive.AudioArchiver", receipt="_closing",
              enabled_setting=("ambient", "enabled")),
    ), (
        _Destination("ambient_db", ("ambient", "db_path"), "data/ambient.db"),
        _Destination("keep_audio_dir", ("ambient", "keep_audio", "dir"), "data/ambient_audio"),
    ), (("capture_runtime_control", "privacy_authority"),)),
    "collectors": _Row("file_tree", (
        _Role("collector_eraser", "src.training.collector_erasure.CollectorEraser", 1, receipt="_closing"),
        _Role("wake_clip_store", "src.wakeword.clip_store.WakeClipStore", 1, receipt="_closing"),
        _Role("wake_clip_writer", "src.wakeword.wake_clip_writer.WakeClipWriter", receipt="_closing",
              enabled_setting=("rooms", "wake_word", "capture_clips", "enabled")),
        _Role("stt_correction_collector", "src.training.stt_correction_collector.STTCorrectionCollector",
              receipt="_closing", enabled_setting=("training", "stt_corrections", "enabled")),
        _Role("conversation_collector", "src.training.conversation_collector.ConversationCollector", 1, receipt="_closing"),
    ), (
        _Destination("wake_clips", ("rooms", "wake_word", "capture_clips", "dir"), "data/wakeword_training/captured"),
        _Destination("stt_corrections", ("training", "stt_corrections", "data_dir"), "data/stt_corrections"),
        _Destination("conversations", ("training", "conversations_dir"), "data/conversations"),
        _Destination("exports", ("training", "exports_dir"), "data/conversation_exports"),
    ), (("collector_index", "privacy_authority"),)),
    "voices": _Row("sqlite", (
        _Role("voice_registry", "src.users.voice_registry.VoiceRegistry", 1, receipt="_closing"),
        _Role("voice_ledger", "src.users.voice_ledger.VoiceLedger", 1),
        _Role("voice_enrollment", "src.users.voice_enrollment.VoiceEnrollment",
              enabled_setting=("speaker_id", "enabled"), enabled_default=True),
    ), references=(("review_clips", "privacy_authority"), ("enrollment", "resource_only"))),
}


def _root_path(state_root: str | PurePosixPath) -> PurePosixPath:
    root = PurePosixPath(state_root)
    _require(root.is_absolute() and root.anchor == "/" and ".." not in root.parts, "audio_publisher_root")
    return root


def _nominal_plan(row: _Row, root: PurePosixPath, enabled: dict[str, bool]) -> dict:
    metadata = {}
    if row.destinations:
        metadata["destinations"] = {item.name: str(root / item.relative_path) for item in row.destinations}
    if row.references:
        metadata["references"] = dict(row.references)
    if enabled:
        metadata["enabled_owners"] = enabled
    return publisher_plan(row.kind, roles=tuple(role.name for role in row.roles), **metadata)


def _plan_shape(row: _Row, plan: dict, header: dict) -> None:
    enabled = plan.get("enabled_owners", {})
    _require(type(enabled) is dict and enabled.keys() == {role.name for role in row.roles if role.enabled_setting}
             and all(type(value) is bool for value in enabled.values()), "audio_publisher_plan")
    _require(plan == _nominal_plan(row, _root_path(header["state_root"]), enabled), "audio_publisher_plan")


def _validate_counts(producer: str, row: _Row, counts: dict, enabled: dict[str, bool]) -> None:
    _require(type(counts) is dict and counts.keys() == {role.name for role in row.roles},
             "audio_publisher_owner_counts")
    _require(all(type(counts[role.name]) is int and role.minimum <= counts[role.name] <= role.maximum
                 for role in row.roles) and any(counts.values()), "audio_publisher_owner_counts")
    if producer == "audio_input":
        multi = counts["multi_room_audio_loop"]
        _require(multi == counts["input_group"] == bool(counts["room_stream"]), "audio_publisher_owner_counts")
    _require(all(counts[role] == int(selected) for role, selected in enabled.items()),
             "audio_publisher_owner_counts")


def _aggregate_shape(producer: str, row: _Row, aggregate: dict, plan: dict, _header: dict) -> None:
    fields = {"owner_counts", "owner_state", "owner_replaced"}
    if producer == "presence_hardware":
        fields.add("amplifier_mode")
    _require(aggregate.keys() == fields and aggregate["owner_state"] == "drained"
             and aggregate["owner_replaced"] is False, "audio_publisher_aggregate")
    counts = aggregate["owner_counts"]
    _validate_counts(producer, row, counts, plan.get("enabled_owners", {}))
    if producer == "presence_hardware":
        modes = ("serial", "ip", "simulation") if counts["ma1260"] else ("absent",)
        _require(aggregate["amplifier_mode"] in modes, "audio_publisher_aggregate")


AUDIO_GRAMMAR_SPECS: tuple[GrammarSpec, ...] = tuple(
    publisher_grammar(
        producer, roles=tuple(role.name for role in row.roles),
        plan_shape=partial(_plan_shape, row), aggregate_shape=partial(_aggregate_shape, producer, row),
    )
    for producer, row in _ROWS.items()
)


def _setting(config: Mapping, path: tuple[str, ...], default: object) -> object:
    section = config
    for key in path[:-1]:
        section = section.get(key)
        if section is None:
            section = {}
        _require(isinstance(section, Mapping), "audio_publisher_config")
    return section.get(path[-1], default)


def _configured_path(config: Mapping, root: PurePosixPath, destination: _Destination) -> PurePosixPath:
    value = _setting(config, destination.settings, destination.relative_path)
    _require(isinstance(value, (str, PurePosixPath)), "audio_publisher_destination")
    path = PurePosixPath(value)
    return root / path


def selects_multi_room_audio(config: Mapping) -> bool:
    """Mirror the composition root's effective microphone selection."""
    rooms = config.get("rooms", {})
    return bool(rooms.get("enabled", False)) and any(
        room.get("mic_device_index") is not None or bool(room.get("mic_usb_port"))
        for name, room in rooms.items() if name not in ROOM_SECTION_KEYS and isinstance(room, Mapping)
    )


def resolve_audio_family_plans(config: Mapping, state_root: str | PurePosixPath) -> dict[str, dict]:
    """Build all seven plans from config and the sealed state root without I/O.

    The /1 grammar pins the reviewed relative destinations. A configured
    destination outside that set needs a reviewed grammar before it can be
    journaled. Relative settings are interpreted against the state root.
    """
    root = _root_path(state_root)
    multi_room = selects_multi_room_audio(config)
    plans = {}
    for producer, row in _ROWS.items():
        for destination in row.destinations:
            _require(_configured_path(config, root, destination) == root / destination.relative_path,
                     "audio_publisher_destination")
        enabled = {}
        for role in row.roles:
            if role.enabled_setting:
                value = _setting(config, role.enabled_setting, role.enabled_default)
                _require(type(value) is bool, "audio_publisher_config")
                if role.name in {"ambient_store", "audio_archiver", "wake_clip_writer"}:
                    value = value and multi_room
                enabled[role.name] = value
        plans[producer] = _nominal_plan(row, root, enabled)
    return plans


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AudioOwner:
    """A live graph slot and its original owner's retained close task.

    For owners with a retained task the binding reads it from the instance.
    For borrower slots supply the original task that covers them (for example,
    the loop task for a room).
    """

    instance: object
    close_receipt: asyncio.Task | None = None


OwnerGraph = Callable[[], Mapping[str, tuple[AudioOwner, ...]]]



def _validate_graph(producer: str, graph: dict[str, dict[int, AudioOwner]]) -> None:
    if producer == "audio_input":
        loop = next(iter(graph["audio_loop"].values())).instance
        _require(id(loop.audio_manager) in graph["audio_manager"]
                 and id(loop._input) in graph["owned_input_stream"], "audio_publisher_owner_graph")
        if graph["multi_room_audio_loop"]:
            multi = next(iter(graph["multi_room_audio_loop"].values())).instance
            _require(id(multi._inputs) in graph["input_group"]
                     and graph["room_stream"].keys() == {id(room) for room in multi.room_streams.values()},
                     "audio_publisher_owner_graph")
    if producer == "audio_output":
        services = next(iter(graph["voice_services"].values())).instance
        _require(id(services.responses) in graph["response_handler"], "audio_publisher_owner_graph")
    if producer == "voices":
        registry = next(iter(graph["voice_registry"].values())).instance
        _require(id(registry.ledger) in graph["voice_ledger"], "audio_publisher_owner_graph")


@dataclass(slots=True, repr=False, eq=False)
class AudioPublisherBinding:
    """Pin reviewed graph instances and observe their original lifecycle."""

    _publisher: GenericOutputPublisher
    _enabled: dict[str, bool]
    _phase: PublisherPhase = field(default=PublisherPhase.INTENT, init=False)
    _read_graph: OwnerGraph | None = field(default=None, init=False)
    _owners: dict[str, tuple[object, ...]] = field(default_factory=dict, init=False)
    _owner_replaced: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def owner_replaced(self) -> bool:
        """Report an invalidation permanently for this binding."""
        return self._owner_replaced

    def _snapshot(self, read_graph: OwnerGraph) -> dict[str, dict[int, AudioOwner]]:
        producer = self._publisher.producer
        row = _ROWS[producer]
        graph = dict(read_graph())
        _require(graph.keys() <= {role.name for role in row.roles}, "audio_publisher_owner_roles")
        snapshot = {}
        for role in row.roles:
            entries = graph.get(role.name, ())
            _require(type(entries) is tuple and all(type(entry) is AudioOwner for entry in entries),
                     "audio_publisher_owner_roles")
            if entries:
                module, name = role.nominal_type.rsplit(".", 1)
                nominal = getattr(import_module(module), name)
                _require(all(type(entry.instance) is nominal and not getattr(entry.instance, "_capture_unreviewed", False)
                             for entry in entries), "audio_publisher_owner_type")
            owners = {id(entry.instance): entry for entry in entries}
            _require(len(owners) == len(entries), "audio_publisher_owner_roles")
            snapshot[role.name] = owners
        _validate_counts(producer, row, {role: len(owners) for role, owners in snapshot.items()}, self._enabled)
        _validate_graph(producer, snapshot)
        return snapshot

    async def attached(self, read_graph: OwnerGraph) -> None:
        """Attach a live accessor after the original shutdown owners are installed."""
        async with self._lock:
            _require(self._phase is PublisherPhase.INTENT, "audio_publisher_phase")
            snapshot = self._snapshot(read_graph)
            self._owners = {role: tuple(entry.instance for entry in owners.values())
                            for role, owners in snapshot.items()}
            self._read_graph = read_graph
            await self._publisher.attached()
            self.require_current()
            self._phase = PublisherPhase.ATTACHED

    def _current(self) -> dict[str, dict[int, AudioOwner]]:
        read_graph = self._read_graph
        if read_graph is None:
            raise RuntimeError("audio_publisher_phase")
        if not self._owner_replaced:
            try:
                observed = self._snapshot(read_graph)
                if all(observed[role].keys() == {id(owner) for owner in owners}
                       for role, owners in self._owners.items()):
                    return observed
            except Exception:
                # An unreadable graph is also an unconfirmed instance match.
                pass
            self._owner_replaced = True
        raise RuntimeError("audio_publisher_owner_replaced")

    def require_current(self) -> None:
        """Check the exact instances before subsequent native admissions."""
        self._current()

    def replace_retired_input(self, previous: object, successor: object) -> None:
        """Track the exact successor created by AudioLoop after a clean pause."""
        from src.audio.owned_input import OwnedInputStream

        _require(self._publisher.producer == "audio_input" and self._phase is PublisherPhase.OPENING
                 and self._owners.get("owned_input_stream") == (previous,)
                 and type(previous) is OwnedInputStream and previous.retired
                 and isinstance(previous._closing, asyncio.Task) and previous._closing.done()
                 and not previous._closing.cancelled() and previous._closing.exception() is None
                 and type(successor) is OwnedInputStream and successor._closing is None,
                 "audio_publisher_input_successor")
        self.require_current()
        self._owners["owned_input_stream"] = (successor,)

    def attach_response_cache(self, cache: object) -> None:
        """Attach the optional cache installed by ApplicationGraph before warmup."""
        from src.tts.response_cache import ResponseCache

        _require(self._publisher.producer == "audio_output" and self._phase is PublisherPhase.OPENING
                 and self._owners.get("response_cache") == () and type(cache) is ResponseCache,
                 "audio_publisher_response_cache")
        if cache not in tuple(slot.instance for slot in self._read_graph().get("response_cache", ())):
            raise RuntimeError("output_registration_audio_publisher_response_cache")
        self._owners["response_cache"] = (cache,)
        self.require_current()

    async def opening(self) -> None:
        """Publish opening and recheck identities before the caller opens."""
        async with self._lock:
            _require(self._phase is PublisherPhase.ATTACHED, "audio_publisher_phase")
            self.require_current()
            await self._publisher.opening()
            self.require_current()
            self._phase = PublisherPhase.OPENING

    async def drained(self) -> None:
        """Observe successful original receipts after close has been initiated."""
        async with self._lock:
            _require(self._phase is PublisherPhase.OPENING, "audio_publisher_phase")
            receipts = self._receipts()
            results = await asyncio.gather(*(asyncio.shield(task) for task in set(receipts.values())),
                                           return_exceptions=True)
            _require(not any(isinstance(result, BaseException) for result in results),
                     "audio_publisher_owner_drain_unconfirmed")
            _require(self._receipts() == receipts, "audio_publisher_owner_drain_unconfirmed")
            aggregate = {
                "owner_counts": {role: len(owners) for role, owners in self._owners.items()},
                "owner_state": "drained",
                "owner_replaced": self.owner_replaced,
            }
            if self._publisher.producer == "presence_hardware":
                amplifier = self._owners["ma1260"]
                mode = amplifier[0].connection_type if amplifier else "absent"
                _require(type(mode) is str and mode in ("serial", "ip", "simulation", "absent"),
                         "audio_publisher_aggregate")
                aggregate["amplifier_mode"] = mode
            await self._publisher.drained(aggregate)
            self._phase = PublisherPhase.DRAINED
            self._read_graph = None
            self._owners.clear()

    def _receipts(self) -> dict[tuple[str, int], asyncio.Task]:
        receipts = {}
        current = self._current()
        for role in _ROWS[self._publisher.producer].roles:
            for key, entry in current[role.name].items():
                task = entry.close_receipt
                if role.receipt is not None:
                    task = getattr(entry.instance, role.receipt)
                    _require(entry.close_receipt is None or entry.close_receipt is task,
                             "audio_publisher_owner_drain_unconfirmed")
                _require(isinstance(task, asyncio.Task) and task is not asyncio.current_task(),
                         "audio_publisher_owner_drain_unconfirmed")
                if role.name == "owned_input_stream" and task.done():
                    _require(entry.instance.retired, "audio_publisher_owner_drain_unconfirmed")
                receipts[role.name, key] = task
        return receipts


async def bind_audio_family_publishers(
    coordinator: OutputRegistrationV2 | None, config: dict, *, selection: Mapping[str, bool] | None = None,
) -> dict[str, AudioPublisherBinding]:
    """Publish all intents or disabled rows before constructing audio owners.

    Use the same selection in the coordinator. The graph retains its shutdown
    callbacks, supplies live accessors at attachment, awaits opening before
    requesting work and calls drained after initiating the original closes.
    Keep graph slots stable from settlement of their close tasks through
    publication of the terminal observation.
    """
    if coordinator is None:
        return {}
    selection = {} if selection is None else dict(selection)
    _require(selection.keys() <= _ROWS.keys() and all(type(value) is bool for value in selection.values()),
             "audio_publisher_selection")
    _require(_config_digest(config) == coordinator.admission._config_digest, "audio_publisher_config")
    _require(all(coordinator._selected.get(producer) is selection.get(producer, True)
                 for producer in _ROWS), "audio_publisher_selection")
    plans = resolve_audio_family_plans(config, coordinator.admission.state_root)
    bindings = {}
    await coordinator._run(coordinator.admission.revalidate)
    for producer, plan in plans.items():
        if selection.get(producer, True):
            publisher = await coordinator.publisher_intent(producer, plan)
            bindings[producer] = AudioPublisherBinding(publisher, plan.get("enabled_owners", {}))
        else:
            await coordinator.publisher_disabled(producer, plan)
    return bindings
