"""Nominal V4 boundaries and owner observations for the T024 memory family."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from ipaddress import ip_address
import logging
import os
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

from src.core.egress import EgressPolicy
from src.core.household_clock import ROOM_SECTION_KEYS
from src.core.paths import models_path, speaker_savedir, tts_settings
from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY, is_unresolved_placeholder
from src.llm.cloud_consent import DEFAULT_COMPACTION_LOCAL_URL, cloud_reasoner_allowed
from src.llm.router_factory import DEFAULT_COOLDOWN_PATH
from src.memory.output_publisher import GenericOutputPublisher, GrammarSpec, publisher_grammar, publisher_plan
from src.memory.output_registration import _require
from src.obs.diagnostic_erasure import DEFAULT_METRICS_PATH, diagnostic_projection_path
from src.vectordb.command_index import LEGACY_COMMANDS_COLLECTION, pointer_name

if TYPE_CHECKING:
    from src.memory.output_registration_v2 import OutputRegistrationV2


class _Mode(StrEnum):
    ACTIVE = "active"
    LAZY = "lazy"
    HISTORY = "history_only"
    DISABLED = "disabled"


@dataclass(frozen=True)
class _Role:
    module: str
    name: str
    mode: _Mode
    optional: bool = False
    drain_task: str | None = None

    def accepts(self, instance: object) -> bool:
        # The caller has already constructed the owner. Do not import its SDK.
        module = sys.modules.get(self.module)
        return module is not None and vars(module).get(self.name) is type(instance)

    def drained(self, instance: object) -> bool:
        if self.drain_task is None:
            return True  # A borrowed role or bootstrap stream has no independent close.
        task = getattr(instance, self.drain_task)
        return isinstance(task, asyncio.Task) and task.done() and not task.cancelled() and task.exception() is None


def _role(module: str, name: str, mode: _Mode = _Mode.ACTIVE, *,
          optional: bool = False, drain_task: str | None = "_closing") -> _Role:
    return _Role(f"src.{module}", name, mode, optional, drain_task)


_KINDS = {
    "house_vector_index": "file_tree",
    "personal_memory": "file_tree",
    "contexts": "file_tree",
    "model_resources": "resource_only",
    "reasoner_transports": "transport_session",
    "diagnostics": "sqlite",
    "logging": "external_effect",
}

_ROLES = {
    "house_vector_index": {"chroma_sync": _role("vectordb.chroma_sync", "ChromaSync", _Mode.LAZY)},
    "personal_memory": {
        "memory_services": _role("memory.services", "MemoryServices", optional=True),
        "long_term_memory": _role("memory.long_term", "LongTermMemory", optional=True, drain_task="_close_task"),
        "policy_index_eraser": _role("memory.index_erasure", "PolicyIndexEraser", _Mode.HISTORY),
    },
    "contexts": {
        "context_eraser": _role("orchestrator.context_erasure", "ContextEraser", _Mode.HISTORY),
        "context_manager": _role("orchestrator.context_manager", "ContextManager", optional=True),
        "context_persister": _role("orchestrator.context_persister", "ContextPersister", _Mode.LAZY,
                                   optional=True, drain_task=None),
    },
    "model_resources": {
        "stt_executors": _role("application", "STTExecutorOwner"),
        "tts_models": _role("application", "TTSModelOwner"),
        "speaker_models": _role("application", "SpeakerModelOwner"),
        "wake_models": _role("application", "WakeModelOwner"),
    },
    "reasoner_transports": {
        "http_reasoners": _role("application", "HttpReasonerOwner", optional=True),
        "hermes_reasoners": _role("application", "HermesReasonerOwner", optional=True),
        "local_reasoner": _role("llm.reasoner", "LLMReasoner", optional=True, drain_task=None),
        "fast_router": _role("llm.reasoner", "FastRouter", _Mode.LAZY, optional=True, drain_task="_adapter_closing"),
        "llm_router": _role("llm.router", "LLMRouter", optional=True, drain_task=None),
        "compactor": _role("orchestrator.compactor", "Compactor", optional=True, drain_task=None),
        "llm_command_router": _role("nlu.llm_router", "LLMCommandRouter", optional=True, drain_task=None),
    },
    "diagnostics": {
        "diagnostic_services": _role("obs.diagnostic_erasure", "DiagnosticServices", _Mode.HISTORY),
        "metrics_emitter": _role("obs.metrics_emitter", "MetricsEmitter", optional=True),
    },
    "logging": {"bootstrap_logging": _Role("logging", "RootLogger", _Mode.ACTIVE)},
}

_HEX = re.compile("[0-9a-f]{64}")
_HOST = re.compile(r"[a-zA-Z0-9.-]+")
_Check = Callable[[object], bool]


def _enum(*choices: str) -> _Check:
    return choices.__contains__


def _digest(value: object) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


def _maybe_digest(value: object) -> bool:
    return value is None or _digest(value)


def _port(value: object) -> bool:
    return type(value) is int and 1 <= value <= 65535


def _fields(value: object, fields: Mapping[str, _Check]) -> bool:
    return (type(value) is dict and value.keys() == fields.keys()
            and all(check(value[key]) for key, check in fields.items()))


def _collector_shape(value: dict) -> bool:
    return _fields(value, {
        "status": _enum("disabled", "configured_inactive", "external_collector_unconfirmed"),
        "port": lambda port: port is None or _port(port),
        "host_sha256": _maybe_digest,
    }) and (value["status"] != "disabled" or value["port"] is value["host_sha256"] is None)


def _model_shape(value: dict) -> bool:
    return _fields(value, {
        "mode": _enum("lazy", "disabled", "shared"),
        "input_kind": _enum("readonly_input", "sdk_model", "shared", "none"),
        "input_sha256": _maybe_digest,
        "sdk_cache": _enum("none", "configured", "unconfirmed"),
        "cache_sha256": _maybe_digest,
    }) and (
        (value["input_sha256"] is not None) == (value["input_kind"] in {"readonly_input", "sdk_model"})
        and (value["cache_sha256"] is not None) == (value["sdk_cache"] == "configured")
        and (value["mode"] == "disabled") == (value["input_kind"] == "none")
        and (value["mode"] == "shared") == (value["input_kind"] == "shared")
        and (value["sdk_cache"] != "none") == (value["input_kind"] == "sdk_model")
    )


def _transport_shape(value: dict) -> bool:
    return _fields(value, {
        "mode": _enum("configured", "blocked", "disabled"),
        "protocol": _enum("http", "https", "cli", "in_process", "unresolved", "none"),
        "processing": _enum("local", "cloud", "unknown", "none"),
        "port": lambda port: port is None or _port(port),
        "origin_sha256": _maybe_digest,
    }) and (
        (value["mode"] == "disabled") == (value["processing"] == "none") == (value["protocol"] == "none")
        and (value["protocol"] in {"http", "https"}) == (value["port"] is not None)
        and (value["protocol"] in {"http", "https", "cli", "in_process"}) == (value["origin_sha256"] is not None)
        and (value["mode"] != "configured" or value["processing"] in {"local", "cloud"})
        and (value["mode"] != "configured" or value["protocol"] != "unresolved")
        and (value["protocol"] != "cli" or value["processing"] == "cloud")
        and (value["protocol"] != "in_process" or value["processing"] == "local")
    )


_MODEL_SLOTS = (
    "stt_primary", "stt_shadow", "stt_ambient", "tts_primary", "tts_secondary",
    "speaker_primary", "speaker_ambient", "wake_room", "wake_parameters", "wake_fallback",
)
_TRANSPORT_SLOTS = ("primary", "fast_router", "compaction")
_CONTEXT_MODES = {
    "history_only": ("disabled", "disabled"), "memory_only": ("active", "disabled"),
    "files_if_compactor_available": ("active", "lazy"),
}
_HISTORY = _enum("erasure_ledger")
_BOUNDARIES = {
    "house_vector_index": {
        "chroma_path_sha256": _digest, "base_collection": _enum(LEGACY_COMMANDS_COLLECTION),
        "pointer_collection": _enum(pointer_name()), "active_collection": _enum("resolved_by_pointer"),
        "sdk_store": _enum("opaque_sdk_store"), "model_cache": _enum("sdk_unconfirmed"),
    },
    "personal_memory": {
        "chroma_path_sha256": _digest, "preferences_path_sha256": _digest,
        "privacy_db_path_sha256": _digest, "policy_collection_sha256": _digest,
        "authority": _enum("privacy_authority"), "historical_targets": _HISTORY,
    },
    "contexts": {
        "policy_v2_sha256": _digest, "policy_v3_sha256": _digest,
        "mode": _enum(*_CONTEXT_MODES), "historical_targets": _HISTORY,
    },
    "model_resources": dict.fromkeys(_MODEL_SLOTS, _model_shape),
    "reasoner_transports": {
        **dict.fromkeys(_TRANSPORT_SLOTS, _transport_shape),
        "local_adapter_sha256": _maybe_digest,
        "cooldown_path_sha256": _digest, "cooldown_mode": _enum("configured", "history_only"),
        "cloud_consent": lambda value: type(value) is bool,
        "cloud_withdrawn": lambda value: type(value) is bool,
        "egress_authority": _enum("original_owner"),
    },
    "diagnostics": {
        "legacy_jsonl_sha256": _digest, "jsonl_sha256": _digest, "rotation_sha256": _digest,
        "sqlite_sha256": _digest, "rotation_bytes": lambda n: type(n) is int and 0 < n < 2**64,
        "historical_targets": _HISTORY, "logstash": _collector_shape,
    },
    "logging": {
        "stderr": _enum("inherited"), "stdout": _enum("inherited"),
        "bootstrap": _enum("standard_stream_handlers"),
        "configured_file_sha256": _maybe_digest, "console_redirection_sha256": _maybe_digest,
        "diagnostic_forwarder": _collector_shape, "sdk_handlers": _enum("unconfirmed"),
        "stream_destinations": _enum("unconfirmed"), "log_authority": _enum("original_owner"),
    },
}


def _boundary_modes(producer: str, plan: dict) -> bool:
    modes, targets = plan["role_modes"], plan["boundaries"]
    if producer == "personal_memory":
        return modes["memory_services"] == modes["long_term_memory"]
    if producer == "contexts":
        return (modes["context_manager"], modes["context_persister"]) == _CONTEXT_MODES[targets["mode"]]
    if producer == "reasoner_transports":
        primary, fast, compaction = (targets[key] for key in _TRANSPORT_SLOTS)
        configured = [item["protocol"] for item in (primary, compaction) if item["mode"] == "configured"]
        return (
            (modes["http_reasoners"] == "active") == any(item in {"http", "https"} for item in configured)
            and (modes["hermes_reasoners"] == "active") == ("cli" in configured)
            and (modes["local_reasoner"] == "active") == ("in_process" in configured)
            and (modes["fast_router"] == "lazy") == (fast["mode"] != "disabled")
            and (modes["compactor"] == "active") == (compaction["mode"] == "configured")
            and (modes["llm_router"] == "active") == (targets["cooldown_mode"] == "configured")
            and (modes["fast_router"] == "lazy" or modes["llm_router"] == modes["llm_command_router"] == "disabled")
            and all(targets[key]["mode"] != "configured" or targets[key]["processing"] != "cloud"
                    or (targets["cloud_consent"] and not targets["cloud_withdrawn"])
                    for key in _TRANSPORT_SLOTS)
        )
    if producer == "diagnostics":
        return (targets["logstash"]["status"] == "disabled"
                or (targets["logstash"]["status"] == "external_collector_unconfirmed")
                == (modes["metrics_emitter"] == "active"))
    return True


def _selected(plan: dict) -> bool:
    return any(mode != "disabled" for mode in plan["role_modes"].values())


def _grammar(producer: str) -> GrammarSpec:
    roles = _ROLES[producer]
    checks = {name: _enum(role.mode.value, "disabled") if role.optional else _enum(role.mode.value)
              for name, role in roles.items()}

    def plan_shape(plan: dict, header: dict) -> None:
        _require(set(plan) == {"kind", "roles", "role_modes", "boundaries"}
                 and plan["kind"] == _KINDS[producer]
                 and _fields(plan["role_modes"], checks)
                 and _fields(plan["boundaries"], _BOUNDARIES[producer])
                 and _boundary_modes(producer, plan), "publisher_plan")
        _require(header["selected"][producer] is _selected(plan), "publisher_selection")

    def aggregate_shape(aggregate: dict, plan: dict, _header: dict) -> None:
        _require(set(aggregate) == {"owners"} and type(aggregate["owners"]) is dict
                 and aggregate["owners"].keys() == roles.keys(), "publisher_aggregate")
        identities = set()
        for name, evidence in aggregate["owners"].items():
            mode = plan["role_modes"][name]
            observations = {"disabled": ("disabled",), "lazy": ("drained", "lazy_unopened")}
            _require(_fields(evidence, {
                "observation": _enum(*observations.get(mode, ("drained",))),
                "instance_sha256": (lambda value: value is None) if mode == "disabled" else _digest,
            }), "publisher_aggregate")
            identity = evidence["instance_sha256"]
            if identity is not None:
                _require(identity not in identities, "publisher_aggregate")
                identities.add(identity)

    return publisher_grammar(producer, roles=tuple(roles), plan_shape=plan_shape, aggregate_shape=aggregate_shape)


MEMORY_GRAMMAR_SPECS: tuple[GrammarSpec, ...] = tuple(map(_grammar, _KINDS))


def _section(config: Mapping, *keys: str) -> dict:
    for key in keys:
        value = config.get(key)
        config = value if isinstance(value, dict) else {}
    return dict(config)


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _at(root: Path, value: str | Path) -> Path:
    # Relative targets are rooted like the managed child. Physical verification
    # of symlinks and actual files belongs to the original owners.
    return Path(os.path.normpath(root / value))


def _path_sha(root: Path, value: str | Path) -> str:
    return _sha(str(_at(root, value)))


def _collector(settings: dict) -> dict:
    host = settings.get("logstash_host")
    if not host:
        return {"status": "disabled", "port": None, "host_sha256": None}
    port = settings.get("logstash_port", 5515)
    normalized = None
    if type(host) is str and "%" not in host:
        try:
            normalized = str(ip_address(host))
        except ValueError:
            if _HOST.fullmatch(host):
                normalized = host.lower()
    return {
        "status": "external_collector_unconfirmed" if settings.get("enabled", True) else "configured_inactive",
        "port": port if _port(port) else None,
        "host_sha256": _sha(normalized) if normalized else None,
    }


def _model(root: Path, value: str | None = None, *, static: bool = False,
           cache: str | Path | None = None, shared: bool = False) -> dict:
    if not value:
        return {"mode": "shared" if shared else "disabled", "input_kind": "shared" if shared else "none",
                "input_sha256": None, "sdk_cache": "none", "cache_sha256": None}
    static = static or value.startswith(("/", "./", "../", "models/"))
    return {
        "mode": "lazy", "input_kind": "readonly_input" if static else "sdk_model",
        "input_sha256": _path_sha(root, value) if static else _sha(value),
        "sdk_cache": "none" if static else ("configured" if cache else "unconfirmed"),
        "cache_sha256": _path_sha(root, cache) if cache and not static else None,
    }


def _command_model(stt: dict, engine: str, model: str) -> str:
    if engine.lower() == "parakeet" or "parakeet" in model.lower():
        return stt.get("parakeet_model", "nemo-parakeet-tdt-0.6b-v3")
    return model


def _model_boundaries(config: Mapping, root: Path) -> dict:
    stt, ambient, rooms = (_section(config, key) for key in ("stt", "ambient", "rooms"))
    model = stt.get("model", "distil-whisper/distil-small.en")
    shadow = stt.get("shadow_engine")
    room_capture = rooms.get("enabled", False) and any(
        key not in ROOM_SECTION_KEYS and isinstance(room, dict)
        and (room.get("mic_device_index") is not None or room.get("mic_usb_port"))
        for key, room in rooms.items()
    )
    ambient_on = room_capture and ambient.get("enabled", False)
    ambient_stt = _section(ambient, "stt")
    ambient_model = (
        ambient_stt.get("parakeet_model", "nemo-parakeet-tdt-0.6b-v3")
        if ambient_stt.get("engine") == "parakeet"
        else ambient_stt.get("model", stt.get("model")) or models_path(config, "whisper-v3-turbo", environ={})
    )

    speaker = _section(config, "speaker_id")
    ambient_speaker = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_model = speaker.get("model", ambient_speaker)
    speaker_cache = (speaker_savedir(config, speaker_model, environ={})
                     if speaker.get("enabled", True) and ("ecapa" in speaker_model.lower()
                                                         or "speechbrain" in speaker_model.lower()) else None)

    tts = tts_settings(config, environ={})
    engine = tts.get("engine", "piper")
    _require(engine in {"piper", "hybrid", "xtts", "dual"}, "publisher_plan")
    xtts = _section(tts, "xtts").get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
    primary = xtts if engine == "xtts" else _section(tts, "piper").get("model")
    secondary = xtts if engine == "hybrid" else None
    if engine == "dual":
        qwen = _section(tts, "qwen3")
        primary = _section(tts, "kokoro").get("model", "hexgrad/Kokoro-82M")
        secondary = qwen.get("model", "Qwen/Qwen3-TTS-0.6B") if qwen else None

    fallback_wake = _section(config, "wake_word")
    wake = rooms.get("wake_word", fallback_wake)
    wake_engine = wake.get("engine", "openwakeword")
    room_model, params = wake.get("model", "hey_jarvis"), None
    if wake_engine == "porcupine":
        porcupine = _section(wake, "porcupine")
        room_model = porcupine.get("keyword_path", models_path(config, "wakeword", "nexa_es.ppn", environ={}))
        params = porcupine.get("model_path", models_path(config, "wakeword", "porcupine_params_es.pv", environ={}))
    return {
        "stt_primary": _model(root, _command_model(stt, stt.get("engine", ""), model)),
        "stt_shadow": _model(root, _command_model(stt, shadow, model) if shadow else None),
        "stt_ambient": _model(root, ambient_model if ambient_on else None),
        "tts_primary": _model(root, primary, static=engine in {"piper", "hybrid"}),
        "tts_secondary": _model(root, secondary),
        "speaker_primary": _model(root, speaker_model if speaker.get("enabled", True) else None, cache=speaker_cache),
        "speaker_ambient": _model(root, ambient_speaker,
                                  cache=speaker_savedir(config, ambient_speaker, environ={})) if ambient_on else _model(root),
        "wake_room": _model(root, room_model if room_capture and wake_engine != "whisper" else None,
                            static=wake_engine == "porcupine" or room_model.endswith(".onnx"),
                            shared=bool(room_capture and wake_engine == "whisper")),
        "wake_parameters": _model(root, params if room_capture else None, static=True),
        "wake_fallback": _model(root, fallback_wake.get("model", "hey_jarvis")),
    }


def _transport(mode: str = "disabled", protocol: str = "none", processing: str = "none",
               *, port: int | None = None, identity: str | None = None) -> dict:
    return {"mode": mode, "protocol": protocol, "processing": processing,
            "port": port, "origin_sha256": _sha(identity) if identity else None}


def _http(url: str, destination: str, consent: bool) -> dict:
    try:
        policy = EgressPolicy.for_http(url, destination, consent)
        parsed = urlsplit(url)
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not _port(port):
            raise ValueError("invalid port")
    except (TypeError, ValueError):
        # No fingerprint is made of a rejected URL; it can contain credentials.
        declared = destination if destination in ("local", "cloud") else "unknown"
        return _transport("blocked", "unresolved", declared)
    return _transport("configured" if policy.allowed else "blocked", parsed.scheme, policy.destination.value,
                      port=port, identity=parsed.hostname.lower())


def _compaction(settings: dict, url: str, destination: str, gate: bool, consent: bool) -> dict:
    # Mirror resolve_compaction_endpoint without its startup logging or its
    # credential-bearing DTO. The original owner still resolves and gates sends.
    explicit = settings.get("base_url")
    transport = _http(explicit or url, settings.get("destination", "cloud") if explicit else destination, consent)
    if transport["mode"] == "configured" and (explicit is not None or gate):
        return transport
    fallback = settings.get("local_base_url")
    local = _http(fallback or DEFAULT_COMPACTION_LOCAL_URL,
                  settings.get("local_destination", "cloud") if fallback else "local", False)
    if local["processing"] == "local" and local["mode"] == "configured":
        return local
    return _http(DEFAULT_COMPACTION_LOCAL_URL, "local", False)


def _reasoner_boundaries(config: Mapping, root: Path, cloud_withdrawn: bool) -> tuple[dict, set[str]]:
    reasoner, router = _section(config, "reasoner"), _section(config, "router")
    compact_cfg = _section(config, "orchestrator", "context", "compaction")
    consent = _section(reasoner, "cloud").get("consent") is True
    cloud_configured = consent and not cloud_withdrawn
    destination = reasoner.get("destination", "cloud")
    url = reasoner.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY)
    declared_gate = cloud_reasoner_allowed(reasoner)
    gate = declared_gate and not cloud_withdrawn
    mode = reasoner.get("mode", "http")
    if mode == "http" and is_unresolved_placeholder(url):
        url, destination = DEFAULT_LOCAL_LLM_GATEWAY, "cloud"

    primary = _transport()
    local_model = reasoner.get("model_path") if mode not in {"http", "hermes_cli"} else None
    if mode == "http":
        primary = _http(url, destination, cloud_configured)
        if not declared_gate:
            primary["mode"] = "blocked"
    elif mode == "hermes_cli":
        primary = _transport("configured" if cloud_configured else "blocked", "cli", "cloud",
                             identity=reasoner.get("hermes_binary_path", "hermes") or "hermes")
    elif local_model:
        primary = _transport("configured", "in_process", "local", identity=str(_at(root, local_model)))

    fast_on = router.get("enabled", True)
    fast = (_http(router.get("base_url", "http://127.0.0.1:8101/v1"),
                  router.get("destination", "cloud"), cloud_configured) if fast_on else _transport())
    compaction = _transport()
    if compact_cfg.get("enabled", False):
        compaction = primary.copy() if mode == "hermes_cli" else _compaction(
            compact_cfg, url, destination, gate, cloud_configured and gate,
        )

    failover = _section(config, "llm", "failover")
    clients = {"fast_router_7b"}
    if primary["mode"] == "configured":
        clients.add("reasoner_cloud")
    failover_on = fast_on and any(ep["id"] in clients for ep in failover.get("endpoints") or ())
    enabled = {
        role for role, active in {
            "http_reasoners": any(ep["mode"] == "configured" and ep["protocol"] in {"http", "https"}
                                  for ep in (primary, compaction)),
            "hermes_reasoners": primary["mode"] == "configured" and primary["protocol"] == "cli",
            "local_reasoner": bool(local_model),
            "fast_router": fast_on,
            "llm_router": failover_on,
            "compactor": compaction["mode"] == "configured",
            "llm_command_router": fast_on and _section(config, "nlu", "llm_router").get("enabled", False),
        }.items() if active
    }
    adapter = reasoner.get("lora_path") if local_model else None
    return {
        "primary": primary, "fast_router": fast, "compaction": compaction,
        "local_adapter_sha256": _path_sha(root, adapter) if adapter else None,
        "cooldown_path_sha256": _path_sha(
            root, _section(failover, "cooldowns").get("persist_path", DEFAULT_COOLDOWN_PATH),
        ),
        "cooldown_mode": "configured" if failover_on else "history_only",
        "cloud_consent": consent, "cloud_withdrawn": cloud_withdrawn,
        "egress_authority": "original_owner",
    }, enabled


def resolve_memory_family_plans(
    config: Mapping, state_root: str | Path, *, cloud_withdrawn: bool = False,
    console_log: str | Path | None = None,
) -> dict[str, dict]:
    """Resolve detached plans without opening files, SDKs or transports.

    Args:
        config: Effective settings at the composition boundary.
        state_root: Absolute managed working root for relative destinations.
        cloud_withdrawn: Observation of the durable/runtime gate at planning.
        console_log: Optional destination declared by the process launcher.

    Returns:
        Metadata plans keyed by matrix producer.
    """
    root = Path(state_root)
    _require(root.is_absolute() and type(cloud_withdrawn) is bool, "publisher_plan")
    memory = _section(config, "memory")
    orchestrator = _section(config, "orchestrator")
    persistence = _section(orchestrator, "context", "persistence")
    obs = _section(config, "observability")
    metrics = _at(root, obs.get("metrics_path", DEFAULT_METRICS_PATH))
    active_metrics = diagnostic_projection_path(metrics)
    transport, transport_roles = _reasoner_boundaries(config, root, cloud_withdrawn)
    context_mode = "history_only"
    if orchestrator.get("enabled", True):
        context_mode = ("files_if_compactor_available"
                        if persistence.get("enabled") and "compactor" in transport_roles else "memory_only")
    context_root = _at(root, persistence.get("base_path", "data/contexts"))

    optional_roles = {
        "personal_memory": {"memory_services", "long_term_memory"} if memory.get("enabled", True) else set(),
        "contexts": {name for name, mode in zip(
            ("context_manager", "context_persister"), _CONTEXT_MODES[context_mode], strict=True,
        ) if mode != "disabled"},
        "reasoner_transports": transport_roles,
        "diagnostics": {"metrics_emitter"} if obs.get("enabled", True) else set(),
    }
    logging_file = _section(config, "logging").get("file")
    collector = _collector(obs)
    boundaries = {
        "house_vector_index": {
            "chroma_path_sha256": _path_sha(root, _section(config, "vectordb").get("path", "./data/chroma_db")),
            "base_collection": LEGACY_COMMANDS_COLLECTION, "pointer_collection": pointer_name(),
            "active_collection": "resolved_by_pointer", "sdk_store": "opaque_sdk_store",
            "model_cache": "sdk_unconfirmed",
        },
        "personal_memory": {
            "chroma_path_sha256": _path_sha(root, memory.get("chroma_path", "./data/memory_db")),
            "preferences_path_sha256": _path_sha(root, memory.get("preferences_path", "./data/preferences.json")),
            "privacy_db_path_sha256": _path_sha(root, memory.get("privacy_db_path", "./data/memory_privacy.db")),
            "policy_collection_sha256": _sha(memory.get("long_term_collection", "user_memories") + "_policy_v1"),
            "authority": "privacy_authority", "historical_targets": "erasure_ledger",
        },
        "contexts": {
            "policy_v2_sha256": _path_sha(root, context_root / "policy_v2"),
            "policy_v3_sha256": _path_sha(root, context_root / "policy_v3"),
            "mode": context_mode, "historical_targets": "erasure_ledger",
        },
        "model_resources": _model_boundaries(config, root),
        "reasoner_transports": transport,
        "diagnostics": {
            "legacy_jsonl_sha256": _path_sha(root, metrics), "jsonl_sha256": _path_sha(root, active_metrics),
            "rotation_sha256": _path_sha(root, active_metrics.with_suffix(active_metrics.suffix + ".1")),
            "sqlite_sha256": _path_sha(root, active_metrics.with_suffix(active_metrics.suffix + ".sqlite3")),
            "rotation_bytes": obs.get("rotate_bytes", 100 * 1024 * 1024),
            "historical_targets": "erasure_ledger", "logstash": collector,
        },
        "logging": {
            "stderr": "inherited", "stdout": "inherited", "bootstrap": "standard_stream_handlers",
            "configured_file_sha256": _path_sha(root, logging_file) if logging_file else None,
            "console_redirection_sha256": _path_sha(root, console_log) if console_log else None,
            "diagnostic_forwarder": collector, "sdk_handlers": "unconfirmed",
            "stream_destinations": "unconfirmed", "log_authority": "original_owner",
        },
    }
    return {
        producer: publisher_plan(
            kind, roles=tuple(_ROLES[producer]), boundaries=boundaries[producer],
            role_modes={name: role.mode.value if not role.optional or name in optional_roles.get(producer, ())
                        else "disabled" for name, role in _ROLES[producer].items()},
        )
        for producer, kind in _KINDS.items()
    }


def memory_family_selection(plans: Mapping[str, dict]) -> dict[str, bool]:
    """Return the selection for coordinator construction with the family specs."""
    return {producer: _selected(plans[producer]) for producer in _KINDS}


@dataclass(frozen=True, eq=False, slots=True)
class _AttachedOwner:
    instance: object = field(repr=False)
    identity: str


@dataclass(eq=False, slots=True)
class MemoryFamilyPublisher:
    """Keep exact owner identity for one row until the original owners drain."""

    _publisher: GenericOutputPublisher
    _modes: dict[str, str]
    _state_root: Path
    _boundaries: dict
    _owners: dict[str, _AttachedOwner] = field(default_factory=dict)
    _opening: set[str] = field(default_factory=set)
    _opened: set[str] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _terminal: bool = False

    def _matches_boundary(self, role: str, owner: object) -> bool:
        targets = self._boundaries
        if role == "chroma_sync":
            return (targets["chroma_path_sha256"] == _path_sha(self._state_root, owner.chroma_path)
                    and targets["base_collection"] == owner.commands_collection_name)
        if role == "context_persister":
            return targets["policy_v3_sha256"] == _path_sha(self._state_root, owner.base_path)
        if role == "metrics_emitter":
            return (targets["jsonl_sha256"] == _path_sha(self._state_root, owner.path)
                    and targets["rotation_bytes"] == owner.rotate_bytes)
        if role == "policy_index_eraser":
            return (targets["chroma_path_sha256"] == _path_sha(self._state_root, owner.descriptor["path"])
                    and targets["policy_collection_sha256"] == _sha(owner.descriptor["collection"]))
        if role == "llm_router":
            return targets["cooldown_path_sha256"] == _path_sha(self._state_root, owner._cd.persistence_path)
        if role == "bootstrap_logging":
            return owner is logging.getLogger()
        return True

    def _matches_role(self, role: str, owner: object) -> bool:
        bound = self._owners.get(role)
        return (bound is not None and bound.instance is owner and not getattr(owner, "_capture_unreviewed", False)
                and self._matches_boundary(role, owner))

    def matches(self, owners: Mapping[str, object]) -> bool:
        """Reconcile the supplied live graph projection by exact identity."""
        active = {role for role, mode in self._modes.items() if mode != "disabled"}
        if owners.keys() != active or not all(self._matches_role(role, owner) for role, owner in owners.items()):
            return False
        if "memory_services" in owners:
            manager = owners["memory_services"].manager
            return (manager is not None and manager.long_term is owners["long_term_memory"]
                    and manager.long_term.ledger is owners["policy_index_eraser"].privacy.ledger
                    and self._boundaries["preferences_path_sha256"]
                    == _path_sha(self._state_root, manager.preferences.file_path))
        if "context_manager" in owners:
            return owners["context_manager"].persister is owners.get("context_persister")
        if "diagnostic_services" in owners:
            return owners["diagnostic_services"].emitter is owners.get("metrics_emitter")
        return True

    async def attach(self, role: str, owner: object) -> None:
        """Pin the original instance after intent and before it can open."""
        async with self._lock:
            _require(not self._terminal and self._modes.get(role, "disabled") != "disabled"
                     and role not in self._owners and _ROLES[self._publisher.producer][role].accepts(owner)
                     and self._matches_boundary(role, owner)
                     and not any(binding.instance is owner for binding in self._owners.values()),
                     "publisher_owner")
            if role == "chroma_sync":
                _require(owner._client is None and not owner._ready and owner._closing is None, "publisher_owner")
            if not self._owners:
                await self._publisher.attached()
            self._owners[role] = _AttachedOwner(owner, _sha(uuid4().hex))

    async def opening(self, role: str, owner: object) -> None:
        """Await the durable opening boundary before invoking this exact owner."""
        async with self._lock:
            _require(not self._terminal and self._matches_role(role, owner), "publisher_owner")
            if not self._opening:
                await self._publisher.opening()
            self._opening.add(role)

    async def opened(self, role: str, owner: object) -> None:
        """Confirm that the original opening or graph handoff succeeded."""
        async with self._lock:
            _require(not self._terminal and role in self._opening and self._matches_role(role, owner),
                     "publisher_owner")
            self._opened.add(role)

    async def drained(self, owners: Mapping[str, object]) -> None:
        """Publish identity evidence after the caller observes original drains."""
        async with self._lock:
            _require(not self._terminal and self.matches(owners)
                     and self._opening == self._opened
                     and all(mode in {"disabled", "lazy"} or role in self._opened
                             for role, mode in self._modes.items())
                     and all(_ROLES[self._publisher.producer][role].drained(owner)
                             for role, owner in owners.items()), "publisher_owner")
            if not self._opened:
                # Complete the journal sequence for a wholly lazy row. Its
                # aggregate records that no native open was requested.
                await self._publisher.opening()
            aggregate = {}
            for role in self._modes:
                binding = self._owners.get(role)
                observation = "disabled"
                if binding is not None:
                    observation = "drained" if role in self._opened else "lazy_unopened"
                aggregate[role] = {"observation": observation,
                                   "instance_sha256": binding.identity if binding else None}
            await self._publisher.drained({"owners": aggregate})
            self._terminal = True


async def bind_memory_family_publishers(
    coordinator: OutputRegistrationV2, plans: Mapping[str, dict],
) -> dict[str, MemoryFamilyPublisher]:
    """Reserve the family on an initialized V4 coordinator.

    Await reservation before invoking the original factories. Attach their
    instances before native use, then pair opening/opened around the original
    work. Reconcile them against the graph at composition and publish evidence
    after their existing drains.
    """
    selection = memory_family_selection(plans)
    header = {"selected": selection}
    events = {}
    for spec in MEMORY_GRAMMAR_SPECS:
        producer = spec.producer
        phase = "intent" if selection[producer] else "disabled"
        event = {"kind": spec.event_kind, "phase": phase, "plan": plans[producer]}
        events[producer] = spec.decode_event(event, None, selection[producer], header)["plan"]

    # Use the same admission snapshot and disk worker as the legacy publishers.
    await coordinator._run(coordinator.admission.revalidate)
    publishers = {}
    for producer, plan in events.items():
        if selection[producer]:
            publisher = await coordinator.publisher_intent(producer, plan)
            publishers[producer] = MemoryFamilyPublisher(
                publisher, plan["role_modes"], coordinator.admission.state_root, plan["boundaries"],
            )
        else:
            await coordinator.publisher_disabled(producer, plan)
    return publishers
