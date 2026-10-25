"""Consent for explicitly declared inference destinations, including gateways."""

from dataclasses import dataclass
import logging

from src.core.egress import EgressPolicy, ProcessingDestination
from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY, is_unresolved_placeholder

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_LOCAL_URL = "http://127.0.0.1:8101/v1"
DEFAULT_COMPACTION_LOCAL_MODEL = "local"


def is_cloud_endpoint(base_url: str, *, destination: str = "cloud") -> bool:
    """Classify declared processing; loopback alone never proves local inference.

    Unknown/invalid declarations and endpoints are treated as cloud. The
    built-in local 7B fallback is declared explicitly at its construction site.
    """
    try:
        policy = EgressPolicy(destination, False, base_url)
    except (TypeError, ValueError):
        return True
    return policy.destination is ProcessingDestination.CLOUD


def cloud_reasoner_allowed(reasoner_config: dict) -> bool:
    """Allow local processing or cloud processing with literal boolean consent."""
    consent = reasoner_config.get("cloud", {}).get("consent", False)
    destination = reasoner_config.get("destination", "cloud")
    # Endpoint validation belongs to resolution/client construction. An opaque
    # placeholder must not acquire a LOCAL grant before it is resolved.
    endpoint = reasoner_config.get("http_base_url")
    if "http_base_url" in reasoner_config and endpoint is None:
        return False
    if is_unresolved_placeholder(endpoint):
        destination, endpoint = "cloud", None
    try:
        policy = EgressPolicy(destination, consent, endpoint)
    except (TypeError, ValueError):
        return False
    return policy.allowed


def resolve_http_reasoner_base_url(reasoner_config: dict, default_local_url: str) -> tuple[bool, str]:
    """Evaluate consent before resolving a missing gateway variable in-place."""
    allowed = cloud_reasoner_allowed(reasoner_config)
    if is_unresolved_placeholder(reasoner_config.get("http_base_url")):
        logger.warning("[Consent] Unresolved reasoner gateway; resolving configured fallback")
        reasoner_config["http_base_url"] = default_local_url
        reasoner_config["destination"] = "cloud"
    return allowed, reasoner_config.get("http_base_url", default_local_url)


def resolve_reasoner_gate(reasoner_config: dict, reasoner_mode: str, default_local_url: str) -> tuple[bool, str | None]:
    """Resolve the HTTP grant also used by compaction; CLI inference is cloud."""
    if reasoner_mode == "http":
        return resolve_http_reasoner_base_url(reasoner_config, default_local_url)
    if reasoner_mode == "hermes_cli":
        return cloud_reasoner_allowed({"cloud": reasoner_config.get("cloud", {})}), None
    return cloud_reasoner_allowed(reasoner_config), reasoner_config.get("http_base_url", default_local_url)


@dataclass(frozen=True, slots=True)
class CompactionEndpoint:
    """A resolved endpoint with its own destination, consent and credentials."""

    base_url: str
    model: str | None
    api_key_env: str | None
    api_style: str
    destination: str = "cloud"
    cloud_consent: bool = False


def resolve_compaction_endpoint(compaction_cfg: dict, reasoner_config: dict, gate_allowed: bool) -> CompactionEndpoint:
    """Resolve compaction without inheriting a LOCAL grant for another backend.

    The global cloud consent and the parent gate must both permit cloud use.
    An explicit endpoint needs its own processing declaration; absent metadata
    defaults to CLOUD. Local processing never inherits a cloud credential.
    """
    consent = gate_allowed is True and reasoner_config.get("cloud", {}).get("consent", False) is True
    explicit = compaction_cfg.get("base_url")
    destination = compaction_cfg.get("destination", "cloud") if explicit else reasoner_config.get("destination", "cloud")
    base_url = explicit or reasoner_config.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY)
    try:
        policy = EgressPolicy(destination, consent, base_url)
        allowed = policy.allowed and (explicit is not None or gate_allowed is True)
    except (TypeError, ValueError):
        allowed = False
    if allowed:
        local = policy.destination is ProcessingDestination.LOCAL
        return CompactionEndpoint(
            base_url, compaction_cfg.get("model") or (DEFAULT_COMPACTION_LOCAL_MODEL if local else reasoner_config.get("http_model")),
            compaction_cfg.get("api_key_env") if local else reasoner_config.get("api_key_env"),
            "chat" if local else reasoner_config.get("api_style", "completions"),
            policy.destination.value, consent,
        )

    # This is the application's existing on-host 7B, not the loopback gateway.
    # A custom fallback must explicitly declare where its backend processes data.
    local_url = compaction_cfg.get("local_base_url") or DEFAULT_COMPACTION_LOCAL_URL
    local_destination = compaction_cfg.get("local_destination", "cloud") if compaction_cfg.get("local_base_url") else "local"
    try:
        fallback = EgressPolicy(local_destination, False, local_url)
        if fallback.destination is not ProcessingDestination.LOCAL:
            raise ValueError("Fallback must process locally")
    except (TypeError, ValueError):
        logger.error("[Consent] Rejected unverified compaction fallback; using the declared local 7B")
        local_url = DEFAULT_COMPACTION_LOCAL_URL
    logger.warning("[Consent] Compaction uses the declared local 7B; no cloud credentials inherited")
    return CompactionEndpoint(
        local_url, compaction_cfg.get("local_model") or DEFAULT_COMPACTION_LOCAL_MODEL,
        None, "chat", "local", False,
    )
