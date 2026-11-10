"""Explicit data-processing destinations, independent of network topology."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import logging
import threading
from urllib.parse import unquote, urlsplit
import weakref

logger = logging.getLogger(__name__)


class ProcessingDestination(StrEnum):
    """Where data is processed, including a gateway's upstream backend."""

    LOCAL = "local"
    CLOUD = "cloud"


class EgressDenied(PermissionError):
    """Data cannot be sent under the declared processing policy."""


class ConsentWithdrawalSource(StrEnum):
    """Why the runtime cloud-consent gate was closed."""

    ACTOR = "actor"
    DURABLE_MARKER = "durable_marker"
    MARKER_UNVERIFIABLE = "marker_unverifiable"


class CloudConsentGate:
    """One-way, process-local latch that withdraws cloud consent at runtime.

    The gate never reopens: a re-grant is an operator procedure across boots.
    ``withdraw`` sets the latch and snapshots the subscribers under one lock,
    then calls them outside it; ``subscribe`` either appends or, when already
    withdrawn, calls immediately. Subscribers may run on the event-loop thread
    or a worker thread, so they must be thread-safe, never block and never
    touch asyncio objects. Bound methods are held weakly. A subscriber error is
    logged and never propagates or reopens the gate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._source: ConsentWithdrawalSource | None = None
        self._subscribers: list[Callable[[], Callable[[], None] | None]] = []

    @property
    def withdrawn(self) -> bool:
        """Whether cloud consent has been withdrawn in this process."""
        return self._event.is_set()

    @property
    def source(self) -> ConsentWithdrawalSource | None:
        """The first withdrawal source, or None while open."""
        return self._source

    @staticmethod
    def _reference(callback: Callable[[], None]) -> Callable[[], Callable[[], None] | None]:
        if getattr(callback, "__self__", None) is not None and hasattr(callback, "__func__"):
            return weakref.WeakMethod(callback)
        return lambda: callback

    @staticmethod
    def _call(callback: Callable[[], None] | None) -> None:
        if callback is None:
            return  # The weakly held owner is gone.
        try:
            callback()
        except Exception:
            logger.exception("[Consent] Withdrawal subscriber failed; the gate stays closed")

    def withdraw(self, source: ConsentWithdrawalSource | str) -> bool:
        """Close the gate once; return True only for the closing call."""
        source = ConsentWithdrawalSource(source)
        with self._lock:
            if self._event.is_set():
                return False
            self._source = source
            self._event.set()
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        for reference in subscribers:
            try:
                callback = reference()
            except Exception:
                logger.exception("[Consent] Withdrawal subscriber unavailable")
                continue
            self._call(callback)
        return True

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` once on withdrawal (immediately if already withdrawn)."""
        with self._lock:
            if not self._event.is_set():
                self._subscribers.append(self._reference(callback))
                return
        self._call(callback)

    def require_open(self) -> None:
        """Raise EgressDenied once consent has been withdrawn."""
        if self._event.is_set():
            raise EgressDenied("cloud_consent_withdrawn")

    def __copy__(self) -> "CloudConsentGate":
        return self

    def __deepcopy__(self, memo: dict) -> "CloudConsentGate":
        return self

    def __reduce_ex__(self, protocol: int):
        raise TypeError("cloud_consent_gate_not_serializable")


def _url_identity(url: str) -> tuple[str, str, int, str]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment
        or any(part in {".", ".."} for part in unquote(parsed.path).split("/"))
        or "\\" in unquote(parsed.path)
    ):
        raise ValueError("Invalid processing endpoint")
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname.lower(), port, parsed.path.rstrip("/")


@dataclass(frozen=True)
class EgressPolicy:
    """Bind a declared destination and strict consent to one processing endpoint.

    LOCAL is an operator declaration about the backend, never a conclusion
    derived from a hostname. Missing declarations default to CLOUD. A network
    redirect or a changed endpoint cannot extend this grant.
    """

    destination: ProcessingDestination = ProcessingDestination.CLOUD
    cloud_consent: bool = False
    endpoint: str | None = None
    consent_gate: CloudConsentGate | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", ProcessingDestination(self.destination))
        if type(self.cloud_consent) is not bool:
            raise ValueError("cloud consent must be a boolean")
        if self.consent_gate is not None and type(self.consent_gate) is not CloudConsentGate:
            raise TypeError("cloud consent gate must be a CloudConsentGate")
        if self.endpoint is not None:
            _url_identity(self.endpoint)
        elif self.destination is ProcessingDestination.LOCAL:
            raise ValueError("Local processing requires an explicit endpoint")

    @classmethod
    def for_http(cls, endpoint: str, destination: str = "cloud", cloud_consent: bool = False, *,
                 consent_gate: CloudConsentGate | None = None) -> "EgressPolicy":
        """Require a real URL; None must never select an SDK's default provider."""
        if not isinstance(endpoint, str):
            raise ValueError("HTTP processing requires an explicit endpoint")
        return cls(destination, cloud_consent, endpoint, consent_gate)

    @property
    def allowed(self) -> bool:
        """Whether this declared processing destination is authorized now."""
        if self.destination is ProcessingDestination.LOCAL:
            return True
        return self.cloud_consent and not (self.consent_gate is not None and self.consent_gate.withdrawn)

    def require_allowed(self, endpoint: str | None = None) -> None:
        """Reject denied processing or retargeting before any network/CLI call."""
        if self.destination is not ProcessingDestination.LOCAL:
            if not self.cloud_consent:
                raise EgressDenied("cloud_consent_required")
            if self.consent_gate is not None and self.consent_gate.withdrawn:
                raise EgressDenied("cloud_consent_withdrawn")
        if endpoint is not None and (
            self.endpoint is None or _url_identity(endpoint) != _url_identity(self.endpoint)
        ):
            raise EgressDenied("processing_endpoint_changed")

    def require_request(self, url: str) -> None:
        """Authorize a fixed operation below the bound endpoint, without redirects."""
        self.require_allowed()
        actual = _url_identity(url)
        expected = _url_identity(self.endpoint) if self.endpoint is not None else None
        if expected is None or actual[:3] != expected[:3] or not (
            actual[3] == expected[3] or actual[3].startswith(expected[3] + "/")
        ):
            raise EgressDenied("processing_endpoint_changed")


def require_consent_gate(policy: EgressPolicy) -> None:
    """Client constructors: a consented CLOUD grant must carry the runtime gate.

    Evaluation-only policies (consent resolution) are not clients and do not
    call this; every object that can send data does, so a missing gate fails
    at construction instead of ignoring a later withdrawal.
    """
    if (policy.destination is ProcessingDestination.CLOUD and policy.cloud_consent
            and policy.consent_gate is None):
        raise ValueError("consent_gate_required")
