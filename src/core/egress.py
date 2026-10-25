"""Explicit data-processing destinations, independent of network topology."""

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import unquote, urlsplit


class ProcessingDestination(StrEnum):
    """Where data is processed, including a gateway's upstream backend."""

    LOCAL = "local"
    CLOUD = "cloud"


class EgressDenied(PermissionError):
    """Data cannot be sent under the declared processing policy."""


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", ProcessingDestination(self.destination))
        if type(self.cloud_consent) is not bool:
            raise ValueError("cloud consent must be a boolean")
        if self.endpoint is not None:
            _url_identity(self.endpoint)
        elif self.destination is ProcessingDestination.LOCAL:
            raise ValueError("Local processing requires an explicit endpoint")

    @classmethod
    def for_http(cls, endpoint: str, destination: str = "cloud", cloud_consent: bool = False) -> "EgressPolicy":
        """Require a real URL; None must never select an SDK's default provider."""
        if not isinstance(endpoint, str):
            raise ValueError("HTTP processing requires an explicit endpoint")
        return cls(destination, cloud_consent, endpoint)

    @property
    def allowed(self) -> bool:
        """Whether this declared processing destination is authorized."""
        return self.destination is ProcessingDestination.LOCAL or self.cloud_consent

    def require_allowed(self, endpoint: str | None = None) -> None:
        """Reject denied processing or retargeting before any network/CLI call."""
        if not self.allowed:
            raise EgressDenied("cloud_consent_required")
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
