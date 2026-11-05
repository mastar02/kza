"""Explicit administrative authority for a single native calibration capture."""
from __future__ import annotations

import ipaddress
import re
import sys
import threading
from collections.abc import Callable
from urllib.parse import urlsplit

import requests


_UNCONFIRMED: set[object] = set()
_UNCONFIRMED_LOCK = threading.Lock()
_CAPABILITY = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def _origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError
        host = parsed.hostname.lower()
        local = host == "localhost"
        if not local:
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        if parsed.scheme == "http" and not local:
            raise ValueError
        port = parsed.port
        netloc = f"[{host}]" if ":" in host else host
        if port is not None:
            netloc += f":{port}"
        return f"{parsed.scheme}://{netloc}"
    except (ValueError, TypeError, AttributeError):
        raise ValueError("calibration_authority_origin_invalid") from None


class CalibrationAuthorityClient:
    """Own a capture capability and observe stop without granting authority."""

    def __init__(self, url: str, token: str, *, timeout_s: float = 1.0,
                 poll_interval_s: float = .25,
                 session_factory: Callable = requests.Session) -> None:
        self._origin = _origin(url)
        if (not isinstance(token, str) or not token or len(token) > 4096
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("calibration_authority_credential_invalid")
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("calibration_authority_timing_invalid")
        self._credential: str | None = token
        self._capability: str | None = None
        self._session = session_factory()
        # The explicitly selected actor and authority must not be replaced by
        # ambient .netrc credentials or redirected through environment proxies.
        self._session.trust_env = False
        self._timeout = timeout_s
        self._interval = poll_interval_s
        self._registered = self._activation_entered = self._active = False
        self._closed = self._retired = False
        self._stop = threading.Event()
        self._monitor_stop = threading.Event()
        self._monitor_exited = threading.Event()
        self._thread = self._handle = self._on_stop = None
        self._monitor_start_entered = False
        self._monitor_done = False
        self._uncertain = False

    def _request(self, action: str, payload: dict, *, actor: bool = True) -> dict:
        response = None
        try:
            if self._closed:
                raise RuntimeError
            headers = {"Origin": self._origin}
            if actor:
                headers["Authorization"] = "Bearer " + self._credential
            response = self._session.post(
                self._origin + "/api/capture/native/calibration/" + action,
                json=payload, headers=headers, timeout=self._timeout,
                allow_redirects=False,
            )
            if response.status_code != 200:
                raise RuntimeError
            data = response.json()
            if (not isinstance(data, dict)
                    or action != "retire" and data.get("kind") != "calibration"):
                raise RuntimeError
            return data
        except Exception:
            raise RuntimeError("calibration_authority_unconfirmed") from None
        finally:
            if response is not None:
                response.close()
            response = payload = None

    def register(self) -> None:
        """Reserve once under the original administrative credential."""
        if self._registered or self._closed:
            raise RuntimeError("calibration_registration_already_attempted")
        self._registered = True
        data = self._request("register", {})
        token = data.get("token")
        if not isinstance(token, str) or _CAPABILITY.fullmatch(token) is None:
            raise RuntimeError("calibration_authority_unconfirmed")
        self._capability = token

    def activate(self) -> None:
        """Require explicit permission before any native capture startup."""
        if self._activation_entered or self._capability is None or self._closed:
            raise RuntimeError("calibration_activation_unavailable")
        self._activation_entered = True
        data = self._request("activate", {"token": self._capability})
        if (data.get("capture_allowed") is not True or data.get("state") != "active"
                or data.get("retirement_required") is not False):
            self._stop.set()
            raise RuntimeError("calibration_capture_not_allowed")
        self._active = True

    def current(self) -> bool:
        """Return a local stop fence without network I/O in audio callbacks."""
        return self._active and not self._closed and not self._stop.is_set()

    def _seal(self) -> None:
        self._stop.set()
        callback = self._on_stop
        try:
            if callback is not None:
                callback()
        finally:
            callback = None

    def _monitor(self) -> None:
        try:
            while not self._monitor_stop.wait(self._interval):
                try:
                    data = self._request("poll", {"token": self._capability})
                    if (data.get("capture_allowed") is not True or data.get("state") != "active"
                            or data.get("retirement_required") is not False):
                        self._seal()
                        return
                except BaseException:
                    self._seal()
                    return
        finally:
            self._monitor_exited.set()

    def start_monitor(self, on_stop: Callable[[], None]) -> None:
        """Retain the actual CPython native thread before dispatch."""
        if not self.current() or self._thread is not None or not callable(on_stop):
            raise RuntimeError("calibration_monitor_unavailable")
        thread = threading.Thread(target=self._monitor, daemon=True)
        handle = getattr(thread, "_handle", None)
        handle_type = getattr(threading, "_ThreadHandle", None)
        if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13)
                or handle_type is None or type(handle) is not handle_type
                or type(getattr(handle, "ident", None)) is not int or handle.ident != 0
                or not callable(getattr(handle, "join", None))
                or not callable(getattr(handle, "is_done", None))):
            raise RuntimeError("calibration_monitor_unsupported_runtime")
        self._on_stop = on_stop
        self._thread, self._handle = thread, handle
        self._monitor_start_entered = True
        try:
            thread.start()
        except BaseException:
            self._stop.set()
            raise

    def stop_monitor(self) -> None:
        """Join the same native handle; observer interruption is not exit."""
        self._monitor_stop.set()
        if self._monitor_done or not self._monitor_start_entered:
            return
        handle, thread = self._handle, self._thread
        if handle is None or handle.ident == 0 or thread is threading.current_thread():
            self._uncertain = True
            raise RuntimeError("calibration_monitor_retirement_unconfirmed")
        controls: list[BaseException] = []
        while True:
            try:
                handle.join()
                if not handle.is_done() or not self._monitor_exited.is_set():
                    raise RuntimeError("calibration_monitor_retirement_unconfirmed")
                self._monitor_done = True
                self._thread = self._handle = self._on_stop = None
                break
            except BaseException as error:
                if not isinstance(error, Exception):
                    controls.append(error)
                    continue
                self._uncertain = True
                raise RuntimeError("calibration_monitor_retirement_unconfirmed") from None
        if controls:
            raise controls[0]

    def retire(self) -> None:
        """Send only the explicit local owner's completed-capture receipt."""
        if self._retired or self._capability is None:
            return
        if self._uncertain or (self._monitor_start_entered and not self._monitor_done):
            raise RuntimeError("calibration_monitor_retirement_unconfirmed")
        data = self._request("retire", {"token": self._capability}, actor=False)
        if data.get("retired") is not True:
            raise RuntimeError("calibration_retirement_unconfirmed")
        self._retired = True
        self._active = False
        self._stop.set()

    def close(self) -> None:
        """Close control-plane resources without inventing capture retirement."""
        if self._closed:
            return
        self._stop.set()
        try:
            self.stop_monitor()
        except BaseException:
            with _UNCONFIRMED_LOCK:
                _UNCONFIRMED.add(self)
            raise
        try:
            self._session.close()
        except Exception:
            raise RuntimeError("calibration_authority_close_unconfirmed") from None
        self._closed = True
        self._session = self._credential = self._capability = self._on_stop = None
