"""Native ``sd_notify`` client (no libsystemd / python-systemd dependency).

Sends ``READY=1``, ``WATCHDOG=1`` and ``STOPPING=1`` datagrams to
``$NOTIFY_SOCKET``. When the variable is absent (macOS, tests, manual runs) every
call is a clean no-op that returns ``False``.
"""

import asyncio
import logging
import os
import socket
from collections.abc import Mapping

logger = logging.getLogger(__name__)

DEFAULT_WATCHDOG_INTERVAL_S = 15.0


class SdNotifier:
    """Minimal systemd notification sender.

    Args:
        env: Environment mapping (injectable for tests); defaults to ``os.environ``.
    """

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        env = os.environ if env is None else env
        raw = env.get("NOTIFY_SOCKET", "")
        self._address: str | bytes | None = None
        if raw.startswith("@"):
            self._address = b"\0" + raw[1:].encode()
        elif raw.startswith("/"):
            self._address = raw
        self._watchdog_usec = self._parse_watchdog(env)

    @staticmethod
    def _parse_watchdog(env: Mapping[str, str]) -> int | None:
        """Return ``WATCHDOG_USEC`` when it targets this process."""
        pid = env.get("WATCHDOG_PID")
        if pid and pid != str(os.getpid()):
            return None
        try:
            usec = int(env.get("WATCHDOG_USEC", ""))
        except ValueError:
            return None
        return usec if usec > 0 else None

    @property
    def enabled(self) -> bool:
        """Whether a notification socket is configured."""
        return self._address is not None

    def watchdog_interval_s(self, default: float = DEFAULT_WATCHDOG_INTERVAL_S) -> float:
        """Ping interval: half of ``WATCHDOG_USEC`` when set, otherwise ``default``."""
        if self._watchdog_usec is None:
            return default
        return min(default, self._watchdog_usec / 2 / 1_000_000)

    def notify(self, *states: str) -> bool:
        """Send newline-joined ``KEY=VALUE`` states.

        Args:
            *states: Assignments such as ``"READY=1"`` or ``"STATUS=loading"``.

        Returns:
            ``True`` if the datagram was sent, ``False`` when disabled or on error.
        """
        if self._address is None or not states:
            return False
        payload = "\n".join(states).encode()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(payload, self._address)
        except OSError as exc:
            logger.warning("sd_notify send failed: %s", exc.strerror or type(exc).__name__)
            return False
        return True

    async def watchdog_loop(self, interval_s: float | None = None) -> None:
        """Send ``WATCHDOG=1`` forever; the ping proves the event loop is not blocked.

        Args:
            interval_s: Override of :meth:`watchdog_interval_s`.
        """
        if not self.enabled:
            return
        interval = self.watchdog_interval_s() if interval_s is None else interval_s
        while True:
            self.notify("WATCHDOG=1")
            await asyncio.sleep(interval)
