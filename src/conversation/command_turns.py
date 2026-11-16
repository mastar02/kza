"""
Command turns - whether a voice command is being captured or dispatched.

FollowUpMode only knows the follow-up window: its timer can end the
conversation while the person is still saying a follow-up command, or while
that command is being transcribed and answered. Observers that must not talk
over the person (a held reminder notice, user answer Q33) also watch this.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class CommandTurns:
    """Read-only view of the audio loop's command activity, with end listeners.

    The owning audio loop gives a probe (a command is being captured in some
    room, or a captured command is being dispatched) and calls ``settle`` on
    its event loop whenever that activity may have ended: after each capture
    step and when a dispatch finishes. Listeners are called only when an
    observer saw the activity (``active`` returned True) and it has ended
    since, so an idle loop never wakes anybody.
    """

    def __init__(self, probe: Callable[[], bool]):
        self._probe = probe
        self._listeners: list[Callable[[], None]] = []
        self._observed = False

    @property
    def active(self) -> bool:
        """Whether a command is being captured or dispatched right now."""
        busy = bool(self._probe())
        if busy:
            self._observed = True
        return busy

    def add_end_listener(self, callback: Callable[[], None]) -> None:
        """Register an observer called once the activity it saw has ended.

        Args:
            callback: Called without arguments on the audio loop's event loop;
                it must not block, and its errors are logged and ignored.
        """
        self._listeners.append(callback)

    def settle(self) -> None:
        """Notify the listeners if observed activity has ended (called by the owning loop)."""
        if not self._observed or self._probe():
            return
        self._observed = False
        for callback in tuple(self._listeners):
            try:
                callback()
            except Exception as error:
                logger.warning("Command turn end listener failed: %s", type(error).__name__)
