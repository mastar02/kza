"""Content-free ownership of one original server authorization and revision."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReaderBinding:
    """An opaque namespace plus the immutable original reader's live guard."""

    namespace: str = field(repr=False)
    _current: Callable[[], bool] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (type(self.namespace) is not str or len(self.namespace) != 64
                or any(char not in '0123456789abcdef' for char in self.namespace)
                or not callable(self._current)):
            raise ValueError('invalid_reader_binding')

    def current(self) -> bool:
        """A failed or retired original grant never selects a replacement reader."""
        try:
            return self._current() is True
        except Exception:
            return False

    def require_current(self) -> None:
        """Use the same guard inside native locks and immediately before commits."""
        if not self.current():
            raise PermissionError('dashboard_reader_retired')
