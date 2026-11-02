"""Immutable account metadata, independent of voice vectors and session grants."""

from dataclasses import dataclass
from enum import StrEnum


class RemovalStatus(StrEnum):
    """Preflight outcomes without copying private account fields."""

    ALLOWED = "allowed"
    MISSING = "missing"
    PROTECTED = "protected"


@dataclass(frozen=True)
class RemovalCheck:
    """Small shared result for preflight and final catalog mutation."""

    status: RemovalStatus
    message: str


@dataclass(frozen=True)
class ActiveAccounts:
    """Nonblocking canonical membership for authority guards."""

    available: bool
    active_ids: frozenset[str]
    revision: int = 0

    def __post_init__(self) -> None:
        if (type(self.available) is not bool or type(self.active_ids) is not frozenset
                or any(type(identity) is not str for identity in self.active_ids)
                or not self.available and bool(self.active_ids)
                or type(self.revision) is not int or self.revision < 0):
            raise ValueError("invalid_active_accounts_snapshot")


@dataclass(frozen=True)
class AccountMetadata:
    """Administrative scalars; account role never selects session authority."""

    id: str
    name: str
    permission_level: str
    is_active: bool

    def __post_init__(self) -> None:
        if (type(self.id) is not str or type(self.name) is not str or type(self.permission_level) is not str
                or self.permission_level not in {'guest', 'child', 'teen', 'adult', 'admin'}
                or type(self.is_active) is not bool):
            raise ValueError('invalid_account_metadata')
