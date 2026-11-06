"""A queued consent withdrawal for the next fully closed Graph generation."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from contextvars import Context
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading

import yaml

from src.core.runtime_admission import RuntimeAdmission
from src.dashboard.authority import DashboardAuthority, DashboardLease


@dataclass(frozen=True)
class PendingConsentWithdrawal:
    """An intention, not a durable-write or resource-retirement receipt."""

    config_sha256: str
    previous_consent: bool
    requested_by: str
    next_generation_only: bool = True
    state: str = "pending"


@dataclass(frozen=True)
class ConsentWithdrawalReceipt:
    previous_config_sha256: str
    config_sha256: str
    consent: bool = False
    next_generation_only: bool = True


class ConfigurationChanges:
    """Write only after later owned exits succeeded, while auth remains live.

    main installs this immediately after the DashboardAuthority exit.  It does
    not change a live client's construction-time EgressPolicy or grant consent.
    """

    def __init__(self, authority, admission):
        if authority is not None and type(authority) is not DashboardAuthority:
            raise TypeError("configuration_authority_required")
        if admission is not None and type(admission) is not RuntimeAdmission:
            raise TypeError("configuration_admission_required")
        self._authority = authority
        self._admission = admission
        self._graph = None
        self._privacy = None
        self._privacy_path = None
        self._installed = False
        self._closed = False
        self._pending = None
        self._actor = None
        self._pin = None
        self._receipt = None
        self._writing = None
        self._lock = threading.RLock()

    @property
    def pending(self) -> PendingConsentWithdrawal | None:
        return self._pending

    @property
    def receipt(self) -> ConsentWithdrawalReceipt | None:
        return self._receipt

    def install_on(self, stack: AsyncExitStack) -> None:
        if not isinstance(stack, AsyncExitStack):
            raise TypeError("configuration_owned_stack_required")
        if self._installed or self._closed:
            raise RuntimeError("configuration_owner_already_installed")
        self._installed = True
        stack.push_async_exit(self._after_graph_close)

    def attach_graph(self, graph) -> None:
        module = sys.modules.get("src.application")
        expected = None if module is None else getattr(module, "ApplicationGraph", None)
        if (self._closed or self._graph is not None or expected is None
                or type(graph) is not expected or graph.dashboard_authority is not self._authority):
            raise RuntimeError("configuration_public_graph_required")
        privacy_module = sys.modules.get("src.memory.privacy_services")
        privacy_type = None if privacy_module is None else getattr(privacy_module, "PrivacyServices", None)
        if privacy_type is None or type(graph.privacy) is not privacy_type:
            raise RuntimeError("configuration_public_graph_required")
        privacy_path = None
        if self._admission is not None:
            privacy_path = Path(graph.privacy.ledger.path).resolve(strict=True)
            if not privacy_path.is_relative_to(Path(self._admission.state_root)):
                raise RuntimeError("configuration_public_graph_required")
        self._privacy = graph.privacy
        self._privacy_path = privacy_path
        self._graph = graph

    def _require_actor(self, actor) -> None:
        if (type(self._authority) is not DashboardAuthority or type(actor) is not DashboardLease
                or actor._authority is not self._authority
                or not actor.current(frozenset({"admin:manage"}))):
            raise PermissionError("configuration_actor_denied")

    def _location(self):
        if type(self._admission) is not RuntimeAdmission:
            raise RuntimeError("configuration_admission_required")
        root = Path(self._admission.state_root)
        directory, path = root / "config", root / "config" / "settings.yaml"
        try:
            if (not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink()
                    or directory.resolve(strict=True) != directory or directory.is_symlink()
                    or path.resolve(strict=True) != path or path.is_symlink()
                    or not directory.is_dir() or not path.is_file()):
                raise ValueError("configuration_path_invalid")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
                raise ValueError("configuration_path_invalid")
            roots = tuple((item.stat().st_dev, item.stat().st_ino) for item in (root, directory))
            return path, roots, (info.st_dev, info.st_ino), stat.S_IMODE(info.st_mode)
        except OSError:
            raise ValueError("configuration_path_invalid") from None

    @staticmethod
    def _parse(raw):
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            raise ValueError("configuration_yaml_invalid") from None
        if type(value) is not dict:
            raise ValueError("configuration_yaml_invalid")
        reasoner = value.get("reasoner", {})
        cloud = reasoner.get("cloud", {}) if type(reasoner) is dict else None
        if type(reasoner) is not dict or type(cloud) is not dict:
            raise ValueError("configuration_consent_invalid")
        consent = cloud.get("consent", False)
        if type(consent) is not bool:
            raise ValueError("configuration_consent_invalid")
        return value, consent

    def _read(self):
        path, roots, identity, mode = self._location()
        raw = path.read_bytes()
        again = self._location()
        if (roots, identity) != (again[1], again[2]):
            raise RuntimeError("configuration_file_changed")
        return path, roots, identity, mode, raw, hashlib.sha256(raw).hexdigest()

    def queue_consent_withdrawal(self, actor) -> PendingConsentWithdrawal:
        """Reserve only false, with the original actor and current head/SHA."""
        with self._lock:
            if not self._installed or self._closed:
                raise RuntimeError("configuration_owner_not_open")
            self._require_actor(actor)
            if self._admission is None:
                raise RuntimeError("configuration_admission_required")
            self._admission.revalidate()
            path, roots, identity, _mode, raw, digest = self._read()
            _value, consent = self._parse(raw)
            self._require_actor(actor)
            self._admission.revalidate()
            pin = (path, roots, identity, digest)
            if self._pending is not None:
                if self._actor is not actor or self._pin != pin:
                    raise RuntimeError("configuration_file_changed")
                return self._pending
            self._pin = pin
            self._actor = actor
            self._pending = PendingConsentWithdrawal(digest, consent, actor.principal.subject)
            return self._pending

    def _require_graph_closed(self) -> None:
        if (self._graph is None or self._graph.privacy is not self._privacy
                or self._graph.dashboard_authority is not self._authority):
            raise RuntimeError("configuration_graph_close_unconfirmed")
        try:
            if (self._privacy_path is not None
                    and Path(self._privacy.ledger.path).resolve(strict=True) != self._privacy_path):
                raise RuntimeError("configuration_graph_close_unconfirmed")
        except OSError:
            raise RuntimeError("configuration_graph_close_unconfirmed") from None
        status = self._privacy.resource_status()
        if (status["writer_lease"] is not False or status["closing"] is not False
                or status["close_phase"] != "complete"):
            raise RuntimeError("configuration_graph_close_unconfirmed")

    def _commit(self):
        """All native errors are returned as fixed data, not Task controls."""
        try:
            with self._lock:
                self._require_graph_closed()
                self._require_actor(self._actor)
                self._admission.revalidate()
                path, roots, identity, mode, raw, digest = self._read()
                if (path, roots, identity, digest) != self._pin:
                    raise RuntimeError("configuration_file_changed")
                value, _consent = self._parse(raw)
                value.setdefault("reasoner", {}).setdefault("cloud", {})["consent"] = False
                encoded = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
                # Work on RAW YAML, never serialize graph.config with resolved secrets.
                fd, temporary = tempfile.mkstemp(prefix=".consent-", dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        os.fchmod(stream.fileno(), mode)
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._require_actor(self._actor)
                    self._admission.revalidate()
                    check = self._read()
                    if (check[0], check[1], check[2], check[5]) != self._pin:
                        raise RuntimeError("configuration_file_changed")
                    os.replace(temporary, path)
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                receipt = ConsentWithdrawalReceipt(digest, hashlib.sha256(encoded).hexdigest())
                return receipt, None
        except BaseException as error:
            # Do not retain YAML, traceback, file paths or native control objects.
            fixed = {"configuration_actor_denied", "configuration_file_changed",
                "configuration_path_invalid", "configuration_yaml_invalid",
                "configuration_consent_invalid", "configuration_graph_close_unconfirmed"}
            code = "configuration_write_unconfirmed"
            if (type(error) in (PermissionError, RuntimeError, ValueError)
                    and len(error.args) == 1 and type(error.args[0]) is str and error.args[0] in fixed):
                code = error.args[0]
            kind = "permission" if type(error) is PermissionError else "error"
            return None, (kind, code)

    async def _after_graph_close(self, exc_type, exc, traceback) -> bool:
        self._closed = True
        if exc_type is not None or self._pending is None:
            return False
        self._require_graph_closed()
        # Construct the retained Task directly: a caller task factory cannot
        # dispatch the writer and then lose its returned handle on publication.
        self._writing = asyncio.Task(asyncio.to_thread(self._commit), context=Context())
        cancelled = False
        while not self._writing.done():
            try:
                await asyncio.shield(self._writing)
            except asyncio.CancelledError:
                cancelled = True
        receipt, failure = self._writing.result()
        self._receipt = receipt
        self._actor = None
        if failure is not None:
            error = PermissionError(failure[1]) if failure[0] == "permission" else RuntimeError(failure[1])
            if cancelled:
                raise BaseExceptionGroup("configuration write and observer cancellation",
                    [error, asyncio.CancelledError()])
            raise error
        if cancelled:
            raise asyncio.CancelledError
        return False
