"""Cloud-consent withdrawal: runtime gate, durable journal marker, shutdown write.

OQ-8 order: admin authority -> gate close -> durable marker -> queued write of
``reasoner.cloud.consent: false`` for the next fully closed Graph generation.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from contextvars import Context
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from io import BufferedWriter, FileIO
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading

import yaml

from src.core.egress import CloudConsentGate, ConsentWithdrawalSource
from src.core.runtime_admission import RuntimeAdmission
from src.dashboard.authority import DashboardAuthority, DashboardLease
from src.memory import output_registration_v2 as output_registration

logger = logging.getLogger(__name__)


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


class MarkerRecordState(StrEnum):
    """Whether this run's withdrawal marker is recorded."""

    DURABLE = "durable"
    REGISTRATION_FAILED = "registration_failed"
    UNMANAGED = "unmanaged"
    ALREADY_RECORDED = "already_recorded"


@dataclass(frozen=True)
class ConsentWithdrawalOutcome:
    """Result of one withdrawal request; the gate stays closed whatever follows."""

    gate_closed: bool
    marker: MarkerRecordState
    pending: PendingConsentWithdrawal | None
    failure: str | None = None


@dataclass(frozen=True)
class CloudConsentStatus:
    """Administrative view of the runtime gate, marker and pending shutdown write."""

    gate_withdrawn: bool
    gate_source: str | None
    boot_decision_basis: str
    marker: str | None
    pending_write: bool
    pending_write_cause: str | None
    registration_failure: str | None
    last_settled: str | None


_QUEUE_CODES = frozenset({
    "configuration_actor_denied", "configuration_file_changed", "configuration_path_invalid",
    "configuration_yaml_invalid", "configuration_consent_invalid", "configuration_admission_required",
    "configuration_owner_not_open", "configuration_marker_required",
})


class ConfigurationChanges:
    """Write only after later owned exits succeeded, while auth remains live.

    main installs this immediately after the DashboardAuthority exit.  It does
    not change a live client's construction-time EgressPolicy or grant consent.
    """

    def __init__(self, authority, admission, consent_gate: CloudConsentGate | None = None):
        if authority is not None and type(authority) is not DashboardAuthority:
            raise TypeError("configuration_authority_required")
        if admission is not None and type(admission) is not RuntimeAdmission:
            raise TypeError("configuration_admission_required")
        if consent_gate is not None and type(consent_gate) is not CloudConsentGate:
            raise TypeError("configuration_consent_gate_required")
        self._authority = authority
        self._admission = admission
        self._consent_gate = consent_gate
        self._marker: MarkerRecordState | None = None
        self._withdrawing: asyncio.Task | None = None
        self._withdrawal_failure: str | None = None
        self._queue_failure: str | None = None
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
        self._registration_failure = None
        self._lock = threading.RLock()

    @property
    def pending(self) -> PendingConsentWithdrawal | None:
        return self._pending

    @property
    def receipt(self) -> ConsentWithdrawalReceipt | None:
        return self._receipt

    @property
    def registration_failure(self) -> str | None:
        """The output-registration failure that kept a queued change unwritten."""
        return self._registration_failure

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

    async def withdraw_cloud_consent(self, actor) -> ConsentWithdrawalOutcome:
        """Withdraw cloud consent for an admin actor: gate, durable marker, queued write.

        The work runs in one retained Task observed through ``_settle``, so a
        cancelled caller never cancels it; close awaits it before reading
        ``_pending``. Authorization failure raises PermissionError with no
        effect. Any later failure keeps the gate closed and either stays
        durable (marker recorded) or poisons the epoch (no clean receipt).

        Args:
            actor: The DashboardLease of the requesting administrator.

        Returns:
            The outcome; ``failure`` names a later step that did not complete.
        """
        if not self._installed or self._closed:
            raise RuntimeError("configuration_owner_not_open")
        if type(self._consent_gate) is not CloudConsentGate:
            raise RuntimeError("configuration_consent_gate_required")
        work = self._withdraw(actor, self._withdrawing)
        task = asyncio.Task(work, context=Context())
        self._withdrawing = task
        actor = None
        cancelled = await self._settle(task)
        try:
            outcome = task.result()
        except asyncio.CancelledError:
            outcome = ConsentWithdrawalOutcome(self._consent_gate.withdrawn, MarkerRecordState.REGISTRATION_FAILED,
                                               None, self._withdrawal_failure)
        finally:
            task = None
            if cancelled:
                raise asyncio.CancelledError
        return outcome

    async def _withdraw(self, actor, previous: asyncio.Task | None) -> ConsentWithdrawalOutcome:
        if previous is not None:
            await self._settle(previous)  # Serialize: one marker per run.
        self._require_actor(actor)  # Step 1: no effect on failure.
        gate = self._consent_gate
        confirmed = False
        try:
            # Step 2: no await between the authority check and the gate.
            gate.withdraw(ConsentWithdrawalSource.ACTOR)
            if self._marker is MarkerRecordState.REGISTRATION_FAILED:
                return ConsentWithdrawalOutcome(True, MarkerRecordState.REGISTRATION_FAILED, None,
                                                self._withdrawal_failure)
            if self._marker is None:
                # Step 3: durable (record and HEAD fsynced) before it returns.
                recorded = await output_registration.configuration_withdrawn(self)
                self._marker = MarkerRecordState.DURABLE if recorded else MarkerRecordState.UNMANAGED
                marker = self._marker
            else:
                marker = MarkerRecordState.ALREADY_RECORDED
            confirmed = True
        except BaseException as error:
            if not confirmed and gate.withdrawn:
                self._withdrawal_unconfirmed(error)
            if isinstance(error, Exception):
                return ConsentWithdrawalOutcome(gate.withdrawn, MarkerRecordState.REGISTRATION_FAILED, None,
                                                self._withdrawal_failure)
            raise
        # Step 4: the queued shutdown write; failures here keep the durable marker.
        try:
            pending = await asyncio.to_thread(self._queue_pending, actor)
            self._queue_failure = None
            return ConsentWithdrawalOutcome(True, marker, pending)
        except Exception as error:
            code = "configuration_queue_unconfirmed"
            if (type(error) in (PermissionError, RuntimeError, ValueError) and len(error.args) == 1
                    and error.args[0] in _QUEUE_CODES):
                code = error.args[0]
            self._queue_failure = code
            logger.warning("Consent withdrawal recorded; shutdown write not queued (%s)", code)
            return ConsentWithdrawalOutcome(True, marker, self._pending, code)
        finally:
            actor = previous = None

    def _withdrawal_unconfirmed(self, error: BaseException) -> None:
        """The gate closed but the marker is not confirmed: poison the epoch (fail closed)."""
        if isinstance(error, asyncio.CancelledError):
            code = "output_registration_withdrawal_cancelled"
        elif isinstance(error, RuntimeError) and str(error).startswith("output_registration_"):
            code = str(error)
        else:
            code = "output_registration_unconfirmed"
        self._marker = MarkerRecordState.REGISTRATION_FAILED
        self._withdrawal_failure = code
        try:
            output_registration.configuration_withdrawal_unconfirmed(self)
        finally:
            self._registration_failed(code)
            logger.error("Consent withdrawal marker not recorded (%s); cloud stays off in this run and "
                         "no closed admission receipt follows", code)

    def _queue_pending(self, actor) -> PendingConsentWithdrawal:
        """Reserve only false, with the original actor and current head/SHA (after the marker)."""
        with self._lock:
            if not self._installed or self._closed:
                raise RuntimeError("configuration_owner_not_open")
            if self._marker not in (MarkerRecordState.DURABLE, MarkerRecordState.UNMANAGED):
                raise RuntimeError("configuration_marker_required")
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

    def status(self) -> CloudConsentStatus:
        """Administrative status; never includes actor identity or file content."""
        gate = self._consent_gate
        coordinator = getattr(self, "_output_coordinator", None)
        decision = None if coordinator is None else coordinator.effective_decision
        settled = None if coordinator is None else coordinator.settled_decision
        cause = self._queue_failure
        if self._pending is not None:
            actor = self._actor
            current = actor is not None and actor.current(frozenset({"admin:manage"}))
            cause = None if current else "actor_lease_expired"
        return CloudConsentStatus(
            gate_withdrawn=gate is not None and gate.withdrawn,
            gate_source=None if gate is None or gate.source is None else gate.source.value,
            boot_decision_basis="none" if decision is None else decision["basis"],
            marker=None if self._marker is None else self._marker.value,
            pending_write=self._pending is not None, pending_write_cause=cause,
            registration_failure=self._registration_failure,
            last_settled=None if settled is None else settled["basis"])

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

    @staticmethod
    def _close_raw_descriptor(descriptor: int) -> tuple[int, int] | None:
        """Observe, then close exactly once, a descriptor no file object owns yet."""
        own = None
        try:
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                own = (info.st_dev, info.st_ino)
        except OSError:
            own = None
        finally:
            # A close failure leaves the name unproven; never retry a reusable fd.
            os.close(descriptor)
        return own

    @staticmethod
    def _close_raw_file(raw: FileIO) -> tuple[int, int] | None:
        """Observe, then close through its owning FileIO, a not yet buffered file."""
        own = None
        try:
            info = os.fstat(raw.fileno())
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                own = (info.st_dev, info.st_ino)
        except OSError:
            own = None
        finally:
            raw.close()
        return own

    @staticmethod
    def _owned_identity(stream) -> tuple[int, int]:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("configuration_temporary_changed")
        return info.st_dev, info.st_ino

    @staticmethod
    def _owned_final(stream, own: tuple[int, int]) -> tuple[int, int]:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (info.st_dev, info.st_ino) != own:
            raise OSError("configuration_temporary_changed")
        return info.st_size, info.st_mtime_ns

    @staticmethod
    def _remove_own_temporary(temporary: str | None, own: tuple[int, int] | None,
                              parent: tuple[int, int]) -> bool:
        """Unlink only the exact own allocation; a foreign or unproven name stays."""
        if temporary is None:
            return True
        try:
            info = os.lstat(temporary)
            if info.st_nlink == 1 and stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == own:
                directory = os.lstat(Path(temporary).parent)
                if stat.S_ISDIR(directory.st_mode) and (directory.st_dev, directory.st_ino) == parent:
                    os.unlink(temporary)
                    return True
        except FileNotFoundError:
            return True
        logger.warning("Configuration temporary cleanup unproven; name retained")
        return False

    def _commit(self, extent=None):
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
                parent = roots[1]
                if extent is not None:
                    # Managed: the registered attempt is admitted before any allocation.
                    extent.before_temporary(self, path, roots, identity, digest)
                temporary = own = None
                published = durable = False
                try:
                    # Work on RAW YAML, never serialize graph.config with resolved secrets.
                    descriptor, temporary = tempfile.mkstemp(prefix=".consent-", dir=path.parent)
                    try:
                        raw_file = FileIO(descriptor, "w", closefd=True)
                    except BaseException:
                        # Not transferred: this owner still holds the raw descriptor.
                        own = self._close_raw_descriptor(descriptor)
                        raise
                    # From here the FileIO object owns the descriptor; never os.close it.
                    try:
                        stream = BufferedWriter(raw_file)
                    except BaseException:
                        own = self._close_raw_file(raw_file)
                        raise
                    finally:
                        raw_file = None
                    with stream:
                        own = self._owned_identity(stream)
                        os.fchmod(stream.fileno(), mode)
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                        final = self._owned_final(stream, own)
                    self._require_actor(self._actor)
                    self._admission.revalidate()
                    check = self._read()
                    if (check[0], check[1], check[2], check[5]) != self._pin:
                        raise RuntimeError("configuration_file_changed")
                    if extent is not None:
                        extent.before_replace(temporary, own, parent, final)
                    os.replace(temporary, path)
                    published = True
                    if extent is not None:
                        extent.replaced(own, hashlib.sha256(encoded).hexdigest())
                    directory = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    durable = True
                finally:
                    proven = False
                    try:
                        proven = self._remove_own_temporary(temporary, own, parent)
                    finally:
                        if extent is not None:
                            extent.finish(temporary=temporary, published=published, durable=durable,
                                          cleanup_proven=proven)
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

    def _registration_failed(self, error: BaseException | str) -> None:
        """Record a failed registration once; the change stays unconfirmed (fail closed)."""
        if self._registration_failure is not None:
            return
        message = error if type(error) is str else str(error)
        self._registration_failure = (message if (type(error) is str or isinstance(error, RuntimeError))
            and message.startswith("output_registration_") else "output_registration_unconfirmed")
        logger.error("Configuration change not confirmed: output registration failed (%s); "
                     "no closed admission receipt follows", self._registration_failure)

    @staticmethod
    async def _settle(task: asyncio.Task) -> bool:
        """Await a retained task to completion; return whether the observer was cancelled."""
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    cancelled = True
            except BaseException:
                pass  # The task's own outcome is read by the caller.
        return cancelled

    async def _write(self, extent, errors: list) -> bool:
        """Run the single writer Task; every failure becomes an error entry, never an early exit."""
        writing = None
        try:
            writing = asyncio.to_thread(self._commit, extent)
            # Construct the retained Task directly: a caller task factory cannot
            # dispatch the writer and then lose its returned handle on publication.
            self._writing = asyncio.Task(writing, context=Context())
        except Exception as error:
            if writing is not None:
                writing.close()
            errors.append(error)
            return False
        cancelled = await self._settle(self._writing)
        try:
            receipt, failure = self._writing.result()
        except asyncio.CancelledError:
            errors.append(RuntimeError("configuration_write_unconfirmed"))
            return cancelled
        except Exception as error:
            errors.append(error)
            return cancelled
        self._receipt = receipt
        if failure is not None:
            errors.append(PermissionError(failure[1]) if failure[0] == "permission" else RuntimeError(failure[1]))
        return cancelled

    async def _after_graph_close(self, exc_type, exc, traceback) -> bool:
        self._closed = True
        errors, cancelled, extent = [], False, None
        # N1: settle any withdrawal before reading _pending or the marker.
        if self._withdrawing is not None:
            cancelled = await self._settle(self._withdrawing)
        with self._lock:
            pass  # Barrier: a step-4 worker left behind by cancellation has finished.
        if self._withdrawal_failure is not None:
            errors.append(RuntimeError("configuration_withdrawal_unconfirmed"))
        if exc_type is not None or self._pending is None:
            outcome = "no_pending" if self._pending is None else "not_attempted_graph_failed"
            if self._pending is not None:
                logger.warning("Configuration change not written: an earlier owned exit failed")
        else:
            outcome = "refused_before_temporary"
            try:
                self._require_graph_closed()
            except Exception as error:
                errors.append(error)
            else:
                try:
                    # Registration precedes the writer Task and every file effect.
                    extent = await output_registration.configuration_opening(self)
                except asyncio.CancelledError:
                    cancelled = True
                    # The queued change is dropped unwritten; keep that visible.
                    self._registration_failed("output_registration_opening_cancelled")
                except Exception as error:
                    self._registration_failed(error)
                    errors.append(error)
                else:
                    outcome = None
                    cancelled = await self._write(extent, errors)
                    if extent is not None and extent.failure is not None:
                        # The registration refused inside the writer (mapped there to a fixed code).
                        self._registration_failed(extent.failure)
        self._actor = None
        # Exactly one terminal per exit; a failed epoch appends nothing.
        drained = asyncio.Task(output_registration.configuration_drained(
            self, extent, outcome, graph_failed=exc_type is not None))
        cancelled = await self._settle(drained) or cancelled
        try:
            drained.result()
        except asyncio.CancelledError:
            cancelled = True
        except Exception as error:
            self._registration_failed(error)
            errors.append(error)
        if len(errors) > 1:
            raise BaseExceptionGroup("configuration write and output registration",
                errors + ([asyncio.CancelledError()] if cancelled else []))
        if errors and cancelled:
            raise BaseExceptionGroup("configuration write and observer cancellation",
                [errors[0], asyncio.CancelledError()])
        if errors:
            raise errors[0]
        if cancelled:
            raise asyncio.CancelledError
        return False
