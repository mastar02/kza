"""The SQLite audit switch, readable without importing any policy package.

U3b-0e (N1; independent review F5/F6). ``hooks.audit_sqlite.enabled`` is the
audit's own switch. Only the boolean ``true`` enables it. Under an admission
ticket R1 refuses it (P2=A: the audit store has no erasure participant and
no admission contract), as it refuses ``KZA_AUDIT_DB``.
"""
from collections.abc import Mapping
import os

AUDIT_ENV = "KZA_AUDIT_DB"
UNSUPPORTED_CODE = "runtime_admission_audit_unsupported"
ENV_CODE = "runtime_admission_audit_env_refused"
TICKET_ENV = "KZA_RUNTIME_ADMISSION"  # same literal as runtime_admission.TICKET_ENV (a test pins it)


def audit_enabled(hooks_cfg: Mapping | None) -> bool:
    """True only for ``hooks.audit_sqlite.enabled: true`` (a real boolean)."""
    section = hooks_cfg.get("audit_sqlite") if isinstance(hooks_cfg, Mapping) else None
    return isinstance(section, Mapping) and section.get("enabled") is True


def refuse_managed_audit(config: Mapping | None, environ: Mapping[str, str] | None = None) -> None:
    """Under a ticket, refuse KZA_AUDIT_DB and an enabled audit, whatever hooks.enabled says.

    Raises:
        RuntimeError: ``runtime_admission_audit_env_refused`` or
            ``runtime_admission_audit_unsupported``.
    """
    environ = os.environ if environ is None else environ
    if TICKET_ENV not in environ:
        return
    if AUDIT_ENV in environ:
        raise RuntimeError(ENV_CODE)
    hooks = config.get("hooks") if isinstance(config, Mapping) else None
    if audit_enabled(hooks):
        raise RuntimeError(UNSUPPORTED_CODE)
