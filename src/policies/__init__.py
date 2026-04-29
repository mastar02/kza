"""KZA plugin policies — handlers registered against the global hook registry.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md

Importing this package triggers each policy module's decorator side-effects,
populating src.hooks.registry._global_registry. Add new policies as new
modules in this directory and import them here so they get loaded.
"""

# Order matters for priority documentation but not execution
# (priorities live on the decorators themselves).
from src.policies import (  # noqa: F401  — side-effect imports
    safety_alarm,
    permissions,
    tts_rewrite_es,
    audit_sqlite,
)
