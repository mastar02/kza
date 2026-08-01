"""Guard against sys.modules pollution breaking collection for other tests."""
import importlib.util
import sys

import pytest

# torch AND soundfile both bit us (see task-1-report.md): transformers calls
# importlib.util.find_spec() on each of them — is_soundfile_available() does
# it for soundfile the same way modeling code does it for torch — and both
# raise ValueError if the module's __spec__ is unset. A bare MagicMock has no
# __spec__, so a module-level `sys.modules['torch'/'soundfile'] = MagicMock()`
# (found unconditionally, without .setdefault, in test_model_manager.py)
# broke collection for every test file collected afterwards. If a new file
# reintroduces that pattern for either module, this guard should catch it
# before it turns into another confusing "Interrupted: N errors during
# collection" instead of a clear assertion failure.
_GUARDED_MODULES = ("torch", "soundfile")


@pytest.mark.parametrize("module_name", _GUARDED_MODULES)
def test_module_has_valid_spec(module_name):
    """Any of _GUARDED_MODULES present in sys.modules must be introspectable.

    See module docstring for why this matters. If the module isn't loaded in
    this run at all, there's nothing to guard — that's a deliberate early
    return, not dead code (e.g. a subprocess/CI run that never imports torch).
    """
    module = sys.modules.get(module_name)
    if module is None:
        return  # not loaded in this run: nothing to guard
    assert getattr(module, "__spec__", None) is not None, (
        f"{module_name}.__spec__ is unset — something replaced "
        f"sys.modules[{module_name!r}] with a mock that importlib cannot "
        "introspect"
    )
    assert importlib.util.find_spec(module_name) is not None
