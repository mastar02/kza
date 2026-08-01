"""Guard against sys.modules pollution breaking collection for other tests."""
import importlib.util
import sys


def test_torch_module_has_valid_spec():
    """Any torch in sys.modules must be introspectable by importlib.

    transformers calls importlib.util.find_spec("torch"), which raises
    ValueError if torch.__spec__ is unset. A bare MagicMock has no __spec__,
    so a module-level sys.modules patch breaks collection for every test
    file collected afterwards.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return  # torch not loaded in this run: nothing to guard
    assert getattr(torch, "__spec__", None) is not None, (
        "torch.__spec__ is unset — something replaced sys.modules['torch'] "
        "with a mock that importlib cannot introspect"
    )
    assert importlib.util.find_spec("torch") is not None
