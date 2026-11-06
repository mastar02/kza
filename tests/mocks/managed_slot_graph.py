"""Use canonical persisted settings and real admission for compiled slot graphs."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def _variable(path):
    encoded = json.dumps(path, separators=(",", ":")).encode()
    return "KZA_SLOT_BIND_" + hashlib.sha256(encoded).hexdigest().upper()


def _leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from _leaves(child, (*path, key))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, index))
    else:
        yield path, value


def _get(value, path):
    for key in path:
        value = value[key]
    return value


def canonical_template(root, configuration):
    """Keep portable raw YAML while binding each runtime's actual owned paths."""
    root = Path(root).resolve()
    home, model = "http://slot-home.invalid", "http://slot-model.invalid"
    config = configuration(root, home, model)
    # The direct offline factory omitted these default-only sections. A real
    # persisted settings document must also satisfy main.load_config's schema.
    for section in ("audio", "wake_word", "stt"):
        config.setdefault(section, {})
    for path, value in list(_leaves(config)):
        if type(value) is str and (
                value.startswith(str(root) + os.sep) or value == str(root)
                or value.startswith(home) or value.startswith(model)):
            owner = _get(config, path[:-1])
            owner[path[-1]] = "${" + _variable(path) + "}"
    return config


def load_canonical_configuration(root, home_url, model_url):
    """Resolve ephemeral loopback leaves and owned paths, then use main's loader."""
    import yaml
    from src.main import load_config
    from tests.mocks.restore_application_graph import configuration

    root = Path(root).resolve()
    path = root.parents[1] / "config/settings.yaml"
    raw = yaml.safe_load(path.read_text())
    generated = configuration(root, home_url, model_url)
    for location, value in _leaves(raw):
        if type(value) is str and value.startswith("${KZA_SLOT_BIND_"):
            variable = _variable(location)
            if value != "${" + variable + "}":
                raise ValueError("slot_configuration_binding_unknown")
            resolved = _get(generated, location)
            if type(resolved) is not str:
                raise ValueError("slot_configuration_binding_not_text")
            if not (resolved.startswith(str(root) + os.sep) or resolved == str(root)
                    or resolved.startswith(home_url) or resolved.startswith(model_url)):
                raise ValueError("slot_configuration_binding_not_owned")
            os.environ[variable] = resolved
    return load_config(str(path))


def _bridge(control):
    directory = Path(control["island_root"]) / "bridge"
    release_path = directory / "release.json"
    release = json.loads(release_path.read_text())
    if release.get("format") != "kza-cpython-bytecode-v1":
        raise ValueError("slot_bridge_release_invalid")
    for relative in ("src/core/runtime_admission.pyc", "src/memory/runtime_closure.pyc",
                     "scripts/stage_runtime.pyc"):
        path = directory / relative
        expected = release["files"][relative]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("slot_bridge_artifact_changed")
        if control["island_files"].get("bridge/" + relative) != expected:
            raise ValueError("slot_bridge_not_hashlisted")
    name = "_kza_slot_admission_bridge"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, directory / "src/core/runtime_admission.pyc")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def managed_application_graph(root, patch, clock, *, control, **kwargs):
    """Run the real older factory with production admission and owned close."""
    import src.main as main
    from tests.mocks.restore_application_graph import application_graph

    admission = _bridge(control)

    async def build(config, *, shutdown, factories):
        if hasattr(main, "verify_runtime_admission"):
            return await main.build_application(config, shutdown=shutdown, factories=factories)
        return await admission.build_compatible_application(main.build_application, config,
            shutdown=shutdown, factories=factories, runtime_root=Path(control["runtime_root"]))

    async with application_graph(root, patch, clock,
            configuration_factory=load_canonical_configuration,
            application_builder=build, **kwargs) as bundle:
        yield bundle
