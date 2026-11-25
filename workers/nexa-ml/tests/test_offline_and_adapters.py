"""Offline enforcement and real-adapter wiring, exercised without torch or model libraries."""

import os
import pathlib
import sys
import types
from typing import Any

import pytest

from nexa_ml.__main__ import apply_offline_env
from nexa_ml.config import MlConfig
from nexa_ml.offline import OfflineViolationError, assert_offline_effective, reject_remote_source
from nexa_ml.runtime import real_loaders
from nexa_ml.slot import InvalidInputError
from nexa_ml.speaker_slot import resolve_ecapa_source
from nexa_ml.tts_slot import KokoroEngine, supported_kwargs


@pytest.fixture
def online_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing flags that explicitly allow network (must be overridden)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------- env forcing
@pytest.mark.usefixtures("online_env")
def test_entrypoint_overrides_preexisting_online_flags() -> None:
    apply_offline_env(MlConfig(offline=True))
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.usefixtures("online_env")
def test_real_loaders_override_preexisting_online_flags() -> None:
    real_loaders(MlConfig(offline=True))  # builds factories only; no heavy import
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.usefixtures("online_env")
def test_offline_false_leaves_env_untouched() -> None:
    real_loaders(MlConfig(offline=False))
    assert os.environ["HF_HUB_OFFLINE"] == "0"


@pytest.mark.usefixtures("online_env")
def test_assert_offline_effective_fails_closed_on_online_flags() -> None:
    with pytest.raises(OfflineViolationError):
        assert_offline_effective()


@pytest.mark.usefixtures("offline_env")
def test_assert_offline_detects_hub_imported_online(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = types.ModuleType("huggingface_hub.constants")
    stale.HF_HUB_OFFLINE = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", stale)
    with pytest.raises(OfflineViolationError):
        assert_offline_effective()
    stale.HF_HUB_OFFLINE = True
    assert_offline_effective()


@pytest.mark.parametrize("source", ["https://hf.co/x", "http://10.0.0.1/m", "s3://bucket/ecapa", "hf://a/b"])
def test_remote_sources_rejected(source: str) -> None:
    with pytest.raises(OfflineViolationError):
        reject_remote_source(source)


def test_hub_ids_and_paths_are_not_uris() -> None:
    reject_remote_source("speechbrain/spkrec-ecapa-voxceleb")
    reject_remote_source("/var/lib/nexa/models/ecapa")


# ---------------------------------------------------------------- ECAPA (SpeechBrain)
@pytest.mark.usefixtures("offline_env")
def test_ecapa_offline_rejects_uri_source(tmp_path: pathlib.Path) -> None:
    with pytest.raises(OfflineViolationError):
        resolve_ecapa_source("https://example.org/ecapa", str(tmp_path), offline=True)


@pytest.mark.usefixtures("online_env")
def test_ecapa_offline_requires_forced_flags(tmp_path: pathlib.Path) -> None:
    with pytest.raises(OfflineViolationError):
        resolve_ecapa_source("speechbrain/spkrec-ecapa-voxceleb", str(tmp_path), offline=True)


@pytest.mark.usefixtures("offline_env")
def test_ecapa_offline_fails_closed_without_local_cache(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_ecapa_source("speechbrain/spkrec-ecapa-voxceleb", str(tmp_path / "cache"), offline=True)


@pytest.mark.usefixtures("offline_env")
def test_ecapa_offline_uses_local_savedir_or_source_dir(tmp_path: pathlib.Path) -> None:
    savedir = tmp_path / "cache"
    savedir.mkdir()
    (savedir / "hyperparams.yaml").write_text("x")
    (savedir / "embedding_model.ckpt").write_bytes(b"x")
    assert resolve_ecapa_source("speechbrain/spkrec-ecapa-voxceleb", str(savedir), offline=True) == str(
        savedir
    )
    assert resolve_ecapa_source(str(savedir), str(tmp_path / "other"), offline=True) == str(savedir)


def test_ecapa_online_passthrough(tmp_path: pathlib.Path) -> None:
    assert resolve_ecapa_source("speechbrain/x", str(tmp_path), offline=False) == "speechbrain/x"


# ---------------------------------------------------------------- Kokoro constructor signatures
class _Module:
    def to(self, _device: str) -> "_Module":
        return self

    def eval(self) -> "_Module":
        return self


def _kokoro_07() -> Any:
    """Constructor signatures of kokoro 0.7.16 (no ``repo_id``)."""
    calls: dict[str, Any] = {}

    class KModel(_Module):
        REPO_ID = "hexgrad/Kokoro-82M"

        def __init__(self, config: Any = None, model: str | None = None) -> None:
            calls["kmodel"] = {"config": config, "model": model}

    class KPipeline:
        def __init__(
            self, lang_code: str, model: Any = True, trf: bool = False, device: str | None = None
        ) -> None:
            calls.setdefault("kpipeline", []).append(
                {"lang_code": lang_code, "model": model, "device": device}
            )
            self.loaded: list[str] = []

        def load_voice(self, voice: str) -> None:
            if voice.startswith("zz_"):
                raise FileNotFoundError("offline cache miss")
            self.loaded.append(voice)

    return types.SimpleNamespace(KModel=KModel, KPipeline=KPipeline, calls=calls)


def _kokoro_09() -> Any:
    """Constructor signatures of kokoro 0.9.4 (``repo_id`` on both)."""
    calls: dict[str, Any] = {}

    class KModel(_Module):
        def __init__(
            self,
            repo_id: str | None = None,
            config: Any = None,
            model: str | None = None,
            disable_complex: bool = False,
        ) -> None:
            calls["kmodel"] = {"repo_id": repo_id, "config": config, "model": model}

    class KPipeline:
        def __init__(
            self,
            lang_code: str,
            repo_id: str | None = None,
            model: Any = True,
            trf: bool = False,
            en_callable: Any = None,
            device: str | None = None,
        ) -> None:
            calls.setdefault("kpipeline", []).append(
                {"lang_code": lang_code, "repo_id": repo_id, "model": model, "device": device}
            )

        def load_voice(self, voice: str) -> None:
            return None

    return types.SimpleNamespace(KModel=KModel, KPipeline=KPipeline, calls=calls)


@pytest.mark.usefixtures("offline_env")
def test_kokoro_07_signature_never_receives_repo_id() -> None:
    mod = _kokoro_07()
    engine = KokoroEngine("hexgrad/Kokoro-82M", "cuda:1", "e", kokoro_module=mod)
    assert mod.calls["kmodel"] == {"config": None, "model": None}
    assert mod.calls["kpipeline"][0]["lang_code"] == "e"
    assert "repo_id" not in mod.calls["kpipeline"][0] and mod.calls["kpipeline"][0]["device"] == "cuda:1"
    pipe = engine.pipeline("e")
    assert engine.prepare_voice(pipe, "ef_dora") == "ef_dora"
    with pytest.raises(InvalidInputError):
        engine.prepare_voice(pipe, "zz_missing")


@pytest.mark.usefixtures("offline_env")
def test_kokoro_07_cannot_honour_custom_repo_without_local_layout() -> None:
    with pytest.raises(ValueError):
        KokoroEngine("someone/other-kokoro", "cpu", "e", kokoro_module=_kokoro_07())


@pytest.mark.usefixtures("offline_env")
def test_kokoro_09_signature_receives_repo_id() -> None:
    mod = _kokoro_09()
    KokoroEngine("hexgrad/Kokoro-82M", "cuda:1", "e", kokoro_module=mod)
    assert mod.calls["kmodel"]["repo_id"] == "hexgrad/Kokoro-82M"
    assert mod.calls["kpipeline"][0]["repo_id"] == "hexgrad/Kokoro-82M"


@pytest.mark.usefixtures("offline_env")
@pytest.mark.parametrize("factory", [_kokoro_07, _kokoro_09], ids=["0.7", "0.9"])
def test_kokoro_local_layout_uses_local_files(tmp_path: pathlib.Path, factory: Any) -> None:
    root = tmp_path / "kokoro"
    (root / "voices").mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "kokoro-v1_0.pth").write_bytes(b"w")
    (root / "voices" / "ef_dora.pt").write_bytes(b"v")
    mod = factory()
    engine = KokoroEngine("hexgrad/Kokoro-82M", "cpu", "e", models_dir=str(tmp_path), kokoro_module=mod)
    assert mod.calls["kmodel"]["config"] == str(root / "config.json")
    assert mod.calls["kmodel"]["model"] == str(root / "kokoro-v1_0.pth")
    assert mod.calls["kmodel"].get("repo_id") is None
    assert mod.calls["kpipeline"][0].get("repo_id") is None
    assert engine.prepare_voice(engine.pipeline("e"), "ef_dora") == str(root / "voices" / "ef_dora.pt")


@pytest.mark.usefixtures("offline_env")
def test_kokoro_offline_rejects_remote_repo() -> None:
    with pytest.raises(OfflineViolationError):
        KokoroEngine("https://example.org/kokoro", "cpu", "e", kokoro_module=_kokoro_09())


@pytest.mark.usefixtures("online_env")
def test_kokoro_offline_fails_closed_when_flags_not_forced() -> None:
    with pytest.raises(OfflineViolationError):
        KokoroEngine("hexgrad/Kokoro-82M", "cpu", "e", kokoro_module=_kokoro_09())


def test_supported_kwargs_filters_by_signature() -> None:
    def narrow(a: int, b: int = 0) -> None: ...

    def wide(a: int, **kw: Any) -> None: ...

    assert supported_kwargs(narrow, {"a": 1, "b": None, "c": 3}) == {"a": 1}
    assert supported_kwargs(wide, {"a": 1, "c": 3}) == {"a": 1, "c": 3}


# ---------------------------------------------------------------- Kokoro English G2P (spaCy)
@pytest.mark.usefixtures("offline_env")
@pytest.mark.parametrize("lang", ["a", "b"])
def test_kokoro_english_offline_without_spacy_model_fails_closed(
    monkeypatch: pytest.MonkeyPatch, lang: str
) -> None:
    downloads: list[str] = []
    fake_spacy = types.ModuleType("spacy")
    fake_spacy.cli = types.SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    mod = _kokoro_09()
    probes: list[str] = []

    def missing(name: str) -> bool:
        probes.append(name)
        return False

    with pytest.raises(FileNotFoundError):
        KokoroEngine("hexgrad/Kokoro-82M", "cpu", lang, kokoro_module=mod, spacy_installed=missing)
    assert probes == ["en_core_web_sm"]
    assert "kpipeline" not in mod.calls and "kmodel" not in mod.calls  # nothing constructed
    assert downloads == []  # spacy.cli.download never invoked


@pytest.mark.usefixtures("offline_env")
def test_kokoro_english_pipeline_on_demand_also_guarded() -> None:
    mod = _kokoro_09()
    engine = KokoroEngine(
        "hexgrad/Kokoro-82M", "cpu", "e", kokoro_module=mod, spacy_installed=lambda _n: False
    )
    with pytest.raises(FileNotFoundError):
        engine.pipeline("a")
    assert [c["lang_code"] for c in mod.calls["kpipeline"]] == ["e"]


@pytest.mark.usefixtures("offline_env")
def test_kokoro_english_with_local_spacy_model_builds_pipeline() -> None:
    mod = _kokoro_09()
    KokoroEngine(
        "hexgrad/Kokoro-82M", "cpu", "a", kokoro_module=mod, spacy_installed=lambda n: n == "en_core_web_sm"
    )
    assert mod.calls["kpipeline"][0]["lang_code"] == "a"


@pytest.mark.usefixtures("online_env")
def test_kokoro_online_mode_skips_spacy_guard() -> None:
    mod = _kokoro_09()
    KokoroEngine(
        "hexgrad/Kokoro-82M", "cpu", "a", offline=False, kokoro_module=mod, spacy_installed=lambda _n: False
    )
    assert mod.calls["kpipeline"][0]["lang_code"] == "a"


def test_spacy_probe_missing_module_is_false() -> None:
    from nexa_ml.tts_slot import spacy_model_installed

    assert spacy_model_installed("nexa_ml_definitely_missing_model") is False


def _importable_model(monkeypatch: pytest.MonkeyPatch, *, spacy_available: bool) -> None:
    """Pretend ``en_core_web_sm`` is importable (and optionally that spaCy is installed)."""
    import importlib.machinery
    import importlib.util

    real_find_spec = importlib.util.find_spec
    present = {"en_core_web_sm"} | ({"spacy"} if spacy_available else set())

    def find_spec(name: str, *args: Any) -> Any:
        if name in present:
            return importlib.machinery.ModuleSpec(name, None)
        if name == "spacy":
            return None
        return real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


@pytest.mark.usefixtures("offline_env")
def test_importable_model_without_distribution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Misaki downloads when metadata is missing even if the module imports: fail before KPipeline."""
    import importlib.metadata

    from nexa_ml.tts_slot import spacy_model_installed

    _importable_model(monkeypatch, spacy_available=False)

    def no_dist(name: str) -> Any:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", no_dist)
    assert spacy_model_installed("en_core_web_sm") is False
    mod = _kokoro_09()
    with pytest.raises(FileNotFoundError):
        KokoroEngine("hexgrad/Kokoro-82M", "cpu", "a", kokoro_module=mod)  # default real probe
    assert "kpipeline" not in mod.calls and "kmodel" not in mod.calls

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: object())
    assert spacy_model_installed("en_core_web_sm") is True


@pytest.mark.usefixtures("offline_env")
def test_spacy_is_package_false_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When spaCy is installed its own ``is_package`` (Misaki's criterion) decides."""
    from nexa_ml.tts_slot import spacy_model_installed

    _importable_model(monkeypatch, spacy_available=True)
    verdict = {"value": False}
    asked: list[str] = []
    fake_util = types.ModuleType("spacy.util")
    fake_util.is_package = lambda name: asked.append(name) or verdict["value"]
    fake_spacy = types.ModuleType("spacy")
    fake_spacy.util = fake_util
    fake_spacy.cli = types.SimpleNamespace(download=lambda name: pytest.fail("download must never run"))
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setitem(sys.modules, "spacy.util", fake_util)

    mod = _kokoro_09()
    with pytest.raises(FileNotFoundError):
        KokoroEngine("hexgrad/Kokoro-82M", "cpu", "b", kokoro_module=mod)
    assert asked == ["en_core_web_sm"] and "kpipeline" not in mod.calls
    verdict["value"] = True
    assert spacy_model_installed("en_core_web_sm") is True


@pytest.mark.usefixtures("offline_env")
@pytest.mark.parametrize("variant", ["A", " a ", "B"])
def test_uppercase_or_padded_english_does_not_bypass_guard(variant: str) -> None:
    mod = _kokoro_09()
    with pytest.raises(FileNotFoundError):
        KokoroEngine(
            "hexgrad/Kokoro-82M", "cpu", variant, kokoro_module=mod, spacy_installed=lambda _n: False
        )
    assert "kpipeline" not in mod.calls
    engine = KokoroEngine(
        "hexgrad/Kokoro-82M", "cpu", "e", kokoro_module=mod, spacy_installed=lambda _n: False
    )
    with pytest.raises(FileNotFoundError):
        engine.pipeline(variant)
    assert [c["lang_code"] for c in mod.calls["kpipeline"]] == ["e"]


@pytest.mark.usefixtures("offline_env")
@pytest.mark.parametrize("alias", ["en-us", "american", "en", "es", ""])
def test_non_canonical_lang_alias_rejected_before_pipeline(alias: str) -> None:
    mod = _kokoro_09()
    with pytest.raises(ValueError):
        KokoroEngine("hexgrad/Kokoro-82M", "cpu", alias, kokoro_module=mod, spacy_installed=lambda _n: True)
    assert "kpipeline" not in mod.calls and "kmodel" not in mod.calls


def test_config_tts_lang_code_canonicalised_or_rejected() -> None:
    from nexa_ml.config import ConfigError, parse_ml_section

    assert parse_ml_section({"ml": {"tts_lang_code": " A "}}).tts_lang_code == "a"
    for alias in ("en-us", "american", "spanish"):
        with pytest.raises(ConfigError):
            parse_ml_section({"ml": {"tts_lang_code": alias}})


# ---------------------------------------------------------------- SpeechBrain fetch config / manifest
def _fake_fetching() -> Any:
    import dataclasses
    import enum

    @dataclasses.dataclass(frozen=True)
    class FetchConfig:
        overwrite: bool = False
        allow_updates: bool = False
        allow_network: bool = True

    class LocalStrategy(enum.Enum):
        SYMLINK = 1
        COPY = 2
        COPY_SKIP_CACHE = 3
        NO_LINK = 4

    return types.SimpleNamespace(FetchConfig=FetchConfig, LocalStrategy=LocalStrategy)


def _ecapa_dir(
    root: pathlib.Path, manifest: str = "pretrained_path: speechbrain/spkrec-ecapa-voxceleb\n"
) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "hyperparams.yaml").write_text(manifest)
    (root / "embedding_model.ckpt").write_bytes(b"x")
    return root


class _FakeClassifier:
    calls: list[dict[str, Any]] = []

    @classmethod
    def from_hparams(cls, **kwargs: Any) -> "_FakeClassifier":
        cls.calls.append(kwargs)
        return cls()


@pytest.mark.usefixtures("offline_env")
def test_ecapa_from_hparams_gets_no_network_fetch_config(tmp_path: pathlib.Path) -> None:
    from nexa_ml.speaker_slot import EcapaEncoder

    local = _ecapa_dir(tmp_path / "cache")
    _FakeClassifier.calls = []
    api = types.SimpleNamespace(EncoderClassifier=_FakeClassifier, fetching=_fake_fetching())
    EcapaEncoder("speechbrain/spkrec-ecapa-voxceleb", "cuda:0", str(local), speechbrain_api=api)
    (kwargs,) = _FakeClassifier.calls
    assert kwargs["fetch_config"].allow_network is False
    assert kwargs["local_strategy"].name == "NO_LINK"
    assert kwargs["overrides"] == {"pretrained_path": str(local)}
    assert kwargs["source"] == str(local) and kwargs["run_opts"] == {"device": "cuda:0"}


@pytest.mark.usefixtures("offline_env")
@pytest.mark.parametrize(
    "manifest",
    [
        "pretrained_path: https://evil.example/ecapa\n",
        "paths:\n  embedding_model: http://10.0.0.9/embedding_model.ckpt\n",
        "pretrained_path: s3://bucket/ecapa\n",
    ],
)
def test_ecapa_remote_uri_in_manifest_rejected_before_from_hparams(
    tmp_path: pathlib.Path, manifest: str
) -> None:
    from nexa_ml.speaker_slot import EcapaEncoder

    local = _ecapa_dir(tmp_path / "cache", manifest)
    _FakeClassifier.calls = []
    api = types.SimpleNamespace(EncoderClassifier=_FakeClassifier, fetching=_fake_fetching())
    with pytest.raises(OfflineViolationError):
        EcapaEncoder("speechbrain/spkrec-ecapa-voxceleb", "cuda:0", str(local), speechbrain_api=api)
    assert _FakeClassifier.calls == []


@pytest.mark.usefixtures("offline_env")
def test_ecapa_fails_closed_without_fetch_config_support(tmp_path: pathlib.Path) -> None:
    from nexa_ml.speaker_slot import EcapaEncoder

    local = _ecapa_dir(tmp_path / "cache")
    _FakeClassifier.calls = []
    api = types.SimpleNamespace(EncoderClassifier=_FakeClassifier, fetching=types.SimpleNamespace())
    with pytest.raises(OfflineViolationError):
        EcapaEncoder("speechbrain/spkrec-ecapa-voxceleb", "cuda:0", str(local), speechbrain_api=api)
    assert _FakeClassifier.calls == []


def test_ecapa_online_mode_passes_plain_kwargs(tmp_path: pathlib.Path) -> None:
    from nexa_ml.speaker_slot import from_hparams_kwargs

    kwargs = from_hparams_kwargs(
        "speechbrain/x", str(tmp_path), "cpu", offline=False, fetching=_fake_fetching()
    )
    assert set(kwargs) == {"source", "savedir", "run_opts"}
