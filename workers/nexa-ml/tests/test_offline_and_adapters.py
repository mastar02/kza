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
