"""paths.models_root, models_path and speaker_id.savedir (U3b-0e 2b, F2/N10)."""

from pathlib import Path

import pytest
import yaml

from src.core import paths
from src.core.runtime_admission import TICKET_ENV

TICKET = {TICKET_ENV: "{}"}
SETTINGS = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def test_repository_settings_document_the_new_keys():
    config = yaml.safe_load(SETTINGS.read_text())
    assert config["paths"]["models_root"] == "./models"
    assert config["speaker_id"]["savedir"] == ""
    assert config["hooks"]["audit_sqlite"] == {"enabled": False, "db_path": "./data/audit.db"}
    assert "audit_sqlite_path" not in config["hooks"]
    text = SETTINGS.read_text()
    assert "paths.models_root" in text and "U3b-0e" in text


def test_development_fallback_without_a_ticket():
    assert paths.models_path({}, "wakeword", environ={}) == "./models/wakeword"
    assert paths.models_path(None, "piper", "v.onnx", environ={}) == "./models/piper/v.onnx"
    assert paths.models_root({"paths": {}}, environ={}) == "./models"
    assert paths.models_path({"paths": {"models_root": "/m"}}, "speaker", environ={}) == "/m/speaker"
    assert paths.models_path({"paths": {"models_root": "rel"}}, "x", environ={}) == "rel/x"


@pytest.mark.parametrize("config", [None, {}, {"paths": None}, {"paths": {}}, {"paths": {"models_root": ""}},
                                    {"paths": {"models_root": None}}])
def test_missing_root_under_a_ticket_refuses(config):
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        paths.models_path(config, "wakeword", environ=TICKET)
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        paths.models_root(config, environ=TICKET)


def test_relative_root_under_a_ticket_refuses():
    for root in ("./models", "models", "../models"):
        with pytest.raises(RuntimeError, match="runtime_admission_models_root_relative"):
            paths.models_path({"paths": {"models_root": root}}, "x", environ=TICKET)
    assert paths.models_path({"paths": {"models_root": "/home/kza/kza/models"}}, "wakeword", "a.ppn",
                             environ=TICKET) == "/home/kza/kza/models/wakeword/a.ppn"


def test_ticket_is_read_from_the_process_environment(monkeypatch):
    monkeypatch.setenv(TICKET_ENV, "{}")
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        paths.models_path({}, "x")
    monkeypatch.delenv(TICKET_ENV)
    assert paths.models_path({}, "x") == "./models/x"


@pytest.mark.parametrize("part", ["", "/abs", "../up", "a/../b"])
def test_parts_must_stay_under_the_root(part):
    with pytest.raises(ValueError, match="models_path_part_invalid"):
        paths.models_path({"paths": {"models_root": "/m"}}, part, environ={})


def test_speaker_savedir_rules():
    name = "speechbrain/spkrec-ecapa-voxceleb"
    leaf = "speechbrain_spkrec-ecapa-voxceleb"
    assert paths.speaker_savedir({}, name, environ={}) == "./models/speaker/" + leaf
    assert paths.speaker_savedir({"paths": {"models_root": "/m"}, "speaker_id": {"savedir": ""}}, name,
                                 environ=TICKET) == "/m/speaker/" + leaf
    assert paths.speaker_savedir({"speaker_id": {"savedir": "/home/kza/kza/models/speaker"}}, name,
                                 environ=TICKET) == "/home/kza/kza/models/speaker/" + leaf
    with pytest.raises(RuntimeError, match="runtime_admission_speaker_savedir_relative"):
        paths.speaker_savedir({"speaker_id": {"savedir": "./models/speaker"}}, name, environ=TICKET)
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        paths.speaker_savedir({"speaker_id": {}}, name, environ=TICKET)


def test_training_output_defaults_are_state_relative():
    assert paths.TRAINING_OUTPUT_LORA == "./data/training_outputs/lora"
    assert paths.TRAINING_OUTPUT_NIGHTLY == "./data/training_outputs/nightly"
    assert paths.TRAINING_OUTPUT_WHISPER == "./data/training_outputs/whisper"


# --- every 2b site goes through the helper ---------------------------------------------------------

def test_speaker_identifier_uses_the_configured_savedir(monkeypatch):
    import sys
    import types
    from src.users.speaker_identifier import SpeakerIdentifier
    calls = []

    class Encoder:
        @staticmethod
        def from_hparams(**kwargs):
            calls.append(kwargs)
            return object()

    module = types.ModuleType("speechbrain.inference.speaker")
    module.EncoderClassifier = Encoder
    monkeypatch.setitem(sys.modules, "speechbrain", types.ModuleType("speechbrain"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference", types.ModuleType("speechbrain.inference"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", module)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    SpeakerIdentifier("speechbrain/x", "cpu", savedir="/abs/speaker/speechbrain_x")._load_speechbrain()
    monkeypatch.delenv(TICKET_ENV, raising=False)
    SpeakerIdentifier("speechbrain/x", "cpu")._load_speechbrain()
    assert [call["savedir"] for call in calls] == ["/abs/speaker/speechbrain_x", "./models/speaker/speechbrain_x"]
    monkeypatch.setenv(TICKET_ENV, "{}")
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        SpeakerIdentifier("speechbrain/x", "cpu")._load_speechbrain()


def test_model_manager_passes_an_explicit_speaker_savedir(monkeypatch):
    """N10: model_manager.py's EncoderClassifier.from_hparams gets a savedir."""
    import asyncio
    import sys
    import types
    from src.pipeline.model_manager import ModelManager, ModelManagerConfig
    calls = []

    class Encoder:
        @staticmethod
        def from_hparams(**kwargs):
            calls.append(kwargs)
            return object()

    module = types.ModuleType("speechbrain.inference.speaker")
    module.EncoderClassifier = Encoder
    monkeypatch.setitem(sys.modules, "speechbrain", types.ModuleType("speechbrain"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference", types.ModuleType("speechbrain.inference"))
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", module)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    monkeypatch.delenv(TICKET_ENV, raising=False)
    asyncio.run(ModelManager(ModelManagerConfig(models_root="/m"))._load_speaker_id())
    asyncio.run(ModelManager(ModelManagerConfig(speaker_savedir="/s"))._load_speaker_id())
    leaf = "speechbrain_spkrec-ecapa-voxceleb"
    assert [call["savedir"] for call in calls] == ["/m/speaker/" + leaf, "/s/" + leaf]


def test_model_manager_piper_path_uses_the_root(monkeypatch):
    import asyncio
    import sys
    import types
    from src.pipeline.model_manager import ModelManager, ModelManagerConfig
    loads = []
    piper = types.ModuleType("piper")
    piper.PiperVoice = types.SimpleNamespace(load=lambda path: loads.append(path))
    monkeypatch.setitem(sys.modules, "piper", piper)
    asyncio.run(ModelManager(ModelManagerConfig(models_root="/m", tts_voice="v"))._load_tts())
    assert loads == ["/m/piper/v.onnx"]


def test_wakeword_defaults_use_the_root(monkeypatch, tmp_path):
    from src.wakeword.detector import WakeWordDetector
    from src.wakeword.trainer import WakeWordTrainer
    monkeypatch.delenv(TICKET_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert WakeWordDetector().custom_models_dir == Path("./models/wakeword")
    assert WakeWordDetector(custom_models_dir="/m/wakeword").custom_models_dir == Path("/m/wakeword")
    assert WakeWordTrainer(models_dir=str(tmp_path / "w")).models_dir == tmp_path / "w"
    monkeypatch.setenv(TICKET_ENV, "{}")
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        WakeWordDetector()
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        WakeWordTrainer()


def test_ambient_whisper_default_uses_the_root(monkeypatch):
    import src.ambient.transcriber as transcriber
    import src.stt.whisper_fast as whisper_fast
    seen = []
    monkeypatch.setattr(whisper_fast, "FastWhisperSTT", lambda **kwargs: seen.append(kwargs["model"]))
    transcriber._build_ambient_stt_engine({}, {}, {"paths": {"models_root": "/m"}})
    transcriber._build_ambient_stt_engine({}, {"model": "/x"}, {})
    transcriber._build_ambient_stt_engine({"model": "/y"}, {"model": "/x"}, {})
    assert seen == ["/m/whisper-v3-turbo", "/x", "/y"]


def test_health_check_models_use_the_root(tmp_path):
    from src.health.health_check import HealthChecker
    (tmp_path / "a.gguf").write_bytes(b"")
    found = HealthChecker({"paths": {"models_root": str(tmp_path)}}).check_models()
    assert found.details == {"models": ["a.gguf"]}
    missing = HealthChecker({"paths": {"models_root": str(tmp_path / "none")}}).check_models()
    assert str(tmp_path / "none") in missing.message


def test_training_defaults_are_the_relocated_roots():
    import inspect
    from src.training.conversation_collector import LoRATrainer
    from src.training.nightly_trainer import NightlyConfig
    assert NightlyConfig().output_dir == "./data/training_outputs/nightly"
    assert inspect.signature(LoRATrainer).parameters["output_dir"].default == "./data/training_outputs/lora"


# --- review F3: tts.piper.model ---------------------------------------------------------------------

def test_model_file_puts_a_bare_name_under_the_models_root():
    config = {"paths": {"models_root": "/m"}}
    assert paths.model_file(config, "v.onnx", "piper", environ={}) == "/m/piper/v.onnx"
    assert paths.model_file(config, "v.onnx", "piper", environ=TICKET) == "/m/piper/v.onnx"
    assert paths.model_file(config, "/abs/v.onnx", "piper", environ=TICKET) == "/abs/v.onnx"
    assert paths.model_file(config, "dir/v.onnx", "piper", environ={}) == "dir/v.onnx"  # preflight judges it
    for value in ("", ".", ".."):
        assert paths.model_file(config, value, "piper", environ={}) == value
    with pytest.raises(RuntimeError, match="runtime_admission_models_root_missing"):
        paths.model_file({}, "v.onnx", "piper", environ=TICKET)


def test_tts_settings_resolve_the_piper_model_only_for_piper_engines():
    config = {"paths": {"models_root": "/m"}, "tts": {"engine": "piper", "piper": {"model": "v.onnx"}}}
    resolved = paths.tts_settings(config, environ=TICKET)
    assert resolved["piper"]["model"] == "/m/piper/v.onnx"
    assert config["tts"]["piper"]["model"] == "v.onnx"  # the caller's mapping is not mutated
    hybrid = paths.tts_settings({"paths": {"models_root": "/m"}, "tts": {"engine": "hybrid"}}, environ=TICKET)
    assert hybrid["piper"]["model"] == f"/m/piper/{paths.PIPER_DEFAULT_MODEL}"
    assert paths.tts_settings({"paths": {"models_root": "/m"}}, environ={})["piper"]["model"] == \
        f"/m/piper/{paths.PIPER_DEFAULT_MODEL}"  # the factory's default engine is piper
    dual = {"tts": {"engine": "dual", "piper": {"model": "v.onnx"}}}
    assert paths.tts_settings(dual, environ=TICKET) == dual["tts"]  # no models root needed


# --- review F10: development defaults follow paths.models_root --------------------------------------

def test_development_settings_follow_the_configured_file(tmp_path):
    settings = tmp_path / "settings.yaml"
    settings.write_text("paths:\n  models_root: /dev-models\nspeaker_id:\n  savedir: /sv\nhome_assistant:\n  token: x\n")
    environ = {"CONFIG_PATH": str(settings)}
    loaded = paths.development_settings(environ=environ)
    assert loaded == {"paths": {"models_root": "/dev-models"}, "speaker_id": {"savedir": "/sv"}}
    assert paths.models_path(loaded, "wakeword", environ=environ) == "/dev-models/wakeword"
    loaded["paths"]["models_root"] = "/mutated"
    assert paths.development_settings(environ=environ)["paths"]["models_root"] == "/dev-models"  # a copy
    settings.write_text("paths:\n  models_root: /changed-root\n")
    assert paths.development_settings(environ=environ) == {"paths": {"models_root": "/changed-root"}}
    assert paths.development_settings(environ={**environ, **TICKET}) == {}
    assert paths.development_settings(environ={"CONFIG_PATH": str(tmp_path / "missing.yaml")}) == {}
    settings.write_text("- not\n- a mapping\n")
    assert paths.development_settings(environ=environ) == {}
    settings.write_text("paths: [unclosed\n")
    assert paths.development_settings(environ=environ) == {}


def test_constructor_defaults_follow_the_development_settings(monkeypatch, tmp_path):
    """Review F10: a configured paths.models_root moves every unconfigured default with it."""
    import asyncio
    import sys
    import types
    from src.pipeline.model_manager import ModelManager, ModelManagerConfig
    from src.wakeword.detector import WakeWordDetector
    from src.wakeword.trainer import WakeWordTrainer
    settings = tmp_path / "settings.yaml"
    root = tmp_path / "dev-models"
    settings.write_text(f"paths:\n  models_root: {root}\n")
    monkeypatch.delenv(TICKET_ENV, raising=False)
    monkeypatch.setenv("CONFIG_PATH", str(settings))
    assert WakeWordDetector().custom_models_dir == root / "wakeword"
    assert WakeWordTrainer().models_dir == root / "wakeword"
    loads = []
    piper = types.ModuleType("piper")
    piper.PiperVoice = types.SimpleNamespace(load=lambda path: loads.append(path))
    monkeypatch.setitem(sys.modules, "piper", piper)
    asyncio.run(ModelManager(ModelManagerConfig(tts_voice="v"))._load_tts())
    assert loads == [f"{root}/piper/v.onnx"]
    asyncio.run(ModelManager(ModelManagerConfig(models_root="/explicit", tts_voice="v"))._load_tts())
    assert loads[-1] == "/explicit/piper/v.onnx"  # an explicit value still wins
