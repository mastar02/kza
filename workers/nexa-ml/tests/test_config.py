"""Typed ``[ml]`` parsing from nexa.toml."""

import pathlib

import pytest

from nexa_ml.config import ConfigError, MlConfig, load_config, parse_ml_section

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_repo_nexa_toml_parses() -> None:
    cfg = load_config(REPO_ROOT / "config" / "nexa" / "nexa.toml")
    assert cfg.command_stt_model == "whisper-v3-turbo"
    assert cfg.command_stt_device == "cuda:0"
    assert cfg.ambient_stt_model == "whisper-v3-turbo"
    assert cfg.ambient_stt_device == "cuda:0"
    assert cfg.ambient_stt_workers == 1
    assert cfg.tts_device == "cuda:0"
    assert cfg.speaker_encoder_device == "cuda:0"
    assert cfg.embeddings_device == "cpu"


def test_defaults_when_section_missing() -> None:
    cfg = parse_ml_section({})
    assert cfg == MlConfig()
    assert cfg.speaker_encoder_device == "cuda:0"
    assert cfg.command_vad_filter is False and cfg.ambient_vad_filter is True


def test_speaker_device_respected_from_config() -> None:
    assert parse_ml_section({"ml": {"speaker_encoder_device": "cuda:1"}}).speaker_encoder_device == "cuda:1"


def test_unknown_keys_are_ignored(caplog: pytest.LogCaptureFixture) -> None:
    cfg = parse_ml_section({"ml": {"future_knob": 1}})
    assert cfg == MlConfig()
    assert "future_knob" in caplog.text


@pytest.mark.parametrize(
    "section",
    [
        {"ambient_stt_workers": "2"},
        {"ambient_stt_workers": True},
        {"ambient_stt_workers": 0},
        {"command_vad_filter": "no"},
        {"tts_device": "gpu1"},
        {"stt_language": "Spanish"},
        {"models_dir": ""},
        {"max_audio_seconds": 0},
    ],
)
def test_invalid_values_raise(section: dict) -> None:
    with pytest.raises(ConfigError):
        parse_ml_section({"ml": section})


def test_float_accepts_int_and_optional_path() -> None:
    cfg = parse_ml_section({"ml": {"max_audio_seconds": 60, "models_dir": "/var/lib/nexa/models"}})
    assert cfg.max_audio_seconds == 60.0 and isinstance(cfg.max_audio_seconds, float)
    assert cfg.models_dir == "/var/lib/nexa/models"


def test_missing_or_broken_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[ml\n")
    with pytest.raises(ConfigError):
        load_config(bad)
    not_table = tmp_path / "nt.toml"
    not_table.write_text('ml = "x"\n')
    with pytest.raises(ConfigError):
        load_config(not_table)


def test_load_config_inherits_command_compute_type_when_shared_worker_omits_it(tmp_path) -> None:
    from pathlib import Path

    toml_file = tmp_path / "nexa.toml"
    toml_file.write_text(
        """
[ml]
command_stt_model = "whisper-v3-turbo"
command_stt_device = "cuda:0"
command_stt_compute_type = "float16"
ambient_stt_model = "whisper-v3-turbo"
ambient_stt_device = "cuda:0"
ambient_stt_workers = 1
""",
        encoding="utf-8",
    )
    cfg = load_config(toml_file)
    assert cfg.ambient_stt_compute_type == "float16"

    repo_cfg = load_config(Path(__file__).resolve().parents[3] / "config" / "nexa" / "nexa.toml")
    assert repo_cfg.command_stt_compute_type == repo_cfg.ambient_stt_compute_type == "float16"
