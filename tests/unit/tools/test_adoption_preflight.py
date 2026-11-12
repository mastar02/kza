"""The controller preflight on the repository settings after the 2a table (U3b-0e1).

Design v3 section 2a: with the A1 hub-id allow-list, the controller's own
path preflight on the repository settings.yaml, transformed by the 2a table,
reports zero problems. The table below is a hand copy of 2a's rows; the
adoption library (U3b-0e2) owns the real editor.
"""

import copy
import os
from pathlib import Path
import sys

import pytest
import yaml

from scripts import managed_launcher as launcher

REPOSITORY_SETTINGS = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
MODELS = "/home/kza/kza/models"
TABLE_2A = {
    ("stt", "model"): MODELS + "/whisper-v3-turbo",
    ("llm", "lora_path"): "",
    ("rooms", "wake_word", "porcupine", "keyword_path"): MODELS + "/wakeword/nexa_es.ppn",
    ("rooms", "wake_word", "porcupine", "model_path"): MODELS + "/wakeword/porcupine_params_es.pv",
    ("logging", "file"): "/home/kza/releases/logs/voice_assistant.log",
    ("training", "lora", "output_dir"): "./data/training_outputs/lora",
    ("training", "nightly", "output_dir"): "./data/training_outputs/nightly",
    ("training", "nightly", "whisper_finetune", "output_dir"): "./data/training_outputs/whisper",
    ("observability", "metrics_path"): "./data/obs/kza-metrics.jsonl",
    ("paths", "models_root"): MODELS,
    ("speaker_id", "savedir"): MODELS + "/speaker",
    ("hooks", "audit_sqlite", "enabled"): False,
}
DELETED_2A = (("hooks", "audit_sqlite_path"), ("code_index",))


def apply_2a(config: dict) -> dict:
    config = copy.deepcopy(config)
    for key, value in TABLE_2A.items():
        owner = config
        for part in key[:-1]:
            owner = owner.setdefault(part, {})
        owner[key[-1]] = value
    tail = config["llm"]["model_path"].split("models/", 1)[1]
    config["llm"]["model_path"] = MODELS + "/" + tail  # '<same tail>'
    for key in DELETED_2A:
        owner = config
        for part in key[:-1]:
            owner = owner[part]
        owner.pop(key[-1], None)
    return config


def preflight(tmp_path: Path, config: dict) -> list[str]:
    """Run the unchanged controller preflight on the prepare_slot layout."""
    state, runtime = tmp_path / "state", tmp_path / "runtime"
    (state / "config").mkdir(parents=True)
    (state / "data").mkdir()
    runtime.mkdir()
    for name in ("config", "data"):
        os.symlink(Path("..") / "state" / name, runtime / name)
    (state / "config" / "settings.yaml").write_text(yaml.safe_dump(config))
    return launcher.path_preflight({"runtime_root": str(runtime), "state_root": str(state)}, sys.executable)


def test_repository_settings_after_the_2a_table_have_zero_problems(tmp_path):
    config = apply_2a(yaml.safe_load(REPOSITORY_SETTINGS.read_text()))
    assert preflight(tmp_path, config) == []


def test_repository_settings_without_the_table_still_refuse(tmp_path):
    problems = preflight(tmp_path, yaml.safe_load(REPOSITORY_SETTINGS.read_text()))
    assert "relative_path:paths.models_root" in problems
    assert "relative_path:stt.model" in problems


HUB_KEYS = [("tts", "kokoro", "model"), ("embeddings", "model"), ("speaker_id", "model"), ("emotion", "model"),
            ("training", "lora", "base_model"), ("training", "nightly", "base_model"),
            ("training", "nightly", "whisper_finetune", "base_model"), ("code_index", "embedder", "model")]


def _nested(key, value):
    config = {}
    owner = config
    for part in key[:-1]:
        owner = owner.setdefault(part, {})
    owner[key[-1]] = value
    return config


@pytest.mark.parametrize("key", HUB_KEYS)
def test_a1_hub_ids_pass_only_at_the_listed_keys(tmp_path, key):
    assert preflight(tmp_path, _nested(key, "BAAI/bge-m3")) == []


@pytest.mark.parametrize("key, value", [
    (("embeddings", "model"), "org/name/extra"),       # two slashes
    (("embeddings", "model"), "./org/name"),           # leading dot
    (("embeddings", "model"), "models/x"),             # a local tree
    (("speaker_id", "model"), "src/x"),              # the runtime code tree
    (("emotion", "model"), "logs/x"),
    (("tts", "kokoro", "model"), "-org/name"),
    (("tts", "kokoro", "model"), "org/name\n"),
    (("stt", "model"), "openai/whisper-large"),       # not a hub key
    (("rooms", "wake_word", "model"), "org/name"),
])
def test_a1_other_relative_values_still_refuse(tmp_path, key, value):
    assert preflight(tmp_path, _nested(key, value)) == ["relative_path:" + ".".join(key)]


def test_a_relative_speaker_savedir_refuses(tmp_path):
    assert preflight(tmp_path, {"speaker_id": {"savedir": "./models/speaker"}}) == [
        "relative_path:speaker_id.savedir"]
    assert preflight(tmp_path / "b", {"speaker_id": {"savedir": MODELS + "/speaker"}}) == []
