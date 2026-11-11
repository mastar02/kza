"""U3b-4b-2 pins: input channels, the composition root, config reading and the GPU contract.

A verification phrase may only arrive through RequestRouter.process_command,
whose single caller is VoicePipeline.process_command, with a CommandEvent
built at one of three known sites. A new input path fails these pins until
the verifier names its channel.
"""

import ast
import copy
from pathlib import Path

import yaml

from src.memory.privacy_services import voice_proposal_settings
from src.memory.voice_attribution import VoiceProposalSettings

ROOT = Path(__file__).resolve().parents[3]
VOICE_MODULES = ("src/users/voice_attributor.py", "src/users/voice_scoring.py", "src/memory/voice_phrase.py")


def _calls(predicate):
    found = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(function):
                    if isinstance(node, ast.Call) and predicate(node):
                        found.append((str(path.relative_to(ROOT)), function.name, ast.unparse(node.func)))
    return sorted(set(found)), len(found)


def test_request_router_process_command_has_a_single_caller():
    sites, count = _calls(lambda node: isinstance(node.func, ast.Attribute) and node.func.attr == "process_command")
    assert sites == [
        ("src/pipeline/request_router.py", "_process_command_legacy", "self.command_processor.process_command"),
        ("src/pipeline/request_router.py", "_process_command_orchestrated", "self.command_processor.process_command"),
        ("src/pipeline/voice_pipeline.py", "process_command", "self.request_router.process_command"),
    ] and count == 3


def test_command_events_are_built_at_exactly_three_sites():
    sites, count = _calls(lambda node: ast.unparse(node.func).split(".")[-1] == "CommandEvent")
    assert [site[:2] for site in sites] == [
        ("src/ambient/textual_wake.py", "maybe_dispatch"),          # textual_wake=True: never matches
        ("src/pipeline/audio_loop.py", "_poll"),                    # room_id '': never matches
        ("src/pipeline/multi_room_audio_loop.py", "_poll_command"),  # the only verification channel
    ] and count == 3


def test_the_single_mic_and_textual_wake_sites_cannot_name_a_room_microphone():
    single = (ROOT / "src/pipeline/audio_loop.py").read_text(encoding="utf-8")
    textual = (ROOT / "src/ambient/textual_wake.py").read_text(encoding="utf-8")
    assert "CommandEvent(audio=audio, room_id=''" in single
    tree = ast.parse(textual)
    event = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "CommandEvent")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in event.keywords}
    assert keywords.get("textual_wake") == "True"


def test_voice_registry_observe_and_encoder_registration_sites_are_unchanged():
    observe, _ = _calls(lambda node: isinstance(node.func, ast.Attribute) and node.func.attr == "observe"
                        and "voice_registry" in ast.unparse(node.func.value))
    assert [site[:2] for site in observe] == [
        ("src/ambient/speaker_tagger.py", "tag"), ("src/orchestrator/dispatcher.py", "_identify_speaker"),
        ("src/pipeline/command_processor.py", "_process_command")]
    register, count = _calls(lambda node: isinstance(node.func, ast.Attribute) and node.func.attr == "register_encoder")
    assert [site[:2] for site in register] == [
        ("src/ambient/speaker_tagger.py", "__init__"), ("src/main.py", "build_application"),
        ("src/pipeline/command_processor.py", "__init__")] and count == 3


def test_the_composition_root_builds_the_stage_once_and_hands_it_to_the_router():
    sites, count = _calls(lambda node: ast.unparse(node.func) == "build_voice_attribution")
    assert sites == [("src/main.py", "build_application", "build_voice_attribution")] and count == 1
    tree = ast.parse((ROOT / "src/main.py").read_text(encoding="utf-8"))
    router = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "RequestRouter")
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in router.keywords}["voice_attributor"] == "voice_attributor"


def test_the_voice_stage_adds_no_inference_and_no_device():
    """4b reuses the registry's stored embeddings: no model, encoder call, torch or device in the new modules."""
    for relative in VOICE_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names} | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not names & {"get_embedding", "SpeakerIdentifier", "identify", "compute_similarity", "cuda", "device",
                            "register_encoder"}, relative
        assert not {name for name in imports if name and name.split(".")[0] in {"torch", "speechbrain", "onnxruntime"}}, relative


def test_gpu_assignment_is_unchanged():
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    assert config["speaker_id"]["device"] == "cuda:1"


def test_the_repository_settings_carry_no_voice_section_and_reading_never_writes():
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    assert "voice_proposals" not in (config.get("memory") or {})
    assert voice_proposal_settings(config) == VoiceProposalSettings() and VoiceProposalSettings().enabled is False
    for section in ({}, {"enabled": "yes"}, {"enabled": True, "min_score": 0.9}, {"enabled": True, "min_score": "x"}):
        given = {"memory": {"voice_proposals": section}}
        before = copy.deepcopy(given)
        voice_proposal_settings(given)
        assert given == before


def test_placeholder_thresholds_are_marked_and_unchanged():
    """0.80 / 0.10 / 1.5 s are placeholders until 4c calibrates them after P7."""
    settings = VoiceProposalSettings()
    assert (settings.min_score, settings.min_margin, settings.min_speech_s, settings.embedding_wait_s,
            settings.min_scoring_profiles) == (0.80, 0.10, 1.5, 2.0, 2)
    source = (ROOT / "src/memory/voice_attribution.py").read_text(encoding="utf-8")
    assert "placeholders until field evidence (pending_4c)" in source
    assert "# PLACEHOLDERS, not calibrated: U3b-4c sets them from the P7 field evidence" in source
    assert source.count("# placeholder") == 5
