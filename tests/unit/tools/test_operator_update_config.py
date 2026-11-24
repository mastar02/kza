"""Tests for operator-sanctioned settings.yaml updates on closed managed heads (update-config)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest
import yaml

from scripts import managed_launcher as launcher, stage_runtime as stage
from tests.mocks.runtime_artifact_cache import cached_build_runtime as build_runtime


@pytest.fixture(scope="module")
def update_config_archive(tmp_path_factory):
    destination = tmp_path_factory.mktemp("update-config-artifact") / "runtime.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], destination)
    return destination


def _write_clean_receipt_child() -> list[str]:
    script = (
        "import importlib.util, os\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('_st', Path.cwd() / 'scripts' / 'stage_runtime.pyc')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "mod.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], {\n"
        "    'graph_closed': True, 'privacy_closed': True, 'sdk_closed': True, 'native_closed': True,\n"
        "})\n"
    )
    return [sys.executable, "-I", "-B", "-c", script]


@pytest.fixture
def closed_world(update_config_archive, tmp_path):
    """Prepare a managed slot and perform one clean closed run so a closure witness exists."""
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    initial_cfg = {
        "stt": {"engine": "whisper", "model": "/models/whisper-v3-turbo", "device": "cuda:1"},
        "rooms": {
            "enabled": True,
            "escritorio": {"capture_channel": 1, "mic_usb_port": "3-1.4"},
        },
        "ambient": {
            "enabled": True,
            "asr_col": 1,
            "vad_col": 1,
            "db_path": "./data/ambient.db",
            "keep_audio": {"enabled": False, "dir": "./data/ambient_audio"},
            "stt": {"engine": "parakeet", "device": "cuda:0"},
        },
        "reasoner": {"cloud": {"consent": False}},
    }
    (source / "config" / "settings.yaml").write_text(
        yaml.safe_dump(initial_cfg, sort_keys=False), encoding="utf-8"
    )
    root = tmp_path / "releases"
    stage.prepare_slot(update_config_archive, root, "slot_v1", source)
    # One clean run establishes a closure witness + authority seal.
    stage.run_slot(root, "slot_v1", _write_clean_receipt_child(), timeout=15)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    return root, head


def test_update_config_dry_run_and_apply_patch(closed_world, tmp_path):
    root, head_before = closed_world
    settings_path = root / "slots" / "slot_v1" / "state" / "config" / "settings.yaml"
    os.chmod(settings_path, 0o600)
    # Re-verify nothing changed before dry-run
    before_bytes = settings_path.read_bytes()
    before_inode = settings_path.lstat().st_ino

    patch_file = tmp_path / "audio_patch.yaml"
    patch_file.write_text(
        yaml.safe_dump(
            {
                "rooms": {"escritorio": {"capture_channel": 0}},
                "ambient": {
                    "asr_col": 0,
                    "vad_col": 0,
                    "stt": {"engine": "hybrid", "workers": 2, "compute_type": "int8_float16"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dry = launcher.update_config(root, patch_path=patch_file, dry_run=True, reason="enable ch0 hybrid stt")
    assert dry["dry_run"] is True
    assert dry["slot"] == "slot_v1"
    assert dry["changed_keys"] == [
        "ambient.asr_col",
        "ambient.stt.compute_type",
        "ambient.stt.engine",
        "ambient.stt.workers",
        "ambient.vad_col",
        "rooms.escritorio.capture_channel",
    ]
    # Dry run must not touch settings.yaml, head.json, or admission/config-changes
    assert settings_path.read_bytes() == before_bytes
    assert stage.read_admission_head(root) == head_before
    assert not stage.config_change_directory(root).exists()

    # Wrong confirmation hash must refuse without writing
    with pytest.raises(launcher.LauncherRefusal) as err:
        launcher.update_config(root, patch_path=patch_file, confirm="0" * 64, reason="wrong")
    assert err.value.code == "runtime_admission_config_confirmation_mismatch"
    assert settings_path.read_bytes() == before_bytes

    # Confirmed apply writes settings atomically, creates immutable audit record & manual head
    applied = launcher.update_config(
        root, patch_path=patch_file, confirm=dry["plan_sha256"], reason="enable ch0 hybrid stt"
    )
    assert applied["dry_run"] is False
    assert applied["head"]["status"] == "closed"
    assert applied["head"]["generation"] == head_before["generation"] + 1
    assert applied["new_binding_id"] != head_before["binding_id"]

    # Atomic replace yields a new inode and preserves 0600 permissions
    st = settings_path.lstat()
    assert st.st_ino != before_inode
    assert stat.S_IMODE(st.st_mode) == 0o600
    updated = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert updated["rooms"]["escritorio"]["capture_channel"] == 0
    assert updated["ambient"]["asr_col"] == 0
    assert updated["ambient"]["stt"]["engine"] == "hybrid"
    assert updated["ambient"]["stt"]["workers"] == 2
    assert updated["ambient"]["db_path"] == "./data/ambient.db"

    # Immutable content-free audit record exists (0600, write-once) and contains no values
    record_path = stage.config_change_record_path(root, applied["head"])
    assert record_path.exists()
    assert stat.S_IMODE(record_path.lstat().st_mode) == 0o600
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["changed_keys"] == dry["changed_keys"]
    assert record["reason"] == "enable ch0 hybrid stt"
    assert "int8_float16" not in record_path.read_text(encoding="utf-8")

    # Closed binding verification and closed_witness succeed immediately after update-config
    with stage.AdmissionController(root, allow_open_head=False, gate="normal"):
        verified = stage.verify_closed_binding(root, applied["new_binding_id"])
        assert verified["slot"] == "slot_v1"
        witness = stage.closed_witness(root, applied["new_binding_id"])
        assert witness["kind"] == "closure"

    # A subsequent managed run boots and seals cleanly against the new config binding
    stage.run_slot(root, "slot_v1", _write_clean_receipt_child(), timeout=15)
    after_run = stage.read_admission_head(root)
    assert after_run["status"] == "closed"
    assert after_run["generation"] == applied["head"]["generation"] + 3
    assert stage.read_manual_head(root, after_run)["kind"] == "closure"


def test_update_config_chained_updates_keep_valid_witness(closed_world, tmp_path):
    root, _ = closed_world
    p1 = tmp_path / "p1.yaml"
    p1.write_text("ambient:\n  asr_col: 0\n", encoding="utf-8")
    d1 = launcher.update_config(root, patch_path=p1, dry_run=True)
    r1 = launcher.update_config(root, patch_path=p1, confirm=d1["plan_sha256"], reason="step 1")

    p2 = tmp_path / "p2.yaml"
    p2.write_text("ambient:\n  vad_col: 0\n", encoding="utf-8")
    d2 = launcher.update_config(root, patch_path=p2, dry_run=True)
    r2 = launcher.update_config(root, patch_path=p2, confirm=d2["plan_sha256"], reason="step 2")

    assert r2["head"]["generation"] == r1["head"]["generation"] + 1
    with stage.AdmissionController(root, allow_open_head=False, gate="normal"):
        stage.verify_closed_binding(root, r2["new_binding_id"])
        assert stage.closed_witness(root, r2["new_binding_id"])["kind"] == "closure"


def test_update_config_refuses_protected_paths_and_cloud_consent_escalation(closed_world, tmp_path):
    root, head_before = closed_world

    # Protected storage path change must be refused
    bad_path = tmp_path / "bad_path.yaml"
    bad_path.write_text("ambient:\n  db_path: /tmp/evil.db\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherRefusal) as err:
        launcher.update_config(root, patch_path=bad_path, dry_run=True)
    assert err.value.code == "runtime_admission_config_protected_path_changed"
    assert "ambient.db_path" in err.value.problems

    # Silent cloud consent escalation must be refused unless explicitly authorized
    consent_up = tmp_path / "consent_up.yaml"
    consent_up.write_text("reasoner:\n  cloud:\n    consent: true\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherRefusal) as err2:
        launcher.update_config(root, patch_path=consent_up, dry_run=True)
    assert err2.value.code == "runtime_admission_config_cloud_consent_grant_refused"

    # Explicit authorization allows granting, and withdrawing needs no flag
    d_grant = launcher.update_config(
        root, patch_path=consent_up, dry_run=True, allow_cloud_consent_grant=True
    )
    launcher.update_config(
        root, patch_path=consent_up, confirm=d_grant["plan_sha256"],
        allow_cloud_consent_grant=True, reason="explicit grant",
    )
    consent_down = tmp_path / "consent_down.yaml"
    consent_down.write_text("reasoner:\n  cloud:\n    consent: false\n", encoding="utf-8")
    d_withdraw = launcher.update_config(root, patch_path=consent_down, dry_run=True)
    r_withdraw = launcher.update_config(
        root, patch_path=consent_down, confirm=d_withdraw["plan_sha256"], reason="withdraw",
    )
    assert r_withdraw["head"]["generation"] == head_before["generation"] + 2


def test_update_config_refuses_noop_invalid_and_open_head(closed_world, tmp_path):
    root, head_before = closed_world

    # No-op patch
    noop = tmp_path / "noop.yaml"
    noop.write_text("ambient:\n  asr_col: 1\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherRefusal) as err:
        launcher.update_config(root, patch_path=noop, dry_run=True)
    assert err.value.code == "runtime_admission_config_unchanged"

    # Non-dict YAML
    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherRefusal) as err2:
        launcher.update_config(root, patch_path=bad_yaml, dry_run=True)
    assert err2.value.code == "runtime_admission_config_invalid"

    # Open/unknown head must be refused
    head_path = root / "admission" / "head.json"
    os.chmod(head_path, 0o600)
    open_head = dict(head_before, status="unknown", run_id="0" * 32)
    head_path.write_text(json.dumps(open_head), encoding="utf-8")
    patch = tmp_path / "ok.yaml"
    patch.write_text("ambient:\n  asr_col: 0\n", encoding="utf-8")
    with pytest.raises(launcher.LauncherRefusal) as err3:
        launcher.update_config(root, patch_path=patch, dry_run=True)
    assert err3.value.code == "runtime_admission_not_closed"


def test_update_config_detects_post_change_tampering_and_preserves_origin_kind(closed_world, tmp_path):
    root, _ = closed_world
    patch = tmp_path / "p.yaml"
    patch.write_text("ambient:\n  asr_col: 0\n", encoding="utf-8")
    dry = launcher.update_config(root, patch_path=patch, dry_run=True)
    applied = launcher.update_config(root, patch_path=patch, confirm=dry["plan_sha256"], reason="ch0")

    manual = stage.read_manual_head(root, applied["head"])
    assert manual["kind"] == "config_change"
    assert launcher._origin_kind(root, manual) == "closure"

    # Out-of-band edit to settings.yaml after update-config must still fail privacy_authority_stale
    settings_path = root / "slots" / "slot_v1" / "state" / "config" / "settings.yaml"
    settings_path.write_text(settings_path.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    with stage.AdmissionController(root, allow_open_head=False, gate="normal"):
        with pytest.raises(ValueError, match="privacy_authority_stale"):
            stage.verify_closed_binding(root, applied["new_binding_id"])
