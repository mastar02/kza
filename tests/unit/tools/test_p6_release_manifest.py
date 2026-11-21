"""T064 reviews real build_runtime archives without contacting a serving system."""

from contextlib import closing
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile

import pytest

from scripts.build_runtime import build_runtime
from tests.mocks.epoch4 import edited_commit
from tools import release_manifest_review as review


REPOSITORY = Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def candidate(tmp_path_factory) -> review.ReviewInputs:
    base = tmp_path_factory.mktemp("p6-release-review")
    archive = base / "candidate.tar.gz"
    manifest = build_runtime(REPOSITORY, archive)
    lock = base / "runtime.lock"
    lock.write_text("aiosqlite==0.22.1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    return review.ReviewInputs(REPOSITORY, archive, manifest["git_commit"], _digest(archive), lock)


def _rewrite(source: Path, target: Path, *, release_fields=None, extra=None, corrupt_pyc=False) -> None:
    with tarfile.open(source, "r:gz") as bundle:
        files = {member.name: bundle.extractfile(member).read() for member in bundle}
    manifest = json.loads(files.pop("release.json"))
    if extra:
        files.update(extra)
    if corrupt_pyc:
        files["src/main.pyc"] = b"\x00" + files["src/main.pyc"][1:]
    manifest.update(release_fields or {})
    manifest["files"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["release.json"] = json.dumps(manifest).encode()
    with tarfile.open(target, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))


def _args(candidate: review.ReviewInputs) -> list[str]:
    return [
        str(candidate.archive), "--commit", candidate.commit, "--archive-sha256", candidate.archive_sha256,
        "--runtime-lock", str(candidate.runtime_lock),
    ]


def test_cli_recomputes_and_verifies_candidate_manifest(candidate, tmp_path, capsys):
    assert review.main(_args(candidate)) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["git_commit"] == candidate.commit
    assert report["artifact"]["python_abi"] == "cp313"
    assert report["artifact"]["cache_tag"] == "cpython-313"
    assert report["artifact"]["python_sources"] == 0
    assert report["artifact"]["sha256"] == candidate.archive_sha256
    assert report["schema_contract"] == {
        "epoch": 2, "sha256": "d0b616eaf5ce55c8995913a0063808235c336945a4a97ca8adee90ad65d71dc5",
    }
    assert report["external_runtime_lock_sha256"] == _digest(candidate.runtime_lock)
    for path in ("requirements.txt", "requirements-test.txt", "pyproject.toml", "config/settings.yaml"):
        committed = subprocess.check_output(["git", "show", f"{candidate.commit}:{path}"], cwd=REPOSITORY)
        assert report["committed_declarations_sha256"][path] == hashlib.sha256(committed).hexdigest()
    assert report["synthetic_adoption_preflight"] == {
        "schema_only": "passed", "reminders_preserved": 1, "home_actions_preserved": 1,
        "timers": 0, "alert_preferences": 0, "legacy_current_verified": True,
        "legacy_links_verified": 5, "frozen_database_unchanged": True,
    }

    sealed = tmp_path / "review.json"
    sealed.write_text(output, encoding="utf-8")
    assert review.main(_args(candidate) + ["--verify", str(sealed)]) == 0
    assert capsys.readouterr().out == ""
    changed_lock = tmp_path / "changed.lock"
    changed_lock.write_bytes(candidate.runtime_lock.read_bytes() + b"# drift\n")
    assert review.main(_args(replace(candidate, runtime_lock=changed_lock)) + ["--verify", str(sealed)]) == 78
    refusal = capsys.readouterr()
    assert refusal.out == ""
    assert "differs from the recomputed evidence" in refusal.err

    report["rollback"]["retire_after_seconds"] -= 1
    sealed.write_text(json.dumps(report), encoding="utf-8")
    assert review.main(_args(candidate) + ["--verify", str(sealed)]) == 78
    assert "differs from the recomputed evidence" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"release_fields": {"git_commit": "a" * 40}}, "differs from the candidate SHA"),
        ({"release_fields": {"schema_contract": 1}}, "must declare schema contract epoch 2"),
        ({"release_fields": {"schema_contract": 2.0}}, "must declare schema contract epoch 2"),
        ({"extra": {"src/injected.py": b"pass\n"}}, "Unexpected or unsafe runtime archive member"),
        ({"extra": {"src/.py": b"pass\n"}}, "contains Python source"),
        ({"corrupt_pyc": True}, "invalid CPython 3.13 bytecode header"),
    ],
)
def test_refuses_mutated_archive_even_with_updated_checksums(candidate, tmp_path, mutation, error):
    archive = tmp_path / "mutated.tar.gz"
    _rewrite(candidate.archive, archive, **mutation)
    with pytest.raises(ValueError, match=error):
        review.review_manifest(replace(candidate, archive=archive, archive_sha256=_digest(archive)))


def test_refuses_broken_seal(candidate):
    with pytest.raises(ValueError, match="differs from the sealed hash"):
        review.review_manifest(replace(candidate, archive_sha256="0" * 64))


def test_refuses_missing_or_empty_lock(candidate, tmp_path):
    lock = tmp_path / "missing.lock"
    with pytest.raises(ValueError, match="lock is missing"):
        review.review_manifest(replace(candidate, runtime_lock=lock))
    lock.write_text("# no dependencies\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lock is empty"):
        review.review_manifest(replace(candidate, runtime_lock=lock))


def test_fixture_matches_frozen_reminders_schema():
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(review._LEGACY_SQL)
        columns = [row[1] for row in db.execute("PRAGMA table_info(reminders)")]
        indexes = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'reminders' "
            "AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    assert columns == [
        "id", "user_id", "text", "trigger_at", "created_at", "recurrence",
        "recurrence_end", "ha_actions", "state", "last_fired_at", "fire_count",
    ]
    assert indexes == [("idx_reminders_trigger",), ("idx_reminders_user",)]


@pytest.mark.parametrize("path", [Path("/home/kza"), Path("/home/kza/releases/candidate.tar.gz")])
def test_review_path_refuses_production_path(path):
    with pytest.raises(ValueError, match="outside the production tree"):
        review._review_path(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("legacy_commit", "0" * 40, "disagrees with the Q4 legacy source or rollback target"),
        ("frozen_state_slot", "unexpected-slot", "disagrees with the Q4 legacy source or rollback target"),
        ("rollback_phrase", "wrong phrase", "disagrees with the Q4 legacy source or rollback target"),
        ("controller_revision", 2, "disagrees with release.json: controller_revision"),
    ],
)
def test_check_controller_refuses_discrepancy(field, value, error):
    manifest = {"controller_epoch": 5, "controller_revision": 1, "launcher_contract": 3}
    compiled = {
        **manifest,
        "schema_contract": review.SCHEMA_EPOCH,
        "schema_contract_sha256": review.SCHEMA_DIGEST,
        "legacy_commits": [review.LEGACY_COMMIT[:7]],
        "legacy_commit": review.LEGACY_COMMIT,
        "adoption_source_commit": review.LEGACY_COMMIT,
        "frozen_state_slot": review.FROZEN_SLOT,
        "legacy_links": 5,
        "rollback_phrase": review.L4_PHRASE,
    }
    review._check_controller(compiled, manifest)
    with pytest.raises(ValueError, match=error):
        review._check_controller({**compiled, field: value}, manifest)


def test_refuses_real_build_with_epoch_two_and_different_contract(candidate, tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPOSITORY), str(repo)],
                   check=True, timeout=120)
    commit = edited_commit(repo, candidate.commit, {
        "scripts/stage_runtime.py": [(
            b'"protected": sorted(_PROTECTED_TABLES),',
            b'"protected": [],',
        )],
    }, message="compile a different digest under the same epoch")
    archive = tmp_path / "drift.tar.gz"
    build_runtime(repo, archive, commit)
    drift = replace(candidate, repo=repo, commit=commit, archive=archive, archive_sha256=_digest(archive))
    with pytest.raises(ValueError, match="compiled schema contract digest differs"):
        review.review_manifest(drift)


def test_l4_plan_targets_only_the_e9669ec_container():
    plan = review.rollback_plan()
    slot = "release_20261029T185008966364Z_e9669ec"
    assert plan["legacy_slot"] == slot
    assert plan["legacy_commit"] == "e9669ec1edb101ca6b190ef8df09314e1ac63562"
    assert plan["frozen_state_slot"] == "release_20261028T145442Z_b0866e8"
    assert plan["retire_after_seconds"] == 1_209_600
    assert plan["day_zero"] == "ended_at of the first clean managed close (W9)"
    assert plan["rollback_argv"][7:] == [
        "uninstall-launcher", "--unit-dir", "<user-unit-drop-in-dir>", "--user-decision", "<L4-decision-id>",
        "--phrase", "revivir datos borrados", "--legacy-slot", slot,
    ]
    assert plan["retirement_dry_run_argv"][-3:] == ["--container-slot", slot, "--dry-run"]
    assert plan["retirement_confirm_argv"][-4:] == ["--container-slot", slot, "--confirm", "<Q5-plan-sha256>"]
    assert plan["retirement_blocked_after_l4"] is True
