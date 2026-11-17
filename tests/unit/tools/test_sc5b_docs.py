"""SC-5b: the release-mode runbook names real commands and codes, in the design's order.

sc-release-design v4 implementation_units SC-5b tests: docs tests in the style
of test_sc5a_docs.py, incl. the rollback path without T0, the stop-and-disable
after T0, HD without L4, the 48-hour decision for CI, T0F and HD as well as SS,
the venv rule, the $PY -I -B T0 form (R4-PRIV-3, R4-OPS-6), the SC-D8 A
procedure with its written confirmation and the SC-D5 A gap stop. Every `$ST`
command of the runbook is parsed by stage_runtime's real parser, and every
refusal code it names must exist in the controller's code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import sys

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage

REPO = Path(__file__).resolve().parents[3]
RUNBOOK = REPO / "docs/runbooks/2026-11-17-release-mode.md"
DEPLOYMENT = REPO / "docs/architecture/DEPLOYMENT.md"
COMPACTION_RUNBOOK = REPO / "docs/runbooks/2026-11-14-owner-compaction.md"
SOURCES = ("scripts/stage_runtime.py", "scripts/compaction.py", "scripts/managed_launcher.py")
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_CODE = re.compile(r"runtime_admission_[a-z0-9_]*[a-z0-9]")


def _text() -> str:
    return RUNBOOK.read_text()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` up to the next heading of the same or a higher level."""
    level = heading.split(" ", 1)[0]
    body = text.split("\n" + heading + "\n", 1)[1]
    stop = re.search(r"\n#{1,%d} " % len(level), body)
    return body[:stop.start()] if stop else body


def _blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", text, flags=re.S)


def _code_lines(text: str) -> list[str]:
    """Bash block lines (comments cut) and inline code spans, flattened (spans may wrap)."""
    found = [line.split(" #", 1)[0].strip() for block in _blocks(text) for line in block.splitlines()]
    prose = _flat(re.sub(r"```.*?```", " ", text, flags=re.S))
    return found + [span.strip() for span in re.findall(r"`([^`]+)`", prose)]


def _commands(text: str) -> list[str]:
    """Every `$ST ...` command of the runbook."""
    return [line for line in _code_lines(text) if line.startswith("$ST ")]


class _Parsed(Exception):
    pass


def _parse(monkeypatch, tokens: list[str]) -> argparse.Namespace:
    """stage_runtime.main()'s own parser (launcher, adoption and compact commands), nothing run."""
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["stage_runtime.py", "--root", "/nonexistent", "--offline", *tokens])
    try:
        stage.main()
    except _Parsed as parsed:
        return parsed.args[0]
    except SystemExit as exited:
        pytest.fail(f"stage_runtime rejects {tokens!r} (exit {exited.code})")
    pytest.fail(f"stage_runtime ran {tokens!r} instead of parsing it")


def _tokens(command: str) -> list[str]:
    return shlex.split(re.sub(r"<[^>]*>", "10", command))


def test_relative_links_resolve():
    for target in _LINK.findall(_text()):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        assert (RUNBOOK.parent / target.split("#", 1)[0]).resolve().exists(), target


def test_every_st_command_parses_with_stage_runtimes_parser(monkeypatch):
    commands = _commands(_text())
    heads = set()
    for command in commands:
        tokens = _tokens(command)[1:]  # drop $ST
        args = _parse(monkeypatch, tokens)
        heads.add(" ".join(tokens[:2]) if tokens[0] == "compact" else tokens[0])
        if tokens[:2] == ["compact", "abort"] and "--discard-k-changes" in tokens:
            assert args.to_h and args.confirm_h and args.discard_k_changes, command
        if tokens[:2] == ["compact", "verify"]:
            assert args.smoke_seconds is not None, command  # release mode: N is required (SC4A-N14)
    assert heads == {"completion-status", "adopt", "compact plan", "compact start", "compact resume",
                     "compact verify", "compact status", "compact retire", "compact abort"}, heads


def test_the_t0_runs_the_extracted_new_archive_with_py_i_b(monkeypatch):
    """R4-OPS-6b / R4-PRIV-3: a .pyc is not executable on its own, and T0 never runs the installed $ST."""
    text = _text()
    assert "$ST install-controller" not in text
    lines = [line for line in _code_lines(text) if "install-controller" in line and "stage_runtime.pyc" in line]
    assert ('"$PY" -I -B "$NEW"/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller '
            '"$NEW_ARCHIVE"') in lines
    for line in lines:
        assert line.startswith('"$PY" -I -B '), line
        args = _parse(monkeypatch, _tokens(line.split("--offline ", 1)[1]))
        assert args.command == "install-controller"
    for document in (RUNBOOK, DEPLOYMENT, COMPACTION_RUNBOOK):
        flat = _flat(document.read_text())
        for found in re.finditer(r"scripts/stage_runtime\.pyc[^`]{0,60}install-controller", flat):
            assert "-I -B" in flat[max(0, found.start() - 30):found.start()], (document.name, found.group(0))
    deployment = _flat(DEPLOYMENT.read_text())
    assert ("`$PY -I -B <tmp>/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller "
            "<archivo>`, con `$PY` = `/home/kza/kza/.venv/bin/python`") in deployment


def test_every_refusal_code_the_runbook_names_exists_in_the_controller():
    source = "\n".join((REPO / relative).read_text() for relative in SOURCES)
    named = set(_CODE.findall(_text()))
    missing = sorted(code for code in named if re.search(r"\b" + code + r"\b", source) is None)
    assert not missing, missing
    required = {compaction.ABORT_PROTECTED_CHANGED, compaction.DISCARD_MISMATCH, compaction.H_CHANGED,
                compaction.H_TRANSIENT, compaction.K_TRANSIENT, compaction.SDK_CHANGED,
                compaction.RETIRE_PROTECTED_CHANGED, compaction.RETIRE_RESURRECTED, compaction.K_BINDING_CHANGED,
                compaction.K_CLOSURE_UNREADABLE, compaction.T0_REQUIRED, compaction.CONTROLLER_BEHIND,
                compaction.PLAIN_MODE_ONLY, compaction.RELEASE_SMOKE_SECONDS, compaction.UNLISTED_COLLECTION,
                stage.CONTROLLER_INSTALL_INCOMPLETE, stage.DELEGATION_FROZEN, stage.OWNERS_PRESENT,
                stage.COMPACTION_OWNERS_PRESENT,
                "runtime_admission_compaction_abort_to_h_before_run", "runtime_admission_compaction_abort_after_run",
                "runtime_admission_compaction_abort_recheck_required", "runtime_admission_compaction_owner_only_data",
                "runtime_admission_compaction_pending", "runtime_admission_not_closed",
                "runtime_admission_compaction_head_moved", "runtime_admission_compaction_verify_consumed",
                "runtime_admission_controller_binding_changed", "runtime_admission_controller_downgrade",
                "runtime_admission_controller_installer_mismatch", "runtime_admission_compaction_release_epoch",
                "runtime_admission_compaction_schema_contract", "runtime_admission_run_record_schema",
                *compaction.W0_TRANSIENT_CODES, "runtime_admission_recovery_child_alive"}
    assert required <= named, sorted(required - named)


def test_the_status_and_w_vocabulary_is_the_codes():
    source = (REPO / "scripts/compaction.py").read_text()
    flat = _flat(_section(_text(), "### Paso 7: estado"))
    for value in ("not armed", "not consumed", "consumed", "yes", "no", "not run", "unreadable"):
        assert f'"{value}"' in source and f"`{value}`" in flat, value
    for key in ("armed_run", "verified", "verify_branch_reached", "phase"):
        assert f"`{key}`" in flat, key
    assert (compaction.W_EQUAL, compaction.W_LOSS_ONLY, compaction.W_RESTRICTIVE) == (
        "equal", "loss_only", "restrictive_or_unclear")
    text = _flat(_text())
    assert all(f"`{value}`" in text for value in ("equal", "loss_only", "restrictive_or_unclear", "w_sha256"))
    assert compaction.RELEASE_SMOKE_SECONDS_MAX == 60 and "N va de 10 a 60" in text


def test_the_steps_run_in_the_design_order():
    """first_transition / each_transition: step 0 read-only while H serves, stop and disable, T0 only
    with a greater triple, H served once, stop and disable again BEFORE plan, then the transition."""
    steps = _section(_text(), "## Pasos de una transición")
    order = ["$ST completion-status", "systemctl --user stop kza-voice.service",
             "systemctl --user disable kza-voice.service", "### Paso 2: ¿hace falta T0?",
             "install-controller \"$NEW_ARCHIVE\"", "ActiveState=active: H llegó a READY",
             "### Paso 4: plan", "$ST compact plan", "$ST compact start", "$ST compact verify",
             "$ST compact status", "$ST compact retire", "ActiveState=active: K' llegó a READY"]
    positions = [steps.index(marker) for marker in order]
    assert positions == sorted(positions), list(zip(order, positions))
    t0 = _section(steps, "### Paso 3: T0, instalar el controlador del NUEVO (sólo con `T0`)")
    serve_h = t0.index("ActiveState=active: H llegó a READY")
    assert t0.index("systemctl --user enable kza-voice.service") < serve_h
    assert serve_h < t0.index("systemctl --user stop kza-voice.service") < t0.index(
        "systemctl --user disable kza-voice.service")
    flat = _flat(t0)
    assert "**detener y deshabilitar otra vez antes del plan**" in flat
    assert "Esta comprobación se saltea **sólo** en dos casos" in flat
    assert ("- cuando H es un head de recuperación de una release que se sabe que falla (fila PA, más abajo)"
            in flat)
    assert ("- en la fila CI (un controlador corregido con un log del modo release pendiente, más abajo): ahí H "
            "no es el head y arrancar la unidad correría la recuperación de `serve` sobre K'.") in flat
    assert "**Sin vuelta atrás (fila T0F).**" in flat
    decide = _flat(_section(steps, "### Paso 2: ¿hace falta T0? (sólo lectura)"))
    assert 'print("T0" if new > installed else "sin T0" if new < installed or a == b else "conflicto: no seguir")' \
        in decide
    assert "Un rollback no necesita T0" in decide
    zero = _flat(_section(steps, "### Paso 0: residuo de resurrección (sólo lectura, el servicio sigue en H)"))
    assert "Cada valor de `resurrected` tiene que ser 0 (respuesta SC4B-Q2" in zero
    assert '"resurrected": resurrected' in (REPO / "scripts/managed_launcher.py").read_text()
    assert launcher.COMMANDS.count("completion-status") == 1


def test_the_gate_states_sc4b_q1_and_q2():
    gate = _flat(_section(_text(), "## Compuerta: cuándo se puede usar"))
    assert "**Primera transición dentro de los 14 días de L4 (SC4B-Q1).**" in gate
    assert "antes de DÍA 0 + 14 × 24 h y antes de que exista la intención del retiro Q5" in gate
    assert "**Paso 0 con `resurrected` en 0 (SC4B-Q2)**" in gate
    assert "Este runbook no autoriza ni programa ninguna transición" in _flat(_text())


def test_the_plan_review_names_the_closed_chroma_list():
    """SC-4c: dropped_collections per store and other_files (SC4C-Q5) are reviewed before start."""
    plan = _flat(_section(_text(), "### Paso 4: plan (sólo lectura; se puede repetir)"))
    for store, name in sorted(compaction.CHROMA_DROPPED_IN_STORE):
        assert f"`{store}`" in plan and f'`["{name}"]`' in plan, (store, name)
    for key in ("`dropped_collections`", "`other_files`", "`other_files: 0`", "`rebuilt: 1`", "`dropped: 1`"):
        assert key in plan, key
    assert "**en cada transición** «listar mis rutinas» por voz queda vacío" in plan
    source = (REPO / "scripts/compaction.py").read_text()
    assert '"dropped_collections"' in source and '"other_files"' in source


def test_h_binding_comes_from_the_intent_record_not_the_plan_view():
    """The plan view does not carry source_binding_id: the runbooks read it from the log's intent."""
    view = compaction._plan_view({"k_slot": "k", "owners": [], "members": [], "untouched": [],
                                  "untouched_sha256": "0" * 64, "coverage": {}, "coverage_members": [],
                                  "capacity": {}, "units": {}, "advisory": {}, "chroma": {}})
    assert "source_binding_id" not in view
    path = "/home/kza/releases/admission/compaction/v1/<compaction_id>/intent.json"
    assert str(stage.compaction_directory(Path("/home/kza/releases"))) + "/<compaction_id>/intent.json" == path
    for document in (RUNBOOK, COMPACTION_RUNBOOK):
        flat = _flat(document.read_text())
        assert ("print(json.load(open(sys.argv[1]))[\"source_binding_id\"])' " + path) in flat, document.name
    assert "Anotar `source_binding_id` de la salida" not in _flat(COMPACTION_RUNBOOK.read_text())


def test_retire_refusals_are_routed_by_code():
    retire = _section(_text(), "### Paso 8: retiro (la aceptación)")
    rows = {}
    for line in retire.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) == 3 and cells[0].startswith("`runtime_admission"):
            for code in _CODE.findall(cells[0]):
                rows[code] = (cells[1], cells[2])
    expected = {"runtime_admission_compaction_identity": "RF", "runtime_admission_compaction_member_missing": "RF",
                "runtime_admission_compaction_owner_only_data": "RF",
                compaction.RETIRE_PROTECTED_CHANGED: "RX", compaction.RETIRE_RESURRECTED: "RX",
                compaction.K_BINDING_CHANGED: "RX", compaction.SDK_CHANGED: "AT", compaction.H_TRANSIENT: "AT",
                compaction.K_TRANSIENT: "AT", compaction.H_CHANGED: "HD"}
    for code, row in expected.items():
        assert rows[code][0] == row, (code, rows.get(code))
    assert "--accept-owner-only-loss <coverage_sha256>" in rows["runtime_admission_compaction_owner_only_data"][1]
    for code in (compaction.RETIRE_PROTECTED_CHANGED, compaction.RETIRE_RESURRECTED, compaction.K_BINDING_CHANGED):
        assert "volver a H con W" in rows[code][1], code
    assert "reinstalar la versión ligada" in rows[compaction.SDK_CHANGED][1]
    # Review SC5B-R5: compact_retire refuses with more codes than the table routes; the last row is the
    # default for all of them.
    last = [line for line in retire.splitlines() if line.startswith("| ")][-1]
    cells = [cell.strip() for cell in last.split("|")[1:-1]]
    assert cells[0].startswith("cualquier otro código") and cells[1] == "ninguna", cells
    assert cells[2] == ("no improvisar: la unidad sigue deshabilitada; guardar la salida y la de "
                        "`$ST compact status <compaction_id>`; entregarlo a desarrollo, con la decisión escrita "
                        "cada 48 horas"), cells[2]
    for code in ("runtime_admission_compaction_phase", "runtime_admission_head_changed",
                 "runtime_admission_compaction_k_delegated", "runtime_admission_compaction_k_foreign_target",
                 "runtime_admission_compaction_database_unreadable", "runtime_admission_compaction_retiring_residue"):
        assert f"`{code}`" in cells[0], code
    # Review SC5B-R2: run_command maps every OSError to io_error, also inside _finish_retire after
    # 'retire_started', where compact_abort refuses runtime_admission_compaction_phase.
    io_row, io_exit = rows["runtime_admission_compaction_io_error"]
    assert io_row == "RS o AT", io_row
    assert ("Con `retire_started`, el retiro ya empezó (fila RS) y el abort rechaza "
            "`runtime_admission_compaction_phase`: quitar la causa del `errno` y repetir "
            "`$ST compact retire <compaction_id>` o `$ST compact resume <compaction_id>`, nunca volver a H.") in io_exit
    assert io_exit.index("Con `retire_started`") < io_exit.index("Sin `retire_started`") < io_exit.index(
        "volver a H con W")
    assert "`phases`" in io_exit
    source = (REPO / "scripts/compaction.py").read_text()
    assert '"phases": [event["kind"] for event in log["events"]]' in source
    assert 'if log["terminal"] or _has(log, "retire_started"):\n            _refuse("runtime_admission_compaction_phase")' \
        in source


def test_armed_run_not_consumed_and_the_interrupted_start():
    text = _flat(_text())
    assert ("`armed_run: not consumed` (la corrida armada no corrió: `serve` murió antes de empezarla, fila A0): "
            "arrancar la unidad otra vez para V") in text
    assert "o el abort simple `$ST compact abort <compaction_id>`, sin pérdida. `--to-h` acá rechaza " \
           "`runtime_admission_compaction_abort_to_h_before_run`" in text
    assert "**Start cortado o rechazado por el trabajador (fila S)**" in text
    assert "Después de un abort simple, habilitar y arrancar la unidad." in text


def test_re_enable_after_retire_and_after_abort():
    text = _text()
    after_retire = _section(text, "### Paso 9: servir K'")
    back = _section(text, "### Volver a H con W (filas V1, V2, V3, V4 no retirable y RX)")
    for section in (after_retire, back):
        assert section.index("systemctl --user enable kza-voice.service") < section.index(
            "systemctl --user start kza-voice.service")


def test_transient_refusals_are_retried_first_not_drift():
    at = _flat(_section(_text(), "### Rechazos transitorios (fila AT)"))
    for code in (*compaction.W0_TRANSIENT_CODES, compaction.SDK_CHANGED, compaction.H_TRANSIENT,
                 compaction.K_TRANSIENT, "runtime_admission_recovery_child_alive"):
        assert f"`{code}`" in at, code
    assert "**Un error al leer H de estos no es una deriva de H**" in at
    assert "repetir el mismo comando, que no escribió nada" in at
    assert "tratarlo como HD" in at


def test_safe_stop_follows_sc_d8_a_with_a_written_confirmation():
    ss = _section(_text(), "### Parada segura SS (`runtime_admission_compaction_abort_protected_changed`)")
    flat = _flat(ss)
    assert "**no prueba que nadie usó el sistema**" in flat
    assert "confirmar sólo por conocimiento directo de que nadie usó el sistema durante V" in flat
    assert "**lo deja por escrito** junto al digest, antes del comando" in flat
    assert "Confirmación SC-D8 A" in ss and "w_sha256 del último rechazo: <64 hex>" in ss
    assert ss.index("Confirmación SC-D8 A") < ss.index("--discard-k-changes <12+ hex del w_sha256 del último rechazo>")
    assert "**Si no puede confirmarlo, el servicio sigue apagado**" in flat
    assert "No hay otra salida de SS" in flat and "tres copias" in flat
    # Review SC5B-R8: _w_checks refuses DISCARD_MISMATCH whenever the class is equal or loss_only
    # (SC4B-N6), so 'take the latest digest' alone would loop.
    assert ("con `equal` o `loss_only` no queda nada que descartar y `--discard-k-changes` rechaza siempre: "
            "sacarlo y correr la vuelta a H sin él, `$ST compact abort <compaction_id> --to-h --confirm-h "
            "<prefijo>`.") in flat
    assert "con `equal` o `loss_only` no hay nada que descartar: repetir la vuelta a H sin `--discard-k-changes`" \
        in _flat(COMPACTION_RUNBOOK.read_text())
    assert "_refuse(DISCARD_MISMATCH, {\"w\": report, \"w_sha256\": digest})  # nothing to discard" in \
        (REPO / "scripts/compaction.py").read_text()


def test_hd_is_terminal_with_no_tool_and_no_l4_exit():
    hd = _flat(_section(_text(), "### Parada terminal HD (`runtime_admission_compaction_h_changed`)"))
    assert "**No hay salida por herramienta, tampoco L4**" in hd
    assert "`uninstall-launcher` de L4 rechazan mientras el log esté pendiente" in hd
    assert "(`runtime_admission_compaction_pending` en L4)" in hd and "repetirlos no daña" in hd
    assert "No cambiar nada bajo `/home/kza/releases` ni en `/home/kza/kza/.venv`" in hd
    assert "Entregarlo a desarrollo" in hd and "uninstall-launcher --" not in hd
    assert compaction.H_CHANGED == "runtime_admission_compaction_h_changed"


def test_the_abort_recheck_takes_the_digest_of_its_own_refusal():
    ab = _flat(_section(_text(), "### Abort cortado (fila AB)"))
    assert "`runtime_admission_compaction_abort_recheck_required`" in ab
    assert "agregarlo con el digest del **propio rechazo de esta re-comprobación**" in ab
    assert "la persona revisa el informe nuevo y confirma por escrito antes (R5-OPS-1)" in ab


def test_the_rollback_needs_no_t0():
    rollback = _flat(_section(_text(), "## Rollback (sin T0)"))
    assert "Los mismos pasos 0, 1, 4 a 9" in rollback and "**Un rollback no lleva T0**" in rollback
    assert "install-controller" in rollback and "`runtime_admission_controller_downgrade`" in rollback
    assert "Paso 3" not in rollback and "paso 3" not in rollback
    assert "R1 no es destino (SC-D7 A, `runtime_admission_compaction_release_epoch`)" in rollback
    build = _flat(_section(_text(), "## Construir NUEVO (bajo `yo`)"))
    assert "**Un arreglo después de un rollback se construye sobre la línea del controlador instalado**" in build
    assert "**toda release que cambia `src` cambia el triple y necesita un T0 con `CONTROLLER_REVISION` subido.**" \
        in build
    assert "**Conservar cada archivo de served(root), con su sha256" in build


def test_the_venv_rule_and_the_sdk_changed_exit():
    rules = _flat(_section(_text(), "## Reglas que valen en todo el procedimiento"))
    assert "no cambiar `/home/kza/kza/.venv` (nada de `pip install` ni `pip upgrade`)" in rules
    assert "La única excepción es la salida de `sdk_changed`: reinstalar la versión ligada" in rules
    assert "**Nunca `adopt` con un log del modo release pendiente.**" in rules
    assert "**Nunca `adopt` durante una transición.**" not in rules  # review SC5B-R3: T0F runs it before plan
    assert "nunca el `$ST` instalado de época 4 para el T0" in rules


def test_every_disabled_period_has_the_48_hour_decision():
    text = _text()
    rule = _flat(_section(text, "## Decisión escrita cada 48 horas"))
    assert "Para **todo** período con la unidad deshabilitada" in rule
    # Mutant M5 (review of this unit): the causes are pinned as the rule's own list, not as words that the
    # template below also contains.
    assert ("sea cual sea la causa: SS, HD, CI, T0F, una release aceptada rota que espera un hotfix (PA), un paso "
            "que no terminó, o una parada del hueco posterior a R1.") in rule
    for heading in ("### Parada segura SS (`runtime_admission_compaction_abort_protected_changed`)",
                    "### Parada terminal HD (`runtime_admission_compaction_h_changed`)",
                    "### Controlador corregido con K' abierto (fila CI)", "## Hueco posterior a R1 (SC-D5 A)"):
        assert "48 horas" in _flat(_section(text, heading)), heading
    t0f = _flat(text.split("**Sin vuelta atrás (fila T0F).**", 1)[1].split("### Paso 4", 1)[0])
    assert "regla de 48 horas" in t0f and "L4 mientras exista" in t0f
    assert "head de H abierto" in t0f and "`$ST adopt --dry-run`" in t0f
    assert "Borrados por plazo (audio ambiental" in text and "Próxima revisión: <fecha + 48 h>" in text


def test_the_post_r1_gap_stops_the_service():
    gap = _flat(_section(_text(), "## Hueco posterior a R1 (SC-D5 A)"))
    assert ("**Si en ese hueco aparece un problema grave de privacidad, detener y deshabilitar la unidad y dejar "
            "el servicio caído**") in gap


def test_after_acceptance_states_what_the_code_does_today(monkeypatch):
    """PA's recovery-head source and --accept-regressed-source are not implemented (fail closed), and
    adopt --dry-run still copies to the system temporary directory: the runbook says so, and these
    pins fail when the code changes, so the runbook is updated with it."""
    after = _flat(_section(_text(), "## Después de la aceptación"))
    assert "**Hoy un head cerrado por recuperación no es fuente de ninguna transición**" in after
    assert "la salida de PA es **L4 mientras exista** (SC-D7 A, SC4B-Q1)" in after
    assert "`kza-adopt-dry-run-*` bajo `$TMPDIR` o `/tmp`" in after
    # Review SC5B-R3: with no pending log adopt is admitted, so the dry run also runs in T0F and on
    # the T0's not_closed refusal, during the first transition; the cleanup sits where it runs.
    assert "Sólo corre después de la aceptación" not in after
    assert ("también durante la primera transición, antes del plan, en la salida T0F (punto 2) y en el rechazo "
            "`runtime_admission_not_closed` del T0") in after
    t0f = _flat(_text().split("**Sin vuelta atrás (fila T0F).**", 1)[1].split("### Paso 4", 1)[0])
    assert 'ls -d "${TMPDIR:-/tmp}"/kza-adopt-dry-run-*' in t0f
    assert t0f.index("`$ST adopt --dry-run`") < t0f.index('ls -d "${TMPDIR:-/tmp}"/kza-adopt-dry-run-*') < \
        t0f.index("borrarla a mano")
    assert "release_mode_log(admission.compaction)" in (REPO / "scripts/managed_launcher.py").read_text()
    assert 'prefix="kza-adopt-dry-run-"' in (REPO / "scripts/managed_launcher.py").read_text()
    for relative in SOURCES:
        assert "accept-regressed-source" not in (REPO / relative).read_text(), relative
    with pytest.raises(pytest.fail.Exception):
        _parse(monkeypatch, ["compact", "plan", "k", "--archive", "a", "--accept-regressed-source", "abc"])
    assert "`adopt --confirm`** pide el `head_sha256` que imprime `adopt --dry-run`" in after
    assert '"head_sha256": head_sha' in (REPO / "scripts/managed_launcher.py").read_text()


def test_a_delegation_record_after_t0_is_split_record_only_or_owners_in_head():
    split = _flat(_section(_text(), "## Registro de delegación después del T0 (filas DL y DO)"))
    assert "**Sólo el registro (fila DL)**" in split and "Quitar las entradas de `admission/delegations/`" in split
    assert "**Dueños en el head (fila DO)**" in split and "**No quitar el registro**" in split
    assert "`runtime_admission_owners_present`" in split and "Ante la duda, tratarlo como DO." in split


def test_deployment_points_to_the_runbook_and_drops_the_owner_limit_text():
    text = DEPLOYMENT.read_text()
    assert "### Procedimiento del modo release (SC-5b) — runbook, sin release" in text
    assert "(../runbooks/2026-11-17-release-mode.md)" in text
    flat = _flat(text)
    assert "La constante del límite sigue en el código" not in flat
    assert "SC-5b retiró del código la constante del límite" in flat
    for item in ("**Punto de aceptación.**", "**Reparaciones que V no ejercita (SC-D6 A).**",
                 "**`home_assistant_routines` se descarta en cada transición**", "**Colas de borrado de audio.**",
                 "**Raíces con dueños sólo en ensayos.**", "**Residuos aceptados.**"):
        assert item in flat, item


def test_the_ci_install_does_not_serve_h():
    """Review SC5B-R1 (mutant R16): under CI the log is pending and H is not the head; starting the unit
    runs serve's recovery on K' before the compaction gate refuses, which moves the head."""
    ci = _section(_text(), "### Controlador corregido con K' abierto (fila CI)")
    flat = _flat(ci)
    assert "**No es un T0 completo: no hay comprobación de que H sirve.**" in flat
    assert "La unidad sigue detenida y deshabilitada y no se arranca después de la instalación." in flat
    assert "`runtime_admission_compaction_abort_recheck_required`" in flat
    assert "K' recibiría escrituras que nadie revisó" in flat
    assert "systemctl" not in ci and "paso 3" not in flat and "como un T0" not in flat
    assert "H sirve (" not in flat and "se comprueba que H sirve" not in flat


def test_a_link_refusal_at_retire_repeats_the_retire_not_resume():
    """Review SC5B-R6: _pre_retire raises symlink/hard_link/identity before 'retire_started', where
    compact_resume with 'activated' logged returns a no-op."""
    retire = _flat(_section(_text(), "### Paso 8: retiro (la aceptación)"))
    assert ("quitar la causa a mano (de un enlace, el enlace, nunca su destino) y repetir el mismo "
            "`$ST compact retire <compaction_id>`, no `compact resume`") in retire
    assert "antes de `retire_started`, `resume` no hace nada" in retire
    source = (REPO / "scripts/compaction.py").read_text()
    assert 'return {"compaction_id": compaction_id, "phase": "activated", "noop": True}' in source


def test_a_deferred_verified_goes_to_retire_first():
    """Review SC5B-R7: serve logs 'compaction verified deferred' when recording 'verified' fails after the
    seal; compact retire records it itself (or refuses not_verified, row V3), so it runs before abort."""
    status = _flat(_section(_text(), "### Paso 7: estado"))
    bullet = "- `armed_run: consumed`, `verify_branch_reached: yes` y `verified: false`:"
    assert bullet in status
    assert status.index(bullet) < status.index("- Cualquier otro resultado")
    assert ("Correr primero el retiro, paso 8, no el abort: `$ST compact retire <compaction_id>` registra él "
            "mismo el `verified` si K' cerró limpio, y si no rechaza `runtime_admission_compaction_not_verified` "
            "(fila V3) sin escribir nada; recién ahí volver a H con W.") in status
    assert '"compaction verified deferred "' in (REPO / "scripts/managed_launcher.py").read_text()
    retire = (REPO / "scripts/compaction.py").read_text().split("def compact_retire(", 1)[1].split("\ndef ", 1)[0]
    assert retire.index("stage.record_compaction_verified(") < retire.index(
        '_refuse("runtime_admission_compaction_not_verified")')


def test_l4_exits_name_their_consequence_and_r_b_stops_again():
    """Review SC5B-R9: L4 brings erased data back with the legacy slot (uninstall_launcher docstring) and
    needs the R1 runbook's phrase and the user's decision; R-b on the T0 not_closed refusal serves H."""
    text = _text()
    after = _flat(_section(text, "## Después de la aceptación"))
    split = _flat(_section(text, "## Registro de delegación después del T0 (filas DL y DO)"))
    for section in (after, split):
        assert "**L4 trae de vuelta los datos borrados**" in section
        assert "pide la frase del runbook de R1 y la decisión del usuario, no del operador" in section.replace(
            "Pide la frase", "pide la frase")
    assert "Erased data comes back with the legacy slot: the user decides, not the operator." in _flat(
        (REPO / "scripts/managed_launcher.py").read_text())
    t0 = _section(text, "### Paso 3: T0, instalar el controlador del NUEVO (sólo con `T0`)")
    row = next(line for line in t0.splitlines() if line.startswith("| `runtime_admission_not_closed` |"))
    assert ("cerrarlo con R-b (arrancar la unidad una vez; R-b cierra el head y sirve H, así que después detener "
            "y deshabilitar la unidad otra vez antes de repetir `install-controller`, que rechaza mientras H "
            "corre)") in row


def test_the_stuck_compaction_exit_installs_from_the_extracted_archive():
    """Review SC5B-R10: the release-mode runbook links 'Compactación trabada' for log_invalid; at epoch 5
    the installed $ST refuses installer_mismatch, so step 3 leads with the extracted-archive form."""
    stuck = _flat(_section(COMPACTION_RUNBOOK.read_text(), "## Compactación trabada"))
    step3 = stuck.split("3. ", 1)[1].split("4. Log inválido", 1)[0]
    lead = ("La salida es un controlador corregido (decisión D4 = A), instalado desde su archivo extraído: "
            "`$PY -I -B <tmp>/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller "
            "<archivo nuevo>` funciona aunque haya una compactación pendiente o un log inválido")
    assert lead in step3
    assert ("Con un controlador de época 5 instalado, `$ST install-controller` rechaza "
            "`runtime_admission_controller_installer_mismatch` sin escribir nada") in step3
    assert step3.index(lead) < step3.index("`$ST install-controller")
    assert "«Compactación trabada» del runbook de compactación" in _flat(_text())
