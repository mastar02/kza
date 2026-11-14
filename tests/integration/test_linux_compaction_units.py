"""U3b-0c P14 under this account's real user systemd: the unit path scanner (scripts/compaction_units.py).

The unit tests drive the scanner with a fake user manager. Here the same
scan reads real ``systemctl --user`` output (list-unit-files, list-units,
show, cat) through the rehearsal guard (tests/mocks/real_systemd.py), on
test-only units ``kza-rehearsal-<token>-*`` written to the runtime unit
directory ``$XDG_RUNTIME_DIR/systemd/user`` (a directory the scanner reads)
and always removed afterwards:

(a) Environment=X=<root> and ExecStart=/bin/true ${X}/slots/m/state: E1
    substitution from the unit's own environment -> unit_path;
(b) the same unit naming a path outside the member passes (and the unit
    is in the report);
(c) StandardOutput=append:<member>/console.log, read through real show (or
    the cat fallback) -> unit_path;
(d) a timer + service pair where only the service names the member ->
    unit_path naming the service;
(e) no user bus (XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS removed from
    the environment ServiceManager builds from os.environ) ->
    scan_unavailable, never an empty listing.
(f) review r2 N4: a unit linked into the runtime unit directory from a file
    outside every unit directory (what ``systemctl --user link --runtime``
    makes), naming the member -> unit_path; its real FragmentPath and unit
    file state are recorded;
(g) review r2 N4: a template ``<prefix>-tpl@.service`` with no instance
    loaded, whose ExecStart is ``<root>/slots/%i/state`` -> unit_path on the
    template; the same template writing under /tmp passes and is reported in
    ``templates_scanned``;
(h) review r3 R3-2: the same template naming the member on a continuation
    line, or through an EnvironmentFile value used as ${SRC}, -> unit_path;
    systemd itself joins the continuation (show of an instance carries the
    member path).
(j) review r5 R5-1/R5-2: a glob over the slots in an ``sh -c`` script of a
    regular unit, a glob above the root in a template, and a template naming
    the root with one character written as a C escape (``\\xHH``, which
    systemd decodes) -> unit_path.

The scan reads every unit of this account in the user unit directories; a
refusal in (b) that names another unit is a real finding about this
account, not a flake. Nothing is started, enabled or stopped by the scan.
Skipped off Linux, without a running user manager, or as root or kza
(real_systemd.unavailable). KZA_0E4_EVIDENCE=<dir> records content-free
results. Run it alone (it daemon-reloads this account's manager).
"""
from __future__ import annotations

import os

import pytest

from scripts import compaction_units as units_module
from scripts import managed_launcher as launcher
from tests.mocks import real_systemd

pytestmark = pytest.mark.skipif(real_systemd.unavailable() is not None,
                                reason=real_systemd.unavailable() or "available")

UNAVAILABLE = "runtime_admission_compaction_unit_scan_unavailable"
PATH_HIT = "runtime_admission_compaction_unit_path:"
RESULTS: list[dict] = []
CLEANUPS: list[dict] = []


@pytest.fixture(scope="module", autouse=True)
def evidence():
    # A killed earlier run skipped its finalizers: sweep its units first.
    real_systemd.record("compaction-units-stale-sweep", real_systemd.sweep_stale())
    yield
    real_systemd.record("compaction-units-results", RESULTS)
    real_systemd.record("compaction-units-cleanup", CLEANUPS)


@pytest.fixture
def systemd(tmp_path):
    manager = real_systemd.UserSystemd(tmp_path / "systemd")
    yield manager
    result = manager.cleanup()
    CLEANUPS.append({key: value for key, value in result.items() if key not in ("before", "after")})
    assert result["rehearsal_units_after"] == [] and result["leftover_paths"] == []
    assert result["guard_breaches"] == [] and result["daemon_reload_exit"] == 0


@pytest.fixture
def layout(tmp_path):
    root = (tmp_path / "releases").resolve()
    member = root / "slots" / "m"
    (member / "state").mkdir(parents=True)
    (root / "elsewhere").mkdir()
    info = member.stat()
    return root, member, [{"path": str(member), "identity": {"device": info.st_dev, "inode": info.st_ino}}]


def _scan(systemd: real_systemd.UserSystemd, layout) -> dict:
    root, _member, members = layout
    manager = launcher.ServiceManager(executable=str(systemd.shim))
    return units_module.scan_units(units_module.strict_systemctl(manager), root=root, members=members)


def _refusal(systemd: real_systemd.UserSystemd, layout) -> str:
    with pytest.raises(units_module.UnitScanRefusal) as caught:
        _scan(systemd, layout)
    return str(caught.value)


def _service(environment: str, exec_start: str, extra: str = "") -> str:
    # No [Install]: the unit file state is 'static' (list-unit-files lists every state).
    return ("[Unit]\nDescription=KZA U3b-0c P14 rehearsal (test only)\n\n[Service]\nType=oneshot\n"
            f"Environment={environment}\nExecStart={exec_start}\n{extra}")


def test_a_variable_naming_a_member_is_refused_and_elsewhere_passes(systemd, layout):
    root, _member, _members = layout
    name = systemd.name("env")
    systemd.write_unit(name, _service(f"X={root}", "/bin/true ${X}/slots/m/state"))
    systemd.reload()
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "a_env_substitution", "code": code})
    assert code == PATH_HIT + name

    systemd.write_unit(name, _service(f"X={root}", "/bin/true ${X}/elsewhere"))
    systemd.reload()
    report = _scan(systemd, layout)
    RESULTS.append({"case": "b_elsewhere", "units_scanned": report["units_scanned"],
                    "rehearsal_units": [unit for unit in report["unit_names"] if unit.startswith(systemd.prefix)]})
    assert name in report["unit_names"]


def test_standard_output_into_a_member_is_refused(systemd, layout):
    _root, member, _members = layout
    name = systemd.name("console")
    systemd.write_unit(name, _service("A=1", "/bin/true", f"StandardOutput=append:{member}/console.log\n"))
    systemd.reload()
    shown = systemd.raw_show(name, "StandardOutput")
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "c_standard_output", "code": code,
                    "show_carries_path": shown["StandardOutput"].startswith("append:/")})
    assert code == PATH_HIT + name
    assert not (member / "console.log").exists()  # nothing was started


def test_a_timer_service_pair_is_refused_on_the_service(systemd, layout):
    root, _member, _members = layout
    service = systemd.name("tick")
    timer = systemd.name("tick", "timer")
    systemd.write_unit(service, _service(f"X={root}", "/bin/true --state ${X}/slots/m/state"))
    systemd.write_unit(timer, f"[Unit]\nDescription=KZA U3b-0c P14 rehearsal timer (test only)\n\n"
                              f"[Timer]\nOnCalendar=2099-01-01\nUnit={service}\n")
    systemd.reload()
    shown = systemd.raw_show(timer, "Triggers", "FragmentPath")
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "d_timer_service", "code": code, "timer_triggers_service": shown["Triggers"] == service})
    assert code == PATH_HIT + service


def test_without_a_user_bus_the_scan_is_unavailable(systemd, layout):
    with pytest.MonkeyPatch.context() as patch:  # restored before the fixture's cleanup needs the bus
        patch.delenv("XDG_RUNTIME_DIR", raising=False)
        patch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        assert "XDG_RUNTIME_DIR" not in os.environ
        code = _refusal(systemd, layout)
    calls = systemd.calls()
    RESULTS.append({"case": "e_no_bus", "code": code, "first_exit": calls[0]["exit"] if calls else None})
    assert code == UNAVAILABLE
    assert calls and calls[0]["exit"] != 0


def test_a_linked_unit_from_outside_the_unit_directories_is_scanned(systemd, layout, tmp_path):
    _root, member, _members = layout
    name = systemd.name("linked")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / name
    target.write_text(_service("A=1", "/bin/true", f"WorkingDirectory={member}\n"))
    (systemd.unit_dir / name).symlink_to(target)
    systemd.reload()
    shown = systemd.raw_show(name, "FragmentPath", "UnitFileState", "LoadState")
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "f_linked_unit", "code": code, "load_state": shown["LoadState"],
                    "unit_file_state": shown["UnitFileState"],
                    "fragment_is_link": shown["FragmentPath"] == str(systemd.unit_dir / name),
                    "fragment_is_target": shown["FragmentPath"] == str(target)})
    assert code == PATH_HIT + name


def test_a_template_without_instances_is_scanned_by_its_fixed_part(systemd, layout):
    root, _member, _members = layout
    name = f"{systemd.prefix}-tpl@.service"
    systemd.write_unit(name, _service("A=1", f"/bin/true {root}/slots/%i/state"))
    systemd.reload()
    listed = name in " ".join(systemd.list_unit_files())
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "g_template_under_root", "code": code, "listed_by_list_unit_files": listed})
    assert code == PATH_HIT + name
    systemd.write_unit(name, _service("A=1", "/bin/true %t/%i.out"))  # not /tmp: the test root is below it
    systemd.reload()
    report = _scan(systemd, layout)
    RESULTS.append({"case": "g_template_elsewhere", "templates": [unit for unit in report["templates_scanned"]
                                                                   if unit.startswith(systemd.prefix)]})
    assert name in report["templates_scanned"]


@pytest.mark.parametrize("form", ["continuation", "environment_file"])
def test_a_template_naming_a_member_through_a_continuation_or_an_environment_file(systemd, layout, tmp_path, form):
    _root, member, _members = layout
    name = f"{systemd.prefix}-r3{form[:4]}@.service"
    if form == "continuation":
        body = f"[Service]\nType=oneshot\nExecStart=/bin/true %t/%i \\\n    {member}/state\n"  # not /tmp: above the root
    else:
        env_file = tmp_path / "r3.env"
        env_file.write_text(f"SRC={member}/state\n")
        body = f"[Service]\nType=oneshot\nEnvironmentFile={env_file}\nExecStart=/bin/true ${{SRC}}\n"
    systemd.write_unit(name, body)
    systemd.reload()
    instance = name.replace("@.", "@x.")
    shown = systemd.raw_show(instance, "ExecStart")
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "h_template_" + form, "code": code,
                    "instance_show_names_member": str(member) in shown["ExecStart"]})
    assert code == PATH_HIT + name


@pytest.mark.parametrize("form", ["sh_c_root_instance", "program_reads_file_value"])
def test_a_template_naming_the_root_in_a_script_or_a_member_in_a_file_value(systemd, layout, tmp_path, form):
    """Review r4 R4-1 (probe_templates4 S2, E3) under real systemd 255: 'sh -c "rm -rf <root>/%I"' and an
    EnvironmentFile value only the program reads both refuse unit_path."""
    root, member, _members = layout
    name = f"{systemd.prefix}-r4{form[:4]}@.service"
    if form == "sh_c_root_instance":
        body = f"[Service]\nType=oneshot\nExecStart=/bin/sh -c 'echo rm -rf {root}/%I'\n"
    else:
        env_file = tmp_path / "r4.env"
        env_file.write_text(f"SRC={member}/state\n")
        body = f"[Service]\nType=oneshot\nEnvironmentFile={env_file}\nExecStart=/usr/bin/env\n"
    systemd.write_unit(name, body)
    systemd.reload()
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "i_template_" + form, "code": code})
    assert code == PATH_HIT + name


_R5_ROLES = {"regular_sh_c_glob": "r5glob", "template_glob_above_root": "r5tglob", "template_c_escape": "r5tesc"}


@pytest.mark.parametrize("form", list(_R5_ROLES))
def test_a_glob_or_a_c_escape_reaching_the_slots_refuses(systemd, layout, form):
    """Review r5 R5-1/R5-2 under real systemd 255: 'sh -c "tar c <root>/slots/*/state"' in a regular unit,
    'sh -c "rm -rf <parent>/rel*/slots/%i"' in a template, and a template naming <root> with its last
    character written as '\\xHH' all refuse unit_path. systemd decodes the escape itself: the show of an
    instance carries the plain root."""
    root, _member, _members = layout
    if form == "regular_sh_c_glob":
        name = systemd.name(_R5_ROLES[form])
        systemd.write_unit(name, _service("A=1", f"/bin/sh -c 'echo tar czf /dev/null {root}/slots/*/state'"))
        shown_unit = name
    else:
        name = f"{systemd.prefix}-{_R5_ROLES[form]}@.service"
        if form == "template_glob_above_root":
            word = f"{root.parent}/rel*/slots/%i"
        else:
            text = str(root)
            word = text[:-1] + "\\x%02x" % ord(text[-1]) + "/slots/%i"
        systemd.write_unit(name, f"[Service]\nType=oneshot\nExecStart=/bin/sh -c 'echo rm -rf {word}'\n")
        shown_unit = name.replace("@.", "@x.")
    systemd.reload()
    shown = systemd.raw_show(shown_unit, "ExecStart", "LoadState")
    code = _refusal(systemd, layout)
    RESULTS.append({"case": "j_" + form, "code": code, "load_state": shown["LoadState"],
                    "show_names_root_slots": f"{root}/slots/" in shown["ExecStart"]})
    assert code == PATH_HIT + name
    if form == "template_c_escape":
        assert f"{root}/slots/x" in shown["ExecStart"]  # decoded by systemd, %i expanded
