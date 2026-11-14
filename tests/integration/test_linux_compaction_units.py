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
    # No [Install]: the unit file state is 'static', listed by list-unit-files --state=...static.
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
