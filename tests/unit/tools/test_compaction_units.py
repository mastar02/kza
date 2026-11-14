"""U3b-0c P14: the user-unit and logrotate path scanner (scripts/compaction_units.py).

Owner-compaction design v4 F2, v5 G1 (tests list, mutants) and the v5 review
findings E1, E2 and E4. systemctl is a fake user manager answering the
strict adapter (``strict_systemctl``); nothing here runs the real systemctl.
"""
from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from scripts import compaction_units as units_module
from scripts import managed_launcher as launcher
from tests.mocks import fake_systemd

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template"
UID = 4242
UNAVAILABLE = "runtime_admission_compaction_unit_scan_unavailable"
UNPARSED = "runtime_admission_compaction_unit_unparsed:"
PATH_HIT = "runtime_admission_compaction_unit_path:"
DEFAULTS = {"StandardOutput": "journal", "StandardError": "inherit"}


def exec_value(*argv: str) -> str:
    """A ``systemctl show`` Exec* value: argv joined with spaces and no quoting, as systemd prints it."""
    return ("{ path=" + argv[0] + " ; argv[]=" + " ".join(argv)
            + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")


def unit(fragment: str, *, state: str | None = "static", loaded: bool = False, cat: str | None = None,
         **props) -> dict:
    props.setdefault("FragmentPath", fragment)
    return {"state": state, "loaded": loaded, "props": props, "cat": cat}


class FakeUserManager:
    """ServiceManager._run's contract over a dict of units; ``fail`` makes a verb (or verb, unit) exit non-zero."""

    def __init__(self, units: dict, fail: tuple = (), manager_environment: dict | None = None) -> None:
        self.units = units
        self.fail = fail
        self.manager_environment = {"HOME": "/home/kza", "PATH": "/usr/bin:/bin"} \
            if manager_environment is None else manager_environment
        self.calls: list[tuple] = []

    def _run(self, *arguments: str, timeout: float | None = None, strict: bool = False) -> str | None:
        self.calls.append(arguments)
        assert arguments[0] == "--user" and strict, arguments
        verb = arguments[1]
        target = arguments[2] if len(arguments) > 2 else None
        if verb in self.fail or (verb, target) in self.fail:
            return None
        if verb == "show-environment":
            return "".join(f"{name}={value}\n" for name, value in self.manager_environment.items())
        if verb == "list-unit-files":
            assert not any(word.startswith("--state") for word in arguments), arguments  # every state
            return "".join(f"{name} {item['state']} enabled\n" for name, item in self.units.items()
                           if item["state"] is not None)
        if verb == "list-units":
            assert "--all" in arguments
            return "".join(f"● {name} loaded failed failed test\n" for name, item in self.units.items()
                           if item["loaded"])
        if verb == "show":
            names = [arguments[index + 1] for index, word in enumerate(arguments) if word == "-p"]
            item = self.units.get(target)
            values = dict(DEFAULTS, LoadState="loaded" if item else "not-found")
            values.update(item["props"] if item else {})
            lines = []
            for name in names:
                value = values.get(name, "")
                lines += [f"{name}={entry}" for entry in (value if isinstance(value, list) else [value])]
            return "\n".join(lines) + "\n"
        if verb == "cat":
            item = self.units.get(target)
            if item is None:
                return None
            return item["cat"] if item["cat"] is not None else f"# {item['props']['FragmentPath']}\n[Service]\n"
        raise AssertionError(arguments)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    base = tmp_path.resolve()
    # The account's home, as for kza in production: kza_markers reads '~' (U3b-0e4 rule), so a
    # unit naming <home>/kza or 'kza' relative to it is kza's.
    monkeypatch.setenv("HOME", str(base / "home"))
    root = base / "releases"
    member = root / "slots" / "m1"
    (member / "state" / "logs").mkdir(parents=True)
    (member / "state" / "logs" / "console.log").write_text("private\n")
    (member / "state" / "x.conf").write_text("/tmp/never-read.log {\n}\n")
    (root / "slots" / "keep" / "state").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "logs" / "console.log").write_text("")
    home = base / "home"
    user_dir = home / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True)
    info = member.stat()
    members = [{"path": str(member), "identity": {"device": info.st_dev, "inode": info.st_ino}}]
    return SimpleNamespace(base=base, root=root, member=member, home=home, user_dir=user_dir, members=members)


def run_scan(layout, units: dict, *, fail: tuple = (), extra: tuple = (), members: list | None = None,
             environ: dict | None = None, manager_environment: dict | None = None, named: tuple = ()) -> dict:
    manager = FakeUserManager(units, fail, manager_environment)
    return units_module.scan_units(units_module.strict_systemctl(manager), root=layout.root,
                                   members=layout.members if members is None else members, home=layout.home,
                                   uid=UID, extra_logrotate=extra, environ=environ or {}, named=named)


def refused(layout, units: dict, **options) -> str:
    with pytest.raises(units_module.UnitScanRefusal) as caught:
        run_scan(layout, units, **options)
    assert isinstance(caught.value, ValueError)
    return str(caught.value)


def ours(layout, name: str, **props) -> dict:
    return {name: unit(str(layout.user_dir / name), **props)}


def rotate_units(layout, conf: Path) -> dict:
    """kza-console-rotate.timer -> .service running logrotate on ``conf`` (the console rotation)."""
    found = ours(layout, "kza-console-rotate.timer", Unit="kza-console-rotate.service",
                 Triggers="kza-console-rotate.service")
    found.update(ours(layout, "kza-console-rotate.service", ExecStart=exec_value(
        "/usr/sbin/logrotate", "-s", str(layout.root / "logrotate.state"), str(conf))))
    return found


# --- the launcher's own unit (E1) ------------------------------------------------------------------

def _managed_units(layout) -> tuple[dict, Path]:
    console = layout.root / "logs" / "console.log"
    settings = launcher.LauncherSettings(python="/usr/bin/python3", stop_grace=120, timeout_start=240,
                                         timeout_stop=200, watchdog_sec=0, console_log=str(console))
    managed = launcher.render_managed_conf(TEMPLATE.read_text(), layout.root, settings)
    secrets = layout.home / "secrets"
    secrets.mkdir()
    (secrets / ".env").write_text("HOME_ASSISTANT_URL=http://192.168.1.100:8123\nHOME_ASSISTANT_TOKEN=abc/def\n")
    base = ("[Service]\nType=simple\nWorkingDirectory=/home/kza/app\nEnvironment=PYTHONUNBUFFERED=1\n"
            "ExecStart=/home/kza/app/.venv/bin/python -m src.main\n")
    directives = fake_systemd.parse_dropins([base, managed])
    dropin = layout.user_dir / "kza-voice.service.d" / "managed.conf"
    found = ours(layout, "kza-voice.service", state="enabled", loaded=True,
                 DropInPaths=str(dropin), WorkingDirectory=directives["WorkingDirectory"],
                 Environment=shlex.join(directives["Environment"]),
                 EnvironmentFiles=[f"{secrets / '.env'} (ignore_errors=no)",
                                   f"{secrets / 'llama-api-key.env'} (ignore_errors=yes)"],
                 ExecStart=exec_value(*shlex.split(directives["ExecStart"][-1])),
                 StandardOutput=directives["StandardOutput"], StandardError=directives["StandardError"])
    conf = layout.root / launcher.LOGROTATE_NAME
    conf.write_text(launcher.render_console_logrotate(str(console)))
    found.update(rotate_units(layout, conf))
    return found, conf


def test_the_rendered_managed_conf_passes(layout):
    """E1: WorkingDirectory=<root> and --root <root> contain every slot but are not references to one."""
    found, conf = _managed_units(layout)
    assert "${" not in found["kza-voice.service"]["props"]["ExecStart"]
    report = run_scan(layout, found, extra=(conf,))
    assert report == {"units_scanned": 3, "unit_names": ["kza-console-rotate.service", "kza-console-rotate.timer",
                                                         "kza-voice.service"], "logrotate_files": 1,
                      "tolerated_unresolved": [], "templates_scanned": []}


def test_a_variable_from_environment_naming_a_member_is_a_path_hit(layout):
    found = ours(layout, "kza-tool.service", Environment=f"X={layout.root}",
                 ExecStart=exec_value("/bin/tool", "${X}/slots/m1"))
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


def test_a_variable_from_an_environment_file_naming_a_member_is_a_path_hit(layout):
    env = layout.home / "tool.env"
    env.write_text(f"X={layout.root}\n")
    found = ours(layout, "kza-tool.service", EnvironmentFiles=f"{env} (ignore_errors=no)",
                 ExecStart=exec_value("/bin/tool", "--data", "${X}/slots/m1/state"))
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


def test_substitution_is_applied_not_refused(layout):
    """Mutant 'skip E1 substitution (refuse every $VAR)': a resolved path elsewhere passes."""
    found = ours(layout, "kza-tool.service", Environment=f"X={layout.root}",
                 ExecStart=exec_value("/bin/tool", "${X}/logs", "$X"))
    assert run_scan(layout, found)["unit_names"] == ["kza-tool.service"]


@pytest.mark.parametrize("argv", [("/bin/tool", "${Y}/x"), ("/bin/tool", "--root", "$Y"),
                                  ("/bin/tool", "--config=$Y"), ("/bin/tool", "/var/log/%Y.log")])
def test_an_unresolved_reference_in_a_path_token_is_unparsed(layout, argv):
    # Named: a unit's own name is not a kza path (review r2 N6), so only naming it keeps it strict.
    found = ours(layout, "kza-tool.service", ExecStart=exec_value(*argv))
    assert refused(layout, found, named=("kza-tool.service",)) == UNPARSED + "kza-tool.service:ExecStart"


@pytest.mark.parametrize("token", ["$Y", "${Z}"])
def test_an_unresolved_bare_reference_is_unparsed(layout, token):
    """Review F9: '/usr/bin/rsync -a $SRC /elsewhere' could reach a member through the variable."""
    found = ours(layout, "kza-tool.service", ExecStartPre=exec_value("/usr/bin/rsync", "-a", token, "/srv/x"))
    assert refused(layout, found, named=("kza-tool.service",)) == UNPARSED + "kza-tool.service:ExecStartPre"


def test_a_plain_specifier_and_systemd_provided_variables_pass(layout):
    found = ours(layout, "kza-tool.service", ExecStartPre=exec_value("/bin/echo", "+%Y"),
                 ExecReload=exec_value("/bin/kill", "-HUP", "$MAINPID"))
    assert run_scan(layout, found)["units_scanned"] == 1


def test_a_manager_environment_variable_naming_a_member_is_a_path_hit(layout):
    """Review F9 / r2 N5: set-environment / the session import name a member: refused as the block."""
    found = ours(layout, "kza-tool.service", ExecStart=exec_value("/usr/bin/cp", "-r", "${KZA_MEMBER}", "/srv/x"))
    hit = {"HOME": str(layout.home), "KZA_MEMBER": str(layout.member)}
    assert refused(layout, found, manager_environment=hit) == PATH_HIT + "show-environment"
    elsewhere = {"HOME": str(layout.home), "KZA_MEMBER": str(layout.home / "elsewhere")}
    assert run_scan(layout, found, manager_environment=elsewhere)["units_scanned"] == 1


def test_a_resolved_manager_reference_is_checked_where_the_unit_uses_it(layout):
    """A manager value outside every member, made a member path by the unit's own suffix. (Review r4: a
    manager value naming the root itself refuses on its own, as the block.)"""
    suffix = "${BASE}/" + layout.root.name + "/slots/m1"
    found = ours(layout, "kza-tool.service", ExecStart=exec_value("/usr/bin/cp", "-r", suffix, "/srv/x"))
    assert refused(layout, found, manager_environment={"BASE": str(layout.root.parent)}) == \
        PATH_HIT + "kza-tool.service"
    assert refused(layout, found, manager_environment={"BASE": str(layout.root)}) == PATH_HIT + "show-environment"


@pytest.mark.parametrize("how", ["dollar_dollar", "program_reads_it", "unit_overrides_it"])
def test_a_manager_environment_value_naming_a_member_refuses_whoever_reads_it(layout, how):
    """Review r2 N5 (probe_p14): '$${KZA_STATE}' (a shell reads it) or a program reading the variable
    itself never shows the value in unit text; the value is held to the unit Environment= rule."""
    argv = {"dollar_dollar": ("/bin/sh", "-c", "cp", "-a", "$${KZA_STATE}", "/tmp/export"),
            "program_reads_it": ("/usr/bin/python3", "/opt/tool/export.py"),
            "unit_overrides_it": ("/bin/tool", "$KZA_STATE")}[how]
    props = {"Environment": f"KZA_STATE={layout.home}"} if how == "unit_overrides_it" else {}
    found = ours(layout, "kza-export.service", ExecStart=exec_value(*argv), **props)
    environment = {"HOME": str(layout.home), "PATH": "/usr/bin", "KZA_STATE": str(layout.member / "state")}
    assert refused(layout, found, manager_environment=environment) == PATH_HIT + "show-environment"
    environment["KZA_STATE"] = str(layout.home / "export")
    assert run_scan(layout, found, manager_environment=environment)["unit_names"] == ["kza-export.service"]


def test_an_escaped_manager_value_stays_unresolved(layout):
    found = ours(layout, "kza-tool.service", ExecStart=exec_value("/bin/tool", "$Q"))
    assert refused(layout, found, manager_environment={"Q": "$'a\\nb'"}, named=("kza-tool.service",)) == \
        UNPARSED + "kza-tool.service:ExecStart"


def test_a_malformed_environment_block_refuses(layout):
    found = ours(layout, "kza-tool.service", ExecStart=exec_value("/bin/tool"))
    assert refused(layout, found, manager_environment={"not a name": "x"}) == UNAVAILABLE


# --- the U3b-0e4 rule for stock units (coordinator follow-up to review F9) ---------------------------

def _shown(path: str, argv: str) -> str:
    """A real ``systemctl show`` Exec* value (argv as systemd joins it), copied from kza-yo-lan1."""
    return ("{ path=" + path + " ; argv[]=" + argv
            + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")


_UNITY_TEXT = """[Unit]
Description=Unity GTK Module Environment variables
Before=unity7.service unity-panel-service.service gnome-session.service
PartOf=graphical-session.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -ec '\\
        GTK_MODULES="$${GTK_MODULES:+$GTK_MODULES:}unity-gtk-module";\\
        dbus-update-activation-environment --verbose --systemd GTK_MODULES'
ExecStopPost=/bin/sh -ec '\\
        GTK_MODULES=$$(echo -n $${GTK_MODULES} | awk -v RS=: -v ORS=: "/^unity-gtk-module$/ {next} {print}" | sed -e "s/:*$//");\\
        dbus-update-activation-environment --verbose --systemd GTK_MODULES'

[Install]
WantedBy=unity-session.target
"""


def _stock_units(layout, *, podman_environment: bool = True) -> dict:
    """The five stock units kza's manager has, each with a user drop-in so that P14 scans it.

    Exec lines and texts are the real ones (Ubuntu 24.04, read on kza-yo-lan1). podman's
    ExecStart passes $LOGGING to 'podman system service'; its unit sets LOGGING, which the
    coordinator's scan of kza did not resolve (``podman_environment=False`` is that reading).
    """
    found = {}

    def stock(name: str, text: str, **props) -> None:
        dropin = layout.user_dir / (name + ".d") / "override.conf"
        cat = f"# /usr/lib/systemd/user/{name}\n{text}\n# {dropin}\n[Service]\nNice=5\n"
        found[name] = unit(f"/usr/lib/systemd/user/{name}", loaded=True, cat=cat, DropInPaths=str(dropin), **props)

    stock("org.freedesktop.IBus.session.generic.service",
          "[Service]\nExecStart=sh -c 'exec /usr/bin/ibus-daemon $IBUS_DAEMON_ARGS'\n",
          ExecStart=_shown("sh", "sh -c exec /usr/bin/ibus-daemon $IBUS_DAEMON_ARGS"))
    stock("org.freedesktop.IBus.session.GNOME.service",
          "[Service]\nExecStart=sh -c 'exec /usr/bin/ibus-daemon --panel disable "
          "$([ \"$XDG_SESSION_TYPE\" = \"x11\" ] && echo \"--xim\")'\n",
          ExecStart=_shown("sh", 'sh -c exec /usr/bin/ibus-daemon --panel disable '
                                 '$([ "$XDG_SESSION_TYPE" = "x11" ] && echo "--xim")'))
    logging = {"Environment": "LOGGING=--log-level=info"} if podman_environment else {}
    stock("podman.service",
          "[Service]\nEnvironment=LOGGING=\"--log-level=info\"\nExecStart=/usr/bin/podman $LOGGING system service\n",
          ExecStart=_shown("/usr/bin/podman", "/usr/bin/podman $LOGGING system service"), **logging)
    stock("podman-restart.service",
          "[Service]\nEnvironment=LOGGING=\"--log-level=info\"\n"
          "ExecStart=/usr/bin/podman $LOGGING start --all --filter restart-policy=always\n"
          "ExecStop=/bin/sh -c '/usr/bin/podman $LOGGING stop $(/usr/bin/podman container ls "
          "--filter restart-policy=always -q)'\n",
          ExecStart=_shown("/usr/bin/podman", "/usr/bin/podman $LOGGING start --all --filter restart-policy=always"),
          ExecStop=_shown("/bin/sh", "/bin/sh -c /usr/bin/podman $LOGGING stop $(/usr/bin/podman container ls "
                                     "--filter restart-policy=always -q)"), **logging)
    stock("unity-gtk-module.service", _UNITY_TEXT,
          ExecStart=_shown("/bin/sh", '/bin/sh -ec          GTK_MODULES="$${GTK_MODULES:+$GTK_MODULES:}unity-gtk-module";'
                                      '         dbus-update-activation-environment --verbose --systemd GTK_MODULES'),
          ExecStopPost=_shown("/bin/sh", '/bin/sh -ec          GTK_MODULES=$$(echo -n $${GTK_MODULES} | awk -v RS=: '
                                         '-v ORS=: "/^unity-gtk-module$/ {next} {print}" | sed -e "s/:*$//");'
                                         '         dbus-update-activation-environment --verbose --systemd GTK_MODULES'))
    return found


# kza's real manager environment keys (coordinator's read-only scan, values made up).
KZA_MANAGER = {"HOME": "/home/kza", "LANG": "C.UTF-8", "LOGNAME": "kza", "PATH": "/usr/bin:/bin", "SHELL": "/bin/bash",
               "USER": "kza", "XDG_RUNTIME_DIR": "/run/user/1001", "QT_ACCESSIBILITY": "1",
               "QTWEBENGINE_DICTIONARIES_PATH": "/usr/share/hunspell-bdic/", "XDG_DATA_DIRS": "/usr/share",
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus", "GSM_SKIP_SSH_AGENT_WORKAROUND": "true",
               "SSH_AUTH_SOCK": "/run/user/1001/keyring/ssh"}


@pytest.mark.parametrize("podman_environment", [True, False])
def test_the_five_stock_units_pass_under_the_0e4_rule(layout, podman_environment):
    """Coordinator scan of kza: IBus generic/GNOME, podman, podman-restart, unity-gtk-module."""
    found, conf = _managed_units(layout)
    found.update(_stock_units(layout, podman_environment=podman_environment))
    report = run_scan(layout, found, extra=(conf,), manager_environment=KZA_MANAGER)
    stock = {"org.freedesktop.IBus.session.generic.service", "org.freedesktop.IBus.session.GNOME.service",
             "unity-gtk-module.service"} | (set() if podman_environment else {"podman.service",
                                                                              "podman-restart.service"})
    assert set(report["tolerated_unresolved"]) == stock
    assert {"kza-voice.service", "podman.service", "unity-gtk-module.service"} <= set(report["unit_names"])


@pytest.mark.parametrize("name", ["kza-voice.service", "kza-llm-fast.service", "kza-code-index.service",
                                  "listed-by-operator.service"])
def test_a_named_unit_with_an_unresolved_reference_still_refuses(layout, name):
    found = {name: unit(str(layout.user_dir / name), loaded=True, cat="[Service]\nExecStart=/bin/tool $Y\n",
                        ExecStart=exec_value("/bin/tool", "$Y"))}
    named = ("listed-by-operator.service",) if name == "listed-by-operator.service" else ()
    assert refused(layout, found, named=named) == UNPARSED + name + ":ExecStart"


def test_a_unit_the_operator_names_must_exist(layout):
    """Review r2 N4: '--unit' with a name the manager does not know (a typo) is not a pass."""
    assert refused(layout, {}, named=("kza-typo.service",)) == UNPARSED + "kza-typo.service:LoadState"


@pytest.mark.parametrize("where", ["text_member", "text_root", "environment_root", "text_kza_home"])
def test_an_unnamed_unit_mentioning_kza_a_member_or_the_root_still_refuses(layout, where):
    text = "[Service]\nExecStart=/bin/tool $Y\n"
    props = {"ExecStart": exec_value("/bin/tool", "$Y")}
    if where == "text_member":
        text += f"# see {layout.member}\n"
    elif where == "text_root":
        text += f"WorkingDirectory=/srv\n# {layout.root}/logs\n"
    elif where == "environment_root":
        props["Environment"] = f"BASE={layout.root}/logs"
    else:
        text += f"# copied from {layout.home / 'kza'}/systemd\n"
    found = {"stock.service": unit("/usr/lib/systemd/user/stock.service", loaded=True, cat=text,
                                   DropInPaths=str(layout.user_dir / "stock.service.d" / "x.conf"), **props)}
    assert refused(layout, found) == UNPARSED + "stock.service:ExecStart"


def test_dollar_dollar_is_a_literal_dollar(layout):
    found = ours(layout, "kza-tool.service", ExecStart=exec_value("/bin/sh", "-c", "echo", "$$HOME/x"))
    run_scan(layout, found)


def test_a_missing_required_environment_file_is_unparsed_and_an_optional_one_is_not(layout):
    missing = layout.home / "missing.env"
    found = ours(layout, "kza-tool.service", EnvironmentFiles=f"{missing} (ignore_errors=yes)")
    run_scan(layout, found)
    found = ours(layout, "kza-tool.service", EnvironmentFiles=f"{missing} (ignore_errors=no)")
    assert refused(layout, found) == UNPARSED + "kza-tool.service:EnvironmentFiles"


def test_an_environment_file_inside_a_member_is_refused_before_it_is_opened(layout, monkeypatch):
    opened = _instrument(monkeypatch)
    inside = layout.member / "state" / "x.conf"
    found = ours(layout, "kza-tool.service", EnvironmentFiles=f"-{inside} (ignore_errors=yes)")
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"
    assert not [path for path in opened if _under(path, layout.member)]


def test_a_semicolon_inside_an_argument_does_not_hide_the_rest(layout):
    found = ours(layout, "kza-tool.service", ExecStart=exec_value(
        "/bin/sh", "-c", "cd /tmp ;", "cat", f"{layout.member}/state/x.conf"))
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


# --- stdout/stderr ------------------------------------------------------------------------------------

def test_a_cat_read_specifier_is_unparsed(layout):
    found = ours(layout, "kza-tool.service", StandardOutput="append",
                 cat="# x\n[Service]\nStandardOutput=append:%h/releases/slots/m1/state/logs/console.log\n")
    assert refused(layout, found) == UNPARSED + "kza-tool.service:StandardOutput"


def test_a_shown_absolute_output_into_a_member_is_a_path_hit(layout):
    found = ours(layout, "kza-tool.service",
                 StandardOutput=f"append:{layout.member}/state/logs/console.log",
                 cat="# x\n[Service]\nStandardOutput=append:%h/releases/slots/m1/state/logs/console.log\n")
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


def test_the_last_cat_assignment_wins_and_empty_resets(layout):
    text = ("# /u/kza-tool.service\n[Service]\nStandardError=append:%h/x.log\n"
            "# /u/kza-tool.service.d/override.conf\n[Service]\nStandardError=\n")
    run_scan(layout, ours(layout, "kza-tool.service", cat=text))
    text = text + f"# /u/kza-tool.service.d/zz.conf\n[Service]\nStandardError=file:{layout.member}/e.log\n"
    assert refused(layout, ours(layout, "kza-tool.service", cat=text)) == PATH_HIT + "kza-tool.service"


# --- resolution (rule 5) ----------------------------------------------------------------------------

def test_a_symlink_into_a_member_is_a_path_hit_that_literal_comparison_misses(layout):
    """Mutant 'literal-path-only comparison': /x/console -> <member>/state/logs."""
    link_parent = layout.base / "x"
    link_parent.mkdir()
    (link_parent / "console").symlink_to(layout.member / "state" / "logs")
    target = f"{link_parent}/console/console.log"
    assert not target.startswith(str(layout.member))  # what a literal comparison would see
    found = ours(layout, "kza-tool.service", StandardOutput=f"append:{target}")
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


def test_a_component_with_the_member_identity_is_a_path_hit(layout):
    """A bind mount: neither the literal nor the resolved path names the member, its inode does."""
    mount = layout.base / "mnt"
    (mount / "state").mkdir(parents=True)
    info = mount.stat()
    members = [{"path": str(layout.member), "identity": {"device": info.st_dev, "inode": info.st_ino}}]
    found = ours(layout, "kza-tool.service", WorkingDirectory=str(mount / "state"))
    assert refused(layout, found, members=members) == PATH_HIT + "kza-tool.service"


def test_an_ancestor_counts_only_strictly_below_the_root(layout):
    found = ours(layout, "kza-tool.service", ReadWritePaths=f"-{layout.root}/slots")
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"
    found = ours(layout, "kza-tool.service", ReadWritePaths=f"{layout.home} {layout.root}",
                 WorkingDirectory=str(layout.root), ExecStart=exec_value("/bin/tool", "--root", str(layout.root)))
    run_scan(layout, found)


def test_bind_paths_and_state_directories_are_checked(layout):
    found = ours(layout, "kza-tool.service", BindReadOnlyPaths=f"{layout.member}/state:/data:rbind")
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"
    state_home = layout.root / "slots"
    found = ours(layout, "kza-tool.service", StateDirectory="m1")
    assert refused(layout, found, environ={"XDG_STATE_HOME": str(state_home)}) == PATH_HIT + "kza-tool.service"


def test_a_relative_argument_resolves_against_the_working_directory(layout):
    found = ours(layout, "kza-tool.service", WorkingDirectory=str(layout.root),
                 ExecStart=exec_value("/bin/tool", "slots/m1/state"))
    assert refused(layout, found) == PATH_HIT + "kza-tool.service"


# --- which units (E2) ---------------------------------------------------------------------------------

def test_every_unit_is_scanned_not_only_kza_voice(layout):
    """Mutant 'scan only kza-voice'."""
    found, _conf = _managed_units(layout)
    found.update(ours(layout, "kza-dashboard-proxy.service", state="enabled",
                      WorkingDirectory=str(layout.member / "state")))
    assert refused(layout, found) == PATH_HIT + "kza-dashboard-proxy.service"


def test_generated_and_static_units_are_scanned_and_vendor_units_are_not(layout):
    generated = {"kza-proxy.service": unit(f"/run/user/{UID}/systemd/generator/kza-proxy.service", state="generated",
                                           SourcePath=str(layout.home / ".config/containers/systemd/kza-proxy.container"),
                                           WorkingDirectory=str(layout.member))}
    assert refused(layout, generated) == PATH_HIT + "kza-proxy.service"
    static = ours(layout, "kza-helper.service", state="static", WorkingDirectory=str(layout.member))
    assert refused(layout, static) == PATH_HIT + "kza-helper.service"
    vendor = {"vendor.service": unit("/usr/lib/systemd/user/vendor.service", state="static",
                                     WorkingDirectory=str(layout.member)),
              "worker@.service": unit("/usr/lib/systemd/user/worker@.service", WorkingDirectory=str(layout.member))}
    manager = FakeUserManager(vendor)
    report = units_module.scan_units(units_module.strict_systemctl(manager), root=layout.root,
                                     members=layout.members, home=layout.home, uid=UID, environ={})
    assert report["unit_names"] == [] and report["templates_scanned"] == []
    assert not [call for call in manager.calls if "worker@.service" in call]


def test_a_disabled_inactive_unit_naming_a_member_is_scanned(layout):
    """Neither loaded nor enabled, still startable: found by the real-systemd rehearsal."""
    disabled = ours(layout, "kza-voice.service", state="disabled", loaded=False,
                    ReadOnlyPaths=str(layout.member / "state" / "logs"))
    assert refused(layout, disabled) == PATH_HIT + "kza-voice.service"


# --- review r2 N4: linked units and templates (probe_p14) --------------------------------------------

def test_a_linked_named_unit_is_scanned_wherever_its_file_is(layout):
    """N4a: kza-voice linked from the legacy checkout; a manager that reports the link target as
    FragmentPath (systemd 255 reports the symlink: both are covered) still has it scanned."""
    legacy = layout.home / "kza" / "systemd"
    found = {"kza-voice.service": unit(str(legacy / "kza-voice.service"), state="linked", loaded=True,
                                       WorkingDirectory=str(layout.member / "runtime"))}
    assert refused(layout, found) == PATH_HIT + "kza-voice.service"
    # Named but not linked (installed elsewhere by hand): still scanned, since it is named.
    found["kza-voice.service"]["state"] = "disabled"
    assert refused(layout, found) == PATH_HIT + "kza-voice.service"


def test_a_linked_or_symlinked_unit_nobody_named_is_scanned(layout):
    linked = {"helper.service": unit("/opt/helper/helper.service", state="linked",
                                     WorkingDirectory=str(layout.member))}
    assert refused(layout, linked) == PATH_HIT + "helper.service"
    (layout.user_dir / "other.service").symlink_to("/opt/other/other.service")
    found = {"other.service": unit("/opt/other/other.service", state="enabled",
                                   WorkingDirectory=str(layout.member))}
    assert refused(layout, found) == PATH_HIT + "other.service"


def _template(layout, name: str, exec_start: str, *, where: Path | None = None) -> dict:
    path = (where or layout.user_dir) / name
    path.write_text(f"[Service]\nType=oneshot\nExecStart={exec_start}\n")
    return {name: unit(str(path), state="static", cat=f"# {path}\n{path.read_text()}")}


def test_a_template_naming_a_member_is_scanned_before_any_instance_exists(layout):
    """N4b: kza-backup@.service in the user unit directory, no instance loaded (list-units does not list it)."""
    found = _template(layout, "kza-backup@.service", f"/usr/bin/tar czf /srv/backup/%i.tgz {layout.member}/state")
    assert refused(layout, found) == PATH_HIT + "kza-backup@.service"


@pytest.mark.parametrize("argument", ["{root}/slots/%i", "%h/../releases/%i", "{root}/%i", "{base}/%i",
                                      "--state={root}/slots/%i/state", "'{root}/slots/%i'"])
def test_a_template_whose_instance_can_land_under_the_root_refuses(layout, argument):
    argument = argument.format(root=layout.root, base=layout.base)
    found = _template(layout, "kza-copy@.service", "/usr/bin/cp -a " + argument + " /srv/x")
    assert refused(layout, found) == PATH_HIT + "kza-copy@.service"


def test_a_template_elsewhere_passes_and_is_reported(layout):
    found = _template(layout, "kza-backup@.service", "/usr/bin/tar czf %t/%i.tgz /srv/data/%i 100%%")
    report = run_scan(layout, found)
    assert report["templates_scanned"] == ["kza-backup@.service"] and report["unit_names"] == []


def test_a_vendor_template_with_a_user_drop_in_is_scanned(layout):
    dropin = layout.user_dir / "worker@.service.d"
    dropin.mkdir()
    (dropin / "x.conf").write_text(f"[Service]\nWorkingDirectory={layout.member}\n")
    found = {"worker@.service": unit("/usr/lib/systemd/user/worker@.service", cat=(
        f"# /usr/lib/systemd/user/worker@.service\n[Service]\nExecStart=/bin/true\n"
        f"# {dropin / 'x.conf'}\n[Service]\nWorkingDirectory={layout.member}\n"))}
    assert refused(layout, found) == PATH_HIT + "worker@.service"


def test_a_loaded_instance_of_a_linked_template_is_scanned(layout):
    found = {"kza-run@.service": unit("/opt/kza-run@.service", state="linked"),
             "kza-run@x.service": unit("/opt/kza-run@.service", state=None, loaded=True,
                                       WorkingDirectory=str(layout.member))}
    assert refused(layout, found) == PATH_HIT + "kza-run@x.service"


# --- review r2 N6: a home whose path contains 'kza' --------------------------------------------------

def test_a_stock_unit_with_a_drop_in_under_a_kza_home_stays_lenient(tmp_path, monkeypatch):
    """With HOME=/home/kza a bare 'kza' marker matched every drop-in under that home (probe_p14 N6)."""
    base = tmp_path.resolve()
    home = base / "home" / "kza"
    monkeypatch.setenv("HOME", str(home))
    root = home / "releases"
    member = root / "slots" / "a"
    (member / "state").mkdir(parents=True)
    user_dir = home / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True)
    info = member.stat()
    layout = SimpleNamespace(base=base, root=root, member=member, home=home, user_dir=user_dir,
                             members=[{"path": str(member), "identity": {"device": info.st_dev,
                                                                         "inode": info.st_ino}}])
    found = _stock_units(layout)
    report = run_scan(layout, found, manager_environment=dict(KZA_MANAGER, HOME=str(home)))
    assert "org.freedesktop.IBus.session.generic.service" in report["tolerated_unresolved"]
    # The same unit naming the account's checkout or root through '~' or '%h' is still kza's.
    for mention in ("~/kza/systemd", "%h/releases/logs", "$HOME/kza", str(home / "kza")):
        name = "org.freedesktop.IBus.session.generic.service"
        found[name]["cat"] += f"# {mention}\n"
        assert refused(layout, found, manager_environment=dict(KZA_MANAGER, HOME=str(home))) == \
            UNPARSED + name + ":ExecStart"
        found = _stock_units(layout)


def test_a_vendor_unit_with_a_user_drop_in_is_scanned(layout):
    found = {"vendor.service": unit("/usr/lib/systemd/user/vendor.service",
                                    DropInPaths=str(layout.user_dir / "vendor.service.d/override.conf"),
                                    WorkingDirectory=str(layout.member))}
    assert refused(layout, found) == PATH_HIT + "vendor.service"


def test_a_loaded_unit_outside_the_unit_files_listing_is_scanned(layout):
    found = ours(layout, "kza-run.service", state=None, loaded=True, WorkingDirectory=str(layout.member))
    assert refused(layout, found) == PATH_HIT + "kza-run.service"


def test_a_timer_scans_the_service_it_triggers(layout):
    found = ours(layout, "kza-tick.timer", Unit="vendor-tick.service", Triggers="vendor-tick.service")
    found["vendor-tick.service"] = unit("/usr/lib/systemd/user/vendor-tick.service", state=None,
                                        ExecStart=exec_value("/bin/tool", str(layout.member / "state")))
    assert refused(layout, found) == PATH_HIT + "vendor-tick.service"


@pytest.mark.parametrize("fail", [("list-unit-files",), ("list-units",), ("show",), ("cat",), ("show-environment",),
                                  (("show", "kza-voice.service"),)])
def test_any_failing_systemctl_call_refuses(layout, fail):
    """Mutant 'treat a systemctl failure as an empty list'."""
    found, conf = _managed_units(layout)
    found["kza-voice.service"]["props"]["StandardOutput"] = "journal"  # so that cat is read
    assert refused(layout, found, fail=fail, extra=(conf,)) == UNAVAILABLE


def test_a_reader_that_raises_or_returns_nothing_refuses(layout):
    def raising(*arguments):
        raise OSError("bus")

    for reader in (raising, lambda *arguments: None):
        with pytest.raises(units_module.UnitScanRefusal, match=UNAVAILABLE):
            units_module.scan_units(reader, root=layout.root, members=layout.members, home=layout.home, uid=UID,
                                    environ={})


def test_failed_to_connect_to_bus_exit_1_refuses(layout, tmp_path):
    """systemctl exits 1 without a bus; ServiceManager's non-strict default would accept that."""
    fake = tmp_path / "systemctl"
    fake.write_text("#!/bin/sh\necho 'Failed to connect to bus: No medium found' >&2\nexit 1\n")
    fake.chmod(0o755)
    manager = launcher.ServiceManager(executable=str(fake))
    assert manager._run("--user", "list-units") == ""  # the trap: exit 1 read as an empty listing
    with pytest.raises(units_module.UnitScanRefusal, match=UNAVAILABLE):
        units_module.scan_units(units_module.strict_systemctl(manager), root=layout.root, members=layout.members,
                                home=layout.home, uid=UID, environ={})


def test_invalid_members_refuse(layout):
    with pytest.raises(units_module.UnitScanRefusal, match="runtime_admission_compaction_unit_members"):
        run_scan(layout, {}, members=[{"path": "relative", "identity": {"device": 1, "inode": 2}}])


# --- logrotate (v5 G1, E4) ------------------------------------------------------------------------------

def _conf(layout, text: str) -> Path:
    conf = layout.root / "console-logrotate.conf"
    conf.write_text(text)
    return conf


def test_a_logrotate_glob_over_the_slots_is_a_path_hit(layout):
    """Mutant 'glob ignored'."""
    conf = _conf(layout, f"{layout.root}/slots/*/state/logs/*.log {{\n    hourly\n    copytruncate\n}}\n")
    assert refused(layout, rotate_units(layout, conf)) == PATH_HIT + "logrotate:console-logrotate.conf"


def test_a_pattern_above_the_root_that_reaches_a_member_is_a_path_hit_without_expansion(layout, monkeypatch):
    calls = []
    real = units_module.glob.glob
    monkeypatch.setattr(units_module.glob, "glob", lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    conf = _conf(layout, f"{layout.base}/*/slots/*/state/logs/*.log {{\n}}\n")
    assert refused(layout, {}, extra=(conf,)) == PATH_HIT + "logrotate:console-logrotate.conf"
    assert calls == []


def test_glob_expansion_happens_only_under_the_root(layout, monkeypatch):
    calls = []
    real = units_module.glob.glob
    monkeypatch.setattr(units_module.glob, "glob",
                        lambda *args, **kwargs: calls.append(kwargs.get("root_dir")) or real(*args, **kwargs))
    other = layout.base / "other"
    other.mkdir()
    (other / "a.log").write_text("")
    run_scan(layout, {}, extra=(_conf(layout, f"{other}/*.log {{\n}}\n"),))
    assert calls == []
    run_scan(layout, {}, extra=(_conf(layout, f"{layout.root}/logs/*.log {{\n}}\n"),))
    assert calls == [str(layout.root / "logs")]


def test_a_glob_through_a_symlink_under_the_root_is_a_path_hit(layout):
    (layout.root / "links").mkdir()
    (layout.root / "links" / "a").symlink_to(layout.member)
    conf = _conf(layout, f"{layout.root}/links/*/console.log {{\n}}\n")
    assert refused(layout, {}, extra=(conf,)) == PATH_HIT + "logrotate:console-logrotate.conf"


def test_an_include_file_into_a_member_is_a_path_hit(layout, monkeypatch):
    """Mutant 'include not followed'."""
    opened = _instrument(monkeypatch)
    conf = _conf(layout, f"include {layout.member}/state/x.conf\n")
    assert refused(layout, rotate_units(layout, conf)) == PATH_HIT + "logrotate:console-logrotate.conf"
    assert not [path for path in opened if _under(path, layout.member)]


def test_an_include_directory_is_read_file_by_file(layout):
    directory = layout.root / "logrotate.d"
    directory.mkdir()
    (directory / "a.conf").write_text(f"{layout.root}/logs/a.log {{\n    daily\n}}\n")
    conf = _conf(layout, f"include {directory}\n")
    assert run_scan(layout, rotate_units(layout, conf))["logrotate_files"] == 2
    (directory / "b.conf").write_text(f"{layout.member}/state/logs/console.log {{\n}}\n")
    assert refused(layout, rotate_units(layout, conf)) == PATH_HIT + "logrotate:b.conf"


def test_a_relative_olddir_into_a_member_is_a_path_hit(layout):
    conf = _conf(layout, f"olddir ../slots/m1/state\n{layout.root}/logs/console.log {{\n    rotate 2\n}}\n")
    assert refused(layout, {}, extra=(conf,)) == PATH_HIT + "logrotate:console-logrotate.conf"
    conf = _conf(layout, f"{layout.root}/logs/console.log {{\n    olddir ../slots/m1/state\n}}\n")
    assert refused(layout, {}, extra=(conf,)) == PATH_HIT + "logrotate:console-logrotate.conf"
    conf = _conf(layout, f"{layout.root}/logs/console.log {{\n    olddir old\n    createolddir\n}}\n")
    run_scan(layout, {}, extra=(conf,))


def test_a_postrotate_script_path_into_a_member_is_a_path_hit(layout):
    conf = _conf(layout, f"{layout.root}/logs/console.log {{\n    postrotate\n"
                         f"        cp {layout.root}/logs/console.log.1 {layout.member}/state/ || true\n"
                         "    endscript\n}\n")
    assert refused(layout, {}, extra=(conf,)) == PATH_HIT + "logrotate:console-logrotate.conf"


def test_a_missing_or_broken_referenced_config_is_unparsed(layout):
    missing = layout.root / "missing.conf"
    assert refused(layout, rotate_units(layout, missing)) == UNPARSED + "logrotate:missing.conf"
    for text in (f"{layout.root}/logs/console.log {{\n    daily\n", "}\n", f"{{\n}}\n",
                 f"{layout.root}/logs/'console.log {{\n}}\n", "include\n"):
        conf = _conf(layout, text)
        assert refused(layout, {}, extra=(conf,)) == UNPARSED + "logrotate:console-logrotate.conf", text


def test_comments_and_unknown_directives_are_ignored(layout):
    conf = _conf(layout, f"# {layout.member}/state/logs/console.log {{\nweekly\nfrobnicate 3\n"
                         f"{layout.root}/logs/console.log {{  # rotated hourly\n    hourly\n    wobble\n}}\n")
    assert run_scan(layout, {}, extra=(conf,))["logrotate_files"] == 1


# --- nothing under a member is opened (rule 7) --------------------------------------------------------

def _under(path, member: Path) -> bool:
    text = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else ""
    return any(text == form or text.startswith(form + "/") for form in {str(member), os.path.realpath(member)})


def _instrument(monkeypatch) -> list:
    """Record every path given to os.open, open, io.open, os.scandir and os.listdir."""
    seen: list = []

    def wrap(function):
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                seen.append(os.fspath(args[0]))
            return function(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(os, "open", wrap(os.open))
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    monkeypatch.setattr(os, "scandir", wrap(os.scandir))
    monkeypatch.setattr(os, "listdir", wrap(os.listdir))
    return seen


def test_a_clean_scan_passes_and_opens_nothing_under_a_member(layout, monkeypatch):
    found, conf = _managed_units(layout)
    directory = layout.root / "logrotate.d"
    directory.mkdir()
    (directory / "a.conf").write_text(f"{layout.root}/logs/*.log {{\n    daily\n}}\n")
    conf.write_text(conf.read_text() + f"include {directory}\n{layout.root}/*/console.log {{\n}}\n")
    opened = _instrument(monkeypatch)
    report = run_scan(layout, found, extra=(conf,))
    assert report["logrotate_files"] == 2 and report["units_scanned"] == 3
    assert str(layout.home / "secrets" / ".env") in opened and str(directory) in opened
    assert str(layout.root) in opened  # the root's own listing for '<root>/*/console.log'
    assert not [path for path in opened if _under(path, layout.member)]


# --- review r3: templates (R3-2), markers (R3-5), escaped manager values (R3-6); probe_templates.py,
# probe_manager_env.py ------------------------------------------------------------------------------

def _template_text(layout, name: str, body: str) -> dict:
    path = layout.user_dir / name
    path.write_text("[Service]\nType=oneshot\n" + body)
    return {name: unit(str(path), state="disabled", cat=f"# {path}\n{path.read_text()}")}


def _home_manager(layout) -> dict:
    return {"HOME": str(layout.home), "PATH": "/usr/bin:/bin"}


@pytest.mark.parametrize("case", ["T1_continuation", "T2_braced_home", "T2_plain_home", "T3_home_instance",
                                  "T4_environment_file", "own_environment", "relative_from_working_directory",
                                  "continuation_after_comment"])
def test_a_user_template_naming_a_member_through_any_form_refuses(layout, case):
    """R3-2 (probe_templates T1-T4 and neighbours): kb@.service, no instance loaded, the member reached by a
    continuation line, $HOME/${HOME}, an EnvironmentFile value, the template's own Environment= or a
    relative path from its WorkingDirectory. The other words write under /srv, never above the test root
    (on Linux pytest's root is under /tmp, and '/tmp/%i' alone refuses by the fixed-part rule)."""
    relative = layout.member.relative_to(layout.home.parent)  # releases/slots/m1 (the root is <base>/releases)
    env_file = layout.home / ".config" / "kb.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(f"SRC={layout.member / 'state'}\n")
    body = {
        "T1_continuation": f"ExecStart=/usr/bin/tar czf /srv/backup/%i.tgz \\\n    {layout.member / 'state'}\n",
        "T2_braced_home": "ExecStart=/usr/bin/tar czf /srv/backup/%i.tgz ${HOME}/../" + str(relative) + "/state\n",
        "T2_plain_home": "ExecStart=/usr/bin/tar czf /srv/backup/%i.tgz $HOME/../" + str(relative) + "/state\n",
        "T3_home_instance": "ExecStart=/usr/bin/tar czf /srv/backup/x.tgz $HOME/../releases/%i\n",
        "T4_environment_file": f"EnvironmentFile=-{env_file}\nExecStart=/usr/bin/tar czf /srv/backup/%i.tgz ${{SRC}}\n",
        "own_environment": f"Environment=\"SRC={layout.member}/state\"\nExecStart=/usr/bin/cp -a $SRC /srv/backup/%i\n",
        "relative_from_working_directory": f"WorkingDirectory={layout.home.parent}\n"
                                           f"ExecStart=/usr/bin/tar czf /srv/backup/%i.tgz {relative}/state\n",
        "continuation_after_comment": "ExecStart=/usr/bin/tar czf /srv/backup/%i.tgz \\\n# a note\n"
                                      f"    {layout.member / 'state'}\n",
    }[case]
    found = _template_text(layout, "kb@.service", body)
    assert refused(layout, found, manager_environment=_home_manager(layout)) == PATH_HIT + "kb@.service"


def test_an_environment_file_inside_a_member_is_refused_before_it_is_read(layout, monkeypatch):
    inside = layout.member / "state" / "x.conf"
    opened = []
    original = os.open
    monkeypatch.setattr(os, "open", lambda path, *a, **k: (opened.append(str(path)), original(path, *a, **k))[1])
    found = _template_text(layout, "kb@.service", f"EnvironmentFile={inside}\nExecStart=/bin/true\n")
    assert refused(layout, found, manager_environment=_home_manager(layout)) == PATH_HIT + "kb@.service"
    assert str(inside) not in opened


@pytest.mark.parametrize("name, tolerated", [("kb@.service", True), ("kza-backup@.service", False)])
@pytest.mark.parametrize("word", ["%i", "${UNSET}/x", "$UNSET"])
def test_a_template_word_that_cannot_be_judged_is_strict_unless_the_0e4_leniency_applies(layout, name, tolerated,
                                                                                         word):
    """R3-2: a word that starts with %i or an unset variable has no fixed part to judge."""
    found = _template_text(layout, name, f"ExecStart=/usr/bin/tool {word}\n")
    if tolerated:
        report = run_scan(layout, found, manager_environment=_home_manager(layout))
        assert report["tolerated_unresolved"] == [name] and report["templates_scanned"] == [name]
    else:
        key = "ExecStart"
        assert refused(layout, found, manager_environment=_home_manager(layout)) == UNPARSED + name + ":" + key


def test_a_template_with_unbalanced_quoting_or_an_unreadable_environment_file_refuses(layout):
    found = _template_text(layout, "kb@.service", "ExecStart=/bin/sh -c 'echo\n")
    assert refused(layout, found, manager_environment=_home_manager(layout)) == UNPARSED + "kb@.service:ExecStart"
    found = _template_text(layout, "kb@.service", f"EnvironmentFile={layout.home}/missing.env\nExecStart=/bin/true\n")
    assert refused(layout, found, manager_environment=_home_manager(layout)) == \
        UNPARSED + "kb@.service:EnvironmentFile"


def test_a_template_reaching_elsewhere_through_every_form_passes(layout):
    env_file = layout.home / "kb.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("SRC=/srv/data\n")
    found = _template_text(layout, "kb@.service", f"EnvironmentFile={env_file}\n"
                                                  "ExecStart=/usr/bin/tar czf %t/%i.tgz \\\n  ${SRC}/%i $HOME/.cache/%i\n")
    report = run_scan(layout, found, manager_environment=_home_manager(layout))
    assert report["templates_scanned"] == ["kb@.service"] and report["tolerated_unresolved"] == []


@pytest.mark.parametrize("value", ["$'{root}/slots/*/state'", "$'{member}/state;'", "$'{member}\\x2fstate'",
                                   "$'{root}/slots/m?/state'"])
def test_an_escaped_manager_value_is_decoded_for_the_member_check(layout, value):
    """R3-6 (probe_manager_env): systemd prints a value holding '*', ';' and the like as $'...'."""
    environment = dict(_home_manager(layout), KZA_SRC=value.format(root=layout.root, member=layout.member))
    assert refused(layout, {}, manager_environment=environment) == PATH_HIT + "show-environment"
    environment["KZA_SRC"] = "$'/srv/*/data;'"
    assert run_scan(layout, {}, manager_environment=environment)["units_scanned"] == 0


def test_a_malformed_escaped_manager_value_refuses(layout):
    environment = dict(_home_manager(layout), BAD="$'unterminated\\")
    assert refused(layout, {}, manager_environment=environment) == UNPARSED + "show-environment:BAD"


def test_a_kza_named_unit_gets_no_leniency(layout):
    """R3-5: strict is the safe side; a unit named kza* is kza's even when its text names no kza path."""
    found = {"kza-sync.service": unit("/usr/lib/systemd/user/kza-sync.service", loaded=True,
                                      cat="[Service]\nExecStart=/usr/bin/rsync -a $SRC /mnt/usb\n",
                                      DropInPaths=str(layout.user_dir / "kza-sync.service.d" / "x.conf"),
                                      ExecStart=exec_value("/usr/bin/rsync", "-a", "$SRC", "/mnt/usb"))}
    assert refused(layout, found) == UNPARSED + "kza-sync.service:ExecStart"
    found["sync.service"] = found.pop("kza-sync.service")
    assert run_scan(layout, found)["tolerated_unresolved"] == ["sync.service"]


# --- review r4: the substring rule (R4-1, R4-3, R4-4) and the fixed part; probe_templates4.py,
# probe_manager_env4.py ------------------------------------------------------------------------------

@pytest.fixture
def kza_home(tmp_path, monkeypatch):
    """Production's shape (probe_templates4): HOME=<base>/home/kza, root=HOME/releases, member root/slots/a."""
    base = tmp_path.resolve()
    home = base / "home" / "kza"
    monkeypatch.setenv("HOME", str(home))
    root = home / "releases"
    member = root / "slots" / "a"
    (member / "state").mkdir(parents=True)
    (member / "runtime").mkdir()
    user_dir = home / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True)
    env_file = home / ".config" / "kza-backup.env"
    env_file.write_text(f"SRC={member / 'state'}\n")
    info = member.stat()
    return SimpleNamespace(base=base, root=root, member=member, home=home, user_dir=user_dir, env_file=env_file,
                           members=[{"path": str(member), "identity": {"device": info.st_dev,
                                                                       "inode": info.st_ino}}])


def _r4_case(kza_home, case: str) -> str:
    member, root, env_file = kza_home.member, kza_home.root, kza_home.env_file
    return {
        # probe_templates4 S1-S3, E1-E3, M1
        "S1_sh_c_home_instance": "ExecStart=/bin/sh -c 'tar czf /srv/b/x.tgz %h/releases/%i'\n",
        "S2_sh_c_root_instance": f"ExecStart=/bin/sh -c 'rm -rf {root}/%I'\n",
        "S3_colon_list": "Environment=PYTHONPATH=/opt/x:%h/releases/%i\nExecStart=/usr/bin/true\n",
        "E1_shell_reads_file_value": f"EnvironmentFile={env_file}\n"
                                     "ExecStart=/bin/sh -c 'tar czf /srv/b/x.tgz \"$$SRC\"'\n",
        "E2_file_overrides_environment": f"EnvironmentFile={env_file}\nEnvironment=SRC=/srv/safe\n"
                                         "ExecStart=/usr/bin/tar czf /srv/b/%i.tgz ${SRC}\n",
        "E3_program_reads_file_value": f"EnvironmentFile={env_file}\nExecStart=/usr/local/bin/kza-backup\n",
        "M1_podman_mount": f"ExecStart=/usr/bin/podman run --rm --mount type=bind,src={member}/state,dst=/data img\n",
        # what only the substring rule sees: a key no word rule reads, a member glued to a name, the root
        "credential_key": "LoadCredential=db:%h/releases/slots/a/state/db\nExecStart=/usr/bin/true\n",
        "glued_to_a_name": f"Description=copy of x{member}/state\nExecStart=/usr/bin/true\n",
        "root_named": f"ExecStart=/usr/bin/tar czf /srv/b/%i.tgz -C {root} slots\n",
        "ancestor_then_instance": "LoadCredential=x:%h/%i\nExecStart=/usr/bin/true\n",
        "home_default_form": "ExecStart=/bin/sh -c 'rm -rf ${HOME:-/nowhere}/releases/%i'\n",
        "double_slash": "ExecStart=/usr/bin/tar czf /srv/b/%i.tgz %h//releases/./slots/a/state\n",
        "partial_root_name": "ExecStart=/usr/bin/tar czf /srv/b/x.tgz %h/rel%i\n",
        # systemd.exec: a file's value wins over Environment= wherever the lines are
        "precedence": f"Environment=BASE=/srv\nEnvironmentFile={kza_home.home / '.config' / 'base.env'}\n"
                      "ExecStart=/usr/bin/tar czf /srv/b/%i.tgz ${BASE}/releases/%i\n",
    }[case]


@pytest.mark.parametrize("case", ["S1_sh_c_home_instance", "S2_sh_c_root_instance", "S3_colon_list",
                                  "E1_shell_reads_file_value", "E2_file_overrides_environment",
                                  "E3_program_reads_file_value", "M1_podman_mount", "credential_key",
                                  "glued_to_a_name", "root_named", "ancestor_then_instance", "home_default_form",
                                  "double_slash", "partial_root_name", "precedence"])
@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r4_a_template_naming_the_root_or_a_member_anywhere_refuses(kza_home, case, name):
    """R4-1/R4-3: before r4 every case passed and was listed in templates_scanned. A path hit refuses in
    any template, the 0e4 leniency notwithstanding."""
    (kza_home.home / ".config" / "base.env").write_text(f"BASE={kza_home.home}\n")
    found = _template_text(kza_home, name, _r4_case(kza_home, case))
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + name


@pytest.mark.parametrize("body", [
    "ExecStart=/bin/sh -c 'tar czf /srv/b/x.tgz ${SRC}'\n",  # S4
    "ExecStart=/usr/bin/tar czf /srv/b/x.tgz -C${SRC} .\n",  # S5
    "ExecStart=/bin/sh -c 'tar czf /srv/b/x.tgz /srv/data/$$NAME'\n",  # any variable in an sh -c script
    "ExecStart=/bin/bash -ec 'tar czf /srv/b/x.tgz /srv/data/${NAME}'\n",
    "ExecStart=/usr/bin/tool --label=backup-%i\n",  # %i after a non-path prefix
], ids=["S4_sh_c_unset", "S5_glued_unset", "shell_variable_in_script", "bash_ec_unset", "instance_in_a_name"])
def test_r4_an_unknown_part_the_scan_cannot_place_is_strict_unless_the_0e4_leniency_applies(kza_home, body):
    """R4-1 S4/S5: an unset variable or %i not at the start of a word, and any variable in an sh -c script,
    refuse unit_unparsed in a kza template; a template not kza's is listed in tolerated_unresolved."""
    found = _template_text(kza_home, "kza-backup@.service", body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == \
        UNPARSED + "kza-backup@.service:ExecStart"
    found = _template_text(kza_home, "kb@.service", body)
    report = run_scan(kza_home, found, manager_environment=_home_manager(kza_home))
    assert report["tolerated_unresolved"] == ["kb@.service"] and report["templates_scanned"] == ["kb@.service"]


def test_r4_a_kza_template_reaching_elsewhere_passes(kza_home):
    """The substring rule is not a ban on paths: these fix nothing under the root and pass in a kza template."""
    (kza_home.home / "backup").mkdir()
    body = (f"EnvironmentFile={kza_home.home / '.config' / 'safe.env'}\nEnvironment=OUT=/srv/b\n"
            "WorkingDirectory=\"%h\"\n"
            "ExecStart=/usr/bin/tar czf %t/%i.tgz %h/backup-%i ${OUT}/%i %h/.cache/%i\n"
            "ExecStartPost=/bin/sh -c 'gzip -t /srv/b/%i.tgz && date +%%s > %t/stamp'\n"
            "LoadCredential=token:%h/.config/kza-token\n")
    (kza_home.home / ".config" / "safe.env").write_text("DATA=/srv/data\n")
    found = _template_text(kza_home, "kza-backup@.service", body)
    report = run_scan(kza_home, found, manager_environment=_home_manager(kza_home))
    assert report["templates_scanned"] == ["kza-backup@.service"] and report["tolerated_unresolved"] == []


@pytest.mark.parametrize("program", ["--mount type=bind,src={state},dst=/data", "--mount=type=bind,source={state}",
                                     "-v {state}:/data"])
def test_r4_a_container_unit_binding_a_member_refuses(kza_home, program):
    """R4-3 (probe_templates4 M2): check_value splits ',' and every '=', so a --mount source is a path;
    the unit is what the Quadlet generator writes for Mount=type=bind,source=<member>."""
    argv = ["/usr/bin/podman", "run", "--rm", *program.format(state=kza_home.member / "state").split(), "img"]
    generated = f"/run/user/{UID}/systemd/generator/kza-chroma.service"
    found = {"kza-chroma.service": unit(generated, ExecStart=exec_value(*argv))}
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "kza-chroma.service"


@pytest.mark.parametrize("value", ["$'x;{member}/state'", "$'a|{member}/state'", "$'a&&{member}/state'",
                                   "$'({member}/state)'", "$'--src={member}/state;x'", "{root}",
                                   "$'cp -a x{member}/state'"])
def test_r4_a_manager_value_gluing_a_member_or_naming_the_root_refuses(kza_home, value):
    """R4-4 (probe_manager_env4 D1-D5): a member glued to a shell character, a member glued to a name, or
    the root itself in the manager's environment refuses as the block."""
    environment = dict(_home_manager(kza_home),
                       KZA_CMD=value.format(member=kza_home.member, root=kza_home.root))
    assert refused(kza_home, {}, manager_environment=environment) == PATH_HIT + "show-environment"


def test_r4_a_double_slash_does_not_hide_a_member_in_a_regular_unit(kza_home):
    found = ours(kza_home, "kza-tool.service",
                 ExecStart=exec_value("/usr/bin/tar", "czf", "/srv/b.tgz", "/" + str(kza_home.member) + "/state"))
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "kza-tool.service"


# --- review r5: a glob reaching the slots (R5-1) and C escapes in template text (R5-2); probe_variants5.py --

def _alias(kza_home) -> Path:
    """A household link in the home to the root (review r4 R4-5 alias), the name 'rl'."""
    link = kza_home.home / "rl"
    link.symlink_to(kza_home.root)
    return link


@pytest.mark.parametrize("argv", [
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {root}/slots/*/state"),  # G3, the household backup
    ("/bin/sh", "-c", "rm -rf {root}/slots/a*"),  # G4
    ("/bin/sh", "-c", "cat {root}/slots/[a]/state/x"),  # G5
    ("/bin/sh", "-c", "du {root}/sl*ts/a"),  # G6
    ("/usr/bin/du", "{root}/slots/*"),  # G7: systemd does not glob, the program may
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/rel*/slots/*/state"),  # above the root
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/r?leases/slots/a/state"),
    ("/bin/sh", "-c", "tar czf /srv/b.tgz ~/rel*/slots/a"),
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/*/slots/a"),
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/r[e]leases/slots/a"),  # a set leaves the rest open
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {base}/home/*/releases/slots/*/state"),  # above the home
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/b*/../releases/slots/a"),  # '..' after a glob
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/r?/slots/*/state"),  # through the home link 'rl'
], ids=["G3_tar_slots_state", "G4_member_prefix", "G5_bracket", "G6_middle_component", "G7_plain_argv",
        "above_root_star", "above_root_question", "tilde", "home_star", "home_set", "above_home",
        "dotdot_after_glob", "home_link"])
def test_r5_a_glob_reaching_the_slots_in_a_regular_unit_refuses(kza_home, argv):
    """R5-1 (probe_variants5 G3-G7): a glob piece is held to the fixed-part rule; before r5 each passed."""
    _alias(kza_home)
    values = {"root": kza_home.root, "home": kza_home.home, "base": kza_home.base}
    found = ours(kza_home, "backup.service", ExecStart=exec_value(*(word.format(**values) for word in argv)))
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "backup.service"


@pytest.mark.parametrize("argv", [
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/backups/*.tgz"),
    ("/bin/sh", "-c", "rm -f {home}/*.log {home}/.cache/kb/*"),
    ("/bin/sh", "-c", "du /srv/*/data {home}/b?ckups"),
    ("/usr/bin/find", "/srv", "-path", "*/cache/*", "-delete"),  # a relative pattern the program reads
    ("/usr/bin/rsync", "-a", "--include=*/", "--exclude=*", "/srv/", "/mnt/usb/"),
    ("/bin/sh", "-c", "sed -e 's/:*$//' /srv/x"),  # unity-gtk-module's ExecStopPost has this
    ("/bin/sh", "-c", "tar czf /srv/b.tgz {home}/rel*"),  # the root itself, as a regular unit may name it
], ids=["backups", "home_logs_and_cache", "elsewhere", "find_path", "rsync_filters", "sed", "root_itself"])
def test_r5_a_glob_elsewhere_in_a_regular_unit_passes(kza_home, argv):
    """The rule is not a ban on globs: one that cannot name anything under the root passes (home above it)."""
    _alias(kza_home)
    found = ours(kza_home, "backup.service",
                 ExecStart=exec_value(*(word.format(home=kza_home.home) for word in argv)))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["unit_names"] == ["backup.service"]


@pytest.mark.parametrize("body", [
    "ExecStart=/bin/sh -c 'rm -rf %h/rel*/slots/%i'\n",  # G1
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/rel*/slots/*/state'\n",  # G1b
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/r?leases/slots/*/state'\n",  # G1c
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz ${HOME}/*/slots/%i'\n",
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/r[e]leases/slots/a'\n",
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/b*/../releases/slots/a'\n",
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/r?/slots/a'\n",  # through the home link 'rl'
    "ExecStart=/bin/sh -c 'tar czf /srv/b/%i.tgz %h/rel*'\n",  # the root itself: refused by name in a template
    "Environment=SRC=%h/rel*/slots\nExecStart=/usr/bin/true\n",
], ids=["G1_glob_then_instance", "G1b_glob_slots_state", "G1c_question_mark", "home_reference_star", "set",
        "dotdot_after_glob", "home_link", "root_itself", "own_environment"])
@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r5_a_glob_reaching_the_slots_in_a_template_refuses(kza_home, body, name):
    """R5-1 (probe_variants5 G1, G1b, G1c): a glob above the root passed every template rule; it now refuses
    unit_path whatever the template's name (no 0e4 leniency for a path hit)."""
    _alias(kza_home)
    found = _template_text(kza_home, name, body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + name


@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r5_a_template_glob_elsewhere_passes(kza_home, name):
    _alias(kza_home)
    body = ("ExecStart=/bin/sh -c 'tar czf %h/backups/%i.tgz /srv/data/*.db'\n"
            "ExecStartPost=/bin/sh -c 'rm -f %h/.cache/kb/*.tmp %t/*.sock; find /srv -name \"*.log\" -delete'\n"
            "ExecStopPost=/usr/bin/rm -f %h/backups/*.part\n")
    found = _template_text(kza_home, name, body)
    report = run_scan(kza_home, found, manager_environment=_home_manager(kza_home))
    assert report["templates_scanned"] == [name]


def test_r5_a_manager_glob_is_matched_by_component(kza_home):
    """check_value now carries the manager environment's glob rule (review r3 R3-6): a glob above the root
    refuses when it can reach the slots, and is matched component by component (``*.log`` cannot be
    ``releases``; before r5 anything directly in the home followed by a glob refused)."""
    environment = dict(_home_manager(kza_home), KZA_GLOB=f"{kza_home.home}/rel*/slots/*/state")
    assert refused(kza_home, {}, manager_environment=environment) == PATH_HIT + "show-environment"
    environment["KZA_GLOB"] = f"{kza_home.home}/*.log"
    assert run_scan(kza_home, {}, manager_environment=environment)["units_scanned"] == 0


@pytest.mark.parametrize("podman_environment", [True, False])
def test_r5_the_five_stock_units_pass_when_the_root_is_in_the_home(kza_home, podman_environment):
    """Production's shape (/home/kza/releases): a glob rule that judged relative patterns from the home would
    refuse unity-gtk-module ('s/:*$//'); the five stock units and kza-voice still pass."""
    found, conf = _managed_units(kza_home)
    found.update(_stock_units(kza_home, podman_environment=podman_environment))
    report = run_scan(kza_home, found, extra=(conf,), manager_environment=dict(KZA_MANAGER, HOME=str(kza_home.home)))
    stock = {"org.freedesktop.IBus.session.generic.service", "org.freedesktop.IBus.session.GNOME.service",
             "unity-gtk-module.service"} | (set() if podman_environment else {"podman.service",
                                                                              "podman-restart.service"})
    assert set(report["tolerated_unresolved"]) == stock
    assert {"kza-voice.service", "podman.service", "unity-gtk-module.service"} <= set(report["unit_names"])


# systemd 255 on kza-yo-lan1 (show of an instance, 2026-11-14): argv as the manager stores it.
_SYSTEMD_WORDS = [
    (r"""A%h/rel\x65ases/a "B%h/rel\x65ases/b" 'C%h/rel\x65ases/c'""",
     ["A%h/releases/a", "B%h/releases/b", "C%h/releases/c"]),
    (r"D\x25h/d E\145 F\u0065 G\d H\x00 I\sJ", ["D%h/d", "Ee", "Fe", r"G\d", r"H\x00", "I J"]),
    (r"""K'a\'b' L\$HOME M"\$x" N\; O"x"y'z'""", ["Ka'b", r"L\$HOME", r"M\$x", r"N\;", "Oxyz"]),
    (r"T\x24{HOME}/t U\x2", ["T${HOME}/t", r"U\x2"]),
]


@pytest.mark.parametrize("value, words", _SYSTEMD_WORDS)
def test_r5_template_words_are_read_as_systemd_reads_them(value, words):
    """R5-2: quotes removed anywhere, C escapes decoded inside quotes and out, unknown ones kept (systemd logs
    'Ignoring unknown escape sequences'); specifiers are expanded after decoding ('D\\x25h' was the home)."""
    assert units_module._systemd_words(value) == words
    with pytest.raises(ValueError):
        units_module._systemd_words(r"/bin/echo S'a\'")  # systemd: 'Unbalanced quoting, ignoring'


@pytest.mark.parametrize("body", [
    r"ExecStart=/bin/rm -rf %h/rel\x65ases/slots/%i" "\n",  # E1
    r'ExecStart=/bin/rm -rf "%h/rel\x65ases/slots/%i"' "\n",  # E2
    r"ExecStart=/bin/sh -c 'rm -rf %h/rel\x65ases/slots/%i'" "\n",
    r"ExecStart=/bin/rm -rf %h/rel\145ases/slots/%i" "\n",
    r"ExecStart=/bin/rm -rf %h/rel\u0065ases/slots/%i" "\n",
    r"LoadCredential=db:%h/rel\x65ases/slots/a/state/db" "\nExecStart=/usr/bin/true\n",  # a key no word rule reads
    r"Environment=D=%h/rel\x65" "\nExecStart=/bin/sh -c 'rm -rf ${D}ases/slots/%i'\n",  # V2, escaped
    r"ExecStart=/bin/sh -c 'tar czf /srv/b.tgz %h/rel\x2a/slots/a'" "\n",  # an escaped glob
], ids=["E1_unquoted", "E2_double_quoted", "single_quoted_script", "octal", "unicode", "credential_key",
        "environment_value", "escaped_glob"])
@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r5_a_c_escape_in_a_template_is_decoded_before_judging(kza_home, body, name):
    """R5-2 (probe_variants5 E1, E2): systemd decodes '\\x65' to 'e', so the program gets <root>/slots/<i>;
    before r5 the scan read 'relx65ases' (shlex) and passed."""
    found = _template_text(kza_home, name, body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + name


@pytest.mark.parametrize("body", [
    r"ExecStart=/usr/bin/tar czf /srv/b/x.tgz \x24{SRC}" "\n",  # an escaped unset variable, systemd substitutes it
    r"ExecStart=/bin/sh -c 'tar czf /srv/b/x.tgz /srv/\x24{SRC}'" "\n",  # the same inside single quotes
], ids=["unquoted", "single_quoted"])
def test_r5_an_escaped_unknown_part_is_strict_unless_the_0e4_leniency_applies(kza_home, body):
    found = _template_text(kza_home, "kza-backup@.service", body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == \
        UNPARSED + "kza-backup@.service:ExecStart"
    found = _template_text(kza_home, "kb@.service", body)
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["tolerated_unresolved"] == \
        ["kb@.service"]


def test_r5_an_escaped_kza_path_makes_a_template_kzas(kza_home):
    """R5-2: the decoded text also decides the 0e4 leniency here ('%h/k\\x7aa' is '%h/kza'); the shared
    adoption rule (mentions_markers) is unchanged."""
    body = "ExecStart=/usr/bin/tool --data %h/k\\x7aa/models %i\n"
    found = _template_text(kza_home, "kb@.service", body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == UNPARSED + "kb@.service:ExecStart"
    found = _template_text(kza_home, "kb@.service", body.replace("k\\x7aa", "k\\x7aq"))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["tolerated_unresolved"] == \
        ["kb@.service"]


def test_r5_escapes_elsewhere_in_a_template_pass(kza_home):
    body = ("ExecStart=/bin/sh -c 'printf \"%%s\\n\" done > /srv/b/%i.stamp'\n"
            "ExecStartPost=/usr/bin/find /srv/b -name x -exec rm {} \\;\n"
            "ExecStopPost=/bin/echo \\x2fsrv\\x2fb \\d\n")
    found = _template_text(kza_home, "kza-backup@.service", body)
    report = run_scan(kza_home, found, manager_environment=_home_manager(kza_home))
    assert report["templates_scanned"] == ["kza-backup@.service"] and report["tolerated_unresolved"] == []


def test_r5_both_readings_of_a_template_value_are_judged(kza_home):
    """R5-2 keeps the r4 (POSIX shell) reading beside systemd's, stricter than either: unquoted 'rel\\eases'
    is 'releases' to shlex (systemd reads ESC). Quoting systemd cannot read refuses even where shlex can."""
    found = _template_text(kza_home, "kb@.service", "Environment=SRC=%h/rel\\eases/slots/a\nExecStart=/usr/bin/true\n")
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "kb@.service"
    found = _template_text(kza_home, "kb@.service", "ExecStart=/bin/echo S'a\\'\n")
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == UNPARSED + "kb@.service:ExecStart"


# --- review r51 (r51-g01-independent-review.json): A-F1 relative globs, A-F2 bracket expressions,
# A-F3 shell quoting in template scripts, A-F4 a variable before a member's tail; probe_r51*.py ----------

@pytest.mark.parametrize("argv", [
    ("/usr/bin/curl", "-fsS", "http://[::1]:8200/health"),  # OVR1
    ("/bin/sh", "-c", "df | awk '/[[:space:]]+/ {{print $1}}'"),  # OVR2
    ("/usr/bin/sed", "-i", "-e", "/[a-z]/d", "/srv/x.txt"),  # OVR3
    ("/usr/bin/grep", "-E", "/[0-9]+$", "/var/log/x"),  # OVR5
    ("/bin/sh", "-c", "ls {home}/[[:digit:]]*/slots/a {home}/[!r]eleases/slots/a {home}/[^r]eleases/slots/a"),
    ("/bin/sh", "-c", "ls {home}/rel[ab/slots/a"),  # an unclosed '[' is a literal
], ids=["curl_ipv6", "awk_class", "sed_address", "grep_regex", "classes_and_negation", "unclosed"])
def test_r51_a_bracket_is_one_character_of_its_set(kza_home, argv):
    """A-F2: '[...]' matched one character, as glob(7) does; before, any '[' opened the rest and these
    refused with no override (they passed at 464c08f)."""
    found = ours(kza_home, "h.service", ExecStart=exec_value(*(word.format(home=kza_home.home) for word in argv)))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["unit_names"] == ["h.service"]


def test_r51_brackets_elsewhere_pass_in_a_template_and_the_manager_environment(kza_home):
    found = _template_text(kza_home, "h@.service", "ExecStart=/usr/bin/curl -fsS http://[::1]:8200/%i\n")  # OVR4
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["templates_scanned"] == \
        ["h@.service"]
    for value in (f"{kza_home.home}/[a-z]*.log", "http://[::1]:631/"):  # refused before r51 (r3's check_pattern)
        environment = dict(_home_manager(kza_home), X=value)
        assert run_scan(kza_home, {}, manager_environment=environment)["units_scanned"] == 0


@pytest.mark.parametrize("word", ["{home}/[r]eleases/slots/*", "{home}/[!x]eleases/slots/a",
                                  "{home}/[^x]eleases/slots/a", "{home}/[[:lower:]]eleases/slots/a",
                                  "{home}/[]r]eleases/slots/a", "{home}/[q-s]eleases/slots/a"])
def test_r51_a_bracket_that_can_be_the_roots_name_refuses(kza_home, word):
    found = ours(kza_home, "b.service", ExecStart=exec_value("/bin/sh", "-c", "ls " + word.format(home=kza_home.home)))
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "b.service"


@pytest.mark.parametrize("component, name, matches", [
    ("[abc", "[abc", True), ("[abc", "a", False),  # unclosed: literal
    ("[!r]eleases", "releases", False), ("[!x]eleases", "releases", True), ("[^r]x", "rx", False),
    ("[]a]", "]", True), ("[[:alpha:]]x", "rx", True), ("[[:digit:]]x", "rx", False),
    ("[[=r=]]eleases", "releases", True), ("[z-a]x", "zx", False), ("r[e]l*", "releases", True),
])
def test_r51_component_pattern_reads_bracket_expressions(component, name, matches):
    expression, opened = units_module._component_pattern(component, None)
    assert not opened
    assert (units_module.re.fullmatch(expression, name) is not None) is matches


@pytest.mark.parametrize("argv, working_directory", [
    (("/bin/sh", "-c", "tar czf /srv/b.tgz releases/slots/*/state"), None),  # REL1, the home is the cwd
    (("/bin/rm", "-rf", "releases/slots/*"), None),  # REL4
    (("/usr/bin/tar", "czf", "/srv/b.tgz", "slots/*/state"), "{root}"),  # REL2
    (("/bin/sh", "-c", "tar czf /srv/b.tgz *"), "{root}"),  # anything from the root
    (("/bin/sh", "-c", "tar czf /srv/b.tgz ./rel*/slots/a"), None),
], ids=["REL1_from_home", "REL4_rm", "REL2_from_root", "star_from_root", "dot_slash"])
def test_r51_a_relative_glob_with_a_directory_is_judged_from_the_working_directory(kza_home, argv,
                                                                                   working_directory):
    """A-F1: the non-glob form 'releases/slots/a/state' already refused; the glob form passed."""
    props = {"WorkingDirectory": working_directory.format(root=kza_home.root)} if working_directory else {}
    found = ours(kza_home, "b.service", ExecStart=exec_value(*argv), **props)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "b.service"


@pytest.mark.parametrize("argv", [
    ("/usr/bin/tar", "czf", "/srv/b.tgz", "backups/*.tgz", "./cache/*"),
    ("/usr/bin/find", "/srv", "-path", "*/cache/*", "-delete"),
    ("/usr/bin/rsync", "-a", "--include=*/", "--exclude=*", "/srv/", "/mnt/usb/"),
], ids=["home_backups", "find_path", "rsync_filters"])
def test_r51_a_relative_glob_elsewhere_passes(kza_home, argv):
    found = ours(kza_home, "b.service", ExecStart=exec_value(*argv))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["unit_names"] == ["b.service"]


@pytest.mark.parametrize("body", [
    "ExecStart=/bin/sh -c 'tar czf /srv/%i.tgz releases/slots/*/state'\n",  # REL3
    "WorkingDirectory=%h/rl\nExecStart=/usr/bin/tar czf /srv/%i.tgz slots/*/state\n",
    "WorkingDirectory=%h/rl\nExecStart=/bin/sh -c 'tar czf /srv/%i.tgz *'\n",
], ids=["REL3_from_home", "from_root_link", "star_from_root_link"])
@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r51_a_relative_glob_in_a_template_is_judged_from_its_base(kza_home, body, name):
    """A-F1 in _judge. The base is the root through the home link 'rl' (naming the root itself would refuse
    by the substring rule, whatever the glob)."""
    _alias(kza_home)
    found = _template_text(kza_home, name, body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + name


@pytest.mark.parametrize("body", [
    "ExecStart=/bin/sh -c \"rm -rf %h/rel''eases/slots/%i\"\n",  # OBF2
    "ExecStart=/bin/bash -c \"rm -rf $'%h/rel\\\\x65ases'/slots/%i\"\n",  # OBF3: bash decodes $'..'
    "ExecStart=/bin/sh -c 'rm -rf %h/rel\"e\"ases/slots/%i'\n",
], ids=["empty_quotes", "ansi_c", "double_quotes"])
@pytest.mark.parametrize("name", ["kza-backup@.service", "kb@.service"])
def test_r51_shell_quoting_in_a_template_script_is_removed_before_judging(kza_home, body, name):
    """A-F3: the shell removes the quotes and deletes <root>/slots/<i>; the cut at quotes hid the root."""
    found = _template_text(kza_home, name, body)
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + name


def test_r51_shell_quoting_elsewhere_in_a_template_script_passes(kza_home):
    found = _template_text(kza_home, "kza-backup@.service",
                           "ExecStart=/bin/sh -c \"echo 'it'\\\"'\\\"'s done' > /srv/b/%i.stamp\"\n")
    report = run_scan(kza_home, found, manager_environment=_home_manager(kza_home))
    assert report["templates_scanned"] == ["kza-backup@.service"] and report["tolerated_unresolved"] == []


def _tolerated(kza_home, script: str) -> dict:
    """A regular unit nobody named and not kza's by its text, so an unset $D is tolerated (0e4 rule)."""
    return {"v.service": unit(str(kza_home.user_dir / "v.service"), cat=f"[Service]\nExecStart=/bin/sh -c '{script}'\n",
                              ExecStart=exec_value("/bin/sh", "-c", script))}


@pytest.mark.parametrize("script", ["ls {home}/$D/slots/a", "ls {home}/$D*/slots", "ls {home}/r$D/slots/*",
                                    "ls {base}/home/$D/slots/a/state"])
def test_r51_a_variable_before_a_members_tail_refuses_in_a_tolerated_unit(kza_home, script):
    """A-F4 (probe_r51_var): the variable may be 'releases' (or 'kza/releases'); the tail names a member or
    <root>/slots. Templates already refused this through the fixed-part rule."""
    found = _tolerated(kza_home, script.format(home=kza_home.home, base=kza_home.base))
    assert refused(kza_home, found, manager_environment=_home_manager(kza_home)) == PATH_HIT + "v.service"


@pytest.mark.parametrize("script", ["ls {home}/$D/x", "ls {home}/$D", "ls {base}/home/$D/releases",
                                    "ls {home}/$D/../x", "ls {home}/x$D/slots/a"])
def test_r51_a_variable_elsewhere_stays_tolerated(kza_home, script):
    """No tail to place, a tail no member has, or the root itself (a regular unit may name it)."""
    found = _tolerated(kza_home, script.format(home=kza_home.home, base=kza_home.base))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["tolerated_unresolved"] == \
        ["v.service"]


def test_r51_an_unclosed_bracket_from_the_root_is_not_a_glob(kza_home):
    """A-F2/A-F1: 'test' brackets are words, not globs; only a closed set makes a token path-bearing."""
    found = ours(kza_home, "b.service", WorkingDirectory=str(kza_home.root),
                 ExecStart=exec_value("/bin/sh", "-c", "[ -f x ] && echo ok"))
    assert run_scan(kza_home, found, manager_environment=_home_manager(kza_home))["unit_names"] == ["b.service"]
