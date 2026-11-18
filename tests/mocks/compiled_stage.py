"""Run a compiled ``scripts/stage_runtime.pyc`` with the rehearsal fixture's models root (U3b-0e4).

Usage: ``python -I -B compiled_stage.py <stage_runtime.pyc> --root <root> --offline <command> ...``

The adoption commands read the 2a table's models root from
``adoption.PRODUCTION_MODELS`` (/home/kza/kza/models), which a rehearsal
fixture cannot use. This wrapper loads the compiled module exactly as
``python -I -B <pyc>`` would, then rebinds only ``production_paths()`` of the
compiled adoption commands to ``KZA_REHEARSAL_MODELS_ROOT`` (the logs and
metrics rows keep their production values, as in the in-process tests).
``KZA_REHEARSAL_WALL_OFFSET`` (seconds) moves the wall clock forward for the
day-14 rehearsal of retire-legacy-state; it is applied before the compiled
modules load, so their ``wall=time.time`` defaults see it.
``KZA_REHEARSAL_KILL_AT_STEP`` (SC-6b, rows T0 and T0S) makes the compiled
``install-controller`` SIGKILL its own process at that ``step_hook`` step
('checked': after the read-only checks, before the switch; 'switched':
after ``current`` moved, before controller.json). The value
'dry-run-mid-copy' (SC-6b-B, row DR) instead makes the compiled
``adopt --dry-run`` SIGKILL its own process mid-copy: two files of the
state copied into its shadow under <root>/admission/dry-run/, the third
not. Everything else is the compiled bytecode of the archive. Not a product
entry point.
"""
import importlib.util
import os
import signal
import sys
import time

pyc, arguments = sys.argv[1], sys.argv[2:]
offset = float(os.environ.get("KZA_REHEARSAL_WALL_OFFSET", "0") or 0)
if offset:
    _real_time = time.time
    time.time = lambda: _real_time() + offset
spec = importlib.util.spec_from_file_location("_kza_rehearsal_stage_runtime", pyc)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
models = os.environ.get("KZA_REHEARSAL_MODELS_ROOT")
if models and any(argument in module.ADOPTION_COMMANDS for argument in arguments):
    commands = module._adoption_commands_module()
    library = commands.adoption
    commands.production_paths = lambda: commands.ProductionPaths(models, library.PRODUCTION_LOGS,
                                                                 library.PRODUCTION_METRICS)
kill_at = os.environ.get("KZA_REHEARSAL_KILL_AT_STEP")
if kill_at and "install-controller" in arguments:
    compiled_launcher = module._launcher_module()
    _install = compiled_launcher.install_controller

    def install_controller(root, archive, **options):
        def hook(step):
            if step == kill_at:
                os.kill(os.getpid(), signal.SIGKILL)
        return _install(root, archive, step_hook=hook, **options)

    compiled_launcher.install_controller = install_controller
if kill_at == "dry-run-mid-copy" and "adopt" in arguments and "--dry-run" in arguments:
    compiled_launcher = module._launcher_module()
    _wanted = compiled_launcher._shadow_wanted
    _copied = []

    def _shadow_wanted(relative):
        if not _wanted(relative):
            return False
        _copied.append(relative)
        if len(_copied) == 3:  # two files copied, the third not: a dry run killed mid-copy
            os.kill(os.getpid(), signal.SIGKILL)
        return True

    compiled_launcher._shadow_wanted = _shadow_wanted
sys.argv = [pyc, *arguments]
module.main()
