"""U3b-0d rehearsal child: the managed stop/readiness contract without the application graph.

Started by the compiled controller (``serve --child-command``) with cwd = the
slot's compiled runtime. It imports ``src.core.process_lifecycle`` from that
runtime's bytecode (the same contract src.main arms) and writes its closure
receipt with the runtime's own compiled ``scripts/stage_runtime.pyc``. The
blocked signal mask is read before arming, so a controller that masked
SIGTERM around Popen is visible (v3 N1).

Scenarios: clean, no_ready, crash, self_end, exit_no_receipt, hang,
receipt_then_hang, watchdog_silent, slow_stop, restore_after_receipt.
"""

import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

RUNTIME = Path.cwd().resolve()
sys.path[:0] = [str(RUNTIME)]
BLOCKED_AT_ENTRY = sorted(int(item) for item in signal.pthread_sigmask(signal.SIG_BLOCK, ())
                          if item in (signal.SIGTERM, signal.SIGINT))

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True)
parser.add_argument("--phase-file", required=True)
parser.add_argument("--seconds", type=float, default=0.0)
parser.add_argument("--touch", default="")
parser.add_argument("--exec", dest="execute", default="")  # in-band state changes under the ticket
ARGS = parser.parse_args()

from src.core.process_lifecycle import StopRequested, arm_process_entry  # noqa: E402

LIFECYCLE = arm_process_entry()


def phase(name: str, value: object = "") -> None:
    with open(ARGS.phase_file, "a") as out:
        out.write(f"{name} {value}\n")
        out.flush()


def receipt() -> None:
    spec = importlib.util.spec_from_file_location("_launcher_child_stage", RUNTIME / "scripts/stage_runtime.pyc")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_admission_receipt(os.environ["KZA_RUNTIME_ADMISSION"], dict.fromkeys(
        ["graph_closed", "privacy_closed", "sdk_closed", "native_closed"], True))
    phase("receipt")


async def main() -> None:
    loop = asyncio.get_running_loop()
    LIFECYCLE.stop.install(loop)
    phase("pid", os.getpid())
    phase("mask", ",".join(map(str, BLOCKED_AT_ENTRY)) or "empty")
    phase("env", ",".join(sorted(key for key in os.environ
                                 if key in {"NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID", "KZA_NOTIFY_FD",
                                            "KZA_CONTROL_FD", "KZA_WATCHDOG_INTERVAL_USEC"})) or "clean")
    phase("pythonpath", os.environ.get("PYTHONPATH", ""))
    phase("config_path", os.environ.get("CONFIG_PATH", ""))
    scenario = ARGS.scenario
    if ARGS.execute:
        exec(Path(ARGS.execute).read_text(), {"state": Path(os.path.realpath("data")).parent})
        phase("executed")
    if scenario == "watchdog_silent":
        LIFECYCLE.notifier.watchdog_usec = None
    if scenario != "no_ready":
        try:
            LIFECYCLE.ready()
            phase("ready")
        except StopRequested:  # a stop before READY: never ready, still a clean close
            phase("ready_skipped")
    if scenario == "crash":
        await asyncio.sleep(ARGS.seconds)
        phase("crash")
        os._exit(1)
    if scenario in {"self_end", "exit_no_receipt"}:
        await asyncio.sleep(ARGS.seconds)
        if scenario == "self_end":
            receipt()
        os._exit(0)
    await LIFECYCLE.stop.wait()
    phase("stop_received", LIFECYCLE.stop.requested_phase.value)
    await asyncio.sleep(0.05)
    phase("fast", LIFECYCLE.fast_close)
    if scenario == "hang":
        await asyncio.Event().wait()
    if scenario == "slow_stop":
        await asyncio.sleep(ARGS.seconds)
        phase("fast_after", LIFECYCLE.fast_close)
    receipt()
    if scenario == "restore_after_receipt":
        Path(ARGS.touch).write_text('{"restored": true}')
    if scenario == "receipt_then_hang":
        time.sleep(3600)
    os._exit(0)


asyncio.run(main())
