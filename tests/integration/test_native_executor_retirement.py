"""Native admission contracts; missing receipt is not an OS-actor leak claim."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import weakref


def _root():
    return next(p for p in Path(__file__).resolve().parents
                if (p / "tools/benchmark_latency.py").is_file())


def _source():
    if os.environ.get("KZA_NATIVE_EXECUTOR_SOURCE"):
        return Path(os.environ["KZA_NATIVE_EXECUTOR_SOURCE"]).resolve()
    live = _root() / "src/core/native_executor.py"
    return live if live.is_file() else Path(__file__).with_name("native-executor-before-contract.py")


def _emit(mode, reason, source, **scalars):
    print("NATIVE_CONTRACT=" + json.dumps({"mode": mode, "reason": reason,
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), **scalars}), flush=True)


def _probe(mode, source):
    source = Path(source)
    spec = importlib.util.spec_from_file_location("synthetic_native_executor", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module._NativeWorker
    workers, starts, callbacks = [], [], []
    executor = module.NativeThreadPoolExecutor(thread_name_prefix="native-contract")

    class Worker(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(weakref.ref(self))

        def start(self):
            starts.append(1)
            if mode == "unconfirmed":
                assert self._handle.ident == 0 and not self._started.is_set() and not self._handle.is_done()
                raise KeyboardInterrupt
            try:
                executor.shutdown(wait=True)
            except RuntimeError as error:
                callbacks.append(str(error))
            else:
                callbacks.append("unexpected_shutdown_return")
            return super().start()

    module._NativeWorker = Worker
    if mode == "unconfirmed":
        first = second = shutdown = None
        try:
            executor.submit(lambda: "first")
        except BaseException as error:
            first = (type(error).__name__, str(error))
        try:
            executor.submit(lambda: "second")
        except BaseException as error:
            second = (type(error).__name__, str(error))
        try:
            executor.shutdown(wait=True)
        except BaseException as error:
            shutdown = (type(error).__name__, str(error))
        retained = any(reference() is not None for reference in workers)
        if len(starts) != 1 or second is None or shutdown is None or "unconfirmed" not in shutdown[1] or not retained:
            _emit(mode, "uncertain_admission_was_not_terminal", source,
                  first=first, second=second, shutdown=shutdown, start_calls=len(starts),
                  worker_metadata_retained=retained, claim="missing_native_receipt_not_os_actor_leak")
            os._exit(1)
        assert first[0] == "KeyboardInterrupt"
        _emit(mode, "confirmed", source, start_calls=1, shutdown_rejected=True,
              worker_metadata_retained=True, claim="missing_native_receipt_not_os_actor_leak")
        os._exit(0)

    result, failures = [], []

    def submit():
        try:
            result.append(executor.submit(lambda: "actual native callable").result(timeout=3))
        except BaseException as error:
            failures.append((type(error).__name__, str(error)))

    caller = threading.Thread(target=submit, daemon=True, name="native-submit-caller")
    caller.start()
    caller._handle.join(1)
    if not caller._handle.is_done():
        _emit(mode, "shutdown_reentered_submit_lock", source,
              caller_native_exited=False, start_calls=len(starts), callbacks=callbacks)
        os._exit(1)
    assert failures == [] and result == ["actual native callable"]
    assert len(starts) == 1 and len(callbacks) == 1
    assert "native_executor" in callbacks[0] and any(word in callbacks[0] for word in ("close", "shutdown"))
    assert callbacks[0] != "unexpected_shutdown_return"
    executor.shutdown(wait=True)
    assert all(reference() is not None and reference()._handle.is_done() for reference in workers)
    _emit(mode, "confirmed", source, caller_native_exited=True,
          worker_native_exited=True, start_calls=1, callback_rejected_before_lock=True)
    os._exit(0)


def _run(mode):
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3])"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode, str(_source())],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("NATIVE_CONTRACT=")) for line in result.stdout.splitlines()
            if line.startswith("NATIVE_CONTRACT=")]
    assert rows, f"fixture failed before contract receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]


def test_unconfirmed_native_start_control_cannot_be_retried_or_acknowledged_by_shutdown():
    _run("unconfirmed")


def test_reentrant_shutdown_during_native_thread_start_is_rejected_before_submit_lock():
    _run("reentrant")
