"""Run after cargo build -p nexa-brain -p nexa-web, with the same CARGO_TARGET_DIR."""

import json
import os
import secrets
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
BIN = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "debug"


class RoutineReceipts(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        config = root / "config"
        config.mkdir()
        source = ROOT / "config/nexa"
        toml = (source / "nexa.toml").read_text()
        for old, new in [
            ("/var/lib/nexa", root / "data"),
            ("/etc/nexa/secrets", root / "secrets"),
            ("/run/user/nexa/nexa", root / "run"),
        ]:
            toml = toml.replace(old, str(new))
        (config / "nexa.toml").write_text(toml)
        (config / "rooms.toml").write_bytes((source / "rooms.example.toml").read_bytes())
        self.token = secrets.token_urlsafe(40)
        env = {
            "PATH": os.environ["PATH"],
            "RUNTIME_DIRECTORY": str(root / "run"),
            "KZA_DASHBOARD_TOKEN": self.token,
            "RUST_LOG": "info",
            "NO_COLOR": "1",
        }
        # Bind port zero and read the actual listener address from its startup log.
        self.origin = "http://127.0.0.1"
        brain = self.start(root, env, "nexa-brain", "--standalone-test", "--config", config)
        web = self.start(root, env, "nexa-web", "--config", config, "--bind", "127.0.0.1:0",
                         "--origin", self.origin)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.assertIsNone(brain.poll(), "brain exited before readiness")
            self.assertIsNone(web.poll(), "web exited before readiness")
            log = (root / "nexa-web.log").read_text()
            if "address=" in log and (root / "run/brain.sock").exists():
                address = log.split("address=", 1)[1].split()[0]
                self.url = f"http://{address}/api/routines"
                try:
                    if self.request("") == []:
                        return
                except URLError:
                    pass
            time.sleep(0.05)
        self.fail("daemons did not become ready")

    def start(self, root, env, name, *args):
        with (root / f"{name}.log").open("w") as log:
            child = subprocess.Popen([BIN / name, *args], env=env, stdout=log, stderr=log)
        self.addCleanup(child.wait, 5)
        self.addCleanup(child.terminate)
        return child

    def request(self, path, body=None, expected=200):
        headers = {"Origin": self.origin, "Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json"}
        data = None if body is None else json.dumps(body).encode()
        try:
            response = urlopen(Request(self.url + path, data=data, headers=headers), timeout=5)
        except HTTPError as error:
            response = error
        with response:
            payload = json.load(response)
            self.assertEqual(response.status, expected, (path, payload))
            return payload

    def test_persisted_receipts(self):
        steps = [{"kind": "delay", "duration": {"secs": 0, "nanos": 1000000}}]
        routine, other = [
            self.request("", {"name": name, "enabled": True, "steps": steps})["routine_id"]
            for name in ("receipt test", "empty routine")
        ]
        base = f"/{routine}"
        self.assertEqual(self.request(base + "/attempts"), {"routine_id": routine, "runs": []})
        execute = base + "/execute"
        first = self.request(execute, {"request_id": "first"}, 202)
        path = f"{base}/runs/{first['run_id']}"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = self.request(path)["run"]
            if run["status"] == "succeeded":
                break
            time.sleep(0.05)
        else:
            self.fail("scheduled work did not finish")
        self.assertGreaterEqual(run["finished_at_unix_ms"], first["accepted_at_unix_ms"])
        self.assertEqual(run, {
            "run_id": first["run_id"], "routine_id": routine, "source": "manual",
            "status": "succeeded", "started_at_unix_ms": first["accepted_at_unix_ms"],
            "finished_at_unix_ms": run["finished_at_unix_ms"],
        })
        replay = self.request(execute, {"request_id": "first"}, 202)
        self.assertEqual(replay["run_id"], run["run_id"])
        second = self.request(execute, {"request_id": "second"}, 202)
        history = self.request(base + "/attempts")
        self.assertEqual([row["run_id"] for row in history["runs"]], [second["run_id"], run["run_id"]])
        self.assertEqual(history["runs"][1], run)
        limited = self.request(base + "/attempts?limit=1")
        self.assertEqual([row["run_id"] for row in limited["runs"]], [second["run_id"]])
        self.assertEqual(self.request(f"/{other}/attempts"), {"routine_id": other, "runs": []})
        self.request(f"/{other}/runs/{run['run_id']}", expected=404)
        self.request(base + "/runs/999999", expected=404)
        self.request("/999999/attempts", expected=404)


if __name__ == "__main__":
    unittest.main()
