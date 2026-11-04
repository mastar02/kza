"""Isolate failed Router retirement: privacy ownership stays until OS exit."""

import json
from pathlib import Path
import subprocess
import sys


def _graph_privacy_failure_child(temporary_root):
    import asyncio
    from contextlib import AsyncExitStack
    import os
    import httpx
    import openai
    import pytest
    from src.application import ApplicationFactories
    from src.main import build_application
    from src.llm import reasoner
    import src.vectordb.chroma_sync as index_module
    from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home

    patch = pytest.MonkeyPatch()
    root = Path(temporary_root)
    patch.chdir(root)
    patch.setattr(index_module, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())
    clients, ledger_closed, sdk_close_calls = [], [], []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "synthetic-router", "object": "model"}]})
        return httpx.Response(200, json={"id": "synthetic", "object": "text_completion", "created": 0,
            "model": "synthetic-router", "choices": [{"index": 0, "text": "synthetic response", "finish_reason": "stop"}]})

    actual_http = httpx.Client

    def http_client(*args, **kwargs):
        value = actual_http(*args, transport=httpx.MockTransport(handle), **kwargs)
        clients.append(value)
        return value

    class HttpxProxy:
        Client = staticmethod(http_client)

        def __getattr__(self, name):
            return getattr(httpx, name)

    patch.setattr(reasoner, "httpx", HttpxProxy())
    patch.setattr(reasoner, "_resolve_api_key", lambda *args, **kwargs: "synthetic-token")

    def failed_sdk_close(client):
        sdk_close_calls.append(True)
        raise RuntimeError("synthetic permanent SDK close failure")

    patch.setattr(openai.OpenAI, "close", failed_sdk_close)

    async def exercise():
        stt, tts = OfflineSTT(), OfflineTTS()
        factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
        async with offline_home() as (url, observations):
            config = offline_config(root, url)
            config["dashboard"]["enabled"] = False
            config["router"] = {"enabled": True, "base_url": "http://synthetic.invalid/v1",
                                "model": "synthetic-router", "destination": "local"}
            config["nlu"] = {"llm_router": {"enabled": True}}
            stack = AsyncExitStack()
            graph = await build_application(config, shutdown=stack, factories=factories)
            fast = graph.pipeline.request_router.llm_command_router.router
            assert await fast.complete("synthetic privacy-close failure") == "synthetic response"
            workers = tuple(fast._router_executor._threads)
            ledger, lease = graph.privacy.ledger, graph.privacy._lease
            assert lease is not None
            original_close = type(ledger).close

            def close_ledger(actual):
                if actual is ledger:
                    ledger_closed.append(True)
                return original_close(actual)

            patch.setattr(type(ledger), "close", close_ledger)
            failure_type = None
            try:
                await stack.aclose()
            except BaseException as error:
                failure_type = type(error).__name__
            assert failure_type is not None and sdk_close_calls
            assert all(worker._handle.is_done() for worker in workers)
            owners = list(fast._router_session._engines.values())
            assert owners and all(owner._handle.is_done() for owner in owners)
            assert any(not owner._confirmed for owner in owners)
            assert fast._client is not None
            protected = graph.privacy._lease == lease and graph.privacy.resource_status()["writer_lease"] and not ledger_closed
            receipt = {"reason": "confirmed" if protected else "privacy_released_after_failed_router_retirement",
                       "writer_lease_retained": graph.privacy.resource_status()["writer_lease"],
                       "ledger_close_calls": len(ledger_closed), "sdk_close_calls": len(sdk_close_calls),
                       "native_workers_exited": True, "native_sdk_closers_exited": True,
                       "router_retirement_unconfirmed": True, "service_calls": observations["service_calls"]}
            print("GRAPH_PRIVACY_FAILURE=" + json.dumps(receipt), flush=True)
            # An unconfirmed model owner and its real privacy lease are kept.
            # Do not clear flags, retry a native close or force ledger release.
            os._exit(0 if protected else 1)

    asyncio.run(exercise())


def test_graph_failed_router_retirement_keeps_actual_privacy_ledger_and_writer_lease_until_process_exit():
    import tempfile

    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "tools/benchmark_latency.py").is_file())
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['_graph_privacy_failure_child'](sys.argv[2])"
    with tempfile.TemporaryDirectory(prefix="kza-router-privacy-synthetic-") as temporary:
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), temporary],
                                cwd=root, capture_output=True, text=True, timeout=20)
    rows = [json.loads(line.removeprefix("GRAPH_PRIVACY_FAILURE=")) for line in result.stdout.splitlines()
            if line.startswith("GRAPH_PRIVACY_FAILURE=")]
    assert rows, f"fixture failed before privacy/native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed" and rows[-1]["service_calls"] == 0
