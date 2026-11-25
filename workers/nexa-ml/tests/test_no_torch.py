"""The worker (server + slots + entrypoint) must never import torch or model libraries."""

import pathlib
import subprocess
import sys
import textwrap

WORKER_ROOT = pathlib.Path(__file__).resolve().parents[1]
HEAVY = (
    "torch",
    "torchaudio",
    "faster_whisper",
    "ctranslate2",
    "kokoro",
    "speechbrain",
    "sentence_transformers",
    "transformers",
)

SCRIPT = textwrap.dedent(
    """
    import asyncio, pathlib, sys, tempfile
    sys.path[:0] = [{src!r}, {tests!r}]
    import grpc
    import nexa_ml.__main__, nexa_ml.server, nexa_ml.runtime
    from nexa_ml.config import MlConfig
    from nexa_ml.pb import common_pb2, ml_pb2, ml_pb2_grpc
    from nexa_ml.runtime import MlRuntime
    from nexa_ml.server import start_server
    from conftest import Fakes, pcm

    async def main():
        cfg = MlConfig()
        rt = MlRuntime(cfg, Fakes().loaders(cfg.ambient_stt_workers))
        await rt.load_all()
        sock = pathlib.Path(tempfile.mkdtemp(prefix="nxml-", dir="/tmp")) / "ml.sock"
        server = await start_server(rt, sock)
        async with grpc.aio.insecure_channel(f"unix:{{sock}}") as ch:
            stub = ml_pb2_grpc.MlServiceStub(ch)
            health = await stub.GetHealth(common_pb2.Empty(), timeout=2)
            assert health.base.state == common_pb2.HEALTH_STATE_OK
            await stub.Transcribe(ml_pb2.TranscribeRequest(pcm_f32le=pcm(), sample_rate=16000,
                                  mode=ml_pb2.STT_MODE_COMMAND), timeout=10)
            tts = stub.SynthesizeStream(ml_pb2.SynthesizeRequest(text="hola"), timeout=10)
            chunks = [c async for c in tts]
            assert chunks[-1].is_final
        await server.stop(0)
        rt.close()
        sock.unlink(missing_ok=True); sock.parent.rmdir()

    asyncio.run(main())
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in {heavy!r})
    print("HEAVY:" + ",".join(loaded))
    """
)


def test_server_and_slots_never_import_torch() -> None:
    script = SCRIPT.format(src=str(WORKER_ROOT / "src"), tests=str(WORKER_ROOT / "tests"), heavy=set(HEAVY))
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, cwd=WORKER_ROOT
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "HEAVY:"
