# `workers/nexa-ml/` — Stateless GPU Inference Worker (Phase 3)

Thin, stateless Python 3.13 worker speaking `nexa.v1.MlService` over `$XDG_RUNTIME_DIR/nexa/ml.sock`.

## Hard Boundaries
- **Zero network access** (`IPAddressDeny=any` in `nexa-ml.service`).
- **Zero SQLite / state** (all persistence lives in `nexa-memoryd` and `nexa-brain`).
- **Zero audio hardware** (ALSA/XVF3800/MA1260 live exclusively in Rust `nexa-audiod`).
- **Zero PyTorch in PR unit CI** (`dev` extra tests wire contracts against lightweight stubs; GPU benchmarks run separately on the server).

## GPU placement

The Linux host has two RTX 3070 GPUs with 8 GiB each.

| Device | Workload | Approximate VRAM |
|---|---|---|
| `cuda:1` | `llama-server` (`Qwen3.5-9B-Q4_K_M`) on port `8101` (dedicated to local LLM) | Dedicated |
| `cuda:0` | Shared `whisper-v3-turbo` (`float16`, `WhisperPriorityGate`), streaming Kokoro-82M TTS, and ECAPA-TDNN speaker encoder | Shared STT+TTS+Speaker |
| CPU | `BAAI/bge-m3` embeddings and ONNX inference for Silero VAD and OpenWakeWord (optional Parakeet shadow veto) | — |

## Offline model cache

Before starting the deployed worker, seed the persistent cache at
`/var/lib/nexa/.cache` (mode `0700`, owned by `nexa:nexa`). For the default
ECAPA model, place the full local SpeechBrain bundle, including
`hyperparams.yaml` and `embedding_model.ckpt`, in
`/var/lib/nexa/.cache/nexa-ml/speechbrain/speechbrain--spkrec-ecapa-voxceleb/`.
Populate `HF_HOME` at `/var/lib/nexa/.cache/huggingface/` with the configured
model and voice assets, preserving the Hub cache layout and snapshot links
to blobs. The worker reuses these files across restarts with network access disabled.

## Layout (`src/nexa_ml/`)
| Module | Role |
|---|---|
| `__main__.py` | `python -m nexa_ml --socket %t/nexa/ml.sock --config /etc/nexa/nexa.toml` |
| `server.py` | `grpc.aio` `MlService` on `unix:<socket>` (umask 077 + `chmod 0600`), fixed status mapping, lifecycle |
| `runtime.py` | Slot composition root (DI of model loaders), startup loads, health, reload routing |
| `slot.py` | `ModelSlot`: one model, own `ThreadPoolExecutor(1)` (or `to_thread` for CPU), drain → unload → `del`+`gc` → recreate |
| `whisper_slot.py` | `stt-command` (v3-turbo, `vad_filter=False`) + AMBIENT pool `stt-ambient-{0..N-1}` (large-v3, `vad_filter=True`) |
| `tts_slot.py` | Kokoro-82M, one `TtsAudioChunk` per `KPipeline` Result, stop at next Result on cancel |
| `speaker_slot.py` | Single ECAPA-TDNN, raw 192-d vector only (voice ≠ person ≠ account ≠ role) |
| `embed_slot.py` | BGE-M3 on CPU, 1024-d, `asyncio.to_thread` |
| `notify.py` | Native `sd_notify` (`READY=1`, `WATCHDOG=1` every 15 s, `STOPPING=1`; no-op without `NOTIFY_SOCKET`) |
| `pb/` | Generated stubs — regenerate with `scripts/gen_pb.py`, never edit |

## Contracts
- **Slot ids** (`GetHealth.models[].slot_id`, `ReloadModel.slot_id`): `stt-command`, `stt-ambient-<i>`, `tts`, `speaker`, `embed`.
- **TTS stream end:** after the last audio chunk the server sends one extra chunk with empty `pcm_f32le` and `is_final=true` (no lookahead, so chunk 0 is never delayed).
- **Status codes:** `INVALID_ARGUMENT` input/voice/slot · `UNAVAILABLE` slot down/reloading · `RESOURCE_EXHAUSTED` queue full/OOM · `CANCELLED` client cancel · `INTERNAL` anything else. Details never echo audio or text.
- **ReloadModel authz:** requires a verified `ActorContext` with `subject` (`UNAUTHENTICATED` otherwise), `ROLE_OPERATOR` and `PERMISSION_ADMIN_MANAGE` (`PERMISSION_DENIED` otherwise), checked before the slot id is resolved or logged.
- **Zero network (`ml.offline=true`, default):** `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are forced to `1` (overwriting any inherited value); every real loader rejects URI sources and fails closed when the Hugging Face client is not offline. ECAPA loads only from local files (`source` dir or `<models_dir>/speechbrain/...` cache); Kokoro prefers a local layout `<models_dir>/kokoro/{config.json,*.pth,voices/*.pt}` and otherwise only the offline HF cache.
- **No lazy reload:** a failed load/reload leaves the slot `UNAVAILABLE` until an explicit `ReloadModel`.
- **PCM:** mono float32 LE at 16 kHz for STT/speaker (no resampling); Kokoro emits 24 kHz.

## Development
```bash
# Stubs (pinned generator; CI fails on drift)
uv run --no-project --python 3.13 --with grpcio-tools==1.81.0 python workers/nexa-ml/scripts/gen_pb.py
# L2 tests without torch (from workers/nexa-ml/)
python -m pytest -m "not gpu" -q
# Server-only smoke with real models
NEXA_ML_CONFIG=/etc/nexa/nexa.toml python -m pytest -m gpu -s tests/gpu
```

### GPU extra on CPython 3.13
No published Kokoro wheel declares 3.13 support (0.9.4: `Requires-Python <3.13`; 0.7.x pins `numpy==1.26.4`). Install the extra with **uv** (it ignores Requires-Python upper bounds), e.g. `uv pip install --python /opt/nexa/ml-venv/bin/python "workers/nexa-ml[gpu]"`; `pip` refuses it. CI only proves the resolution; runtime behaviour on 3.13 is validated by `-m gpu` on the server.
