# `workers/nexa-ml/` — Stateless GPU Inference Worker (Phase 3)

Thin, stateless Python 3.13 worker speaking `nexa.v1.MlService` over `$XDG_RUNTIME_DIR/nexa/ml.sock`.

## Hard Boundaries
- **Zero network access** (`IPAddressDeny=any` in `nexa-ml.service`).
- **Zero SQLite / state** (all persistence lives in `nexa-memoryd` and `nexa-brain`).
- **Zero audio hardware** (ALSA/XVF3800/MA1260 live exclusively in Rust `nexa-audiod`).
- **Zero PyTorch in PR unit CI** (`dev` extra tests wire contracts against lightweight stubs; GPU benchmarks run separately on the server).

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
