# KZA Deployment Analysis — Single-Room Testing

**Date:** 2026-03-14
**Goal:** Deploy KZA on the production server and validate end-to-end voice pipeline in the escritorio room.
**Hardware:** Threadripper PRO 7965WX, 128GB DDR5, 4x RTX 3070 8GB
**Testing room:** Escritorio (mic_device_index TBD, ma1260_zone 4, bt_adapter hci3)

---

## 1. Code Fixes Required Before Deploy

### Fix 1: Router model — fp16 → AWQ

**Problem:** `Qwen/Qwen2.5-7B-Instruct` in fp16 is ~14GB — does not fit in 8GB VRAM.

**Files:**
- `config/settings.yaml` key `router.model`: change to `Qwen/Qwen2.5-7B-Instruct-AWQ`

**Dependencies to add to `requirements.txt`:**
- `autoawq` — required by vLLM for AWQ model loading

**Note:** Verify the exact HuggingFace model ID before download. The canonical AWQ quant is `Qwen/Qwen2.5-7B-Instruct-AWQ`.

### Fix 2: Nightly training LoRA compatibility with AWQ router

**Problem:** Nightly training generates LoRA adapters for the router. QLoRA training expects a non-quantized base model — applying QLoRA on top of an AWQ-quantized model is not standard.

**Solution:** Train LoRA on the fp16 base (`Qwen/Qwen2.5-7B-Instruct`), then load the adapter at inference into the AWQ model. vLLM supports loading LoRA adapters on AWQ models at runtime.

**Files:**
- `config/settings.yaml` key `training.nightly.base_model`: keep as `Qwen/Qwen2.5-7B-Instruct` (fp16 for training)
- The router runs AWQ for inference, the nightly trains on fp16 — the LoRA adapter is compatible with both

**Out of scope for day one:** Nightly training is disabled until conversation data exists. This compatibility path should be validated when nightly training is first enabled.

### Fix 3: Wire missing config params to FastRouter and LLMReasoner in main.py

**Problem A — FastRouter:** `FastRouter.__init__` accepts `enable_lora`, `lora_path`, and `max_lora_rank`, but `src/main.py` does not pass them from `router_config`. The LoRA hot-swap feature is silently disabled.

**Problem B — LLMReasoner:** `LLMReasoner.__init__` accepts `chat_format`, `lora_path`, `lora_scale`, `n_batch`, `n_gpu_layers`, `rope_freq_base`, `rope_freq_scale`, but `src/main.py` only passes `model_path`, `n_ctx`, and `n_threads`. **Critical:** The `chat_format` defaults to `"llama-3"` in code, but `settings.yaml` specifies `"chatml"` for Qwen2.5-72B. Using the wrong chat template will cause malformed prompts and garbage output.

**File:** `src/main.py` — expand both constructor calls:

```python
# FastRouter (GPU 2)
fast_router = FastRouter(
    model=router_config.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    device=router_config.get("device", "cuda:2"),
    gpu_memory_utilization=router_config.get("gpu_memory_utilization", 0.85),
    enable_prefix_caching=router_config.get("enable_prefix_caching", True),
    enable_lora=router_config.get("enable_lora", False),
    lora_path=router_config.get("lora_path"),
    max_lora_rank=router_config.get("max_lora_rank", 32),
)

# LLMReasoner (CPU)
llm = LLMReasoner(
    model_path=model_path,
    lora_path=reasoner_config.get("lora_path"),
    lora_scale=reasoner_config.get("lora_scale", 1.0),
    n_ctx=reasoner_config.get("n_ctx", 32768),
    n_threads=reasoner_config.get("n_threads", 24),
    n_batch=reasoner_config.get("n_batch", 512),
    n_gpu_layers=reasoner_config.get("n_gpu_layers", 0),
    chat_format=reasoner_config.get("chat_format", "chatml"),
    rope_freq_base=reasoner_config.get("rope_freq_base", 1000000.0),
    rope_freq_scale=reasoner_config.get("rope_freq_scale", 1.0),
)
```

**Also fix:** The current default for `n_ctx` in `main.py` is `8192` but `settings.yaml` specifies `32768`. Update the fallback default to `32768`.

### Fix 4: download_models.sh is outdated

**Problem:** The script downloads wrong models — mismatches with current `settings.yaml`.

| Script downloads | Config expects |
|-----------------|----------------|
| `distil-whisper/distil-small.en` | `marianbasti/distil-whisper-large-v3-es` |
| `BAAI/bge-small-en-v1.5` | `BAAI/bge-m3` |
| Piper TTS only | Kokoro-82M + Qwen3-TTS-0.6B |
| Qwen 72B Q8_0 options only | Qwen2.5-72B-Instruct Q6_K (`bartowski/Qwen2.5-72B-Instruct-GGUF`) |
| No router model | Qwen2.5-7B-Instruct-AWQ |
| No Kokoro/Qwen3-TTS | Dual TTS on GPU 3 |

**Fix:** Rewrite the script. Requirements for the rewrite:
- **Non-interactive** — no `read -p` prompts. Accept model selection via CLI args or default to configured models. Must work when called from systemd or non-interactive shells.
- Download order optimized for early testing:
  1. Small models first (~5 min): Kokoro-82M, ECAPA-TDNN, OpenWakeWord
  2. Medium models (~10-15 min): distil-whisper-large-v3-es, BGE-M3, Qwen3-TTS-0.6B, Qwen2.5-7B-Instruct-AWQ
  3. Large model last (background): Qwen2.5-72B-Instruct-Q6_K from `bartowski/Qwen2.5-72B-Instruct-GGUF` (~64GB)
- Note: wav2vec2 (emotion) and ECAPA-TDNN (speaker ID) are auto-downloaded by `transformers`/`speechbrain` on first use. The script should trigger these downloads explicitly.

### Fix 5: validate_hardware.py — Python version check + CPU comment

**File:** `scripts/validate_hardware.py`
- `check_python_version()`: change condition `version.minor >= 10` → `version.minor >= 13`
- Same function: change message strings `>=3.10 requerido` → `>=3.13 requerido`
- `check_cpu()`: fix comment `# Threadripper 9965WX` → `# Threadripper PRO 7965WX`

### Fix 6: Missing dependencies in requirements.txt

**Problem:** Several packages used in the codebase are not in `requirements.txt`:
- `autoawq` — required for vLLM AWQ model loading
- `speechbrain` — used by `SpeakerIdentifier` (`src/users/speaker_identifier.py`)
- `psutil` — used by `scripts/validate_hardware.py`
- `fastapi` — used by `src/dashboard/api.py`
- `uvicorn` — ASGI server for dashboard API
- `spotipy` — Spotify API client (if used by `src/spotify/client.py`; verify import)

**Note:** Kokoro-82M and Qwen3-TTS-0.6B dependencies need investigation — check their PyPI package names and add to requirements.

### Fix 7: Dashboard API not started in main.py

**Problem:** `src/dashboard/api.py` has a `DashboardAPI` class with a `start()` method, but `src/main.py` never imports or starts it. The dashboard is unreachable.

**Fix:** Import and start the dashboard in `main.py` before the pipeline runs. The dashboard should run as an asyncio task alongside the voice pipeline.

**Impact on deployment:** Phase 5 validation step "Dashboard API responds" will fail without this fix. If dashboard is not critical for day one, move this validation to Phase 7 (extra features).

### Fix 8: .env.example is stale

**Problem:** `.env.example` references `TTS_ENGINE=piper` and `GPU 3: TTS (Piper/XTTS)` — should reference dual TTS (Kokoro/Qwen3-TTS). Missing `CONFIG_PATH` variable.

**Fix:** Update `.env.example` to match current architecture.

---

## 2. Deployment Phases

### Phase 1: Server Base

Run `sudo ./scripts/setup_ubuntu.sh` which handles:
- System packages (portaudio, libsndfile, ffmpeg, alsa, bluetooth)
- Python 3.13 from deadsnakes PPA
- NVIDIA drivers + CUDA Toolkit 12.4
- Service user `kza` with audio/video/bluetooth groups
- Installation directory `/opt/kza` with data/models/logs subdirs
- Python 3.13 venv + pip dependencies
- systemd service installation
- System optimizations (file limits, audio rtprio, hugepages)

**Validation:**
- `nvidia-smi` shows 4x RTX 3070
- `nvcc --version` shows CUDA
- `arecord -l` shows audio devices
- Ping to Home Assistant IP

### Phase 2: Python Environment

Handled by setup_ubuntu.sh, but verify:

**Validation:**
- `python3.13 --version` → 3.13.x
- venv activated
- `pip install -r requirements.txt` clean (with updated deps)
- `python3 -c "import torch; print(torch.cuda.device_count())"` → 4

### Phase 3: Models

Run the updated `download_models.sh`.

**Validation:**
- All model directories populated
- `smoke_test.sh --quick` passes

### Phase 4: Configuration

- Copy `.env.example` → `.env`, fill in HA token + Spotify credentials
- Verify `settings.yaml` has correct model paths (AWQ router, bge-m3, whisper-es)
- Detect actual mic device_index: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Update `rooms.escritorio.mic_device_index` if different from 4

**Validation:**
- `.env` has real values (not placeholders)
- `settings.yaml` model paths match downloaded files

### Phase 5: Home Assistant + Pipeline Start

**Validation:**
- `curl -H "Authorization: Bearer $TOKEN" http://HA_IP:8123/api/` → HTTP 200
- `python -m src.main` starts without crash
- ChromaDB entity sync completes (check logs — may take 2-10 min on first run with many entities)
- All 4 GPUs show memory usage in `nvidia-smi`

### Phase 6: End-to-End Pipeline (escritorio)

**Validation:**
- Wake word "hey jarvis" detected
- STT transcribes Spanish correctly
- "Prende la luz del escritorio" → HA executes turn_on on `light.escritorio`
- TTS responds through escritorio speaker
- Total latency < 300ms

### Phase 7: Extra Features

**Validation:**
- "Que hora es" → Router 7B responds via TTS
- "Pon musica en el escritorio" → Spotify plays
- "Agrega leche a la lista de compras" → list updated
- Dashboard API: `curl http://127.0.0.1:8080/api/health` responds (if Fix 7 applied)
- Speaker identification works (if enrolled)

### Follow-up: systemd Production Activation

After manual testing succeeds:
```bash
sudo systemctl enable kza-voice
sudo systemctl start kza-voice
journalctl -u kza-voice -f  # Monitor logs
```

---

## 3. Risk Matrix

| Risk | Probability | Impact | Mitigation | Fallback | Time |
|------|------------|--------|------------|----------|------|
| Mic device_index differs from config | High | Low | Run `sounddevice.query_devices()`, update YAML | — | 2 min |
| vLLM can't load AWQ model | Medium | Medium | Ensure `autoawq` installed | Use Qwen2.5-3B fp16 (~6GB) | 5-15 min |
| Qwen 72B download too slow | Low | Low | System works without LLM (fast path only) | Skip slow path, test later | 0 min |
| HA not reachable from server | Medium | High | Check IP, port 8123, firewall, token | Blocker — must fix | 5-30 min |
| Kokoro/Qwen3-TTS fail on GPU 3 | Low-Medium | Medium | — | Switch to `tts.engine: "piper"` (CPU) | 2 min |
| Wake word too sensitive/insensitive | Medium | Low | Adjust `wake_word.threshold` (0.3-0.7) | Iterative tuning | 5-10 min |
| ChromaDB initial sync slow | Medium | Low | Wait for completion, monitor logs | Skip sync, test with manual commands | 2-10 min |

---

## 4. Out of Scope

- **Multi-room:** Configure after single-room validated
- **Nightly training:** Activate once conversation data exists; validate LoRA-on-AWQ compatibility then
- **Speaker enrollment:** Requires working pipeline first
- **Dashboard frontend:** Nice-to-have, not critical for day one
- **Spotify follow mode:** Requires presence detection working
- **Wake word custom training:** Uses pre-trained "hey_jarvis" for now

---

## 5. Implementation Summary

| # | Item | Type | Files |
|---|------|------|-------|
| 1 | Router model → AWQ | Config change | `config/settings.yaml` |
| 2 | Nightly base_model stays fp16 (LoRA compat) | Design decision | `config/settings.yaml` (no change needed) |
| 3 | Wire all config params to FastRouter + LLMReasoner | Bug fix | `src/main.py` |
| 4 | Rewrite download_models.sh (non-interactive) | Script rewrite | `scripts/download_models.sh` |
| 5 | Fix Python version check + CPU comment | Script fix | `scripts/validate_hardware.py` |
| 6 | Add missing deps (autoawq, speechbrain, psutil, fastapi, uvicorn) | Dependency add | `requirements.txt` |
| 7 | Start dashboard API in main.py | Feature wiring | `src/main.py` |
| 8 | Update .env.example | Doc fix | `.env.example` |
