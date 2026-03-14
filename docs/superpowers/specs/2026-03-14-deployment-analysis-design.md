# KZA Deployment Analysis — Single-Room Testing

**Date:** 2026-03-14
**Goal:** Deploy KZA on the production server and validate end-to-end voice pipeline in the escritorio room.
**Hardware:** Threadripper PRO 9965WX, 128GB DDR5, 4x RTX 3070 8GB
**Testing room:** Escritorio (mic_device_index TBD, ma1260_zone 4, bt_adapter hci3)

---

## 1. Code Fixes Required Before Deploy

### Fix 1: Router model — fp16 → AWQ

**Problem:** `Qwen/Qwen2.5-7B-Instruct` in fp16 is ~14GB — does not fit in 8GB VRAM.

**Files:**
- `config/settings.yaml` line 175: change model to `Qwen/Qwen2.5-7B-Instruct-AWQ`

**Dependency:** `autoawq` package must be in `requirements.txt` for vLLM AWQ support.

### Fix 2: Nightly training base_model must match router

**Problem:** Nightly training generates LoRA adapters for the router. If the router uses AWQ, the base_model must also be AWQ for adapter compatibility.

**Files:**
- `config/settings.yaml` line 466: change base_model to `Qwen/Qwen2.5-7B-Instruct-AWQ`

### Fix 3: download_models.sh is outdated

**Problem:** The script downloads wrong models — English-only STT, English-only embeddings, Piper TTS instead of Kokoro/Qwen3-TTS, no router download, no emotion model.

| Script downloads | Config expects |
|-----------------|----------------|
| `distil-whisper/distil-small.en` | `marianbasti/distil-whisper-large-v3-es` |
| `BAAI/bge-small-en-v1.5` | `BAAI/bge-m3` |
| Piper TTS only | Kokoro-82M + Qwen3-TTS-0.6B |
| Qwen 72B Q8_0 options | Qwen2.5-72B-Instruct Q6_K |
| No router model | Qwen2.5-7B-Instruct-AWQ |
| No emotion model | wav2vec2 (auto-downloaded by transformers) |
| No Kokoro/Qwen3-TTS | Dual TTS on GPU 3 |

**Fix:** Rewrite the script to download the correct models matching `settings.yaml`.

**Download order (optimized):**
1. Small models first (~5 min): Kokoro-82M, ECAPA-TDNN, OpenWakeWord, wav2vec2
2. Medium models (~10-15 min): distil-whisper-large-v3-es, BGE-M3, Qwen3-TTS-0.6B, Qwen2.5-7B-Instruct-AWQ
3. Large model last (background): Qwen2.5-72B-Instruct-Q6_K (~64GB)

### Fix 4: validate_hardware.py checks Python >= 3.10, should be >= 3.13

**File:** `scripts/validate_hardware.py` line 47: change `version.minor >= 10` to `version.minor >= 13`

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
- `pip install -r requirements.txt` clean
- `python3 -c "import torch; print(torch.cuda.device_count())"` → 4

### Phase 3: Models

Run the updated `download_models.sh`.

**Validation:**
- All model directories populated
- `smoke_test.sh --quick` passes

### Phase 4: Configuration

- Copy `.env.example` → `.env`, fill in HA token + Spotify credentials
- Update `settings.yaml`: router AWQ, nightly base_model AWQ
- Detect actual mic device_index: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Update `rooms.escritorio.mic_device_index` if different from 4

**Validation:**
- `.env` has real values (not placeholders)
- `settings.yaml` model paths match downloaded files

### Phase 5: Home Assistant

**Validation:**
- `curl -H "Authorization: Bearer $TOKEN" http://HA_IP:8123/api/` → HTTP 200
- `python -m src.main` starts without crash
- ChromaDB entity sync completes (check logs)
- Dashboard API: `curl http://127.0.0.1:8080/api/health` responds

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
- Speaker identification works (if enrolled)

---

## 3. Risk Matrix

| Risk | Probability | Impact | Mitigation | Fallback | Time |
|------|------------|--------|------------|----------|------|
| Mic device_index differs from config | High | Low | Run `sounddevice.query_devices()`, update YAML | — | 2 min |
| vLLM can't load AWQ model | Medium | Medium | Add `autoawq` to requirements | Use Qwen2.5-3B fp16 (~6GB) | 5-15 min |
| Qwen 72B download too slow | Low | Low | System works without LLM (fast path only) | Skip slow path, test later | 0 min |
| HA not reachable from server | Medium | High | Check IP, port 8123, firewall, token | Blocker — must fix | 5-30 min |
| Kokoro/Qwen3-TTS fail on GPU 3 | Low-Medium | Medium | — | Switch to `tts.engine: "piper"` (CPU) | 2 min |
| Wake word too sensitive/insensitive | Medium | Low | Adjust `wake_word.threshold` (0.3-0.7) | Iterative tuning | 5-10 min |

---

## 4. Out of Scope

- **Multi-room:** Configure after single-room validated
- **Nightly training:** Activate once conversation data exists
- **Speaker enrollment:** Requires working pipeline first
- **Dashboard frontend:** Nice-to-have, not critical for day one
- **Spotify follow mode:** Requires presence detection working
- **Wake word custom training:** Uses pre-trained "hey_jarvis" for now

---

## 5. Implementation Summary

| Item | Type | Files |
|------|------|-------|
| Router AWQ | Config change | `config/settings.yaml` |
| Nightly base_model AWQ | Config change | `config/settings.yaml` |
| `autoawq` dependency | Dependency add | `requirements.txt` |
| download_models.sh rewrite | Script rewrite | `scripts/download_models.sh` |
| validate_hardware.py fix | Script fix | `scripts/validate_hardware.py` |
