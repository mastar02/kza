# KZA Deployment Fixes — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 issues in the codebase so KZA can deploy on the production server tomorrow.

**Architecture:** Config corrections, dependency additions, two bug fixes in `main.py` (constructor wiring), one script rewrite (`download_models.sh`), and two minor script/doc fixes. No new modules, no architectural changes.

**Tech Stack:** Python 3.13, vLLM (AWQ), llama-cpp-python, FastAPI/uvicorn, bash

**Spec:** `docs/superpowers/specs/2026-03-14-deployment-analysis-design.md`

---

## Chunk 1: Config + Dependencies (Tasks 1-3)

### Task 1: Router model AWQ + settings.yaml config fix

**Files:**
- Modify: `config/settings.yaml`

- [ ] **Step 1: Change router model to AWQ**

In `config/settings.yaml`, find key `router.model` and change:

```yaml
# BEFORE
router:
  model: "Qwen/Qwen2.5-7B-Instruct"

# AFTER
router:
  model: "Qwen/Qwen2.5-7B-Instruct-AWQ"
```

- [ ] **Step 2: Verify nightly base_model stays fp16**

Confirm `training.nightly.base_model` is already `Qwen/Qwen2.5-7B-Instruct` (fp16). This is correct — QLoRA trains on fp16, adapter loads into AWQ at inference. **No change needed.**

- [ ] **Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "fix: change router model to AWQ (7B fp16 doesn't fit 8GB VRAM)"
```

---

### Task 2: Add missing dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Verify which deps are actually imported**

Check these imports exist in the codebase:
- `speechbrain` → `src/users/speaker_identifier.py`
- `fastapi` → `src/dashboard/api.py`
- `psutil` → `scripts/validate_hardware.py`

Spotify uses `aiohttp` directly (no `spotipy`). TTS uses custom integration (check Kokoro/Qwen3-TTS package names on PyPI).

- [ ] **Step 2: Add dependencies**

Add these lines to `requirements.txt` in the appropriate sections:

```
# Speaker Identification
speechbrain>=1.0.0

# Dashboard
fastapi>=0.110.0
uvicorn[standard]>=0.27.0

# AWQ quantization (for vLLM)
autoawq>=0.2.0

# System monitoring
psutil>=5.9.0
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "fix: add missing deps (speechbrain, fastapi, uvicorn, autoawq, psutil)"
```

---

### Task 3: Fix validate_hardware.py

**Files:**
- Modify: `scripts/validate_hardware.py`

- [ ] **Step 1: Fix Python version check**

In function `check_python_version()`, change:

```python
# BEFORE (line 47)
if version.major >= 3 and version.minor >= 10:

# AFTER
if version.major >= 3 and version.minor >= 13:
```

- [ ] **Step 2: Fix version message strings**

In same function, change both occurrences of the message:

```python
# BEFORE (lines 49, 57)
message=f"{version_str} (>=3.10 requerido)"

# AFTER
message=f"{version_str} (>=3.13 requerido)"
```

- [ ] **Step 3: Fix CPU comment**

In function `check_cpu()`, change:

```python
# BEFORE (line 152)
expected_threads = 48  # Threadripper 9965WX

# AFTER
expected_threads = 48  # Threadripper PRO 7965WX
```

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_hardware.py
git commit -m "fix: update validate_hardware.py — Python >=3.13, correct CPU name"
```

---

## Chunk 2: Wire config params in main.py (Tasks 4-5)

### Task 4: Wire all LLMReasoner config params

**Files:**
- Modify: `src/main.py`

**Context:** `LLMReasoner.__init__` (in `src/llm/reasoner.py:55-66`) accepts: `model_path`, `lora_path`, `lora_scale`, `n_ctx`, `n_threads`, `n_batch`, `n_gpu_layers`, `chat_format`, `rope_freq_base`, `rope_freq_scale`. Currently `main.py` only passes 3 of these. The `chat_format` mismatch (default `"llama-3"` vs configured `"chatml"`) would cause garbage output with Qwen2.5.

- [ ] **Step 1: Expand LLMReasoner constructor call**

In `src/main.py`, find the `LLMReasoner()` call (inside the `else:` block around line 149, indented 8 spaces) and replace:

```python
        # BEFORE (inside else: block, 8-space indent)
        llm = LLMReasoner(
            model_path=model_path,
            n_ctx=reasoner_config.get("n_ctx", 8192),
            n_threads=reasoner_config.get("n_threads", 24)
        )

        # AFTER
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

**Note on `rope_freq_base`:** The fallback default `1000000.0` intentionally differs from `LLMReasoner.__init__`'s default of `500000.0`. Qwen2.5 uses RoPE base 1M; the class default of 500K was for Llama models. Since we're now using Qwen2.5-72B, the correct production default is 1M.

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `python3 -m pytest tests/unit/llm/ -v 2>&1 | tail -20`
Expected: all existing LLM tests pass (we only added optional params with defaults)

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "fix: wire all LLMReasoner config params (critical: chat_format for Qwen2.5)"
```

---

### Task 5: Wire FastRouter LoRA params + update default model

**Files:**
- Modify: `src/main.py`

**Context:** `FastRouter.__init__` (in `src/llm/reasoner.py:371-379`) accepts: `model`, `device`, `gpu_memory_utilization`, `enable_prefix_caching`, `enable_lora`, `lora_path`, `max_lora_rank`. Currently `main.py` only passes the first 4.

- [ ] **Step 1: Expand FastRouter constructor call**

In `src/main.py`, find the `FastRouter()` call (inside the `if router_config.get("enabled", True):` block around line 162, indented 8 spaces) and replace:

```python
        # BEFORE (inside if: block, 8-space indent)
        fast_router = FastRouter(
            model=router_config.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            device=router_config.get("device", "cuda:2"),
            gpu_memory_utilization=router_config.get("gpu_memory_utilization", 0.85),
            enable_prefix_caching=router_config.get("enable_prefix_caching", True)
        )

        # AFTER
        fast_router = FastRouter(
            model=router_config.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
            device=router_config.get("device", "cuda:2"),
            gpu_memory_utilization=router_config.get("gpu_memory_utilization", 0.85),
            enable_prefix_caching=router_config.get("enable_prefix_caching", True),
            enable_lora=router_config.get("enable_lora", False),
            lora_path=router_config.get("lora_path"),
            max_lora_rank=router_config.get("max_lora_rank", 32),
        )
```

- [ ] **Step 2: Run existing tests**

Run: `python3 -m pytest tests/unit/llm/ tests/unit/pipeline/ -v 2>&1 | tail -20`
Expected: all pass (optional params with defaults)

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "fix: wire FastRouter LoRA params + default to AWQ model"
```

---

## Chunk 3: Dashboard wiring (Task 6)

### Task 6: Start Dashboard API in main.py

**Files:**
- Modify: `src/main.py`

**Context:** `DashboardAPI` (in `src/dashboard/api.py:116`) needs these constructor args: `routine_scheduler`, `routine_executor`, `presence_detector`, `ha_client`, `list_manager`, `reminder_manager`, `health_aggregator`, `reminder_scheduler`, `host`, `port`, `cors_config`. All are optional (default `None`). The `start()` method is async and runs uvicorn.

- [ ] **Step 1: Add imports**

At the top of `src/main.py`, add after the existing imports:

```python
from src.dashboard.api import DashboardAPI
from src.monitoring.health_aggregator import HealthAggregator
```

- [ ] **Step 2: Create HealthAggregator and DashboardAPI after pipeline assembly**

In `src/main.py`, after the `feature_manager` block (around line 588) and before the `# Nightly Training` section, add:

```python
# ----------------------------------------------------------------
# Dashboard API + Health Aggregator
# ----------------------------------------------------------------
dashboard_config = config.get("dashboard", {})
dashboard = None

if dashboard_config.get("enabled", True):
    health_aggregator = HealthAggregator(
        ha_client=ha_client,
        latency_monitor=latency_monitor,
        priority_queue=getattr(orchestrator, "_queue", None) if orchestrator else None,
        reminder_scheduler=reminder_scheduler,
    )

    dashboard = DashboardAPI(
        routine_scheduler=None,   # RoutineScheduler not yet wired in main.py
        routine_executor=None,    # RoutineExecutor not yet wired in main.py
        presence_detector=presence_detector,
        ha_client=ha_client,
        list_manager=list_manager,
        reminder_manager=reminder_manager,
        health_aggregator=health_aggregator,
        reminder_scheduler=reminder_scheduler,
        host=dashboard_config.get("host", "127.0.0.1"),
        port=dashboard_config.get("port", 8080),
        cors_config=dashboard_config.get("cors"),
    )
    logger.info(f"Dashboard API configured on {dashboard_config.get('host', '127.0.0.1')}:{dashboard_config.get('port', 8080)}")
    # Note: /api/routines/* endpoints will return 500 until RoutineScheduler/Executor
    # are wired. Health, lists, reminders, and presence endpoints work.
```

- [ ] **Step 3: Start dashboard as asyncio task before pipeline.run()**

In `src/main.py`, before `await pipeline.run()` (around line 673), add:

```python
# Start dashboard API as background task
dashboard_task = None
if dashboard:
    dashboard_task = asyncio.create_task(dashboard.start())
    logger.info("Dashboard API started")
```

- [ ] **Step 4: Cancel dashboard task in finally block**

In the `finally` block (around line 677), add before `await pipeline.stop()`:

```python
if dashboard_task:
    dashboard_task.cancel()
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/ -k "dashboard or health" -v 2>&1 | tail -20`
Expected: existing dashboard/health tests pass

- [ ] **Step 6: Commit**

```bash
git add src/main.py
git commit -m "feat: wire Dashboard API + HealthAggregator into main.py"
```

---

## Chunk 4: Scripts (Tasks 7-8)

### Task 7: Rewrite download_models.sh

**Files:**
- Rewrite: `scripts/download_models.sh`

**Requirements:** Non-interactive, downloads models matching `settings.yaml`, ordered small → large.

- [ ] **Step 1: Write the new script**

Replace `scripts/download_models.sh` entirely with:

```bash
#!/bin/bash
#
# KZA Voice Assistant - Model Downloader
# Non-interactive. Downloads all models matching config/settings.yaml.
#
# Usage:
#   ./scripts/download_models.sh             # Download all models
#   ./scripts/download_models.sh --skip-llm  # Skip the 72B LLM (saves ~64GB)
#
# Models are downloaded in order: small → medium → large
# so you can start testing before the big model finishes.
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

SKIP_LLM=false
for arg in "$@"; do
    case $arg in
        --skip-llm) SKIP_LLM=true ;;
    esac
done

# Resolve project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"
mkdir -p "$MODELS_DIR/lora_adapters"
mkdir -p "./data/chroma_db"
mkdir -p "./data/memory_db"
mkdir -p "./logs"

# Determine Python binary
PYTHON="python3"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

echo ""
echo "============================================================"
echo "  KZA Voice Assistant - Model Downloader"
echo "  Non-interactive — downloads all models for settings.yaml"
echo "============================================================"
echo ""

# Verify huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    log_info "Installing huggingface_hub..."
    $PYTHON -m pip install huggingface_hub --quiet
fi

PASS=0
FAIL=0

download_ok()   { log_ok "$1"; PASS=$((PASS + 1)); }
download_fail() { log_fail "$1"; FAIL=$((FAIL + 1)); }

# ==================================================================
# Phase 1: Small models (~5 min total)
# ==================================================================
echo ""
echo -e "${BLUE}Phase 1: Small models${NC}"
echo "------------------------------------------------------------"

# 1a. OpenWakeWord (hey_jarvis)
log_info "OpenWakeWord (hey_jarvis)..."
if $PYTHON -c "
from openwakeword import Model
Model(wakeword_models=['hey_jarvis'], inference_framework='onnx')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "OpenWakeWord hey_jarvis"
else
    download_fail "OpenWakeWord (will retry on first use)"
fi

# 1b. ECAPA-TDNN (speaker identification)
log_info "ECAPA-TDNN (speaker ID)..."
if $PYTHON -c "
from speechbrain.pretrained import EncoderClassifier
EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='./models/speaker_id'
)
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)"
else
    download_fail "ECAPA-TDNN (will retry on first use)"
fi

# 1c. wav2vec2 (emotion detection) — auto-downloaded by transformers
log_info "wav2vec2 (emotion detection)..."
if $PYTHON -c "
from transformers import pipeline
pipe = pipeline('audio-classification', model='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition', device='cpu')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "wav2vec2 emotion detection"
else
    download_fail "wav2vec2 (will retry on first use)"
fi

# 1d. Kokoro-82M (fast TTS)
log_info "Kokoro-82M (fast TTS)..."
if $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('hexgrad/Kokoro-82M', local_dir='./models/kokoro-82m')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "Kokoro-82M"
else
    download_fail "Kokoro-82M"
fi

# ==================================================================
# Phase 2: Medium models (~10-15 min)
# ==================================================================
echo ""
echo -e "${BLUE}Phase 2: Medium models${NC}"
echo "------------------------------------------------------------"

# 2a. distil-whisper-large-v3-es (STT, GPU 0)
log_info "distil-whisper-large-v3-es (STT)..."
if $PYTHON -c "
from faster_whisper import WhisperModel
model = WhisperModel('marianbasti/distil-whisper-large-v3-es', device='cpu', compute_type='int8')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "distil-whisper-large-v3-es"
else
    download_fail "distil-whisper-large-v3-es"
fi

# 2b. BGE-M3 (embeddings, GPU 1)
log_info "BGE-M3 (embeddings)..."
if $PYTHON -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "BGE-M3 (BAAI/bge-m3)"
else
    download_fail "BGE-M3"
fi

# 2c. Qwen3-TTS-0.6B (conversational TTS, GPU 3)
log_info "Qwen3-TTS-0.6B (conversational TTS)..."
if $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-0.6B', local_dir='./models/qwen3-tts')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "Qwen3-TTS-0.6B"
else
    download_fail "Qwen3-TTS-0.6B"
fi

# 2d. Qwen2.5-7B-Instruct-AWQ (router, GPU 2)
log_info "Qwen2.5-7B-Instruct-AWQ (router)..."
if $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', local_dir='./models/qwen2.5-7b-awq')
print('ok')
" 2>/dev/null | grep -q "ok"; then
    download_ok "Qwen2.5-7B-Instruct-AWQ"
else
    download_fail "Qwen2.5-7B-Instruct-AWQ"
fi

# 2e. Piper TTS fallback (CPU) — keep for fallback
PIPER_DIR="$MODELS_DIR/piper"
mkdir -p "$PIPER_DIR"
if [ ! -f "$PIPER_DIR/es_ES-davefx-medium.onnx" ]; then
    log_info "Piper TTS (fallback)..."
    wget -q --show-progress -O "$PIPER_DIR/es_ES-davefx-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx" && \
    wget -q -O "$PIPER_DIR/es_ES-davefx-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json" && \
    download_ok "Piper TTS fallback (es_ES-davefx)" || \
    download_fail "Piper TTS fallback"
else
    download_ok "Piper TTS fallback (already exists)"
fi

# ==================================================================
# Phase 3: Large model (background-friendly)
# ==================================================================
echo ""
echo -e "${BLUE}Phase 3: Large model${NC}"
echo "------------------------------------------------------------"

LLM_FILE="Qwen2.5-72B-Instruct-Q6_K.gguf"
LLM_REPO="bartowski/Qwen2.5-72B-Instruct-GGUF"

if [ "$SKIP_LLM" = true ]; then
    log_warn "Skipping LLM download (--skip-llm flag)"
elif [ -f "$MODELS_DIR/$LLM_FILE" ]; then
    download_ok "$LLM_FILE (already exists)"
else
    log_info "Downloading $LLM_FILE (~64GB) — this will take a while..."
    if huggingface-cli download \
        "$LLM_REPO" \
        "$LLM_FILE" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False; then
        download_ok "$LLM_FILE"
    else
        download_fail "$LLM_FILE"
    fi
fi

# ==================================================================
# Summary
# ==================================================================
echo ""
TOTAL=$((PASS + FAIL))
echo "============================================================"
echo -e "  Results: ${GREEN}${PASS} downloaded${NC}, ${RED}${FAIL} failed${NC} (${TOTAL} total)"
echo "============================================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${YELLOW}Some models failed to download.${NC}"
    echo "  Failed models will be retried on first use."
    echo ""
fi

echo "  Models directory:"
ls -lhS "$MODELS_DIR" 2>/dev/null | grep -v "^total" | head -10
echo ""
echo "  Next: ./scripts/smoke_test.sh"
echo "============================================================"

exit "$FAIL"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/download_models.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/download_models.sh
git commit -m "fix: rewrite download_models.sh — non-interactive, matches settings.yaml"
```

---

### Task 8: Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Replace .env.example contents**

```bash
# Home Assistant Voice - Environment Variables
# Copy this file to .env and fill in your values

# ==================== Home Assistant ====================
# Your Home Assistant instance URL
HOME_ASSISTANT_URL=http://192.168.1.x:8123

# Long-lived access token from Home Assistant
# Create at: Settings > Security > Long-lived access tokens
HOME_ASSISTANT_TOKEN=your_long_lived_token_here

# ==================== GPU Configuration ====================
# GPU 0: STT (distil-whisper-large-v3-es)
# GPU 1: Embeddings (BGE-M3) + Speaker ID (ECAPA) + Emotion (wav2vec2)
# GPU 2: Router (Qwen2.5-7B-Instruct-AWQ via vLLM)
# GPU 3: TTS (Kokoro-82M + Qwen3-TTS-0.6B)
# CPU:   LLM (Qwen2.5-72B-Instruct Q6_K via llama-cpp)
CUDA_VISIBLE_DEVICES=0,1,2,3

# ==================== Configuration ====================
# Path to main configuration file
CONFIG_PATH=config/settings.yaml

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# ==================== Spotify ====================
# Create app at: https://developer.spotify.com/dashboard
# Add redirect URI: http://localhost:8888/callback
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "fix: update .env.example to match current architecture"
```

---

## Execution Order Summary

| Task | Fix # | What | Risk if skipped |
|------|-------|------|-----------------|
| 1 | Fix 1,2 | Router AWQ config | vLLM OOM crash on GPU 2 |
| 2 | Fix 6 | Missing deps | Import crashes |
| 3 | Fix 5 | validate_hardware.py | Wrong Python version check |
| 4 | Fix 3a | LLMReasoner params | **Garbage LLM output** (wrong chat_format) |
| 5 | Fix 3b | FastRouter params | LoRA disabled, wrong default model |
| 6 | Fix 7 | Dashboard wiring | Dashboard unreachable |
| 7 | Fix 4 | download_models.sh | Wrong models downloaded |
| 8 | Fix 8 | .env.example | Confusing for setup |

Tasks 1-5 are **critical path**. Tasks 6-8 are important but won't prevent the voice pipeline from working.
