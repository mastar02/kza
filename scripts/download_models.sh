#!/bin/bash
#
# KZA Voice Assistant - Model Downloader
# Non-interactive. Downloads all models matching config/settings.yaml.
#
# Usage:
#   ./scripts/download_models.sh             # Download all models
#   ./scripts/download_models.sh --skip-llm  # Skip the 72B LLM (saves ~64GB)
#

set -euo pipefail

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"
mkdir -p "$MODELS_DIR/lora_adapters"
mkdir -p "./data/chroma_db"
mkdir -p "./data/memory_db"
mkdir -p "./logs"

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
