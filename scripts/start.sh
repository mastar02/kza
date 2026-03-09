#!/bin/bash
#
# KZA Voice Assistant - Startup Script
# For Ubuntu Server with Threadripper PRO + 4x RTX 3070
#
# Usage: ./scripts/start.sh [--skip-checks]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Resolve project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error(){ echo -e "${RED}[FAIL]${NC} $1"; }

SKIP_CHECKS=false
if [ "${1:-}" = "--skip-checks" ]; then
    SKIP_CHECKS=true
fi

echo ""
echo "=================================================="
echo "  KZA Voice Assistant - Home Automation"
echo "  Hardware: Threadripper PRO + 4x RTX 3070"
echo "=================================================="
echo ""

# ----------------------------------------------------------
# 1. Activate Python 3.13 venv
# ----------------------------------------------------------
log_info "Activating Python virtual environment..."

if [ ! -d "venv" ]; then
    log_error "Python venv not found at ${PROJECT_DIR}/venv"
    log_info "Run scripts/setup_ubuntu.sh first, or create manually:"
    log_info "  python3.13 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

PYTHON_VERSION=$(python3 --version 2>&1)
if echo "$PYTHON_VERSION" | grep -q "3\.13"; then
    log_ok "Venv activated: ${PYTHON_VERSION}"
else
    log_warn "Expected Python 3.13, got: ${PYTHON_VERSION}"
fi

# ----------------------------------------------------------
# 2. GPU environment variables
# ----------------------------------------------------------
# All 4 GPUs visible to the process; each component selects its own via config
# cuda:0=STT, cuda:1=Embeddings/Speaker, cuda:2=Router, cuda:3=TTS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Prevent CUDA from caching kernels (saves disk in /tmp)
export CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-0}"

# CUDA library path
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

log_ok "GPU config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ----------------------------------------------------------
# 3. Load .env
# ----------------------------------------------------------
if [ ! -f ".env" ]; then
    log_error ".env not found"
    log_info "Copy .env.example to .env and configure your values"
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
log_ok ".env loaded"

# Verify critical variables
if [ -z "${HOME_ASSISTANT_URL:-}" ] || [ -z "${HOME_ASSISTANT_TOKEN:-}" ]; then
    log_error "HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN must be set in .env"
    exit 1
fi
log_ok "HA config: ${HOME_ASSISTANT_URL}"

# ----------------------------------------------------------
# 4. Pre-flight checks (smoke test)
# ----------------------------------------------------------
if [ "$SKIP_CHECKS" = false ]; then
    log_info "Running pre-flight checks..."

    if [ -x "${SCRIPT_DIR}/smoke_test.sh" ]; then
        if "${SCRIPT_DIR}/smoke_test.sh" --quick; then
            log_ok "Pre-flight checks passed"
        else
            log_error "Pre-flight checks failed"
            log_info "Run ${SCRIPT_DIR}/smoke_test.sh for detailed diagnostics"
            log_info "Or use --skip-checks to bypass"
            exit 1
        fi
    else
        # Minimal inline checks if smoke_test.sh is not available
        log_info "smoke_test.sh not found, running minimal checks..."

        # Check GPUs
        if command -v nvidia-smi &>/dev/null; then
            GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
            log_info "GPUs detected: ${GPU_COUNT}"
            if [ "$GPU_COUNT" -lt 4 ]; then
                log_warn "Expected 4 GPUs, found: ${GPU_COUNT}"
            fi
        else
            log_warn "nvidia-smi not found"
        fi

        # Check PyTorch
        python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
            log_error "PyTorch not importable"
            exit 1
        }

        # Check config
        if [ ! -f "config/settings.yaml" ]; then
            log_error "config/settings.yaml not found"
            exit 1
        fi

        # Ensure data dirs exist
        mkdir -p data/chroma_db data/memory_db data/contexts logs

        log_ok "Minimal checks passed"
    fi
else
    log_warn "Pre-flight checks skipped (--skip-checks)"
fi

# ----------------------------------------------------------
# 5. Verify models (non-blocking warnings)
# ----------------------------------------------------------
log_info "Checking models..."

MODELS_DIR="${MODELS_PATH:-./models}"

if [ -d "$MODELS_DIR" ]; then
    MODEL_COUNT=$(find "$MODELS_DIR" -type f \( -name "*.gguf" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.onnx" \) 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        log_ok "Models directory: ${MODEL_COUNT} model file(s) found"
    else
        log_warn "No model files found in ${MODELS_DIR}"
        log_info "Download with: ./scripts/download_models.sh"
    fi
else
    log_warn "Models directory not found: ${MODELS_DIR}"
fi

# ----------------------------------------------------------
# 6. System resource limits
# ----------------------------------------------------------
ulimit -n 65535 2>/dev/null || true

# ----------------------------------------------------------
# 7. Graceful shutdown handling
# ----------------------------------------------------------
CHILD_PID=""

cleanup() {
    log_info "Shutdown signal received, stopping KZA..."
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        # Send SIGTERM and wait up to 25 seconds for graceful shutdown
        kill -TERM "$CHILD_PID" 2>/dev/null
        local WAIT=0
        while kill -0 "$CHILD_PID" 2>/dev/null && [ $WAIT -lt 25 ]; do
            sleep 1
            WAIT=$((WAIT + 1))
        done
        # Force kill if still running
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            log_warn "Process did not stop gracefully, sending SIGKILL..."
            kill -KILL "$CHILD_PID" 2>/dev/null
        fi
    fi
    log_info "KZA stopped"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

# ----------------------------------------------------------
# 8. Start KZA
# ----------------------------------------------------------
echo ""
log_info "Starting KZA Voice Assistant..."
echo ""
echo "  Wake word:   ${WAKE_WORD:-hey_jarvis}"
echo "  Latency:     <300ms target"
echo "  HA:          ${HOME_ASSISTANT_URL}"
echo "  GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "  Python:      ${PYTHON_VERSION}"
echo ""
echo "  Press Ctrl+C to stop"
echo ""
echo "--------------------------------------------------"
echo ""

# Run as background process so trap can handle signals
python3 -m src.main &
CHILD_PID=$!

# Wait for the child process
wait "$CHILD_PID"
EXIT_CODE=$?
CHILD_PID=""

if [ $EXIT_CODE -ne 0 ]; then
    log_error "KZA exited with code ${EXIT_CODE}"
fi

exit $EXIT_CODE
