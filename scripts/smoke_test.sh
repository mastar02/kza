#!/bin/bash
#
# KZA Voice Assistant - Smoke Test
# Verifies that the environment is ready to run KZA.
#
# Usage:
#   ./scripts/smoke_test.sh          # Full smoke test
#   ./scripts/smoke_test.sh --quick  # Quick check (for systemd ExecStartPre)
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#

set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
WARN=0

QUICK_MODE=false
if [ "${1:-}" = "--quick" ]; then
    QUICK_MODE=true
fi

# Resolve project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

check_pass() {
    echo -e "  ${GREEN}PASS${NC}  $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo -e "  ${RED}FAIL${NC}  $1"
    FAIL=$((FAIL + 1))
}

check_warn() {
    echo -e "  ${YELLOW}WARN${NC}  $1"
    WARN=$((WARN + 1))
}

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo "============================================================"
    echo "  KZA Voice Assistant - Smoke Test"
    echo "============================================================"
    echo ""
fi

# ----------------------------------------------------------
# 1. Python 3.13 venv
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo -e "${BLUE}[Python Environment]${NC}"
fi

if [ -d "venv" ] && [ -f "venv/bin/python3" ]; then
    PY_VER=$(venv/bin/python3 --version 2>&1)
    if echo "$PY_VER" | grep -q "3\.13"; then
        check_pass "Python 3.13 venv exists (${PY_VER})"
    else
        check_fail "Venv exists but wrong Python version: ${PY_VER} (expected 3.13)"
    fi
else
    check_fail "Python venv not found at ./venv"
fi

# Verify pip works
if venv/bin/pip --version &>/dev/null; then
    check_pass "pip is functional"
else
    check_fail "pip not working in venv"
fi

# ----------------------------------------------------------
# 2. Required Python packages
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Python Packages]${NC}"

    REQUIRED_PACKAGES=(
        "torch:PyTorch"
        "faster_whisper:faster-whisper (STT)"
        "chromadb:ChromaDB (vector store)"
        "sentence_transformers:sentence-transformers (embeddings)"
        "vllm:vLLM (router model)"
        "sounddevice:sounddevice (audio I/O)"
        "soundfile:soundfile (audio files)"
        "yaml:PyYAML (config)"
        "dotenv:python-dotenv (.env)"
        "aiohttp:aiohttp (HA client)"
        "websockets:websockets (HA WS)"
        "openwakeword:openwakeword (wake word)"
        "pydantic:pydantic (validation)"
        "numpy:numpy"
    )

    for pkg_entry in "${REQUIRED_PACKAGES[@]}"; do
        PKG_IMPORT="${pkg_entry%%:*}"
        PKG_NAME="${pkg_entry##*:}"
        if venv/bin/python3 -c "import ${PKG_IMPORT}" 2>/dev/null; then
            check_pass "${PKG_NAME}"
        else
            check_fail "${PKG_NAME} (cannot import ${PKG_IMPORT})"
        fi
    done
fi

# ----------------------------------------------------------
# 3. GPU access
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[GPU Access]${NC}"
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -ge 4 ]; then
        check_pass "nvidia-smi: ${GPU_COUNT} GPUs detected"
    elif [ "$GPU_COUNT" -ge 1 ]; then
        check_warn "nvidia-smi: ${GPU_COUNT} GPUs (expected 4)"
    else
        check_fail "nvidia-smi: no GPUs detected"
    fi
else
    check_fail "nvidia-smi not found"
fi

# Verify torch CUDA access
if [ "$QUICK_MODE" = false ]; then
    CUDA_RESULT=$(venv/bin/python3 -c "
import torch
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print(f'{available}|{count}')
" 2>/dev/null || echo "error|0")

    CUDA_AVAIL="${CUDA_RESULT%%|*}"
    CUDA_COUNT="${CUDA_RESULT##*|}"

    if [ "$CUDA_AVAIL" = "True" ]; then
        if [ "$CUDA_COUNT" -ge 4 ]; then
            check_pass "torch.cuda: ${CUDA_COUNT} devices available"
        else
            check_warn "torch.cuda: ${CUDA_COUNT} devices (expected 4)"
        fi
    else
        check_fail "torch.cuda not available"
    fi
fi

# ----------------------------------------------------------
# 4. Audio devices
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Audio Devices]${NC}"

    if command -v arecord &>/dev/null; then
        AUDIO_DEVS=$(arecord -l 2>/dev/null | grep -c "card" || echo "0")
        if [ "$AUDIO_DEVS" -gt 0 ]; then
            check_pass "Audio capture devices: ${AUDIO_DEVS}"
        else
            check_warn "No audio capture devices (arecord -l)"
        fi
    else
        check_warn "arecord not installed (alsa-utils)"
    fi

    # Check sounddevice can list devices
    SD_RESULT=$(venv/bin/python3 -c "
import sounddevice as sd
devs = sd.query_devices()
inputs = [d for d in devs if d['max_input_channels'] > 0]
print(len(inputs))
" 2>/dev/null || echo "error")

    if [ "$SD_RESULT" != "error" ] && [ "$SD_RESULT" -gt 0 ] 2>/dev/null; then
        check_pass "sounddevice input devices: ${SD_RESULT}"
    else
        check_warn "sounddevice cannot detect input devices"
    fi
fi

# ----------------------------------------------------------
# 5. Home Assistant connectivity
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Home Assistant]${NC}"

    # Load .env
    if [ -f ".env" ]; then
        HA_URL=$(grep "^HOME_ASSISTANT_URL" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs)
        HA_TOKEN=$(grep "^HOME_ASSISTANT_TOKEN" .env 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs)

        if [ -n "$HA_URL" ] && [ "$HA_URL" != "http://192.168.1.x:8123" ] && [ -n "$HA_TOKEN" ] && [ "$HA_TOKEN" != "your_long_lived_token_here" ]; then
            # Test connectivity
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer ${HA_TOKEN}" \
                -H "Content-Type: application/json" \
                "${HA_URL}/api/" \
                --connect-timeout 5 --max-time 10 2>/dev/null || echo "000")

            if [ "$HTTP_CODE" = "200" ]; then
                check_pass "Home Assistant reachable at ${HA_URL} (HTTP 200)"
            elif [ "$HTTP_CODE" = "401" ]; then
                check_fail "Home Assistant reachable but token invalid (HTTP 401)"
            elif [ "$HTTP_CODE" = "000" ]; then
                check_fail "Home Assistant not reachable at ${HA_URL} (connection timeout)"
            else
                check_warn "Home Assistant returned HTTP ${HTTP_CODE}"
            fi
        else
            check_warn "HOME_ASSISTANT_URL or TOKEN not configured in .env"
        fi
    else
        check_fail ".env file not found"
    fi
fi

# ----------------------------------------------------------
# 6. Configuration files
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Configuration]${NC}"
fi

if [ -f "config/settings.yaml" ]; then
    # Validate YAML syntax
    YAML_OK=$(venv/bin/python3 -c "
import yaml
try:
    with open('config/settings.yaml') as f:
        yaml.safe_load(f)
    print('ok')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "error")

    if [ "$YAML_OK" = "ok" ]; then
        check_pass "config/settings.yaml exists and is valid YAML"
    else
        check_fail "config/settings.yaml has invalid YAML: ${YAML_OK}"
    fi
else
    check_fail "config/settings.yaml not found"
fi

if [ -f ".env" ]; then
    check_pass ".env file exists"
else
    check_fail ".env file not found"
fi

# ----------------------------------------------------------
# 7. Data directories
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Data Directories]${NC}"
fi

REQUIRED_DIRS=(
    "data:data/"
    "data/chroma_db:data/chroma_db/"
    "logs:logs/"
    "models:models/"
)

for dir_entry in "${REQUIRED_DIRS[@]}"; do
    DIR_PATH="${dir_entry%%:*}"
    DIR_NAME="${dir_entry##*:}"
    if [ -d "$DIR_PATH" ]; then
        if [ "$QUICK_MODE" = false ]; then
            check_pass "${DIR_NAME}"
        fi
    else
        check_fail "${DIR_NAME} does not exist"
    fi
done

# Check write permissions on data/
if [ -w "data/" ] 2>/dev/null; then
    if [ "$QUICK_MODE" = false ]; then
        check_pass "data/ is writable"
    fi
else
    check_fail "data/ is not writable"
fi

# ----------------------------------------------------------
# 8. Source code
# ----------------------------------------------------------
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}[Source Code]${NC}"

    if [ -f "src/main.py" ]; then
        check_pass "src/main.py exists"
    else
        check_fail "src/main.py not found"
    fi

    # Verify basic import works
    IMPORT_OK=$(venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from src.health.health_check import HealthChecker
print('ok')
" 2>/dev/null || echo "error")

    if [ "$IMPORT_OK" = "ok" ]; then
        check_pass "Core imports work (HealthChecker)"
    else
        check_warn "Core imports failed (may need dependencies)"
    fi
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""

if [ "$QUICK_MODE" = true ]; then
    if [ "$FAIL" -gt 0 ]; then
        echo -e "${RED}Quick check: ${FAIL} failure(s)${NC}" >&2
        exit 1
    fi
    exit 0
fi

TOTAL=$((PASS + FAIL + WARN))
echo "============================================================"
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC} (${TOTAL} total)"
echo "============================================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}SMOKE TEST FAILED${NC} — fix the failures above before starting KZA."
    echo ""
    exit 1
else
    if [ "$WARN" -gt 0 ]; then
        echo -e "  ${YELLOW}SMOKE TEST PASSED WITH WARNINGS${NC} — KZA may start but some features could be limited."
    else
        echo -e "  ${GREEN}SMOKE TEST PASSED${NC} — environment is ready."
    fi
    echo ""
    exit 0
fi
