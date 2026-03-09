#!/bin/bash
#
# KZA Voice Assistant - Ubuntu Setup Script
# For Ubuntu 22.04/24.04 LTS Server
# Hardware: Threadripper PRO 7965WX + 4x RTX 3070 8GB
#
# Usage: sudo ./scripts/setup_ubuntu.sh
#
# Environment variables (optional overrides):
#   KZA_USER     - service user (default: kza)
#   INSTALL_DIR  - installation path (default: /opt/kza)
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
log_error(){ echo -e "${RED}[FAIL]${NC} $1"; }

echo ""
echo "============================================================"
echo "  KZA Voice Assistant - Ubuntu Setup"
echo "  Target: Ubuntu 22.04/24.04 + Threadripper PRO + 4x RTX 3070"
echo "  Python: 3.13 (deadsnakes PPA)"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# 0. Pre-checks
# ----------------------------------------------------------
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root"
    log_info "Usage: sudo $0"
    exit 1
fi

KZA_USER="${KZA_USER:-kza}"
INSTALL_DIR="${INSTALL_DIR:-/opt/kza}"
PYTHON_VERSION="3.13"
PYTHON_BIN="python${PYTHON_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SUMMARY=()
summary_ok()   { SUMMARY+=("${GREEN}[ OK ]${NC} $1"); }
summary_warn() { SUMMARY+=("${YELLOW}[WARN]${NC} $1"); }

# ----------------------------------------------------------
# 1. System update
# ----------------------------------------------------------
log_info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
log_ok "System updated"

# ----------------------------------------------------------
# 2. Install Python 3.13 from deadsnakes PPA
# ----------------------------------------------------------
log_info "Installing Python ${PYTHON_VERSION} from deadsnakes PPA..."

apt-get install -y -qq software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq

apt-get install -y -qq \
    "${PYTHON_BIN}" \
    "${PYTHON_BIN}-venv" \
    "${PYTHON_BIN}-dev" \
    "${PYTHON_BIN}-distutils" 2>/dev/null || true

# Verify Python 3.13
if command -v "$PYTHON_BIN" &> /dev/null; then
    INSTALLED_PY=$("$PYTHON_BIN" --version 2>&1)
    log_ok "Python installed: ${INSTALLED_PY}"
    summary_ok "Python ${INSTALLED_PY}"
else
    log_error "Failed to install Python ${PYTHON_VERSION}"
    exit 1
fi

# ----------------------------------------------------------
# 3. Install system dependencies
# ----------------------------------------------------------
log_info "Installing system dependencies..."

apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    htop \
    nvtop \
    tmux \
    portaudio19-dev \
    libsndfile1-dev \
    ffmpeg \
    alsa-utils \
    pulseaudio \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    espeak-ng \
    libespeak-ng-dev \
    jq \
    bluetooth \
    bluez \
    libbluetooth-dev

log_ok "System dependencies installed"
summary_ok "System dependencies (portaudio, libsndfile, ffmpeg, alsa, bluetooth)"

# ----------------------------------------------------------
# 4. Verify / install NVIDIA drivers and CUDA
# ----------------------------------------------------------
log_info "Checking NVIDIA drivers..."

if ! command -v nvidia-smi &> /dev/null; then
    log_info "Installing NVIDIA drivers..."
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update -qq
    ubuntu-drivers autoinstall

    log_warn "NVIDIA drivers installed. REBOOT the system, then re-run this script."
    summary_warn "NVIDIA drivers installed — REBOOT REQUIRED"
    exit 0
else
    DRIVER_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
    log_ok "NVIDIA driver: ${DRIVER_INFO}"
    summary_ok "NVIDIA driver: ${DRIVER_INFO}"
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_info "GPUs detected: ${GPU_COUNT}"
if [ "$GPU_COUNT" -lt 4 ]; then
    log_warn "Expected 4 GPUs, found: ${GPU_COUNT} — some features may be limited"
    summary_warn "GPUs: ${GPU_COUNT}/4 detected"
else
    summary_ok "GPUs: ${GPU_COUNT} detected (4x RTX 3070 expected)"
fi

# CUDA Toolkit
if ! command -v nvcc &> /dev/null; then
    log_info "Installing CUDA Toolkit 12.4..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4

    # Add to system PATH
    cat > /etc/profile.d/cuda.sh << 'CUDAEOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
CUDAEOF

    log_ok "CUDA Toolkit 12.4 installed"
    summary_ok "CUDA Toolkit 12.4 installed"
else
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_ok "CUDA already installed: ${CUDA_VER}"
    summary_ok "CUDA Toolkit ${CUDA_VER}"
fi

# ----------------------------------------------------------
# 5. Create service user
# ----------------------------------------------------------
log_info "Configuring user ${KZA_USER}..."

if ! id "$KZA_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash -G audio,video,bluetooth "$KZA_USER"
    log_ok "User ${KZA_USER} created"
else
    log_ok "User ${KZA_USER} already exists"
fi

usermod -a -G audio,video,bluetooth "$KZA_USER"
summary_ok "Service user: ${KZA_USER}"

# ----------------------------------------------------------
# 6. Create installation directory and required subdirs
# ----------------------------------------------------------
log_info "Setting up installation directory ${INSTALL_DIR}..."

mkdir -p "${INSTALL_DIR}/data/chroma_db"
mkdir -p "${INSTALL_DIR}/data/memory_db"
mkdir -p "${INSTALL_DIR}/data/contexts"
mkdir -p "${INSTALL_DIR}/logs"
mkdir -p "${INSTALL_DIR}/models/lora_adapters"

# Copy project files if running from repo checkout
if [ -f "$PROJECT_DIR/src/main.py" ]; then
    log_info "Copying project files from ${PROJECT_DIR}..."
    cp -r "$PROJECT_DIR/src" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/config" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/scripts" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/systemd" "$INSTALL_DIR/"
    cp "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    cp "$PROJECT_DIR/.env.example" "$INSTALL_DIR/" 2>/dev/null || true
    log_ok "Project files copied"
fi

chown -R "${KZA_USER}:${KZA_USER}" "$INSTALL_DIR"
log_ok "Directory ${INSTALL_DIR} configured"
summary_ok "Install dir: ${INSTALL_DIR} (data/, models/, logs/ created)"

# ----------------------------------------------------------
# 7. Create Python 3.13 venv and install pip dependencies
# ----------------------------------------------------------
log_info "Setting up Python ${PYTHON_VERSION} virtual environment..."

sudo -u "$KZA_USER" bash << VENVEOF
set -e
cd "$INSTALL_DIR"

# Create venv with Python 3.13
${PYTHON_BIN} -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools -q

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Install project requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
fi

echo "Python venv ready: \$(python --version), torch \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
VENVEOF

log_ok "Python ${PYTHON_VERSION} venv created and dependencies installed"
summary_ok "Python venv: ${INSTALL_DIR}/venv (Python ${PYTHON_VERSION})"

# ----------------------------------------------------------
# 8. Copy .env.example to .env if not exists
# ----------------------------------------------------------
if [ ! -f "$INSTALL_DIR/.env" ]; then
    if [ -f "$INSTALL_DIR/.env.example" ]; then
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
        chown "${KZA_USER}:${KZA_USER}" "$INSTALL_DIR/.env"
        chmod 600 "$INSTALL_DIR/.env"
        log_warn ".env created from .env.example — EDIT IT with your HA token and Spotify credentials"
        summary_warn ".env copied from .env.example — NEEDS EDITING"
    else
        log_warn ".env.example not found; create .env manually"
        summary_warn ".env not created — no .env.example found"
    fi
else
    log_ok ".env already exists"
    summary_ok ".env already configured"
fi

# ----------------------------------------------------------
# 9. Install systemd service
# ----------------------------------------------------------
log_info "Installing systemd service..."

if [ -f "$INSTALL_DIR/systemd/kza-voice.service" ]; then
    cp "$INSTALL_DIR/systemd/kza-voice.service" /etc/systemd/system/

    # Adjust paths if INSTALL_DIR is not /opt/kza
    if [ "$INSTALL_DIR" != "/opt/kza" ]; then
        sed -i "s|/opt/kza|${INSTALL_DIR}|g" /etc/systemd/system/kza-voice.service
    fi
    sed -i "s|User=kza|User=${KZA_USER}|g" /etc/systemd/system/kza-voice.service
    sed -i "s|Group=kza|Group=${KZA_USER}|g" /etc/systemd/system/kza-voice.service

    systemctl daemon-reload
    log_ok "systemd service installed (kza-voice.service)"
    summary_ok "systemd service: kza-voice"
else
    log_warn "systemd/kza-voice.service not found — skipped"
    summary_warn "systemd service not installed"
fi

# ----------------------------------------------------------
# 10. Verify audio devices
# ----------------------------------------------------------
log_info "Checking audio devices..."

AUDIO_DEVS=$(arecord -l 2>/dev/null | grep -c "card" || echo "0")
if [ "$AUDIO_DEVS" -gt 0 ]; then
    log_ok "Audio capture devices found: ${AUDIO_DEVS}"
    arecord -l 2>/dev/null | grep "card" | while read -r line; do
        echo "  ${line}"
    done
    summary_ok "Audio devices: ${AUDIO_DEVS} capture device(s)"
else
    log_warn "No audio capture devices detected (ReSpeaker may need USB connection)"
    summary_warn "No audio capture devices found"
fi

# ----------------------------------------------------------
# 11. System optimizations
# ----------------------------------------------------------
log_info "Applying system optimizations..."

# File descriptor limits
if ! grep -q "# KZA Voice Assistant" /etc/security/limits.conf 2>/dev/null; then
    cat >> /etc/security/limits.conf << LIMEOF

# KZA Voice Assistant
${KZA_USER} soft nofile 65535
${KZA_USER} hard nofile 65535
${KZA_USER} soft nproc  4096
${KZA_USER} hard nproc  4096
LIMEOF
fi

# Audio real-time priority
if [ ! -f /etc/security/limits.d/audio.conf ]; then
    cat > /etc/security/limits.d/audio.conf << AUDIOEOF
@audio - rtprio 95
@audio - memlock unlimited
AUDIOEOF
fi

# Hugepages for LLM 70B (optional, helps with large memory allocations)
if ! grep -q "kza" /etc/sysctl.d/99-kza.conf 2>/dev/null; then
    cat > /etc/sysctl.d/99-kza.conf << SYSEOF
# KZA Voice Assistant — system tuning
vm.nr_hugepages=1024
net.core.rmem_max=16777216
net.core.wmem_max=16777216
SYSEOF
    sysctl -p /etc/sysctl.d/99-kza.conf 2>/dev/null || true
fi

log_ok "System optimizations applied"
summary_ok "System tuning: file limits, audio rtprio, hugepages"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Summary:"
echo ""
for item in "${SUMMARY[@]}"; do
    echo -e "    ${item}"
done
echo ""
echo "  Next steps:"
echo ""
echo "    1. Edit configuration:"
echo "       sudo -u ${KZA_USER} nano ${INSTALL_DIR}/.env"
echo ""
echo "    2. Download models:"
echo "       sudo -u ${KZA_USER} ${INSTALL_DIR}/scripts/download_models.sh"
echo ""
echo "    3. Run smoke test:"
echo "       sudo -u ${KZA_USER} ${INSTALL_DIR}/scripts/smoke_test.sh"
echo ""
echo "    4. Test manually:"
echo "       sudo -u ${KZA_USER} ${INSTALL_DIR}/scripts/start.sh"
echo ""
echo "    5. Enable and start the service:"
echo "       sudo systemctl enable kza-voice"
echo "       sudo systemctl start kza-voice"
echo ""
echo "    6. View logs:"
echo "       journalctl -u kza-voice -f"
echo ""
echo "============================================================"
