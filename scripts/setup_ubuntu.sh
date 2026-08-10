#!/bin/bash
#
# KZA Voice Assistant - Ubuntu Setup Script
# For Ubuntu 22.04/24.04 LTS Server
# Hardware: Threadripper PRO 7965WX + 2x RTX 3070 8GB (more GPUs may be added)
#
# Usage: sudo ./scripts/setup_ubuntu.sh
#
# Run once as root to provision the host. After that, KZA runs ROOTLESS:
# everything is managed as the kza user via `systemctl --user` + linger.
# The kza account must NOT have sudo nor root-equivalent groups
# (docker/lxd/kvm/libvirt) — see docs/architecture/ROOTLESS_MIGRATION.md.
#
# Environment variables (optional overrides):
#   KZA_USER     - service user (default: kza)
#   INSTALL_DIR  - installation path (default: /home/kza/kza, symlinked as ~/app)
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
echo "  Target: Ubuntu 22.04/24.04 + Threadripper PRO + 2x RTX 3070"
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
KZA_HOME="/home/${KZA_USER}"
INSTALL_DIR="${INSTALL_DIR:-${KZA_HOME}/kza}"
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
if [ "$GPU_COUNT" -lt 2 ]; then
    log_warn "Expected at least 2 GPUs, found: ${GPU_COUNT} — some features may be limited"
    summary_warn "GPUs: ${GPU_COUNT}/2 detected"
else
    summary_ok "GPUs: ${GPU_COUNT} detected (2x RTX 3070 baseline; assignment in config/settings.yaml)"
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

# Regular (non-system) user. Groups: audio (ReSpeaker mics), dialout (MA1260
# RS-232), bluetooth (BLE presence), video (GPU device nodes on some installs).
# NEVER add sudo/docker/lxd/kvm/libvirt — the account must stay non-privileged
# (docs/architecture/ROOTLESS_MIGRATION.md).
if ! id "$KZA_USER" &>/dev/null; then
    useradd -m -s /bin/bash -G audio,video,dialout,bluetooth "$KZA_USER"
    log_ok "User ${KZA_USER} created"
else
    log_ok "User ${KZA_USER} already exists"
fi

usermod -a -G audio,video,dialout,bluetooth "$KZA_USER"
summary_ok "Service user: ${KZA_USER} (non-privileged; audio,video,dialout,bluetooth)"

# ----------------------------------------------------------
# 6. Create installation directory and required subdirs
# ----------------------------------------------------------
log_info "Setting up installation directory ${INSTALL_DIR}..."

mkdir -p "${INSTALL_DIR}/data/chroma_db"
mkdir -p "${INSTALL_DIR}/data/memory_db"
mkdir -p "${INSTALL_DIR}/data/contexts"
mkdir -p "${INSTALL_DIR}/logs"
mkdir -p "${INSTALL_DIR}/models/lora_adapters"
mkdir -p "${KZA_HOME}/secrets"
chmod 700 "${KZA_HOME}/secrets"

# Canonical layout: ~/app is a symlink to the checkout (systemd units and docs
# reference /home/kza/app)
if [ ! -e "${KZA_HOME}/app" ]; then
    ln -s "$INSTALL_DIR" "${KZA_HOME}/app"
fi

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

chown -R "${KZA_USER}:${KZA_USER}" "$INSTALL_DIR" "${KZA_HOME}/secrets"
chown -h "${KZA_USER}:${KZA_USER}" "${KZA_HOME}/app" 2>/dev/null || true
log_ok "Directory ${INSTALL_DIR} configured"
summary_ok "Install dir: ${INSTALL_DIR} (~/app symlink, data/, models/, logs/, ~/secrets/)"

# ----------------------------------------------------------
# 7. Create Python 3.13 venv and install pip dependencies
# ----------------------------------------------------------
log_info "Setting up Python ${PYTHON_VERSION} virtual environment..."

sudo -u "$KZA_USER" bash << VENVEOF
set -e
cd "$INSTALL_DIR"

# Create venv with Python 3.13 (.venv — the path the systemd user unit executes)
${PYTHON_BIN} -m venv .venv
source .venv/bin/activate

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
summary_ok "Python venv: ${INSTALL_DIR}/.venv (Python ${PYTHON_VERSION})"

# ----------------------------------------------------------
# 8. Seed ~/secrets/.env if not exists (units read it from there, chmod 600)
# ----------------------------------------------------------
ENV_FILE="${KZA_HOME}/secrets/.env"
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$INSTALL_DIR/.env.example" ]; then
        cp "$INSTALL_DIR/.env.example" "$ENV_FILE"
        chown "${KZA_USER}:${KZA_USER}" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        log_warn "${ENV_FILE} created from .env.example — EDIT IT with your HA token and Spotify credentials"
        summary_warn "${ENV_FILE} copied from .env.example — NEEDS EDITING"
    else
        log_warn ".env.example not found; create ${ENV_FILE} manually"
        summary_warn "${ENV_FILE} not created — no .env.example found"
    fi
else
    log_ok "${ENV_FILE} already exists"
    summary_ok "${ENV_FILE} already configured"
fi

# ----------------------------------------------------------
# 9. Install systemd USER unit + linger (rootless model)
# ----------------------------------------------------------
# kza-voice runs as a user unit (`systemctl --user`), never as a system unit.
# `loginctl enable-linger` keeps the user manager (and the service) running
# without an open session and across reboots.
log_info "Installing systemd user unit..."

USER_UNIT_DIR="${KZA_HOME}/.config/systemd/user"
if [ -f "$INSTALL_DIR/systemd/kza-voice.service" ]; then
    install -d -o "$KZA_USER" -g "$KZA_USER" "$USER_UNIT_DIR"
    cp "$INSTALL_DIR/systemd/kza-voice.service" "$USER_UNIT_DIR/"

    # Adjust paths if the user/home differs from the canonical /home/kza
    if [ "$KZA_HOME" != "/home/kza" ]; then
        sed -i "s|/home/kza|${KZA_HOME}|g" "$USER_UNIT_DIR/kza-voice.service"
    fi
    chown "${KZA_USER}:${KZA_USER}" "$USER_UNIT_DIR/kza-voice.service"

    loginctl enable-linger "$KZA_USER"
    sudo -u "$KZA_USER" XDG_RUNTIME_DIR="/run/user/$(id -u "$KZA_USER")" \
        systemctl --user daemon-reload
    log_ok "systemd user unit installed (kza-voice.service) + linger enabled"
    summary_ok "systemd user unit: kza-voice (rootless, linger on)"
else
    log_warn "systemd/kza-voice.service not found — skipped"
    summary_warn "systemd user unit not installed"
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

# Hugepages (optional, helps with large model memory allocations)
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
echo "  Next steps (run as ${KZA_USER} — e.g. 'ssh ${KZA_USER}@<host>'; no sudo needed):"
echo ""
echo "    1. Edit configuration:"
echo "       nano ${KZA_HOME}/secrets/.env"
echo ""
echo "    2. Download models:"
echo "       ${INSTALL_DIR}/scripts/download_models.sh"
echo ""
echo "    3. Run smoke test:"
echo "       ${INSTALL_DIR}/scripts/smoke_test.sh"
echo ""
echo "    4. Test manually:"
echo "       ${INSTALL_DIR}/scripts/start.sh"
echo ""
echo "    5. Enable and start the service (user unit, rootless):"
echo "       systemctl --user enable --now kza-voice"
echo ""
echo "    6. View logs:"
echo "       journalctl --user-unit kza-voice -f"
echo ""
echo "============================================================"
