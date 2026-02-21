#!/bin/bash
#
# KZA Voice Assistant - Ubuntu Setup Script
# Para Ubuntu 22.04 LTS Server
#
# Ejecutar como: sudo ./scripts/setup_ubuntu.sh
#

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "============================================================"
echo "  KZA Voice Assistant - Ubuntu Setup"
echo "  Para: Ubuntu 22.04 LTS + Threadripper PRO + 4x RTX 3070"
echo "============================================================"
echo ""

# Verificar root
if [ "$EUID" -ne 0 ]; then
    log_error "Este script debe ejecutarse como root"
    log_info "Usa: sudo $0"
    exit 1
fi

# Usuario que ejecutará el servicio
KZA_USER="${KZA_USER:-kza}"
INSTALL_DIR="${INSTALL_DIR:-/opt/kza}"

# ----------------------------------------------------------
# 1. Actualizar sistema
# ----------------------------------------------------------
log_info "Actualizando sistema..."
apt-get update
apt-get upgrade -y
log_ok "Sistema actualizado"

# ----------------------------------------------------------
# 2. Instalar dependencias del sistema
# ----------------------------------------------------------
log_info "Instalando dependencias del sistema..."

apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    htop \
    nvtop \
    tmux \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
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
    jq

log_ok "Dependencias del sistema instaladas"

# ----------------------------------------------------------
# 3. Instalar NVIDIA Drivers + CUDA
# ----------------------------------------------------------
log_info "Verificando NVIDIA drivers..."

if ! command -v nvidia-smi &> /dev/null; then
    log_info "Instalando NVIDIA drivers..."

    # Agregar repositorio de NVIDIA
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update

    # Instalar driver recomendado
    ubuntu-drivers autoinstall

    log_warn "NVIDIA drivers instalados. REINICIA el sistema antes de continuar."
    log_info "Después del reinicio, ejecuta este script nuevamente."
    exit 0
else
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    log_ok "NVIDIA drivers ya instalados"
fi

# Verificar CUDA
if ! command -v nvcc &> /dev/null; then
    log_info "Instalando CUDA Toolkit..."

    # CUDA 12.x para Ubuntu 22.04
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-12-4

    # Agregar al PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh

    log_ok "CUDA Toolkit instalado"
else
    nvcc --version | head -n1
    log_ok "CUDA ya instalado"
fi

# ----------------------------------------------------------
# 4. Crear usuario para el servicio
# ----------------------------------------------------------
log_info "Configurando usuario $KZA_USER..."

if ! id "$KZA_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash -G audio,video "$KZA_USER"
    log_ok "Usuario $KZA_USER creado"
else
    log_ok "Usuario $KZA_USER ya existe"
fi

# Agregar al grupo de audio
usermod -a -G audio "$KZA_USER"

# ----------------------------------------------------------
# 5. Crear directorio de instalación
# ----------------------------------------------------------
log_info "Creando directorio de instalación..."

mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/models"

# Si estamos en el repo, copiar archivos
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/src/main.py" ]; then
    log_info "Copiando archivos del proyecto..."
    cp -r "$PROJECT_DIR/src" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/config" "$INSTALL_DIR/"
    cp -r "$PROJECT_DIR/scripts" "$INSTALL_DIR/"
    cp "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    cp "$PROJECT_DIR/.env.example" "$INSTALL_DIR/" 2>/dev/null || true
fi

chown -R "$KZA_USER:$KZA_USER" "$INSTALL_DIR"
log_ok "Directorio $INSTALL_DIR configurado"

# ----------------------------------------------------------
# 6. Crear entorno virtual e instalar dependencias Python
# ----------------------------------------------------------
log_info "Configurando entorno Python..."

sudo -u "$KZA_USER" bash << EOF
cd "$INSTALL_DIR"

# Crear venv
python3 -m venv venv
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip wheel setuptools

# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Instalar requirements si existe
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Instalar paquetes adicionales comunes
pip install \
    chromadb \
    sentence-transformers \
    faster-whisper \
    openai-whisper \
    sounddevice \
    soundfile \
    numpy \
    scipy \
    pyyaml \
    python-dotenv \
    aiohttp \
    websockets \
    openwakeword \
    llama-cpp-python \
    speechbrain \
    piper-tts

EOF

log_ok "Entorno Python configurado"

# ----------------------------------------------------------
# 7. Configurar systemd service
# ----------------------------------------------------------
log_info "Configurando servicio systemd..."

if [ -f "$PROJECT_DIR/systemd/kza-voice.service" ]; then
    cp "$PROJECT_DIR/systemd/kza-voice.service" /etc/systemd/system/

    # Ajustar paths en el service file
    sed -i "s|/opt/kza|$INSTALL_DIR|g" /etc/systemd/system/kza-voice.service
    sed -i "s|User=kza|User=$KZA_USER|g" /etc/systemd/system/kza-voice.service
    sed -i "s|Group=kza|Group=$KZA_USER|g" /etc/systemd/system/kza-voice.service

    systemctl daemon-reload
    log_ok "Servicio systemd configurado"
else
    log_warn "Archivo de servicio no encontrado"
fi

# ----------------------------------------------------------
# 8. Configurar audio
# ----------------------------------------------------------
log_info "Configurando audio..."

# Asegurar que PulseAudio o PipeWire están configurados
if command -v pulseaudio &> /dev/null; then
    # Habilitar PulseAudio para el usuario
    sudo -u "$KZA_USER" pulseaudio --start 2>/dev/null || true
    log_ok "PulseAudio configurado"
fi

# Listar dispositivos de audio
log_info "Dispositivos de audio disponibles:"
arecord -l 2>/dev/null || log_warn "No se detectaron dispositivos de grabación"

# ----------------------------------------------------------
# 9. Optimizaciones del sistema
# ----------------------------------------------------------
log_info "Aplicando optimizaciones del sistema..."

# Aumentar límite de archivos abiertos
cat >> /etc/security/limits.conf << EOF
# KZA Voice Assistant
$KZA_USER soft nofile 65535
$KZA_USER hard nofile 65535
EOF

# Optimizar para baja latencia de audio
if [ ! -f /etc/security/limits.d/audio.conf ]; then
    cat > /etc/security/limits.d/audio.conf << EOF
@audio - rtprio 95
@audio - memlock unlimited
EOF
fi

# Configurar hugepages para LLM (opcional)
echo "vm.nr_hugepages=1024" >> /etc/sysctl.d/99-kza.conf
sysctl -p /etc/sysctl.d/99-kza.conf 2>/dev/null || true

log_ok "Optimizaciones aplicadas"

# ----------------------------------------------------------
# 10. Crear archivo .env
# ----------------------------------------------------------
if [ ! -f "$INSTALL_DIR/.env" ]; then
    log_info "Creando archivo .env..."

    cat > "$INSTALL_DIR/.env" << EOF
# Home Assistant
HOME_ASSISTANT_URL=http://192.168.1.100:8123
HOME_ASSISTANT_TOKEN=tu_token_aqui

# Paths
MODELS_PATH=$INSTALL_DIR/models
CHROMA_PATH=$INSTALL_DIR/data/chroma_db
CONFIG_PATH=$INSTALL_DIR/config/settings.yaml

# GPU Assignment (interno, no modificar)
# GPU 0: STT
# GPU 1: Embeddings + Speaker ID
# GPU 2: Fast Router
# GPU 3: TTS
EOF

    chown "$KZA_USER:$KZA_USER" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"

    log_warn "Archivo .env creado. EDÍTALO con tus valores de Home Assistant"
fi

# ----------------------------------------------------------
# Resumen
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup completado!"
echo "============================================================"
echo ""
echo "  Directorio: $INSTALL_DIR"
echo "  Usuario: $KZA_USER"
echo ""
echo "  Próximos pasos:"
echo ""
echo "  1. Edita la configuración:"
echo "     sudo -u $KZA_USER nano $INSTALL_DIR/.env"
echo ""
echo "  2. Descarga los modelos:"
echo "     sudo -u $KZA_USER $INSTALL_DIR/scripts/download_models.sh"
echo ""
echo "  3. Prueba manualmente:"
echo "     sudo -u $KZA_USER $INSTALL_DIR/scripts/start.sh"
echo ""
echo "  4. Habilita el servicio:"
echo "     sudo systemctl enable kza-voice"
echo "     sudo systemctl start kza-voice"
echo ""
echo "  5. Ver logs:"
echo "     journalctl -u kza-voice -f"
echo ""
echo "============================================================"
