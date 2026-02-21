#!/bin/bash
#
# KZA Voice Assistant - Startup Script
# Para Ubuntu Server con 4x RTX 3070
#

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "=================================================="
echo "  KZA Voice Assistant - Home Automation"
echo "  Hardware: Threadripper PRO + 4x RTX 3070"
echo "=================================================="
echo ""

# ----------------------------------------------------------
# 1. Verificar entorno
# ----------------------------------------------------------
log_info "Verificando entorno..."

# Verificar que estamos en Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_warn "Este script está optimizado para Ubuntu/Linux"
fi

# Verificar .env
if [ ! -f ".env" ]; then
    log_error ".env no encontrado"
    log_info "Copia .env.example a .env y configura tus valores"
    exit 1
fi
log_ok ".env encontrado"

# Cargar variables de entorno
set -a
source .env
set +a

# Verificar variables críticas
if [ -z "$HOME_ASSISTANT_URL" ] || [ -z "$HOME_ASSISTANT_TOKEN" ]; then
    log_error "HOME_ASSISTANT_URL y HOME_ASSISTANT_TOKEN deben estar configurados en .env"
    exit 1
fi
log_ok "Variables de entorno cargadas"

# ----------------------------------------------------------
# 2. Verificar GPUs
# ----------------------------------------------------------
log_info "Verificando GPUs..."

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi no encontrado. Instala NVIDIA drivers."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_info "GPUs detectadas: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 4 ]; then
    log_warn "Se esperaban 4 GPUs, encontradas: $GPU_COUNT"
    log_warn "Algunas funciones pueden estar limitadas"
fi

# Mostrar estado de GPUs
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# ----------------------------------------------------------
# 3. Asignación de GPUs (CUDA_VISIBLE_DEVICES por proceso)
# ----------------------------------------------------------
# GPU 0: STT (Whisper)
# GPU 1: Embeddings + Speaker ID
# GPU 2: Fast Router (Qwen2.5-7B)
# GPU 3: TTS (Piper/XTTS)
#
# Nota: Cada componente lee su GPU del config/settings.yaml
# CUDA_VISIBLE_DEVICES se usa solo para el LLM 70B que corre en CPU

# ----------------------------------------------------------
# 4. Verificar modelos
# ----------------------------------------------------------
log_info "Verificando modelos..."

MODELS_DIR="${MODELS_PATH:-./models}"

check_model() {
    local name="$1"
    local path="$2"

    if [ -e "$path" ]; then
        log_ok "$name"
    else
        log_warn "$name no encontrado: $path"
        return 1
    fi
}

MODELS_OK=true

# Verificar modelo LLM (opcional pero recomendado)
LLM_MODEL="${MODELS_DIR}/llama-3.1-70b-instruct-q4_k_m.gguf"
if [ -f "$LLM_MODEL" ]; then
    log_ok "LLM 70B encontrado"
else
    log_warn "LLM 70B no encontrado (razonamiento profundo deshabilitado)"
    log_info "  Descarga con: ./scripts/download_models.sh"
fi

# Verificar directorio de datos
mkdir -p ./data/chroma_db
mkdir -p ./data/memory_db
log_ok "Directorios de datos creados"

# ----------------------------------------------------------
# 5. Verificar Python y dependencias
# ----------------------------------------------------------
log_info "Verificando Python..."

if ! command -v python3 &> /dev/null; then
    log_error "Python3 no encontrado"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
log_ok "Python $PYTHON_VERSION"

# Verificar venv
if [ ! -d "venv" ]; then
    log_info "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar venv
source venv/bin/activate
log_ok "Entorno virtual activado"

# Verificar dependencias críticas
python3 -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>/dev/null || {
    log_error "PyTorch no instalado. Ejecuta: pip install -r requirements.txt"
    exit 1
}

python3 -c "import chromadb; print(f'  ChromaDB OK')" 2>/dev/null || {
    log_warn "ChromaDB no instalado"
}

# ----------------------------------------------------------
# 6. Configurar límites del sistema (opcional, requiere root)
# ----------------------------------------------------------
# Aumentar límite de archivos abiertos para ChromaDB
ulimit -n 65535 2>/dev/null || true

# ----------------------------------------------------------
# 7. Health check inicial
# ----------------------------------------------------------
log_info "Ejecutando health check..."

python3 -c "
from src.health.health_check import HealthChecker
import asyncio

async def check():
    checker = HealthChecker()
    result = await checker.quick_check()
    return result['status'] == 'healthy'

try:
    ok = asyncio.run(check())
    exit(0 if ok else 1)
except Exception as e:
    print(f'Health check error: {e}')
    exit(1)
" && log_ok "Health check pasado" || log_warn "Health check con advertencias"

# ----------------------------------------------------------
# 8. Iniciar aplicación
# ----------------------------------------------------------
echo ""
log_info "Iniciando KZA Voice Assistant..."
echo ""
echo "  Wake word: hey_jarvis"
echo "  Target latency: 300ms"
echo "  Home Assistant: $HOME_ASSISTANT_URL"
echo ""
echo "  Presiona Ctrl+C para detener"
echo ""
echo "--------------------------------------------------"
echo ""

# Configurar manejo de señales
trap 'log_info "Recibida señal de terminación..."; exit 0' SIGTERM SIGINT

# Ejecutar
exec python3 -m src.main
