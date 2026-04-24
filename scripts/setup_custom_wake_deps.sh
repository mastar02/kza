#!/usr/bin/env bash
# Setup dependencies para entrenar un wake word custom con openwakeword.
#
# Instala piper-sample-generator, TensorFlow, audiomentations, openwakeword,
# tf2onnx. Opcionalmente descarga MUSAN (~11GB) para augmentation/negatives.
#
# Uso:
#   bash scripts/setup_custom_wake_deps.sh            # solo deps
#   bash scripts/setup_custom_wake_deps.sh --musan    # deps + descarga MUSAN
#
# Asume que corre en el server (kza@192.168.1.2) con el .venv del repo activo.
set -euo pipefail

REPO="${REPO:-$HOME/app}"
VENV="${VENV:-$REPO/.venv}"
MUSAN_DIR="${MUSAN_DIR:-$HOME/data/musan}"
DOWNLOAD_MUSAN=false

for arg in "$@"; do
    case "$arg" in
        --musan) DOWNLOAD_MUSAN=true ;;
        *) echo "Arg desconocido: $arg"; exit 1 ;;
    esac
done

if [[ ! -d "$VENV" ]]; then
    echo "No encuentro venv en $VENV. Set VENV=path antes de correr."
    exit 1
fi

echo ">> Activando venv: $VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo ">> Instalando deps de training..."
pip install --upgrade pip
pip install \
    piper-tts \
    piper-sample-generator \
    "tensorflow>=2.12,<2.16" \
    audiomentations \
    "openwakeword>=0.6" \
    tf2onnx \
    onnx \
    onnxruntime \
    librosa \
    soundfile

echo ">> Verificando imports..."
python - <<'PY'
import sys
errs = []
for pkg in ("piper", "piper_sample_generator", "tensorflow",
            "audiomentations", "openwakeword", "tf2onnx", "librosa"):
    try:
        __import__(pkg.replace("-", "_"))
    except Exception as e:
        errs.append(f"{pkg}: {e}")
if errs:
    print("FAIL:"); [print("  -", e) for e in errs]; sys.exit(1)
print("OK: todas las deps importan correctamente.")
PY

if $DOWNLOAD_MUSAN; then
    if [[ -d "$MUSAN_DIR" ]]; then
        echo ">> MUSAN ya existe en $MUSAN_DIR, skip download."
    else
        echo ">> Descargando MUSAN (~11GB) a $MUSAN_DIR..."
        mkdir -p "$(dirname "$MUSAN_DIR")"
        cd "$(dirname "$MUSAN_DIR")"
        curl -L -O https://www.openslr.org/resources/17/musan.tar.gz
        echo ">> Extrayendo..."
        tar xf musan.tar.gz
        rm musan.tar.gz
        echo ">> MUSAN listo: $MUSAN_DIR"
    fi
else
    echo ">> Skip MUSAN (pasa --musan para descargarlo)"
fi

echo ">> Setup completo. Próximos pasos:"
echo "   1. systemctl --user stop kza-voice.service"
echo "   2. python -m scripts.train_custom_wake all"
echo "   3. systemctl --user start kza-voice.service"
echo "   4. Editar config/settings.yaml: rooms.wake_word.engine = 'openwakeword'"
