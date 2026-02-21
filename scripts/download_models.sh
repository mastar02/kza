#!/bin/bash
# Download Models Script
# Descarga todos los modelos necesarios para 128GB RAM setup

set -e

MODELS_DIR="./models"
mkdir -p $MODELS_DIR

echo "=========================================="
echo "  KZA Voice - Model Downloader"
echo "  Optimizado para 128GB RAM"
echo "=========================================="
echo ""

# Verificar huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Instalando huggingface_hub..."
    pip install huggingface_hub --quiet
fi

# ============ Seleccion de modelo LLM ============
echo ""
echo "🧠 MODELO LLM PRINCIPAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Opciones para 128GB RAM:"
echo "  1) Llama 3.3 70B Q8_0 (~70GB) - RECOMENDADO"
echo "  2) Qwen 2.5 72B Q8_0 (~72GB) - Excelente en español"
echo "  3) Llama 3.1 70B Q8_0 (~70GB) - Muy estable"
echo "  4) Llama 3.1 70B Q4_K_M (~45GB) - Mas ligero"
echo ""

read -p "Selecciona modelo [1-4, default=1]: " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1)
        MODEL_NAME="Llama-3.3-70B-Instruct-Q8_0.gguf"
        MODEL_URL="bartowski/Llama-3.3-70B-Instruct-GGUF"
        MODEL_SIZE="70GB"
        ;;
    2)
        MODEL_NAME="Qwen2.5-72B-Instruct-Q8_0.gguf"
        MODEL_URL="Qwen/Qwen2.5-72B-Instruct-GGUF"
        MODEL_SIZE="72GB"
        ;;
    3)
        MODEL_NAME="Meta-Llama-3.1-70B-Instruct-Q8_0.gguf"
        MODEL_URL="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"
        MODEL_SIZE="70GB"
        ;;
    4)
        MODEL_NAME="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
        MODEL_URL="bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"
        MODEL_SIZE="45GB"
        ;;
    *)
        echo "Opcion invalida, usando Llama 3.3"
        MODEL_NAME="Llama-3.3-70B-Instruct-Q8_0.gguf"
        MODEL_URL="bartowski/Llama-3.3-70B-Instruct-GGUF"
        MODEL_SIZE="70GB"
        ;;
esac

echo ""
echo "📦 Descargando $MODEL_NAME (~$MODEL_SIZE)..."
echo "   Esto puede tomar bastante tiempo..."
echo ""

if [ ! -f "$MODELS_DIR/$MODEL_NAME" ]; then
    huggingface-cli download \
        "$MODEL_URL" \
        "$MODEL_NAME" \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks False

    echo "✅ $MODEL_NAME descargado"
else
    echo "✅ $MODEL_NAME ya existe"
fi

# Crear symlink al modelo configurado
ln -sf "$MODELS_DIR/$MODEL_NAME" "$MODELS_DIR/llm_model.gguf" 2>/dev/null || true

# ============ Whisper STT ============
echo ""
echo "🎤 SPEECH-TO-TEXT (Whisper)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Descargando Whisper (distil-small)..."

python3 -c "
from faster_whisper import WhisperModel
print('Descargando modelo Whisper...')
model = WhisperModel('distil-whisper/distil-small.en', device='cpu', compute_type='int8')
print('✅ Whisper descargado')
" 2>/dev/null || echo "⚠️ Whisper se descargara al primer uso"

# ============ Piper TTS ============
echo ""
echo "🔊 TEXT-TO-SPEECH (Piper)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PIPER_DIR="$MODELS_DIR/piper"
mkdir -p "$PIPER_DIR"

if [ ! -f "$PIPER_DIR/es_ES-davefx-medium.onnx" ]; then
    echo "📦 Descargando voz española (davefx)..."
    wget -q --show-progress -O "$PIPER_DIR/es_ES-davefx-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
    wget -q -O "$PIPER_DIR/es_ES-davefx-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
    echo "✅ Piper TTS descargado"
else
    echo "✅ Piper TTS ya existe"
fi

# ============ Embeddings ============
echo ""
echo "🧲 EMBEDDINGS (BGE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Descargando BGE-small..."

python3 -c "
from sentence_transformers import SentenceTransformer
print('Descargando modelo de embeddings...')
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('✅ BGE-small descargado')
" 2>/dev/null || echo "⚠️ BGE se descargara al primer uso"

# ============ Wake Word ============
echo ""
echo "👂 WAKE WORD (OpenWakeWord)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Descargando OpenWakeWord..."

python3 -c "
from openwakeword import Model
print('Descargando modelo wake word...')
model = Model(wakeword_models=['hey_jarvis'], inference_framework='onnx')
print('✅ OpenWakeWord descargado')
" 2>/dev/null || echo "⚠️ OpenWakeWord se descargara al primer uso"

# ============ Speaker ID ============
echo ""
echo "👤 SPEAKER IDENTIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Descargando modelo de identificacion de voz..."

python3 -c "
from speechbrain.pretrained import EncoderClassifier
print('Descargando modelo SpeechBrain...')
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='./models/speaker_id'
)
print('✅ Speaker ID descargado')
" 2>/dev/null || echo "⚠️ Speaker ID se descargara al primer uso"

# ============ Crear directorios LoRA ============
echo ""
echo "📁 Creando estructura de directorios..."
mkdir -p "$MODELS_DIR/lora_adapters"
mkdir -p "./data/conversations"
mkdir -p "./data/chroma_db"
mkdir -p "./data/memory_db"
mkdir -p "./logs"

# ============ Resumen ============
echo ""
echo "=========================================="
echo "  ✅ DESCARGA COMPLETADA"
echo "=========================================="
echo ""
echo "Modelo LLM: $MODEL_NAME ($MODEL_SIZE)"
echo ""
echo "Contenido de $MODELS_DIR:"
ls -lh "$MODELS_DIR" 2>/dev/null | grep -v "^total"
echo ""
echo "Distribucion de memoria estimada:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LLM ($MODEL_SIZE):        ~${MODEL_SIZE}"
echo "  Contexto 32K:             ~8GB"
echo "  OS + Python + buffers:    ~10GB"
echo "  ────────────────────────────────"
echo "  Total estimado:           ~88GB de 128GB"
echo "  Libre:                    ~40GB"
echo ""
echo "Para iniciar: ./scripts/start.sh"
echo "=========================================="
