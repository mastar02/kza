#!/bin/bash
#
# KZA Voice Assistant - GPU Monitor
# Muestra estado y asignación de GPUs en tiempo real
#

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Asignación de GPUs
declare -A GPU_ASSIGNMENT
GPU_ASSIGNMENT[0]="STT (Whisper)"
GPU_ASSIGNMENT[1]="Embeddings + Speaker ID"
GPU_ASSIGNMENT[2]="Fast Router (Qwen 7B)"
GPU_ASSIGNMENT[3]="TTS (Piper/XTTS)"

# Verificar nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi no encontrado${NC}"
    exit 1
fi

# Función para mostrar barra de uso
progress_bar() {
    local percent=$1
    local width=20
    local filled=$((percent * width / 100))
    local empty=$((width - filled))

    local color=$GREEN
    if [ "$percent" -gt 80 ]; then
        color=$RED
    elif [ "$percent" -gt 50 ]; then
        color=$YELLOW
    fi

    printf "${color}["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "]${NC} %3d%%" "$percent"
}

# Función para obtener datos de GPU
get_gpu_info() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
}

# Modo watch (actualización continua)
if [ "$1" == "-w" ] || [ "$1" == "--watch" ]; then
    INTERVAL="${2:-2}"

    while true; do
        clear
        echo ""
        echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║${NC}                    ${BLUE}KZA Voice Assistant - GPU Monitor${NC}                     ${CYAN}║${NC}"
        echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════════════╣${NC}"

        while IFS=',' read -r idx name mem_used mem_total gpu_util temp power; do
            # Limpiar espacios
            idx=$(echo "$idx" | tr -d ' ')
            name=$(echo "$name" | tr -d ' ' | cut -c1-20)
            mem_used=$(echo "$mem_used" | tr -d ' ')
            mem_total=$(echo "$mem_total" | tr -d ' ')
            gpu_util=$(echo "$gpu_util" | tr -d ' ')
            temp=$(echo "$temp" | tr -d ' ')
            power=$(echo "$power" | tr -d ' ')

            # Calcular porcentaje de memoria
            mem_percent=$((mem_used * 100 / mem_total))

            # Obtener asignación
            assignment="${GPU_ASSIGNMENT[$idx]:-Desconocido}"

            echo -e "${CYAN}║${NC}"
            printf "${CYAN}║${NC}  ${YELLOW}GPU %s${NC}: %-20s │ ${BLUE}%-25s${NC}\n" "$idx" "$name" "$assignment"
            printf "${CYAN}║${NC}    VRAM: %5d/%5d MB  " "$mem_used" "$mem_total"
            progress_bar "$mem_percent"
            echo ""
            printf "${CYAN}║${NC}    Uso:  %3d%%    Temp: %2d°C    Power: %5.1f W\n" "$gpu_util" "$temp" "$power"

        done <<< "$(get_gpu_info)"

        echo -e "${CYAN}║${NC}"
        echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════════════╣${NC}"

        # CPU info
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        MEM_INFO=$(free -m | awk 'NR==2{printf "%d/%d MB (%.1f%%)", $3, $2, $3*100/$2}')

        printf "${CYAN}║${NC}  ${YELLOW}CPU${NC}: %5.1f%%    ${YELLOW}RAM${NC}: %s\n" "$CPU_USAGE" "$MEM_INFO"

        echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  Actualización cada ${INTERVAL}s | Presiona Ctrl+C para salir"

        sleep "$INTERVAL"
    done
else
    # Modo single shot
    echo ""
    echo -e "${BLUE}KZA Voice Assistant - GPU Status${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    while IFS=',' read -r idx name mem_used mem_total gpu_util temp power; do
        idx=$(echo "$idx" | tr -d ' ')
        name=$(echo "$name" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        mem_total=$(echo "$mem_total" | tr -d ' ')
        gpu_util=$(echo "$gpu_util" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')

        assignment="${GPU_ASSIGNMENT[$idx]:-Desconocido}"
        mem_percent=$((mem_used * 100 / mem_total))

        echo -e "${YELLOW}GPU $idx${NC}: $name"
        echo -e "  Asignación: ${CYAN}$assignment${NC}"
        printf "  VRAM: %d/%d MB (%d%%)\n" "$mem_used" "$mem_total" "$mem_percent"
        printf "  Uso: %d%%  |  Temp: %d°C\n" "$gpu_util" "$temp"
        echo ""

    done <<< "$(get_gpu_info)"

    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Uso: $0 -w [intervalo]  para modo watch"
fi
