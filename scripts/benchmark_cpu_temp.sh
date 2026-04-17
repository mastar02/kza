#!/usr/bin/env bash
# =============================================================================
# benchmark_cpu_temp.sh — Benchmark CPU con monitoreo de temperatura en tiempo real
# KZA Server — Threadripper PRO 7965WX
#
# Uso: sudo bash scripts/benchmark_cpu_temp.sh
# Genera: /tmp/kza_benchmark_<timestamp>.csv  +  reporte final
# =============================================================================
set -uo pipefail

RED='\033[0;31m'
YEL='\033[1;33m'
GRN='\033[0;32m'
CYN='\033[0;36m'
BLU='\033[0;34m'
MAG='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

TS=$(date '+%Y%m%d_%H%M%S')
CSV="/tmp/kza_benchmark_${TS}.csv"
PHASE_FILE="/tmp/kza_current_phase_${TS}.txt"   # archivo compartido con subshell
STRESS_PID=""

# Fases del benchmark: (nombre, threads, duracion_seg, descripcion)
declare -A PHASES
PHASE_ORDER=(
    "idle"
    "light_8t"
    "medium_16t"
    "heavy_24t"
    "heavy_38t"
    "cooldown"
)

declare -A PHASE_THREADS=(
    ["idle"]=0
    ["light_8t"]=8
    ["medium_16t"]=16
    ["heavy_24t"]=24
    ["heavy_38t"]=38
    ["cooldown"]=0
)

declare -A PHASE_DURATION=(
    ["idle"]=60
    ["light_8t"]=120
    ["medium_16t"]=120
    ["heavy_24t"]=120
    ["heavy_38t"]=120
    ["cooldown"]=90
)

declare -A PHASE_DESC=(
    ["idle"]="Sistema en reposo (baseline)"
    ["light_8t"]="Carga ligera — deja 40t libres para KZA + LLM"
    ["medium_16t"]="Carga media — deja 32t libres para LLM"
    ["heavy_24t"]="Carga alta — deja 24t libres"
    ["heavy_38t"]="Carga máxima minería — 80% threads"
    ["cooldown"]="Enfriamiento post-benchmark"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo -e "${CYN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GRN}[OK]${NC} $*"; }
warn() { echo -e "${YEL}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

bar() {
    local val=$1 max=$2 width=${3:-30}
    local filled=$(( val * width / max ))
    local empty=$(( width - filled ))
    printf '%s' "["
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null || printf '%*s' $filled '' | tr ' ' '█'
    printf '%*s' $empty '' | tr ' ' '░'
    printf '%s' "]"
}

get_temp() {
    # Intenta múltiples fuentes para Threadripper PRO
    local t
    t=$(sensors 2>/dev/null | grep -E "Tctl|Tdie" | grep -oP '[+\-]?\d+\.\d+' | \
        sort -rn | head -1 | cut -d'.' -f1)
    # Fallback: k10temp directo
    if [ -z "$t" ]; then
        t=$(cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | \
            sort -rn | head -1 | awk '{printf "%d", $1/1000}')
    fi
    echo "${t:-0}"
}

get_freq_mhz() {
    # Frecuencia media actual de todos los cores
    awk 'BEGIN{s=0;n=0} /cpu MHz/{s+=$4; n++} END{if(n>0) printf "%d", s/n; else print "?"}' \
        /proc/cpuinfo 2>/dev/null
}

get_load() {
    awk '{printf "%.1f", $1}' /proc/loadavg 2>/dev/null
}

get_mem_gb() {
    free -g 2>/dev/null | awk '/Mem:/{printf "%d/%d", $3, $2}'
}

# Hashrate simulado RandomX (kH/s) basado en threads activos y frecuencia
# En el benchmark real usamos stress-ng; XMRig daría los hashes reales
estimate_hashrate() {
    local threads=$1 freq_mhz=$2
    if [ "$threads" -eq 0 ]; then echo "0.00"; return; fi
    # ~1.0 kH/s por core a 4600 MHz (estimado conservador 7965WX)
    local rate
    rate=$(awk "BEGIN{printf \"%.2f\", $threads * 0.95 * ($freq_mhz / 4600.0)}")
    echo "$rate"
}

# -----------------------------------------------------------------------------
# Monitor en background — escribe CSV
# -----------------------------------------------------------------------------
start_monitor() {
    local phase=$1
    echo "timestamp,phase,temp_c,freq_mhz,load_1m,mem_gb_used,threads_active" >> "$CSV"

    while true; do
        local ts temp freq load mem
        ts=$(date '+%H:%M:%S')
        temp=$(get_temp)
        freq=$(get_freq_mhz)
        load=$(get_load)
        mem=$(get_mem_gb)
        local threads=${PHASE_THREADS[$CURRENT_PHASE]:-0}

        echo "${ts},${CURRENT_PHASE},${temp},${freq},${load},${mem},${threads}" >> "$CSV"
        sleep 5
    done
}

# -----------------------------------------------------------------------------
# Mostrar dashboard en tiempo real
# -----------------------------------------------------------------------------
print_dashboard() {
    local phase=$1 elapsed=$2 total=$3 temp=$4 freq=$5 load=$6

    local threads=${PHASE_THREADS[$phase]}
    local hr=$(estimate_hashrate "$threads" "$freq")

    # Color de temperatura
    local tcol="$GRN"
    [ "$temp" -ge 75 ] && tcol="$YEL"
    [ "$temp" -ge 82 ] && tcol="$RED"

    # Barra de progreso de fase
    local pbar
    pbar=$(bar "$elapsed" "$total" 25)

    printf "\r${BOLD}%-14s${NC} ${pbar} %3ds/%-3ds │ " "$phase" "$elapsed" "$total"
    printf "T: ${tcol}%2d°C${NC} │ " "$temp"
    printf "Freq: ${BLU}%4d MHz${NC} │ " "$freq"
    printf "Load: ${MAG}%5s${NC} │ " "$load"
    printf "kH/s~: ${GRN}%s${NC}    " "$hr"
}

# -----------------------------------------------------------------------------
# Ejecutar una fase del benchmark
# -----------------------------------------------------------------------------
run_phase() {
    local phase=$1
    CURRENT_PHASE="$phase"
    echo "$phase" > "$PHASE_FILE"   # visible al subshell monitor
    local threads=${PHASE_THREADS[$phase]}
    local duration=${PHASE_DURATION[$phase]}
    local desc=${PHASE_DESC[$phase]}

    echo ""
    echo -e "${BOLD}${BLU}━━━ FASE: ${phase} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "    ${desc}"
    echo -e "    Threads: ${threads} | Duración: ${duration}s"
    echo ""

    # Matar stress anterior si existe
    if [ -n "$STRESS_PID" ] && kill -0 "$STRESS_PID" 2>/dev/null; then
        kill "$STRESS_PID" 2>/dev/null
        sleep 2
    fi
    STRESS_PID=""

    # Iniciar carga si threads > 0
    if [ "$threads" -gt 0 ]; then
        # stress-ng con RandomX-like: memoria intensiva + CPU
        stress-ng --cpu "$threads" \
                  --cpu-method=all \
                  --vm 4 --vm-bytes 2G \
                  --timeout "${duration}s" \
                  --quiet &>/dev/null &
        STRESS_PID=$!
    fi

    # Loop de monitoreo de la fase
    local start_ts peak_temp=0
    start_ts=$(date +%s)

    for (( elapsed=0; elapsed<=duration; elapsed+=5 )); do
        local temp freq load
        temp=$(get_temp)
        freq=$(get_freq_mhz)
        load=$(get_load)

        [ "$temp" -gt "$peak_temp" ] && peak_temp=$temp

        print_dashboard "$phase" "$elapsed" "$duration" "$temp" "$freq" "$load"

        # Alerta si temperatura crítica
        if [ "$temp" -ge 85 ]; then
            echo ""
            warn "TEMPERATURA CRÍTICA: ${temp}°C — abortando fase y enfriando"
            [ -n "$STRESS_PID" ] && kill "$STRESS_PID" 2>/dev/null
            STRESS_PID=""
            break
        fi

        sleep 5
    done

    # Guardar peak de la fase en resumen
    echo "${phase},${threads},${peak_temp}" >> "/tmp/kza_peak_temps_${TS}.txt"

    # Esperar a que stress-ng termine
    if [ -n "$STRESS_PID" ] && kill -0 "$STRESS_PID" 2>/dev/null; then
        wait "$STRESS_PID" 2>/dev/null || true
    fi
    STRESS_PID=""

    echo ""
    echo -e "    ${GRN}Peak temperatura fase: ${peak_temp}°C${NC}"
}

# -----------------------------------------------------------------------------
# Reporte final
# -----------------------------------------------------------------------------
print_report() {
    echo ""
    echo -e "${BOLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "  REPORTE BENCHMARK KZA — Threadripper PRO 7965WX @ 4.6 GHz"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    local freq_actual
    freq_actual=$(get_freq_mhz)

    printf "${BOLD}%-20s │ %-10s │ %-12s │ %-10s${NC}\n" \
        "Fase" "Threads" "Peak Temp" "Estado"
    echo "─────────────────────┼────────────┼──────────────┼──────────────"

    if [ -f "/tmp/kza_peak_temps_${TS}.txt" ]; then
        while IFS=',' read -r phase threads peak; do
            local status="OK"
            local scol="$GRN"
            [ "$peak" -ge 75 ] && status="CALIENTE" && scol="$YEL"
            [ "$peak" -ge 82 ] && status="LIMITE"   && scol="$RED"
            printf "%-20s │ %-10s │ ${scol}%-12s${NC} │ ${scol}%-10s${NC}\n" \
                "$phase" "${threads}t" "${peak}°C" "$status"
        done < "/tmp/kza_peak_temps_${TS}.txt"
    fi

    echo ""
    echo -e "  ${BOLD}CSV completo:${NC} ${CSV}"
    echo -e "  ${BOLD}Frecuencia actual:${NC} ${freq_actual} MHz (límite: 4600 MHz)"
    echo ""

    # Recomendación threads para minería
    local safe_threads=0
    if [ -f "/tmp/kza_peak_temps_${TS}.txt" ]; then
        while IFS=',' read -r phase threads peak; do
            if [ "$peak" -lt 80 ]; then
                safe_threads=$threads
            fi
        done < "/tmp/kza_peak_temps_${TS}.txt"
    fi

    echo -e "${BOLD}  RECOMENDACIÓN:${NC}"
    if [ "$safe_threads" -gt 0 ]; then
        echo -e "  ${GRN}Threads seguros para XMRig: ${safe_threads}t (pico < 80°C)${NC}"
        local est_hr
        est_hr=$(estimate_hashrate "$safe_threads" 4600)
        echo -e "  ${GRN}Hashrate estimado RandomX: ~${est_hr} kH/s${NC}"
    else
        echo -e "  ${RED}Revisar sistema de refrigeración antes de minar${NC}"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    log "Limpiando procesos..."
    [ -n "$STRESS_PID" ] && kill "$STRESS_PID" 2>/dev/null || true
    pkill -f stress-ng 2>/dev/null || true
    print_report
    exit 0
}
trap cleanup INT TERM EXIT

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
    err "Ejecutar como root: sudo bash $0"
    exit 1
fi

echo -e "${BOLD}${CYN}"
echo "  ██╗  ██╗███████╗ █████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗"
echo "  ██║ ██╔╝╚══███╔╝██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║"
echo "  █████╔╝   ███╔╝ ███████║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║"
echo "  ██╔═██╗  ███╔╝  ██╔══██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║"
echo "  ██║  ██╗███████╗██║  ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║"
echo "  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${BOLD}  Benchmark CPU con monitoreo de temperatura en tiempo real${NC}"
echo -e "  CPU: Threadripper PRO 7965WX | Frecuencia límite: 4600 MHz"
echo -e "  Duración total estimada: ~${BOLD}$(( (60+120+120+120+120+90) / 60 )) minutos${NC}"
echo ""

# Verificar que el límite de frecuencia está activo
CURRENT_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "0")
CURRENT_MAX_MHZ=$(( CURRENT_MAX / 1000 ))
if [ "$CURRENT_MAX_MHZ" -gt 4650 ]; then
    warn "Frecuencia máxima actual: ${CURRENT_MAX_MHZ} MHz — por encima del objetivo"
    warn "Ejecuta primero: sudo bash scripts/setup_cpu_mining.sh"
    echo ""
    read -rp "¿Continuar de todas formas? [s/N] " ans
    [[ "$ans" =~ ^[sS]$ ]] || exit 0
else
    ok "Frecuencia limitada correctamente: ${CURRENT_MAX_MHZ} MHz"
fi

# Inicializar CSV y estado compartido
echo "timestamp,phase,temp_c,freq_mhz,load_1m,mem_gb_used,threads_active" > "$CSV"
CURRENT_PHASE="idle"
echo "$CURRENT_PHASE" > "$PHASE_FILE"

# Iniciar monitor en background — lee la fase desde PHASE_FILE (compartido entre procesos)
(
    declare -A _MON_THREADS=(
        ["idle"]=0 ["light_8t"]=8 ["medium_16t"]=16
        ["heavy_24t"]=24 ["heavy_38t"]=38 ["cooldown"]=0
    )
    while true; do
        ts=$(date '+%H:%M:%S')
        temp=$(sensors 2>/dev/null | grep -E "Tctl|Tdie" | grep -oP '[+\-]?\d+\.\d+' | sort -rn | head -1 | cut -d'.' -f1)
        freq=$(awk 'BEGIN{s=0;n=0} /cpu MHz/{s+=$4; n++} END{if(n>0) printf "%d", s/n}' /proc/cpuinfo 2>/dev/null)
        load=$(awk '{printf "%.1f", $1}' /proc/loadavg 2>/dev/null)
        mem=$(free -g 2>/dev/null | awk '/Mem:/{printf "%d/%d", $3, $2}')
        phase=$(cat "$PHASE_FILE" 2>/dev/null || echo "idle")
        threads=${_MON_THREADS[$phase]:-0}
        echo "${ts},${phase},${temp:-0},${freq:-0},${load},${mem},${threads}" >> "$CSV"
        sleep 5
    done
) &
MONITOR_PID=$!

log "CSV iniciado: ${CSV}"
log "Monitor PID: ${MONITOR_PID}"
echo ""

# Ejecutar fases
for phase in "${PHASE_ORDER[@]}"; do
    run_phase "$phase"
done

kill "$MONITOR_PID" 2>/dev/null || true
rm -f "$PHASE_FILE"
print_report
