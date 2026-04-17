#!/usr/bin/env bash
# =============================================================================
# benchmark_freq_sweep.sh — Encuentra la frecuencia óptima para 48 threads
# KZA Server — Threadripper PRO 7965WX
#
# Lógica: sube la frecuencia escalonadamente con TODOS los threads activos
# y encuentra el punto donde la temperatura se estabiliza bajo 78°C.
#
# Uso: sudo bash scripts/benchmark_freq_sweep.sh
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
CSV="/tmp/kza_freq_sweep_${TS}.csv"
RESULTS_FILE="/tmp/kza_freq_results_${TS}.txt"
THREADS=48          # Todos los threads
SOAK_TIME=90        # Segundos por frecuencia (tiempo de estabilización térmica)
SAFE_TEMP=78        # °C — límite objetivo
ABORT_TEMP=85       # °C — abortar esta frecuencia y no subir más
COOLDOWN_TIME=60    # Segundos entre frecuencias

# Frecuencias a testear (MHz) — empezamos conservador y subimos
FREQ_STEPS=(2800 3200 3600 3800 4000 4200 4400 4600)

# -----------------------------------------------------------------------------
get_temp() {
    sensors 2>/dev/null | grep -E "Tctl|Tdie" | grep -oP '[+\-]?\d+\.\d+' | \
        sort -rn | head -1 | cut -d'.' -f1
    # fallback hwmon
    [ -z "$(sensors 2>/dev/null | grep -E 'Tctl|Tdie')" ] && \
        cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | \
        sort -rn | head -1 | awk '{printf "%d", $1/1000}'
}

get_freq_mhz() {
    awk 'BEGIN{s=0;n=0} /cpu MHz/{s+=$4; n++} END{if(n>0) printf "%d", s/n; else print 0}' \
        /proc/cpuinfo 2>/dev/null
}

get_load() { awk '{printf "%.1f", $1}' /proc/loadavg 2>/dev/null; }

set_freq() {
    local mhz=$1
    cpupower frequency-set --max "${mhz}MHz" &>/dev/null
    # Verificar
    local actual
    actual=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0) / 1000 ))
    echo "$actual"
}

bar() {
    local val=$1 max=$2 width=${3:-20}
    local filled=$(( val * width / max ))
    local empty=$(( width - filled ))
    printf '['
    [ $filled -gt 0 ] && printf '%*s' $filled '' | tr ' ' '█'
    [ $empty  -gt 0 ] && printf '%*s' $empty  '' | tr ' ' '░'
    printf ']'
}

temp_color() {
    local t=$1
    if   [ "$t" -ge "$ABORT_TEMP" ]; then echo -e "${RED}${t}°C${NC}"
    elif [ "$t" -ge "$SAFE_TEMP"  ]; then echo -e "${YEL}${t}°C${NC}"
    else                                   echo -e "${GRN}${t}°C${NC}"
    fi
}

# -----------------------------------------------------------------------------
# Soak una frecuencia: carga 48t durante SOAK_TIME y registra temperatura
# Retorna: peak_temp (via RESULT_PEAK_TEMP global)
# -----------------------------------------------------------------------------
RESULT_PEAK_TEMP=0
STRESS_PID=""

soak_freq() {
    local target_mhz=$1
    RESULT_PEAK_TEMP=0
    local abort=false

    echo ""
    echo -e "${BOLD}${BLU}━━━ Testeando: ${target_mhz} MHz con ${THREADS} threads ━━━━━━━━━━━━━━━${NC}"

    # Aplicar frecuencia
    local actual_mhz
    actual_mhz=$(set_freq "$target_mhz")
    echo -e "  Frecuencia aplicada: ${actual_mhz} MHz"

    # Esperar 5s para que el CPU adopte el nuevo límite
    sleep 5

    # Lanzar stress-ng CPU-only (sin vm, para simular mejor RandomX puro)
    # RandomX es CPU+memory pero el bottleneck real es el hash loop en CPU
    stress-ng --cpu "$THREADS" \
              --cpu-method=ackermann \
              --timeout "${SOAK_TIME}s" \
              --quiet &>/dev/null &
    STRESS_PID=$!

    local samples=0 sum_temp=0 sum_freq=0 first_stable_temp=0

    echo -e "  Estabilizando temperatura durante ${SOAK_TIME}s..."
    printf "  "

    for (( elapsed=5; elapsed<=SOAK_TIME; elapsed+=5 )); do
        local temp freq load
        temp=$(get_temp)
        freq=$(get_freq_mhz)
        load=$(get_load)
        temp=${temp:-0}

        # Guardar en CSV
        echo "${elapsed},${target_mhz},${temp},${freq},${load},${THREADS}" >> "$CSV"

        # Acumular para promedio (últimos 60s solamente = estado estable)
        if [ "$elapsed" -ge 30 ]; then
            sum_temp=$(( sum_temp + temp ))
            sum_freq=$(( sum_freq + freq ))
            samples=$(( samples + 1 ))
        fi

        # Track peak
        [ "$temp" -gt "$RESULT_PEAK_TEMP" ] && RESULT_PEAK_TEMP=$temp

        # Guardar primera temp estable (a los 30s)
        [ "$elapsed" -eq 30 ] && first_stable_temp=$temp

        # Barra visual
        local tbar
        tbar=$(bar "$temp" 100 15)
        printf "\r  %3ds │ T: $(temp_color "$temp") │ %s │ Freq real: ${BLU}%4d MHz${NC} │ Load: ${MAG}%4s${NC}   " \
            "$elapsed" "$tbar" "$freq" "$load"

        # Abort si temperatura demasiado alta
        if [ "$temp" -ge "$ABORT_TEMP" ]; then
            echo ""
            echo -e "  ${RED}ABORT: ${temp}°C ≥ ${ABORT_TEMP}°C — esta frecuencia es demasiado alta${NC}"
            abort=true
            break
        fi

        sleep 5
    done

    echo ""

    # Matar stress
    if kill -0 "$STRESS_PID" 2>/dev/null; then
        kill "$STRESS_PID" 2>/dev/null
        wait "$STRESS_PID" 2>/dev/null || true
    fi
    STRESS_PID=""

    # Calcular temp media estable
    local avg_temp=0 avg_freq=0
    if [ "$samples" -gt 0 ]; then
        avg_temp=$(( sum_temp / samples ))
        avg_freq=$(( sum_freq / samples ))
    fi

    # Guardar resultado
    local status="OK"
    [ "$RESULT_PEAK_TEMP" -ge "$SAFE_TEMP"  ] && status="CALIENTE"
    [ "$RESULT_PEAK_TEMP" -ge "$ABORT_TEMP" ] && status="ABORT"
    $abort && status="ABORT"

    echo "${target_mhz},${THREADS},${RESULT_PEAK_TEMP},${avg_temp},${avg_freq},${status}" >> "$RESULTS_FILE"

    echo -e "  Peak: $(temp_color "$RESULT_PEAK_TEMP") │ Media estable: ${avg_temp}°C │ Freq real promedio: ${avg_freq} MHz │ Estado: ${status}"

    # Cooldown entre frecuencias
    if ! $abort; then
        echo -e "  ${CYN}Cooldown ${COOLDOWN_TIME}s antes de siguiente frecuencia...${NC}"
        local target_cool=$(( RESULT_PEAK_TEMP - 12 ))
        for (( t=0; t<COOLDOWN_TIME; t+=5 )); do
            local ct
            ct=$(get_temp)
            printf "\r  Enfriando: $(temp_color "${ct:-0}") │ %ds restantes   " "$(( COOLDOWN_TIME - t ))"
            # Si ya enfrió suficiente, no esperar el cooldown completo
            [ "${ct:-99}" -le "$target_cool" ] && echo "" && break
            sleep 5
        done
        echo ""
    fi

    $abort && return 1
    return 0
}

# -----------------------------------------------------------------------------
# Reporte final
# -----------------------------------------------------------------------------
print_final_report() {
    echo ""
    echo -e "${BOLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "  REPORTE FREQ SWEEP — ${THREADS} threads — Threadripper PRO 7965WX"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    printf "${BOLD}%-10s │ %-8s │ %-10s │ %-10s │ %-10s │ %-12s │ %-8s${NC}\n" \
        "Freq MHz" "Threads" "Peak Temp" "Med Temp" "Freq real" "kH/s ~" "Estado"
    echo "───────────┼──────────┼────────────┼────────────┼────────────┼──────────────┼──────────"

    local best_freq=0 best_hr=0 best_temp=0

    if [ -f "$RESULTS_FILE" ]; then
        while IFS=',' read -r freq threads peak avg real_freq status; do
            # Estimado hashrate RandomX (kH/s): ~0.95 kH/s/core @ 4600 MHz
            local hr
            hr=$(awk "BEGIN{printf \"%.1f\", $threads * 0.95 * ($real_freq / 4600.0)}")

            local tcol="$GRN"
            [ "$peak" -ge 75 ] && tcol="$YEL"
            [ "$peak" -ge "$ABORT_TEMP" ] && tcol="$RED"

            local scol="$GRN"
            [ "$status" = "CALIENTE" ] && scol="$YEL"
            [ "$status" = "ABORT"    ] && scol="$RED"

            printf "%-10s │ %-8s │ ${tcol}%-10s${NC} │ ${tcol}%-10s${NC} │ %-10s │ ${GRN}%-12s${NC} │ ${scol}%-8s${NC}\n" \
                "${freq}" "${threads}t" "${peak}°C" "${avg}°C" "${real_freq} MHz" "~${hr} kH/s" "$status"

            # Guardar mejor configuración (mayor hashrate con status OK o CALIENTE)
            if [[ "$status" != "ABORT" ]]; then
                local hr_int
                hr_int=$(awk "BEGIN{printf \"%d\", $hr * 10}")
                local best_hr_int
                best_hr_int=$(awk "BEGIN{printf \"%d\", $best_hr * 10}")
                if [ "$hr_int" -gt "$best_hr_int" ]; then
                    best_freq=$freq
                    best_hr=$hr
                    best_temp=$peak
                fi
            fi
        done < "$RESULTS_FILE"
    fi

    echo ""
    if [ "$best_freq" -gt 0 ]; then
        echo -e "${BOLD}${GRN}  CONFIGURACIÓN ÓPTIMA DETECTADA:${NC}"
        echo -e "  ${GRN}Frecuencia: ${best_freq} MHz${NC}"
        echo -e "  ${GRN}Threads:    ${THREADS}t (todos los disponibles)${NC}"
        echo -e "  ${GRN}Hashrate ~: ${best_hr} kH/s (estimado — XMRig dará el valor real)${NC}"
        echo -e "  ${GRN}Temp peak:  ${best_temp}°C${NC}"
        echo ""
        echo -e "  Para aplicar permanentemente:"
        echo -e "  ${CYN}sudo cpupower frequency-set --max ${best_freq}MHz${NC}"
        echo -e "  ${CYN}sudo sed -i 's/4600MHz/${best_freq}MHz/' /etc/systemd/system/kza-cpu-freq-limit.service${NC}"
        echo -e "  ${CYN}sudo systemctl daemon-reload${NC}"
    fi
    echo ""
    echo -e "  CSV detallado: ${CSV}"
}

# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    [ -n "$STRESS_PID" ] && kill "$STRESS_PID" 2>/dev/null || true
    pkill -f stress-ng 2>/dev/null || true
    # Restaurar límite de frecuencia a 4600 MHz
    cpupower frequency-set --max 4600MHz &>/dev/null
    print_final_report
    exit 0
}
trap cleanup INT TERM EXIT

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if [[ "$EUID" -ne 0 ]]; then
    echo "Ejecutar como root: sudo bash $0"
    exit 1
fi

echo -e "${BOLD}${CYN}"
echo "  KZA — Frequency Sweep: ${THREADS} threads @ múltiples frecuencias"
echo -e "${NC}"
echo -e "  Objetivo: encontrar máxima frecuencia estable con ${THREADS}t bajo ${SAFE_TEMP}°C"
echo -e "  Frecuencias a testear: ${FREQ_STEPS[*]} MHz"
echo -e "  Soak por frecuencia: ${SOAK_TIME}s | Cooldown: ${COOLDOWN_TIME}s"
echo -e "  Duración total estimada: ~$(( (${#FREQ_STEPS[@]} * (SOAK_TIME + COOLDOWN_TIME)) / 60 )) minutos"
echo ""

# Inicializar CSV
echo "elapsed_s,target_mhz,temp_c,freq_mhz,load_1m,threads" > "$CSV"
echo "" > "$RESULTS_FILE"

# Ejecutar sweep
for freq in "${FREQ_STEPS[@]}"; do
    soak_freq "$freq" || {
        echo -e "${YEL}Frecuencias superiores abortadas — temperatura límite alcanzada${NC}"
        break
    }
done

# Restaurar y reportar (también lo hace cleanup via trap)
cpupower frequency-set --max 4600MHz &>/dev/null
print_final_report
