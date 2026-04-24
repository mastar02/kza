#!/usr/bin/env bash
# Monitor térmico/VRAM de las GPUs (RTX 3070 consumer no exponen VRAM temp).
# Usa proxies: core temp, power draw, VRAM usage, memory clock.
#
# Flags de alerta:
#   🔥  core temp > 75°C  (VRAM probablemente > 85°C)
#   💾  VRAM usada > 95%  (riesgo OOM)
#   ⚡  power > 180W sostenido
#
# Uso:
#   bash scripts/gpu_thermal.sh            # snapshot
#   bash scripts/gpu_thermal.sh watch 2    # refresh cada 2s
#   bash scripts/gpu_thermal.sh log        # append CSV a logs/gpu_thermal.csv
set -u

ALERT_TEMP=75
ALERT_VRAM_PCT=95
ALERT_POWER=180

query() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,\
utilization.gpu,utilization.memory,memory.used,memory.total,\
fan.speed,clocks.current.graphics,clocks.current.memory \
      --format=csv,noheader,nounits
}

fmt() {
    IFS=',' read -r idx name temp pw pwmax utilg utilm mused mtot fan clkg clkm <<< "$1"
    temp=$(echo "$temp" | xargs)
    pw=$(echo "$pw" | xargs | cut -d. -f1)
    utilg=$(echo "$utilg" | xargs)
    utilm=$(echo "$utilm" | xargs)
    mused=$(echo "$mused" | xargs)
    mtot=$(echo "$mtot" | xargs)
    fan=$(echo "$fan" | xargs)
    pct=$(( mused * 100 / mtot ))

    flag=""
    [ "$temp" -gt "$ALERT_TEMP" ] 2>/dev/null && flag+=" 🔥"
    [ "$pct" -gt "$ALERT_VRAM_PCT" ] 2>/dev/null && flag+=" 💾"
    [ "$pw" -gt "$ALERT_POWER" ] 2>/dev/null && flag+=" ⚡"

    printf "GPU %s | %d°C | %3dW | util %3s%% gpu / %3s%% mem | VRAM %4d/%4d MiB (%d%%) | fan %s%% | clk %s/%s MHz%s\n" \
        "$idx" "$temp" "$pw" "$utilg" "$utilm" "$mused" "$mtot" "$pct" "$fan" "$clkg" "$clkm" "$flag"
}

print_once() {
    echo "== $(date +'%H:%M:%S') =="
    while IFS= read -r line; do fmt "$line"; done < <(query)
}

mode="${1:-once}"
case "$mode" in
    once) print_once ;;
    watch)
        interval="${2:-2}"
        while true; do
            clear
            print_once
            echo ""
            echo "(Ctrl-C para salir — refresh ${interval}s)"
            sleep "$interval"
        done
        ;;
    log)
        mkdir -p logs
        [ ! -f logs/gpu_thermal.csv ] && echo "ts,gpu,name,temp_c,power_w,power_limit_w,util_gpu,util_mem,vram_used_mib,vram_total_mib,fan_pct,clk_gpu_mhz,clk_mem_mhz" > logs/gpu_thermal.csv
        while IFS= read -r line; do
            ts=$(date -Iseconds)
            echo "$ts,$(echo "$line" | sed 's/, */,/g')" >> logs/gpu_thermal.csv
        done < <(query)
        tail -2 logs/gpu_thermal.csv
        ;;
    *)
        echo "Uso: $0 once | watch [secs] | log"
        exit 1
        ;;
esac
