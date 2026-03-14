# KZA Deployment Day — Runbook para Claude

**Fecha:** 2026-03-15
**Trigger prompt:** `Estamos en el servidor de la KZA. Ubuntu está instalado. Arrancá el deployment.`

---

## Contexto que Claude debe recordar

- Spec completo: `docs/superpowers/specs/2026-03-14-deployment-analysis-design.md`
- Plan de fixes (ya implementados): `docs/superpowers/plans/2026-03-14-deployment-fixes.md`
- Los 8 fixes del código ya están en el repo (commits d20487c → 4a839cb)
- Habitación de testing: **escritorio**
- Red: TP-Link AX4000 → switch → KZA server + HA (misma subred)

---

## Secuencia de ejecución

### Fase 1: Verificar estado base
```bash
# Verificar que estamos en el servidor correcto
uname -a
lscpu | head -5
free -h
nvidia-smi
```
**Gate:** 4x RTX 3070 visibles, 128GB RAM, Ubuntu instalado.

### Fase 2: Setup del sistema
```bash
sudo ./scripts/setup_ubuntu.sh
```
**Gate:** Script termina sin errores. Python 3.13, CUDA, venv listos.

### Fase 3: Configurar .env
Preguntar al usuario:
- "¿Cuál es la IP de Home Assistant?"
- "¿Tenés el token de HA listo? Pegalo acá."
- "¿Tenés las credenciales de Spotify? (client_id y client_secret)"

Escribir `.env` con los valores reales.

### Fase 4: Descargar modelos
```bash
./scripts/download_models.sh
```
Si el LLM de 64GB tarda mucho, usar `--skip-llm` y seguir con las fases siguientes. Descargar el LLM en background después.

**Gate:** smoke_test.sh pasa.

### Fase 5: Detectar hardware de audio
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
arecord -l
```
Identificar el `device_index` del ReSpeaker del escritorio. Actualizar `config/settings.yaml` key `rooms.escritorio.mic_device_index` con el índice correcto.

### Fase 6: Verificar Home Assistant
```bash
source .env
curl -s -H "Authorization: Bearer $HOME_ASSISTANT_TOKEN" "$HOME_ASSISTANT_URL/api/" | head -5
```
**Gate:** HTTP 200.

### Fase 7: Primer arranque
```bash
python -m src.main
```
Monitorear logs. Verificar:
- Todos los modelos cargan sin OOM
- ChromaDB sync completa
- 4 GPUs muestran uso de VRAM (`nvidia-smi`)
- Dashboard responde: `curl http://127.0.0.1:8080/api/health`

### Fase 8: Test end-to-end
Pedir al usuario que diga:
1. **"Hey Jarvis"** → verificar wake word
2. **"Prende la luz del escritorio"** → verificar STT + HA action
3. **"Qué hora es"** → verificar Router 7B
4. **"Pon música en el escritorio"** → verificar Spotify (si configurado)

Verificar latencia < 300ms en logs.

### Fase 9: Activar systemd (si todo funciona)
```bash
sudo systemctl enable kza-voice
sudo systemctl start kza-voice
journalctl -u kza-voice -f
```

---

## Si algo falla

| Problema | Acción |
|----------|--------|
| nvidia-smi no ve GPUs | Verificar drivers, posible reboot post-install |
| pip install falla | Verificar Python 3.13, venv activado |
| vLLM OOM en GPU 2 | Verificar que el modelo es AWQ, no fp16 |
| HA no responde | Verificar IP, puerto 8123, token, firewall |
| Mic no detectado | Verificar USB, extensores Cat5e, `arecord -l` |
| Wake word no detecta | Ajustar threshold (0.3-0.7) |
| TTS no suena | Verificar MA1260 serial, zona correcta, fallback a Piper |
| LLM garbage output | Verificar chat_format=chatml en logs |

---

## Prompt de inicio

Cuando el usuario escriba algo como:

> "Estamos en el servidor de la KZA. Ubuntu está instalado. Arrancá el deployment."

O variantes como:
> "Ya estamos en el server, dale"
> "Ubuntu listo, arrancá"
> "Iniciá el deployment de la KZA"

Claude debe:
1. Leer este runbook
2. Leer el spec y el plan de referencia
3. Ejecutar las fases en orden
4. Pedir input del usuario solo para: tokens/credenciales y confirmación de tests de voz
5. No pedir confirmación entre fases — avanzar automáticamente si la gate pasa
