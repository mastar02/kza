# Sesión 2026-05-30 — XVF3800 swap + cadena wake→NLU→HA arreglada

**Estado final:** la luz del escritorio **funciona end-to-end** con el mic XVF3800. 10 commits merged a `main` (fast-forward `75d4685` → `3706e89`). Servicio `kza-voice` corriendo.

> Para retomar: leer también la memoria persistente `project_tv_mode_hallucination_rootcause_2026-05-30.md` y el plan `docs/superpowers/plans/2026-05-30-wake-fastpath-hallucination-quality.md`.

---

## Qué motivó la sesión
El usuario reportó que los comandos de la luz del escritorio "no tenían efecto". Diagnóstico inicial: el wake nunca se aceptaba por **alucinaciones de Whisper sobre silencio** activando TV-mode espurio. Durante la sesión el usuario **swappeó el mic del escritorio al XVF3800** (array + beamforming/DOA del living), lo que destapó múltiples capas rotas (whack-a-mole) porque el pipeline estaba calibrado para el mic viejo (UAC1.0 en cielorraso).

## Fixes aplicados (todos en `main`)

| Commit | Qué |
|--------|-----|
| `e0231f6` | Alucinaciones (`tv_stop_phrase`, etc.) ya no alimentan TV-mode (`_NON_TV_MODE_REJECT_REASONS`). Era activación espuria (716 veces/870h). |
| `61a3e1d` | Captura `no_speech_prob`/`avg_logprob` en el wake detector + métricas (instrumentación). |
| `cccbb12`/`b117e05` | Denylist de alucinaciones de utterance-completa ("¡Gracias!" ×12.440, etc.), match exacto, no cuenta TV-mode. |
| `2156401` | **Canales**: el XVF3800 es 2ch; se abría a `channels=1` → audio interleaved → garbled. Ahora abre con canales nativos del device + toma ch0. (El mic viejo UAC1.0 reporta 0ch → fallback a 1.) |
| `a0d17c4` | **Parseo JSON del router**: el modelo en `:8101` antepone preámbulo/pseudo-objetos; scanner de llaves balanceadas extrae el JSON real. |
| `39ab74e` | **Fuzzy wake → canónico**: el detector acepta "dexa"~"nexa" pero el CommandGate hace match exacto → `missing_wake`. Ahora se reescribe a "Nexa" en el texto pretranscripto. |
| `ed9f3b9` | **`use_silero_vad: false`** (en settings.yaml, working tree): el XVF3800 hace su propio DSP → el Silero VAD del pipeline devolvía `prob≈0` sobre voz fuerte → gateaba el comando. Ahora gate **RMS-only**. |
| `3706e89` | **Gramática autoritativa en domótica**: el LLMRouter (Qwen-7B Q4_K_M en :8101, modelo débil) clasificaba comandos válidos como `noise` de forma inconsistente. Ahora `parse_partial_command` (intent+entity, conf≥0.75) saltea el LLMRouter para domótica clara. |

## Hallazgos de arquitectura (importantes)
- El `--detect` de `room_context` usa `create_default_rooms()` HARDCODEADO y resuelve mics por NOMBRE+orden — **engaña**. El servicio real lee `rooms.*` de settings.yaml y resuelve por **puerto USB** (`resolve_mic_usb_port`). El XVF3800 quedó en puerto **`3-1.4`** (card1) → escritorio. Living sin mic (temporal).
- Hay DOS detectores: `WhisperWakeDetector` (activo) y `StreamingWhisperWakeDetector`, clases separadas. Silero carga lazy en `load()`.
- El fast-path regex (`regex_extractor`/`llm_gate` en request_router) está **apagado** (None, sin logs) — por eso todo caía al LLMRouter. El grammar fast-path nuevo lo suple.
- La llamada HA por WebSocket (`call_service_ws`) **falla en el primer comando post-arranque** (WS conecta lazy, transitorio), se auto-resetea (`_ws_connected=False`) y el 2º comando anda. Fire-and-forget NO reintenta.

## Latencia medida (comandos exitosos)
- **Grammar fast-path: ~508 ms** pipeline (STT=0 porque reusa transcripción del wake; vector ~58-155 ms) + **55-165 ms** HA call ≈ **~567 ms voz→acción**.
- Vía LLMRouter (sin fast-path): 1457-1946 ms.
- **+ delay físico Hue Zigbee ~1-1.2 s** (bridge→bombilla, NO es KZA; baja migrando a Zigbee2MQTT, infra prep en Notion 9.6).
- Objetivo CLAUDE.md <300 ms: hoy ~500 ms el pipeline → margen de optimización.

---

## PENDIENTE para la próxima sesión

1. ~~**`config/settings.yaml` tiene WIP sin commitear**~~ **✅ RESUELTO 2026-05-30 (sesión siguiente, commits `87e4315` + `61a8e3c`).** El WIP de settings se commiteó separado por concern. **CORRECCIÓN al diagnóstico de esta sesión:** el gateway **NO estaba roto**.
   - `reasoner.http_base_url` → `http://192.168.1.2:8200/v1`: **es la arquitectura nueva correcta**, no un WIP a revertir. Cutover documentado en Notion pág 8: `:8200` es un **gateway LiteLLM centralizado bajo `infra`** que proxya a MiniMax-M2.7 (auditoría + egress central; el modelo local gpt-oss-120b fue borrado, `kza-llm-ik` quedó `.disabled`). El `400 "No connected db"` era **key incorrecta del consumer** (LiteLLM con master_key estática trata una key distinta como virtual key y la busca en DB inexistente), **NO gateway/DB caído**. El reasoner principal de KZA siempre conectó `200 OK` (su `MINIMAX_API_KEY` ya == `LITELLM_MASTER_KEY`).
   - **El único bug real era el compactor**: `main.py` construía el compaction reasoner hardcodeado a `127.0.0.1:8200` SIN api_key → 400 → `Compactor disabled`. Fix (`87e4315`): reusa el gateway autenticado del reasoner principal (`base_url`+`api_style`+`api_key_env` de config). Verificado: `[main] Compactor enabled`.
   - Mic rebind escritorio (`mic_usb_port: "3-1.4"`, XVF3800) + living `null` + `use_silero_vad: false` → commiteados en `61a8e3c`.
   - Backups `settings.yaml.bak.*` quedaron como untracked (no commiteados, intencional).
   - Ver memoria `project_litellm_gateway_cutover_2026-05-30.md`.
2. ~~**Push a `origin/main`**~~ **✅ RESUELTO 2026-05-30**: `git push origin main` (fast-forward limpio, origin estaba 13 commits atrás, sin divergencia).
3. **Notion desactualizado** (stack LLM): pág "2.5 LLM y razonamiento" (2026-04-17) y "7.3 LLM stack overhaul" (2026-04-28) describen vLLM AWQ :8100 + ik_llama :8200. **Realidad actual**: fast_router = Qwen-7B **Q4_K_M en :8101** (reemplazó vLLM :8100 deshabilitado), reasoner = gateway/MiniMax :8200. No hay página del gateway. Actualizar cuando el stack esté estable.
4. **LLMRouter Q4_K_M débil**: el grammar fast-path lo evita para domótica, pero para queries conversacionales/no-domótica el router sigue siendo poco fiable. Considerar revertir a vLLM AWQ o subir quant.
5. **7 tests pre-existentes rotos** en `tests/unit/nlu/test_llm_router.py` (`fake_generate()` signature mismatch — el mock no matchea la firma real de `classify`) + 1 en `test_endpointing.py`. Ajenos a esta sesión, pero conviene arreglar el mock.
6. **HA call WS fallback**: considerar fallback a REST en `call_service_ws` cuando `success=False` (REST está probado y funciona) para blindar el primer-comando-post-arranque.
7. **Optimización de latencia** hacia <300 ms: reactivar fast-path regex o quedarse con grammar fast-path; optimizar vector search / wake Whisper.
8. **XVF3800 / array**: explorar DOA/beamforming (Fase 5 del plan) — el living quedó sin mic; reconectar UAC1.0 al living si se quiere multi-room.

## Cómo verificar que sigue andando
```bash
ssh kza 'cd ~/app && git log --oneline -1'   # debe ser 3706e89 o posterior
ssh kza 'systemctl --user is-active kza-voice.service'
# Decir "Nexa, prendé la luz del escritorio" → ver:
ssh kza 'journalctl --user -u kza-voice.service -f | grep -E "GRAMMAR_FASTPATH|HA-CALL"'
```
