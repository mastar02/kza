# SOTA ASR español local (2025-2026) — ¿reemplazar el stack de transcripción?

**Fecha**: 2026-06-07 · **Método**: deep-research (105 agentes, 22 fuentes, 104 claims extraídos, 25 verificados adversarialmente con 3 votos c/u → 21 confirmados, 4 refutados) · **Contexto**: faster-whisper large-v3-turbo (CT2) en RTX 3070 8GB; command path <300ms + ambient path 24/7; bugs confirmados del turbo (alucinaciones sobre no-voz, repetition loops, `no_speech_prob` degenerado ~1e-10).

## TL;DR — recomendación por path

| Path | Veredicto | Modelo |
|---|---|---|
| **Ambient 24/7** | **Cambiar — candidato fuerte, validar con benchmark A/B propio** | **NVIDIA Parakeet-TDT-0.6B-v3** |
| **Command <300ms** | **NO tocar por ahora** | turbo beam 1 se queda (nadie midió <300ms en 3070 para los candidatos) |

## Hallazgos confirmados (3-0 salvo indicación)

1. **El problema es estructural de Whisper, cuantificado**: large-v3 alucina en el **40.3% del audio no-voz** (52.1% en clips de 1s); "thank you" (24.76%) y "thanks for watching" (10.32%) dominan — los equivalentes exactos de nuestros "Gracias." / "Gracias por ver el video." (ICASSP 2025, arxiv 2501.11378). Mecanismo: entrenamiento sobre transcripciones de video.

2. **Parakeet-TDT-0.6B-v3 y Canary-1B-v2 fueron diseñados contra esto**: NVIDIA los entrenó con **36.000 horas de audio no-voz con target string-vacío** ("teach the model when not to generate transcriptions", arxiv 2509.14128). Ataca exactamente nuestra clase de falla.

3. **Parakeet-TDT es el mejor fit para el ambient path**: 600M FastConformer-**TDT transducer** (frame-synchronous — arquitecturalmente no puede "free-runear" texto como el decoder autoregresivo de Whisper, lo que también mata los repetition loops), el más robusto a ruido del par NVIDIA (12.21% WER @ -5dB SNR vs 19.38% de Canary), ~40× más rápido que Whisper en long-form, español entre 25 idiomas con autodetección, entra cómodo en 8GB. Licencia CC-BY-4.0.

4. **Canary-1B-v2**: viable pero peor para nosotros — decoder AED (familia de Whisper, más propenso a alucinar), menos robusto a ruido, y sus features extra (traducción) no aportan.

5. **Qwen3-ASR-1.7B** (Apache-2.0, ene-2026): supera a large-v3 en Fleurs/CommonVoice multilingüe (4.90/9.18 vs 5.27/10.77; español 3.36/4.65) — candidato para el command path A FUTURO, pero: números self-reported de Alibaba, stack de inferencia local inmaduro (sin CT2/path GPU optimizado), y su claim de robustez a ruido fue **REFUTADO 0-3**.

6. **Voxtral Mini 4B Realtime (Mistral)**: el mejor WER español crudo (3.31% Fleurs @ 480ms) pero **descartado**: requiere ≥16GB VRAM BF16, int8 no soportado oficialmente.

7. **Calm-Whisper** (Interspeech 2025): fine-tunear solo los 3 decoder heads "locos" con ~105h de no-voz/blank reduce alucinación **84%** con <0.1% costo WER — PERO validado en large-v3 (32 capas decoder), **no transferible directo al turbo (4 capas)**. Solo relevante si volviéramos a large-v3 no-turbo.

8. **Transducers/CTC vs attention-decoders**: 10-100× throughput con WER levemente peor — la familia correcta para ambient continuo. ⚠️ El 10-100× es throughput batch en A100, NO latencia single-utterance en 3070: el <300ms del command path hay que **medirlo**, no asumirlo.

## Claims REFUTADOS (no asumir)

- ❌ "Parakeet español WER ~3.45 Fleurs" (1-2) — su calidad en español es plausible pero **no está sólidamente establecida**. → por eso el benchmark propio es obligatorio.
- ❌ "Qwen3-ASR 16.17 vs Whisper 63.17 en ExtremeNoise" (0-3) — la ventaja de ruido de Qwen3 NO está demostrada.
- ❌ "Silero VAD pre-gating lleva alucinación/WER a ~0" (0-3) — nuestro `vad_prob` es valioso pero NO bala de plata.
- ❌ "Parakeet RTFx 3332 / Canary 749 ⇒ headroom directo en 3070" (1-2) — números de A100, no transferibles.

## Caveats transversales

- La evidencia en **español es el eslabón débil**: casi todos los números de robustez/velocidad son inglés o ruido sintético (LibriSpeech+MUSAN), no rioplatense far-field con TV.
- La propia model card de Parakeet avisa que "can hallucinate extra repetitions" — es comparativamente mejor, no perfecto.
- Stack: Parakeet/Canary requieren **NeMo u onnx export** (stack nuevo en el server); Whisper/CT2 es lo maduro que ya corremos.
- Área que se mueve rápido (Qwen3-ASR es de ene-2026): re-chequear en meses.

## Preguntas abiertas (las responde el benchmark A/B en el server)

1. Latencia wall-clock single-utterance de Parakeet-TDT v3 (y Qwen3-ASR) en la 3070 para comandos cortos en español — ¿le gana a turbo ~200ms?
2. Comportamiento REAL sobre nuestro audio: capturas XVF3800 far-field + TV (`/home/kza/bench-ambient`: `tv_capture_2ch.wav`, `voz_capture_2ch.wav` con ground truth) + corpus ambient.db — tasa de alucinación sobre no-voz y WER sobre comandos reales.
3. VRAM/convivencia: Parakeet via NeMo/onnx junto al resto de la carga de cuda:0.

## Plan propuesto (gated)

1. **Ahora**: dejar correr las 24h de ingesta (calibra `vad_prob` para el stack actual — útil con CUALQUIER modelo).
2. **Benchmark A/B en el server** (worktree bench-ambient, sin tocar prod): turbo vs Parakeet-TDT-0.6B-v3 sobre las capturas de Fase 0 + muestras de ambient.db. Medir: alucinaciones sobre tramos no-voz, WER sobre tramos con ground truth, RTF y VRAM reales.
3. **Si Parakeet gana**: swap SOLO del ambient path (`AmbientSTT` ya está aislado por DI — es cambiar el modelo que recibe). Command path queda en turbo.
4. Command path: re-evaluar Qwen3-ASR cuando su stack local madure (o si el benchmark sorprende).

## ✅ Benchmark A/B ejecutado (2026-06-07, server, audio real XVF3800)

Setup: `/home/kza/bench-ambient/bench_ab_stt.py`, venv aislado `.venv-bench` (onnx-asr 0.11 + faster-whisper), 17 clips de 10s/3s del canal ch1 ASR (voz en llamada, far-field con ground truth, TV-only, silencio/ruido sintético). Resultados en `bench_ab_results.json`. **Nota: Parakeet corrió en CPU** (onnxruntime sin CUDA EP) — la calidad no depende del provider y el RTF CPU ya alcanza.

| Métrica | turbo (CT2, cuda:0) | Parakeet-TDT-0.6B-v3 (onnx, **CPU**) |
|---|---|---|
| **Alucinación sobre no-voz (5 clips silencio/ruido)** | **5/5** ("Gracias por ver el video.") | **0/5** (string vacío) |
| Calidad voz (anime en llamada) | peor ("sinobi", "a cumplir las velas... opciones") | mejor ("shinobi", "al cumplir... superiores", captó "¿Me llamó?" omitido por turbo) |
| **Pérdida de contenido** | tv_10-20s → alucinó "Gracias." sobre diálogo real | transcribió "Sí, excelente trabajo, Pakashi." |
| Far-field 3-4m + TV (usuario real) | garble | garble (≈) — **es problema de señal, ningún modelo lo salva** |
| Latencia clip 3s | ~150ms (GPU) | **~130ms (CPU!)** |
| Latencia clip 10s | ~160-210ms (GPU) | ~250-340ms (CPU), RTF ~0.03 |
| VRAM | ~1.5GB cuda:0 | **0 (CPU)** |

**Veredicto: Parakeet-TDT-0.6B-v3 gana el ambient path por KO** — elimina la clase entera de alucinaciones (el target #1), mejor calidad en voz real, y corre en CPU (libera ~1.5GB de cuda:0 si se quita el Whisper ambient). El far-field garbleado queda igual → sigue siendo el caso del flasheo 6ch/DoA. Command path: sin cambios (turbo GPU ~150ms).

## Fuentes principales

- arxiv 2501.11378 (ICASSP 2025 — cuantificación alucinaciones Whisper)
- arxiv 2509.14128 (NVIDIA — Parakeet/Canary multilingüe, 36kh no-voz)
- arxiv 2505.12969 (Calm-Whisper, Interspeech 2025)
- huggingface.co/nvidia/parakeet-tdt-0.6b-v3 · huggingface.co/blog/open-asr-leaderboard
- github.com/QwenLM/Qwen3-ASR · arxiv 2601.21337
- huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
