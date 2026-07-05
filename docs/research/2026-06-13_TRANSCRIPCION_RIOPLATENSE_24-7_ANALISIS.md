# INFORME: Cómo KZA transcribe español rioplatense uruguayo en tiempo real 24/7 — foco en la calidad de escucha

> **Fecha:** 2026-06-13 · **Rama:** `feat/nexa-command-detection-fixes`
> **Método:** investigación multi-agente (Workflow, 28 agentes: 7 lectores de
> código + 4 investigadores web + 16 verificaciones adversarias + síntesis).
> Las afirmaciones llevan etiqueta CONFIRMADO / INCIERTO / REFUTADO según
> verificación independiente (ver §8). Verificado además localmente:
> firmware 2ch (`docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md`), tests
> `test_language_quality.py` (14/14), 3 tests rojos del schema `lang` en
> `test_store.py`.

## 1. Resumen ejecutivo (veredicto central)

- **El cuello de botella del rioplatense far-field NO es un solo problema, son tres apilados, y el más caro está al PRINCIPIO de la cadena (la señal), no al final (el modelo).** El front-end del XVF3800 (beamforming + AEC + NS + AGC ×8) entrega al ASR una señal far-field 3-4m de la que ningún modelo recupera rioplatense limpio. Esto está medido en el A/B propio (turbo=garble, Parakeet=garble) — pero, ojo, la conclusión "ningún modelo lo salva" quedó marcada **incierta** en verificación (ver §8), porque se generalizó desde N=2 modelos sobre N=17 clips sin WER etiquetado.

- **`language='es'` en el ambient path (Parakeet) es un NO-OP CONFIRMADO al 100% en triple fuente independiente** (código onnx-asr + arquitectura/código NeMo + un maintainer de NVIDIA). Parakeet-TDT es un transducer que no recibe, no usa ni emite identificador de idioma; el kwarg entra por `**kwargs` y se descarta en silencio. El proyecto cree forzar español y no lo hace. El **command path SÍ fuerza idioma de verdad** (faster-whisper, `language='es'` real) — la asimetría entre los dos paths es estructural y favorece al comando.

- **El discriminante real de la calidad del rioplatense es el IDIOMA DEL TEXTO (langid), no la energía (`vad_prob`).** Esto está confirmado por mecanismo (LID neuronal sesgado a inglés sobre señal débil) y por el propio diseño del repo. El módulo que ataca esto correctamente — `src/ambient/language_quality.py` (langid + marcadores rioplatenses + flag-no-drop) — está escrito, testeado (14/14) **pero COMPLETAMENTE SIN CABLEAR** (untracked en git, 0 imports en `src/`). Es la palanca de software de máximo apalancamiento, hoy inerte.

- **El langid desplegado (py3langid) misclasifica rioplatense LIMPIO con altísima confianza** a francés/catalán/gallego/italiano/walloon (verificado empíricamente). Esto contamina el conflicto sin resolver "inglés=100% TV vs ~30-47% inglés": parte de ese "inglés/no-es" es rioplatense bien transcripto que el LID estadístico tira a otra lengua romance. El gate desplegado `drop_language='en'@0.9` NO dropea esos casos (solo dropea 'en'), pero invalida cualquier medición futura por 'es' y, en Fase 3 (enforce activo), corre el riesgo de borrar conversación real del hogar sin rastro.

- **No se está midiendo ni conservando la calidad del rioplatense.** Con Parakeet, las columnas `confidence`/`no_speech_prob` de `ambient.db` son siempre NULL; la única señal es `vad_prob` (energía, no idioma). No hay columna `lang`, no hay etiqueta de audit, y la retención de 12h borra la evidencia far-field antes de poder auditarla o juntar dataset de re-entrenamiento. El orden de ejecución está invertido: se promovió el gate enforce ANTES de cablear el flag-no-drop.

---

## 2. El pipeline de escucha end-to-end — etapa por etapa

### Front-end común (hardware)
Micrófono XVF3800 (XMOS), far-field 3-4m, DSP por hardware: AEC + noise suppression + beamforming + AGC ×8 (`PP_AGCMAXGAIN=8.0`, `apply_on_start`, `config/settings.yaml:1090-1093`). **Firmware actual CONFIRMADO 2ch** (`docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md:7`, snapshot del chip real 06-04, VERSION 2.0.6): ch0=Conference (beam + post-procesador con NS), ch1=ASR (mismo beam, tap pre-post-processor, sin NS no-lineal). **Los mics crudos ch2-5 NO EXISTEN** — el flasheo 6ch está GATED y nunca se ejecutó.

**Dónde se gana/pierde español:** el AGC ×8 amplifica el piso de ruido de sala junto con la voz a 3-4m; RMS y SPENERGY quedaron "muertos" como discriminadores justamente por el AGC (`settings.yaml:1210`). La literatura (arxiv 2512.17562, "When De-noising Hurts": 40/40 configs, audio crudo gana hasta +46.6% WER absoluto sobre el "mejorado") indica que este front-end agresivo **daña** el ASR neuronal moderno. Esta es la pérdida de español más temprana y probablemente la mayor.

### COMMAND PATH (<300ms, GPU cuda:1)

1. **Wake "Nexa"** (openwakeword/nexa.onnx, acústico).
2. **Captura ch1** (beam ASR) → `CommandProcessor.process_command`.
3. **faster-whisper large-v3-turbo** (`src/stt/whisper_fast.py:166-202`): `language="es"` **REAL y forzado** (whisper_fast.py:169 — fuerza el token de idioma en el decoder, sin auto-detect), `beam_size=5`, `best_of=1`, `temperature=0`, `condition_on_previous_text=False`, `vad_filter=False`, `initial_prompt` de dominio. `compute_type=int8_float16` (VRAM compartida con el 7B :8101). Thread pineado (max_workers=1) para evitar el `CUDA invalid argument` multi-GPU.
4. **CommandAcceptanceGate** (`command_gate.py`): reglas HARD (siempre enforce: empty, noise phrases TV/YouTube, filler, word_repetition, missing_wake) + CONFIDENCE (shadow) + COMPRESSION (shadow propio).
5. **RequestRouter** → grammar fastpath / LLM router / orchestrator.

**Dónde gana español:** `language='es'` real + initial_prompt de dominio + beam_size=5 + el usuario habla más cerca/fuerte tras el wake. El command path es **estructuralmente inmune** al language-collapse a inglés.

**Dónde pierde español:**
- El `initial_prompt` (`settings.yaml:168-171`) NO contiene voseo imperativo (`prendé/apagá/subí/bajá/poné`) — solo rooms y dominios. Whisper puede normalizar `prendé→prende/prender`. Único lever léxico del path, infrautilizado.
- `no_speech_prob ≈ 1e-10` degenerado y `avg_logprob` INVERTIDO (medido sobre 14.288 eventos) → ambas señales de confianza en shadow. `compression_ratio` es la única señal viva, pero **también en shadow sin calibrar** (`enforce_compression_ratio=false`, `settings.yaml:274`).
- `temperature=0` escalar sin lista de fallback desactiva el re-decode adaptativo de faster-whisper sobre segmentos garbleados.
- El command path NO consume `vad_prob` — un garble far-field con TV de fondo que no matchee una noise-phrase pasa el gate.

### AMBIENT PATH (24/7 continuo, Fase 3 enforce activo)

1. **Tap** (`src/pipeline/multi_room_audio_loop.py:565-578`): el callback C de PortAudio hace `self._ambient_tap.push(rs.room_id, indata.copy(), ...)` — empuja el array **multicanal completo** (con fw 2ch: shape `(1280, 2)`). Es **fail-open y O(1)**: try/except que traga toda excepción → el command path nunca se bloquea por el ambient. `MultiChannelTap` = ring buffer por room (deque maxlen 1250 ≈ 100s, FIFO-drop).

2. **UtteranceSegmenter** (`src/ambient/segmenter.py`): VAD Silero sobre `vad_col=2`. **CONFIRMADO: con fw 2ch hace fallback a col 0** (segmenter.py:92-104) = **ch0 Conference POST-NS** — contradice el diseño nominal (VAD sobre crudo), aunque el docstring ya documenta el fallback (por eso la acusación de "contradice el docstring" quedó **refutada**, ver §8). `speech_threshold=0.3` (calibrado far-field). Máquina de estados: pre-roll 500ms, `min_speech_ms=300` (≥4 chunks de 80ms), `close_silence_ms=700`, `max_segment_s=30`.
   - **`vad_prob` = mean-of-maxes que INCLUYE la cola de 700ms de silencio** (segmenter.py:50,133,168 — CONFIRMADO). El mismo enunciado da `vad_prob` distinto según cuánta cola le tocó: voz idéntica de 1s vs 2s da ~0.41 vs ~0.63, **cruzando el gate de 0.45**. No es interpretable como "calidad de voz" estable.
   - **Cobertura de ventana incompleta** (CONFIRMADO): chunk=1280, ventanas Silero de 512 step 512 → solo se evalúan 1024/1280 samples (64 de 80ms); los últimos 16ms de cada chunk nunca pasan por el VAD.

3. **AmbientSTT → ParakeetSTT** (`src/ambient/ambient_stt.py` + `parakeet_stt.py`): toma `audio[:, asr_col]` con `asr_col=1` = beam ASR (**canal correcto**, existe con 2ch). `parakeet_stt.py:89`: `text = self._model.recognize(audio, language=self.language) or ""` — **`language='es'` es NO-OP confirmado**. CPU, 0 VRAM. `STTResult` con `no_speech_prob/avg_logprob/compression_ratio = None`.

4. **SpeakerTagger** (`speaker_tagger.py`): corre sobre `seg.audio[:, 0]` = ch0 Conference (asimetría deliberada: ECAPA quiere full-band post-proc). Pero `embeddings_loader` apunta a `./data/users` que **no existe** → SIEMPRE `("unknown", 0.0)`.

5. **DoAEstimator** (`doa.py:62-65`): requiere `shape[1] ≥ raw_first_col + n_raw = 6`. **Con 2ch → retorna None SIEMPRE** (CONFIRMADO). azimuth=None.

6. **SourceClassifier** (`source_classifier.py`): `during_tts→self`; speaker conocido→live (nunca); azimuth None→unknown (SIEMPRE). **Resultado CONFIRMADO: ~100% 'unknown' salvo 'self'** (prod: 1833/1833 unknown). `tv_azimuth=null`. **La separación TV/voz-del-hogar por dirección NO EXISTE hoy.**

7. **AmbientStore.add** (`store.py`): persiste a `data/ambient.db` sin tocar idioma. `AmbientUtterance` (`types.py:39-56`) **no tiene campos de idioma**. `confidence`/`no_speech_prob` quedan NULL; `vad_prob` es la única señal real.

8. **Distiller** (Fase 3, `shadow_mode=false`, corre cada 6h): `undistilled_live(min_vad_prob=0.45)` → `_log_language_shadow` (cuenta idiomas, efímero a stdout) → `_apply_language_gate` (`distiller.py:254-277`): dropea `lang=='en' AND prob>=0.9` con py3langid. **Las filas dropeadas se `mark_distilled` igual** → la purga de 12h las borra. Pérdida irreversible.

**Dónde gana español el ambient:** ch1 es el canal ASR correcto; Parakeet elimina la clase alucinación-sobre-silencio (0/5 vs 5/5); 0 VRAM. **Dónde pierde:** todo lo anterior — language-collapse a inglés sin control, sin red de seguridad por-utterance, gate de idioma ciego al rioplatense y aguas abajo, y retención hostil al re-entrenamiento.

---

## 3. EL problema central: ¿`language='es'` es NO-OP? ¿Parakeet colapsa el rioplatense a inglés-garble?

### `language='es'` es NO-OP — **CONFIRMADO al 100%, triple fuente independiente**

1. **Código onnx-asr** (`src/onnx_asr/adapters.py`): el `TypedDict RecognizeOptions` documenta textualmente `language: str | None` = *"Speech language (only for Whisper and Canary models)"*. Como `total=False` y entra por `**kwargs`, `recognize(audio, language='es')` con Parakeet **no lanza error: se descarta**.
2. **Arquitectura NeMo** (`src/onnx_asr/models/nemo.py`): `NemoConformerTdt` (Parakeet) hereda de RNNT y **NO lee `language`**; el decode es greedy-argmax puro sobre el transducer, solo consume `need_logprobs`. En cambio `NemoConformerAED` (Canary) SÍ inyecta `batch_tokens[:, 4] = self._tokens['<|{language}|>']`. Esta es la diferencia arquitectónica exacta.
3. **Maintainer de NVIDIA** (NeMo issue #14799, CLOSED): nithinraok responde literal *"unfortunately parakeet-v3 model doesn't receive or output language id"*. El workaround `change_decoding_strategy(decoding_cfg.language=...)` fue PROBADO y falla (*"decoding_cfg doesn't have language key"*). NeMo #15097: *"It is not supported by the model to output language. You may use langid_ambernet."*

El propio docstring de `parakeet_stt.py:1-19` y de `language_quality.py:3-6` ya documentan que es no-op. **Residual flagueado:** no se pudo ejecutar en el venv del server (onnx_asr no instalado local), pero el código vuelve el comportamiento determinista — confirmaría, no decidiría.

⚠️ **Bug latente de creencia:** el comentario de `parakeet_stt.py:45-47` dice *"language: Idioma forzado para recognize() ... fijarlo evita drift"* — esto es **falso y peligroso**: ninguna sesión futura debe creer que fija el idioma.

### ¿Parakeet colapsa el rioplatense a inglés-garble? — **PLAUSIBLE pero NO PROBADO sobre audio etiquetado (incierto)**

El **mecanismo está confirmado por literatura independiente**, no es conjetura: ASR multilingüe ante fonemas degradados far-field los mapea al idioma dominante (inglés) en vez de `[UNK]` (Gladia, code-switching); la precisión de LID cae con SNR, peor bajo babble multi-locutor (TV) y con sesgo fuerte al inglés en modelos English-dominant (Springer/NCBI). El LID de Parakeet es **implícito** en un tokenizer SentencePiece compartido (8192 tokens), reportado independientemente como "language contamination" en idiomas chicos (NeMo #15097).

**PERO la afirmación causal específica "el rioplatense far-field colapsa a inglés-garble por-utterance" NO está probada:** `ambient.db` local está vacía (0 bytes — el corpus vive en el server, purgado cada 12h); no existe audit ciego etiquetado; el A/B midió alucinación-sobre-no-voz, no WER ni tasa de colapso-de-idioma; y que py3langid etiquete un texto como inglés NO prueba que Parakeet produjera inglés a partir de voz española (el propio LID misclasifica — §4). El A/B "gana por KO" es sobre alucinaciones, no sobre fidelidad del español débil.

---

## 4. Brechas de cobertura del rioplatense (voseo / sheísmo / léxico uruguayo / langid)

**El stack tiene dos niveles de conciencia dialectal MUY desparejos:**

| Capa | Command path | Ambient path |
|---|---|---|
| Voseo imperativo | **SÍ**: `src/nlu/regex/vocab.py:32` `IMPERATIVES_VOSEO` (`apagá/prendé/bajá/subí/poné/ponele/abrí/cerrá...`), voseo-only por diseño | **NO**: nada en el ASR |
| Acentos preservados | **SÍ**: `normalize.py:5` (distingue `apagá` imperativo de indicativo) | N/A |
| Prompt en rioplatense | LLM gate: `llm_gate.py:52` "validador binario en español rioplatense" | Solo el `_SYSTEM_PROMPT` del distiller (`distiller.py:22`, voseo) — pero es prompt, no garantiza que el 7B entienda léxico UY |
| Léxico uruguayo (championes/túper/gurí/garúa/ta/bo/botija) | NO modelado | NO modelado |

**Dónde se pierde el rioplatense:**

1. **El ASR ambient no fue entrenado/evaluado con rioplatense ni far-field.** El doc de research lo admite: *"La evidencia en español es el eslabón débil... no rioplatense far-field con TV"* (`2026-06-07_SOTA...:39`). Whisper-v3 vio castellano+latam mezclado; ningún modelo público es es-AR/es-UY. El léxico (vos/tenés/dale/boludo) NO es problema acústico sino **prior de lenguaje** → se ataca con fine-tune/LoRA o initial_prompt, no con cambio de modelo.

2. **py3langid misclasifica rioplatense LIMPIO** (hallazgo refinado y VERIFICADO empíricamente con el detector exacto del gate, `norm_probs=True`):
   - `'che boludo prendé la tele'` → **fr@1.000**
   - `'ponele que sí'` → **pt@0.787**
   - `'che boludo prendé la luz del living dale'` → **wa (walloon) 0.985**
   - `'vos sabés que no me gusta'` → **ca 0.985**
   - `'a ver qué onda'` → **fr 0.997**
   - El voseo (tenés/sabés/prendé) y el sheísmo empujan a langid a hermanas romances con prob >0.9. **Refina el hallazgo:** no es solo "idioma vs energía" — el LID estadístico colapsa el dialecto LIMPIO a pt/gl/ca/fr/it/wa. Esto es un modo de falla DOCUMENTADO de todos los LID (FastSpell arXiv 2404.08345: *"Spanish and Galician often get mixed up"*). py3langid además es de los **peores en texto corto <70 chars** (~0.567 accuracy vs 0.912 en largo) — exactamente el régimen ambient.

   ⚠️ **Corrección a un dato previo:** el ejemplo `'no se viste'→sk` que circulaba en notas es FALSO — py3langid lo da `es@0.521`. El `sk` solo aparece con `'no sé viste'` (con tilde), que además dispara `has_spanish_markers=True` y sería rescatado. (verdict: incierto, ver §8).

3. **El léxico controlado del command path** (`CARDINALS`, `COLOR_*`, `ALIAS_INDEX`) cubre solo domótica; cualquier cosa con sabor local cae al LLM. No hay alias para variantes uruguayas de dispositivos.

---

## 5. El módulo `language_quality.py` sin cablear + el gate desplegado: diagnóstico y plan

### Diagnóstico

**`src/ambient/language_quality.py` (CONFIRMADO dead code):** untracked en git (`?? `), `git log --all` vacío, `grep` de sus 4 símbolos públicos (`has_spanish_markers/is_spanish_keepable/make_quality_fn/SPANISH_STOPWORDS`) en `src/` devuelve solo el propio archivo. No lo importa `transcriber.py` (que sí importa explícitamente cada submódulo en `transcriber.py:236-243`), ni `store.py`, ni `main.py`. Único consumidor: su test (14/14 PASS). `src/ambient/__init__.py` vacío.

**Es el lever correcto, bien diseñado:**
- `has_spanish_markers` (líneas 35-44): acento/ñ/¿¡ → True directo; si no, ≥2 stopwords españolas distintas (excluyendo ambiguas es/en como 'no','a','y','son'). **Verificado: rescata 6/6 casos rioplatenses que py3langid mandó a wa/fr/ca/it, y rechaza 3/3 ingleses reales de TV.** Es exactamente la arquitectura validada por FastSpell (LID + segunda opinión léxica) y la recomendada por NVIDIA (langid sobre el texto).
- Política **flag-no-drop** (preserva la señal far-field para re-entrenamiento, como WakeClipWriter) — la correcta.

**Pero tiene problemas que hay que resolver ANTES de cablear:**

1. **TDD a medias:** los 3 tests nuevos de `test_store.py` (`test_lang_fields_roundtrip`, `test_undistilled_live_spanish_only`, `test_init_migrates_old_schema_adding_lang_columns`) FALLAN (`TypeError: AmbientUtterance got unexpected kwarg 'lang'`, `KeyError: 'lang'`) — describen columnas `lang/lang_prob/lang_ok` + migración + filtro `spanish_only` que el código de producción NO implementa.

2. **Política CONFLICTIVA con el gate desplegado, no complementaria:**
   - Gate distiller: dropea SOLO `en@prob>=0.9` (conserva fr/pt/eo/garble-no-en).
   - Módulo: keep solo si markers OR `lang=='es'` — dropea todo lo demás.
   - Sobre las mismas frases dan resultados OPUESTOS: `'apaga la tele por favor'`→**eo@0.56** → el gate distiller NO lo dropea (sobrevive), pero el módulo lo descarta (`keepable=False`). Si se cablea sin reconciliar, doble filtro inconsistente.

3. **`min_len=8` + fallback `lang=='es'` castiga comandos cortos sin tilde** — el caso típico far-field. `'dale, prende'`→fr@0.20, markers False, `keepable=False` (DESCARTADO). La literatura pide ~20 chars mínimo para confiar en LID puro; el módulo confía a 8.

4. **`min_vad=0.40`** (módulo) vs **`0.45`** (prod) — dos verdades sobre el mismo lever.

5. **`lang_prob` se recibe pero NO se usa** (línea 65, "reservado") — el gate desplegado SÍ lo usa (`drop_language_min_prob=0.9`). Asimetría de criterio.

6. **Falso-positivo con TV en español:** una utterance de TV en español con tildes pasa `markers=True` y sería conservable, pero es bleed que no debería ir a memoria. El módulo resuelve idioma, NO fuente — y la separación de fuente (SourceClassifier) está rota (1833/1833 unknown).

### Plan de integración (orden correcto)

**Paso 1 — FLAG primero, nunca drop.** Degradar el gate `drop_language='en'` del distiller a **flag** mientras corra py3langid crudo (hoy podría estar dropeando rioplatense que él mismo manda a en@0.9 sobre garble). Persistir el motivo.

**Paso 2 — Implementar el schema objetivo** (cerrar el TDD rojo): columnas `lang/lang_prob/lang_ok` en `store.py` + migración por `ALTER TABLE` (el patrón ya existe para `vad_prob`, `store.py:71-77`) + campos en `AmbientUtterance` + param `spanish_only` en `undistilled_live`. Hacer pasar los 3 tests.

**Paso 3 — Cablear `make_quality_fn` en `transcriber._handle_segment` ANTES de `store.add`**, envolviendo `make_langid_fn()` con `make_quality_fn()`. Decisión a tomar: meter py3langid en el hot-path del worker ambient (es CPU, determinista, liviano — aceptable). Persistir `(lang, lang_prob, lang_ok)` como **flag**, sin dropear.

**Paso 4 — Reconciliar umbrales y reglas:** unificar `min_vad` a 0.45 (o re-validar sobre `ambient.db` real); subir `min_len` a ~20 para la rama `lang=='es'` puro (mantener 8 solo cuando hay markers); usar `lang_prob`; agregar a `SPANISH_STOPWORDS` los imperativos voseo que faltan (`prendé/apagá/poné/pasame/ponelo/dale/onda`).

**Paso 5 — Subir retención 12h→48h** (la propuesta de la memoria) para que la evidencia far-field sobreviva al audit.

---

## 6. Medición de calidad sin ground truth: el conflicto "47% inglés vs 100%-TV" y el audit ciego

### El conflicto

- **Usuario (06-08):** "inglés=100% TV".
- **Shadow logs:** `~47% del crudo es inglés` / `~30% del post-vad sigue siendo inglés` (`distiller.py:284`, `settings.yaml:1512`).

**Sin resolver.** La investigación de langid (§4) aporta la hipótesis más fuerte para reconciliarlo: **una fracción grande de ese "inglés/no-es" es rioplatense bien transcripto que py3langid manda a otra etiqueta romance** (el % de "inglés" está inflado por el LID, no necesariamente por garble far-field). Si eso es cierto, el lever es LANGID+markers, no `drop_language='en'`. Si el 30% es TV-inglés-real, el gate actual basta. **No se puede decidir sin escuchar audio.**

### Por qué hoy NO se puede medir
- Con Parakeet, `confidence`/`no_speech_prob` son NULL — solo `vad_prob` (energía, no idioma).
- No hay columna `lang` persistida — `_log_language_shadow` es efímero a stdout. No se puede `SELECT lang, count(*) GROUP BY lang`.
- No hay columna de etiqueta humana (`audit_label`).
- Retención 12h: si se audita semanalmente, el 85% de la evidencia ya se purgó.
- `ambient.db` local = 0 bytes; toda la evidencia vive en el server y se autodestruye.

### Procedimiento de audit ciego concreto (priorizado)

1. **Pre-requisito:** subir `retention_hours` 12→48 (`settings.yaml:1464`) y agregar una **tabla de muestreo persistente fuera del TTL** (que copie N utterances/día con su audio-ref si se puede, o al menos el texto + `vad_prob` + `lang` re-derivado).

2. **Estrechar el subset SIN escuchar primero** (barato, cierra parte del conflicto): re-correr sobre el histórico de `ambient.db` el LID + `has_spanish_markers` del módulo. Medir, dentro del 30-47% marcado "en/no-es": ¿qué fracción tiene marcadores españoles (acento/ñ/¿¡/≥2 stopwords)? Eso es rioplatense mal clasificado. El resto es candidato a TV-inglés-real o garble sin markers.

3. **Audit ciego manual N≥50** sobre las utterances marcadas `lang!='es'` (idealmente con su audio): el revisor etiqueta cada una en {**TV-inglés-real**, **español-mal-transcrito (garble)**, **code-switch es+en**, **rioplatense-limpio-mal-clasificado**}, **sin ver la etiqueta de langid** (ciego). Guardar en la columna `audit_label`.

4. **Veredicto operativo:**
   - Si predomina TV-inglés-real → `drop_language='en'` es seguro, mantener.
   - Si hay rioplatense-limpio o garble-de-voz-del-hogar → degradar a flag-no-drop y mover el discriminante a markers+`lang=='es'` con longitud mínima.

5. **NO tocar el umbral de `vad_prob` como si separase idioma** — confirmado que es ortogonal al LID.

---

## 7. Recomendaciones priorizadas

| # | Acción | Por qué | Esfuerzo | Impacto | Riesgo |
|---|---|---|---|---|---|
| 1 | **Cablear `language_quality.py` en `transcriber._handle_segment` con FLAG-NO-DROP** + implementar columnas `lang/lang_prob/lang_ok` + migración (cerrar TDD rojo) | Es la palanca de software de máximo apalancamiento, hoy inerte; permite medir y conservar el rioplatense. Coincide con la recomendación de NVIDIA (langid sobre texto) | Medio | **Alto** | Bajo (flag no borra nada) |
| 2 | **Degradar el gate `drop_language='en'` enforce a FLAG** hasta resolver el audit ciego | Hoy en Fase 3 (enforce) podría estar borrando conversación real del hogar sin rastro; py3langid es no fiable en texto corto rioplatense | Bajo | **Alto** | Bajo (revierte pérdida de datos) |
| 3 | **Subir `retention_hours` 12→48 + tabla de muestreo persistente** | La evidencia far-field se borra antes de auditarla o juntar dataset; es la pérdida de evidencia más cara | Bajo | Medio | Bajo |
| 4 | **Audit ciego manual N≥50** sobre utterances `lang!='es'`, con columna `audit_label` | Resuelve el conflicto "47% vs 100%-TV"; decide si el gate de idioma es correcto o un bug de pérdida de datos | Medio | **Alto** | Bajo |
| 5 | **A/B acústico: bajar/congelar AGC (`PP_AGCONOFF:0` + gain fijo, "Vía B" de `settings.yaml:1082`) y/o probar ch0 vs ch1** en el tap del ambient, medido contra ground truth | La literatura (40/40 configs, +46.6% WER) muestra que el DSP agresivo daña el ASR neuronal; es el lever de SEÑAL más barato, antes de flashear | Medio | **Alto** (potencial) | Medio (toca el único mic de prod — gated, hacer en bench) |
| 6 | **Neutralizar/comentar el `language='es'` de `parakeet_stt.py` como NO-OP confirmado** + corregir el comentario engañoso (líneas 45-47) | Elimina una creencia falsa cara; no cambia comportamiento | Bajo | Bajo | Nulo |
| 7 | **Enriquecer `initial_prompt` del command path con voseo** (`vos/tenés/prendé/apagá/dale`), A/B-testeado | Prior de lenguaje barato para el dialecto; el único lever léxico del command path | Bajo | Medio | Medio (Whisper podría regurgitar sobre TV → acción fantasma, como en 2026-05-29 — por eso A/B, no asumir) |
| 8 | **A/B gated por números: Canary-1b-v2/flash** (SÍ respeta `language='es'`, inyecta `<\|es\|>` en el decoder AED), midiendo (a) WER rioplatense far-field, (b) tasa de alucinación sobre silencio | Única vía de forzado de idioma a nivel modelo. PERO es GPU-only (rompe el diseño 0-VRAM) y AED (puede reintroducir la alucinación que Parakeet eliminó) | Alto | Medio | Alto (VRAM + alucinación) |
| 9 | **Fine-tune/LoRA con audio rioplatense PROPIO** (cerrar el seam del LoRA Whisper nocturno "que nadie consume"), eval set auto-etiquetado con `has_spanish_markers` | No existe corpus es-UY/es-AR público suficiente; el `ambient.db`+WakeClipWriter ya juntan corpus | Alto | Alto (largo plazo) | Medio |
| 10 | **NO migrar el ambient a Canary; NO subir el umbral de `vad_prob` esperando que separe idioma; NO adoptar FastSpell/GlotLID** | Canary GPU-only+AED; vad es ortogonal a LID; FastSpell ~50-100x más lento que langid (innecesario para es/en); GlotLID entrenado en oraciones largas | — | — | — |

**Orden de máximo apalancamiento sobre la calidad del rioplatense:** primero #2+#1+#3 (software, flag-no-drop, conservar evidencia) → #4 (audit, cerrar el conflicto) → #5 (señal, el lever físico más barato) → recién después #8/#9 (modelo/entreno, gated por números).

---

## 8. Afirmaciones REFUTADAS / INCIERTAS (no actuar sobre supuestos falsos)

**REFUTADAS:**

- **"El firmware 2ch contradice el docstring de `segmenter.py`."** REFUTADO: el docstring (`segmenter.py:1-6`) ya documenta EXPLÍCITAMENTE el fallback 2ch→col 0 ("Con firmware 2ch la columna no existe y se degrada a col 0"). El código coincide con su docstring; no hay contradicción oculta. (Los hechos de hardware — fw 2ch, VAD sobre ch0 post-NS — sí son ciertos.)

- **"`min_speech_ms=300` elimina confirmaciones rioplatenses cortas legítimas."** REFUTADO: el umbral exige ≥4 chunks de voz (320ms); confirmaciones típicas (`dale`~400ms, `prendé`~450ms, `sí claro`~700ms) se conservan; solo `'sí'` aislado roza el borde. La afirmación confunde la tasa stream-wide (20.8% chunks>0.3, que incluye silencio) con la densidad intra-utterance (densa/contigua). `_speech_ms` acumula a través de micro-pausas, lo que hace MÁS difícil descartar voz real. El diseño existe para filtrar TV espuria, no confirmaciones.

- **"`'no se viste'→sk` sobrevive el drop como no-es."** REFUTADO: py3langid lo da `es@0.521`; 'sk' solo aparece con `'no sé viste'` (con tilde), que además sería rescatado por markers. Error factual material en el dato medido.

**INCIERTAS (no probadas):**

- **"El garble far-field es 'problema de señal, ningún modelo lo salva' / el modelo no es el cuello de botella."** INCIERTO: el dato empírico acotado existe en el doc (`2026-06-07...:66`), pero la conclusión causal fuerte se generalizó desde N=2 modelos sobre N=17 clips, fila far-field **cualitativa sin WER**. Contradice (a) la hipótesis activa del propio proyecto de que Canary lo arreglaría, (b) literatura de ganancias far-field por el lado acústico/modelo (data-aug reverberante, Mega-ASR), (c) los propios caveats del doc ("español es el eslabón débil", "no sólidamente establecida"). Tratar la mejora de señal como **una** palanca, no como la única.

- **"Parakeet colapsa el rioplatense far-field a inglés-garble per-utterance."** INCIERTO: mecanismo confirmado por literatura + premisa no-op confirmada, pero NO probado sobre audio etiquetado (ambient.db local vacía, sin audit ciego, A/B midió no-voz). Que py3langid diga "en" no prueba que Parakeet produjera inglés desde voz española — el LID también misclasifica.

- **"`vad_prob` NO puede separar español-hogar de bleed-TV; el discriminante es langid (único)."** INCIERTO/mixto: la parte "el VAD-gating como bala de plata está refutado" es literal en el doc (`2026-06-07...:34`). PERO el absolutismo "NO PUEDE" se contradice con `store.py:147-149` (vad_prob ES la compuerta validada, cruce hogar/bleed-TV ≈0.45) y la separación es multi-señal (vad + DoA + langid), no langid-only. langid es el lever solo para el sub-caso español-sobre-español.

- **"py3langid manda rioplatense con marcadores a fr/pt/sk y sobreviven el drop como no-es."** INCIERTO: 2 de 3 ejemplos reproducen (fr@1.000, pt@0.787), pero el encuadre "no-es que sobrevive" no se sostiene: en el gate de calidad real (`is_spanish_keepable`+markers) los tres dan `keepable=True` — el módulo SÍ los rescata (es su propósito). El daño de langid hoy es **potencial, no actual** (porque el módulo no está cableado y el gate solo tira 'en').

---

**Archivos clave citados (rutas absolutas):**
- `/Users/yo/Documents/kza/src/ambient/parakeet_stt.py` (línea 89: `language='es'` NO-OP; 45-47: comentario engañoso a corregir)
- `/Users/yo/Documents/kza/src/ambient/language_quality.py` (dead code, flag-no-drop; líneas 35-44 markers, 47-76 keepable)
- `/Users/yo/Documents/kza/src/ambient/segmenter.py` (50,133,168 vad_prob mean-of-maxes con cola; 92-104 fallback 2ch→col 0; 37 ventana incompleta)
- `/Users/yo/Documents/kza/src/ambient/distiller.py` (254-277 `_apply_language_gate`; 63-88 `make_langid_fn`; 279-291 shadow)
- `/Users/yo/Documents/kza/src/ambient/store.py` (sin columnas lang; 136-165 `undistilled_live`; 71-77 patrón de migración)
- `/Users/yo/Documents/kza/src/ambient/doa.py` (62-65 None con 2ch), `source_classifier.py` (1833/1833 unknown)
- `/Users/yo/Documents/kza/src/ambient/transcriber.py` (108-148 `_handle_segment`; 236-243 imports; no cablea language_quality)
- `/Users/yo/Documents/kza/src/pipeline/multi_room_audio_loop.py` (565-578 tap fail-open)
- `/Users/yo/Documents/kza/src/stt/whisper_fast.py` (169 `language='es'` REAL command path)
- `/Users/yo/Documents/kza/src/nlu/regex/vocab.py:32` (IMPERATIVES_VOSEO), `normalize.py:5`, `llm_gate.py:52`
- `/Users/yo/Documents/kza/config/settings.yaml` (1082/1090-1093 AGC; 1460/1464/1506/1514 ambient; 168-171/259-275 command)
- `/Users/yo/Documents/kza/tests/unit/ambient/test_store.py` (3 tests rojos del diseño objetivo)
- `/Users/yo/Documents/kza/docs/research/2026-06-07_SOTA_ASR_ESPANOL_INVESTIGACION.md` (34 VAD refutado; 39 español eslabón débil; 66/71 far-field garble)
- `/Users/yo/Documents/kza/docs/runbooks/2026-06-06-xvf3800-flasheo-6ch.md:7` (fw 2ch confirmado, flasheo GATED)

Fuentes web load-bearing: onnx-asr `adapters.py`/`nemo.py` (github.com/istupakov/onnx-asr); NeMo issues #14799 y #15097 (maintainer NVIDIA: Parakeet-TDT no recibe/emite language id); FluidAudio #303; FastSpell arXiv 2404.08345 (confusión es↔gl/ca/pt); arxiv 2512.17562 (denoise daña ASR); model card nvidia/parakeet-tdt-0.6b-v3 y nvidia/canary-1b-flash.
