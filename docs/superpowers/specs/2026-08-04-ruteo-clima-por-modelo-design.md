# Ruteo de la ambiguedad clima/AC por inferencia — diseño

Fecha: 2026-08-04
Rama: `feat/mundo-real-fase1`
Estado: aprobado, pendiente de plan de implementacion

## El problema

En rioplatense "clima" significa dos cosas: el tiempo y el aire acondicionado.

    "prendé el clima"       -> encendé el aire      (comando)
    "está el clima lindo"   -> qué tiempo hace      (pregunta)

Se intentaron tres rondas de reglas sobre substrings, cada una revisada
adversarialmente ejecutando frases contra el dispatcher. Cada ronda cerró una
clase de error y abrió la de al lado; la ronda 3 se revirtió (`ef651e2`).

Diagnostico raiz del ultimo review, que este diseño acepta: `WEATHER_KEYWORDS`
es un scan de substrings que corre antes que domotica, y `DOMOTICS_KEYWORDS` no
distingue imperativo de infinitivo (a proposito: Whisper produce los dos).
Cada regla nueva es un proxy de "esto es una orden o una observacion", y ese
proxy es leaky en español hablado sin puntuacion confiable.

## Estado medido antes de decidir

Todas las mediciones son del 2026-08-04 contra el server de produccion.

### Los tres enfoques sobre las mismas 22 frases de las tres rondas

| enfoque | accuracy | latencia | falla en |
|---|---|---|---|
| Reglas de hoy (`ef651e2`) | 17/22 = 77.3% | ~5 us | comandos con relleno o clausula de justificacion |
| Motor de gramatica (`src/nlu/command_grammar.py`, mejor caso) | 17/22 = 77.3% | ~us | negacion y alcance interrogativo |
| 7B `:8101` con prompt dedicado | 21/22 = 95.5% | +71ms p50 / 118ms p95 | 1 caso genuinamente ambiguo |

Los dos enfoques por reglas empatan fallando en **casos distintos**. La gramatica
resuelve los cinco comandos con relleno que las reglas pierden, pero da `turn_on`
con confianza alta para `"no hace falta prender nada"`: un regex de verbo no ve
negacion. Ese es el techo del metodo, no un defecto de implementacion.

Las 5 fallas de las reglas estan **vivas en la rama hoy**: `"prendé ya el clima,
hace calor"` rutea `fast_weather`, o sea que pedis el aire y te contesta el
pronostico.

### Latencia del clasificador

`:8101` (llama-server, Qwen2.5-7B-Instruct Q4_K_M, `-ngl 99`):

- warm: p50 110ms con el prompt generico actual, **71ms** con el prompt dedicado (mas corto)
- cold-ish (prompt ajeno intercalado que desaloja el prefix): p50 124ms
- p95 del prompt dedicado: 118ms
- determinista: 3 corridas a `temperature=0.0` dieron output identico

El comentario de `config/settings.yaml:299` que dice "latencia warm medida:
245-272ms" quedo viejo: el drop-in de systemd ya corre `-ngl 99` (todas las capas
en GPU) y el ExecStart original con `-ngl 20` fue reemplazado.

### Ley de presupuesto de prompt

Costo de prefill medido en `:8101`:

| prompt | latencia |
|---:|---:|
| 63 tok | 89 ms |
| 1.066 tok | 334 ms |
| 4.262 tok | 1.049 ms |
| 14.887 tok | 4.141 ms |

Aproximadamente 0.28ms por token. **El fast path aguanta ~250 tokens de prompt
en total.** Con 1.000 tokens ya se consume el presupuesto entero de 300ms sin
haber generado nada. Esta ley descarta por aritmetica cualquier diseño que
infle el prompt.

### Alternativas de endpoint descartadas

- **`:8180`**: no existe. Nada escucha ahi.
- **`:8200`**: hoy es el gateway LiteLLM hacia **MiniMax cloud**; el GLM-Air local
  quedo comentado como rollback de emergencia (`settings.yaml:397-402`). Medicion
  propia del 2026-06-06: TTFT 0.87-1.56s. Es 3-5x el presupuesto completo.
- **Hermes como modelo**: no existe un MoE chico (solo hay un pedido abierto en
  HuggingFace, sin release); la linea fue al 36B denso. El 14B denso Q4_K_M son
  ~8.5GB y no entra: cuda:0 tiene ~6.3GB libres y cuda:1 ~0.7GB.
- **hermes-agent como harness**: 13.9K tokens de overhead fijo por call medidos en
  el analisis de junio = **4.1s de prefill** en este endpoint; ademas su inferencia
  va por Nous Portal (cloud). Descartado para el fast path, no para KZA en general
  (el plan de junio ya tiene su propia ruta por sandbox).

### Frecuencia real del gate

Corpus de `events.db`: 1.116 frases con texto. **151 (13.5%)** contienen un
sustantivo climatico, dominadas por una frase repetida 113 veces (`"Nexa bajá la
temperatura del aire."`). No hay ninguna pregunta de clima en el historico: el
path `FAST_WEATHER` se agrego el 2026-08-04.

El input real viene destrozado por el STT y el diseño tiene que asumirlo:

    Nexa bajá la luz del aire.
    Nexa, prendela luces, aire acondicionado.
    Nexa bajá la temperatura de la luz del escritorio,

## Decision

Enfoque hibrido: **gate de vocabulario + clasificador binario**. El modelo
interviene solo cuando el texto contiene vocabulario contestado; el resto del
ruteo sigue por reglas, sin latencia agregada.

Alternativas consideradas y por que no:

- **Todo el ruteo por inferencia**: paga +71ms p50 / 118ms p95 en cada comando
  sobre un fast path que hoy esta en 150-280ms con presupuesto de 300ms. La
  aritmetica no cierra.
- **Gate por confianza de la gramatica** (usar `parse_command()` y llamar al
  modelo solo si `confidence < 0.75`): filtraria justo los casos que el modelo
  tiene que arreglar, porque la gramatica esta *confiadamente equivocada* en la
  negacion.
- **Sin modelo** (reordenar + extender la gramatica): medido, llega a 77.3% y
  falla en las tres preguntas hibridas. Es la cuarta ronda que el ledger predijo.

### La propiedad que sostiene el diseño

El **vocabulario contestado** es exactamente `_CLIMATE_DOMAIN_NOUNS`, que ya
existe en `dispatcher.py:153` y se deriva de `_NON_LIGHT_DOMAIN_NOUNS`:
`clima`, `temperatura`, `termostato`, `calefaccion`, `aire`, `grados`. El gate
es un test de presencia de esas palabras (accent-insensitive, con limite de
palabra, igual que `_conflicting_domain`). No se agrega vocabulario nuevo.

El gate pregunta algo trivial — *el texto contiene alguna de esas palabras?* — y
no pregunta nada sobre intencion. Por eso **cuando el gate se equivoca, el costo
es latencia, no ruteo incorrecto**: puede disparar de mas y se pagan 71ms, pero
no puede mandar un comando al path equivocado.

Los guards de las rondas 1-3 eran proxies de *intencion*; cuando fallaban,
misrouteaban. Esa es la diferencia estructural, no una mejora de grado.

## Arquitectura

```
_classify_request(text) -> (PathType, Priority)      [sin cambios: sync, puro, determinista]
        |
        +-- devolvio FAST_WEATHER o FAST_DOMOTICS
            y el texto tiene vocabulario contestado?  -> contested = True
        |
dispatch()  [ya es async]
        |
        +-- si contested: asyncio.gather(clasificador_7B, vector_search)
            el clasificador decide el path final
            si falla, devuelve None o vence el timeout -> vale lo que dijo _classify_request
```

Tres invariantes:

1. **`_classify_request` sigue puro y determinista.** Los ~300 tests que la
   llaman siguen valiendo. Su resultado pasa a ser la opinion por defecto, no la
   final.
2. **El modelo solo puede corregir en un eje**: `FAST_DOMOTICS <-> FAST_WEATHER`.
   No puede mandar nada a musica, ni al slow path, ni inventar un path. Cualquier
   salida fuera de las dos etiquetas se descarta.
3. **El guard se borra, no se apila.** Se eliminan
   `_DOMOTICS_CLIMATE_ADJACENCY_RE`, el bail-out por `?`/`¿`, y las entradas
   `"hace calor"` / `"hace frío"` / `"hace frio"` de `WEATHER_KEYWORDS`. El
   clasificador ocupa ese lugar.

Limitacion aceptada a proposito: el gate corre **despues** de que las reglas
eligieron, asi que si las reglas mandaran una frase climatica a `SLOW_LLM` el
clasificador no se enteraria. Ninguna de las 22 frases lo hace. Se acota asi
para que el mecanismo solo pueda tocar dos paths.

### Costo real

**Version que se implementa (secuencial): +71ms p50 sobre el 13.5% del trafico =
~9.6ms de promedio.** El subconjunto contestado pasa de 150-280ms a 221-351ms, o
sea que puede rozar el techo de 300ms en su cola.

Durante el diseño se proyecto solaparlo con el vector search — que ya cuesta
~48ms en CPU en ese mismo path (`settings.yaml:286`) — para un costo neto de
`max(71, 48) - 48 = ~23ms`. Al escribir el plan se verifico que **no es posible
sin reestructurar `_fast_path()`**: el vector search vive dentro de ese metodo
(`dispatcher.py:611`), o sea despues de la bifurcacion que el arbitraje tiene que
decidir. Solaparlos mezclaria dos cambios sin relacion.

Queda como follow-up medido: si el p95 del path contestado rompe los 300ms en
produccion, reestructurar `_fast_path` para solapar. La correccion del ruteo no
depende de eso.

## Componentes

### `src/nlu/climate_intent.py` (nuevo)

Modulo chico, un solo proposito: dado un texto, devolver `ACCION | CONSULTA |
None`. Recibe el router por constructor (DI, patron dominante del proyecto).
`FastRouter` queda generico: la logica de dominio no se le mete adentro.

**Las etiquetas no pueden llamarse "clima".** Es un hallazgo medido, no una
preferencia: con el prompt generico actual y la opcion `clima` en la lista, el 7B
devolvio `clima` para `"prendé el clima"` — hereda la misma ambiguedad de la
palabra. Con `ACCION_AIRE` / `PREGUNTA_TIEMPO` acierta 21/22.

Prompt (~150 tokens, dentro del presupuesto de 250):

```
Sos el router de un asistente de hogar. Decidí qué hace el usuario.

ACCION_AIRE     = ordena encender, apagar o ajustar el aire / termostato / calefacción.
PREGUNTA_TIEMPO = pregunta o comenta cómo está el tiempo afuera.

En rioplatense "clima" significa las dos cosas: el aparato y el tiempo.
Decidí por lo que el usuario QUIERE que pase, no por la palabra.

Texto: apagá el aire                       -> ACCION_AIRE
Texto: ¿va a llover mañana?                -> PREGUNTA_TIEMPO
Texto: está lindo el día, no prendas nada  -> PREGUNTA_TIEMPO
Texto: poné el aire en 22 que hace calor   -> ACCION_AIRE
```

Parametros: `temperature=0.0`, `max_tokens=10`, `stop=["\n", "Texto:", "Etiqueta:"]`.

Los 4 ejemplos few-shot son parte del **contrato**, no decoracion: cambiarlos
cambia la accuracy. Van versionados en el modulo con un comentario que obliga a
re-correr el eval si se tocan (ver tripwire mas abajo).

**Parseo estricto**: `startswith` sobre las dos etiquetas exactas. Cualquier otra
cosa — texto vacio, `OTRO`, una alucinacion, la salida contaminada tipo
`"clima\nRespuesta: Lo s..."` que se observo con el prompt generico — devuelve
`None`, y `None` significa que decidan las reglas. Sin matching difuso.

### `src/llm/reasoner.py` (cambio minimo)

`FastRouter.complete()` es async pero **no soporta `stop`**, y su firma termina en
`**_ignored`: si se le pasa `stop=[...]` lo descarta en silencio. Se agrega
`stop: list[str] | None = None` y se reenvia. Aditivo, sin cambio de
comportamiento para quien no lo use.

Sin esto el clasificador pierde los stop tokens sin aviso, y el benchmark de mayo
(`benchmarks/router/REPORT.md`) midio que los stops valen **+18 puntos** de
accuracy.

### `src/orchestrator/dispatcher.py`

- Se borra el guard de clima completo (regex de adyacencia + bail-out por
  interrogacion) y las 3 keywords de la ronda 2.
- `_classify_request` queda sin tocar en firma ni semantica.
- `dispatch()` incorpora la rama `contested` con `asyncio.gather` y el timeout.

## Degradado

Timeout de **150ms**. Si vence, si `:8101` no responde, o si el parseo da `None`,
vale el resultado que las reglas ya calcularon (se computan igual, cuestan ~5us).

El timeout esta puesto justo arriba del p95 medido (118ms) a proposito: corta la
cola sin sacrificar el caso tipico. Por eso el go/no-go del eval exige
`p95 <= 150ms` — si el p95 se acercara al timeout, el degradado dejaria de ser
excepcional y el clasificador estaria aportando latencia sin aportar decisiones.
Las dos cifras se mueven juntas: cambiar una obliga a revisar la otra.

**El peor caso del hibrido es exactamente el comportamiento de hoy, nunca peor.**

Se loguea el degradado con un contador para que no sea un fallo silencioso.

No hay segundo LLM en la cadena para este caso: `:8200` es cloud (~1s TTFT) y no
entra en el presupuesto. La cadena de failover que existe en
`config/settings.yaml:404-433` es un fallback de *razonamiento*, no de ruteo.

## Eval

El 95.5% medido durante el diseño **esta inflado**: el prompt se escribio mirando
esas 22 frases. Es tuning y evaluacion sobre el mismo set. Ademas esas frases
estan limpias, y el input real llega destrozado por el STT.

Dos sets separados:

| set | origen | para que |
|---|---|---|
| A — desarrollo | las 22 de las tres rondas | iterar el prompt; se puede mirar libremente |
| B — held-out | escrito **antes** de tocar el clasificador y no vuelto a mirar, mas las ~15 frases reales distintas del corpus con su ruido de STT | el numero que se reporta |

Vive en `benchmarks/router/` reusando la forma del bench de mayo
(`golden_set.yaml` + `runner.py` + `analyze.py`): se agrega `climate_set.yaml`.

Los dos errores no valen lo mismo y se miden por separado:

- **Consulta -> accion** (`"está el clima lindo"` prende el aire): accion fisica
  no pedida. Es el error caro.
- **Accion -> consulta** (`"prendé el clima"` da el pronostico): molesto, no hace
  nada, se repite. Es el barato, y es el que esta vivo hoy 5 veces.

Criterio de go/no-go sobre el set B:

- accuracy global >= **90%** (contra 77.3% de las reglas, que es la vara)
- **cero** consulta -> accion
- p95 <= **150ms**

Si B no llega, la conclusion es no adoptar, y se escribe con los numeros.

## Tests

### Impacto sobre los 2832 existentes

Verificado por ejecucion, simulando el borrado del guard contra los 9 casos de
`test_domotics_climate_adjacency_guard_finding_3`: **8 de 9 no se mueven.** El
guard aportaba un solo caso; las frases de dos palabras de `WEATHER_KEYWORDS` ya
cubrian el resto.

- **0 tests borrados**
- **1 caso cambia de capa**: `"¿tengo que prender el clima o hace calor afuera?"`
  deja de decidirse en `_classify_request` y pasa a testearse contra `dispatch()`
  con el router mockeado, que es donde ahora vive esa decision. No se relaja: se
  mueve al nivel donde el comportamiento existe.

### Tres capas

1. **Unit con router mockeado** (pytest, sin red). Aca vive casi todo el codigo
   nuevo y es 100% determinista, o sea `assertEqual` comun: que el gate dispare
   con el vocabulario esperado; que el parseo devuelva `None` ante `""`, ante
   salida contaminada y ante `OTRO`; que `None` haga valer la regla; que el
   clasificador no pueda devolver un path fuera de los dos; que a los 150ms se
   corte y caiga a reglas.
2. **Eval scoreado** contra `:8101` vivo. Umbral, no igualdad. Fuera de pytest, a
   demanda y antes de deploy.
3. **Tripwire**: un test que fija el prompt por hash. Si alguien lo edita, falla
   con el mensaje de re-correr el eval. Sin esto el prompt se degrada en silencio
   y ningun test se entera.

`temperature=0.0` es lo que sostiene el esquema: medido, 3 corridas con output
identico. El eval no es flaky, asi que si su numero se mueve es porque se movio
algo real (prompt, modelo o server).

### Lo que sigue sin testearse

La verificacion end-to-end por voz. El paso 8 de la Task 3 nunca se hizo. Nada de
esto prueba que `"prendé el clima"` prenda el aire de verdad contra el HA real:
eso necesita una persona hablandole al dispositivo.

## Resultado del eval: no adoptar

Corrida el 2026-08-04 contra `:8101` en vivo (tunel SSH, bearer auth), set B
completo (50 casos: 20 corpus + 20 sinteticos CONSULTA + 10 borde ACCION).
`benchmarks/router/climate_eval.py` construido segun este diseño, sin tocar
`src/nlu/climate_intent.py` ni el set.

Un defecto del brief se corrigio antes de correr: el `--model` default
(`qwen2.5-7b-instruct`) no existe en el catalogo de este `:8101` -- el
`llama-server` de ik_llama.cpp expone como id el path completo del `.gguf`
cargado, y el cliente OpenAI exige match exacto. Se cambio el default a
`/home/kza/kza/models/Qwen2.5-7B-Instruct-Q4_K_M/Qwen2.5-7B-Instruct-Q4_K_M.gguf`,
confirmado contra `/v1/models`.

4 corridas completas (1 inicial + 3 de determinismo), **accuracy y errores de
clasificacion identicos en las 4** (garantia de `temperature=0.0`, tal como
predijo este diseño):

| check | valor | piso/techo | resultado |
|---|---|---|---|
| accuracy global | 47/50 = 94.0% | >= 90% | OK |
| consulta -> accion | 2 (`sint-011`, `sint-012`) | == 0 | **NO** |
| p95 | 118-122ms en 3/4 corridas, 179ms en la primera (cold) | <= 150ms | inconsistente, ver abajo |

**El gate es NO-GO por el segundo criterio, que falla de forma 100%
reproducible.** La accuracy global (94.0%) supera holgadamente el piso y a las
reglas de hoy (77.3%), pero el criterio de cero-tolerancia sobre el error caro
no se sostiene.

### Los dos casos que rompen el gate

Ambos son CONSULTA con el verbo de dominio negado explicitamente:

- `sint-011`: *"no hace falta prender nada, está fresco"* -> clasificado ACCION
- `sint-012`: *"no hay que tocar el aire, se está bien así"* -> clasificado ACCION

Este diseño ya habia medido el mismo techo para el motor de gramatica (linea
33 de la tabla de arriba: "falla en... negacion y alcance interrogativo") y lo
atribuyo a que un regex de verbo no ve negacion. El 7B con este prompt
hereda el mismo techo: el prompt de pocos ejemplos no ensaya explicitamente la
negacion de un verbo de accion (el unico ejemplo con negacion en el prompt,
*"está lindo el día, no prendas nada"*, tiene el verbo ya en imperativo negado
al lado del sujeto climatico, no un verbo generico como "prender"/"tocar" en
una clausula de necesidad negada como "no hace falta" / "no hay que").

Ademas, `corpus-011` (*"Nexa bajá la temperatura de la luz del escritorio,"*)
abstuvo en las 4 corridas -- no cuenta contra ningun check (el degradado cae a
reglas) pero es la unica abstencion consistente del set.

### Latencia: no concluyente por si sola, pero no cambia el veredicto

p95 vario entre corridas (179ms en la primera contra ~120ms en las 3
siguientes), consistente con jitter de trafico real en un endpoint de
produccion que sirve el pipeline de voz en simultaneo, no con no-determinismo
del clasificador (los labels no se movieron). No se investigo mas a fondo
porque el criterio de consulta->accion ya es un NO-GO deterministic e
independiente del ruido de latencia: aunque el p95 hubiera dado OK en las 4
corridas, el veredicto seguiria siendo NO-GO.

### Decision

**No adoptar** el clasificador por inferencia con el prompt actual. No se
intento ajustar el prompt para pasar el gate -- hacerlo seria tunear contra el
held-out (`benchmarks/router/climate_set.yaml`) y invalidaria la medicion. El
prompt, el modulo `src/nlu/climate_intent.py` y el set quedan tal como estan;
la Task 5 (integracion en `dispatcher.py`) no se ejecuta.

Lo que sigue vivo de este diseño, sin cambios: el diagnostico raiz (los guards
por regex son proxies de intencion), el hallazgo de latencia (71-124ms warm en
`:8101`, no 245-272ms), y el error barato que persiste hoy en la rama
(`"prendé ya el clima, hace calor"` -> `fast_weather`). Si se retoma esta
via en el futuro, el punto de partida es resolver la negacion explicita de
verbos de dominio antes de volver a correr `climate_eval.py` contra el mismo
set B -- sin tocarlo.

## Hallazgos laterales (fuera de alcance, anotados para no perderlos)

1. `config/settings.yaml:435-437` dice que si `:8101` cae el router rota al
   `:8200` (GLM-Air **local**). Dejo de ser cierto el 2026-05-29: GLM-Air se
   reemplazo por MiniMax cloud. El comentario describe un fallback local que ya
   no existe, y el real sale a internet con `timeout_s: 60.0`.
2. `config/settings.yaml:299` dice "latencia warm medida: 245-272ms"; medido hoy,
   71-124ms.
3. `FastRouter.classify()` es sincrona y bloqueante (`generate()` es un for de
   HTTP): llamarla desde el dispatcher bloquearia el event loop. Ademas devuelve
   el texto crudo sin parsear.
4. La colision substring `prende ⊂ prender` sigue viva en el loop general de
   domotica. Este diseño no la toca porque ningun caso del alcance la atraviesa.
