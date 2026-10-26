# H2: respuestas completas y captura disponible

Este hito implementa R04/R05 del [roadmap](2026-10-25-stabilization-roadmap.md).
H1 tiene contratos y pruebas conectadas; la selección real sigue deshabilitada
hasta verificar miembros y efectos. H2 se desarrolla bajo `yo`, sin reproducción
ni llamadas al reasoner de producción. La única release sigue siendo H6.

## Contrato y límites previos a la evaluación

El reasoner produce texto y `ResponseHandler` es el responsable de locutar cada
respuesta. Los resultados distinguen generación, entrega, fallo y cancelación;
un error del slow path también recibe una respuesta audible si la salida está
disponible. Cancelar no puede publicar texto o iniciar audio de una sesión vieja.
Los callbacks del dispositivo sólo consumen PCM y notifican espacio: nunca
esperan al productor ni hacen síntesis o red.

| Criterio | Prueba y límite de aceptación |
|---|---|
| Event loop | Productor simulado con espera de 200 ms; heartbeat de 5 ms, diez repeticiones, p95 de retraso ≤25 ms y máximo ≤75 ms. Son límites offline de regresión, no latencia de respuesta del proveedor. |
| Cancelación | Petición de cancelación atendida en ≤100 ms en los grafos simulados; reproducción detenida en ≤200 ms y limpieza de recursos cooperativos en ≤1 s. La medida acústica real se hace en H6. |
| PCM completo | Igualdad sample por sample y mismo orden al reproducir normalmente, con chunks de 24.000 samples a 24 kHz, chunks mayores que el buffer y tamaños variables de callback. Sin truncado ni duplicación; el padding final del dispositivo no cuenta como audio del productor. |
| Presión de retorno | Buffer PCM fijo y colas de texto acotadas; un consumidor lento frena al productor. Cancelar/fallar despierta a productores bloqueados. No crece una cola de respuestas o un conjunto de workers abandonados sin límite. |
| Finalización | El fin del productor no cierra el dispositivo antes de consumir el tail. Se espera la notificación de drenado; timeout o caída del dispositivo produce fallo, no éxito silencioso. |
| Voz única | Cada respuesta y error tiene un único responsable de locución. No se repite la respuesta completa después de haber emitido sus fragmentos. |
| Procedencia | La respuesta sale por la zona de la petición; una zona global mutable no redirige una respuesta tardía. Audio de una sesión cancelada no entra en la siguiente. |
| Contexto | Sólo una generación final válida se registra como respuesta completa. Una cancelación o un fallo no inventa un turno exitoso. |

Los tests usarán datos sintéticos, un dispositivo callback simulado y transportes
locales/en memoria. La aceptación acústica exige una respuesta larga no cacheada,
barge-in y captura real durante la validación de H6. No se heredan las cifras
históricas de latencia como resultado medido de este cambio.

## Implementación

1. Corregir `StreamingAudioPlayer`: escritura completa por tramos, presión de
   retorno cancelable, callbacks que consuman incluso restos parciales y un
   drenado observable antes de cerrar. Registrar el handle que usa cada salida.
2. Consumir generación sin bloquear el loop, con cierre de streams y recursos
   ante timeout, cancelación y fallo. Preservar destino/consentimiento de H1.
3. Conectar el slow path y el modo legacy con un responsable de respuestas,
   preservando streaming, zona de origen, orden y cancelación. Retirar la
   exclusión por tipo de path como sustituto de saber si ya se entregó voz.
4. Recorrer el grafo de `main` y los productores conectados, incluidos feedback
   y fallos. Serializar la síntesis cuando el motor no admite concurrencia y
   mantener la captura disponible durante generación/reproducción.
5. Ejecutar corpus conectado, suite completa, checks y smoke compilado. Registrar
   métricas y límites reales pendientes; publicar sólo con el resto de H6.

Al abrir el hito se observaron tres defectos por lectura: el worker
consume `generate_stream` síncrono dentro del loop, el router omite locución del
slow path y el player recorta lo que no cabe en el buffer. También falta exponer
los handles de reproducción por zona al controlador de cancelación.

El drenado se apoyará en `finished_callback` y `CallbackStop`; cancelar usa
aborto. La [API de sounddevice 0.5.5](https://python-sounddevice.readthedocs.io/en/0.5.5/api/streams.html)
y sus [excepciones de callback](https://python-sounddevice.readthedocs.io/en/0.5.5/api/misc.html)
distinguen el fin normal, que conserva buffers pendientes, del aborto, que los
descarta. La versión instalada de la venv consultada es 0.5.5; su comportamiento con el
driver se contrasta en H6; la simulación no prueba ese driver.

## Primera implementación: PCM y ciclo del dispositivo

`StreamingAudioPlayer` escribe por tramos y espera espacio mediante una
condición que la cancelación y el callback despiertan. El ring permanece fijo;
no se recorta el resto del chunk. El callback consume todo lo disponible aun
cuando su bloque es mayor que el ring, rellena el resto con silencio y registra
ese underrun de software. Se conserva el orden de los frames del productor;
ese padding por falta de datos no se presenta como audio generado.

El fin normal espera `finished_callback`, incluyendo el tail del dispositivo.
Un buffer sin progreso, un drenado agotado o la caída del driver generan fallo.
Detener aborta, despierta escritores y conserva el estado cancelado. Un productor
cancelado que entregue tarde no puede reiniciar ni contaminar la siguiente
sesión. El source se cierra incluso al fallar o cancelar. El timeout de drenado
predeterminado es al menos 2 s y supera la duración del ring en 1 s; ambos
plazos se pueden pasar por el contrato del player y sus adaptadores.

Los adaptadores de Piper y `ZoneManager` exponen el handle antes de arrancar,
aceptan un token de cancelación y conservan el resultado. La zona desconocida
se rechaza y un error de salida se propaga; no se convierte en éxito ni en una
reproducción por otra zona. Falta conectar ese contrato a todas las sesiones de
`ResponseHandler`, incluidos cache, síntesis completa y barge-in del grafo real.

Se agregó un driver sintético con callback, tail diferido y fallos observables.
Las pruebas recorren player y zonas reales, chunks de 24.000 frames, múltiples
canales, callbacks mayores que el ring, error y cancelación. Los tests de
barge-in ahora importan los módulos reales de Python 3.13: se retiran los stubs
globales de paquetes usados para un entorno histórico de Python 3.9, que rompían
la recolección aislada y podían reemplazar DTOs del sistema.

Verificación del commit `5143e82`: **4032 passed, 1 failed, 1 xfailed** en
111,67 s; el fallo de idioma/VAD es el mismo del registro anterior. Pasan 77
pruebas dirigidas, incluidas 24 nuevas del player y seis grafos de zona/player,
Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4, doce tests Node, el smoke del
dashboard y el artefacto compilado fuera del checkout.

En diez corridas sintéticas se conservaron los 24.000 frames de cada respuesta
con ring máximo de 3.600 frames. Cancelar audio activo tardó como máximo
3,13 ms y el productor terminó en 3,25 ms. Es una medida del driver de prueba,
no del altavoz ni del barge-in completo. El
[registro PCM](../research/2026-10-25-h2-pcm-validation.json) conserva alcance,
comandos, métricas y criterios de H2 todavía pendientes.

## Segunda implementación: generación asíncrona conectada

El worker de `MultiUserOrchestrator` consume `GenerationStream`. El reasoner
HTTP usa el SDK asíncrono instalado, con los mismos estilos chat/completions,
destino y modelo configurados. La construcción del transporte y el primer
acceso a los recursos lazy del SDK quedan fuera del loop. La primera suite
conectada en frío midió un bloqueo de 135 ms por esa inicialización diferida;
se corrigió sin ampliar el límite de 75 ms. Las pruebas conservan el arranque
en frío y una inicialización lazy simulada de 200 ms, incluido su fallo.
Cada generación posee su cliente y stream; cancelar, agotar el watchdog,
fallar o cerrar el consumidor libera ambos. El consumidor cierra el generador
con `aclosing`, también al salir antes del EOF.

El camino nuevo comprueba consentimiento antes de construir y antes de enviar
el prompt, rechaza cambios de configuración durante discovery y no sustituye
un modelo explícito por otro. No sigue redirecciones ni reintenta una respuesta
parcial. Requiere fin `stop`: truncado, EOF prematuro y respuesta vacía fallan;
el tail retenido no se publica como respuesta completa después de un fallo.
El texto crudo se limita a 65.536 caracteres, incluido el razonamiento oculto.

Los motores síncronos quedan aislados en un productor con cola de dos chunks.
La cancelación despierta al productor cuando espera espacio y descarta su
resultado tardío. Si está dentro de una llamada bloqueante, conserva la
ocupación hasta que termine: nuevas generaciones de ese propietario fallan
como ocupado. No se crean workers sucesivos abandonados ni se afirma que
cancelar asyncio mate una llamada nativa o un subprocess síncrono. Su cierre
real y recuperación siguen siendo una limitación, a resolver según el backend
durante H4; no sustituyen el cierre cooperativo del HTTP nativo.

El grafo conectado descubrió que `RequestDispatcher` reemplazaba la cola
inyectada cuando estaba vacía, por su valor falso. Ahora conserva exactamente
la cola que atiende el worker. Las nuevas consultas del mismo usuario cancelan
también la petición en proceso; una prioridad mayor puede interrumpir durante
una espera de red. Timeout, cancelación del llamador y apagado despiertan al
waiter, cierran la generación y conservan su razón. Una petición terminal no
puede revivir por una respuesta tardía ni notificar su resultado dos veces.
Sólo una generación completa válida incorpora el par de turnos al contexto.

Las pruebas usan el SDK 2.24.0 y HTTPX instalados, con HTTP/SSE en memoria.
Se contrastó el ciclo de vida con la
[referencia de Python](https://developers.openai.com/api/reference/python) y la
[guía de streaming](https://developers.openai.com/api/docs/guides/streaming-responses).
No se cambia el SDK, proveedor, modelo ni familia de API para este trabajo.

Verificación del commit `892f422`: **4081 passed, 1 failed, 1 xfailed** en
116,67 s. El fallo es el mismo de VAD/idioma de la línea base; pasan las 49
pruebas nuevas de generación, Ruff 0.15.1, el gate medio/alto de Bandit 1.9.4,
doce tests Node, el smoke autenticado y el artefacto compilado. En diez corridas
del grafo cola/dispatcher/worker/contexto, con espera de red de 200 ms y
heartbeat de 5 ms, el retraso p95 fue 0,38 ms y el máximo 5,44 ms. La petición
de cancelación tardó como máximo 0,08 ms y el cierre del stream/cliente
cooperativo 9,97 ms. Son medidas sintéticas de generación; no prueban captura,
altavoz ni cancelación del grafo completo de voz. El
[registro de generación](../research/2026-10-25-h2-generation-validation.json)
conserva métricas sin redondear, alcance y limitaciones.

## Estado

Plan abierto. PCM, generación y entrega conversacional tienen implementación
y grafos sintéticos conectados. La siguiente sección conserva el alcance
conectado y los productores que todavía requieren auditoría; el grafo construido
por `main` y su cierre completo siguen siendo requisitos antes de cerrar H2.
La prueba acústica queda en H6.

## Tercera implementación: propietario de respuesta y entrega de PCM

`ResponseHandler` conserva un propietario hasta que el worker de síntesis y
salida termina realmente. Las esperas asíncronas de ese propietario no crean
threads; se admiten hasta 16 respuestas pendientes. Una sesión cancelada dentro
de una llamada bloqueante a TTS conserva la ocupación, descarta su PCM tardío
y no permite empezar otro worker TTS. No se presupone que cancelar una task
mate el motor. Los adaptadores síncronos usan el mismo propietario.

La sesión captura la zona antes de esperar o generar. El valor global de zona
activa sólo sirve de default para nuevas sesiones. Un hook puede reescribir
texto pero no cambiar esa procedencia. Cache, síntesis completa, síntesis por
chunks, earcons y broadcast usan `StreamingAudioPlayer` con handle registrado,
token previo al arranque y resultado explícito. Un fallo de salida no reintenta
la respuesta completa. El estado `is_speaking` refleja un dispositivo arrancado;
`is_responding` incluye espera de generación/síntesis, sin tratarla como eco.

La generación produce frases con cola de dos elementos y máximo de 400
caracteres por frase. La falta de delimitador fuerza corte acotado; un punto
al final de un delta no divide un decimal antes de recibir el siguiente delta.
El texto completo sigue limitado por `GenerationStream`. Un fallo de generación
descarta frases todavía pendientes y entrega un único error por el mismo
propietario, si la salida está disponible. Cancelar no genera ese error. Se
conserva el filler opcional, también serializado y cancelable.

El worker encolado llama `arespond` y registra en la petición si la entrega fue
atendida. El router deja de excluir `SLOW_LLM` por su tipo: usa ese resultado
para evitar repetir voz, y locuta errores que aún no se atendieron. El modo
legacy consume el mismo stream asíncrono y sólo registra una conversación
exitosa. Confirmaciones, feedback, sugerencias, earcons y los avisos del
ejecutor HA se esperan por la API asíncrona. El clasificador y la búsqueda
vectorial legacy se ejecutan fuera del loop. Cada captura admitida cancela la
respuesta anterior de su zona; el barge-in también limita la cancelación a su
zona, incluyendo broadcast.

El presupuesto de respuesta del dispatcher se configura en
`orchestrator.response_timeout_s`, por defecto 180 s para cola, generación y
entrega, en lugar del timeout fijo heredado de 5 s. Los límites de cola de
frases, tamaño de frase, espera de locución y plazos del dispositivo se
inyectan desde `tts.streaming` en el YAML existente. Esta implementación no
prueba todavía que 180 s sea el presupuesto doméstico adecuado: la evaluación
de respuestas largas no cacheadas y la separación de presupuestos por etapa
siguen pendientes.

Las pruebas de routers usan un doble explícito de la API asíncrona. Las pruebas
de entrega usan clases reales de respuesta, zonas y player con SDK/HTTP en
memoria y driver callback. Se retiran tests que sólo cambiaban flags privados
o suponían `tts.speak`/`sd.play`; se comprueban estados y cancelación sobre el
dispositivo simulado que realmente consume PCM.

Verificación del commit `5182a01`: **4101 passed, 1 failed, 1 xfailed** en
123,08 s. El único fallo sigue siendo VAD/idioma. Pasan 19 grafos de entrega,
cuatro combinaciones de router real con éxito/fallo de generación, Ruff,
Bandit medio/alto, doce tests Node, smoke autenticado y smoke compilado. En
diez respuestas de 24.000 frames con espera de red de 200 ms se conservó el
PCM completo; el retraso p95 del loop fue 0,36 ms y el máximo 1,57 ms. La
cancelación con limpieza de la salida simulada tardó como máximo 1,63 ms.
El [registro del propietario](../research/2026-10-25-h2-owned-response-validation.json)
conserva métricas, límites y alcance; no son medidas físicas ni una validación
del ciclo de vida completo de `main`.

### Trabajo pendiente de H2

- Auditar los demás consumidores TTS conectados por `main`, incluidos
  recordatorios, rutinas, avisos y alertas, para que no compitan con este
  propietario ni bloqueen la captura; su funcionalidad de producto se completa
  en H5, pero su disciplina de entrega pertenece también a este contrato.
- Verificar el cierre del propietario y los workers dentro del ciclo de vida
  completo de la aplicación, con cancelación y fallos durante arranque/parada.
- Probar timeout de petición durante entrega parcial y distinguir quién anuncia
  ese error de la cancelación normal, sin duplicar un error ya en reproducción.
- Ampliar el corpus conectado con respuesta larga, captura/barge-in del loop
  real simulado y presupuesto por etapa; no confundir los grafos de respuesta
  aislados con el grafo completo de `main`.
- Validar salida, micrófono, amplificador, latencia física y barge-in en H6.
