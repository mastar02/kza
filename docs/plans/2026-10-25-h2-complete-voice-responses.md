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

## Estado

Plan abierto. La capa PCM tiene implementación y pruebas sintéticas; quedan el
consumo no bloqueante del reasoner, la responsabilidad única de locución,
la conexión completa de cancelación y sus métricas del event loop. La prueba
acústica queda en H6. H2 no está cerrado.
