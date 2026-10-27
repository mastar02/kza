# Evaluación de agrupación de voces — candidato 1

Protocolo escrito antes de ejecutar la evaluación del agrupador. Es parte de H3;
no acepta la agrupación acústica ni autoriza una release por sí mismo.

Se reutiliza el encoder ya observado por el registro, sin cambiar modelo,
dispositivo ni asignación de GPUs. El candidato usa similitud coseno mínima
0,88 con todos los miembros del grupo, una diferencia mínima de 0,08 frente
al segundo grupo y segmentos de al menos dos segundos. Un grupo admite hasta
32 muestras del mismo descriptor y dimensión del encoder. No une dos grupos
por una cadena de vecinos. Los grupos corregidos manualmente no reciben nuevas
asignaciones automáticas. Un segmento corto, una coincidencia insuficiente o
ambigua puede quedar sin asociación. El texto y la habitación no deciden quién
habla. Estos parámetros son una hipótesis inicial, no una calibración medida.

El corpus real permanece fuera de Git bajo `data/`, con procedencia, permiso
de uso, plazos de retención y etiquetas manuales. Separar sesiones completas
entre calibración y evaluación; nunca dividir copias o fragmentos de una misma
captura entre conjuntos. Registrar hash del manifiesto y del encoder. Para
cada segmento: ID opaco, hablante de referencia o desconocido, sesión,
habitación/distancia, duración y condiciones de ruido, TV, eco y solapamiento.
No exportar nombres, WAV, texto ni vectores a la evidencia versionada.

Exigir al menos tres voces consentidas, con un mínimo de diez segmentos claros
por voz para calibración y treinta por voz para evaluación, en dos habitaciones
y dos distancias. Añadir al menos diez segmentos por condición adversa: cortos
(<2 s), TV, eco y voces superpuestas. Las sesiones del conjunto de evaluación
no se usan para elegir parámetros. Conservar el orden de entrada y evaluar
además diez permutaciones con semillas 0–9, sin seleccionar sólo la mejor.

Antes de evaluar se fijan estos criterios: cero uniones entre personas distintas
en evaluación, precisión de pares >=0,99, recuperación de pares de la misma voz
>=0,80 en audio claro y cobertura >=0,70 de los segmentos claros elegibles.
Reportar también separaciones erróneas, abstenciones por condición y tiempo de
procesamiento. Los segmentos cortos y las superposiciones etiquetadas deben
abstenerse; la similitud sola no demuestra que detecte solapamiento o TV. Una
falla en esa condición deja la aceptación abierta y exige mejorar las señales
de calidad y volver a separar calibración/evaluación. No relajar el criterio
para hacer pasar el mismo conjunto ni presentar coseno como probabilidad.

Las pruebas sintéticas sólo comprueban mecánica: enlace completo, ambigüedad,
independencia de texto/habitación, encoders incompatibles y límites de capacidad.
La aceptación exige el corpus real y todos los criterios anteriores, además de
la verificación de TTL, procedencia, retiro de vistas y correcciones persistentes.
Nombrar o corregir un grupo no verifica identidad ni concede permisos.
