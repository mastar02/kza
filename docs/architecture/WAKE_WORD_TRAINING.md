# Guía de Entrenamiento de Wake Word Personalizado

## Resumen

Esta guía te ayudará a crear tu propio wake word personalizado para KZA (por ejemplo: "Oye KZA", "Hey Casa", "Jarvis", etc.)

## Requisitos

- Micrófono de buena calidad
- Ambiente relativamente silencioso
- 10-15 minutos de tu tiempo
- Al menos 50 muestras de voz (30 mínimo)

## Proceso de Entrenamiento

### Paso 1: Decidir tu Wake Word

Elige una palabra o frase que:
- Sea fácil de pronunciar
- No se confunda con palabras comunes
- Tenga 2-4 sílabas (ideal)

**Buenos ejemplos:**
- "Oye KZA"
- "Hey Casa"
- "Hola Jarvis"
- "Ok Computer"

**Malos ejemplos:**
- "Hola" (muy común)
- "Luz" (muy corto)
- "Enciende las luces de la cocina" (muy largo)

### Paso 2: Grabar Muestras Positivas

Estas son grabaciones de TI diciendo el wake word.

```bash
cd /ruta/a/kza
python scripts/train_wakeword.py record --name "oye kza" --positive --count 50
```

**Consejos para mejores resultados:**
1. Varía tu tono de voz (normal, susurro, en voz alta)
2. Graba en diferentes posiciones (cerca, lejos del mic)
3. Graba a diferentes horas (mañana, noche)
4. Incluye variaciones naturales ("Oye KZA", "OYE kza", "oye KZA")

### Paso 3: Grabar Muestras Negativas

Estas son grabaciones de frases que NO son el wake word.

```bash
python scripts/train_wakeword.py record --name "oye kza" --negative --count 50
```

**Qué decir:**
- Frases similares: "Oye papá", "Hey", "Casa"
- Comandos de domótica: "Prende la luz", "Apaga el aire"
- Conversación normal: "¿Qué hora es?", "Hace calor"
- Nombres: "María", "Juan"
- Números: "Uno, dos, tres"

### Paso 4: Entrenar el Modelo

```bash
python scripts/train_wakeword.py train --name "oye kza" --epochs 100
```

**Parámetros opcionales:**
- `--epochs 150` - Más épocas = mejor precisión (pero más tiempo)
- `--data-dir ./mis_datos` - Directorio de datos personalizado
- `--output-dir ./mis_modelos` - Directorio de salida personalizado

El entrenamiento toma aproximadamente:
- 50 muestras + 100 épocas ≈ 5-10 minutos
- 100 muestras + 150 épocas ≈ 15-20 minutos

### Paso 5: Probar el Wake Word

```bash
python scripts/train_wakeword.py test --model "oye_kza" --threshold 0.5
```

**Ajustar el umbral:**
- `0.3` - Más sensible (detecta más, más falsos positivos)
- `0.5` - Balanceado (recomendado para empezar)
- `0.7` - Más estricto (menos falsos positivos, puede no detectar)

### Paso 6: Activar en Configuración

Edita `config/settings.yaml`:

```yaml
wake_word:
  model: "oye_kza"  # Nombre de tu modelo entrenado
  threshold: 0.5
  inference_framework: "onnx"
```

## Comandos Útiles

### Listar modelos disponibles
```bash
python scripts/train_wakeword.py list
```

### Ver estadísticas de grabación
```bash
# Se muestra automáticamente al grabar
# También puedes revisar:
ls -la data/wakeword_training/oye_kza/positive/
ls -la data/wakeword_training/oye_kza/negative/
```

### Agregar más muestras
```bash
# Agregar 20 muestras positivas más
python scripts/train_wakeword.py record --name "oye kza" --positive --count 20

# Re-entrenar con más datos
python scripts/train_wakeword.py train --name "oye kza" --epochs 100
```

## Estructura de Archivos

```
data/wakeword_training/
└── oye_kza/
    ├── positive/
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    └── negative/
        ├── sample_001.wav
        ├── sample_002.wav
        └── ...

models/wakeword/
├── oye_kza.onnx          # Modelo entrenado
└── oye_kza.json          # Metadata
```

## Múltiples Wake Words

Puedes tener varios wake words activos. Edita `config/settings.yaml`:

```yaml
wake_word:
  model: "oye_kza,hey_jarvis"  # Separados por coma
  threshold: 0.5
```

## Solución de Problemas

### "No se detecta mi wake word"
1. Reduce el threshold a 0.3
2. Graba más muestras positivas
3. Asegúrate de pronunciar igual que en las grabaciones

### "Demasiados falsos positivos"
1. Aumenta el threshold a 0.6 o 0.7
2. Graba más muestras negativas con frases similares
3. Re-entrena con más épocas

### "Error de entrenamiento"
1. Verifica que tienes al menos 30 muestras de cada tipo
2. Verifica que los archivos .wav no están corruptos
3. Intenta con menos épocas primero (50)

### "El modelo es muy grande"
Los modelos típicos son de 100-500 KB. Si es mucho más grande:
1. Usa menos épocas
2. Considera grabar muestras más cortas (1-2 segundos)

## Mejores Prácticas

1. **Graba en el ambiente real** - Donde usarás KZA
2. **Incluye ruido de fondo** - Algunas muestras con TV, música, etc.
3. **Varias personas** - Si varios usuarios usarán el wake word, que todos graben
4. **Re-entrena periódicamente** - Si cambia tu voz o el ambiente
5. **Backup de muestras** - Guarda tus grabaciones por si necesitas re-entrenar

## Integración con Sistema Multi-Usuario

Si tienes identificación de speaker habilitada, el wake word detectará la activación y luego el sistema identificará quién habló:

```
Wake Word detectado → STT transcribe → Speaker ID identifica → Respuesta personalizada
```

Esto significa que el wake word debe ser el mismo para todos, pero las respuestas se personalizan por usuario.

---

## Ejemplo Completo

```bash
# 1. Navegar al proyecto
cd /ruta/a/kza

# 2. Grabar positivas (di "Oye KZA" 50 veces)
python scripts/train_wakeword.py record --name "oye kza" --positive --count 50

# 3. Grabar negativas (di otras frases 50 veces)
python scripts/train_wakeword.py record --name "oye kza" --negative --count 50

# 4. Entrenar
python scripts/train_wakeword.py train --name "oye kza" --epochs 100

# 5. Probar
python scripts/train_wakeword.py test --model "oye_kza" --threshold 0.5

# 6. Activar (editar config/settings.yaml)
# wake_word:
#   model: "oye_kza"
#   threshold: 0.5

# 7. Reiniciar KZA
python src/main.py
```

---

*Guía actualizada: Febrero 2026*
*Proyecto KZA*
