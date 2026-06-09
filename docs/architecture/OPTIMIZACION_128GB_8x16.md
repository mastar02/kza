# Optimización para 128GB (8x16GB - 8 Canales)

## Tu Hardware Real

```
RAM: 8x16GB DDR5-5600 RDIMM ECC
Canales: 8 de 8 (máximo)
Ancho de banda: ~358 GB/s (teórico)
Capacidad: 128GB (fijo, no expandible)
```

## Ventaja Clave: Ancho de Banda

Tu configuración 8x16GB es **MEJOR para LLMs** que 2x64GB porque:

| Métrica | 2x64GB | 8x16GB |
|---------|--------|--------|
| Ancho de banda | ~102 GB/s | ~358 GB/s |
| Tokens/segundo* | ~6-8 | ~15-20 |
| Latencia primer token | ~3s | ~1.2s |

*Para Llama 70B Q4_K_M

## El Problema: Capacidad Fija

Con 128GB no puedes usar Q8_0 cómodamente:

| Modelo | Tamaño | RAM Libre | ¿Viable? |
|--------|--------|-----------|----------|
| Llama 70B Q8_0 | ~70GB | ~34GB | ⚠️ Justo |
| Llama 70B Q6_K | ~55GB | ~49GB | ✅ Bien |
| Llama 70B Q4_K_M | ~40GB | ~64GB | ✅ Óptimo |
| Qwen 72B Q4_K_M | ~42GB | ~62GB | ✅ Óptimo |

## Recomendación: Q4_K_M + Contexto Grande

Con Q4_K_M y el ancho de banda de 8 canales:
- **Velocidad similar** a Q8_0 con 2 canales
- **Más contexto** disponible (hasta 64K tokens)
- **Sin riesgo de swap**

## Configuración Recomendada

```yaml
# config/settings.yaml

reasoner:
  # RECOMENDADO para 8x16GB (128GB, 8 canales)
  model_path: "./models/Llama-3.3-70B-Instruct-Q4_K_M.gguf"

  # Contexto grande aprovechando RAM libre
  n_ctx: 65536        # 64K tokens (antes 32K)
  n_batch: 1024       # Batch más grande

  # Todos los cores
  n_threads: 24
  n_gpu_layers: 0

  # RoPE para contexto extendido
  rope_freq_base: 1000000.0
  rope_freq_scale: 1.0
```

## Alternativa: Modelo Dual

Con la RAM libre puedes tener DOS modelos:

```yaml
reasoner:
  # Modelo principal para razonamiento complejo
  model_path: "./models/Qwen2.5-32B-Instruct-Q8_0.gguf"  # ~34GB

  # El router en GPU 2 ya maneja consultas simples
  # Esto te da ~70GB libres para contexto y servicios
```

## Distribución de Memoria Óptima

```
┌─────────────────────────────────────────────────────────────────┐
│  MEMORIA RAM: 128GB (8x16GB, 8 canales, ~358 GB/s)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OPCIÓN A: Llama 70B Q4_K_M                                     │
│  ──────────────────────────────────────────────────────────────│
│  Modelo Q4_K_M:               ~40GB  ████████████               │
│  Contexto 64K tokens:         ~16GB  █████                      │
│  ChromaDB + embeddings:       ~5GB   ██                         │
│  Sistema + servicios:         ~15GB  █████                      │
│  ──────────────────────────────────────────────────────────────│
│  USADO:                       ~76GB                             │
│  LIBRE:                       ~52GB  (margen de seguridad)      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OPCIÓN B: Qwen 32B Q8_0 (máxima calidad en español)            │
│  ──────────────────────────────────────────────────────────────│
│  Modelo Q8_0:                 ~34GB  ██████████                  │
│  Contexto 128K tokens:        ~32GB  ██████████                  │
│  ChromaDB + embeddings:       ~5GB   ██                         │
│  Sistema + servicios:         ~15GB  █████                      │
│  ──────────────────────────────────────────────────────────────│
│  USADO:                       ~86GB                             │
│  LIBRE:                       ~42GB  (margen de seguridad)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark Esperado

Con tu configuración 8x16GB y Q4_K_M:

| Métrica | Valor Esperado |
|---------|----------------|
| Tokens/segundo (generación) | 15-22 tok/s |
| Latencia primer token | 1.0-1.5s |
| Throughput prompt | 200-300 tok/s |
| Contexto máximo útil | 64K tokens |

## Comandos para Verificar

```bash
# Ver uso de memoria en tiempo real
watch -n 1 free -h

# Monitorear durante inferencia
htop

# Verificar ancho de banda de memoria
sudo dmidecode -t memory | grep -E "Speed|Size|Locator"

# Test de ancho de banda
sudo apt install sysbench
sysbench memory run
```

## Modelos Recomendados (Descarga)

```bash
# OPCIÓN A: Llama 3.3 70B Q4_K_M (RECOMENDADO)
# ~40GB, excelente balance calidad/velocidad
huggingface-cli download TheBloke/Llama-3.3-70B-Instruct-GGUF \
  llama-3.3-70b-instruct.Q4_K_M.gguf --local-dir ./models

# OPCIÓN B: Qwen 2.5 72B Q4_K_M (Mejor en español)
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-GGUF \
  qwen2.5-72b-instruct-q4_k_m.gguf --local-dir ./models

# OPCIÓN C: Qwen 32B Q8_0 (Máxima calidad con margen)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \
  qwen2.5-32b-instruct-q8_0.gguf --local-dir ./models
```

## Conclusión

Tu decisión de 8x16GB es **técnicamente correcta** para LLM inference:

✅ **3.5x más ancho de banda** = tokens más rápidos
✅ **8 canales saturados** = máximo rendimiento del Threadripper
✅ **128GB suficientes** para 70B Q4_K_M + contexto grande

⚠️ **Única limitación**: No podrás expandir a 256GB+ en el futuro sin cambiar módulos.

**Recomendación final**: Usa **Q4_K_M** y aprovecha el ancho de banda extra para contexto más grande (64K). La diferencia de calidad Q4 vs Q8 es mínima (~2-3%) pero ganarás mucho en velocidad y margen de RAM.

---

*Documento para KZA con hardware 8x16GB DDR5*
