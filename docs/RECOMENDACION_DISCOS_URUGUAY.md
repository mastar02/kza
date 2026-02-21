# Recomendaciones de Almacenamiento para KZA

**Fecha**: Febrero 2026
**Mercado**: Uruguay

---

## Necesidades de Almacenamiento KZA

### Proyección a 3 años (sistema en crecimiento)

| Componente | Año 1 | Año 3 | Tipo Recomendado |
|------------|-------|-------|------------------|
| Llama 70B Q4_K_M | ~40GB | ~40GB | NVMe (rápido) |
| Modelos alternativos | ~100GB | ~300GB | NVMe |
| ChromaDB + embeddings | ~50GB | ~200GB | NVMe |
| LoRA adapters (nightly) | ~5GB | ~50GB | NVMe |
| Contextos usuarios | ~1GB | ~5GB | Cualquiera |
| Datos de entrenamiento | ~10GB | ~100GB | NVMe |
| Logs + sistema | ~50GB | ~100GB | Cualquiera |
| **Total activo** | **~250GB** | **~800GB** | - |
| **Recomendado** | **4TB NVMe + 12TB HDD** | - | Híbrido |

⚠️ **2TB se queda corto** - Con crecimiento de ChromaDB, adapters LoRA y modelos alternativos, necesitas margen.

---

## Configuración Recomendada

```
┌─────────────────────────────────────────────────────────────┐
│  ALMACENAMIENTO KZA - CONFIGURACIÓN PARA 3+ AÑOS            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DISCO 1: NVMe 4TB (Sistema + Modelos + Datos activos)      │
│  ────────────────────────────────────────────────────────── │
│  /boot, /, /home                          ~100GB            │
│  /opt/kza/models/                         ~500GB            │
│    ├─ Llama-70B-Q4_K_M.gguf              ~40GB             │
│    ├─ Qwen2.5-72B-Q4_K_M.gguf            ~45GB             │
│    ├─ modelos de backup/swap             ~200GB            │
│    └─ lora_adapters/nightly/             ~50GB (3 años)    │
│  /opt/kza/data/chroma_db/                 ~200GB            │
│  /opt/kza/data/nightly_training/          ~100GB            │
│  /opt/kza/data/contexts/                  ~10GB             │
│  SWAP                                     ~64GB             │
│  ────────────────────────────────────────────────────────── │
│  Libre para crecimiento                   ~3TB              │
│                                                              │
│  DISCO 2: HDD 12TB NAS (Backup + Histórico + Expansión)     │
│  ────────────────────────────────────────────────────────── │
│  /mnt/storage/                                              │
│    ├─ models_archive/        Modelos descargados            │
│    ├─ training_archive/      Datasets históricos            │
│    ├─ chroma_backups/        Snapshots semanales            │
│    ├─ conversation_logs/     Todo el historial              │
│    └─ system_backups/        Backups incrementales          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Opciones de Compra en Uruguay

### 1. SSD NVMe 4TB (Prioridad Alta) ⭐

| Modelo | Velocidad | Precio (USD) | Tienda |
|--------|-----------|--------------|--------|
| **Crucial P3 Plus 4TB** | 5000/4200 MB/s | ~$350-380 | [PC Gamer UY](https://www.pcgamer-uy.com/discos-ssd) |
| **Samsung 990 PRO 4TB** | 7450/6900 MB/s | ~$400-450 | Importación |
| WD Black SN850X 4TB | 7300/6600 MB/s | ~$380-420 | [MercadoLibre](https://mercadolibre.com.uy) |
| Sabrent Rocket 4 Plus 4TB | 7100/6600 MB/s | ~$350-400 | Importación |
| Kingston KC3000 4TB | 7000/7000 MB/s | ~$370-400 | [NewTek](https://www.newtek.com.uy) |

**Recomendación**: Crucial P3 Plus 4TB - mejor precio, suficiente velocidad para LLM.

### Alternativa: 2x NVMe 2TB en RAID0

Si no conseguís 4TB, podés usar 2x 2TB en RAID0:
- Crucial P3 Plus 2TB x2 = ~$390
- Duplica velocidad de lectura secuencial
- ⚠️ Riesgo: si falla uno, perdés todo (usar backups)

### 2. HDD NAS 12TB (Para backup/histórico)

| Modelo | Capacidad | Precio Ref. (USD) | Notas |
|--------|-----------|-------------------|-------|
| **Seagate IronWolf** | 12TB | ~$250-300 | Optimizado NAS 24/7 |
| **Seagate IronWolf Pro** | 12TB | ~$300-350 | + Health Management |
| WD Red Plus | 12TB | ~$280-320 | CMR, silencioso |
| WD Red Pro | 12TB | ~$320-380 | Para NAS profesional |
| Seagate Exos | 12TB | ~$220-260 | Enterprise, ruidoso |

**Recomendación**: Seagate IronWolf 12TB - diseñado para operación 24/7, 3 años garantía.

### Tiendas en Uruguay

1. **PC Gamer UY** - [pcgamer-uy.com](https://www.pcgamer-uy.com)
   - Precios en USD con IVA incluido
   - Stock de NVMe Crucial, ADATA

2. **NewTek Computers** - [newtek.com.uy](https://www.newtek.com.uy)
   - WD, Kingston, ADATA
   - Local en Montevideo

3. **MercadoLibre Uruguay** - [mercadolibre.com.uy](https://listado.mercadolibre.com.uy)
   - WD Red disponible
   - Cuotas sin interés
   - Envío gratis

4. **HARD PC** - [hardpc.com.uy](https://www.hardpc.com.uy)
   - Asesoramiento especializado
   - 14 de julio 1362, Pocitos

5. **Infinit** - [infinit.com.uy](https://www.infinit.com.uy)
   - SSDs y NVMe
   - Ofertas frecuentes

6. **Uruguay Portátil** - [uruguayportatil.com](https://www.uruguayportatil.com)
   - Patriot, otras marcas

---

## ⚠️ Contexto de Precios 2026

Los precios de almacenamiento han subido significativamente debido a la demanda de IA:

- **HDDs**: Aumento promedio ~46% desde septiembre 2025
- **SSDs**: Precios de NAND flash al alza por demanda de datacenters AI

**Consejo**: Si ves buen precio, comprar ahora. La tendencia es al alza.

---

## Configuración Recomendada para KZA

### Opción Económica (~$300-350 USD)

```yaml
# Mínimo viable
sistema:
  nvme: "Crucial P3 Plus 2TB"  # ~$195
  hdd: "Seagate BarraCuda 4TB"  # ~$100
```

### Opción Recomendada (~$600-650 USD) ⭐

```yaml
# Capacidad para 3+ años de crecimiento
sistema:
  nvme: "Crucial P3 Plus 4TB"     # ~$350-380
  hdd: "Seagate IronWolf 12TB"    # ~$250-300
```

### Opción Premium (~$750-850 USD)

```yaml
# Máximo rendimiento + redundancia
sistema:
  nvme: "Samsung 990 PRO 4TB"     # ~$400-450
  hdd: "Seagate IronWolf Pro 12TB" # ~$300-350
  # O 2x IronWolf 8TB en RAID1 para redundancia
```

### Opción Mínima (~$450-500 USD)

```yaml
# Si el presupuesto es muy ajustado (cuidado con crecimiento)
sistema:
  nvme: "Crucial P3 Plus 2TB x2 RAID0"  # ~$390
  hdd: "Seagate IronWolf 8TB"            # ~$180
```

---

## Estructura de Directorios KZA

```bash
# En NVMe (rápido)
/opt/kza/
├── models/                    # Modelos activos
│   ├── Llama-3.3-70B-Q4_K_M.gguf
│   └── lora_adapters/
│       └── nightly/          # Adapters del entrenamiento nocturno
├── data/
│   ├── chroma_db/            # Vector database
│   ├── contexts/             # Contextos por usuario
│   ├── conversations/        # ConversationCollector
│   └── nightly_training/     # Datasets de entrenamiento
├── config/
└── logs/

# En HDD (backup/histórico)
/mnt/storage/
├── models_backup/            # Otros modelos descargados
├── historical_data/          # Datos históricos
├── training_archives/        # Datasets antiguos
└── system_backups/           # Backups periódicos
```

---

## Script de Setup de Almacenamiento

```bash
#!/bin/bash
# setup_storage.sh - Configurar almacenamiento para KZA

# Variables
NVME_MOUNT="/opt/kza"
HDD_MOUNT="/mnt/storage"

# Crear estructura en NVMe
sudo mkdir -p $NVME_MOUNT/{models,data,config,logs}
sudo mkdir -p $NVME_MOUNT/models/lora_adapters/nightly
sudo mkdir -p $NVME_MOUNT/data/{chroma_db,contexts,conversations,nightly_training}

# Crear estructura en HDD
sudo mkdir -p $HDD_MOUNT/{models_backup,historical_data,training_archives,system_backups}

# Permisos
sudo chown -R $USER:$USER $NVME_MOUNT
sudo chown -R $USER:$USER $HDD_MOUNT

# Link simbólico para modelos de backup
ln -s $HDD_MOUNT/models_backup $NVME_MOUNT/models/backup

echo "Estructura de almacenamiento creada"
```

---

## Backup Automático

Agregar al crontab para backup diario:

```bash
# Backup diario de contextos y ChromaDB a HDD
0 4 * * * rsync -av /opt/kza/data/ /mnt/storage/system_backups/daily/

# Backup semanal de modelos LoRA
0 5 * * 0 rsync -av /opt/kza/models/lora_adapters/ /mnt/storage/models_backup/lora/
```

---

## Conclusión

**Compra recomendada para KZA:**

| Item | Modelo | Precio Aprox. |
|------|--------|---------------|
| **NVMe** | Crucial P3 Plus 4TB | ~$350-380 USD |
| **HDD** | Seagate IronWolf 12TB | ~$250-300 USD |
| **Total** | | **~$600-680 USD** |

Esta configuración te da:
- ✅ **4TB NVMe**: Espacio para múltiples modelos LLM + ChromaDB en crecimiento
- ✅ **12TB HDD**: Backup extenso + histórico de entrenamientos
- ✅ Margen para 3+ años de operación
- ✅ SSD rápido para inferencia y entrenamiento nocturno
- ✅ HDD NAS confiable para operación 24/7

### Comparativa de opciones

| Opción | NVMe | HDD | Total | Para quién |
|--------|------|-----|-------|------------|
| **Mínima** | 2TB x2 RAID0 | 8TB | ~$570 | Presupuesto ajustado |
| **Recomendada** | 4TB | 12TB | ~$650 | Mayoría de usuarios ⭐ |
| **Premium** | 4TB Pro | 12TB Pro | ~$800 | Máximo rendimiento |

---

*Documento generado para KZA - Febrero 2026*
*Precios sujetos a cambios - verificar en tiendas locales*
