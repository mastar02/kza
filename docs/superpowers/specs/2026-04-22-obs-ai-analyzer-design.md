# OBS AI Analyzer — Diseño

**Fecha:** 2026-04-22  
**Estado:** Aprobado  
**Scope:** Phase 3 del stack de observabilidad — analizador IA de tráfico L2+L3

---

## Objetivo

Un servicio Python autónomo (`obs-analyzer`) que analiza logs de red en tiempo cuasi-real usando los modelos LLM ya corriendo en el servidor. Dos modos de operación:

- **AlertLoop** (cada 5 min): detecta anomalías, dispara Ntfy + anotación Grafana si `anomaly_score > 0.75`
- **ReportLoop** (03:00 diario): análisis profundo 24h, documento indexado en ES + anotación Grafana

---

## Fuentes de datos

### L3 — OPNsense filterlog (ya existente)
Índice: `opnsense-logs-YYYY.MM.dd`  
Campos actuales: `action, direction, protocol, src_ip, dst_ip, src_port, dst_port, interface`  
Campos a agregar: `ttl, tcp_flags, pkt_length, rule_tracker` + enriquecimiento `geo.*` y `asn.*`

### L2 — MikroTik CRS310 via SNMP polling + syslog (nuevo — Step 1 del plan)
**Nota:** El CRS310 usa ASIC Marvell-98DX226S para switching hardware — el bridge filter no intercepta tráfico forwarded en hardware. Se usa SNMP polling en su lugar.

**SNMP polling** (cada 5 min desde collector.py, community `obsread`):
- `dot1dTpFdbTable` (OID 1.3.6.1.2.1.17.4.3.1) → MAC FDB table: detección de MACs nuevas y moves entre puertos
- `ipNetToMediaTable` (OID 1.3.6.1.2.1.4.22.1) → ARP table: IP→MAC, detección de ARP spoofing
- `ifTable` (OID 1.3.6.1.2.1.2.2.1) → bytes/paquetes por puerto: spikes de tráfico por interfaz

**Syslog** (continúa enviando eventos de sistema a Logstash :5514):
Índice: `mikrotik-logs-YYYY.MM.dd`  
Formato RouterOS nativo: `{topic1},{topic2} {mensaje}` (sin wrapper RFC3164)  
Eventos útiles: interface link up/down, login attempts al switch, cambios de config

---

## Arquitectura

```
OPNsense syslog UDP :5514 ──────────────────────────────────┐
MikroTik syslog UDP :5514 ──────────────────────────────────┤
                                                             ▼
                                                  Logstash 8.15.5
                                          ┌────────────────────────────┐
                                          │ opnsense.conf (existente)  │
                                          │   + geoip filter (nuevo)   │
                                          │   + asn filter (nuevo)     │
                                          │   + ttl/tcp_flags (nuevo)  │
                                          ├────────────────────────────┤
                                          │ mikrotik.conf (nuevo)      │
                                          │   grok RouterOS syslog     │
                                          └────────────┬───────────────┘
                                                       │
                              ┌────────────────────────┼──────────────────────────┐
                              ▼                        ▼                          ▼
                 opnsense-logs-* (30d)    mikrotik-logs-* (30d)    obs-ai-* indices
                              │                        │
                              └────────────┬───────────┘
                                           │ ES aggregations
                                           ▼
                              obs-analyzer.service (Python, Quadlet rootless, usuario obs)
                              ┌────────────────────────────────────────────────────────┐
                              │  AlertLoop — cada 5 min                               │
                              │    collector.py → stats L3+L2 merged (~500 tokens)    │
                              │    analyzer.py → Qwen 7B :8100 → anomaly JSON         │
                              │    if score > 0.75:                                   │
                              │      notifier.py → Ntfy push + Grafana annotation     │
                              │    indexer.py → obs-ai-analysis-YYYY.MM.dd            │
                              │                                                        │
                              │  ReportLoop — 03:00 diario                            │
                              │    collector.py → stats 24h extended (~2000 tokens)   │
                              │    analyzer.py → Qwen 72B :8200 → reporte narrativo   │
                              │    indexer.py → obs-ai-reports-YYYY.MM.dd             │
                              │    notifier.py → Grafana annotation (daily-report)    │
                              └────────────────────────────────────────────────────────┘
```

---

## Step 1 del plan (prerequisito): Configurar MikroTik bridge firewall logging

Antes de cualquier código Python, habilitar logging L2 en el switch:

```routeros
# 1. Agregar action de logging remoto (si no existe)
/system logging action
add name=remote-obs target=remote remote=192.168.1.2 remote-port=5514 \
    src-address=192.168.1.5 bsd-syslog=yes syslog-facility=local0

# 2. Reglas bridge firewall — forward chain, log únicamente (no block)
/interface bridge filter
add chain=forward action=log log-prefix="L2-FWD:" \
    in-bridge=bridge comment="obs-analyzer L2 visibility"

# 3. Habilitar log topics relevantes
/system logging
add topics=firewall action=remote-obs
add topics=interface action=remote-obs
add topics=system action=remote-obs
```

**Consideraciones:**
- `action=log` sin `passthrough=no` — solo registra, no bloquea
- El volumen puede ser alto inicialmente; monitorear antes de ajustar retención
- Si el volumen es excesivo, acotar con `src-address` o `dst-address` para filtrar solo tráfico inter-dispositivo (excluir gateway 192.168.1.1)

---

## Modelos LLM

| Loop | Modelo | Endpoint | Justificación |
|---|---|---|---|
| AlertLoop (5 min) | Qwen 7B AWQ | :8100 (vLLM infra compartido) | Rápido, bajo costo por token, suficiente para clasificación |
| ReportLoop (diario) | Qwen 72B Q8_0 | :8200 (kza-72b.service) | Ya caliente, mayor capacidad de razonamiento, 1 llamada/día |

---

## Detección de anomalías

### AlertLoop — anomaly_types (7B)

| Tipo | Señal |
|---|---|
| `port_scan` | Muchos dst_ports distintos desde misma IP en 5 min |
| `volume_spike` | total_events > 2x baseline_avg |
| `new_ip_suspicious` | IP nunca vista en 7d + puertos sensibles (22, 3389, 5900, 23) |
| `exfiltration_hint` | Tráfico outbound desde LAN hacia IP externa nueva, protocolo inusual |
| `mac_spoofing` | Misma MAC vista en >1 puerto físico del bridge |
| `unknown_device` | MAC nueva no vista en 24h aparece en bridge table |
| `port_flapping` | Interfaz up/down >3 veces en ventana |
| `lateral_scan` | Dispositivo LAN generando tráfico hacia muchos dst_ip internos |
| `switch_mgmt_attack` | Login attempts a 192.168.1.5 (IP switch) |
| `none` | Sin anomalía |

### Umbral de alerta
`anomaly_score > 0.75` → Ntfy push + Grafana annotation  
`anomaly_score 0.5-0.75` → solo indexado (visible en Kibana)  
`anomaly_score < 0.5` → indexado, sin acción

---

## Schema ES — nuevos índices

```
obs-ai-analysis-YYYY.MM.dd   retención 30 días
{
  "@timestamp": "...",
  "window_start": "...",
  "window_end": "...",
  "anomaly_score": 0.87,
  "anomaly_type": "port_scan",
  "reasons": ["spike 4x baseline", "312 hits SSH desde IP nueva DE"],
  "recommendation": "considerar agregar 185.220.101.5 a scanners_known",
  "stats_snapshot": { ...stats JSON completo... },
  "model": "qwen-7b-awq"
}

obs-ai-reports-YYYY.MM.dd    retención 90 días
{
  "@timestamp": "...",
  "report_date": "2026-04-22",
  "narrative": "...",
  "top_threats": [...],
  "new_ips_seen": [...],
  "recommended_blocks": [...],
  "model": "qwen-72b-q8"
}
```

Cambio en retención existente: `opnsense-logs-*` de 1 día → **30 días** (nueva política ILM `logs-30d`).

---

## Prompts

### System prompt — AlertLoop (7B, fijo)
```
You are a network security analyst reviewing firewall logs.
Analyze the provided traffic statistics and respond ONLY with valid JSON:
{
  "anomaly_score": <float 0.0-1.0>,
  "anomaly_type": <"port_scan"|"volume_spike"|"new_ip_suspicious"|"exfiltration_hint"|
                   "mac_spoofing"|"unknown_device"|"port_flapping"|"lateral_scan"|
                   "switch_mgmt_attack"|"none">,
  "reasons": [<list of strings, max 3>],
  "recommendation": <string, one sentence, null if none>
}
Score guide: 0.0=normal, 0.5=worth noting, 0.75=alert, 1.0=critical.
Be conservative — only score >0.75 for clear threats.
```

### User message — AlertLoop (`build_alert_prompt()`)
Texto narrativo generado desde stats dict con: spike ratio, top IPs con país/ASN/tcp_flags/TTL, puertos sensibles, IPs nunca vistas en 7d, eventos L2 (MAC moves, nuevas MACs, port flaps), tráfico outbound nuevo.

### System prompt — ReportLoop (72B)
Análisis narrativo completo: tendencias 24h, IPs de interés, recomendaciones de bloqueo, comparativa con día anterior.

---

## Layout de archivos

```
/home/obs/app/
├── logstash/pipeline/
│   ├── opnsense.conf          # existente + geoip/asn/ttl/tcp_flags
│   └── mikrotik.conf          # nuevo — grok RouterOS syslog L2
└── analyzer/
    ├── src/
    │   ├── collector.py       # ES aggregations L3+L2
    │   ├── analyzer.py        # LLM calls (7B / 72B) + response parsing
    │   ├── notifier.py        # Ntfy HTTP API + Grafana Annotations API
    │   ├── indexer.py         # write to obs-ai-* indices
    │   └── scheduler.py       # asyncio entry point — dos loops
    ├── config.yaml            # umbral (0.75), URLs, Ntfy topic, horario report
    └── requirements.txt       # elasticsearch>=8, httpx, pyyaml

/home/obs/.config/containers/systemd/
└── obs-analyzer.container     # Quadlet rootless, obs-internal.network
```

---

## Secuencia de implementación (orden crítico)

1. **[PRIMERO] MikroTik bridge firewall logging** — habilitar en switch + verificar que llegan logs a Logstash :5514
2. **Logstash mikrotik.conf** — parser grok para RouterOS syslog
3. **Logstash opnsense.conf** — agregar geoip + asn + ttl + tcp_flags
4. **ES retención** — cambiar `opnsense-logs-*` a 30d + crear políticas para obs-ai-*
5. **collector.py** — aggregations L3 + L2
6. **analyzer.py** — llamadas LLM + parsing JSON response
7. **notifier.py** — Ntfy + Grafana annotations
8. **indexer.py** — escritura obs-ai-* 
9. **scheduler.py** — asyncio loops (5min + 03:00 diario)
10. **Quadlet + config.yaml** — containerizar y deployar

---

## Dependencias Python

```
elasticsearch>=8.15
httpx>=0.27          # async HTTP para LLM APIs + Ntfy + Grafana
pyyaml>=6.0
```

Sin dependencias pesadas — el servicio es liviano, toda la IA vive en los servidores externos (:8100/:8200).

---

## Criterios de éxito

- [ ] Logs L2 del MikroTik indexando en `mikrotik-logs-*`
- [ ] Logs OPNsense con campos geo + ttl + tcp_flags
- [ ] AlertLoop corriendo cada 5 min sin errores por 24h
- [ ] Al menos 1 anomalía real detectada y notificada via Ntfy
- [ ] Reporte diario indexado y visible en Kibana
- [ ] Anotaciones visibles en Grafana dashboard
