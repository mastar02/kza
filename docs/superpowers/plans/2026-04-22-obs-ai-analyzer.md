# OBS AI Analyzer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Desplegar un servicio Python que analiza logs L2 (MikroTik) + L3 (OPNsense) vía LLMs locales, alertando por Ntfy cada 5 min y generando un reporte diario en ES + Grafana.

**Architecture:** `obs-analyzer.service` (Quadlet rootless, usuario `obs`) con dos loops asyncio: AlertLoop (Qwen 7B :8100, cada 5 min) y ReportLoop (Qwen 72B :8200, 03:00 UTC). El collector agrega estadísticas en ES antes de enviar ~500 tokens al modelo — nunca logs crudos.

**Tech Stack:** Python 3.12, elasticsearch-py 8.x async, httpx async, pysnmp (o easysnmp), pyyaml, pytest, Podman Quadlet, Logstash 8.15.5, Elasticsearch 8.15.5.

---

## Archivos a crear / modificar

```
SERVIDOR /home/obs/app/
├── logstash/pipeline/
│   ├── opnsense.conf          MOD: + geoip, asn, ttl, tcp_flags
│   └── mikrotik.conf          CREATE: grok RouterOS syslog L2
└── analyzer/
    ├── src/
    │   ├── collector.py       CREATE: ES aggregations L3+L2
    │   ├── analyzer.py        CREATE: LLM calls + build_alert_prompt
    │   ├── notifier.py        CREATE: Ntfy + Grafana annotations
    │   ├── indexer.py         CREATE: write obs-ai-* indices
    │   └── scheduler.py       CREATE: asyncio entry point
    ├── tests/
    │   ├── conftest.py        CREATE: fixtures mock ES + httpx
    │   ├── test_collector.py  CREATE
    │   ├── test_analyzer.py   CREATE
    │   ├── test_notifier.py   CREATE
    │   └── test_indexer.py    CREATE
    ├── config.yaml            CREATE: umbrales, URLs, topics
    ├── requirements.txt       CREATE
    └── Containerfile          CREATE

SERVIDOR /home/obs/.config/containers/systemd/
└── obs-analyzer.container     CREATE: Quadlet rootless
```

---

## Task 1: MikroTik — SNMP + syslog remoto ✅ COMPLETADO

**Prerequisito bloqueante.** Completado 2026-04-22.

**Hallazgo clave:** El CRS310 usa ASIC Marvell-98DX226S — bridge filter NO intercepta tráfico forwarded en hardware. Se usa SNMP polling en su lugar.

**Config aplicada en 192.168.1.5:**
- Bridge: `bridgeLocal` (no `bridge`)
- SNMP: habilitado, community `obsread` restringida a 192.168.1.2/32
- Syslog: action `remoteobs` → 192.168.1.2:5514 UDP, topics: system + interface + firewall
- Syslog format RouterOS nativo: `{topic1},{topic2} {mensaje}` (sin RFC3164 header)

**SNMP OIDs verificados desde 192.168.1.2:**
- `1.3.6.1.2.1.17.4.3.1` — Bridge FDB (MAC→port): ✅ 20+ MACs visibles
- `1.3.6.1.2.1.4.22.1` — ARP table (IP→MAC): ✅ 9 dispositivos activos
- `1.3.6.1.2.1.2.2.1` — Interface stats: ✅ ether1-8, sfp-sfpplus1-2, bridgeLocal

**Dispositivos LAN detectados (ARP table al 2026-04-22):**
- 192.168.1.1 → 52:54:00:22:E9:CD (gateway OPNsense)
- 192.168.1.2 → 30:C5:99:70:02:59 (servidor)
- 192.168.1.115, .173, .179, .180, .194, .195, .199

- [ ] **Step 4: Habilitar topics de log para syslog remoto**

```bash
ssh admin@192.168.1.5 "
/system logging
add topics=firewall action=remote-obs
add topics=interface action=remote-obs
add topics=system action=remote-obs
"
```

- [ ] **Step 5: Verificar que llegan logs a Logstash**

En el servidor, capturar UDP :5514 durante 30 segundos:

```bash
ssh kza@192.168.1.2 "sudo tcpdump -i any udp port 5514 -A -l 2>/dev/null | head -40"
```

Deberían verse líneas con `L2-FWD:` y eventos de interface. Copiar 3-5 líneas de muestra — las necesitarás en Task 2 para ajustar el grok.

- [ ] **Step 6: Verificar volumen — si es excesivo, acotar el filtro**

Si el paso anterior muestra >100 líneas/segundo, acotar el bridge filter para excluir tráfico hacia/desde el gateway:

```bash
ssh admin@192.168.1.5 "
/interface bridge filter
remove [find log-prefix=L2-FWD:]
add chain=forward action=log log=yes log-prefix=L2-FWD: \
    in-bridge=bridge dst-address=!192.168.1.1 src-address=!192.168.1.1 \
    comment=obs-analyzer-filtered
"
```

- [ ] **Step 7: Commit de documentación**

```bash
# En el repo local (kza), documentar el cambio de infra
git add docs/superpowers/plans/2026-04-22-obs-ai-analyzer.md
git commit -m "docs: add obs-ai-analyzer implementation plan"
```

---

## Task 2: Logstash — mikrotik.conf (parser L2)

**Files:**
- Create: `/home/obs/app/logstash/pipeline/mikrotik.conf`

- [ ] **Step 1: Crear el pipeline con grok basado en las muestras del Task 1**

Usando las líneas capturadas en Task 1 Step 5, crear `/home/obs/app/logstash/pipeline/mikrotik.conf`:

```ruby
input {
  udp {
    port => 5514
    type => "mikrotik"
    tags => ["mikrotik"]
  }
}

filter {
  if "mikrotik" in [tags] {

    # Bridge firewall forward log (L2-FWD prefix)
    grok {
      match => { "message" => [
        "%{SYSLOGTIMESTAMP:log_timestamp} %{HOSTNAME:switch_host} %{WORD:topic},%{WORD:log_level} %{DATA:log_prefix}: in:%{WORD:in_interface} out:%{WORD:out_interface} src-mac %{MAC:src_mac}(?:, dst-mac %{MAC:dst_mac})?, proto %{WORD:l2_protocol}(?: \(%{WORD:tcp_flags}\))?, %{IP:src_ip}:%{INT:src_port}->%{IP:dst_ip}:%{INT:dst_port}, len %{INT:pkt_len}",
        "%{SYSLOGTIMESTAMP:log_timestamp} %{HOSTNAME:switch_host} interface,%{WORD:log_level} %{WORD:if_name} link %{WORD:link_state}(?: \(%{GREEDYDATA:link_detail}\))?",
        "%{SYSLOGTIMESTAMP:log_timestamp} %{HOSTNAME:switch_host} %{WORD:topic},%{WORD:log_level} %{GREEDYDATA:raw_message}"
      ] }
      tag_on_failure => ["_mikrotik_grok_fail"]
    }

    # Normalizar timestamp
    date {
      match => ["log_timestamp", "MMM  d HH:mm:ss", "MMM dd HH:mm:ss"]
      target => "@timestamp"
      timezone => "UTC"
    }

    # Convertir tipos
    if [src_port] { mutate { convert => { "src_port" => "integer" "dst_port" => "integer" "pkt_len" => "integer" } } }

    mutate { remove_field => ["log_timestamp", "message"] }
  }
}

output {
  if "mikrotik" in [tags] {
    elasticsearch {
      hosts => ["http://127.0.0.1:9200"]
      index => "mikrotik-logs-%{+YYYY.MM.dd}"
      user => "${ES_USER}"
      password => "${ES_PASSWORD}"
    }
  }
}
```

- [ ] **Step 2: Verificar que no hay conflicto con opnsense.conf en el input UDP**

El puerto 5514 ya es usado por `opnsense.conf`. Logstash puede tener un único input UDP por puerto. Verificar:

```bash
ssh kza@192.168.1.2 "grep -r '5514' /home/obs/app/logstash/pipeline/"
```

Si `opnsense.conf` ya tiene `udp { port => 5514 }`, **no duplicar el input**. En cambio, mover el input a un archivo compartido o discriminar por `host`:

```ruby
# En mikrotik.conf — reemplazar el bloque input por un filter condicional
# y agregar en opnsense.conf un input compartido o usar el mismo input
# discriminando por switch_host:

filter {
  if [host] == "192.168.1.5" or [host] == "MikroTik" {
    mutate { add_tag => ["mikrotik"] }
  }
}
```

Ajustar según lo que revele el grep.

- [ ] **Step 3: Reiniciar Logstash y verificar**

```bash
ssh kza@192.168.1.2 "sudo -u obs systemctl --user restart obs-logstash.service && sleep 5 && sudo -u obs systemctl --user status obs-logstash.service"
```

- [ ] **Step 4: Verificar índice en ES**

```bash
ssh kza@192.168.1.2 "curl -s -u elastic:\$ES_PASSWORD 'http://127.0.0.1:9200/mikrotik-logs-*/_count' | python3 -m json.tool"
```

Esperado: `"count": >0` después de ~60 segundos.

- [ ] **Step 5: Verificar campos parseados**

```bash
ssh kza@192.168.1.2 "curl -s -u elastic:\$ES_PASSWORD 'http://127.0.0.1:9200/mikrotik-logs-*/_search?size=2&pretty' | grep -E 'src_ip|src_mac|in_interface|l2_protocol|link_state'"
```

- [ ] **Step 6: Commit**

```bash
# Guardar mikrotik.conf en el repo (si obs/app está en git)
ssh kza@192.168.1.2 "cd /home/obs/app && git add logstash/pipeline/mikrotik.conf && git commit -m 'feat: add MikroTik L2 Logstash pipeline'"
```

---

## Task 3: Logstash — enriquecer opnsense.conf (geoip + asn + ttl + tcp_flags)

**Files:**
- Modify: `/home/obs/app/logstash/pipeline/opnsense.conf`

- [ ] **Step 1: Ver la estructura actual del pipeline**

```bash
ssh kza@192.168.1.2 "cat /home/obs/app/logstash/pipeline/opnsense.conf"
```

Identificar: dónde está el grok/csv filter, qué campos ya extrae.

- [ ] **Step 2: Extender el csv filter para capturar ttl, pkt_len y tcp_flags**

El filterlog de OPNsense es un CSV posicional. Los campos clave:

```
pos 0:  rule_num
pos 4:  interface     ← ya extraído
pos 6:  action        ← ya extraído
pos 7:  direction     ← ya extraído
pos 8:  ip_version
pos 11: ttl           ← NUEVO
pos 16: protocol      ← ya extraído
pos 17: pkt_len       ← NUEVO
pos 18: src_ip        ← ya extraído
pos 19: dst_ip        ← ya extraído
pos 20: src_port      ← ya extraído
pos 21: dst_port      ← ya extraído
pos 23: tcp_flags     ← NUEVO (solo para TCP)
```

Agregar después del bloque csv/grok existente:

```ruby
# En el filter block, después del csv/grok que extrae los campos base:

# Enriquecimiento GeoIP (requiere MaxMind GeoLite2-City.mmdb)
geoip {
  source => "src_ip"
  target => "geo"
  tag_on_failure => ["_geoip_fail"]
}

# Enriquecimiento ASN (requiere MaxMind GeoLite2-ASN.mmdb)
geoip {
  source => "src_ip"
  target => "asn"
  database => "/usr/share/GeoIP/GeoLite2-ASN.mmdb"
  tag_on_failure => ["_asn_fail"]
  fields => ["autonomous_system_number", "autonomous_system_organization"]
}

# Convertir tipos para campos nuevos
mutate {
  convert => {
    "ttl"     => "integer"
    "pkt_len" => "integer"
  }
}
```

Si el CSV filter ya usa columnas nombradas, agregar `ttl`, `pkt_len`, `tcp_flags` a la lista de columnas en el orden correcto. Si usa grok con campos posicionales, extender el patrón.

- [ ] **Step 3: Verificar que existen las bases de datos MaxMind**

```bash
ssh kza@192.168.1.2 "ls -la /usr/share/GeoIP/*.mmdb"
```

Si faltan, instalar:

```bash
ssh kza@192.168.1.2 "sudo apt-get install -y geoipupdate && sudo geoipupdate"
```

O descargar manualmente desde MaxMind (requiere cuenta gratuita) y copiar a `/usr/share/GeoIP/`.

- [ ] **Step 4: Reiniciar Logstash y verificar campos nuevos**

```bash
ssh kza@192.168.1.2 "sudo -u obs systemctl --user restart obs-logstash.service && sleep 10"
ssh kza@192.168.1.2 "curl -s -u elastic:\$ES_PASSWORD 'http://127.0.0.1:9200/opnsense-logs-*/_search?size=1&pretty' | grep -E 'geo|asn|ttl|tcp_flags|pkt_len'"
```

- [ ] **Step 5: Commit**

```bash
ssh kza@192.168.1.2 "cd /home/obs/app && git add logstash/pipeline/opnsense.conf && git commit -m 'feat: enrich opnsense logs with geoip, asn, ttl, tcp_flags'"
```

---

## Task 4: ES — políticas ILM y retención

**Files:**
- Modifica políticas ILM vía API REST de ES

- [ ] **Step 1: Cambiar retención opnsense-logs a 30 días**

```bash
ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_ilm/policy/logs-30d' \
  -H 'Content-Type: application/json' \
  -d '{\"policy\":{\"phases\":{\"delete\":{\"min_age\":\"30d\",\"actions\":{\"delete\":{}}}}}}'  | python3 -m json.tool"
```

- [ ] **Step 2: Reasignar template de opnsense-logs a política 30d**

```bash
ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_index_template/opnsense-logs' \
  -H 'Content-Type: application/json' \
  -d '{
    \"index_patterns\": [\"opnsense-logs-*\"],
    \"priority\": 600,
    \"template\": {
      \"settings\": {
        \"index.lifecycle.name\": \"logs-30d\"
      }
    }
  }' | python3 -m json.tool"
```

- [ ] **Step 3: Crear política para mikrotik-logs (30 días)**

```bash
ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_index_template/mikrotik-logs' \
  -H 'Content-Type: application/json' \
  -d '{
    \"index_patterns\": [\"mikrotik-logs-*\"],
    \"priority\": 600,
    \"template\": {
      \"settings\": {\"index.lifecycle.name\": \"logs-30d\"}
    }
  }' | python3 -m json.tool"
```

- [ ] **Step 4: Crear política para obs-ai-analysis (30 días)**

```bash
ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_index_template/obs-ai-analysis' \
  -H 'Content-Type: application/json' \
  -d '{
    \"index_patterns\": [\"obs-ai-analysis-*\"],
    \"priority\": 600,
    \"template\": {
      \"settings\": {\"index.lifecycle.name\": \"logs-30d\"}
    }
  }' | python3 -m json.tool"
```

- [ ] **Step 5: Crear política para obs-ai-reports (90 días)**

```bash
ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_ilm/policy/logs-90d' \
  -H 'Content-Type: application/json' \
  -d '{\"policy\":{\"phases\":{\"delete\":{\"min_age\":\"90d\",\"actions\":{\"delete\":{}}}}}}' | python3 -m json.tool"

ssh kza@192.168.1.2 "curl -s -X PUT -u elastic:\$ES_PASSWORD \
  'http://127.0.0.1:9200/_index_template/obs-ai-reports' \
  -H 'Content-Type: application/json' \
  -d '{
    \"index_patterns\": [\"obs-ai-reports-*\"],
    \"priority\": 600,
    \"template\": {
      \"settings\": {\"index.lifecycle.name\": \"logs-90d\"}
    }
  }' | python3 -m json.tool"
```

- [ ] **Step 6: Verificar políticas**

```bash
ssh kza@192.168.1.2 "curl -s -u elastic:\$ES_PASSWORD 'http://127.0.0.1:9200/_ilm/policy/logs-30d,logs-90d' | python3 -m json.tool"
```

---

## Task 5: Scaffold del proyecto Python

**Files:**
- Create: `/home/obs/app/analyzer/requirements.txt`
- Create: `/home/obs/app/analyzer/config.yaml`
- Create: `/home/obs/app/analyzer/Containerfile`
- Create: `/home/obs/app/analyzer/tests/conftest.py`

- [ ] **Step 1: Crear estructura de directorios**

```bash
ssh kza@192.168.1.2 "mkdir -p /home/obs/app/analyzer/{src,tests}"
```

- [ ] **Step 2: Crear requirements.txt**

```
elasticsearch[async]==8.15.2
httpx==0.27.2
pyyaml==6.0.2
pytest==8.3.2
pytest-asyncio==0.24.0
pytest-mock==3.14.0
```

- [ ] **Step 3: Crear config.yaml**

```yaml
elasticsearch:
  url: "http://127.0.0.1:9200"
  user: "elastic"
  password: ""  # set via ES_PASSWORD env var

llm:
  fast:
    url: "http://127.0.0.1:8100/v1/chat/completions"
    model: "qwen-7b-awq"
    timeout_seconds: 30
  deep:
    url: "http://127.0.0.1:8200/v1/chat/completions"
    model: "qwen-72b-q8"
    timeout_seconds: 120

alerting:
  anomaly_threshold: 0.75
  ntfy:
    url: "https://ntfy.sh"
    topic: "kza-security"
    token: ""  # set via NTFY_TOKEN env var
  grafana:
    url: "http://127.0.0.1:3000"
    api_key: ""  # set via GRAFANA_API_KEY env var

schedule:
  alert_interval_minutes: 5
  alert_window_minutes: 5
  report_hour_utc: 3
```

- [ ] **Step 4: Crear Containerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY config.yaml .
CMD ["python", "-m", "src.scheduler"]
```

- [ ] **Step 5: Crear tests/conftest.py**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_es():
    es = AsyncMock()
    es.search = AsyncMock()
    es.count = AsyncMock()
    es.index = AsyncMock()
    return es


@pytest.fixture
def sample_l3_es_response():
    return {
        "hits": {"total": {"value": 847}},
        "aggregations": {
            "top_src_ips": {"buckets": [
                {
                    "key": "185.220.101.5", "doc_count": 312,
                    "countries": {"buckets": [{"key": "DE"}]},
                    "actions": {"buckets": [{"key": "block", "doc_count": 312}]},
                    "dst_ports": {"buckets": [{"key": 22}, {"key": 3389}]},
                    "tcp_flags": {"buckets": [{"key": "S", "doc_count": 310}]},
                    "ttl": {"buckets": [{"key": 64}]}
                }
            ]},
            "dst_ports": {"buckets": [
                {"key": 22, "doc_count": 401},
                {"key": 443, "doc_count": 198}
            ]},
            "action_dist": {"buckets": [
                {"key": "block", "doc_count": 610},
                {"key": "pass", "doc_count": 237}
            ]},
            "protocol_dist": {"buckets": [
                {"key": "tcp", "doc_count": 720},
                {"key": "udp", "doc_count": 127}
            ]},
            "country_dist": {"buckets": [
                {
                    "key": "DE", "doc_count": 320,
                    "block_count": {"doc_count": 314}
                }
            ]}
        }
    }


@pytest.fixture
def sample_l2_es_response():
    return {
        "hits": {"total": {"value": 45}},
        "aggregations": {
            "mac_moves": {"buckets": [
                {
                    "key": "aa:bb:cc:dd:ee:ff",
                    "doc_count": 12,
                    "ports": {"buckets": [{"key": "ether3"}, {"key": "ether7"}]}
                }
            ]},
            "port_events": {"doc_count": 3, "ports": {"buckets": []}}
        }
    }


@pytest.fixture
def sample_config():
    return {
        "elasticsearch": {"url": "http://127.0.0.1:9200", "user": "elastic", "password": "test"},
        "llm": {
            "fast": {"url": "http://127.0.0.1:8100/v1/chat/completions", "model": "qwen-7b", "timeout_seconds": 30},
            "deep": {"url": "http://127.0.0.1:8200/v1/chat/completions", "model": "qwen-72b", "timeout_seconds": 120}
        },
        "alerting": {
            "anomaly_threshold": 0.75,
            "ntfy": {"url": "https://ntfy.sh", "topic": "kza-security", "token": "test"},
            "grafana": {"url": "http://127.0.0.1:3000", "api_key": "test"}
        }
    }
```

- [ ] **Step 6: Verificar que pytest corre**

```bash
cd /home/obs/app/analyzer && pip install -r requirements.txt && pytest tests/ -v
```

Esperado: 0 tests collected, 0 errors (no hay tests todavía, solo conftest).

- [ ] **Step 7: Commit**

```bash
cd /home/obs/app/analyzer && git add . && git commit -m "feat: scaffold obs-analyzer project"
```

---

## Task 6: collector.py — aggregaciones ES

**Files:**
- Create: `/home/obs/app/analyzer/src/collector.py`
- Create: `/home/obs/app/analyzer/tests/test_collector.py`

- [ ] **Step 1: Escribir tests**

`tests/test_collector.py`:

```python
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, call
from src.collector import fetch_alert_stats, build_stats


@pytest.mark.asyncio
async def test_fetch_alert_stats_returns_merged_l3_l2(
    mock_es, sample_l3_es_response, sample_l2_es_response, sample_config
):
    mock_es.search = AsyncMock(side_effect=[
        sample_l3_es_response,    # l3 main query
        {"aggregations": {"ips": {"buckets": []}}},  # known IPs 1h
        {"aggregations": {"ips": {"buckets": []}}},  # known IPs 7d
        sample_l2_es_response,    # l2 query
        {"aggregations": {"macs": {"buckets": []}}},  # known MACs 24h
    ])
    mock_es.count = AsyncMock(return_value={"count": 2520})

    stats = await fetch_alert_stats(mock_es, window_minutes=5)

    assert stats["total_events"] == 847
    assert stats["baseline_avg"] == 210.0
    assert stats["spike_ratio"] == pytest.approx(4.03, rel=0.01)
    assert stats["top_src_ips"][0]["ip"] == "185.220.101.5"
    assert stats["top_src_ips"][0]["country"] == "DE"
    assert stats["top_src_ips"][0]["tcp_flags"] == {"S": 310}
    assert stats["l2_mac_moves"][0]["mac"] == "aa:bb:cc:dd:ee:ff"
    assert stats["l2_mac_moves"][0]["ports"] == ["ether3", "ether7"]


@pytest.mark.asyncio
async def test_new_ips_unseen_detected(mock_es, sample_l3_es_response, sample_l2_es_response):
    mock_es.search = AsyncMock(side_effect=[
        sample_l3_es_response,
        {"aggregations": {"ips": {"buckets": [{"key": "1.2.3.4"}]}}},  # known 1h — nueva IP no está
        {"aggregations": {"ips": {"buckets": []}}},
        sample_l2_es_response,
        {"aggregations": {"macs": {"buckets": []}}},
    ])
    mock_es.count = AsyncMock(return_value={"count": 2520})

    stats = await fetch_alert_stats(mock_es, window_minutes=5)

    assert "185.220.101.5" in stats["new_src_ips_unseen_1h"]


@pytest.mark.asyncio
async def test_spike_ratio_no_division_by_zero(mock_es, sample_l3_es_response, sample_l2_es_response):
    mock_es.count = AsyncMock(return_value={"count": 0})
    mock_es.search = AsyncMock(side_effect=[
        sample_l3_es_response,
        {"aggregations": {"ips": {"buckets": []}}},
        {"aggregations": {"ips": {"buckets": []}}},
        sample_l2_es_response,
        {"aggregations": {"macs": {"buckets": []}}},
    ])

    stats = await fetch_alert_stats(mock_es, window_minutes=5)
    assert stats["spike_ratio"] == 1.0
```

- [ ] **Step 2: Correr tests — deben fallar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_collector.py -v
```

Esperado: `ModuleNotFoundError: No module named 'src.collector'`

- [ ] **Step 3: Implementar collector.py**

`src/collector.py`:

```python
import logging
from datetime import datetime, timedelta, timezone
from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

SENSITIVE_PORTS = {22, 23, 25, 3389, 5900, 445, 1433, 3306, 5432, 6379, 27017}


async def fetch_alert_stats(es: AsyncElasticsearch, window_minutes: int = 5) -> dict:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=window_minutes)
    baseline_start = now - timedelta(hours=1)

    l3 = await _fetch_l3_stats(es, window_start, now, baseline_start)
    l2 = await _fetch_l2_stats(es, window_start, now)
    return {"window_minutes": window_minutes, **l3, **l2}


async def _fetch_l3_stats(es, window_start, window_end, baseline_start):
    resp = await es.search(
        index="opnsense-logs-*",
        body={
            "query": {"range": {"@timestamp": {
                "gte": window_start.isoformat(), "lte": window_end.isoformat()
            }}},
            "size": 0,
            "aggs": {
                "top_src_ips": {
                    "terms": {"field": "src_ip.keyword", "size": 20},
                    "aggs": {
                        "countries": {"terms": {"field": "geo.country_code.keyword", "size": 1}},
                        "actions": {"terms": {"field": "action.keyword"}},
                        "dst_ports": {"terms": {"field": "dst_port", "size": 10}},
                        "tcp_flags": {"terms": {"field": "tcp_flags.keyword", "size": 5}},
                        "ttl": {"terms": {"field": "ttl", "size": 3}},
                    },
                },
                "dst_ports": {"terms": {"field": "dst_port", "size": 20}},
                "action_dist": {"terms": {"field": "action.keyword"}},
                "protocol_dist": {"terms": {"field": "protocol.keyword"}},
                "country_dist": {
                    "terms": {"field": "geo.country_code.keyword", "size": 20},
                    "aggs": {"block_count": {"filter": {"term": {"action.keyword": "block"}}}},
                },
            },
        },
    )

    baseline_resp = await es.count(
        index="opnsense-logs-*",
        body={"query": {"range": {"@timestamp": {
            "gte": baseline_start.isoformat(), "lte": window_end.isoformat()
        }}}},
    )
    baseline_count = baseline_resp["count"]
    baseline_avg = baseline_count / 12

    known_1h = await _known_ips(es, window_start - timedelta(hours=1), window_start)
    known_7d = await _known_ips(es, window_end - timedelta(days=7), window_start)

    aggs = resp["aggregations"]
    total = resp["hits"]["total"]["value"]

    top_src_ips = [
        {
            "ip": b["key"],
            "count": b["doc_count"],
            "country": b["countries"]["buckets"][0]["key"] if b["countries"]["buckets"] else "unknown",
            "action_dist": {x["key"]: x["doc_count"] for x in b["actions"]["buckets"]},
            "dst_ports": [x["key"] for x in b["dst_ports"]["buckets"]],
            "tcp_flags": {x["key"]: x["doc_count"] for x in b["tcp_flags"]["buckets"]},
            "ttl_mode": b["ttl"]["buckets"][0]["key"] if b["ttl"]["buckets"] else None,
        }
        for b in aggs["top_src_ips"]["buckets"]
    ]

    current_ips = {ip["ip"] for ip in top_src_ips}

    country_dist = [
        {
            "country": b["key"],
            "count": b["doc_count"],
            "action_block_pct": int(b["block_count"]["doc_count"] / b["doc_count"] * 100)
            if b["doc_count"] > 0 else 0,
        }
        for b in aggs["country_dist"]["buckets"]
    ]

    return {
        "total_events": total,
        "baseline_avg": round(baseline_avg, 1),
        "spike_ratio": round(total / baseline_avg, 2) if baseline_avg > 0 else 1.0,
        "top_src_ips": top_src_ips,
        "new_src_ips_unseen_1h": list(current_ips - known_1h),
        "new_src_ips_unseen_7d": list(current_ips - known_7d),
        "dst_ports_top": [
            {"port": b["key"], "count": b["doc_count"], "is_sensitive": b["key"] in SENSITIVE_PORTS}
            for b in aggs["dst_ports"]["buckets"]
        ],
        "action_dist": {b["key"]: b["doc_count"] for b in aggs["action_dist"]["buckets"]},
        "protocol_dist": {b["key"]: b["doc_count"] for b in aggs["protocol_dist"]["buckets"]},
        "country_dist": country_dist,
    }


async def _known_ips(es, gte, lte):
    resp = await es.search(
        index="opnsense-logs-*",
        body={
            "query": {"range": {"@timestamp": {"gte": gte.isoformat(), "lte": lte.isoformat()}}},
            "size": 0,
            "aggs": {"ips": {"terms": {"field": "src_ip.keyword", "size": 10000}}},
        },
    )
    return {b["key"] for b in resp["aggregations"]["ips"]["buckets"]}


async def _fetch_l2_stats(es, window_start, window_end):
    resp = await es.search(
        index="mikrotik-logs-*",
        body={
            "query": {"range": {"@timestamp": {"gte": window_start.isoformat(), "lte": window_end.isoformat()}}},
            "size": 0,
            "aggs": {
                "mac_moves": {
                    "terms": {"field": "src_mac.keyword", "size": 100},
                    "aggs": {"ports": {"terms": {"field": "in_interface.keyword", "size": 10}}},
                },
                "port_events": {
                    "filter": {"term": {"topic.keyword": "interface"}},
                    "aggs": {"ports": {"terms": {"field": "if_name.keyword", "size": 20}}},
                },
            },
        },
    )
    aggs = resp.get("aggregations", {})

    mac_moves = [
        {"mac": b["key"], "ports": [p["key"] for p in b["ports"]["buckets"]], "count": b["doc_count"]}
        for b in aggs.get("mac_moves", {}).get("buckets", [])
        if len(b["ports"]["buckets"]) > 1
    ]

    known_macs_resp = await es.search(
        index="mikrotik-logs-*",
        body={
            "query": {"range": {"@timestamp": {
                "gte": (window_end - timedelta(hours=24)).isoformat(),
                "lte": window_start.isoformat(),
            }}},
            "size": 0,
            "aggs": {"macs": {"terms": {"field": "src_mac.keyword", "size": 1000}}},
        },
    )
    known_macs = {b["key"] for b in known_macs_resp.get("aggregations", {}).get("macs", {}).get("buckets", [])}
    current_macs = {b["key"] for b in aggs.get("mac_moves", {}).get("buckets", [])}

    return {
        "l2_mac_moves": mac_moves,
        "l2_new_macs_unseen_24h": list(current_macs - known_macs),
        "l2_total_events": resp["hits"]["total"]["value"],
    }
```

- [ ] **Step 4: Correr tests — deben pasar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_collector.py -v
```

Esperado: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/obs/app/analyzer && git add src/collector.py tests/test_collector.py && git commit -m "feat: add collector with L3+L2 ES aggregations"
```

---

## Task 7: analyzer.py — LLM calls + build_alert_prompt

**Files:**
- Create: `/home/obs/app/analyzer/src/analyzer.py`
- Create: `/home/obs/app/analyzer/tests/test_analyzer.py`

- [ ] **Step 1: Escribir tests**

`tests/test_analyzer.py`:

```python
import pytest
import json
from unittest.mock import AsyncMock, patch
from src.analyzer import build_alert_prompt, call_llm, parse_llm_response


def test_build_alert_prompt_includes_spike_ratio():
    stats = {
        "window_minutes": 5,
        "total_events": 847,
        "baseline_avg": 210.0,
        "spike_ratio": 4.03,
        "top_src_ips": [
            {"ip": "185.220.101.5", "country": "DE", "asn_org": "Tor Project",
             "count": 312, "dst_ports": [22, 3389], "tcp_flags": {"S": 310}, "ttl_mode": 64,
             "action_dist": {"block": 312}}
        ],
        "new_src_ips_unseen_7d": ["185.220.101.5"],
        "new_src_ips_unseen_1h": ["185.220.101.5"],
        "dst_ports_top": [{"port": 22, "count": 401, "is_sensitive": True}],
        "country_dist": [{"country": "DE", "count": 320, "action_block_pct": 98}],
        "l2_mac_moves": [],
        "l2_new_macs_unseen_24h": [],
        "l2_total_events": 0,
        "outbound_new_dst_ips": [],
    }
    prompt = build_alert_prompt(stats)

    assert "spike_ratio=4.03" in prompt
    assert "185.220.101.5" in prompt
    assert "SSH" in prompt
    assert "never seen in 7 days" in prompt


def test_build_alert_prompt_includes_l2_mac_moves():
    stats = {
        "window_minutes": 5, "total_events": 50, "baseline_avg": 50.0, "spike_ratio": 1.0,
        "top_src_ips": [], "new_src_ips_unseen_7d": [], "new_src_ips_unseen_1h": [],
        "dst_ports_top": [], "country_dist": [],
        "l2_mac_moves": [{"mac": "aa:bb:cc:dd:ee:ff", "ports": ["ether3", "ether7"], "count": 12}],
        "l2_new_macs_unseen_24h": [],
        "l2_total_events": 12,
        "outbound_new_dst_ips": [],
    }
    prompt = build_alert_prompt(stats)
    assert "aa:bb:cc:dd:ee:ff" in prompt
    assert "MAC SPOOFING" in prompt


@pytest.mark.asyncio
async def test_call_llm_returns_parsed_json():
    mock_response = {
        "choices": [{"message": {"content": json.dumps({
            "anomaly_score": 0.87,
            "anomaly_type": "port_scan",
            "reasons": ["spike 4x baseline"],
            "recommendation": "block IP"
        })}}]
    }
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=AsyncMock(json=AsyncMock(return_value=mock_response), raise_for_status=AsyncMock())
        )
        result = await call_llm(
            url="http://127.0.0.1:8100/v1/chat/completions",
            model="qwen-7b",
            system_prompt="You are a security analyst.",
            user_message="test",
            timeout=30,
        )
    assert result["anomaly_score"] == 0.87
    assert result["anomaly_type"] == "port_scan"


def test_parse_llm_response_handles_malformed_json():
    result = parse_llm_response("not valid json at all {{{")
    assert result["anomaly_score"] == 0.0
    assert result["anomaly_type"] == "none"
```

- [ ] **Step 2: Correr tests — deben fallar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_analyzer.py -v
```

Esperado: `ModuleNotFoundError: No module named 'src.analyzer'`

- [ ] **Step 3: Implementar analyzer.py**

`src/analyzer.py`:

```python
import json
import logging
import httpx

logger = logging.getLogger(__name__)

ALERT_SYSTEM_PROMPT = """You are a network security analyst reviewing firewall logs.
Analyze the provided traffic statistics and respond ONLY with valid JSON:
{
  "anomaly_score": <float 0.0-1.0>,
  "anomaly_type": <"port_scan"|"volume_spike"|"new_ip_suspicious"|"exfiltration_hint"|"mac_spoofing"|"unknown_device"|"port_flapping"|"lateral_scan"|"switch_mgmt_attack"|"none">,
  "reasons": [<list of strings, max 3>],
  "recommendation": <string or null>
}
Score guide: 0.0=normal, 0.5=worth noting, 0.75=alert, 1.0=critical.
Be conservative — only score >0.75 for clear threats. Respond with JSON only, no other text."""

PORT_LABELS = {22: "SSH", 23: "Telnet", 25: "SMTP", 3389: "RDP", 5900: "VNC",
               445: "SMB", 1433: "MSSQL", 3306: "MySQL", 5432: "PostgreSQL",
               6379: "Redis", 27017: "MongoDB"}


def build_alert_prompt(stats: dict) -> str:
    lines = []

    lines.append(
        f"Network window: last {stats['window_minutes']} minutes — "
        f"{stats['total_events']} events (baseline {stats['baseline_avg']}/window, "
        f"spike_ratio={stats['spike_ratio']:.2f}x)."
    )

    if stats["top_src_ips"]:
        lines.append("Top source IPs:")
        for ip_info in stats["top_src_ips"][:5]:
            flags_str = ", ".join(f"{k}={v}" for k, v in ip_info.get("tcp_flags", {}).items())
            all_blocked = ip_info["action_dist"].get("pass", 0) == 0
            lines.append(
                f"  {ip_info['ip']} ({ip_info.get('country','?')}, "
                f"{ip_info.get('asn_org', 'unknown ASN')}): {ip_info['count']} events, "
                f"ports={ip_info['dst_ports']}, flags=[{flags_str}], "
                f"TTL={ip_info.get('ttl_mode','?')}, all_blocked={all_blocked}"
            )

    sensitive = [p for p in stats["dst_ports_top"] if p.get("is_sensitive")]
    if sensitive:
        labeled = [f"{PORT_LABELS.get(p['port'], str(p['port']))}({p['port']})×{p['count']}" for p in sensitive]
        lines.append("Sensitive destination ports hit: " + ", ".join(labeled))

    if stats.get("new_src_ips_unseen_7d"):
        lines.append(f"IPs never seen in 7 days: {stats['new_src_ips_unseen_7d']}")

    suspicious_countries = [
        c for c in stats.get("country_dist", [])
        if c["action_block_pct"] > 90 and c["count"] > 50
    ]
    if suspicious_countries:
        lines.append("High-block-rate countries: " + ", ".join(
            f"{c['country']}({c['count']} events, {c['action_block_pct']}% blocked)"
            for c in suspicious_countries
        ))

    if stats.get("l2_mac_moves"):
        for move in stats["l2_mac_moves"]:
            lines.append(
                f"WARNING MAC SPOOFING DETECTED: {move['mac']} seen on ports {move['ports']} "
                f"({move['count']} events)"
            )

    if stats.get("l2_new_macs_unseen_24h"):
        lines.append(f"New unknown devices (MAC never seen 24h): {stats['l2_new_macs_unseen_24h']}")

    if stats.get("outbound_new_dst_ips"):
        lines.append(f"WARNING — LAN devices contacted new external IPs: {stats['outbound_new_dst_ips']}")

    return "\n".join(lines)


async def call_llm(url: str, model: str, system_prompt: str, user_message: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    return parse_llm_response(content)


def parse_llm_response(content: str) -> dict:
    try:
        # Strip markdown code fences if present
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse LLM response: %s — raw: %.200s", exc, content)
        return {"anomaly_score": 0.0, "anomaly_type": "none", "reasons": [], "recommendation": None}


async def analyze_alert(stats: dict, llm_config: dict) -> dict:
    prompt = build_alert_prompt(stats)
    return await call_llm(
        url=llm_config["fast"]["url"],
        model=llm_config["fast"]["model"],
        system_prompt=ALERT_SYSTEM_PROMPT,
        user_message=prompt,
        timeout=llm_config["fast"]["timeout_seconds"],
    )


async def analyze_report(stats_24h: dict, llm_config: dict) -> dict:
    system = (
        "You are a senior network security analyst. Analyze 24h of firewall traffic statistics. "
        "Provide a structured JSON report with: summary (string), top_threats (list of objects with ip/type/severity), "
        "new_ips_of_interest (list), recommended_blocks (list of IPs), trends (string). "
        "Respond with JSON only."
    )
    prompt = build_alert_prompt(stats_24h)
    return await call_llm(
        url=llm_config["deep"]["url"],
        model=llm_config["deep"]["model"],
        system_prompt=system,
        user_message=prompt,
        timeout=llm_config["deep"]["timeout_seconds"],
    )
```

- [ ] **Step 4: Correr tests — deben pasar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_analyzer.py -v
```

Esperado: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/obs/app/analyzer && git add src/analyzer.py tests/test_analyzer.py && git commit -m "feat: add LLM analyzer with build_alert_prompt"
```

---

## Task 8: notifier.py — Ntfy + Grafana annotations

**Files:**
- Create: `/home/obs/app/analyzer/src/notifier.py`
- Create: `/home/obs/app/analyzer/tests/test_notifier.py`

- [ ] **Step 1: Escribir tests**

`tests/test_notifier.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.notifier import send_ntfy, post_grafana_annotation


@pytest.mark.asyncio
async def test_send_ntfy_posts_with_correct_headers(sample_config):
    ntfy_cfg = sample_config["alerting"]["ntfy"]
    with patch("httpx.AsyncClient") as mock_client:
        mock_post = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))
        mock_client.return_value.__aenter__.return_value.post = mock_post

        await send_ntfy(ntfy_cfg, title="Security Alert", message="Port scan detected", priority="high")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "kza-security" in call_kwargs[0][0]
        assert call_kwargs[1]["headers"]["Title"] == "Security Alert"
        assert call_kwargs[1]["headers"]["Priority"] == "high"


@pytest.mark.asyncio
async def test_post_grafana_annotation_sends_correct_payload(sample_config):
    grafana_cfg = sample_config["alerting"]["grafana"]
    with patch("httpx.AsyncClient") as mock_client:
        mock_post = AsyncMock(return_value=MagicMock(raise_for_status=MagicMock()))
        mock_client.return_value.__aenter__.return_value.post = mock_post

        await post_grafana_annotation(grafana_cfg, text="Port scan from DE", tags=["security", "port_scan"])

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["text"] == "Port scan from DE"
        assert "security" in payload["tags"]
```

- [ ] **Step 2: Correr tests — deben fallar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_notifier.py -v
```

- [ ] **Step 3: Implementar notifier.py**

`src/notifier.py`:

```python
import logging
import httpx

logger = logging.getLogger(__name__)


async def send_ntfy(ntfy_config: dict, title: str, message: str, priority: str = "default") -> None:
    url = f"{ntfy_config['url'].rstrip('/')}/{ntfy_config['topic']}"
    headers = {"Title": title, "Priority": priority, "Tags": "shield"}
    if ntfy_config.get("token"):
        headers["Authorization"] = f"Bearer {ntfy_config['token']}"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, content=message.encode(), headers=headers)
        resp.raise_for_status()
    logger.info("Ntfy alert sent: %s", title)


async def post_grafana_annotation(grafana_config: dict, text: str, tags: list[str]) -> None:
    url = f"{grafana_config['url'].rstrip('/')}/api/annotations"
    headers = {
        "Authorization": f"Bearer {grafana_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {"text": text, "tags": tags}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    logger.info("Grafana annotation posted: %s", text)


async def notify_anomaly(config: dict, analysis: dict, stats: dict) -> None:
    anomaly_type = analysis.get("anomaly_type", "none")
    score = analysis.get("anomaly_score", 0.0)
    reasons = "; ".join(analysis.get("reasons", []))
    title = f"[{anomaly_type.upper()}] Score {score:.2f}"
    message = f"{reasons}\nRecommendation: {analysis.get('recommendation', 'none')}"

    priority = "urgent" if score >= 0.9 else "high"

    await send_ntfy(config["alerting"]["ntfy"], title=title, message=message, priority=priority)
    await post_grafana_annotation(
        config["alerting"]["grafana"],
        text=f"🔴 {title} — {reasons}",
        tags=["security", anomaly_type, "obs-analyzer"],
    )
```

- [ ] **Step 4: Correr tests — deben pasar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_notifier.py -v
```

Esperado: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/obs/app/analyzer && git add src/notifier.py tests/test_notifier.py && git commit -m "feat: add Ntfy + Grafana notifier"
```

---

## Task 9: indexer.py — escritura ES

**Files:**
- Create: `/home/obs/app/analyzer/src/indexer.py`
- Create: `/home/obs/app/analyzer/tests/test_indexer.py`

- [ ] **Step 1: Escribir tests**

`tests/test_indexer.py`:

```python
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from src.indexer import index_alert_result, index_daily_report


@pytest.mark.asyncio
async def test_index_alert_result_uses_correct_index(mock_es):
    analysis = {"anomaly_score": 0.87, "anomaly_type": "port_scan", "reasons": [], "recommendation": None}
    stats = {"total_events": 100, "window_minutes": 5}
    ts = datetime(2026, 4, 22, 10, 0, 0, tzinfo=timezone.utc)

    await index_alert_result(mock_es, analysis=analysis, stats=stats, timestamp=ts)

    mock_es.index.assert_called_once()
    call_kwargs = mock_es.index.call_args[1]
    assert call_kwargs["index"] == "obs-ai-analysis-2026.04.22"
    assert call_kwargs["document"]["anomaly_score"] == 0.87
    assert call_kwargs["document"]["model"] == "qwen-7b"


@pytest.mark.asyncio
async def test_index_daily_report_uses_correct_index(mock_es):
    report = {"summary": "Quiet day", "top_threats": [], "recommended_blocks": []}
    ts = datetime(2026, 4, 22, 3, 0, 0, tzinfo=timezone.utc)

    await index_daily_report(mock_es, report=report, timestamp=ts)

    call_kwargs = mock_es.index.call_args[1]
    assert call_kwargs["index"] == "obs-ai-reports-2026.04.22"
    assert call_kwargs["document"]["model"] == "qwen-72b"
```

- [ ] **Step 2: Correr tests — deben fallar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_indexer.py -v
```

- [ ] **Step 3: Implementar indexer.py**

`src/indexer.py`:

```python
import logging
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)


async def index_alert_result(
    es: AsyncElasticsearch,
    analysis: dict,
    stats: dict,
    timestamp: datetime | None = None,
) -> None:
    ts = timestamp or datetime.now(timezone.utc)
    doc = {
        "@timestamp": ts.isoformat(),
        "anomaly_score": analysis.get("anomaly_score", 0.0),
        "anomaly_type": analysis.get("anomaly_type", "none"),
        "reasons": analysis.get("reasons", []),
        "recommendation": analysis.get("recommendation"),
        "total_events": stats.get("total_events", 0),
        "spike_ratio": stats.get("spike_ratio", 1.0),
        "model": "qwen-7b",
    }
    index = f"obs-ai-analysis-{ts.strftime('%Y.%m.%d')}"
    await es.index(index=index, document=doc)
    logger.debug("Indexed alert result score=%.2f to %s", doc["anomaly_score"], index)


async def index_daily_report(
    es: AsyncElasticsearch,
    report: dict,
    timestamp: datetime | None = None,
) -> None:
    ts = timestamp or datetime.now(timezone.utc)
    doc = {
        "@timestamp": ts.isoformat(),
        "report_date": ts.strftime("%Y-%m-%d"),
        "model": "qwen-72b",
        **report,
    }
    index = f"obs-ai-reports-{ts.strftime('%Y.%m.%d')}"
    await es.index(index=index, document=doc)
    logger.info("Indexed daily report to %s", index)
```

- [ ] **Step 4: Correr tests — deben pasar**

```bash
cd /home/obs/app/analyzer && pytest tests/test_indexer.py -v
```

Esperado: 2 passed.

- [ ] **Step 5: Correr suite completa**

```bash
cd /home/obs/app/analyzer && pytest tests/ -v
```

Esperado: todos los tests pasan (collector + analyzer + notifier + indexer).

- [ ] **Step 6: Commit**

```bash
cd /home/obs/app/analyzer && git add src/indexer.py tests/test_indexer.py && git commit -m "feat: add ES indexer for analysis results"
```

---

## Task 10: scheduler.py — asyncio entry point

**Files:**
- Create: `/home/obs/app/analyzer/src/scheduler.py`
- Create: `/home/obs/app/analyzer/src/__init__.py`

- [ ] **Step 1: Crear src/__init__.py vacío**

```bash
touch /home/obs/app/analyzer/src/__init__.py
```

- [ ] **Step 2: Implementar scheduler.py**

`src/scheduler.py`:

```python
import asyncio
import logging
import os
from datetime import datetime, timezone

import yaml
from elasticsearch import AsyncElasticsearch

from src.collector import fetch_alert_stats
from src.analyzer import analyze_alert, analyze_report
from src.notifier import notify_anomaly, post_grafana_annotation
from src.indexer import index_alert_result, index_daily_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["elasticsearch"]["password"] = os.environ.get("ES_PASSWORD", cfg["elasticsearch"].get("password", ""))
    cfg["alerting"]["ntfy"]["token"] = os.environ.get("NTFY_TOKEN", cfg["alerting"]["ntfy"].get("token", ""))
    cfg["alerting"]["grafana"]["api_key"] = os.environ.get("GRAFANA_API_KEY", cfg["alerting"]["grafana"].get("api_key", ""))
    return cfg


async def alert_loop(es: AsyncElasticsearch, config: dict) -> None:
    interval = config.get("schedule", {}).get("alert_interval_minutes", 5) * 60
    window = config.get("schedule", {}).get("alert_window_minutes", 5)
    threshold = config["alerting"]["anomaly_threshold"]

    logger.info("AlertLoop started — every %d seconds, threshold=%.2f", interval, threshold)

    while True:
        try:
            stats = await fetch_alert_stats(es, window_minutes=window)
            analysis = await analyze_alert(stats, config["llm"])
            await index_alert_result(es, analysis=analysis, stats=stats)

            score = analysis.get("anomaly_score", 0.0)
            if score >= threshold:
                logger.warning("Anomaly detected: type=%s score=%.2f", analysis.get("anomaly_type"), score)
                await notify_anomaly(config, analysis, stats)
            else:
                logger.debug("AlertLoop cycle OK — score=%.2f", score)

        except Exception:
            logger.exception("AlertLoop error — continuing")

        await asyncio.sleep(interval)


async def report_loop(es: AsyncElasticsearch, config: dict) -> None:
    report_hour = config.get("schedule", {}).get("report_hour_utc", 3)
    logger.info("ReportLoop started — daily at %02d:00 UTC", report_hour)

    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=report_hour, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run.replace(day=next_run.day + 1)
        wait_seconds = (next_run - now).total_seconds()
        logger.info("ReportLoop — next run in %.0f seconds (%s UTC)", wait_seconds, next_run.isoformat())
        await asyncio.sleep(wait_seconds)

        try:
            stats = await fetch_alert_stats(es, window_minutes=1440)
            report = await analyze_report(stats, config["llm"])
            await index_daily_report(es, report=report)
            await post_grafana_annotation(
                config["alerting"]["grafana"],
                text=f"📊 Daily security report — {now.strftime('%Y-%m-%d')}",
                tags=["security", "daily-report", "obs-analyzer"],
            )
            logger.info("Daily report complete")
        except Exception:
            logger.exception("ReportLoop error")


async def main() -> None:
    config = load_config()
    es_cfg = config["elasticsearch"]
    es = AsyncElasticsearch(
        hosts=[es_cfg["url"]],
        basic_auth=(es_cfg["user"], es_cfg["password"]),
    )
    try:
        await asyncio.gather(
            alert_loop(es, config),
            report_loop(es, config),
        )
    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Smoke test local con ES mockeado**

```bash
cd /home/obs/app/analyzer
ES_PASSWORD=test NTFY_TOKEN=test GRAFANA_API_KEY=test timeout 12 python -m src.scheduler 2>&1 | head -20
```

Esperado: líneas de log `AlertLoop started` y `ReportLoop started`, luego errores de conexión a ES (esperado en local sin ES real). Sin crashes por import o syntax.

- [ ] **Step 4: Commit**

```bash
cd /home/obs/app/analyzer && git add src/scheduler.py src/__init__.py && git commit -m "feat: add asyncio scheduler with alert + report loops"
```

---

## Task 11: Quadlet + deploy en servidor

**Files:**
- Create: `/home/obs/.config/containers/systemd/obs-analyzer.container`

- [ ] **Step 1: Build de la imagen en el servidor**

```bash
ssh kza@192.168.1.2 "cd /home/obs/app/analyzer && sudo -u obs podman build -t obs-analyzer:latest ."
```

- [ ] **Step 2: Crear el Quadlet**

`/home/obs/.config/containers/systemd/obs-analyzer.container`:

```ini
[Unit]
Description=OBS AI Network Analyzer
After=network-online.target obs-elasticsearch.service

[Container]
Image=localhost/obs-analyzer:latest
ContainerName=obs-analyzer
Network=obs-internal.network
Environment=ES_PASSWORD=%s
Environment=NTFY_TOKEN=%s
Environment=GRAFANA_API_KEY=%s
EnvironmentFile=/home/obs/secrets/.env
AutoUpdate=local
Restart=on-failure

[Install]
WantedBy=default.target
```

- [ ] **Step 3: Crear archivo de secrets**

```bash
ssh kza@192.168.1.2 "
sudo -u obs sh -c '
cat >> /home/obs/secrets/.env << EOF
ES_PASSWORD=<password_de_elastic>
NTFY_TOKEN=<token_de_ntfy>
GRAFANA_API_KEY=<api_key_de_grafana>
EOF
chmod 600 /home/obs/secrets/.env
'
"
```

Reemplazar los valores placeholder con los reales. El `GRAFANA_API_KEY` se genera en Kibana → Administration → API keys.

- [ ] **Step 4: Habilitar y arrancar el servicio**

```bash
ssh kza@192.168.1.2 "sudo -u obs systemctl --user daemon-reload && sudo -u obs systemctl --user enable --now obs-analyzer.service"
```

- [ ] **Step 5: Verificar que el servicio está corriendo**

```bash
ssh kza@192.168.1.2 "sudo -u obs systemctl --user status obs-analyzer.service"
ssh kza@192.168.1.2 "sudo -u obs journalctl --user -u obs-analyzer.service -n 30 --no-pager"
```

Esperado: `AlertLoop started` y `ReportLoop started` en los logs, sin errores de conexión a ES.

- [ ] **Step 6: Verificar primer ciclo en ES**

Esperar 5 minutos, luego:

```bash
ssh kza@192.168.1.2 "curl -s -u elastic:\$ES_PASSWORD 'http://127.0.0.1:9200/obs-ai-analysis-*/_search?size=3&pretty' | grep -E 'anomaly_score|anomaly_type|@timestamp'"
```

Esperado: documentos con `anomaly_score`, `anomaly_type`, `@timestamp`.

- [ ] **Step 7: Verificar anotación en Grafana**

Abrir `http://192.168.1.2:8081` → cualquier dashboard → verificar que aparecen anotaciones de `obs-analyzer`.

- [ ] **Step 8: Commit final**

```bash
ssh kza@192.168.1.2 "cd /home/obs/app && git add .config/containers/systemd/obs-analyzer.container && git commit -m 'feat: deploy obs-analyzer Quadlet service'"
```

---

## Self-Review

**Cobertura del spec:**
- ✅ Bridge firewall MikroTik como Step 1 bloqueante (Task 1)
- ✅ mikrotik.conf Logstash (Task 2)
- ✅ opnsense.conf geoip + asn + ttl + tcp_flags (Task 3)
- ✅ ES ILM 30d/90d (Task 4)
- ✅ collector.py L3+L2 (Task 6)
- ✅ analyzer.py + build_alert_prompt (Task 7)
- ✅ notifier.py Ntfy + Grafana (Task 8)
- ✅ indexer.py (Task 9)
- ✅ scheduler.py AlertLoop 5min + ReportLoop 03:00 (Task 10)
- ✅ Quadlet deploy (Task 11)
- ✅ Umbral 0.75 configurable en config.yaml
- ✅ 7B para alertas, 72B para reporte diario

**Consistencia de tipos:**
- `fetch_alert_stats` → `dict` con keys `total_events`, `spike_ratio`, `l2_mac_moves` — usados consistentemente en `build_alert_prompt` y `index_alert_result`
- `call_llm` → `dict` con `anomaly_score`, `anomaly_type`, `reasons`, `recommendation` — consistente en `notify_anomaly` e `index_alert_result`

**Sin placeholders:** ningún TBD o TODO en el código.
