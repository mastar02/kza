/**
 * KZA Operator Dashboard
 * Vanilla JS — fetches from /api/* and updates DOM every 5 seconds.
 * Uses safe DOM manipulation (createElement/textContent) — no innerHTML.
 */

"use strict";

const REFRESH_MS = 5000;
const API_ENDPOINTS = {
    health:    "/api/health",
    metrics:   "/api/metrics",
    subsystems:"/api/subsystems",
    failures:  "/api/failures?limit=20",
    reminders: "/api/reminders/status",
    routines:  "/api/routines",
    presence:  "/api/presence",
};

let refreshTimer = null;
let connected = true;

// ==================== Bootstrap ====================

document.addEventListener("DOMContentLoaded", () => {
    refreshAll();
    startAutoRefresh();

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            stopAutoRefresh();
        } else {
            refreshAll();
            startAutoRefresh();
        }
    });
});

function startAutoRefresh() {
    stopAutoRefresh();
    refreshTimer = setInterval(refreshAll, REFRESH_MS);
}

function stopAutoRefresh() {
    if (refreshTimer !== null) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
}

// ==================== DOM Helpers ====================

/** Create an element with optional className, textContent, and style. */
function el(tag, opts) {
    const node = document.createElement(tag);
    if (opts) {
        if (opts.cls) node.className = opts.cls;
        if (opts.text !== undefined) node.textContent = String(opts.text);
        if (opts.style) Object.assign(node.style, opts.style);
    }
    return node;
}

/** Remove all children from an element. */
function clearEl(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
}

/** Append multiple children to a parent. */
function appendAll(parent, children) {
    for (const child of children) {
        parent.appendChild(child);
    }
}

// ==================== Fetch helpers ====================

async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error("HTTP " + response.status);
    }
    return response.json();
}

function setConnected(isConnected) {
    connected = isConnected;
    const dot = document.getElementById("connection-dot");
    const label = document.getElementById("connection-label");
    if (dot && label) {
        if (isConnected) {
            dot.classList.remove("disconnected");
            label.textContent = "Conectado";
        } else {
            dot.classList.add("disconnected");
            label.textContent = "Sin conexion";
        }
    }
}

function updateTimestamp() {
    const node = document.getElementById("last-update");
    if (node) {
        node.textContent = new Date().toLocaleTimeString("es-MX");
    }
}

// ==================== Main refresh ====================

async function refreshAll() {
    const tasks = [
        refreshHealth(),
        refreshMetrics(),
        refreshSubsystems(),
        refreshFailures(),
        refreshReminders(),
        refreshRoutines(),
        refreshPresence(),
    ];

    const results = await Promise.allSettled(tasks);

    const allFailed = results.every(r => r.status === "rejected");
    const anyOk = results.some(r => r.status === "fulfilled");

    setConnected(!allFailed);

    if (anyOk) {
        updateTimestamp();
    }
}

// ==================== Health ====================

async function refreshHealth() {
    const data = await fetchJSON(API_ENDPOINTS.health);
    const container = document.getElementById("health-content");
    if (!container) return;

    const statusClass = normalizeStatus(data.status);
    const subsystems = data.subsystems || [];
    const healthyCount = subsystems.filter(s => normalizeStatus(s.status) === "healthy").length;
    const degradedCount = subsystems.filter(s => normalizeStatus(s.status) === "degraded").length;
    const unhealthyCount = subsystems.filter(s => normalizeStatus(s.status) === "unhealthy").length;

    clearEl(container);

    const overview = el("div", { cls: "health-overview" });

    // Big status circle
    const bigStatus = el("div", { cls: "health-big-status" });
    const circle = el("div", { cls: "health-circle " + statusClass, text: statusIcon(statusClass) });
    const labelEl = el("span", { cls: "health-label", text: statusLabel(statusClass) });
    appendAll(bigStatus, [circle, labelEl]);

    // Stats grid
    const statsGrid = el("div", { cls: "health-stats" });
    statsGrid.appendChild(buildHealthStat("Subsistemas", String(subsystems.length), null));
    statsGrid.appendChild(buildHealthStat("Saludables", String(healthyCount), "var(--healthy)"));
    statsGrid.appendChild(buildHealthStat("Degradados", String(degradedCount), "var(--degraded)"));
    statsGrid.appendChild(buildHealthStat("Con fallos", String(unhealthyCount), "var(--unhealthy)"));

    appendAll(overview, [bigStatus, statsGrid]);
    container.appendChild(overview);
}

function buildHealthStat(labelText, valueText, color) {
    const stat = el("div", { cls: "health-stat" });
    stat.appendChild(el("span", { cls: "label", text: labelText }));
    const val = el("span", { cls: "value", text: valueText });
    if (color) val.style.color = color;
    stat.appendChild(val);
    return stat;
}

// ==================== Metrics ====================

async function refreshMetrics() {
    const data = await fetchJSON(API_ENDPOINTS.metrics);
    const container = document.getElementById("metrics-content");
    if (!container) return;

    const latency = data.latency || {};

    clearEl(container);

    const grid = el("div", { cls: "metrics-grid" });
    grid.appendChild(buildMetricItem("P50 Latencia", formatMs(latency.p50_ms), "ms"));
    grid.appendChild(buildMetricItem("P95 Latencia", formatMs(latency.p95_ms), "ms"));
    grid.appendChild(buildMetricItem("P99 Latencia", formatMs(latency.p99_ms), "ms"));
    grid.appendChild(buildMetricItem("Cola", String(data.queue_depth ?? 0), "pendientes"));
    grid.appendChild(buildMetricItem("Comandos", String(data.command_count ?? 0), "total"));
    grid.appendChild(buildMetricItem("Zonas activas", String(data.active_zones ?? 0), "zonas"));

    container.appendChild(grid);
}

function buildMetricItem(labelText, valueText, unitText) {
    const item = el("div", { cls: "metric-item" });
    item.appendChild(el("span", { cls: "label", text: labelText }));
    item.appendChild(el("span", { cls: "value", text: valueText }));
    item.appendChild(el("span", { cls: "unit", text: unitText }));
    return item;
}

// ==================== Subsystems ====================

async function refreshSubsystems() {
    const data = await fetchJSON(API_ENDPOINTS.subsystems);
    const container = document.getElementById("subsystems-content");
    if (!container) return;

    const subs = data.subsystems || [];
    clearEl(container);

    if (subs.length === 0) {
        container.appendChild(el("div", { cls: "state-message", text: "Sin subsistemas registrados" }));
        return;
    }

    const list = el("div", { cls: "subsystem-list" });

    for (const s of subs) {
        const sc = normalizeStatus(s.status);

        const item = el("div", { cls: "subsystem-item" });

        const info = el("div", { cls: "subsystem-info" });
        info.appendChild(el("span", { cls: "status-dot " + sc }));
        info.appendChild(el("span", { cls: "subsystem-name", text: s.name }));

        const right = el("div");
        right.appendChild(el("span", { cls: "status-badge " + sc, text: statusLabel(sc) }));
        if (s.detail) {
            right.appendChild(el("span", { cls: "subsystem-detail", text: " " + s.detail }));
        }

        appendAll(item, [info, right]);
        list.appendChild(item);
    }

    container.appendChild(list);
}

// ==================== Failures ====================

async function refreshFailures() {
    const data = await fetchJSON(API_ENDPOINTS.failures);
    const container = document.getElementById("failures-content");
    if (!container) return;

    const failures = data.failures || [];
    clearEl(container);

    if (failures.length === 0) {
        container.appendChild(el("div", { cls: "failures-empty", text: "Sin fallos recientes" }));
        return;
    }

    const table = el("table", { cls: "failures-table" });

    // Header
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const h of ["Hora", "Origen", "Mensaje"]) {
        headerRow.appendChild(el("th", { text: h }));
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement("tbody");
    for (const f of failures) {
        const row = document.createElement("tr");
        row.appendChild(el("td", { cls: "timestamp", text: f.timestamp ? formatTimestamp(f.timestamp) : "\u2014" }));
        row.appendChild(el("td", { cls: "source", text: f.source || f.subsystem || "\u2014" }));
        row.appendChild(el("td", { cls: "message", text: f.message || f.error || "\u2014" }));
        tbody.appendChild(row);
    }
    table.appendChild(tbody);

    container.appendChild(table);
}

// ==================== Reminders ====================

async function refreshReminders() {
    const data = await fetchJSON(API_ENDPOINTS.reminders);
    const container = document.getElementById("reminders-content");
    if (!container) return;

    clearEl(container);

    const running = data.scheduler_running;
    const nextTrigger = data.next_trigger_at ? formatTimestamp(data.next_trigger_at) : "\u2014";
    const deliveryFailures = data.delivery_failures || 0;

    const grid = el("div", { cls: "reminder-stats" });

    // Scheduler status
    const schedulerStat = el("div", { cls: "reminder-stat" });
    schedulerStat.appendChild(el("span", { cls: "label", text: "Scheduler" }));
    schedulerStat.appendChild(el("span", {
        cls: "value " + (running ? "running" : "stopped"),
        text: running ? "Activo" : "Detenido",
    }));
    grid.appendChild(schedulerStat);

    // Pending count
    const pendingStat = el("div", { cls: "reminder-stat" });
    pendingStat.appendChild(el("span", { cls: "label", text: "Pendientes" }));
    pendingStat.appendChild(el("span", { cls: "value", text: String(data.pending_count ?? 0) }));
    grid.appendChild(pendingStat);

    // Next trigger
    const nextStat = el("div", { cls: "reminder-stat" });
    nextStat.appendChild(el("span", { cls: "label", text: "Proximo" }));
    nextStat.appendChild(el("span", { cls: "value", text: nextTrigger, style: { fontSize: "0.85rem" } }));
    grid.appendChild(nextStat);

    // Delivery failures
    const failStat = el("div", { cls: "reminder-stat" });
    failStat.appendChild(el("span", { cls: "label", text: "Fallos entrega" }));
    const failVal = el("span", { cls: "value", text: String(deliveryFailures) });
    if (deliveryFailures > 0) failVal.style.color = "var(--unhealthy)";
    failStat.appendChild(failVal);
    grid.appendChild(failStat);

    container.appendChild(grid);
}

// ==================== Routines ====================

async function refreshRoutines() {
    const data = await fetchJSON(API_ENDPOINTS.routines);
    const container = document.getElementById("routines-content");
    if (!container) return;

    clearEl(container);

    if (!Array.isArray(data) || data.length === 0) {
        container.appendChild(el("div", { cls: "state-message", text: "Sin rutinas configuradas" }));
        return;
    }

    const list = el("div", { cls: "routines-list" });

    for (const r of data) {
        const item = el("div", { cls: "routine-item" });

        const info = el("div", { cls: "routine-info" });
        info.appendChild(el("span", { cls: "routine-name", text: r.name }));
        info.appendChild(el("span", { cls: "routine-desc", text: r.description || "" }));

        const meta = el("div", { cls: "routine-meta" });
        const enabledBadge = el("span", {
            cls: "routine-enabled " + (r.enabled ? "on" : "off"),
            text: r.enabled ? "On" : "Off",
        });
        meta.appendChild(enabledBadge);
        meta.appendChild(el("span", { text: (r.execution_count ?? 0) + " ejecuciones" }));

        appendAll(item, [info, meta]);
        list.appendChild(item);
    }

    container.appendChild(list);
}

// ==================== Presence ====================

async function refreshPresence() {
    const data = await fetchJSON(API_ENDPOINTS.presence);
    const container = document.getElementById("presence-content");
    if (!container) return;

    clearEl(container);

    if (!Array.isArray(data) || data.length === 0) {
        container.appendChild(el("div", { cls: "state-message", text: "Sin datos de presencia" }));
        return;
    }

    const list = el("div", { cls: "presence-list" });

    for (const u of data) {
        const item = el("div", { cls: "presence-item" });

        const user = el("div", { cls: "presence-user" });

        const initials = (u.name || u.user_id || "?")
            .split(" ")
            .map(w => w[0])
            .join("")
            .substring(0, 2)
            .toUpperCase();

        user.appendChild(el("div", { cls: "presence-avatar", text: initials }));

        const details = document.createElement("div");
        details.appendChild(el("div", { cls: "presence-name", text: u.name || u.user_id }));
        details.appendChild(el("div", { cls: "presence-location", text: u.current_room || "\u2014" }));
        user.appendChild(details);

        const homeClass = u.is_home ? "home" : "away";
        const homeText = u.is_home ? "En casa" : "Fuera";
        const homeBadge = el("span", { cls: "presence-home " + homeClass, text: homeText });

        appendAll(item, [user, homeBadge]);
        list.appendChild(item);
    }

    container.appendChild(list);
}

// ==================== Utilities ====================

function normalizeStatus(raw) {
    if (!raw) return "unhealthy";
    const s = String(raw).toLowerCase().replace("systemstatus.", "").replace("subsystemstatus.", "");
    if (s === "ok" || s === "healthy" || s === "running") return "healthy";
    if (s === "degraded" || s === "warning") return "degraded";
    return "unhealthy";
}

function statusIcon(statusClass) {
    switch (statusClass) {
        case "healthy": return "OK";
        case "degraded": return "!";
        case "unhealthy": return "X";
        default: return "?";
    }
}

function statusLabel(statusClass) {
    switch (statusClass) {
        case "healthy": return "Saludable";
        case "degraded": return "Degradado";
        case "unhealthy": return "Con fallos";
        default: return "Desconocido";
    }
}

function formatMs(val) {
    if (val === undefined || val === null) return "\u2014";
    return Number(val).toFixed(1);
}

function formatTimestamp(ts) {
    if (!ts) return "\u2014";

    var d;
    // Handle Unix timestamp (seconds as float)
    if (typeof ts === "number") {
        d = new Date(ts * 1000);
    } else {
        // Handle ISO string
        try {
            d = new Date(ts);
        } catch (_e) {
            return String(ts);
        }
    }

    if (isNaN(d.getTime())) return String(ts);

    return d.toLocaleString("es-MX", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        day: "2-digit",
        month: "short",
    });
}
