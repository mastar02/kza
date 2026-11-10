/* Observed collections and authorized operations for the technical panel.
 *
 * Data comes only from authenticated API responses. A failed or unproven read
 * leaves the collection empty and marked unavailable; there are no built-in
 * examples. Server demo mode (X-KZA-Source: mock) is kept and labelled as such.
 */
(function (root) {
  'use strict';

  const COLLECTIONS = Object.freeze([
    ['ZONES', '/api/zones', 'Zonas'],
    ['LLM_ENDPOINTS', '/api/llm/endpoints', 'Endpoints LLM'],
    ['HA_ENTITIES', '/api/ha/entities', 'Entidades de Home Assistant'],
    ['HA_ACTIONS', '/api/ha/actions', 'Acciones de Home Assistant'],
    ['ALERTS', '/api/alerts', 'Alertas'],
    ['GPUS', '/api/system/gpus', 'GPUs'],
    ['SERVICES', '/api/system/services', 'Servicios'],
  ]);

  // Mirror of RESTARTABLE_SERVICES in src/dashboard/admin.py. The server
  // remains authoritative and rejects any other name.
  const RESTARTABLE_SERVICES = Object.freeze(['kza-voice', 'kza-llm-ik', 'kza-llm-fast', 'chromadb']);
  const ADMIN = 'admin:manage';

  const STATUS = Object.freeze({
    PENDING: 'sin consultar', REAL: 'real', DEMO: 'demostración', DEGRADED: 'degradada',
  });

  function header(headers, name) {
    const value = headers && typeof headers.get === 'function' ? headers.get(name) : null;
    return typeof value === 'string' ? value : null;
  }

  function parseObservation(headers) {
    const unobserved = (header(headers, 'x-kza-unobserved-fields') || '')
      .split(',').map(item => item.trim()).filter(Boolean);
    return {
      source: (header(headers, 'x-kza-source') || '').toLowerCase(),
      sourceId: header(headers, 'x-kza-source-id'),
      state: header(headers, 'x-kza-observation-state'),
      observedAt: header(headers, 'x-kza-observed-at'),
      expiresAt: header(headers, 'x-kza-expires-at'),
      reason: header(headers, 'x-kza-observation-reason'),
      unobserved,
    };
  }

  // A bound above the server's own 15 s restart timeout; never changes it.
  const OPERATION_TIMEOUT_MS = 20000;

  function createObservationStore(options = {}) {
    const now = options.now || (() => new Date());
    const setTimer = options.setTimer || ((callback, ms) => setTimeout(callback, ms));
    const clearTimer = options.clearTimer || (handle => clearTimeout(handle));
    let expiryTimer = null;
    const entries = new Map();
    const sources = {};
    const channels = new Map();
    const listeners = new Set();
    let permissions = new Set();
    let notice = null;

    function notify() { for (const listener of [...listeners]) listener(); }
    // Freshness is time-driven: wake subscribers when the earliest current
    // observation expires, so an expired read is relabelled with no new read.
    function scheduleExpiry() {
      if (expiryTimer !== null) { clearTimer(expiryTimer); expiryTimer = null; }
      const current = now().getTime();
      const deadlines = [...entries.values()].map(entry => Date.parse(entry.expiresAt || ''))
        .filter(value => Number.isFinite(value) && value > current);
      if (!deadlines.length) return;
      expiryTimer = setTimer(() => { expiryTimer = null; notify(); scheduleExpiry(); },
        Math.min(...deadlines) - current + 1);
    }
    function reset() {
      if (expiryTimer !== null) { clearTimer(expiryTimer); expiryTimer = null; }
      entries.clear(); channels.clear(); permissions = new Set(); notice = null;
      for (const key of Object.keys(sources)) delete sources[key];
      for (const [key] of COLLECTIONS) sources[key] = STATUS.PENDING;
    }
    reset();

    return {
      sources,
      get(key) { return entries.get(key)?.data || []; },
      entry(key) { return entries.get(key) || { data: [], status: sources[key] || STATUS.PENDING, unobserved: [] }; },
      status(key) { return sources[key] || STATUS.PENDING; },
      available(key) { const status = sources[key]; return status === STATUS.REAL || status === STATUS.DEMO; },
      unobserved(key, field) { return (entries.get(key)?.unobserved || []).includes(field); },
      freshness(key) { return freshness(this.entry(key), now()); },
      now() { return now(); },
      accept(key, observation, data) {
        const status = data === null || data === undefined || !Array.isArray(data) ? STATUS.DEGRADED
          : observation.source === 'real' ? STATUS.REAL
          : observation.source === 'mock' ? STATUS.DEMO : STATUS.DEGRADED;
        const kept = status === STATUS.DEGRADED ? [] : data;
        entries.set(key, { ...observation, data: kept, status });
        sources[key] = status;
        scheduleExpiry();
        notify();
        return status;
      },
      fail(key) {
        entries.set(key, { data: [], status: STATUS.DEGRADED, unobserved: [] });
        sources[key] = STATUS.DEGRADED;
        notify();
      },
      setSource(key, value) { sources[key] = value; notify(); },
      setPermissions(list) { permissions = new Set(Array.isArray(list) ? list : []); notify(); },
      can(permission) { return permissions.has(permission); },
      setChannel(name, state) { channels.set(name, state); notify(); },
      channel(name) { return channels.get(name) || 'sin conectar'; },
      setNotice(value) { notice = value ? { ...value } : null; notify(); },
      notice() { return notice; },
      summary() {
        const keys = COLLECTIONS.map(([key]) => key);
        const pick = status => keys.filter(key => sources[key] === status);
        return { total: keys.length, real: pick(STATUS.REAL), demo: pick(STATUS.DEMO),
          degraded: pick(STATUS.DEGRADED), pending: pick(STATUS.PENDING) };
      },
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      clear() { reset(); notify(); },
    };
  }

  class AuthRequired extends Error {}

  async function readCollection({ fetch, store, key, path, signal }) {
    let response;
    try {
      response = await fetch(path, { credentials: 'same-origin', cache: 'no-store', signal });
    } catch (error) {
      if (signal?.aborted) throw error;
      store.fail(key);
      return STATUS.DEGRADED;
    }
    if (signal?.aborted) return store.status(key);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if (response.status === 401 || response.status === 403 || body?.detail === 'dashboard_auth_not_configured') {
        throw new AuthRequired();
      }
      store.fail(key);
      return STATUS.DEGRADED;
    }
    let data = null;
    try { data = await response.json(); } catch { data = null; }
    if (signal?.aborted) return store.status(key);
    return store.accept(key, parseObservation(response.headers), data);
  }

  async function readPermissions({ fetch, store, signal }) {
    const response = await fetch('/api/admin/auth/whoami', { credentials: 'same-origin', cache: 'no-store', signal });
    if (response.status === 401) throw new AuthRequired();
    if (!response.ok) { store.setPermissions([]); return; }
    const body = await response.json().catch(() => ({}));
    store.setPermissions(body?.permissions);
  }

  async function hydrate({ fetch, store, signal }) {
    await readPermissions({ fetch, store, signal });
    await Promise.all(COLLECTIONS.map(([key, path]) => readCollection({ fetch, store, key, path, signal })));
  }

  const enc = value => encodeURIComponent(String(value));
  const OPERATIONS = Object.freeze({
    ackAlert: id => ({ method: 'POST', path: `/api/admin/alerts/${enc(id)}/ack`, refresh: 'ALERTS',
      done: 'El servidor marcó la alerta como atendida.' }),
    dismissAlert: id => ({ method: 'DELETE', path: `/api/admin/alerts/${enc(id)}`, refresh: 'ALERTS',
      done: 'El servidor descartó la alerta.' }),
    restartService: name => ({ method: 'POST', path: `/api/admin/services/${enc(name)}/restart`, refresh: 'SERVICES',
      done: `systemctl completó el reinicio de ${name}. Revisá su estado en la tabla.` }),
    clearCooldown: id => ({ method: 'POST', path: `/api/llm/endpoints/${enc(id)}/clear-cooldown`, refresh: 'LLM_ENDPOINTS',
      done: `El servidor quitó el cooldown de ${id}.` }),
  });

  function availability(store, kind, target) {
    if (!Object.hasOwn(OPERATIONS, kind)) return { enabled: false, reason: 'Operación no disponible.' };
    if (store.status(OPERATIONS[kind](target).refresh) === STATUS.DEMO) {
      return { enabled: false, reason: 'No disponible en modo demostración: los datos mostrados son de ejemplo.' };
    }
    if (!store.can(ADMIN)) return { enabled: false, reason: 'Requiere permiso de administración.' };
    if (kind === 'restartService' && !RESTARTABLE_SERVICES.includes(target)) {
      return { enabled: false, reason: 'Este servicio no se reinicia desde el panel.' };
    }
    return { enabled: true, reason: '' };
  }

  function failure(status, detail) {
    const uncertain = detail && typeof detail === 'object' && detail.effect === 'uncertain';
    if (status === 504 || uncertain) return { state: 'uncertain', message: 'Resultado incierto: el servidor no pudo confirmar el efecto. Revisá el estado antes de reintentar.' };
    if (status === 401) return { state: 'auth', message: 'La sesión venció. Volvé a iniciar sesión.' };
    if (status === 403) return { state: 'denied', message: 'La sesión no tiene permiso para esta operación.' };
    if (status === 404) return { state: 'not_found', message: 'El servidor no encontró el elemento.' };
    if (status === 409 && detail === 'observability_demo_mode') {
      return { state: 'rejected', message: 'No disponible en modo demostración: el servidor no aplicó ningún cambio.' };
    }
    if (status === 409) return { state: 'conflict', message: 'El estado cambió. Actualizá antes de reintentar.' };
    if (status === 503) return { state: 'unavailable', message: 'La operación no está disponible en este momento.' };
    return { state: 'error', message: `No se pudo completar la operación (HTTP ${status}).` };
  }

  const UNCERTAIN_TIMEOUT = 'Resultado incierto: el servidor no respondió a tiempo. Revisá el estado antes de reintentar.';

  async function runOperation({ fetch, store, kind, target, timeoutMs = OPERATION_TIMEOUT_MS,
                                setTimer = (callback, ms) => setTimeout(callback, ms),
                                clearTimer = handle => clearTimeout(handle) }) {
    const allowed = availability(store, kind, target);
    if (!allowed.enabled) return { state: 'denied', message: allowed.reason };
    const operation = OPERATIONS[kind](target);
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimer(() => { timedOut = true; controller.abort(); }, timeoutMs);
    let response, body;
    try {
      response = await fetch(operation.path, { method: operation.method, credentials: 'same-origin',
        cache: 'no-store', signal: controller.signal });
      body = await response.json().catch(error => {
        if (timedOut) throw error;
        return {};
      });
    } catch {
      return { state: 'uncertain', message: timedOut ? UNCERTAIN_TIMEOUT
        : 'Resultado incierto: no llegó respuesta del servidor. Revisá el estado antes de reintentar.' };
    } finally {
      clearTimer(timer);
    }
    if (timedOut) return { state: 'uncertain', message: UNCERTAIN_TIMEOUT };
    if (!response.ok) return failure(response.status, body?.detail);
    if (body && body.mocked === true) {
      return { state: 'demo', message: 'El servidor está en modo demostración: no se aplicó ningún cambio.' };
    }
    const path = COLLECTIONS.find(([key]) => key === operation.refresh)[1];
    let refreshed;
    try {
      refreshed = await readCollection({ fetch, store, key: operation.refresh, path });
    } catch (error) {
      if (error instanceof AuthRequired) {
        return { state: 'confirmed', sessionExpired: true,
          message: `${operation.done} No se pudo actualizar la lista porque la sesión venció.` };
      }
      refreshed = STATUS.DEGRADED;
    }
    return { state: 'confirmed', message: refreshed === STATUS.DEGRADED
      ? `${operation.done} No se pudo actualizar la lista.` : operation.done };
  }

  // Seconds until an ISO-8601 instant with an explicit offset; null otherwise.
  function remainingUntil(instant, now = new Date()) {
    if (typeof instant !== 'string' || !/(Z|[+-]\d{2}:\d{2})$/.test(instant)) return null;
    const target = Date.parse(instant);
    if (Number.isNaN(target)) return null;
    return Math.max(0, Math.ceil((target - now.getTime()) / 1000));
  }

  function formatInstant(value) {
    if (typeof value !== 'string' || !value) return 'no disponible';
    // Without an explicit offset the instant's zone is unknown; never guess it.
    if (!/(Z|[+-]\d{2}:?\d{2})$/.test(value)) return 'no disponible (sin zona)';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'no disponible'
      : date.toLocaleTimeString('es', { hour12: false, timeZoneName: 'short' });
  }

  function freshness(entry, now = new Date()) {
    if (!entry?.expiresAt) return entry?.status === STATUS.REAL ? 'sin vencimiento declarado' : 'no disponible';
    const expiry = new Date(entry.expiresAt).getTime();
    if (Number.isNaN(expiry)) return 'no disponible';
    return expiry > now.getTime() ? 'vigente' : 'vencida';
  }

  const api = {
    COLLECTIONS, RESTARTABLE_SERVICES, STATUS, OPERATIONS, AuthRequired,
    parseObservation, createObservationStore, readCollection, hydrate,
    OPERATION_TIMEOUT_MS, availability, runOperation, remainingUntil, formatInstant, freshness,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KzaObsData = api;
})(typeof window === 'undefined' ? globalThis : window);
