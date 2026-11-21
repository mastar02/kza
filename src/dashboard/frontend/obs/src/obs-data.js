/* Observed collections and authorized operations for the technical panel.
 *
 * Data comes only from authenticated API responses. A failed or unproven read
 * leaves the collection empty and marked unavailable; there are no built-in
 * examples. Server demo mode (X-KZA-Source: mock) is kept and labelled as such.
 */
(function (root) {
  'use strict';

  const { SessionRequired: AuthRequired, requestAccess, checkAccess } =
    typeof module !== 'undefined' && module.exports ? require('./session.js') : root.KzaDashboardSession;

  const COLLECTIONS = Object.freeze([
    ['ZONES', '/api/zones', 'Zonas'],
    ['LLM_ENDPOINTS', '/api/llm/endpoints', 'Endpoints LLM'],
    ['HA_ENTITIES', '/api/ha/entities', 'Entidades de Home Assistant'],
    ['HA_ACTIONS', '/api/ha/actions', 'Acciones de Home Assistant'],
    ['ALERTS', '/api/alerts', 'Alertas'],
    ['GPUS', '/api/system/gpus', 'GPUs'],
    ['SERVICES', '/api/system/services', 'Servicios'],
  ]);

  const COLLECTION_PATHS = Object.freeze(Object.fromEntries(COLLECTIONS.map(([key, path]) => [key, path])));

  // Collections displayed by each /obs/ view; inactive views are never polled.
  const VIEW_COLLECTIONS = Object.freeze({
    live: Object.freeze(['ZONES', 'LLM_ENDPOINTS']),
    conversations: Object.freeze([]),
    privacy: Object.freeze([]),
    alerts: Object.freeze(['ALERTS']),
    ha: Object.freeze(['HA_ENTITIES', 'HA_ACTIONS']),
    llm: Object.freeze(['LLM_ENDPOINTS']),
    users: Object.freeze([]),
    system: Object.freeze(['GPUS', 'SERVICES']),
    admin: Object.freeze([]),
  });

  // Bounded retry interval when an active-view observation is degraded or
  // returned already expired by the server, avoiding tight retry loops.
  const DEGRADED_RETRY_MS = 10000;

  // Mirror of RESTARTABLE_SERVICES in src/dashboard/admin.py. The server
  // remains authoritative and rejects any other name.
  const RESTARTABLE_SERVICES = Object.freeze(['kza-voice', 'kza-llm-ik', 'kza-llm-fast', 'chromadb']);
  const ADMIN = 'admin:manage';

  const STATUS = Object.freeze({
    PENDING: 'sin consultar', REAL: 'real', DEMO: 'demostración', DEGRADED: 'degradada', FORBIDDEN: 'sin permiso',
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
    const received = new Map();
    const sources = {};
    const channels = new Map();
    const operations = new Map();
    const listeners = new Set();
    let readSequence = 0;
    let permissions = new Set();
    let notice = null;

    const operationKey = (kind, target) => JSON.stringify([kind, String(target)]);
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
      entries.clear(); received.clear(); channels.clear(); operations.clear(); permissions = new Set(); notice = null;
      for (const key of Object.keys(sources)) delete sources[key];
      for (const [key] of COLLECTIONS) sources[key] = STATUS.PENDING;
    }
    reset();

    return {
      sources,
      get(key) { return entries.get(key)?.data || []; },
      entry(key) { return entries.get(key) || { data: [], status: sources[key] || STATUS.PENDING, unobserved: [] }; },
      receivedAt(key) { return received.has(key) ? received.get(key) : null; },
      status(key) { return sources[key] || STATUS.PENDING; },
      available(key) {
        const entry = entries.get(key);
        if (entry?.alertStatus) return entry.alertStatus.available && freshness(entry, now()) === 'vigente';
        const status = sources[key];
        return status === STATUS.REAL || status === STATUS.DEMO;
      },
      unobserved(key, field) { return (entries.get(key)?.unobserved || []).includes(field); },
      freshness(key) { return freshness(this.entry(key), now()); },
      now() { return now(); },
      beginRead() { return ++readSequence; },
      accept(key, observation, data, readId) {
        if (key === 'ALERTS' && readId < entries.get(key)?.readId) return this.status(key);
        const alertStatus = key === 'ALERTS' && ['real', 'degraded'].includes(observation.source)
          && typeof data?.available === 'boolean' && Array.isArray(data?.producers) && Array.isArray(data?.alerts)
          ? { available: data.available, producers: data.producers, output: data.output } : null;
        if (alertStatus) data = data.alerts;
        const status = data === null || data === undefined || !Array.isArray(data) ? STATUS.DEGRADED
          : observation.source === 'real' ? STATUS.REAL
          : observation.source === 'mock' ? STATUS.DEMO : STATUS.DEGRADED;
        const kept = status === STATUS.DEGRADED && !alertStatus ? [] : data;
        entries.set(key, { ...observation, data: kept, status, alertStatus, readId });
        received.set(key, now().getTime());
        sources[key] = status;
        const current = status === STATUS.REAL && (!observation.expiresAt || freshness(observation, now()) === 'vigente');
        if (current || alertStatus) {
          for (const [id, attempt] of operations) {
            const verified = attempt.kind === 'setAlertZones' ? alertStatus?.output?.available === true : current;
            if (verified && attempt.refresh === key && attempt.outcome.state === 'uncertain' && readId > attempt.afterRead) {
              operations.delete(id);
            }
          }
        }
        scheduleExpiry();
        notify();
        return status;
      },
      fail(key, status = STATUS.DEGRADED, reason = null, readId) {
        if (key === 'ALERTS' && readId < entries.get(key)?.readId) return this.status(key);
        entries.set(key, { data: [], status, reason, unobserved: [], readId });
        received.set(key, now().getTime());
        sources[key] = status;
        scheduleExpiry();
        notify();
        return status;
      },
      setSource(key, value) { sources[key] = value; notify(); },
      setPermissions(list) { permissions = new Set(Array.isArray(list) ? list : []); notify(); },
      can(permission) { return permissions.has(permission); },
      setChannel(name, state) { channels.set(name, state); notify(); },
      channel(name) { return channels.get(name) || 'sin conectar'; },
      setNotice(value) { notice = value ? { ...value } : null; notify(); },
      notice() { return notice; },
      operationOutcome(kind, target) { return operations.get(operationKey(kind, target))?.outcome; },
      beginOperation(kind, target, refresh) {
        const id = operationKey(kind, target);
        const attempt = { id, kind, refresh, outcome: { state: 'pending', message: 'Hay una solicitud en curso.' } };
        operations.set(id, attempt);
        this.setNotice(null);
        return attempt;
      },
      finishOperation(attempt, outcome) {
        if (operations.get(attempt.id) !== attempt) return false;
        if (outcome.state === 'uncertain') {
          attempt.outcome = outcome;
          // Reads already in flight may still describe the state before the write.
          attempt.afterRead = readSequence;
        } else operations.delete(attempt.id);
        this.setNotice(outcome);
        return true;
      },
      summary() {
        const keys = COLLECTIONS.map(([key]) => key);
        const pick = status => keys.filter(key => sources[key] === status);
        return { total: keys.length, real: pick(STATUS.REAL), demo: pick(STATUS.DEMO),
          degraded: pick(STATUS.DEGRADED), forbidden: pick(STATUS.FORBIDDEN), pending: pick(STATUS.PENDING) };
      },
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      clear() { reset(); notify(); },
    };
  }

  async function readCollection({ fetch, store, key, path, signal }) {
    const readId = store.beginRead();
    let result;
    try {
      result = await requestAccess(fetch, path, { signal }, async response => ({
        observation: parseObservation(response.headers),
        data: await response.json(),
      }));
    } catch (error) {
      if (signal?.aborted || error instanceof AuthRequired) throw error;
      const status = error.kind === 'forbidden' ? STATUS.FORBIDDEN : STATUS.DEGRADED;
      return store.fail(key, status, error.message, readId);
    }
    if (signal?.aborted) return store.status(key);
    return store.accept(key, result.observation, result.data, readId);
  }

  async function readPermissions({ fetch, store, signal }) {
    const body = await checkAccess(fetch, signal, response => response.json());
    if (!signal?.aborted) store.setPermissions(body?.permissions);
  }

  async function readCollections({ fetch, store, keys, signal }) {
    const unique = [...new Set(keys)].filter(key => Object.hasOwn(COLLECTION_PATHS, key));
    if (!unique.length) return [];
    return await Promise.all(unique.map(key => readCollection({
      fetch, store, key, path: COLLECTION_PATHS[key], signal,
    })));
  }

  async function hydrate({ fetch, store, signal }) {
    await readPermissions({ fetch, store, signal });
    await Promise.all(COLLECTIONS.map(([key, path]) => readCollection({ fetch, store, key, path, signal })));
  }

  function defaultVisible() {
    if (typeof document === 'undefined') return true;
    return document.visibilityState === 'visible' && !document.hidden;
  }

  function wsDisconnected(state) {
    return state !== 'sin conectar' && state !== 'conectando' && state !== 'conectado';
  }

  // Deadline-driven poller for the active /obs/ view only. Never extends
  // server-emitted expires_at locally, pauses while the tab is hidden, stops
  // immediately on 401 or session abort, and rehydrates active/expired
  // inventories when /ws/live returns to 'conectado' after a disconnect.
  function createObservationPoller({
    fetch,
    store,
    view = null,
    signal = null,
    isVisible = defaultVisible,
    setTimer = (callback, ms) => setTimeout(callback, ms),
    clearTimer = handle => clearTimeout(handle),
    onUnauthorized = () => {},
    retryMs = DEGRADED_RETRY_MS,
  }) {
    let activeView = view;
    let stopped = Boolean(signal?.aborted);
    let timer = null;
    let sawDisconnect = wsDisconnected(store.channel('ws'));
    let pendingReconnect = false;
    const inFlight = new Map();

    function activeKeys() {
      return Object.hasOwn(VIEW_COLLECTIONS, activeView) ? VIEW_COLLECTIONS[activeView] : [];
    }

    function reconnectKeys() {
      const keys = new Set(activeKeys());
      for (const [key] of COLLECTIONS) {
        if (store.freshness(key) === 'vencida') keys.add(key);
      }
      return [...keys];
    }

    function clearScheduled() {
      if (timer !== null) {
        clearTimer(timer);
        timer = null;
      }
    }

    function abortKeys(filter = () => true) {
      for (const [key, controller] of [...inFlight.entries()]) {
        if (!filter(key)) continue;
        inFlight.delete(key);
        controller.abort();
      }
    }

    function dueAt(key, currentMs) {
      const entry = store.entry(key);
      const status = entry.status;
      const receivedAt = store.receivedAt(key);
      if (status === STATUS.PENDING) {
        return receivedAt === null ? currentMs : receivedAt + retryMs;
      }
      if (status === STATUS.REAL) {
        const expiry = Date.parse(entry.expiresAt || '');
        if (!Number.isFinite(expiry)) return null;
        if (expiry > currentMs) return expiry + 1;
        // Expired: read immediately unless a read already landed at/after
        // expiry (e.g. server clock skew or already-stale cache).
        if (receivedAt === null || receivedAt < expiry) return currentMs;
        return receivedAt + retryMs;
      }
      if (status === STATUS.DEGRADED) {
        return receivedAt === null ? currentMs : receivedAt + retryMs;
      }
      return null;
    }

    function fetchKeys(keys) {
      if (stopped || !isVisible()) return;
      for (const key of keys) {
        if (!Object.hasOwn(COLLECTION_PATHS, key) || inFlight.has(key)) continue;
        const controller = new AbortController();
        inFlight.set(key, controller);
        const requestSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
        readCollection({ fetch, store, key, path: COLLECTION_PATHS[key], signal: requestSignal })
          .catch(error => {
            if (error instanceof AuthRequired) {
              const wasRunning = !stopped;
              stop();
              if (wasRunning) onUnauthorized(error);
            }
          })
          .finally(() => {
            if (inFlight.get(key) === controller) inFlight.delete(key);
            if (!stopped) schedule();
          });
      }
    }

    function schedule() {
      clearScheduled();
      if (stopped) return;
      if (!isVisible()) {
        abortKeys();
        return;
      }
      if (pendingReconnect) {
        pendingReconnect = false;
        fetchKeys(reconnectKeys());
      }
      const current = store.now().getTime();
      const dueNow = [];
      let nextAt = Infinity;
      for (const key of activeKeys()) {
        if (inFlight.has(key)) continue;
        const at = dueAt(key, current);
        if (at === null) continue;
        if (at <= current) dueNow.push(key);
        else if (at < nextAt) nextAt = at;
      }
      if (dueNow.length) fetchKeys(dueNow);
      if (!stopped && isVisible() && Number.isFinite(nextAt)) {
        timer = setTimer(() => {
          timer = null;
          schedule();
        }, Math.max(1, nextAt - store.now().getTime()));
      }
    }

    function onStoreChange() {
      if (stopped) return;
      const ws = store.channel('ws');
      if (wsDisconnected(ws)) {
        sawDisconnect = true;
      } else if (ws === 'conectado' && sawDisconnect) {
        sawDisconnect = false;
        pendingReconnect = true;
      }
      schedule();
    }

    function setView(nextView) {
      if (stopped || activeView === nextView) return;
      activeView = nextView;
      const keep = new Set(activeKeys());
      abortKeys(key => !keep.has(key));
      schedule();
    }

    function stop() {
      if (stopped) return;
      stopped = true;
      pendingReconnect = false;
      clearScheduled();
      abortKeys();
      unsubscribe();
      signal?.removeEventListener('abort', stop);
    }

    const unsubscribe = store.subscribe(onStoreChange);
    if (stopped) unsubscribe();
    else {
      signal?.addEventListener('abort', stop, { once: true });
      schedule();
    }

    return {
      setView,
      sync: schedule,
      stop,
      activeKeys,
      running: () => !stopped,
      pendingKeys: () => [...inFlight.keys()],
    };
  }

  const enc = value => encodeURIComponent(String(value));
  const OPERATIONS = Object.freeze({
    ackAlert: id => ({ method: 'POST', path: `/api/admin/alerts/${enc(id)}/ack`, refresh: 'ALERTS',
      done: 'El servidor marcó la alerta como atendida.' }),
    dismissAlert: id => ({ method: 'DELETE', path: `/api/admin/alerts/${enc(id)}`, refresh: 'ALERTS',
      done: 'El servidor descartó la alerta.' }),
    setAlertZones: () => ({ method: 'PUT', path: '/api/alerts/zones', refresh: 'ALERTS',
      done: 'El servidor guardó las zonas de avisos hablados.' }),
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
    if (kind === 'setAlertZones' && store.entry('ALERTS').alertStatus?.output?.writable !== true) {
      return { enabled: false, reason: 'La configuración de zonas no está disponible.' };
    }
    if (kind === 'restartService' && !RESTARTABLE_SERVICES.includes(target)) {
      return { enabled: false, reason: 'Este servicio no se reinicia desde el panel.' };
    }
    return { enabled: true, reason: '' };
  }

  function failure(status, detail) {
    if (status === 401) return { state: 'auth', message: 'La sesión venció. Volvé a iniciar sesión.' };
    if (status === 403) return { state: 'denied', message: 'La sesión no tiene permiso para esta operación.' };
    if (status === 409 && detail === 'observability_demo_mode') {
      return { state: 'rejected', message: 'No disponible en modo demostración: el servidor no aplicó ningún cambio.' };
    }
    if (status === 409) return { state: 'conflict', message: 'Conflicto: el estado cambió. Actualizá antes de reintentar.' };
    const uncertain = detail && typeof detail === 'object' && detail.effect === 'uncertain';
    if (status === 504 || uncertain) return { state: 'uncertain', message: 'Resultado incierto: el servidor no pudo confirmar el efecto. Revisá el estado antes de reintentar.' };
    if (status === 404) return { state: 'not_found', message: 'El servidor no encontró el elemento.' };
    if (status === 503) return { state: 'unavailable', message: 'La operación no está disponible en este momento.' };
    return { state: 'error', message: `No se pudo completar la operación (HTTP ${status}).` };
  }

  const UNCERTAIN_TIMEOUT = 'Resultado incierto: el servidor no respondió a tiempo. Revisá el estado antes de reintentar.';

  async function sendOperation({ fetch, store, operation, timeoutMs = OPERATION_TIMEOUT_MS,
                                 setTimer = (callback, ms) => setTimeout(callback, ms),
                                 clearTimer = handle => clearTimeout(handle) }) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimer(() => { timedOut = true; controller.abort(); }, timeoutMs);
    let response, body;
    try {
      const request = { method: operation.method, credentials: 'same-origin', cache: 'no-store', signal: controller.signal };
      if (operation.payload !== undefined) {
        request.headers = { 'Content-Type': 'application/json' };
        request.body = JSON.stringify(operation.payload);
      }
      response = await fetch(operation.path, request);
      if (response.status === 401 || response.status === 403) return failure(response.status);
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
      if (operation.payload !== undefined && refreshed !== STATUS.FORBIDDEN) {
        refreshed = store.entry('ALERTS').alertStatus?.output?.available === true ? STATUS.REAL : STATUS.DEGRADED;
      }
    } catch (error) {
      if (error instanceof AuthRequired) {
        return { state: 'confirmed', sessionExpired: true,
          message: `${operation.done} No se pudo actualizar la lista porque la sesión venció.` };
      }
      refreshed = STATUS.DEGRADED;
    }
    const refreshMessage = refreshed === STATUS.FORBIDDEN ? ' Sin permiso para actualizar la lista.'
      : refreshed === STATUS.DEGRADED ? ' No se pudo actualizar la lista.' : '';
    return { state: 'confirmed', message: operation.done + refreshMessage };
  }

  async function runOperation({ fetch, store, kind, target, payload, ...options }) {
    const held = store.operationOutcome(kind, target);
    if (held) return held;
    const allowed = availability(store, kind, target);
    if (!allowed.enabled) {
      const outcome = { state: 'denied', message: allowed.reason };
      store.setNotice(outcome);
      return outcome;
    }
    const operation = { ...OPERATIONS[kind](target), payload };
    const attempt = store.beginOperation(kind, target, operation.refresh);
    const outcome = await sendOperation({ ...options, fetch, store, operation });
    if (!store.finishOperation(attempt, outcome)) return { state: 'retired' };
    return outcome;
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

  function alertProducerLabel(producer, now = new Date()) {
    if (producer.state === 'stale' || (['fresh', 'partial'].includes(producer.state)
        && freshness({ expiresAt: producer.expires_at }, now) === 'vencida')) return 'vencida';
    if (producer.reason === 'source_unconfigured') return 'sin iniciar';
    if (producer.reason === 'source_closed') return 'detenida';
    if (producer.connected === false) return 'desconectada';
    if (producer.available) return 'disponible';
    if (producer.state === 'partial') return 'parcial';
    return 'incierta';
  }

  const api = {
    COLLECTIONS, VIEW_COLLECTIONS, DEGRADED_RETRY_MS, RESTARTABLE_SERVICES, STATUS, OPERATIONS, AuthRequired,
    parseObservation, createObservationStore, readCollection, readCollections, hydrate, createObservationPoller, alertProducerLabel,
    OPERATION_TIMEOUT_MS, availability, runOperation, remainingUntil, formatInstant, freshness,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KzaObsData = api;
})(typeof window === 'undefined' ? globalThis : window);
