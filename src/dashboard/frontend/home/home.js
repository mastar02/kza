/* KZA home — shared NEXO helpers (classic script), also loadable by the native Node test runner. */
(function (root) {
  'use strict';
  const OPERATION_LABELS = { activate_scene: 'Activar escena', turn_on: 'Encender', turn_off: 'Apagar', brightness: 'Brillo', color_temp_kelvin: 'Temperatura de color', set_temperature: 'Temperatura objetivo', set_hvac_mode: 'Modo del aire', open_cover: 'Abrir', close_cover: 'Cerrar', stop_cover: 'Detener', set_cover_position: 'Posición' };
  const OPERATION_PARAMETERS = { brightness: ['brightness_pct', 'brightness'], color_temp_kelvin: ['color_temp_kelvin', 'brightness_pct', 'brightness'], set_temperature: ['temperature'], set_hvac_mode: ['hvac_mode'], set_cover_position: ['position'] };
  const MODE_LABELS = { off: 'Apagado', cool: 'Frío', heat: 'Calor', dry: 'Deshumidificar', fan_only: 'Ventilación', auto: 'Automático', heat_cool: 'Frío y calor' };
  function controlFields(entity, operation, parameters = {}) {
    const limits = entity?.control_limits || {};
    const number = (name, label, min, max, step = 1, optional = false) => ({ name, label, min, max, step, optional });
    if (parameters.brightness !== undefined && parameters.brightness !== null && ['brightness', 'color_temp_kelvin'].includes(operation)) {
      const fields = [number('brightness', 'Brillo (0–255)', Math.max(operation === 'color_temp_kelvin' ? 1 : 0, Math.ceil((limits.brightness_min ?? 0) * 255 / 100)), Math.floor((limits.brightness_max ?? 100) * 255 / 100))];
      if (operation === 'color_temp_kelvin') fields.unshift(number('color_temp_kelvin', 'Temperatura de color (K)', limits.color_temp_min, limits.color_temp_max));
      return fields;
    }
    if (operation === 'brightness') return [number('brightness_pct', 'Brillo (%)', limits.brightness_min ?? 0, limits.brightness_max ?? 100)];
    if (operation === 'color_temp_kelvin') return [number('color_temp_kelvin', 'Temperatura de color (K)', limits.color_temp_min, limits.color_temp_max), ...(entity?.capabilities?.includes('brightness') ? [number('brightness_pct', 'Brillo opcional (%)', Math.max(1, limits.brightness_min), limits.brightness_max, 1, true)] : [])];
    if (operation === 'set_temperature') return [number('temperature', 'Temperatura objetivo (°C)', limits.temperature_min, limits.temperature_max, limits.temperature_step)];
    if (operation === 'set_hvac_mode') return [{ name: 'hvac_mode', label: 'Modo', options: (limits.hvac_modes || []).map(mode => [mode, MODE_LABELS[mode] || mode]) }];
    if (operation === 'set_cover_position') return [number('position', 'Posición (% abierto)', limits.position_min, limits.position_max)];
    return [];
  }
  function validControlValue(field, value) {
    if (field.optional && (value === null || value === undefined || value === '')) return true;
    if (field.options) return field.options.some(([option]) => option === value);
    return Number.isFinite(value) && Number.isFinite(field.min) && Number.isFinite(field.max) && value >= field.min && value <= field.max && (field.name === 'temperature' ? Number.isFinite(field.step) && field.step > 0 && Math.abs((value - field.min) / field.step - Math.round((value - field.min) / field.step)) <= 1e-7 : Number.isInteger(value));
  }
  function selectFreshResponse(currentSeq, incomingSeq) { return incomingSeq >= currentSeq; }
  function actionLabel(status) {
    return ({ confirmed: 'Confirmado por HA', failed: 'Falló', uncertain: 'Sin confirmar', missing: 'Resultado desconocido', in_progress: 'En curso' })[status] || 'Resultado desconocido';
  }
  function effectSummary(effects) {
    if (!Array.isArray(effects) || !effects.length) return '';
    const count = status => effects.filter(effect => effect.status === status).length;
    const confirmed = count('confirmed'), unconfirmed = count('failed') + count('uncertain'), remaining = count('not_executed');
    return [`${confirmed} confirmado${confirmed === 1 ? '' : 's'}`, unconfirmed ? `${unconfirmed} sin confirmar` : '', remaining ? `${remaining} sin ejecutar` : ''].filter(Boolean).join(' · ');
  }
  function observationMillis(value) {
    if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value * 1000 : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const parsed = Date.parse(value); return Number.isFinite(parsed) ? parsed : null;
  }
  function createSnapshotClock(data, { monotonic = () => root.performance.now(), wallClock = Date.now } = {}) {
    const receivedMono = monotonic(), receivedWall = wallClock();
    const explicit = Object.prototype.hasOwnProperty.call(data, 'observation');
    const observation = explicit && data.observation && typeof data.observation === 'object' ? data.observation : {};
    const observed = observationMillis(explicit ? observation.observed_at : data.observed_at);
    const expires = observationMillis(observation.expires_at);
    // Explicit server expiry is anchored to the original observation. Older
    // microphone schemas retain their existing local freshness budget.
    const ttlMs = explicit ? (observed !== null && expires !== null && expires > observed ? expires - observed : 0)
      : (Number.isFinite(data.stale_after_s) && data.stale_after_s > 0 ? data.stale_after_s : 6) * 1000;
    const sourceUnavailable = explicit && !['fresh', 'partial'].includes(observation.state);
    const elapsed = () => Math.max(0, monotonic() - receivedMono, wallClock() - receivedWall);
    const age = timestamp => {
      const parsed = observationMillis(timestamp);
      return parsed === null ? null : Math.max(0, receivedWall - parsed) + elapsed();
    };
    function directionFreshness(direction) {
      const timestampAge = age(direction?.observed_at);
      const reportedAge = Number.isFinite(direction?.age_ms) && direction.age_ms >= 0 ? direction.age_ms + elapsed() : 0;
      const ageMs = timestampAge === null ? null : Math.max(timestampAge, reportedAge);
      const ttlMs = direction?.stale_after_ms ?? 2000;
      const stale = ageMs === null || !Number.isFinite(ttlMs) || ttlMs <= 0 || ageMs >= ttlMs;
      return { ageMs, stale, remainingMs: stale ? Infinity : ttlMs - ageMs };
    }
    return {
      age, directionFreshness, now: () => receivedWall + elapsed(),
      read() {
        const ageMs = observed === null ? null : Math.max(0, receivedWall - observed) + elapsed();
        const stale = data.stale === true || sourceUnavailable || ageMs === null || ageMs >= ttlMs || observed > receivedWall;
        // The existing presentation timer also wakes at each direction deadline.
        // No sample sequence or refresh completion renews the observation time.
        const directionDeadlines = (data.microphones || []).filter(mic => ['observed', 'no_speech'].includes(mic.direction?.state)).map(mic => directionFreshness(mic.direction).remainingMs);
        const entityDeadlines = (data.entities || []).map(entity => {
          const expires = observationMillis(entity.observation?.expires_at);
          const remaining = expires === null ? 0 : expires - (receivedWall + elapsed());
          return remaining > 0 ? remaining : Infinity;
        });
        return { ageMs, unknown: ageMs === null, stale, remainingMs: Math.min(stale ? Infinity : ttlMs - ageMs, ...directionDeadlines, ...entityDeadlines) };
      }
    };
  }
  function createFreshnessTicker({ read, onTick, schedule = setTimeout, cancel = clearTimeout }) {
    let active = false, timer;
    function clear() { if (timer !== undefined) cancel(timer); timer = undefined; }
    function tick() {
      if (!active) return;
      const state = read(); onTick(state);
      // Wake at the expiry boundary even if it falls between age updates.
      if (active) timer = schedule(tick, Math.max(1, Math.min(1000, state.remainingMs ?? 1000)));
    }
    return {
      start() { if (!active) { active = true; tick(); } },
      refresh() { if (active) { clear(); tick(); } },
      stop() { active = false; clear(); }
    };
  }
  // An entity observation is current only inside its own server-issued window.
  function entityObservationCurrent(entity, now = Date.now()) {
    if (!Object.prototype.hasOwnProperty.call(entity, 'observation')) return true;
    const observation = entity.observation || {};
    const observed = observationMillis(observation.observed_at), expires = observationMillis(observation.expires_at);
    return ['fresh', 'partial'].includes(observation.state) && observed !== null && expires !== null && observed <= now && now < expires;
  }
  function entityCanControl(entity, operation, stale, now = Date.now()) {
    const domain = entity.domain || entity.entity_id?.split('.')[0];
    const states = { light: ['on', 'off'], switch: ['on', 'off'], climate: Object.keys(MODE_LABELS), cover: ['open', 'closed', 'opening', 'closing'] };
    const current = entityObservationCurrent(entity, now);
    return !stale && current && entity.available === true && (domain === 'scene' ? entity.state !== 'unavailable' : states[domain]?.includes(entity.state)) && Array.isArray(entity.capabilities) && entity.capabilities.includes(operation);
  }
  function createApi({ fetch: fetcher = root.fetch.bind(root), onDemo = () => {}, onUnauthorized = () => {}, timeoutMs = 12000 } = {}) {
    let sessionGeneration = 0;
    async function api(path, options = {}) {
      const requestGeneration = sessionGeneration;
      const controller = new AbortController();
      const abort = () => controller.abort();
      if (options.signal?.aborted) abort();
      options.signal?.addEventListener('abort', abort, { once: true });
      const timeout = setTimeout(abort, timeoutMs);
      try {
        const { body, ...rest } = options;
        const response = await fetcher(path, { ...rest, credentials: 'same-origin', signal: controller.signal, cache: 'no-store', headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...options.headers }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
        if (response.headers?.get('X-KZA-Demo')?.toLowerCase() === 'true') onDemo();
        if (!response.ok) {
          let detail;
          try { detail = (await response.json())?.detail; } catch (_) { /* Non-JSON errors still carry their HTTP status. */ }
          if (response.status === 401 && requestGeneration === sessionGeneration && !controller.signal.aborted) onUnauthorized();
          throw Object.assign(new Error(response.status === 401 ? 'La sesión no está autorizada.' : response.status === 403 ? 'La sesión no tiene permiso para esta operación.' : `El servicio no respondió correctamente (${response.status}).`), { status: response.status, detail });
        }
        return response.status === 204 ? null : await response.json();
      } finally {
        clearTimeout(timeout);
        options.signal?.removeEventListener('abort', abort);
      }
    }
    // Includes callers without their own AbortSignal, such as Casa actions.
    // Their late HTTP errors still reach the caller but cannot expire a new session.
    api.invalidateSession = () => { sessionGeneration++; };
    return api;
  }
  // The catalog is re-read at most every POLL_INTERVAL_MS. When the held
  // observation expires sooner, the next read starts early enough to land
  // before that expiry: the refresh budget is twice the slowest recent round
  // trip (at least REFRESH_LEAD_MIN_MS). Expiry itself is never extended, so
  // a refresh that misses the budget still leaves the data stale.
  const POLL_INTERVAL_MS = 5000, POLL_MIN_DELAY_MS = 1000, REFRESH_LEAD_MIN_MS = 1000, ROUND_TRIP_WINDOW = 3;
  function nextPollDelay(remainingMs, roundTripMs = 0) {
    if (!Number.isFinite(remainingMs)) return POLL_INTERVAL_MS;
    const lead = Math.max(REFRESH_LEAD_MIN_MS, 2 * (Number.isFinite(roundTripMs) ? roundTripMs : 0));
    return Math.max(POLL_MIN_DELAY_MS, Math.min(POLL_INTERVAL_MS, remainingMs - lead));
  }
  function createPoller({ load, onData, onError, onInvalidate = () => {}, deadline = () => Infinity, now = () => root.performance.now(), schedule = setTimeout, cancel = clearTimeout }) {
    let active = false, hidden = false, busy = false, sequence = 0, accepted = 0, timer, controller;
    const roundTrips = [];
    function clear() { if (timer !== undefined) cancel(timer); timer = undefined; }
    function invalidate() { sequence++; controller?.abort(); onInvalidate(); busy = false; clear(); }
    function delay() {
      let remaining;
      try { remaining = deadline(); } catch (_) { remaining = Infinity; }
      return nextPollDelay(remaining, Math.max(0, ...roundTrips));
    }
    async function refresh() {
      if (!active || hidden || busy) return;
      clear(); busy = true; const seq = ++sequence, started = now();
      controller = new AbortController();
      try {
        const data = await load(controller.signal);
        if (active && !hidden && seq === sequence && selectFreshResponse(accepted, seq)) { accepted = seq; onData(data); }
      } catch (error) {
        if (active && !hidden && seq === sequence && error.name !== 'AbortError') onError(error);
        // A timeout abort is still a failed observation, unless this poll was invalidated.
        else if (active && !hidden && seq === sequence) onError(new Error('La consulta agotó el tiempo de espera.'));
      } finally {
        if (seq === sequence) {
          busy = false;
          roundTrips.push(Math.max(0, now() - started)); if (roundTrips.length > ROUND_TRIP_WINDOW) roundTrips.shift();
          if (active && !hidden) timer = schedule(refresh, delay());
        }
      }
    }
    return {
      start() { if (active) return; active = true; refresh(); },
      stop() { active = false; invalidate(); }, refresh,
      setHidden(value) { if (hidden === value) return; hidden = value; if (hidden) invalidate(); else refresh(); }
    };
  }
  const { newRequestId } = typeof module !== 'undefined' && module.exports ? require('./request-id.js') : root.HomeRequestIdentity;
  function createActionController({ api, makeId = newRequestId, onChange = () => {} }) {
    const records = new Map();
    let generation = 0;
    const keyFor = (entity, operation) => `${entity}/${operation}`;
    function publish(key, record) { records.set(key, record); onChange(key, { ...record }); }
    async function receipt(key, record) {
      if (record.checking) return;
      const epoch = generation;
      record.checking = true;
      try {
        const result = await api(`/api/home/actions?request_id=${encodeURIComponent(record.request_id)}`);
        if (epoch !== generation) return;
        const status = ['confirmed', 'failed', 'uncertain', 'in_progress'].includes(result?.status) ? result.status : 'missing';
        publish(key, { ...record, status, effects: result.effects, pending: status === 'in_progress', checking: false });
      } catch (error) {
        if (epoch !== generation) return;
        publish(key, { ...record, status: error.status === 404 ? 'missing' : 'uncertain', pending: error.status !== 404, checking: false });
      }
    }
    return {
      get(entity, operation) { return records.get(keyFor(entity, operation)); },
      async execute(entity, operation, parameters) {
        const key = keyFor(entity, operation);
        if (records.get(key)?.pending) return;
        const epoch = generation;
        if (operation === 'brightness' && typeof parameters === 'number') parameters = { brightness_pct: parameters };
        const selected = Object.fromEntries((OPERATION_PARAMETERS[operation] || []).filter(name => parameters?.[name] !== undefined && parameters[name] !== null).map(name => [name, parameters[name]]));
        const record = { entity_id: entity, operation, parameters: selected, request_id: makeId(), pending: true, sending: true, status: 'in_progress' };
        publish(key, record);
        try {
          const body = { entity_id: entity, operation, request_id: record.request_id, ...selected };
          const result = await api('/api/home/actions', { method: 'POST', body });
          if (epoch !== generation) return;
          const status = ['confirmed', 'failed', 'uncertain', 'in_progress'].includes(result?.status) ? result.status : 'missing';
          publish(key, { ...record, status, effects: result.effects, sending: false, pending: status === 'in_progress' });
        } catch (error) {
          if (epoch !== generation) return;
          record.sending = false;
          // Even an HTTP error is not a receipt. Reconcile the original identity once.
          await receipt(key, record);
        }
      },
      async reconcile() { await Promise.all([...records].filter(([, record]) => record.pending && !record.sending).map(([key, record]) => receipt(key, record))); },
      clear() { generation++; records.clear(); }
    };
  }
  const exported = { OPERATION_LABELS, OPERATION_PARAMETERS, controlFields, validControlValue, selectFreshResponse, actionLabel, effectSummary, entityCanControl, entityObservationCurrent, createApi, createPoller, nextPollDelay, createActionController, observationMillis, createSnapshotClock, createFreshnessTicker };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  root.HomeUI = exported;
})(typeof globalThis !== 'undefined' ? globalThis : this);
