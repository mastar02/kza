/* KZA home — classic script, also loadable by the native Node test runner. */
(function (root) {
  'use strict';
  function selectFreshResponse(currentSeq, incomingSeq) { return incomingSeq >= currentSeq; }
  function actionLabel(status) {
    return ({ confirmed: 'Confirmado por HA', failed: 'Falló', uncertain: 'Sin confirmar', missing: 'Resultado desconocido', in_progress: 'En curso' })[status] || 'Resultado desconocido';
  }
  function known(value, fallback = 'Desconocido') { return value === null || value === undefined || value === '' ? fallback : String(value); }
  function observationMillis(value) {
    if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value * 1000 : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const parsed = Date.parse(value); return Number.isFinite(parsed) ? parsed : null;
  }
  function createSnapshotClock(data, { monotonic = () => root.performance.now(), wallClock = Date.now } = {}) {
    const receivedMono = monotonic(), receivedWall = wallClock();
    const observed = observationMillis(data.observed_at);
    // Casa has no server TTL in schema v1: six seconds is the explicit local
    // freshness budget. Only observed_at anchors evidence, never last_changed.
    const ttlMs = (Number.isFinite(data.stale_after_s) && data.stale_after_s > 0 ? data.stale_after_s : 6) * 1000;
    const elapsed = () => Math.max(0, monotonic() - receivedMono);
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
      age, directionFreshness,
      read() {
        const ageMs = observed === null ? null : Math.max(0, receivedWall - observed) + elapsed();
        const stale = data.stale === true || ageMs === null || ageMs >= ttlMs;
        // The existing presentation timer also wakes at each direction deadline.
        // No sample sequence or refresh completion renews the observation time.
        const directionDeadlines = (data.microphones || []).filter(mic => ['observed', 'no_speech'].includes(mic.direction?.state)).map(mic => directionFreshness(mic.direction).remainingMs);
        return { ageMs, unknown: ageMs === null, stale, remainingMs: Math.min(stale ? Infinity : ttlMs - ageMs, ...directionDeadlines) };
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
  function formatAge(ageMs) {
    if (ageMs === null || !Number.isFinite(ageMs)) return 'Sin fecha de observación';
    const seconds = Math.max(0, Math.floor(ageMs / 1000));
    return seconds < 60 ? `hace ${seconds} s` : seconds < 3600 ? `hace ${Math.floor(seconds / 60)} min` : `hace ${Math.floor(seconds / 3600)} h`;
  }
  function directionLabel(direction) {
    if (!direction || direction.state === 'unverified') return 'Dirección no verificada';
    if (direction.state === 'disabled') return 'Observación de dirección desactivada';
    if (direction.state === 'stale') return 'Dirección desactualizada';
    if (direction.state === 'no_speech') return 'Sin habla detectada';
    if (direction.state !== 'observed' || !Number.isFinite(direction.relative_angle_deg)) return 'Dirección desconocida';
    return `${Math.round(direction.relative_angle_deg)}° respecto del micrófono`;
  }
  function describeMicrophone(mic) {
    const capture = mic.capture || {};
    return {
      signal: capture.nonzero_seen === true ? 'Señal no-cero observada' : capture.nonzero_seen === false ? 'Sin señal no-cero observada' : 'Sin comprobar',
      frames: known(capture.frames_received, 'Desconocidos'),
      voice: mic.voice?.detected === true ? 'Voz detectada' : mic.voice?.detected === false ? 'Sin voz detectada' : 'Sin verificar',
      direction: directionLabel(mic.direction),
      // Metadata never certifies the coordinated physical walkthrough.
      physical: 'Pendiente de prueba humana'
    };
  }
  function entityCanControl(entity, operation, stale) {
    return !stale && entity.available === true && ['on', 'off'].includes(entity.state) && Array.isArray(entity.capabilities) && entity.capabilities.includes(operation);
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
  function createPoller({ load, onData, onError, onInvalidate = () => {}, schedule = setTimeout, cancel = clearTimeout }) {
    let active = false, hidden = false, busy = false, sequence = 0, accepted = 0, timer, controller;
    function clear() { if (timer !== undefined) cancel(timer); timer = undefined; }
    function invalidate() { sequence++; controller?.abort(); onInvalidate(); busy = false; clear(); }
    async function refresh() {
      if (!active || hidden || busy) return;
      clear(); busy = true; const seq = ++sequence;
      controller = new AbortController();
      try {
        const data = await load(controller.signal);
        if (active && !hidden && seq === sequence && selectFreshResponse(accepted, seq)) { accepted = seq; onData(data); }
      } catch (error) {
        if (active && !hidden && seq === sequence && error.name !== 'AbortError') onError(error);
        // A timeout abort is still a failed observation, unless this poll was invalidated.
        else if (active && !hidden && seq === sequence) onError(new Error('La consulta agotó el tiempo de espera.'));
      } finally {
        if (seq === sequence) { busy = false; if (active && !hidden) timer = schedule(refresh, 5000); }
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
        publish(key, { ...record, status, pending: status === 'in_progress', checking: false });
      } catch (error) {
        if (epoch !== generation) return;
        publish(key, { ...record, status: error.status === 404 ? 'missing' : 'uncertain', pending: error.status !== 404, checking: false });
      }
    }
    return {
      get(entity, operation) { return records.get(keyFor(entity, operation)); },
      async execute(entity, operation, brightness) {
        const key = keyFor(entity, operation);
        if (records.get(key)?.pending) return;
        const epoch = generation;
        const record = { entity_id: entity, operation, request_id: makeId(), pending: true, sending: true, status: 'in_progress' };
        publish(key, record);
        try {
          const body = { entity_id: entity, operation, request_id: record.request_id };
          if (operation === 'brightness') body.brightness_pct = brightness;
          const result = await api('/api/home/actions', { method: 'POST', body });
          if (epoch !== generation) return;
          const status = ['confirmed', 'failed', 'uncertain', 'in_progress'].includes(result?.status) ? result.status : 'missing';
          publish(key, { ...record, status, sending: false, pending: status === 'in_progress' });
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
  function createViewLifecycle() {
    let handle;
    function destroy() { const previous = handle; handle = null; previous?.destroy?.(); }
    return {
      mount(renderer, container, context) { destroy(); handle = renderer(container, context) || null; },
      refresh() { return handle?.refresh?.(); }, destroy
    };
  }
  function createScriptLoader({ document: doc, src, getValue }) {
    let pending;
    return function load() {
      if (pending) return pending;
      pending = new Promise((resolve, reject) => {
        const script = doc.createElement('script'); script.src = src; script.async = true;
        function failed() { script.remove(); pending = null; reject(new Error('No se pudo cargar la vista.')); }
        script.addEventListener('load', () => { const value = getValue(); if (value) resolve(value); else failed(); }, { once: true });
        script.addEventListener('error', failed, { once: true }); doc.head.append(script);
      });
      return pending;
    };
  }
  // The shell receives a synchronous lifecycle handle while the classic script
  // loads. Leaving or logging out invalidates the continuation, not shared code.
  function createLazyView({ load, onLoading = () => {}, onError = () => {} }) {
    return function renderLazy(container, context) {
      let alive = true, handle, pending;
      function begin() {
        if (!alive || pending) return pending;
        onLoading(container);
        pending = Promise.resolve().then(load).then(renderer => { if (alive) handle = renderer(container, context) || {}; })
          .catch(error => { if (alive) onError(container, error); }).finally(() => { pending = null; });
        return pending;
      }
      begin();
      return { refresh() { return handle ? handle.refresh?.() : begin(); }, destroy() { if (!alive) return; alive = false; handle?.destroy?.(); } };
    };
  }
  function clearRoutineSession(storage = root.sessionStorage) {
    if (root.RoutinesUI?.clearRoutineSession) { root.RoutinesUI.clearRoutineSession(storage); return; }
    if (!storage) return;
    // Logout must clear attempts even before the lazy module was downloaded.
    const keys = [];
    for (let index = 0; index < storage.length; index++) { const key = storage.key(index); if (key?.startsWith('kza:routine-attempt:')) keys.push(key); }
    for (const key of keys) storage.removeItem(key);
  }
  const views = new Map();
  let application = null;
  const exported = { selectFreshResponse, actionLabel, directionLabel, describeMicrophone, entityCanControl, createApi, createPoller, createActionController, createViewLifecycle, createLazyView, clearRoutineSession, observationMillis, createSnapshotClock, createFreshnessTicker,
    /** Register a synchronous renderer(container, context), mounted once per activation.
     * It may return {refresh, destroy}. Refresh delegates without remounting; destroy
     * runs before leaving/replacing the view, on logout, and on pagehide. The view
     * owns listener/poll cleanup; keep drafts in its registration closure to survive
     * navigation. There is no shell poll for extensions. API uses JSON body objects
     * and same-origin HttpOnly session cookies. No token storage is provided.
     */
    registerView(id, renderer) {
      if (typeof id !== 'string' || !/^[a-z][a-z0-9-]*$/.test(id) || typeof renderer !== 'function' || ['house', 'microphones'].includes(id)) throw new Error('Vista no válida');
      views.set(id, renderer); application?.register(id);
    },
    api: createApi({ fetch: (...args) => root.fetch(...args), onDemo: () => application?.markDemo(), onUnauthorized: () => application?.sessionExpired() })
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  root.HomeUI = exported;

  function boot() {
    const doc = root.document;
    const byId = id => doc.getElementById(id);
    if (!byId('workspace')) return;
    function node(tag, className, text) {
      const element = doc.createElement(tag); if (className) element.className = className;
      if (text !== undefined) element.textContent = String(text); return element;
    }
    function button(text, handler, className = 'button') {
      const element = node('button', className, text); element.type = 'button'; element.addEventListener('click', handler); return element;
    }
    function badge(text, tone = '') { return node('span', `badge ${tone}`, text); }
    function age(timestamp) {
      return formatAge(stateFor().clock?.age(timestamp) ?? null);
    }
    let current = 'house', poller, authenticated = false, demo = false;
    let temporalBindings = [];
    const viewLifecycle = createViewLifecycle();
    const cache = new Map();
    const freshnessTicker = createFreshnessTicker({ read: () => {
      const currentFreshness = freshnessFor(stateFor());
      const microphoneFreshness = current === 'house' ? freshnessFor(cache.get('microphones') || {}) : null;
      return { ...currentFreshness, remainingMs: Math.min(currentFreshness.remainingMs, microphoneFreshness?.remainingMs ?? Infinity) };
    }, onTick: updatePresentation });
    const actions = createActionController({ api: exported.api, onChange: () => { if (current === 'house' && authenticated) render(); } });
    function markDemo() { demo = true; byId('demo-banner').hidden = false; }
    function showLogin(message) {
      exported.api.invalidateSession();
      authenticated = false; poller?.stop(); freshnessTicker.stop(); temporalBindings = []; viewLifecycle.destroy(); clearRoutineSession(); cache.clear(); actions.clear(); byId('content').replaceChildren();
      byId('workspace').hidden = true; byId('login-panel').hidden = false; byId('logout').hidden = true;
      byId('login-error').textContent = message || ''; byId('token').value = '';
    }
    function stateFor() { return cache.get(current) || { data: null, error: null, loading: true }; }
    function freshnessFor(state) {
      const freshness = state.clock?.read() || { stale: !!state.data, unknown: !!state.data, ageMs: null, remainingMs: Infinity };
      return state.error ? { ...freshness, stale: true } : freshness;
    }
    function isStale(state) { return freshnessFor(state).stale; }
    function bindTemporal(update) {
      temporalBindings.push(update); update(freshnessFor(stateFor()));
    }
    function updatePresentation(freshness = freshnessFor(stateFor())) {
      const state = stateFor(), stale = freshness.stale;
      const status = byId('connection-status'); status.replaceChildren(node('span', 'status-dot')); status.className = `connection-status${stale ? ' warning' : ''}`;
      let message = state.data ? `${demo ? 'Datos sintéticos' : 'Última observación'} · ${formatAge(freshness.ageMs)}` : 'Consultando el estado…';
      if (state.error) message = state.data ? `Datos desactualizados · ${formatAge(freshness.ageMs)}. No se pudo actualizar. ${state.error.message}` : `No se pudo consultar. ${state.error.message}`;
      else if (state.data && freshness.unknown) message = 'Antigüedad desconocida · sin fecha de observación válida. Actualizá antes de actuar.';
      else if (stale) message = `Datos desactualizados · ${formatAge(freshness.ageMs)}. Actualizá antes de actuar.`;
      status.append(node('span', '', message));
      for (const update of temporalBindings) update(freshness);
    }
    function empty(title, text, loading = false) {
      const panel = node('div', 'empty-state'); panel.append(node('span', 'section-icon', loading ? '…' : '⌂'), node('h3', '', title), node('p', '', text));
      if (loading) panel.append(node('div', 'loading-line')); return panel;
    }
    function render() {
      const state = stateFor(), stale = isStale(state), content = byId('content');
      temporalBindings = [];
      content.setAttribute('aria-busy', state.loading ? 'true' : 'false'); content.replaceChildren();
      byId('connection-status').hidden = views.has(current);
      updatePresentation(); freshnessTicker.refresh();
      if (views.has(current)) {
        const house = cache.get('house');
        viewLifecycle.mount(views.get(current), content, { api: exported.api, refresh: () => viewLifecycle.refresh(), catalog: house && !isStale(house) ? house.data : null, data: state.data, error: state.error, loading: state.loading, stale }); return;
      }
      if (!state.data) {
        content.append(state.error ? empty('No pudimos obtener el estado', 'Comprobá la conexión y tocá Actualizar. No se muestran datos de ejemplo.') : empty('Conectando con tu casa', 'Estamos consultando los dispositivos configurados.', true)); if (current === 'house') renderMicrophoneSummary(content); return;
      }
      if (current === 'house') { renderHouse(content, state.data); renderMicrophoneSummary(content); } else renderMicrophones(content, state.data);
    }
    function renderEntity(entity) {
      const card = node('div', 'entity'); card.append(node('div', 'entity-name', known(entity.name)), node('div', 'entity-id', known(entity.entity_id)));
      const row = node('div', 'entity-row');
      const state = entity.available !== true ? 'No disponible' : ({ on: 'Encendido', off: 'Apagado', unavailable: 'No disponible', unknown: 'Desconocido' })[entity.state] || 'Desconocido';
      const stateBadge = badge(state);
      bindTemporal(freshness => { stateBadge.className = `badge${!freshness.stale && entity.available === true && entity.state === 'on' ? ' good' : ''}`; });
      row.append(stateBadge);
      const controls = node('div', 'entity-controls');
      for (const [operation, label] of [['turn_on', 'Encender'], ['turn_off', 'Apagar']]) {
        if (!entity.capabilities?.includes(operation)) continue;
        const control = button(label, () => {
          if (entityCanControl(entity, operation, isStale(stateFor())) && !actions.get(entity.entity_id, operation)?.pending) actions.execute(entity.entity_id, operation);
        });
        control.setAttribute('aria-label', `${label} ${known(entity.name)}`);
        bindTemporal(freshness => { control.disabled = !entityCanControl(entity, operation, freshness.stale) || !!actions.get(entity.entity_id, operation)?.pending; });
        controls.append(control);
      }
      row.append(controls); card.append(row);
      if (entity.capabilities?.includes('brightness')) {
        const brightness = node('div', 'brightness'), input = node('input'); input.type = 'range'; input.min = '0'; input.max = '100'; input.step = '1';
        const valid = Number.isFinite(entity.brightness_pct) && entity.brightness_pct >= 0 && entity.brightness_pct <= 100;
        input.value = valid ? entity.brightness_pct : 50; // A starting choice is not an observed value.
        input.setAttribute('aria-label', `Elegir brillo de ${known(entity.name)}`);
        const output = node('output', '', valid ? `${entity.brightness_pct}%` : 'Sin dato');
        let selected = false;
        const allowed = () => entityCanControl(entity, 'brightness', isStale(stateFor())) && !actions.get(entity.entity_id, 'brightness')?.pending;
        const apply = button('Aplicar', () => { if (selected && allowed()) actions.execute(entity.entity_id, 'brightness', Number(input.value)); });
        bindTemporal(() => { input.disabled = !allowed(); apply.disabled = !selected || !allowed(); });
        input.addEventListener('input', () => { selected = true; output.textContent = `${input.value}% elegido`; apply.disabled = !allowed(); });
        brightness.append(node('label', '', 'Brillo · elegí un valor y aplicalo'), input, output, apply); card.append(brightness);
      }
      for (const operation of ['turn_on', 'turn_off', 'brightness']) {
        const record = actions.get(entity.entity_id, operation);
        if (!record) continue;
        const text = node('p', `action-result ${record.status === 'confirmed' ? 'confirmed' : ''}`, `${({ turn_on: 'Encender', turn_off: 'Apagar', brightness: 'Brillo' })[operation]} · ${actionLabel(record.status)}`);
        text.setAttribute('role', 'status');
        if (['uncertain', 'missing'].includes(record.status)) text.append(node('span', '', '. Actualizá el estado antes de decidir. Una nueva orden necesita otro toque.'));
        card.append(text);
      }
      return card;
    }
    function renderHouse(content, data) {
      const rooms = Array.isArray(data.rooms) ? data.rooms : [], entities = Array.isArray(data.entities) ? data.entities : [];
      const unassigned = (data.unassigned_entities || []).map(item => typeof item === 'string' ? entities.find(entity => entity.entity_id === item) : item).filter(Boolean);
      const groups = rooms.map(room => ({ name: room.name, entities: entities.filter(entity => (room.entity_ids || []).includes(entity.entity_id) || entity.room_id === room.room_id) }));
      if (unassigned.length) groups.push({ name: 'Sin ambiente', entities: unassigned });
      if (!groups.length) { content.append(empty('Tu casa todavía no tiene dispositivos', data.reason ? `El catálogo informa: ${data.reason}` : 'Los ambientes y entidades aparecerán cuando estén configurados en Home Assistant.')); return; }
      const grid = node('div', 'room-grid');
      for (const group of groups) {
        const card = node('section', 'room-card'), head = node('div', 'room-head'), title = node('div');
        title.append(node('h3', '', known(group.name, 'Ambiente sin nombre')), node('span', 'count', `${group.entities.length} ${group.entities.length === 1 ? 'dispositivo' : 'dispositivos'}`));
        head.append(title, node('span', 'room-symbol', '⌂')); card.append(head);
        for (const entity of group.entities) card.append(renderEntity(entity));
        if (!group.entities.length) card.append(node('p', 'room-empty', 'Sin dispositivos asignados a este ambiente.'));
        grid.append(card);
      }
      content.append(grid);
      if (data.reason) content.append(node('p', 'fine-print', `Información del catálogo: ${data.reason}`));
    }
    function renderMicrophoneSummary(content) {
      const summary = node('section', 'room-card'), evidence = node('div'), status = node('p', 'fine-print');
      summary.append(node('h3', '', 'Micrófonos de la casa'), status, evidence, button('Ver Micrófonos', () => activate('microphones')));
      let displayed = null;
      bindTemporal(() => {
        const snapshot = cache.get('microphones');
        if (snapshot !== displayed) {
          displayed = snapshot; evidence.replaceChildren();
          const microphones = snapshot?.data?.microphones || [];
          for (const mic of microphones) {
            const description = describeMicrophone(mic);
            evidence.append(node('p', '', `${known(mic.room_name, 'Sin ambiente')} · ${description.frames} frames · ${description.signal}. Voz: ${description.voice}. Prueba física: ${description.physical}.`));
          }
          if (!microphones.length) evidence.append(node('p', '', 'Sin evidencia de micrófonos disponible. No confirma una desconexión física.'));
        }
        const freshness = freshnessFor(snapshot || {});
        status.textContent = !snapshot?.data ? 'Telemetría indisponible · sin observación confirmada.' : freshness.unknown ? 'Antigüedad desconocida · no confirma la captura actual.' : freshness.stale ? `Telemetría desactualizada · ${formatAge(freshness.ageMs)}. No confirma la conexión ni la captura actual.` : `Telemetría del propietario de captura · ${formatAge(freshness.ageMs)}. La señal no-cero no verifica voz ni funcionamiento físico.`;
      });
      content.append(summary);
    }
    function renderMicrophones(content, data) {
      const microphones = Array.isArray(data.microphones) ? data.microphones : [];
      if (!microphones.length) content.append(empty('Sin micrófonos para mostrar', 'No hay telemetría de micrófonos configurados. La ausencia de datos no confirma una desconexión física.'));
      const grid = node('div', 'microphone-grid');
      for (const mic of microphones) {
        const description = describeMicrophone(mic), capture = mic.capture || {}, card = node('section', 'mic-card'), head = node('div', 'room-head'), title = node('div');
        title.append(node('h3', '', known(mic.room_name, 'Sin ambiente')), node('span', 'count', known(mic.product, 'Micrófono configurado')));
        head.append(title, node('span', 'room-symbol', '♩')); card.append(head);
        card.append(node('div', 'mic-id', known(mic.mic_id)));
        const metrics = node('div', 'mic-grid');
        const labels = { connected: 'Conectado', disconnected: 'Desconectado', matched: 'Coincide', mismatch: 'No coincide', unassigned: 'Sin ambiente', closed: 'Cerrada', waiting: 'Esperando callbacks', receiving: 'Recibiendo frames', stale: 'Sin callbacks recientes', unknown: 'Desconocido', unavailable: 'No disponible', ambiguous: 'Identidad ambigua', unverified: 'Sin verificar' };
        for (const [label, value] of [['Conexión USB', labels[mic.connection] || 'Desconocida'], ['Asociación', labels[mic.binding] || 'Desconocida'], ['Captura', labels[capture.state] || 'Desconocida'], ['Señal no-cero', description.signal], ['Frames recibidos', description.frames], ['Último callback', age(capture.last_callback_at)]]) {
          const metric = node('div', 'metric'), valueNode = node('span', 'metric-value', value);
          if (label === 'Último callback') bindTemporal(() => { valueNode.textContent = age(capture.last_callback_at); });
          metric.append(node('span', 'metric-label', label), valueNode); metrics.append(metric);
        }
        card.append(metrics);
        const verification = node('div', 'verification');
        for (const [label, value] of [['Voz', description.voice], ['Prueba física', description.physical]]) { const line = node('p'); line.append(node('span', '', label), node('span', '', value)); verification.append(line); }
        card.append(verification);
        const direction = mic.direction, directionPanel = node('div', 'direction-panel');
        const directionText = node('span', 'direction-label'), arrow = node('span', 'direction-arrow', '↑');
        arrow.hidden = true; arrow.setAttribute('aria-hidden', 'true');
        const directionHeading = node('p', 'direction-heading'); directionHeading.append(node('span', '', 'Dirección'), arrow, directionText); directionPanel.append(directionHeading);
        const supportText = ({ unknown: 'Desconocido', supported: 'Compatible', unsupported: 'No compatible' })[direction?.support] || 'Desconocido';
        const version = direction?.version;
        const versionText = Array.isArray(version) && version.length === 3 && version.every(value => Number.isInteger(value) && value >= 0) ? `${version.join('.')} · informativa` : 'Desconocida';
        const directionAge = node('span', 'direction-age'), calibrated = node('span', 'direction-calibrated'); calibrated.hidden = true;
        for (const [label, valueNode] of [['Soporte', node('span', 'direction-support', supportText)], ['Versión', node('span', 'direction-version', versionText)], ['Última muestra válida', directionAge]]) {
          const line = node('p'); line.append(node('span', '', label), valueNode); directionPanel.append(line);
        }
        directionPanel.append(calibrated, node('p', 'fine-print', 'La dirección no identifica personas ni un punto físico exacto. Su estabilidad no verifica su exactitud; requiere comprobación física.'));
        bindTemporal(freshness => {
          const sample = stateFor().clock?.directionFreshness(direction) || { ageMs: null, stale: true };
          const expired = freshness.stale || sample.stale;
          const visible = expired && ['observed', 'no_speech'].includes(direction?.state) ? { ...direction, state: 'stale' } : direction;
          const observed = visible?.state === 'observed' && Number.isFinite(visible.relative_angle_deg);
          directionText.textContent = directionLabel(visible);
          directionAge.textContent = sample.ageMs === null ? 'Sin muestra válida' : formatAge(sample.ageMs);
          arrow.hidden = !observed;
          arrow.style.transform = observed ? `rotate(${visible.relative_angle_deg}deg)` : '';
          calibrated.hidden = !(observed && Number.isFinite(visible.calibrated_room_angle) && visible.calibration_id != null);
          calibrated.textContent = calibrated.hidden ? '' : `${Math.round(visible.calibrated_room_angle)}° respecto de la sala · calibración vinculada`;
        });
        card.append(directionPanel);
        const identity = node('details', 'identity'); identity.append(node('summary', '', 'Identidad y detalles técnicos'));
        const list = node('dl');
        for (const [label, value] of [['Puerto configurado', mic.configured_usb_port], ['Puerto observado', mic.usb_port], ['Serie', mic.serial], ['ALSA', mic.alsa_card], ['PortAudio', mic.portaudio_index], ['Identidad observada', age(mic.identity_observed_at)], ['Generación', capture.generation], ['Frecuencia (Hz)', capture.sample_rate_hz], ['Canales', capture.channels], ['Canal seleccionado', capture.selected_channel], ['Formato', capture.host_dtype], ['Última señal no-cero', age(capture.last_nonzero_at)]]) {
          const valueNode = node('dd', '', known(value));
          if (label === 'Identidad observada') bindTemporal(() => { valueNode.textContent = age(mic.identity_observed_at); });
          if (label === 'Última señal no-cero') bindTemporal(() => { valueNode.textContent = age(capture.last_nonzero_at); });
          list.append(node('dt', '', label), valueNode);
        }
        identity.append(list); card.append(identity);
        const expired = node('p', 'action-result', 'Telemetría desactualizada. No confirma la conexión ni la captura actual.');
        bindTemporal(freshness => { expired.hidden = !freshness.stale; }); card.append(expired);
        grid.append(card);
      }
      content.append(grid);
      if (data.unconfigured_rooms?.length) content.append(node('p', 'unconfigured', `Ambientes sin micrófono configurado: ${data.unconfigured_rooms.map(room => typeof room === 'string' ? room : room.name || room.room_name || room.room_id).join(', ')}.`));
      const guide = node('section', 'guide'); guide.append(node('h3', '', 'Comprobación guiada, ambiente por ambiente'), node('p', '', 'Los metadatos pueden mostrar captura y señal no-cero; eso no prueba voz inteligible, dirección correcta ni funcionamiento físico. Este asistente no graba ni activa equipos.'));
      const steps = node('ol');
      for (const text of ['Elegí un ambiente y confirmá físicamente qué micrófono le corresponde. Compará el puerto y la identidad observada.', 'Coordiná la prueba con una persona presente. Hablá cerca del equipo y observá si cambian los frames, el callback y la señal no-cero.', 'Comprobá por separado la voz y la dirección con el procedimiento acordado. La señal no-cero puede ser ruido.', 'Registrá la evidencia y el resultado humano fuera de esta pantalla. La validación permanece pendiente hasta esa comprobación.']) steps.append(node('li', '', text));
      guide.append(steps, badge('Prueba física pendiente · sin verificación automática', 'warning')); content.append(guide);
    }
    function activate(id) {
      if (!['house', 'microphones'].includes(id) && !views.has(id)) return;
      poller?.stop(); freshnessTicker.stop(); temporalBindings = []; viewLifecycle.destroy(); current = id;
      for (const element of byId('view-nav').querySelectorAll('button')) {
        const selected = element.dataset.view === id; element.classList.toggle('active', selected);
        if (selected) element.setAttribute('aria-current', 'page'); else element.removeAttribute('aria-current');
      }
      byId('view-title').textContent = id === 'house' ? 'Casa' : id === 'microphones' ? 'Micrófonos' : id === 'routines' ? 'Rutinas' : id;
      byId('view-kicker').textContent = id === 'house' ? 'AMBIENTES Y DISPOSITIVOS' : id === 'microphones' ? 'IDENTIDAD, CAPTURA Y VERIFICACIÓN' : 'CONTROL DE LA CASA';
      const previous = cache.get(id); cache.set(id, { data: previous?.data || null, clock: previous?.clock, error: previous?.error || null, loading: !previous?.data }); render();
      if (views.has(id)) return;
      if (!doc.hidden) freshnessTicker.start();
      let microphoneRequest = null;
      function invalidateMicrophones() { microphoneRequest?.abort(); microphoneRequest = null; }
      async function refreshMicrophones() {
        if (microphoneRequest) return;
        const request = new AbortController(); microphoneRequest = request;
        const currentRequest = () => microphoneRequest === request && !request.signal.aborted;
        try {
          const data = await exported.api('/api/home/microphones', { signal: request.signal });
          if (!currentRequest()) return;
          cache.set('microphones', { data, clock: createSnapshotClock(data), error: null, loading: false });
          freshnessTicker.refresh();
        } catch (error) {
          if (!currentRequest()) return;
          const previous = cache.get('microphones');
          cache.set('microphones', { data: previous?.data || null, clock: previous?.clock, error, loading: false });
          freshnessTicker.refresh();
        } finally { if (microphoneRequest === request) microphoneRequest = null; }
      }
      poller = createPoller({
        onInvalidate: invalidateMicrophones,
        async load(signal) {
          if (id !== 'house') return exported.api('/api/home/microphones', { signal });
          // The same poll timer renews each source only when it is idle. A slow
          // diagnostic request never delays publication or renewal of Casa.
          refreshMicrophones();
          const catalog = await exported.api('/api/home/catalog', { signal });
          if (signal.aborted) throw Object.assign(new Error('Aborted'), { name: 'AbortError' });
          await actions.reconcile();
          return catalog;
        },
        onData(data) { authenticated = true; byId('logout').hidden = false; cache.set(id, { data, clock: createSnapshotClock(data), error: null, loading: false }); render(); },
        onError(error) { if (error.status === 401) { showLogin('Iniciá sesión para consultar tu casa.'); return; } const previous = cache.get(id); cache.set(id, { data: previous?.data || null, clock: previous?.clock, error, loading: false }); render(); }
      });
      poller.setHidden(doc.hidden); poller.start();
    }
    const loadRoutines = createScriptLoader({ document: doc, src: '/home/routines.js', getValue: () => root.RoutinesUI?.mountRoutines });
    exported.registerView('routines', createLazyView({
      load: async () => { const mount = await loadRoutines(); return (container, context) => mount(container, { apiFetch: context.api, catalog: context.catalog }); },
      onLoading(container) { container.replaceChildren(empty('Abriendo Rutinas', 'Cargando el editor de tu casa.', true)); },
      onError(container) { container.replaceChildren(empty('No pudimos abrir Rutinas', 'Tocá Actualizar para volver a intentar.')); }
    }));
    application = {
      markDemo, sessionExpired() { showLogin('Iniciá sesión para consultar tu casa.'); },
      register(id) {
        if (byId('view-nav').querySelector(`[data-view="${id}"]`)) return;
        const control = button(id === 'routines' ? 'Rutinas' : id, () => activate(id), 'view-button'); control.dataset.view = id; byId('view-nav').append(control);
      }
    };
    for (const id of views.keys()) application.register(id);
    for (const control of byId('view-nav').querySelectorAll('[data-view]')) {
      if (['house', 'microphones'].includes(control.dataset.view)) control.addEventListener('click', () => activate(control.dataset.view));
    }
    byId('refresh').addEventListener('click', () => { actions.reconcile(); if (views.has(current)) viewLifecycle.refresh(); else poller?.refresh(); });
    doc.addEventListener('visibilitychange', () => {
      if (doc.hidden) freshnessTicker.stop();
      else if (!byId('workspace').hidden && !views.has(current)) freshnessTicker.start();
      // start() synchronously updates expiry and age before resuming the GET.
      poller?.setHidden(doc.hidden);
    });
    byId('login-form').addEventListener('submit', async event => {
      event.preventDefault(); const submit = event.currentTarget.querySelector('button'); submit.disabled = true; byId('login-error').textContent = '';
      const token = byId('token').value; byId('token').value = '';
      try { await exported.api('/api/admin/auth/login', { method: 'POST', body: { token } }); exported.api.invalidateSession(); viewLifecycle.destroy(); clearRoutineSession(); authenticated = true; byId('workspace').hidden = false; byId('login-panel').hidden = true; byId('logout').hidden = false; activate(current); }
      catch (error) { byId('login-error').textContent = error.status === 401 ? 'El token no es válido. Revisalo e intentá de nuevo.' : error.message; }
      finally { submit.disabled = false; }
    });
    byId('logout').addEventListener('click', async () => {
      byId('logout').disabled = true;
      try { await exported.api('/api/admin/auth/logout', { method: 'POST' }); showLogin('Sesión cerrada.'); }
      catch (error) { byId('connection-status').textContent = `No se pudo cerrar la sesión. ${error.message}`; }
      finally { byId('logout').disabled = false; }
    });
    root.addEventListener('pagehide', () => { poller?.stop(); freshnessTicker.stop(); temporalBindings = []; viewLifecycle.destroy(); });
    root.addEventListener('pageshow', event => { if (event.persisted && !byId('workspace').hidden) activate(current); });
    activate(current);
  }
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true }); else boot();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
