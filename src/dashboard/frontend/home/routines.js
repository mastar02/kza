(function (host) {
  'use strict';

  const PREFIX = 'kza:routine-attempt:';
  const TIMEZONE = 'America/Montevideo';
  const OPERATIONS = { turn_on: 'Encender', turn_off: 'Apagar', brightness: 'Ajustar brillo' };
  const TERMINAL = ['succeeded', 'failed', 'uncertain', 'skipped', 'interrupted'];
  const LABELS = { running: 'En curso', succeeded: 'Confirmada', confirmed: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta', not_executed: 'No ejecutada', skipped: 'Omitida por condición', interrupted: 'Interrumpida', pending: 'Pendiente', sending: 'Enviando', active: 'En curso', terminal: 'Finalizada', unresolved: 'Resultado pendiente de consulta', rejected: 'Intento rechazado', auth_required: 'Requiere iniciar sesión' };
  const copy = value => value == null ? value : JSON.parse(JSON.stringify(value));
  const entityList = catalog => Array.isArray(catalog?.entities) ? catalog.entities : [];
  const emptyDraft = () => ({ name: '', description: '', enabled: true, timezone: TIMEZONE, schedule: null, conditions: [], actions: [] });

  // Only these editable fields may leave the browser. Actor and execution context
  // belong to the authenticated server, including during preview.
  function definition(value) {
    return { name: value.name, description: value.description, enabled: value.enabled, timezone: value.timezone,
      schedule: value.schedule == null ? null : { at: value.schedule.at, weekdays: copy(value.schedule.weekdays) },
      conditions: (value.conditions || []).map(item => ({ entity_id: item.entity_id, state: item.state })),
      actions: (value.actions || []).map(item => ({ entity_id: item.entity_id, operation: item.operation, ...(item.operation === 'brightness' ? { brightness_pct: item.brightness_pct } : {}) })) };
  }

  function validateRoutineDraft(draft, catalog) {
    const errors = [], value = draft || {};
    if (typeof value.name !== 'string' || !value.name.trim()) errors.push('Escribí un nombre');
    if (typeof value.description !== 'string') errors.push('Revisá la descripción');
    if (typeof value.enabled !== 'boolean') errors.push('Elegí si la rutina está activa');
    if (value.timezone !== TIMEZONE) errors.push('La zona horaria debe ser America/Montevideo');
    if (value.schedule !== null) {
      if (!/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value.schedule?.at || '')) errors.push('Elegí una hora válida');
      const days = value.schedule?.weekdays;
      if (!Array.isArray(days) || !days.length || days.some(day => !Number.isInteger(day) || day < 0 || day > 6) || new Set(days).size !== days.length) errors.push('Elegí al menos un día, sin repetir');
    }
    if (!Array.isArray(value.conditions) || value.conditions.some(item => !item || typeof item.entity_id !== 'string' || !item.entity_id || typeof item.state !== 'string' || !item.state.trim())) errors.push('Completá la entidad y el estado de cada condición');
    if (!Array.isArray(value.actions) || !value.actions.length) errors.push('Agregá al menos una acción');
    else value.actions.forEach((item, index) => {
      const entity = entityList(catalog).find(entry => entry.entity_id === item?.entity_id);
      if (!item || typeof item.entity_id !== 'string' || !item.entity_id || !Object.hasOwn(OPERATIONS, item.operation) || (catalog && (!entity || !entity.capabilities?.includes(item.operation)))) errors.push(`Revisá la entidad y operación de la acción ${index + 1}`);
      if (item?.operation === 'brightness' && (!Number.isInteger(item.brightness_pct) || item.brightness_pct < 0 || item.brightness_pct > 100)) errors.push(`El brillo de la acción ${index + 1} debe estar entre 0 y 100`);
    });
    return { valid: errors.length === 0, errors };
  }

  function attemptAfterResponse(attempt, outcome) {
    if (outcome.kind === 'conflict') return { ...attempt, state: 'rejected' };
    if (outcome.kind === 'auth_required') return { ...attempt, state: 'auth_required' };
    const run = outcome.run;
    if (run && typeof run.run_id === 'string' && run.run_id && (run.status === 'running' || TERMINAL.includes(run.status))) {
      return { ...attempt, run_id: run.run_id, state: run.status === 'running' ? 'active' : 'terminal', run: copy(run) };
    }
    return { ...attempt, state: 'unresolved' };
  }

  let editorSession = null, editorGeneration = 0;
  const { newRequestId } = typeof module !== 'undefined' && module.exports ? require('./request-id.js') : host.HomeRequestIdentity;

  function clearRoutineSession(storage) {
    editorSession = null; editorGeneration++;
    if (!storage) return;
    const keys = [];
    for (let n = 0; n < storage.length; n++) { const key = storage.key(n); if (key?.startsWith(PREFIX)) keys.push(key); }
    keys.forEach(key => storage.removeItem(key));
  }

  /** Transport and attempt lifecycle shared by the DOM view and native Node tests. */
  function createRoutineController({ apiFetch, catalog = null, storage, makeId = newRequestId, editorState = null, onChange = () => {}, schedule = setTimeout, cancel = clearTimeout }) {
    let alive = true, visible = true, generation = 0, draftVersion = 0, listVersion = 0, timer, pollDelay = 1500, refreshing = false;
    const requests = new Set(), busy = new Set(), restored = new Set();
    const state = { routines: [], catalog, observedAt: null, error: null, loading: false, draft: null, editingId: null, editingRevision: null, message: '', preview: null, validation: [], attempts: {}, busy: [], saving: false, previewing: false };
    if (editorState) Object.assign(state, copy(editorState));
    function emit() { if (alive) { state.busy = [...busy]; onChange(state); } }
    function fail(error) {
      state.error = { status: error.status || null, message: error.status === 409 ? 'La rutina cambió o está en curso. Conservamos tu borrador; revisá antes de recargar.' : error.status === 401 ? 'Iniciá sesión para continuar.' : error.status === 403 ? 'La sesión no tiene permiso para esta operación.' : error.status === 422 ? 'El servidor rechazó la definición. Revisá sus campos; conservamos tu borrador.' : 'Servicio indisponible. Conservamos los últimos datos y tu borrador.' };
      state.message = '';
    }
    async function request(url, init = {}) {
      const controller = new AbortController(); requests.add(controller);
      try { return await apiFetch(url, { ...init, signal: controller.signal }); }
      finally { requests.delete(controller); }
    }
    function stopTimer() { if (timer !== undefined) cancel(timer); timer = undefined; }
    function planPoll() {
      stopTimer();
      if (!alive || !visible || [401, 403].includes(state.error?.status) || !state.routines.some(item => !busy.has(item.routine_id) && (state.attempts[item.routine_id]?.state === 'active' || item.latest_run?.status === 'running'))) return;
      timer = schedule(async () => {
        timer = undefined;
        for (const item of state.routines) {
          if (!alive) return;
          const latestRunning = item.latest_run?.status === 'running' ? item.latest_run.run_id : null;
          if (state.attempts[item.routine_id]?.state === 'active' || latestRunning) await consultAttempt(item.routine_id, true, latestRunning);
        }
        pollDelay = Math.min(pollDelay * 2, 15000); planPoll();
      }, pollDelay);
    }
    function restore(id) {
      if (restored.has(id)) return;
      restored.add(id);
      try {
        const raw = storage?.getItem(PREFIX + id);
        if (!raw) return;
        const value = JSON.parse(raw);
        if (typeof value.request_id !== 'string' || !value.request_id || (value.run_id !== null && typeof value.run_id !== 'string')) throw new Error('Invalid attempt');
        // A terminal flag in storage is not a fresh terminal receipt. Query it
        // before enabling a new identity, even if a previous page saw it finish.
        state.attempts[id] = { request_id: value.request_id, run_id: value.run_id, state: value.state === 'active' ? 'active' : 'unresolved' };
      } catch (error) { fail(error); state.attempts[id] = { request_id: null, run_id: null, state: 'unresolved' }; }
    }
    function persist(id, attempt) {
      if (!storage) throw new Error('Session storage unavailable');
      storage.setItem(PREFIX + id, JSON.stringify({ request_id: attempt.request_id, run_id: attempt.run_id, state: attempt.state }));
      state.attempts[id] = attempt;
    }
    function routine(id) { return state.routines.find(item => item.routine_id === id); }
    function locked(id) { const item = routine(id); return busy.has(id) || state.attempts[id]?.state === 'active' || item?.latest_run?.status === 'running'; }
    function replaceRoutine(value) {
      listVersion++;
      const index = state.routines.findIndex(item => item.routine_id === value.routine_id);
      if (index === -1) state.routines.push(value); else state.routines[index] = value;
    }
    async function refresh() {
      if (!alive || refreshing) return;
      refreshing = true; const ticket = generation, readVersion = listVersion; state.loading = true; emit();
      const results = await Promise.allSettled([request('/api/routines', { method: 'GET' }), state.catalog ? Promise.resolve(state.catalog) : request('/api/home/catalog', { method: 'GET' })]);
      if (!alive || ticket !== generation) return;
      refreshing = false; state.loading = false; state.error = null;
      if (results[0].status === 'fulfilled' && Array.isArray(results[0].value)) {
        if (readVersion === listVersion) {
          state.routines = results[0].value; state.observedAt = new Date().toISOString();
          state.routines.forEach(item => restore(item.routine_id));
        }
      } else fail(results[0].status === 'rejected' ? results[0].reason : new Error('Invalid routines response'));
      if (results[1].status === 'fulfilled' && Array.isArray(results[1].value?.entities)) state.catalog = results[1].value;
      else fail(results[1].status === 'rejected' ? results[1].reason : new Error('Invalid catalog response'));
      emit(); planPoll();
    }
    function newDraft() { if (!alive || state.saving) return; state.draft = emptyDraft(); state.editingId = null; state.editingRevision = null; state.message = ''; state.preview = null; state.validation = []; draftVersion++; emit(); }
    function edit(id) {
      if (!alive || locked(id) || state.saving || !routine(id)) return;
      const item = routine(id); state.draft = definition(item); state.editingId = id; state.editingRevision = item.revision; state.message = ''; state.preview = null; state.validation = []; draftVersion++; emit();
    }
    function setDraft(value) { if (!alive || state.saving || (state.editingId && locked(state.editingId))) return; state.draft = definition(value); state.message = ''; state.preview = null; state.validation = []; draftVersion++; }
    function validate() {
      state.validation = validateRoutineDraft(state.draft, state.catalog).errors;
      if (!state.catalog) state.validation.push('Actualizá el catálogo de dispositivos antes de continuar');
      if (state.validation.length) { emit(); return false; } return true;
    }
    async function save() {
      if (!alive || state.saving || !state.draft || (state.editingId && locked(state.editingId)) || !validate()) return;
      const id = state.editingId, revision = state.editingRevision, value = definition(state.draft), ticket = generation, version = draftVersion;
      state.saving = true; state.message = ''; state.error = null; emit();
      try {
        const saved = await request(id ? '/api/routines/' + encodeURIComponent(id) : '/api/routines', { method: id ? 'PUT' : 'POST', body: id ? { ...value, expected_revision: revision } : value });
        if (!alive || ticket !== generation) return;
        if (!saved?.routine_id || !Number.isInteger(saved.revision)) throw new Error('Missing persistence receipt');
        replaceRoutine(saved); state.message = 'Guardada';
        if (version === draftVersion) { state.draft = definition(saved); state.editingId = saved.routine_id; state.editingRevision = saved.revision; }
      } catch (error) { if (alive && ticket === generation) fail(error); }
      finally { if (alive && ticket === generation) { state.saving = false; emit(); } }
    }
    async function preview() {
      if (!alive || state.previewing || !state.draft || !validate()) return;
      const ticket = generation, version = draftVersion; state.previewing = true; state.preview = null; state.error = null; emit();
      try { const result = await request('/api/routines/preview', { method: 'POST', body: definition(state.draft) }); if (alive && ticket === generation && version === draftVersion) state.preview = result; }
      catch (error) { if (alive && ticket === generation) fail(error); }
      finally { if (alive && ticket === generation) { state.previewing = false; emit(); } }
    }
    async function mutate(id, operation) {
      if (!alive || locked(id) || !routine(id) || (state.saving && state.editingId === id)) return;
      const revision = routine(id).revision, ticket = generation; busy.add(id); state.error = null; emit();
      try {
        const result = await request('/api/routines/' + encodeURIComponent(id) + (operation === 'DELETE' ? '?expected_revision=' + encodeURIComponent(revision) : '/' + operation), { method: operation === 'DELETE' ? 'DELETE' : 'POST', ...(operation === 'DELETE' ? {} : { body: { expected_revision: revision } }) });
        if (!alive || ticket !== generation) return;
        if (operation === 'DELETE') { listVersion++; state.routines = state.routines.filter(item => item.routine_id !== id); if (state.editingId === id) { state.draft = null; state.editingId = null; } }
        else { if (!result?.routine_id || !Number.isInteger(result.revision)) throw new Error('Missing persistence receipt'); replaceRoutine(result); }
      } catch (error) { if (alive && ticket === generation) fail(error); }
      finally { if (alive && ticket === generation) { busy.delete(id); emit(); planPoll(); } }
    }
    async function sendAttempt(id, mode) {
      if (!alive || locked(id) || !routine(id) || (state.saving && state.editingId === id)) return;
      restore(id); let attempt = state.attempts[id];
      if (mode === 'first' && attempt) return;
      if (mode === 'again' && attempt?.state !== 'terminal') return;
      if (mode === 'retry' && (!attempt?.request_id || ['terminal', 'active'].includes(attempt.state))) return;
      const ticket = generation, latestAtSend = routine(id).latest_run?.run_id || null;
      try {
        if (mode !== 'retry') attempt = { request_id: makeId(), run_id: null, state: 'unresolved' };
        persist(id, attempt); // Must succeed before any side effect is sent.
      } catch (error) { fail(error); emit(); return; }
      busy.add(id); state.error = null; pollDelay = 1500; emit();
      try {
        const run = await request('/api/routines/' + encodeURIComponent(id) + '/execute', { method: 'POST', body: { request_id: attempt.request_id } });
        if (alive && ticket === generation) applyReceipt(id, attempt, run, mode === 'retry' ? undefined : latestAtSend);
      } catch (error) { if (alive && ticket === generation) applyFailure(id, attempt, error); }
      finally { if (alive && ticket === generation) { busy.delete(id); emit(); planPoll(); } }
    }
    function applyReceipt(id, attempt, run, replaceLatestId) {
      if (run?.routine_id !== id || (attempt.run_id && run.run_id !== attempt.run_id)) throw new Error('Receipt identity mismatch');
      const next = attemptAfterResponse(attempt, { kind: 'receipt', run });
      persist(id, next);
      const item = routine(id), latestId = item?.latest_run?.run_id;
      // Correlation with this manual attempt does not establish that its receipt
      // is the latest run. Only an own new POST can replace the unchanged baseline.
      if (item && next.run && (!latestId || latestId === run.run_id || (replaceLatestId !== undefined && latestId === replaceLatestId))) {
        listVersion++; item.latest_run = next.run;
      }
      state.error = null;
    }
    function applyFailure(id, attempt, error) {
      const kind = error.status === 409 ? 'conflict' : error.status === 401 ? 'auth_required' : 'network_error';
      const next = attemptAfterResponse(attempt, { kind });
      state.attempts[id] = next;
      try { persist(id, next); } catch (_) { /* Original identity remains stored. */ }
      fail(error);
    }
    async function consultAttempt(id, polling = false, trackedRunId = null) {
      if (!alive || (polling && !visible) || busy.has(id)) return;
      restore(id); const attempt = state.attempts[id], latest = routine(id)?.latest_run;
      if (!attempt?.request_id && !latest?.run_id) return;
      const ticket = generation, runId = trackedRunId || attempt?.run_id || latest?.run_id;
      const followsManual = attempt && (!trackedRunId || trackedRunId === attempt.run_id);
      busy.add(id); emit();
      try {
        const run = await request('/api/routines/' + encodeURIComponent(id) + (runId ? '/runs/' + encodeURIComponent(runId) : '/attempts?request_id=' + encodeURIComponent(attempt.request_id)), { method: 'GET' });
        if (!alive || ticket !== generation) return;
        if (followsManual) applyReceipt(id, attempt, run);
        else {
          if (run?.routine_id !== id || run.run_id !== runId || (run.status !== 'running' && !TERMINAL.includes(run.status))) throw new Error('Receipt identity mismatch');
          // A list refresh may already have observed a later execution while
          // this GET was pending. Never reinstall the previously tracked run.
          if (routine(id)?.latest_run?.run_id === runId) { listVersion++; routine(id).latest_run = run; }
          state.error = null;
        }
      } catch (error) { if (alive && ticket === generation) { if (followsManual) applyFailure(id, attempt, error); else fail(error); } }
      finally { if (alive && ticket === generation) { busy.delete(id); emit(); if (!polling) { pollDelay = 1500; planPoll(); } } }
    }
    function destroy() { if (!alive) return; alive = false; generation++; stopTimer(); requests.forEach(controller => controller.abort()); requests.clear(); }
    function setVisible(value) { visible = !!value; if (visible) planPoll(); else stopTimer(); }
    function clearSession() { destroy(); clearRoutineSession(storage); state.attempts = {}; state.draft = null; state.routines = []; state.catalog = null; }
    return { getState: () => copy(state), refresh, newDraft, edit, setDraft, save, preview, execute: id => sendAttempt(id, 'first'), executeAgain: id => sendAttempt(id, 'again'), retryAttempt: id => sendAttempt(id, 'retry'), consultAttempt,
      setEnabled: (id, enabled) => mutate(id, enabled ? 'enable' : 'disable'), remove: id => mutate(id, 'DELETE'), locked, setVisible, destroy, clearSession };
  }

  /** Mount once; refresh updates data without replacing an in-progress draft. */
  function mountRoutines(root, { apiFetch, catalog, storage = host.sessionStorage } = {}) {
    const doc = root.ownerDocument, listeners = [], ui = doc.createElement('section'), session = editorGeneration;
    let destroyed = false, rendering = false, list, feedback, previewPanel, controller;
    root.replaceChildren(ui);
    function node(tag, className, text) { const element = doc.createElement(tag); if (className) element.className = className; if (text !== undefined) element.textContent = String(text); return element; }
    function listen(element, event, callback) { element.addEventListener(event, callback); listeners.push(() => element.removeEventListener(event, callback)); }
    function button(text, callback, disabled = false, primary = false) { const element = node('button', 'button' + (primary ? ' primary' : ''), text); element.type = 'button'; element.disabled = disabled; listen(element, 'click', callback); return element; }
    function field(label, control) { const wrap = node('label', 'routine-field'); control.dataset.routineField = label; wrap.append(node('span', 'metric-label', label), control); return wrap; }
    function input(type, value, callback) { const element = node('input'); element.type = type; if (type === 'checkbox') element.checked = value; else element.value = value ?? ''; listen(element, 'input', () => callback(type === 'checkbox' ? element.checked : element.value)); return element; }
    function select(options, value, callback) { const element = node('select'); options.forEach(([id, label]) => { const option = node('option', '', label); option.value = id; element.append(option); }); element.value = value; listen(element, 'change', () => callback(element.value)); return element; }
    function update(mutator, redraw = false) {
      const value = controller.getState().draft; if (!value) return;
      mutator(value); controller.setDraft(value);
      feedback.textContent = 'Cambios sin guardar'; previewPanel?.remove(); previewPanel = null;
      if (redraw) render(controller.getState());
    }
    function entitySelect(value, catalogValue, callback, allowedOnly = false) {
      const options = [['', 'Elegí un dispositivo'], ...entityList(catalogValue).filter(item => !allowedOnly || item.capabilities?.some(cap => Object.hasOwn(OPERATIONS, cap))).map(item => [item.entity_id, item.name || item.entity_id])];
      if (value && !options.some(([id]) => id === value)) options.push([value, value + ' (fuera del catálogo)']);
      return select(options, value || '', callback);
    }
    function renderReceipt(run, target, preview = false) {
      if (!run) return;
      target.append(node('p', 'badge', preview ? 'Vista previa · sin guardar ni ejecutar' : LABELS[run.status] || 'Estado desconocido'));
      if (run.reason) target.append(node('p', 'fine-print', run.reason));
      const results = node('ol');
      for (const item of run.actions || []) {
        const entity = entityList(controller.getState().catalog).find(entry => entry.entity_id === item.entity_id);
        const line = node('li', 'action-result', (entity?.name || item.entity_id || 'Acción') + ' · ' + (preview ? (OPERATIONS[item.operation] || 'Prevista') : LABELS[item.status] || 'Estado desconocido'));
        if (item.reason) line.append(node('span', '', ' · ' + item.reason)); results.append(line);
      }
      target.append(results);
      if (!preview) target.append(node('p', 'fine-print', 'Confirmada indica el estado informado por Home Assistant; no comprueba el resultado físico.'));
    }
    function renderEditor(state) {
      if (!state.draft) return;
      const value = state.draft, form = node('form', 'room-card routine-editor');
      form.append(node('h3', '', state.editingId ? 'Editar rutina · revisión ' + state.editingRevision : 'Nueva rutina'));
      const fields = node('fieldset'); fields.disabled = state.saving || (state.editingId && controller.locked(state.editingId));
      fields.append(field('Nombre', input('text', value.name, text => update(d => { d.name = text; }))));
      fields.append(field('Descripción (opcional)', input('text', value.description, text => update(d => { d.description = text; }))));
      fields.append(field('Activa', input('checkbox', value.enabled, checked => update(d => { d.enabled = checked; }))));
      fields.append(field('Cuándo', select([['manual', 'Sólo manual'], ['scheduled', 'Horario y días']], value.schedule ? 'scheduled' : 'manual', mode => update(d => { d.schedule = mode === 'manual' ? null : { at: '22:00', weekdays: [0, 1, 2, 3, 4, 5, 6] }; }, true))));
      if (value.schedule) {
        fields.append(field('Hora · Montevideo', input('time', value.schedule.at, text => update(d => { d.schedule.at = text; }))));
        const days = node('div', 'entity-controls');
        ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'].forEach((day, index) => days.append(field(day, input('checkbox', value.schedule.weekdays.includes(index), checked => update(d => { d.schedule.weekdays = checked ? [...d.schedule.weekdays, index].sort() : d.schedule.weekdays.filter(n => n !== index); })))));
        fields.append(days);
      }
      const conditions = node('section', 'entity'); conditions.append(node('h3', '', 'Condiciones de estado (opcionales)'));
      value.conditions.forEach((item, index) => {
        const row = node('div', 'entity');
        row.append(field('Dispositivo de la condición ' + (index + 1), entitySelect(item.entity_id, state.catalog, id => update(d => { d.conditions[index].entity_id = id; }))));
        row.append(field('Estado requerido (por ejemplo, on)', input('text', item.state, text => update(d => { d.conditions[index].state = text; }))));
        row.append(button('Quitar condición ' + (index + 1), () => update(d => { d.conditions.splice(index, 1); }, true))); conditions.append(row);
      });
      conditions.append(button('Agregar condición', () => update(d => { d.conditions.push({ entity_id: '', state: 'on' }); }, true))); fields.append(conditions);
      const actions = node('section', 'entity'); actions.append(node('h3', '', 'Acciones en orden'));
      value.actions.forEach((item, index) => {
        const row = node('div', 'entity'), entity = entityList(state.catalog).find(entry => entry.entity_id === item.entity_id);
        const options = Object.entries(OPERATIONS).filter(([key]) => entity?.capabilities?.includes(key));
        if (item.operation && !options.some(([key]) => key === item.operation)) options.unshift([item.operation, (OPERATIONS[item.operation] || item.operation) + ' (no disponible)']);
        row.append(field('Dispositivo de la acción ' + (index + 1), entitySelect(item.entity_id, state.catalog, id => update(d => { const target = entityList(state.catalog).find(entry => entry.entity_id === id); d.actions[index] = { entity_id: id, operation: target?.capabilities?.find(cap => Object.hasOwn(OPERATIONS, cap)) || '' }; }, true), true)));
        row.append(field('Operación ' + (index + 1), select([['', 'Elegí una operación'], ...options], item.operation, operation => update(d => { d.actions[index] = { entity_id: item.entity_id, operation, ...(operation === 'brightness' ? { brightness_pct: 50 } : {}) }; }, true))));
        if (item.operation === 'brightness') { const control = input('number', item.brightness_pct, text => update(d => { d.actions[index].brightness_pct = text === '' ? null : Number(text); })); control.min = '0'; control.max = '100'; control.step = '1'; row.append(field('Brillo de la acción ' + (index + 1) + ' (%)', control)); }
        const controls = node('div', 'entity-controls');
        function move(offset) { update(d => { const entry = d.actions.splice(index, 1)[0]; d.actions.splice(index + offset, 0, entry); }, true); }
        controls.append(button('Subir acción ' + (index + 1), () => move(-1), index === 0), button('Bajar acción ' + (index + 1), () => move(1), index === value.actions.length - 1), button('Quitar acción ' + (index + 1), () => update(d => { d.actions.splice(index, 1); }, true))); row.append(controls); actions.append(row);
      });
      actions.append(button('Agregar acción', () => update(d => { d.actions.push({ entity_id: '', operation: '' }); }, true))); fields.append(actions);
      const controls = node('div', 'entity-controls');
      const save = button(state.saving ? 'Guardando…' : 'Guardar', () => {}, !state.catalog, true); save.type = 'submit';
      controls.append(save, button(state.previewing ? 'Consultando vista previa…' : 'Previsualizar', () => controller.preview(), state.previewing || !state.catalog)); fields.append(controls); form.append(fields);
      listen(form, 'submit', event => { event.preventDefault(); controller.save(); });
      if (state.validation.length) { const errors = node('ul', 'error-text'); errors.setAttribute('role', 'alert'); state.validation.forEach(text => errors.append(node('li', '', text))); form.append(errors); }
      if (state.error?.status === 409 && state.editingId) form.append(button('Descartar borrador y recargar', async () => { const id = state.editingId; await controller.refresh(); if (!controller.getState().error) controller.edit(id); }));
      if (state.preview) { previewPanel = node('section'); renderReceipt(state.preview, previewPanel, true); form.append(previewPanel); }
      ui.append(form);
    }
    function render(state) {
      if (destroyed || rendering) return; rendering = true;
      const focused = doc.activeElement, focusKey = root.contains(focused) ? focused.dataset?.routineField : null;
      const selection = focusKey && typeof focused.selectionStart === 'number' ? [focused.selectionStart, focused.selectionEnd] : null;
      // Detached controls lose their listeners; a mounted view keeps only the
      // current render's handlers, rather than retaining every prior form.
      listeners.splice(0).forEach(remove => remove()); ui.replaceChildren(); previewPanel = null;
      const heading = node('div', 'view-heading'); heading.append(node('h3', '', 'Rutinas de tu casa'), button('Crear rutina', () => controller.newDraft(), state.saving)); ui.append(heading);
      feedback = node('p', state.error ? 'connection-status warning' : 'connection-status'); feedback.setAttribute('role', 'status'); feedback.setAttribute('aria-live', 'polite');
      const timestamp = state.observedAt ? 'Última consulta: ' + new Date(state.observedAt).toLocaleString('es-UY') : 'Sin consulta confirmada';
      feedback.textContent = state.error ? state.error.message + ' ' + timestamp : state.message || (state.loading ? 'Consultando rutinas…' : timestamp); ui.append(feedback);
      renderEditor(state); list = node('div', 'room-grid');
      state.routines.forEach(item => {
        const card = node('article', 'room-card'), id = item.routine_id, attempt = state.attempts[id], locked = controller.locked(id);
        card.append(node('h3', '', item.name), node('p', 'muted', item.description || 'Sin descripción'), node('p', 'badge', item.enabled ? 'Activa' : 'Desactivada'), node('p', 'fine-print', 'Revisión ' + item.revision));
        card.append(node('p', 'fine-print', item.schedule ? item.schedule.at + ' · ' + item.schedule.weekdays.map(day => ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'][day]).join(', ') + ' · Montevideo' : 'Sólo manual'));
        const controls = node('div', 'entity-controls'); controls.append(button('Editar', () => controller.edit(id), locked), button(item.enabled ? 'Desactivar' : 'Activar', () => controller.setEnabled(id, !item.enabled), locked));
        // Deletion requires a second explicit gesture inside this card.
        const deletion = node('details'); deletion.append(node('summary', '', 'Borrar rutina'), button('Confirmar borrado', () => controller.remove(id), locked)); controls.append(deletion);
        if (!attempt) controls.append(button('Ejecutar', () => controller.execute(id), locked, true));
        else {
          card.append(node('p', 'action-result', LABELS[attempt.state] || 'Resultado desconocido'));
          if (attempt.state === 'terminal') controls.append(button('Ejecutar otra vez', () => controller.executeAgain(id), locked, true));
          else if (attempt.state !== 'active') controls.append(button('Reintentar este intento', () => controller.retryAttempt(id), locked || !attempt.request_id));
        }
        if (attempt || item.latest_run) controls.append(button('Consultar resultado', () => controller.consultAttempt(id), state.busy.includes(id)));
        card.append(controls);
        if (attempt?.run) { card.append(node('p', 'fine-print', 'Resultado de este intento manual')); renderReceipt(attempt.run, card); }
        if (item.latest_run && item.latest_run.run_id !== attempt?.run?.run_id) {
          card.append(node('p', 'fine-print', 'Última ejecución registrada')); renderReceipt(item.latest_run, card);
        }
        list.append(card);
      });
      if (!state.routines.length && !state.loading && !state.error) list.append(node('p', 'muted', 'Todavía no hay rutinas. Creá la primera con dispositivos de tu casa.'));
      ui.append(list);
      if (focusKey) {
        const next = [...ui.querySelectorAll('[data-routine-field]')].find(element => element.dataset.routineField === focusKey);
        if (next && !next.disabled) { next.focus({ preventScroll: true }); if (selection && next.setSelectionRange) next.setSelectionRange(...selection); }
      }
      rendering = false;
    }
    controller = createRoutineController({ apiFetch, catalog, storage, editorState: editorSession, onChange: render });
    const visibility = () => controller.setVisible(!doc.hidden);
    doc.addEventListener('visibilitychange', visibility); visibility();
    render(controller.getState()); controller.refresh();
    function destroy() {
      if (destroyed) return;
      if (session === editorGeneration) {
        const { draft, editingId, editingRevision } = controller.getState();
        editorSession = { draft, editingId, editingRevision };
      }
      destroyed = true; controller.destroy(); doc.removeEventListener('visibilitychange', visibility); listeners.splice(0).forEach(remove => remove()); root.replaceChildren(); }
    return { refresh: controller.refresh, clearSession() { controller.clearSession(); destroy(); }, destroy };
  }

  const exported = { mountRoutines, validateRoutineDraft, attemptAfterResponse, createRoutineController, clearRoutineSession };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  host.RoutinesUI = exported;
  host.mountRoutines = mountRoutines;
})(typeof globalThis !== 'undefined' ? globalThis : this);
