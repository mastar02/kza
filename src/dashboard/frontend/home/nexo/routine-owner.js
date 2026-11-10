/* NEXO routines owner. Reads may be aborted; writes never are, and their pending
 * identities survive reload until the session that owns them ends. */
(function (root) {
  'use strict';
  const isNode = typeof module !== 'undefined' && module.exports;
  const home = isNode ? require('../home.js') : root.HomeUI;
  const { newRequestId } = isNode ? require('../request-id.js') : root.HomeRequestIdentity;
  // Shared with legacy routines.js so either surface restores and wipes the other's attempts.
  const PREFIX = 'kza:routine-attempt:';
  const WRITE_PREFIX = PREFIX + '~write:';
  // Outside the attempt prefix so a session-end wipe can leave a content-free reminder.
  const ENDED_KEY = 'kza:routine-pending-ended';
  const TIMEZONE = 'America/Montevideo';
  const OPERATIONS = home.OPERATION_LABELS;
  const RUN_STATUSES = ['running', 'succeeded', 'failed', 'uncertain', 'skipped', 'interrupted'];
  const WRITE_KINDS = ['create', 'update', 'enable', 'disable', 'delete'];
  const CATALOG_STALE_MS = 300000, CATALOG_REFRESH_MS = 60000;
  const RETIRED = ['dashboard_session_retired', 'dashboard_authority_retired'];
  const copy = value => value == null ? value : JSON.parse(JSON.stringify(value));
  const entityList = catalog => Array.isArray(catalog?.entities) ? catalog.entities : [];
  const emptyDraft = () => ({ name: '', description: '', enabled: true, timezone: TIMEZONE, schedule: null, conditions: [], actions: [] });
  const string = value => typeof value === 'string';

  /** Editable fields only; actor, revision and execution context belong to the server. */
  function definition(value) {
    return { name: value.name, description: value.description, enabled: value.enabled, timezone: value.timezone,
      schedule: value.schedule == null ? null : { at: value.schedule.at, weekdays: Array.isArray(value.schedule.weekdays) ? [...value.schedule.weekdays].sort((a, b) => a - b) : value.schedule.weekdays },
      conditions: (value.conditions || []).map(item => ({ entity_id: item.entity_id, state: item.state })),
      actions: (value.actions || []).map(item => ({ entity_id: item.entity_id, operation: item.operation, ...Object.fromEntries((home.OPERATION_PARAMETERS[item.operation] || [])
        .filter(name => item[name] !== undefined && item[name] !== null).map(name => [name, item[name]])) })) };
  }
  const sameDefinition = (left, right) => JSON.stringify(definition(left)) === JSON.stringify(definition(right));

  function validateRoutineDraft(draft, catalog) {
    const errors = [], value = draft || {};
    if (!string(value.name) || !value.name.trim()) errors.push('Escribí un nombre.');
    if (!string(value.description)) errors.push('Revisá la descripción.');
    if (typeof value.enabled !== 'boolean') errors.push('Elegí si la rutina está activa.');
    if (value.timezone !== TIMEZONE) errors.push('La zona horaria debe ser America/Montevideo.');
    if (value.schedule !== null) {
      if (!/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value.schedule?.at || '')) errors.push('Elegí una hora válida.');
      const days = value.schedule?.weekdays;
      if (!Array.isArray(days) || !days.length || days.some(day => !Number.isInteger(day) || day < 0 || day > 6) || new Set(days).size !== days.length) errors.push('Elegí al menos un día, sin repetir.');
    }
    if (!Array.isArray(value.conditions)) errors.push('Revisá las condiciones.');
    else value.conditions.forEach((item, index) => {
      if (!item || !string(item.entity_id) || !item.entity_id || !string(item.state) || !item.state.trim()) errors.push(`Completá el dispositivo y el estado de la condición ${index + 1}.`);
    });
    if (!Array.isArray(value.actions) || !value.actions.length) errors.push('Agregá al menos una acción.');
    else value.actions.forEach((item, index) => {
      const entity = entityList(catalog).find(entry => entry.entity_id === item?.entity_id);
      if (!item || !string(item.entity_id) || !item.entity_id || !Object.hasOwn(OPERATIONS, item.operation) || (catalog && (!entity || !entity.capabilities?.includes(item.operation)))) errors.push(`Revisá el dispositivo y la operación de la acción ${index + 1}.`);
      if (item?.brightness != null && item?.brightness_pct != null) errors.push(`Elegí una única medida de brillo en la acción ${index + 1}.`);
      for (const field of home.controlFields(entity, item?.operation, item || {})) if (!home.validControlValue(field, item?.[field.name])) errors.push(`Revisá ${field.label.toLowerCase()} de la acción ${index + 1}.`);
    });
    return { valid: errors.length === 0, errors };
  }

  function parseRun(value) {
    if (!value || typeof value !== 'object' || !string(value.run_id) || !value.run_id || !string(value.routine_id) || !string(value.status)
        || !Array.isArray(value.actions)) throw Error('invalid_routine_run');
    const actions = value.actions.map(item => {
      if (!item || !Number.isInteger(item.index) || !string(item.entity_id) || !string(item.status)) throw Error('invalid_routine_run');
      return { index: item.index, entity_id: item.entity_id, status: item.status };
    });
    return { run_id: value.run_id, routine_id: value.routine_id, status: value.status, source: string(value.source) ? value.source : null,
      scheduled_for: string(value.scheduled_for) ? value.scheduled_for : null, reason: string(value.reason) ? value.reason : null, actions };
  }
  function parseRoutine(value) {
    const schedule = value?.schedule;
    if (!value || typeof value !== 'object' || !string(value.routine_id) || !value.routine_id || !Number.isSafeInteger(value.revision) || value.revision < 1
        || !string(value.name) || !string(value.description) || typeof value.enabled !== 'boolean' || !string(value.timezone)
        || !(schedule === null || (schedule && string(schedule.at) && Array.isArray(schedule.weekdays)))
        || !Array.isArray(value.conditions) || !Array.isArray(value.actions)) throw Error('invalid_routine');
    // created_by is disclosed by the server but never kept for presentation.
    return { routine_id: value.routine_id, revision: value.revision, ...definition(value), latest_run: value.latest_run == null ? null : parseRun(value.latest_run) };
  }

  function detailCode(error) {
    const detail = error?.detail;
    return string(detail) ? detail : string(detail?.code) ? detail.code : null;
  }
  function describeError(error) {
    if (error?.sent === false) return 'Este acceso no permite la operación.';
    if (error?.answered === true) return 'La sesión dejó de estar vigente y no pudimos leer la respuesta.';
    const status = error?.status, code = detailCode(error);
    if (status === 401) return 'Iniciá sesión para continuar.';
    if (status === 403) return RETIRED.includes(code) ? 'La sesión dejó de estar vigente.' : code === 'origin denied' ?
      'Este navegador no está autorizado para operar KZA desde este origen.' : 'La sesión no tiene permiso para esta operación.';
    if (status === 404) return 'La rutina ya no existe. Actualizá la lista.';
    if (status === 409) return ({ revision_conflict: 'La rutina cambió en otra sesión. Revisá antes de volver a guardar.', routine_busy: 'La rutina está en curso. Esperá a que termine.',
      routine_disabled: 'La rutina está desactivada.' })[code] || 'La rutina no aceptó el cambio; actualizá.';
    // A pydantic list or the store's string: never rendered raw.
    if (status === 422) return 'El servidor rechazó la definición. Revisá sus campos; conservamos tu borrador.';
    if (status === 503) return 'Rutinas no disponibles en este servidor.';
    if (Number.isInteger(status) && status >= 500) return 'El servidor no pudo completar la operación.';
    return 'No se pudo contactar al servidor.';
  }
  /** 'none' only when the answer proves this request_id has no claim; otherwise 'unknown'. */
  function executeOutcome(error, first) {
    if (error?.sent === false) return 'none';
    if (error?.answered === true || !Number.isInteger(error?.status)) return 'unknown';
    const status = error.status, code = detailCode(error);
    if (status === 404 || status === 422 || (status === 409 && ['routine_disabled', 'routine_busy'].includes(code))) return 'none';
    if (first && (status === 401 || status === 403)) return 'none';
    return 'unknown';
  }
  /** A server answer below 500 is definitive for a write: authorization and CAS run before commit. */
  function writeOutcome(error) {
    if (error?.sent === false) return 'rejected';
    return error?.answered !== true && Number.isInteger(error?.status) && error.status < 500 ? 'rejected' : 'unknown';
  }
  function riskyRun(run) {
    if (!run || run.status === 'running') return false;
    if (['uncertain', 'interrupted'].includes(run.status) || !RUN_STATUSES.includes(run.status)) return true;
    return run.status === 'failed' && run.actions.some(item => ['confirmed', 'uncertain', 'sending'].includes(item.status));
  }
  const aborted = error => error?.name === 'AbortError';
  const encode = encodeURIComponent;

  function createRoutineOwner({ api, session, storage = root.sessionStorage, makeId = newRequestId, schedule = setTimeout, cancel = clearTimeout, now = Date.now, onAccountChanged = () => {} }) {
    let epoch = 0, opened = false, timer, pollDelay = 1500, listVersion = 0, refreshing = null, again = false, serial = 0;
    const reads = new Set(), listeners = new Set(), busy = new Set(), consulting = new Set();
    let state = initial();
    function initial() {
      return { routines: [], list: { loading: false, loaded: false, observedAt: null, error: null }, catalog: null,
        catalogStatus: { loading: false, observedAt: null, error: null }, attempts: {}, writes: {}, notices: {}, draft: null, announcement: null };
    }
    function capabilities() {
      const manage = session.can('routines:manage');
      return { read: session.can('home:read'), manage, execute: manage && session.can('home:control') };
    }
    function emit() { for (const listener of [...listeners]) listener(); }
    function announce(text) { state.announcement = { serial: ++serial, text }; }
    function notice(id, text) { state.notices[id] = text || null; }
    function find(id) { return state.routines.find(item => item.routine_id === id); }
    function locked(id) {
      const routine = find(id), attempt = state.attempts[id];
      return busy.has(id) || Boolean(state.writes[id]) || attempt?.state === 'active' || routine?.latest_run?.status === 'running';
    }
    async function read(path, options = {}) {
      const controller = new AbortController(); reads.add(controller);
      try { return await api(path, { ...options, signal: controller.signal }); } finally { reads.delete(controller); }
    }
    function stopTimer() { if (timer !== undefined) cancel(timer); timer = undefined; }

    function storageKeys() {
      const keys = [];
      if (!storage) return keys;
      for (let index = 0; index < storage.length; index++) { const key = storage.key(index); if (key?.startsWith(PREFIX)) keys.push(key); }
      return keys;
    }
    function remove(key) { try { storage?.removeItem(key); } catch (_) { /* Storage is best effort when removing. */ } }
    // Attempts are actor-scoped on the server: every stored record names the account that made it.
    function subject() { return typeof session.subject === 'function' ? session.subject() : null; }
    /**
     * Asks the server which account the shared cookie names now. Another tab may have switched
     * accounts while this one was hidden; the cached session subject cannot tell.
     * Resolves 'same', 'changed' (another, no or an unidentified account) or 'unknown' (not asked).
     */
    async function confirmAccount() {
      const expected = subject();
      if (!expected) return 'changed';
      try {
        const identity = await read('/api/admin/auth/whoami');
        return identity?.ok === true && typeof identity.subject === 'string' && identity.subject === expected ? 'same' : 'changed';
      } catch (error) {
        return [401, 403].includes(error?.status) ? 'changed' : 'unknown';
      }
    }
    /** The account changed under this tab: send nothing, forget every identity, leave a reminder. */
    function accountChanged() {
      clear({ wipe: true });
      notice('session', 'La cuenta de esta pestaña cambió en otra pestaña. No se envió nada. Revisá el último resultado de tus rutinas antes de repetirlo.');
      emit();
      try { onAccountChanged(); } catch (_) { /* The page re-reads the session on its own. */ }
    }
    const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se envió nada. Probá de nuevo en un momento.';
    function persistAttempt(id, attempt) {
      if (!storage) throw Error('storage_unavailable');
      if (!attempt.subject) throw Error('subject_unavailable');
      storage.setItem(PREFIX + id, JSON.stringify({ request_id: attempt.request_id, run_id: attempt.run_id, state: attempt.state, subject: attempt.subject }));
    }
    function dropAttempt(id) { delete state.attempts[id]; remove(PREFIX + id); }
    function persistWrite(target, marker) {
      if (!storage) throw Error('storage_unavailable');
      if (!marker.subject) throw Error('subject_unavailable');
      storage.setItem(WRITE_PREFIX + target, JSON.stringify(marker));
    }
    function dropWrite(target) { delete state.writes[target]; remove(WRITE_PREFIX + target); }
    function restore() {
      let keys;
      try { keys = storageKeys(); } catch (_) { return; }
      const current = subject();
      let foreignPending = false;
      for (const key of keys) {
        try {
          const value = JSON.parse(storage.getItem(key));
          // Another account's (or a legacy, unbound) record is never restored, consulted or retried.
          if (!current || !value || value.subject !== current) {
            if (key.startsWith(WRITE_PREFIX) || value?.state !== 'terminal') foreignPending = true;
            throw Error('foreign');
          }
          if (key.startsWith(WRITE_PREFIX)) {
            const target = key.slice(WRITE_PREFIX.length);
            if (state.writes[target]) continue;
            if (!value || !WRITE_KINDS.includes(value.kind) || (value.kind !== 'create' && (!string(value.routine_id) || !Number.isSafeInteger(value.expected_revision)))) throw Error('invalid');
            state.writes[target] = { ...value, state: 'unknown', inFlight: false };
            notice(target, 'No sabemos si se aplicó el último cambio. Consultando…');
          } else {
            const id = key.slice(PREFIX.length);
            if (state.attempts[id]) continue;
            if (!value || !string(value.request_id) || !value.request_id || (value.run_id !== null && !string(value.run_id))) throw Error('invalid');
            // A stored terminal flag is not a fresh receipt; only 'active' survives as is.
            state.attempts[id] = { request_id: value.request_id, run_id: value.run_id, state: value.state === 'active' ? 'active' : 'unresolved', run: null, first: false, subject: value.subject };
          }
        } catch (_) { remove(key); }
      }
      if (foreignPending) notice('session', 'Una ejecución o un cambio de rutina de otra sesión quedó sin confirmar y no se puede consultar desde esta cuenta. Revisá el último resultado de tus rutinas antes de repetirlo.');
    }

    function resolveWrites() {
      for (const [target, write] of Object.entries(state.writes)) {
        if (write.inFlight || write.state !== 'unknown') continue;
        const routine = write.routine_id ? find(write.routine_id) : null;
        let outcome;
        if (write.kind === 'create') {
          const match = state.routines.find(item => !(write.known_ids || []).includes(item.routine_id) && sameDefinition(item, write.definition));
          outcome = match ? 'applied' : 'not_applied';
          if (match) write.routine_id = match.routine_id;
        } else if (write.kind === 'delete') {
          outcome = !routine ? 'applied' : routine.revision === write.expected_revision ? 'not_applied' : 'changed';
        } else if (!routine) outcome = 'changed';
        else if (routine.revision === write.expected_revision) outcome = 'not_applied';
        else if (routine.revision === write.expected_revision + 1 && (write.kind === 'update' ? sameDefinition(routine, write.definition) : routine.enabled === (write.kind === 'enable'))) outcome = 'applied';
        else outcome = 'changed';
        dropWrite(target);
        const text = outcome === 'applied' ? ({ create: 'La rutina se guardó.', update: 'El cambio se aplicó.', enable: 'La rutina quedó activa.', disable: 'La rutina quedó desactivada.', delete: 'La rutina se borró.' })[write.kind]
          : outcome === 'not_applied' ? 'El cambio no se aplicó. Podés volver a intentarlo.' : 'La rutina cambió en otra sesión. Revisá antes de volver a intentarlo.';
        notice(target, text);
        if (outcome === 'applied' && write.kind === 'create') notice(write.routine_id, text);
        announce(text);
        const draft = state.draft;
        if (draft && ((write.kind === 'create' && target === 'new' && !draft.editingId) || (write.kind === 'update' && draft.editingId === write.routine_id))) {
          draft.unknown = false;
          if (outcome === 'applied') { const saved = find(write.routine_id); state.draft = null; if (saved) announce(`Rutina guardada · revisión ${saved.revision}.`); }
          else { draft.message = text; draft.recoverable = !draft.open; draft.error = outcome === 'changed' ? { message: text, conflict: write.kind === 'update' } : null; }
        }
      }
    }
    function reconcile() {
      const ids = new Set(state.routines.map(item => item.routine_id));
      for (const id of Object.keys(state.attempts)) if (!ids.has(id) && !busy.has(id)) dropAttempt(id);
      resolveWrites();
    }

    async function readOnce(ticket) {
      const version = listVersion;
      state.list.loading = state.catalogStatus.loading = true; emit();
      const [list, catalog] = await Promise.allSettled([read('/api/routines'), read('/api/home/catalog')]);
      if (ticket !== epoch || !opened) return;
      state.list.loading = state.catalogStatus.loading = false;
      if (list.status === 'fulfilled') {
        try {
          if (!Array.isArray(list.value)) throw Error('invalid');
          const rows = list.value.map(parseRoutine);
          // A write applied while this read was pending is newer than the read: read again.
          if (version === listVersion) {
            state.routines = rows;
            state.list = { loading: false, loaded: true, observedAt: now(), error: null };
            reconcile();
          } else again = true;
        } catch (_) { state.list.error = { status: null, message: 'El servidor devolvió rutinas con un formato desconocido.' }; }
      } else if (!aborted(list.reason)) state.list.error = { status: list.reason?.status ?? null, message: describeError(list.reason) };
      if (catalog.status === 'fulfilled' && Array.isArray(catalog.value?.entities)) { state.catalog = catalog.value; state.catalogStatus = { loading: false, observedAt: now(), error: null }; }
      else if (catalog.status === 'rejected' && aborted(catalog.reason)) { /* Left the view. */ }
      else state.catalogStatus.error = { message: catalog.status === 'rejected' ? describeError(catalog.reason) : 'El catálogo no está disponible.' };
      emit(); planPoll();
    }
    /** Coalesces concurrent calls; a call made during a read waits for a fresh read. */
    function refresh() {
      if (!opened || !session.can('home:read')) return Promise.resolve();
      if (refreshing) { again = true; return refreshing; }
      const ticket = epoch;
      const current = (async () => {
        try {
          do { again = false; await readOnce(ticket); } while (again && ticket === epoch && opened);
        } finally { if (refreshing === current) refreshing = null; }
      })();
      refreshing = current;
      return current;
    }

    function needsPolling() {
      return state.routines.some(item => state.attempts[item.routine_id]?.state === 'active' || item.latest_run?.status === 'running');
    }
    function planPoll() {
      stopTimer();
      if (!opened || [401, 403].includes(state.list.error?.status) || !needsPolling()) return;
      const ticket = epoch;
      timer = schedule(async () => {
        timer = undefined;
        if (!opened || ticket !== epoch) return;
        if (capabilities().manage) {
          for (const item of [...state.routines]) {
            if (state.attempts[item.routine_id]?.state === 'active' || item.latest_run?.status === 'running') await consult(item.routine_id, true);
          }
        } else await refresh();
        if (!opened || ticket !== epoch) return;
        pollDelay = Math.min(pollDelay * 2, 15000); planPoll();
      }, pollDelay);
    }

    function applyReceipt(id, value, latestAtSend, attempt = state.attempts[id]) {
      const run = parseRun(value);
      if (run.routine_id !== id || (attempt?.run_id && run.run_id !== attempt.run_id)) throw Error('receipt_mismatch');
      if (attempt) {
        attempt.run_id = run.run_id; attempt.run = run; attempt.first = false;
        attempt.state = run.status === 'running' ? 'active' : RUN_STATUSES.includes(run.status) ? 'terminal' : 'unresolved';
        try { persistAttempt(id, attempt); } catch (_) { /* The receipt is in memory; storage keeps the identity. */ }
      }
      const routine = find(id), latest = routine?.latest_run?.run_id || null;
      // A receipt replaces the latest run only if nothing newer was observed since this send.
      if (routine && (!latest || latest === run.run_id || (latestAtSend !== undefined && latest === latestAtSend))) { routine.latest_run = run; listVersion++; }
      return run;
    }

    async function send(id, first) {
      const ticket = epoch, attempt = state.attempts[id], latestAtSend = find(id)?.latest_run?.run_id || null;
      busy.add(id); notice(id, null); pollDelay = 1500; emit();
      try {
        // Never aborted: leaving the view must not cut a request the server may commit.
        const value = await api('/api/routines/' + encode(id) + '/execute', { method: 'POST', body: { request_id: attempt.request_id } });
        if (ticket !== epoch) return { stale: true };
        const run = applyReceipt(id, value, latestAtSend);
        announce(run.status === 'running' ? 'La rutina se está ejecutando.' : 'La rutina terminó.');
        return { sent: true, run };
      } catch (error) {
        if (ticket !== epoch) return { stale: true };
        if (executeOutcome(error, first) === 'none') {
          dropAttempt(id);
          const text = 'No se ejecutó. ' + describeError(error);
          notice(id, text); announce(text);
          return { sent: true, executed: false };
        }
        const current = state.attempts[id];
        if (current) { current.state = 'unresolved'; current.first = false; try { persistAttempt(id, current); } catch (_) { /* Identity stored before sending. */ } }
        const text = 'Resultado pendiente de consulta. ' + describeError(error);
        notice(id, text); announce(text);
        return { sent: true, uncertain: true };
      } finally {
        if (ticket === epoch) { busy.delete(id); emit(); planPoll(); }
      }
    }
    function riskyActions(id) {
      const runs = [find(id)?.latest_run, state.attempts[id]?.run].filter(riskyRun);
      if (!runs.length) return null;
      const seen = new Set(), actions = [];
      for (const run of runs) for (const item of run.actions) {
        if (item.status === 'confirmed' || seen.has(item.index + ':' + item.entity_id)) continue;
        seen.add(item.index + ':' + item.entity_id); actions.push({ index: item.index, entity_id: item.entity_id, status: item.status });
      }
      return actions;
    }
    async function execute(id, { confirmed = false } = {}) {
      const routine = find(id), attempt = state.attempts[id];
      if (!routine || !capabilities().execute || locked(id) || (attempt && attempt.state !== 'terminal')) return { sent: false };
      const risky = riskyActions(id);
      if (risky && !confirmed) return { sent: false, needsConfirmation: risky };
      const next = { request_id: makeId(), run_id: null, state: 'unresolved', run: null, first: true, subject: subject() };
      try { persistAttempt(id, next); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el intento; no se envió.';
        notice(id, text); announce(text); emit(); return { sent: false };
      }
      state.attempts[id] = next;
      return send(id, true);
    }
    async function retryAttempt(id) {
      const attempt = state.attempts[id];
      if (!capabilities().execute || busy.has(id) || !attempt?.request_id || attempt.state !== 'unresolved') return { sent: false };
      if (!attempt.subject || attempt.subject !== subject()) {
        // Never resend another account's identity: the server would treat it as a new run.
        dropAttempt(id); notice(id, 'Este intento pertenece a otra cuenta y se descartó. Revisá el último resultado antes de ejecutar.'); emit();
        return { sent: false };
      }
      // The cookie may name another account than the cached session: confirm before resending.
      const ticket = epoch;
      busy.add(id); notice(id, null); emit();
      let account;
      try { account = await confirmAccount(); } finally { if (ticket === epoch) busy.delete(id); }
      if (ticket !== epoch || state.attempts[id] !== attempt) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      if (account === 'unknown') { notice(id, ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      return send(id, false);
    }
    /** Polls are quiet (no disabled controls); a user consult marks the routine busy. */
    async function consult(id, polling = false) {
      if (!opened || consulting.has(id) || busy.has(id) || !find(id)) return;
      if (!capabilities().manage) return refresh();
      const attempt = state.attempts[id], latest = find(id).latest_run;
      // A poll follows an active own attempt, otherwise the running latest receipt.
      const followAttempt = Boolean(attempt) && (!polling || attempt.state === 'active');
      const runId = followAttempt ? attempt.run_id : latest?.run_id;
      if (!runId && !(followAttempt && attempt.request_id)) return;
      const ticket = epoch;
      consulting.add(id);
      if (!polling) { busy.add(id); emit(); }
      try {
        if (followAttempt) {
          // An own attempt is only consulted under the account that made it.
          const account = await confirmAccount();
          if (ticket !== epoch) return;
          if (account === 'changed') { accountChanged(); return; }
          if (account === 'unknown') { if (!polling) notice(id, ACCOUNT_UNKNOWN); return; }
          if (state.attempts[id] !== attempt) return;
        }
        const value = await read('/api/routines/' + encode(id) + (runId ? '/runs/' + encode(runId) : '/attempts?request_id=' + encode(attempt.request_id)));
        if (ticket !== epoch) return;
        if (followAttempt) {
          // A newer own attempt started meanwhile: this receipt belongs to the old one.
          if (state.attempts[id] !== attempt) return;
          const before = attempt.state;
          applyReceipt(id, value, undefined, attempt);
          if (before === 'active' && attempt.state === 'terminal') announce('La ejecución terminó.');
        } else {
          const run = parseRun(value), routine = find(id);
          if (run.routine_id !== id || run.run_id !== runId) throw Error('receipt_mismatch');
          if (routine?.latest_run?.run_id === runId) { routine.latest_run = run; listVersion++; }
        }
        if (!polling) notice(id, null);
      } catch (error) {
        if (ticket !== epoch || aborted(error)) return;
        if (error?.status === 404 && followAttempt && !attempt.run_id) notice(id, 'El servidor no registró este intento para esta cuenta. Podés reintentarlo: antes de enviarlo se confirma la cuenta y se usa el mismo identificador.');
        else notice(id, 'No se pudo consultar el resultado. ' + describeError(error));
      } finally {
        if (ticket === epoch) { consulting.delete(id); if (!polling) busy.delete(id); emit(); if (!polling) { pollDelay = 1500; planPoll(); } }
      }
    }

    async function write(kind, target, path, options, input) {
      const ticket = epoch, marker = { ...input, subject: subject() };
      try { persistWrite(target, marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado del cambio; no se envió.';
        notice(target, text); announce(text); emit(); return { ok: false, sent: false };
      }
      state.writes[target] = { ...marker, state: 'pending', inFlight: true };
      // Any list read already in flight may predate this commit: it must not resolve the outcome.
      listVersion++;
      busy.add(target); notice(target, null); emit();
      let value, failure = null;
      // Never aborted by leave(): the response is applied whenever it arrives.
      try { value = await api(path, options); } catch (error) { failure = error; }
      if (ticket !== epoch) return { ok: false, stale: true };
      busy.delete(target);
      try {
        if (failure) throw failure;
        if (kind === 'delete') {
          if (value?.success !== true) throw Error('missing_receipt');
          listVersion++; state.routines = state.routines.filter(item => item.routine_id !== marker.routine_id); dropAttempt(marker.routine_id);
        } else {
          const routine = parseRoutine({ ...value, latest_run: value?.latest_run ?? find(value?.routine_id)?.latest_run ?? null });
          if (kind !== 'create' && routine.routine_id !== marker.routine_id) throw Error('receipt_mismatch');
          const index = state.routines.findIndex(item => item.routine_id === routine.routine_id);
          listVersion++;
          if (index === -1) state.routines.push(routine); else state.routines[index] = routine;
          value = routine;
        }
        dropWrite(target);
        const text = ({ create: 'Rutina guardada.', update: 'Cambios guardados.', enable: 'La rutina quedó activa.', disable: 'La rutina quedó desactivada.', delete: 'La rutina se borró.' })[kind];
        notice(kind === 'create' ? value.routine_id : target, text); announce(kind === 'create' || kind === 'update' ? `${text} Revisión ${value.revision}.` : text);
        emit(); return { ok: true, routine: kind === 'delete' ? null : value };
      } catch (error) {
        if (writeOutcome(error) === 'rejected') {
          dropWrite(target);
          const text = describeError(error);
          notice(target, text); announce(text); emit();
          return { ok: false, error, code: detailCode(error), status: error.status };
        }
        state.writes[target] = { ...marker, state: 'unknown', inFlight: false };
        // Only a read issued from now on may decide the outcome; in-flight reads are discarded.
        listVersion++;
        const text = 'No sabemos si se aplicó. Consultando…';
        notice(target, text); announce(text); emit();
        void refresh();
        return { ok: false, unknown: true };
      }
    }

    function catalogUsable() {
      return Boolean(state.catalog) && !state.catalogStatus.error && state.catalogStatus.observedAt !== null && now() - state.catalogStatus.observedAt <= CATALOG_STALE_MS;
    }
    function openDraft(value, id = null, revision = null) {
      state.draft = { value, editingId: id, editingRevision: revision, validation: [], preview: null, message: '', error: null, saving: false, previewing: false, unknown: false, open: true };
      if (!catalogUsable() || now() - (state.catalogStatus.observedAt ?? 0) > CATALOG_REFRESH_MS) void refresh();
      emit();
    }
    /** A kept (hidden) draft is only replaced when the caller confirmed it. */
    function hiddenDraft() { return Boolean(state.draft && !state.draft.open); }
    function newDraft({ replace = false } = {}) {
      if (!capabilities().manage || state.draft?.saving || state.draft?.unknown || state.writes.new) return false;
      if (hiddenDraft() && !replace) return false;
      notice('new', null); openDraft(emptyDraft()); return true;
    }
    function edit(id, { replace = false } = {}) {
      const routine = find(id);
      if (!capabilities().manage || !routine || locked(id) || state.draft?.saving || state.draft?.unknown) return false;
      if (hiddenDraft() && !replace) return false;
      // A new edit supersedes the previous outcome message for this routine.
      notice(id, null); openDraft(definition(routine), id, routine.revision); return true;
    }
    function setDraft(value) {
      const draft = state.draft;
      if (!draft || draft.saving || draft.unknown) return;
      draft.value = definition(value); draft.preview = null; draft.message = '';
      if (draft.validation.length) draft.validation = [];
      emit();
    }
    /** Closing during an uncertain save keeps the draft (hidden) so it can be recovered if not applied. */
    function closeEditor() {
      const draft = state.draft;
      if (!draft || draft.saving) return;
      if (draft.unknown) draft.open = false; else state.draft = null;
      emit();
    }
    function reopenDraft() { if (state.draft && !state.draft.open) { state.draft.open = true; state.draft.recoverable = false; emit(); } }
    function discardDraft() { if (state.draft && !state.draft.saving && !state.draft.unknown) { state.draft = null; emit(); } }
    function validateDraft() {
      const draft = state.draft;
      draft.validation = validateRoutineDraft(draft.value, state.catalog).errors;
      if (!catalogUsable()) draft.validation.push('Actualizá el catálogo de dispositivos antes de guardar.');
      return draft.validation.length === 0;
    }
    async function save() {
      const draft = state.draft;
      if (!draft || draft.saving || draft.unknown || !capabilities().manage) return { ok: false };
      if (draft.editingId && locked(draft.editingId)) return { ok: false };
      if (!validateDraft()) { emit(); return { ok: false, invalid: true }; }
      const body = definition(draft.value), id = draft.editingId;
      draft.saving = true; draft.error = null; draft.message = ''; emit();
      const result = id
        ? await write('update', id, '/api/routines/' + encode(id), { method: 'PUT', body: { ...body, expected_revision: draft.editingRevision } }, { kind: 'update', routine_id: id, expected_revision: draft.editingRevision, definition: body })
        : await write('create', 'new', '/api/routines', { method: 'POST', body }, { kind: 'create', routine_id: null, expected_revision: null, definition: body, known_ids: state.routines.map(item => item.routine_id) });
      if (result.stale || state.draft !== draft) return result;
      draft.saving = false;
      if (result.ok) { state.draft = null; emit(); return result; }
      if (result.unknown) { draft.unknown = true; draft.message = 'No sabemos si se guardó. Consultando…'; }
      else draft.error = { message: result.sent === false ? 'El navegador no permite guardar el estado del cambio; no se envió.' : describeError(result.error), conflict: result.status === 409 && result.code === 'revision_conflict' };
      emit(); return result;
    }
    async function discardAndReload() {
      const draft = state.draft;
      if (!draft?.editingId) return;
      const id = draft.editingId;
      state.draft = null; await refresh();
      if (find(id)) edit(id); else emit();
    }
    async function preview() {
      const draft = state.draft;
      if (!draft || draft.previewing || !capabilities().manage) return;
      if (!validateDraft()) { emit(); return { ok: false, invalid: true }; }
      const ticket = epoch, body = definition(draft.value);
      draft.previewing = true; draft.preview = null; draft.error = null; emit();
      try {
        const value = await read('/api/routines/preview', { method: 'POST', body });
        if (ticket !== epoch || state.draft !== draft) return;
        if (!Array.isArray(value?.actions)) throw Error('invalid_preview');
        if (sameDefinition(draft.value, body)) draft.preview = definition(value);
      } catch (error) {
        if (ticket !== epoch || state.draft !== draft || aborted(error)) return;
        draft.error = { message: error?.message === 'invalid_preview' ? 'La vista previa no está disponible.' : describeError(error), conflict: false };
      } finally { if (ticket === epoch && state.draft === draft) { draft.previewing = false; emit(); } }
    }
    function setEnabled(id, enabled) {
      const routine = find(id);
      if (!capabilities().manage || !routine || locked(id) || routine.enabled === enabled) return Promise.resolve({ ok: false });
      const kind = enabled ? 'enable' : 'disable';
      return write(kind, id, '/api/routines/' + encode(id) + '/' + kind, { method: 'POST', body: { expected_revision: routine.revision } }, { kind, routine_id: id, expected_revision: routine.revision });
    }
    function removeRoutine(id) {
      const routine = find(id);
      if (!capabilities().manage || !routine || locked(id)) return Promise.resolve({ ok: false });
      return write('delete', id, '/api/routines/' + encode(id) + '?expected_revision=' + encode(routine.revision), { method: 'DELETE' }, { kind: 'delete', routine_id: id, expected_revision: routine.revision });
    }

    /** Opening (also on returning to a hidden tab) confirms the account before using any stored identity. */
    async function open() {
      opened = true; restore();
      try {
        if (storage?.getItem(ENDED_KEY)) { remove(ENDED_KEY); notice('session', 'Una ejecución o un cambio de rutina quedó sin confirmar al terminar la sesión anterior. Revisá el último resultado de tus rutinas antes de repetirlo.'); }
      } catch (_) { /* Reminder is best effort. */ }
      emit();
      // Without a cached subject nothing can be sent or restored: no check, and no re-bootstrap loop.
      if (subject()) {
        const ticket = epoch, account = await confirmAccount();
        if (ticket !== epoch || !opened) return true;
        if (account === 'changed') { accountChanged(); return true; }
      }
      void refresh();
      return true;
    }
    /** Stop reads and polling; keep drafts, attempts and markers. Resolves immediately. */
    function leave() {
      opened = false; stopTimer(); again = false;
      for (const controller of reads) controller.abort();
      reads.clear(); refreshing = null;
      return Promise.resolve(true);
    }
    /** Session boundary: forget everything; wipe stored identities unless the page is only being hidden. */
    function clear({ wipe = true } = {}) {
      epoch++; leave(); busy.clear(); consulting.clear(); state = initial(); pollDelay = 1500; listVersion++;
      if (wipe) {
        let keys = [];
        try { keys = storageKeys(); } catch (_) { /* Nothing to wipe. */ }
        // A wiped unconfirmed run or change cannot be consulted by the next account: leave only a reminder.
        const pending = keys.some(key => { try { return key.startsWith(WRITE_PREFIX) || JSON.parse(storage.getItem(key))?.state !== 'terminal'; } catch (_) { return true; } });
        keys.forEach(remove);
        if (pending) { try { storage.setItem(ENDED_KEY, '1'); } catch (_) { /* Reminder is best effort. */ } }
      }
      emit();
      return Promise.resolve(true);
    }
    function getState() {
      return { ...copy(state), busy: [...busy], capabilities: capabilities(), catalogUsable: catalogUsable(), locked: Object.fromEntries(state.routines.map(item => [item.routine_id, locked(item.routine_id)])) };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, subscribe, getState, capabilities, newDraft, edit, setDraft, closeEditor, reopenDraft, discardDraft, save, preview, discardAndReload,
      setEnabled, remove: removeRoutine, execute, retryAttempt, consult, riskyActions, locked, hiddenDraft, isOpen: () => opened };
  }

  const exported = { ENDED_KEY, createRoutineOwner, validateRoutineDraft, definition, sameDefinition, describeError, executeOutcome, writeOutcome, parseRoutine, parseRun, riskyRun, emptyDraft, PREFIX, WRITE_PREFIX, TIMEZONE };
  root.NexoRoutineOwner = exported;
  if (isNode) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
