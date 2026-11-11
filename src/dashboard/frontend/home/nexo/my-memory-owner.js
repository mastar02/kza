/* NEXO 'Mis recuerdos' owner (no DOM), for a household account bound to a person.
 * Same rules as the Memoria owner, with no shared state: reads may be aborted,
 * writes never are; a pending operation is bound to the account that sent it and
 * stored without any fact content; only a read that started after a loss, or the
 * answer to a resend, settles it. A read walks every keyset page of
 * GET /api/memory/mine, so absence from a complete walk is evidence. */
(function (root) {
  'use strict';
  const PREFIX = 'kza:my-memory-op:';
  const ENDED_KEY = 'kza:my-memory-op-ended';
  const PAGE = 100;
  const MAX_PAGES = 100;
  const OPS = ['delete', 'return'];
  const FACT_ID = /^fact_[0-9a-f]{64}$/;
  const CURSOR = /^k1\.[0-9a-f]{16}\.fact_[0-9a-f]{64}$/;
  const STATES = ['automatic', 'confirmed', 'proposed'];
  const METHODS = ['operator_assignment', 'person_session', 'voice_proposal'];
  // memory_api C7 projection: exactly these fields; anything else (sources, subjects, counters) fails closed.
  const ITEM_KEYS = ['category', 'content', 'created_at', 'expires_at', 'has_hidden_dependents', 'id', 'owned_dependents', 'owner_assigned_at', 'owner_method', 'state'];
  const NOT_APPLIED_503 = ['memory_unavailable', 'dashboard_authority_unavailable', 'dashboard_auth_not_configured'];
  // memory_api.RETURN_NOTICE, shown before and after 'No es mío' (v4 return_semantics.person_notice).
  const RETURN_NOTICE = 'Lo devolviste a la administración. No se borró: sigue guardado hasta que la administración lo reasigne o lo elimine. Si querés que se borre, usá Eliminar antes de devolverlo.';
  const string = value => typeof value === 'string';
  const finite = value => typeof value === 'number' && Number.isFinite(value);

  function detailCode(error) {
    const detail = error?.detail;
    return string(detail) ? detail : string(detail?.code) ? detail.code : null;
  }
  function parseItem(value) {
    if (!value || typeof value !== 'object' || Object.keys(value).sort().join(',') !== ITEM_KEYS.join(',')
        || !string(value.id) || !FACT_ID.test(value.id) || !string(value.content) || !string(value.category) || !STATES.includes(value.state)
        || !finite(value.created_at) || !(value.expires_at === null || finite(value.expires_at)) || (value.state === 'confirmed') !== (value.expires_at === null)
        || !METHODS.includes(value.owner_method) || !finite(value.owner_assigned_at) || typeof value.has_hidden_dependents !== 'boolean'
        || !Array.isArray(value.owned_dependents) || value.owned_dependents.some(id => !string(id) || !FACT_ID.test(id) || id === value.id)
        || new Set(value.owned_dependents).size !== value.owned_dependents.length) throw Error('invalid_my_memory');
    return { id: value.id, content: value.content, category: value.category, state: value.state, created_at: value.created_at,
      expires_at: value.expires_at, method: value.owner_method, assigned_at: value.owner_assigned_at,
      owned_dependents: [...value.owned_dependents], hidden_dependents: value.has_hidden_dependents };
  }
  /** One page of GET /api/memory/mine: {facts, next, limit} and nothing else. */
  function parsePage(value) {
    if (!value || typeof value !== 'object' || Object.keys(value).sort().join(',') !== 'facts,limit,next' || !Array.isArray(value.facts)
        || !Number.isSafeInteger(value.limit) || value.limit < 1 || value.limit > PAGE || value.facts.length > value.limit
        || !(value.next === null || (string(value.next) && CURSOR.test(value.next)))) throw Error('invalid_my_memory');
    const facts = value.facts.map(parseItem);
    if (new Set(facts.map(item => item.id)).size !== facts.length) throw Error('invalid_my_memory');
    return { facts, next: value.next };
  }

  const UNCONFIRMED = 'La respuesta del servidor no confirma el resultado.';
  const GONE_ALREADY = 'Ya no figuraba entre tus recuerdos.';
  /** applied | not_applied | provisional | uncertain, as in the Memoria owner (401/403 are uncertain). */
  function classifyWrite(op, outcome) {
    if (!OPS.includes(op)) throw Error('invalid_my_memory_operation');
    if (outcome && 'value' in outcome) {
      const value = outcome.value;
      if (op === 'delete') {
        if (typeof value?.deleted !== 'boolean') return { kind: 'uncertain', text: UNCONFIRMED };
        return { kind: 'applied', text: value.deleted ? 'Recuerdo eliminado.' : GONE_ALREADY, done: value.deleted };
      }
      if (typeof value?.returned !== 'boolean') return { kind: 'uncertain', text: UNCONFIRMED };
      return value.returned ? { kind: 'applied', text: string(value.notice) && value.notice ? value.notice : RETURN_NOTICE, done: true, returned: true }
        : { kind: 'applied', text: GONE_ALREADY, done: false };
    }
    const error = outcome?.error;
    if (error?.sent === false) return { kind: 'not_applied', text: 'No se envió: este acceso no permite la operación.' };
    if (error?.answered === true) return { kind: 'uncertain', text: 'La sesión dejó de estar vigente y no pudimos leer la respuesta.' };
    const status = error?.status, code = detailCode(error);
    if (!Number.isInteger(status)) return { kind: 'uncertain', text: 'No se pudo contactar al servidor.' };
    if (status === 401 || status === 403) return { kind: 'uncertain', text: 'La sesión no permitió confirmar el resultado.' };
    if (status === 422) return { kind: 'provisional', text: 'El servidor rechazó el identificador del recuerdo.' };
    if (status === 503 && NOT_APPLIED_503.includes(code)) return { kind: 'provisional', text: 'Tus recuerdos no están disponibles en este momento.' };
    return { kind: 'uncertain', text: status >= 500 ? 'El servidor no pudo completar la operación.' : 'El servidor respondió de forma inesperada.' };
  }
  function writeDisposition(kind, first) {
    if (kind === 'applied') return 'drop';
    if (kind === 'not_applied') return first ? 'final' : 'unresolved';
    return kind === 'provisional' ? 'provisional' : 'unresolved';
  }
  const STILL = { delete: 'Todavía figura entre tus recuerdos; puede que el cambio no se haya aplicado.',
    return: 'Todavía figura entre tus recuerdos; puede que no se haya devuelto.' };
  const INCONCLUSIVE = 'No pudimos comprobarlo: tu lista de recuerdos no se pudo recorrer completa.';
  const RESTORED_ABSENT = 'Ya no figura entre tus recuerdos, pero la operación quedó de una sesión anterior y no podemos confirmar que la hayas aplicado vos.';
  const RESTORED_STILL = 'Todavía figura entre tus recuerdos. La operación quedó de una sesión anterior y no se reenvía sola: si querés, hacela de nuevo.';
  /**
   * Resolve from a read that started after the loss. Absence from a complete walk only says the
   * memory is no longer yours (deleted, returned or reassigned): the copy never claims which.
   */
  function resolveMarker(op, observation, provisional = false, reason = '', restored = false) {
    // whoami names only a subject: it cannot prove that the login (or the person bound to it) is the
    // one that sent an operation stored by an earlier page. Such a marker is never 'Comprobado' (R3).
    if (restored) {
      if (!observation.complete && !observation.present) return { result: 'inconclusive', text: INCONCLUSIVE };
      return { result: 'final', text: observation.present ? RESTORED_STILL : RESTORED_ABSENT };
    }
    if (!observation.present) {
      return observation.complete ? { result: 'applied', text: 'Comprobado: ya no figura entre tus recuerdos.' } : { result: 'inconclusive', text: INCONCLUSIVE };
    }
    return provisional ? { result: 'final', text: reason || STILL[op] } : { result: 'pending', text: STILL[op] };
  }
  function describeReadError(error) {
    const status = error?.status;
    if (error?.sent === false || status === 403) return 'Este acceso no permite consultar tus recuerdos.';
    if (status === 503) return detailCode(error) === 'memory_unavailable' ? 'Tus recuerdos no están disponibles en este servidor.' : 'El servicio no está disponible en este momento.';
    if (status === 401) return 'Iniciá sesión para consultar tus recuerdos.';
    if (Number.isInteger(status)) return 'El servidor no pudo completar la consulta.';
    return 'No pudimos consultar tus recuerdos.';
  }
  const aborted = error => error?.name === 'AbortError';
  const statusless = error => !Number.isInteger(error?.status) && error?.sent !== false && error?.answered !== true;
  const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.';

  function createMyMemoryOwner({ api, session, storage = root.sessionStorage, schedule = setTimeout, cancel = clearTimeout,
    monotonic = () => root.performance.now(), wall = Date.now, onAccountChanged = () => {}, maxPages = MAX_PAGES }) {
    let epoch = 0, visit = 0, opened = false, timer, readSeq = 0, readFloor = 0, refreshing = null, again = false, serial = 0;
    const reads = new Set(), listeners = new Set();
    let state = initial();
    function initial() {
      return { list: null, complete: true, received: null, read: { loading: false, loaded: false, error: null, observedWall: null },
        markers: {}, notices: {}, announcement: null, returnNotice: null };
    }
    function capabilities() { return { read: session.can('memory:own:read'), write: session.can('memory:own:write') }; }
    function emit() { for (const listener of [...listeners]) { try { listener(); } catch (_) { /* A view must not break the owner. */ } } }
    function announce(text) { state.announcement = { serial: ++serial, text }; }
    function notice(id, text) { if (text) state.notices[id] = text; else delete state.notices[id]; }
    function subject() { return typeof session.subject === 'function' ? session.subject() : null; }
    function stopTimer() { if (timer !== undefined) cancel(timer); timer = undefined; }
    async function read(path) {
      const controller = new AbortController(); reads.add(controller);
      try { return await api(path, { signal: controller.signal }); } finally { reads.delete(controller); }
    }
    // The person projection has no server clock: expiry is judged against the local wall clock.
    function visibleList() {
      if (!state.list) return [];
      const now = wall() / 1000;
      return state.list.filter(item => state.markers[item.id] || item.expires_at === null || item.expires_at > now);
    }
    function plan() {
      stopTimer();
      if (!opened || !state.list) return;
      const now = wall() / 1000;
      const deadlines = state.list.map(item => item.expires_at).filter(value => value !== null && value > now);
      if (!deadlines.length) return;
      const ticket = epoch;
      timer = schedule(() => { timer = undefined; if (ticket === epoch && opened) { emit(); plan(); } }, Math.max(50, Math.min(60000, (Math.min(...deadlines) - now) * 1000 + 50)));
    }

    function storageKeys() {
      const keys = [];
      if (!storage) return keys;
      for (let index = 0; index < storage.length; index++) { const key = storage.key(index); if (key?.startsWith(PREFIX)) keys.push(key); }
      return keys;
    }
    function remove(key) { try { storage?.removeItem(key); } catch (_) { /* Best effort. */ } }
    function persist(id, marker) {
      if (!storage) throw Error('storage_unavailable');
      if (!marker.subject) throw Error('subject_unavailable');
      storage.setItem(PREFIX + id, JSON.stringify({ op: marker.op, subject: marker.subject, sent_at: marker.sent_at, state: marker.state, ...(marker.reason ? { reason: marker.reason } : {}) }));
    }
    function setMarker(id, marker) { state.markers[id] = marker; try { persist(id, marker); } catch (_) { /* Stored before sending. */ } }
    function dropMarker(id) { delete state.markers[id]; remove(PREFIX + id); }
    function restore() {
      let keys;
      try { keys = storageKeys(); } catch (_) { return; }
      const current = subject();
      let foreign = false;
      for (const key of keys) {
        const id = key.slice(PREFIX.length);
        try {
          const value = JSON.parse(storage.getItem(key));
          if (!FACT_ID.test(id) || !value || !OPS.includes(value.op) || !string(value.subject)) throw Error('invalid');
          if (!current || value.subject !== current) { foreign = true; throw Error('foreign'); }
          if (state.markers[id]) continue;
          state.markers[id] = { op: value.op, subject: value.subject, sent_at: finite(value.sent_at) ? value.sent_at : null,
            state: value.state === 'provisional' ? 'provisional' : 'unresolved', reason: string(value.reason) ? value.reason : '', after: -1, restored: true };
          notice(id, 'No sabemos si se aplicó la última operación. Consultando…');
        } catch (_) { remove(key); }
      }
      if (foreign) notice('session', 'Una operación de otra cuenta quedó sin confirmar; no se reintentó. Revisá tu lista.');
    }
    /** 'same' needs the cached subject, the home role and memory:own:read in the server's answer (F5). */
    async function confirmAccount() {
      const expected = subject(), seen = visit, wasOpen = opened;
      if (!expected) return 'changed';
      try {
        const identity = await read('/api/admin/auth/whoami');
        return identity?.ok === true && identity.subject === expected && identity.role === 'home' && session.role() === 'home'
          && Array.isArray(identity.permissions) && identity.permissions.includes('memory:own:read') ? 'same' : 'changed';
      } catch (error) {
        if (aborted(error) && (wasOpen && (!opened || seen !== visit))) return 'aborted';
        return [401, 403].includes(error?.status) ? 'changed' : 'unknown';
      }
    }
    function accountChanged() {
      clear({ wipe: true });
      notice('session', 'La cuenta de esta pestaña cambió en otra pestaña. No se consultó ni se envió nada.');
      emit();
      try { onAccountChanged(); } catch (_) { /* The page re-reads the session on its own. */ }
    }
    function dropContent() { state.list = null; state.complete = true; stopTimer(); }

    function settle(id, verdict) {
      const marker = state.markers[id];
      if (verdict.result === 'applied' || verdict.result === 'final') { dropMarker(id); notice(id, verdict.text); announce(verdict.text); return; }
      marker.state = 'unresolved'; setMarker(id, marker); notice(id, verdict.text);
    }
    async function walk(current) {
      const facts = [], seen = new Set(), cursors = new Set();
      let next = null, pages = 0, complete = true;
      do {
        if (pages >= maxPages) { complete = false; break; }
        if (next !== null) { if (cursors.has(next)) throw Error('invalid_my_memory'); cursors.add(next); }
        const page = parsePage(await read('/api/memory/mine' + (next ? '?after=' + encodeURIComponent(next) : '')));
        if (!current()) return null;
        for (const item of page.facts) { if (seen.has(item.id)) throw Error('invalid_my_memory'); seen.add(item.id); facts.push(item); }
        next = page.next; pages++;
      } while (next);
      return { facts, complete };
    }
    async function readOnce(ticket) {
      const started = ++readSeq, seen = visit;
      const current = () => ticket === epoch && opened && seen === visit;
      state.read.loading = true; emit();
      let result = null, failure = null, malformed = false;
      try { result = await walk(current); } catch (error) { if (error?.message === 'invalid_my_memory') malformed = true; else failure = error; }
      if (!current()) return;
      state.read.loading = false;
      if (malformed) { state.read.error = 'El servidor devolvió tus recuerdos con un formato desconocido.'; emit(); return; }
      if (failure) {
        if (statusless(failure)) {
          const account = await confirmAccount();
          if (!current()) return;
          if (account === 'changed') { accountChanged(); return; }
          if (account !== 'same') dropContent();
        } else dropContent();
        state.read.error = describeReadError(failure);
        emit(); plan(); return;
      }
      if (!result) return;
      if (started <= readFloor) { again = true; emit(); return; }
      state.list = result.facts; state.complete = result.complete;
      state.read = { loading: false, loaded: true, error: null, observedWall: wall() };
      for (const [id, marker] of Object.entries(state.markers)) {
        if (marker.state === 'sending' || marker.after >= started) continue;
        settle(id, resolveMarker(marker.op, { present: result.facts.some(item => item.id === id), complete: result.complete }, marker.state === 'provisional', marker.reason, marker.restored === true));
      }
      emit(); plan();
    }
    function refresh() {
      if (!opened || !capabilities().read) return Promise.resolve();
      if (refreshing) { again = true; return refreshing; }
      const ticket = epoch;
      const current = (async () => {
        try { do { again = false; await readOnce(ticket); } while (again && ticket === epoch && opened); }
        finally { if (refreshing === current) refreshing = null; }
      })();
      refreshing = current;
      return current;
    }
    let reloading = null;
    function reload() {
      if (!opened) return Promise.resolve();
      if (!reloading) reloading = reloadOnce().finally(() => { reloading = null; });
      return reloading;
    }
    async function reloadOnce() {
      const ticket = epoch, seen = visit, account = await confirmAccount();
      if (ticket !== epoch || !opened || seen !== visit || account === 'aborted') return;
      if (account === 'changed') { accountChanged(); return; }
      if (account === 'unknown') { state.read.error = ACCOUNT_UNKNOWN; dropContent(); emit(); return; }
      await refresh();
    }

    const find = id => visibleList().find(item => item.id === id) || null;
    function ownedDependents(id) {
      const item = find(id);
      if (!item) return [];
      const listed = visibleList();
      return item.owned_dependents.map(other => listed.find(entry => entry.id === other) || { id: other, content: null });
    }
    function blocked(id) {
      const item = find(id);
      return Boolean(state.markers[id]) || Boolean(item && item.owned_dependents.some(other => state.markers[other]));
    }
    async function send(id, marker, first) {
      const ticket = epoch;
      const path = '/api/memory/mine/facts/' + id + (marker.op === 'return' ? '/return' : '');
      let outcome;
      // Never aborted: leaving the view must not cut a write the server may commit.
      try { outcome = { value: await api(path, { method: marker.op === 'delete' ? 'DELETE' : 'POST' }) }; } catch (error) { outcome = { error }; }
      if (ticket !== epoch || state.markers[id] !== marker) return { stale: true };
      const verdict = classifyWrite(marker.op, outcome);
      if (verdict.kind === 'applied') {
        dropMarker(id); readFloor = readSeq; again = again || Boolean(refreshing);
        if (state.list && opened) {
          const gone = new Set([id, ...(marker.op === 'delete' && verdict.done ? (find(id)?.owned_dependents || []) : [])]);
          state.list = state.list.filter(item => !gone.has(item.id));
        }
        if (verdict.returned) state.returnNotice = verdict.text;
        notice(id, verdict.text); announce(verdict.text); emit();
        void refresh();
        return { applied: true };
      }
      const disposition = writeDisposition(verdict.kind, first);
      if (disposition === 'final') { dropMarker(id); notice(id, verdict.text); announce(verdict.text); emit(); return { applied: false }; }
      marker.state = disposition; marker.reason = disposition === 'provisional' ? verdict.text : ''; marker.after = readSeq;
      setMarker(id, marker);
      const text = verdict.kind === 'provisional' ? verdict.text + ' Comprobando…' : 'Sin confirmar. ' + verdict.text + ' Consultando…';
      notice(id, text); announce(text); emit();
      void refresh();
      return { uncertain: true };
    }
    /** Both operations change what the person keeps: always confirmed by a dialog and a whoami. */
    async function act(op, id, { confirmed = false } = {}) {
      if (!opened || !OPS.includes(op) || !capabilities().write || !confirmed || blocked(id) || !find(id)) return { sent: false };
      const ticket = epoch, account = await confirmAccount();
      if (ticket !== epoch) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      if (account !== 'same') { notice(id, ACCOUNT_UNKNOWN); announce(ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      if (blocked(id) || !find(id)) return { sent: false };
      const marker = { op, subject: subject(), sent_at: wall(), state: 'sending', reason: '', after: Infinity };
      try { persist(id, marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado de la operación; no se envió.';
        notice(id, text); announce(text); emit(); return { sent: false };
      }
      state.markers[id] = marker; notice(id, null); emit();
      return send(id, marker, true);
    }
    async function retry(id) {
      const marker = state.markers[id];
      if (!opened || !capabilities().write || !marker || marker.state === 'sending') return { sent: false };
      if (marker.subject !== subject()) { dropMarker(id); notice(id, 'Esta operación pertenece a otra cuenta y se descartó sin reenviarla.'); emit(); return { sent: false }; }
      // An operation stored by an earlier page may come from another login: never resent with one click (R3).
      if (marker.restored) return { sent: false, restored: true };
      const ticket = epoch, previous = { state: marker.state, reason: marker.reason };
      marker.state = 'sending'; emit();
      const account = await confirmAccount();
      if (ticket !== epoch || state.markers[id] !== marker) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      if (account !== 'same') { Object.assign(marker, previous); notice(id, ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      marker.after = Infinity; setMarker(id, marker); notice(id, null); emit();
      return send(id, marker, false);
    }
    async function open() {
      opened = true; visit++; restore();
      try { if (storage?.getItem(ENDED_KEY)) { remove(ENDED_KEY); notice('session', 'Una operación sobre tus recuerdos quedó sin confirmar al terminar la sesión anterior. Revisá la lista antes de repetirla.'); } }
      catch (_) { /* Reminder is best effort. */ }
      emit();
      const ticket = epoch, seen = visit, account = await confirmAccount();
      if (ticket !== epoch || !opened || seen !== visit || account === 'aborted') return true;
      if (account === 'changed') { accountChanged(); return true; }
      if (account === 'unknown') { state.read.error = ACCOUNT_UNKNOWN; emit(); return true; }
      void refresh();
      return true;
    }
    function leave() {
      opened = false; again = false; stopTimer();
      for (const controller of reads) controller.abort();
      reads.clear(); refreshing = null;
      dropContent(); state.returnNotice = null;
      state.read = { loading: false, loaded: false, error: null, observedWall: state.read.observedWall };
      return Promise.resolve(true);
    }
    function clear({ wipe = true } = {}) {
      const pending = Object.keys(state.markers).length > 0;
      epoch++; leave(); state = initial(); readFloor = readSeq;
      if (wipe) {
        let keys = [];
        try { keys = storageKeys(); } catch (_) { /* Nothing to wipe. */ }
        keys.forEach(remove);
        if (pending || keys.length) { try { storage.setItem(ENDED_KEY, '1'); } catch (_) { /* Best effort. */ } }
      }
      emit();
      return Promise.resolve(true);
    }
    function getState() {
      return { facts: visibleList(), loaded: Boolean(state.list), complete: state.complete, read: { ...state.read },
        markers: Object.fromEntries(Object.entries(state.markers).map(([id, marker]) => [id, { op: marker.op, state: marker.state, restored: marker.restored === true }])),
        notices: { ...state.notices }, announcement: state.announcement, returnNotice: state.returnNotice, capabilities: capabilities(), subject: subject() };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, reload, subscribe, getState, capabilities, ownedDependents, blocked,
      remove: (id, options) => act('delete', id, options), giveBack: (id, options) => act('return', id, options), retry, isOpen: () => opened };
  }

  const exported = { createMyMemoryOwner, parsePage, parseItem, classifyWrite, writeDisposition, resolveMarker, describeReadError, PREFIX, ENDED_KEY, RETURN_NOTICE, MAX_PAGES, RESTORED_ABSENT, RESTORED_STILL };
  root.NexoMyMemoryOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
