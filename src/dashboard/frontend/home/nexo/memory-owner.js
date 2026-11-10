/* NEXO Memoria owner (no DOM). Reads may be aborted; writes never are. A pending
 * operation is bound to the account that sent it, survives reload without any
 * fact content, and is settled only by evidence: a visible effect in a read that
 * started after the loss, or the answer to a resend of the same operation. */
(function (root) {
  'use strict';
  const PREFIX = 'kza:memory-op:';
  // Outside the prefix so a session-end wipe can leave a content-free reminder.
  const ENDED_KEY = 'kza:memory-op-ended';
  // FactLedger.snapshot returns at most this many items per list, silently.
  const CAP = 100;
  const OPS = ['confirm', 'reject', 'delete'];
  const FACT_ID = /^fact_[0-9a-f]{64}$/;
  const FACT_STATES = ['automatic', 'confirmed'];
  const ORIGINS = ['ambient', 'voice', 'interaction'];
  // Pre-handler or pre-commit answers that say "not applied"; still checked by a read (F3).
  const NOT_APPLIED_503 = ['memory_unavailable', 'dashboard_authority_unavailable', 'dashboard_auth_not_configured'];
  const string = value => typeof value === 'string';
  const finite = value => typeof value === 'number' && Number.isFinite(value);

  function detailCode(error) {
    const detail = error?.detail;
    return string(detail) ? detail : string(detail?.code) ? detail.code : null;
  }

  function parseSource(value) {
    if (!value || typeof value !== 'object' || !ORIGINS.includes(value.origin) || !finite(value.captured_at)) throw Error('invalid_memory_source');
    const dependencies = value.fact_dependencies === undefined ? [] : value.fact_dependencies;
    if (!Array.isArray(dependencies) || dependencies.some(item => !item || !string(item.fact_id) || !FACT_ID.test(item.fact_id))) throw Error('invalid_memory_source');
    // Capture and conversation identifiers are never kept for presentation.
    return { origin: value.origin, captured_at: value.captured_at, depends_on: dependencies.map(item => item.fact_id) };
  }
  function parseFact(value, proposed) {
    if (!value || typeof value !== 'object' || !string(value.id) || !FACT_ID.test(value.id) || !string(value.content)
        || !string(value.category) || !(proposed ? value.state === 'proposed' : FACT_STATES.includes(value.state))
        || !(value.expires_at === null || finite(value.expires_at)) || (value.state === 'confirmed') !== (value.expires_at === null)
        || !Number.isSafeInteger(value.revision) || value.revision < 1
        || !(value.confirmed_at === null || value.confirmed_at === undefined || finite(value.confirmed_at))
        || !(value.confirmed_by === null || value.confirmed_by === undefined || string(value.confirmed_by))
        || !Array.isArray(value.sources)) throw Error('invalid_memory_fact');
    return { id: value.id, content: value.content, category: value.category, state: value.state, expires_at: value.expires_at,
      revision: value.revision, confirmed_at: value.confirmed_at ?? null, confirmed_by: value.confirmed_by ?? null,
      sources: value.sources.map(parseSource) };
  }
  const count = value => value === undefined || value === null ? null : Number.isSafeInteger(value) && value >= 0 ? value : (() => { throw Error('invalid_memory_snapshot'); })();
  /** Fail closed: one malformed item makes the whole snapshot unusable. */
  function parseSnapshot(value) {
    if (!value || typeof value !== 'object' || !finite(value.observed_at) || !Array.isArray(value.facts) || !Array.isArray(value.proposals)
        || value.facts.length > CAP || value.proposals.length > CAP) throw Error('invalid_memory_snapshot');
    const facts = value.facts.map(item => parseFact(item, false)), proposals = value.proposals.map(item => parseFact(item, true));
    const ids = new Set();
    for (const item of [...facts, ...proposals]) { if (ids.has(item.id)) throw Error('invalid_memory_snapshot'); ids.add(item.id); }
    return { observed_at: value.observed_at, facts, proposals, pending_index: count(value.pending_index),
      pending_index_cleanup: count(value.pending_index_cleanup), legacy_policy: string(value.legacy_policy) ? value.legacy_policy : null };
  }

  const APPLIED_TEXT = { confirm: 'Recuerdo conservado sin vencimiento.', reject: 'Propuesta descartada. No se puede recuperar.', delete: 'Recuerdo eliminado.' };
  const ALREADY_TEXT = { reject: 'Ya no estaba guardada.', delete: 'Ya no estaba guardado.' };
  /**
   * Classify one write answer. applied: a valid 200; not_applied: never sent;
   * provisional: sent and the server says "not applied" (a read must agree, F3);
   * uncertain: anything else, including every 401/403 (a 403 can follow a commit).
   */
  function classifyWrite(op, id, outcome) {
    if (!OPS.includes(op)) throw Error('invalid_memory_operation');
    if (outcome && 'value' in outcome) {
      const value = outcome.value;
      if (op === 'confirm') {
        let fact = null;
        try { fact = parseFact(value?.fact, false); } catch (_) { /* Malformed receipt. */ }
        return fact && fact.id === id && fact.state === 'confirmed' ? { kind: 'applied', text: APPLIED_TEXT.confirm, fact }
          : { kind: 'uncertain', text: 'La respuesta del servidor no confirma el resultado.' };
      }
      if (typeof value?.deleted !== 'boolean') return { kind: 'uncertain', text: 'La respuesta del servidor no confirma el resultado.' };
      return { kind: 'applied', text: value.deleted ? APPLIED_TEXT[op] : ALREADY_TEXT[op], deleted: value.deleted };
    }
    const error = outcome?.error;
    if (error?.sent === false) return { kind: 'not_applied', text: 'No se envió: este acceso no permite la operación.' };
    if (error?.answered === true) return { kind: 'uncertain', text: 'La sesión dejó de estar vigente y no pudimos leer la respuesta.' };
    const status = error?.status, code = detailCode(error);
    if (!Number.isInteger(status)) return { kind: 'uncertain', text: 'No se pudo contactar al servidor.' };
    if (status === 401 || status === 403) return { kind: 'uncertain', text: 'La sesión no permitió confirmar el resultado.' };
    if (op === 'confirm' && status === 404 && code === 'memory_proposal_unavailable') return { kind: 'provisional', text: 'La propuesta venció o ya no está disponible.' };
    if (op === 'confirm' && status === 409 && code === 'memory_proposal_required') return { kind: 'provisional', text: 'Este recuerdo no es una propuesta.' };
    if (op === 'reject' && status === 409 && code === 'memory_proposal_state_changed') return { kind: 'provisional', text: 'Otra sesión ya la confirmó; no se descartó.' };
    if (status === 422) return { kind: 'provisional', text: 'El servidor rechazó el identificador del recuerdo.' };
    if (status === 503 && NOT_APPLIED_503.includes(code)) return { kind: 'provisional', text: 'La memoria no está disponible en este momento.' };
    return { kind: 'uncertain', text: status >= 500 ? 'El servidor no pudo completar la operación.' : 'El servidor respondió de forma inesperada.' };
  }

  /**
   * What a write answer does to its marker. A local refusal is final only on the first send:
   * on a retry the earlier send may already have committed, so the marker stays unresolved.
   */
  function writeDisposition(kind, first) {
    if (kind === 'applied') return 'drop';
    if (kind === 'not_applied') return first ? 'final' : 'unresolved';
    return kind === 'provisional' ? 'provisional' : 'unresolved';
  }
  /** Where an id is visible in a snapshot and whether each list hit the silent cap. */
  function observe(snapshot, id) {
    const fact = snapshot.facts.find(item => item.id === id), proposal = snapshot.proposals.find(item => item.id === id);
    return { where: fact ? 'facts' : proposal ? 'proposals' : null, state: (fact || proposal)?.state ?? null,
      factsFull: snapshot.facts.length >= CAP, proposalsFull: snapshot.proposals.length >= CAP };
  }
  const STILL = { confirm: 'Todavía figura como propuesta; puede que el cambio no se haya aplicado.',
    reject: 'Todavía figura como propuesta; puede que el cambio no se haya aplicado.', delete: 'Todavía figura guardado; puede que el cambio no se haya aplicado.' };
  const INCONCLUSIVE = 'No pudimos comprobarlo: no aparece entre los 100 más recientes.';
  /**
   * Resolve a marker from a read that started after the loss (or after the provisional answer).
   * A read never proves "not applied" for an uncertain write (F2): it may still commit later.
   * Returns {result: applied|final|pending|inconclusive, text}.
   */
  function resolveMarker(op, observation, provisional = false, reason = '') {
    const { where, state, factsFull, proposalsFull } = observation;
    const agree = text => provisional ? { result: 'final', text: reason || text } : { result: 'pending', text: STILL[op] };
    if (op === 'confirm') {
      if (where === 'facts' && state === 'confirmed') return { result: 'applied', text: 'Comprobado: el recuerdo quedó conservado sin vencimiento.' };
      // An automatic fact can never be confirmed: the operation is impossible from now on.
      if (where === 'facts') return { result: 'final', text: reason || 'Este recuerdo no es una propuesta; no se conservó.' };
      if (where === 'proposals') return agree();
      if (factsFull) return { result: 'inconclusive', text: INCONCLUSIVE };
      // Below the cap a confirmed fact would be listed; an absent proposal cannot become confirmed later
      // unless it is a proposal beyond the proposals cap.
      if (proposalsFull) return provisional ? { result: 'final', text: reason } : { result: 'inconclusive', text: INCONCLUSIVE };
      return { result: 'final', text: 'Ya no está guardada: venció o se eliminó.' };
    }
    if (where === null) {
      if (factsFull || proposalsFull) return { result: 'inconclusive', text: INCONCLUSIVE };
      return { result: 'applied', text: op === 'delete' ? 'Comprobado: ya no está guardado.' : 'Comprobado: la propuesta ya no está guardada.' };
    }
    // A proposal now listed as a fact was confirmed elsewhere; a reject can never apply to it.
    if (op === 'reject' && where === 'facts') return { result: 'final', text: state === 'confirmed' ? 'Otra sesión ya la confirmó; no se descartó.' : 'Este recuerdo no es una propuesta; no se descartó.' };
    return agree();
  }

  function describeReadError(error) {
    const status = error?.status, code = detailCode(error);
    if (error?.sent === false) return 'Este acceso no permite consultar la memoria.';
    if (status === 503) return code === 'memory_unavailable' ? 'La memoria no está disponible en este servidor.' : 'El servicio no está disponible en este momento.';
    if (status === 403) return 'Este acceso no permite consultar la memoria.';
    if (status === 401) return 'Iniciá sesión para consultar la memoria.';
    if (Number.isInteger(status)) return 'El servidor no pudo completar la consulta.';
    return 'No pudimos consultar la memoria.';
  }

  /** Visible items that the server would delete together with `id` (transitively). */
  function dependents(items, id) {
    const affected = new Set([id]);
    let grown = true;
    while (grown) {
      grown = false;
      for (const item of items) {
        if (affected.has(item.id)) continue;
        if (item.sources.some(source => source.depends_on.some(parent => affected.has(parent)))) { affected.add(item.id); grown = true; }
      }
    }
    return items.filter(item => item.id !== id && affected.has(item.id));
  }

  const aborted = error => error?.name === 'AbortError';
  const statusless = error => !Number.isInteger(error?.status) && error?.sent !== false && error?.answered !== true;
  const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.';

  function createMemoryOwner({ api, session, storage = root.sessionStorage, schedule = setTimeout, cancel = clearTimeout,
    monotonic = () => root.performance.now(), wall = Date.now, onAccountChanged = () => {} }) {
    let epoch = 0, visit = 0, opened = false, timer, readSeq = 0, readFloor = 0, refreshing = null, again = false, serial = 0;
    const reads = new Set(), listeners = new Set();
    let state = initial();
    function initial() {
      return { snapshot: null, received: null, read: { loading: false, loaded: false, error: null, observedWall: null },
        markers: {}, kept: {}, notices: {}, announcement: null };
    }
    function capabilities() { return { read: session.can('memory:read'), review: session.can('memory:review') }; }
    function emit() { for (const listener of [...listeners]) { try { listener(); } catch (_) { /* A view must not break the owner. */ } } }
    function announce(text) { state.announcement = { serial: ++serial, text }; }
    function notice(id, text) { if (text) state.notices[id] = text; else delete state.notices[id]; }
    function subject() { return typeof session.subject === 'function' ? session.subject() : null; }
    function stopTimer() { if (timer !== undefined) cancel(timer); timer = undefined; }
    async function read(path) {
      const controller = new AbortController(); reads.add(controller);
      try { return await api(path, { signal: controller.signal }); } finally { reads.delete(controller); }
    }

    // Server clock: observed_at plus the larger of monotonic and wall elapsed time since receipt.
    function serverNow() {
      const received = state.received;
      if (!received) return null;
      return received.observed + Math.max(0, monotonic() - received.mono, wall() - received.wall) / 1000;
    }
    const live = (item, now) => item.expires_at === null || item.expires_at > now;
    function visible() {
      const snapshot = state.snapshot;
      if (!snapshot) return { facts: [], proposals: [], kept: [] };
      const now = serverNow(), marked = id => Boolean(state.markers[id]);
      const facts = snapshot.facts.filter(item => marked(item.id) || live(item, now));
      const proposals = snapshot.proposals.filter(item => marked(item.id) || live(item, now));
      const listed = new Set(facts.map(item => item.id));
      return { facts, proposals, kept: Object.values(state.kept).filter(item => !listed.has(item.id)) };
    }
    function plan() {
      stopTimer();
      if (!opened || !state.snapshot) return;
      const now = serverNow();
      const deadlines = [...state.snapshot.facts, ...state.snapshot.proposals].map(item => item.expires_at).filter(value => value !== null && value > now);
      if (!deadlines.length) return;
      const ticket = epoch;
      timer = schedule(() => { timer = undefined; if (ticket === epoch && opened) { emit(); plan(); } }, Math.max(50, Math.min(60000, (Math.min(...deadlines) - now) * 1000 + 50)));
    }

    // Markers: one per fact id, never with content.
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
    function setMarker(id, marker) {
      state.markers[id] = marker;
      try { persist(id, marker); } catch (_) { /* The identity was stored before sending. */ }
    }
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
          // Another account's operation is never resolved, shown as its own or resent here.
          if (!current || value.subject !== current) { foreign = true; throw Error('foreign'); }
          if (state.markers[id]) continue;
          // A stored 'sending' marker is not proof of anything: the page that sent it is gone.
          state.markers[id] = { op: value.op, subject: value.subject, sent_at: finite(value.sent_at) ? value.sent_at : null,
            state: value.state === 'provisional' ? 'provisional' : 'unresolved', reason: string(value.reason) ? value.reason : '', after: -1 };
          notice(id, 'No sabemos si se aplicó la última operación. Consultando…');
        } catch (_) { remove(key); }
      }
      if (foreign) notice('session', 'Una operación de Memoria de otra cuenta quedó sin confirmar; no se reintentó. Revisá la lista.');
    }

    /**
     * Asks the server which account the shared cookie names now (F5). 'same' requires the cached
     * subject and role and memory:read; 'changed' covers another or no account; 'unknown' = not asked.
     */
    async function confirmAccount() {
      const expected = subject(), seen = visit, wasOpen = opened;
      if (!expected) return 'changed';
      try {
        const identity = await read('/api/admin/auth/whoami');
        return identity?.ok === true && identity.subject === expected && identity.role === session.role()
          && Array.isArray(identity.permissions) && identity.permissions.includes('memory:read') ? 'same' : 'changed';
      } catch (error) {
        // leave() aborts reads of a finished visit; any other abort is the client timeout: not asked.
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
    function dropContent() { state.snapshot = null; state.received = null; state.kept = {}; stopTimer(); }

    function settle(id, marker, verdict) {
      if (verdict.result === 'applied' || verdict.result === 'final') {
        dropMarker(id); notice(id, verdict.text); announce(verdict.text);
        if (verdict.result === 'applied' && marker.op !== 'confirm') delete state.kept[id];
        return;
      }
      marker.state = 'unresolved'; marker.reason = marker.reason || '';
      setMarker(id, marker);
      notice(id, verdict.text);
    }
    function resolveAll(started) {
      for (const [id, marker] of Object.entries(state.markers)) {
        if (marker.state === 'sending' || marker.after >= started) continue;
        const verdict = resolveMarker(marker.op, observe(state.snapshot, id), marker.state === 'provisional', marker.reason);
        settle(id, marker, verdict);
      }
      // A kept confirmation stays visible only while the full facts list can hide it (F1).
      for (const id of Object.keys(state.kept)) {
        const listed = state.snapshot.facts.some(item => item.id === id);
        if (listed || state.snapshot.facts.length < CAP) delete state.kept[id];
      }
    }

    async function readOnce(ticket) {
      const started = ++readSeq, seen = visit;
      const current = () => ticket === epoch && opened && seen === visit;
      state.read.loading = true; emit();
      let value, failure = null;
      try { value = await read('/api/memory'); } catch (error) { failure = error; }
      // Reads aborted by leave() belong to a finished visit; an abort seen here is the client timeout.
      if (!current()) return;
      state.read.loading = false;
      if (failure) {
        if (statusless(failure)) {
          // A body cut by a retired session looks like a network error: keep content only for the same account (F9).
          const account = await confirmAccount();
          if (!current()) return;
          if (account === 'changed') { accountChanged(); return; }
          if (account !== 'same') dropContent();
        } else dropContent();
        state.read.error = describeReadError(failure);
        emit(); plan(); return;
      }
      let snapshot;
      try { snapshot = parseSnapshot(value); } catch (_) {
        state.read.error = 'El servidor devolvió la memoria con un formato desconocido.'; emit(); return;
      }
      // A write applied after this read started is newer than it: read again.
      if (started <= readFloor) { again = true; emit(); return; }
      state.snapshot = snapshot; state.received = { observed: snapshot.observed_at, mono: monotonic(), wall: wall() };
      state.read = { loading: false, loaded: true, error: null, observedWall: wall() };
      resolveAll(started);
      emit(); plan();
    }
    /** Coalesces concurrent calls; a call made during a read waits for a fresh read. */
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
    /** User refresh: confirm the account first, then read. */
    let reloading = null;
    function reload() {
      if (!opened) return Promise.resolve();
      // Coalesced: a second press while the first is in progress joins it.
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

    function find(id) {
      const view = visible();
      return view.proposals.find(item => item.id === id) || view.facts.find(item => item.id === id) || view.kept.find(item => item.id === id) || null;
    }
    function affected(id) {
      const view = visible();
      return dependents([...view.facts, ...view.proposals, ...view.kept], id);
    }
    function deleteBlocked(id) { return [id, ...affected(id).map(item => item.id)].some(item => state.markers[item]); }

    async function send(id, marker, first) {
      const ticket = epoch;
      const path = marker.op === 'delete' ? '/api/memory/facts/' + id : '/api/memory/proposals/' + id + '/' + marker.op;
      let outcome;
      // Never aborted: leaving the view must not cut a write the server may commit.
      try { outcome = { value: await api(path, { method: marker.op === 'delete' ? 'DELETE' : 'POST' }) }; } catch (error) { outcome = { error }; }
      if (ticket !== epoch || state.markers[id] !== marker) return { stale: true };
      const verdict = classifyWrite(marker.op, id, outcome);
      if (verdict.kind === 'applied') {
        dropMarker(id); readFloor = readSeq; again = again || Boolean(refreshing);
        if (state.snapshot && opened) {
          const proposals = state.snapshot.proposals.filter(item => item.id !== id);
          let facts = state.snapshot.facts.filter(item => item.id !== id);
          if (marker.op === 'confirm') { facts = [verdict.fact, ...facts]; state.kept[id] = verdict.fact; }
          state.snapshot = { ...state.snapshot, facts, proposals };
        }
        if (marker.op !== 'confirm') delete state.kept[id];
        notice(id, verdict.text); announce(verdict.text); emit();
        void refresh();
        return { applied: true };
      }
      const disposition = writeDisposition(verdict.kind, first);
      if (disposition === 'final') {
        dropMarker(id); notice(id, verdict.text); announce(verdict.text); emit();
        return { applied: false };
      }
      // Provisional answers and losses wait for a read that starts from now on (F2, F3).
      marker.state = disposition;
      marker.reason = disposition === 'provisional' ? verdict.text : '';
      marker.after = readSeq;
      setMarker(id, marker);
      const text = verdict.kind === 'provisional' ? verdict.text + ' Comprobando…' : 'Sin confirmar. ' + verdict.text + ' Consultando…';
      notice(id, text); announce(text); emit();
      void refresh();
      return { uncertain: true };
    }
    async function act(op, id, { confirmed = false } = {}) {
      if (!opened || !OPS.includes(op) || !capabilities().review || state.markers[id]) return { sent: false };
      const item = find(id);
      if (!item) return { sent: false };
      if (op !== 'delete' && item.state !== 'proposed') return { sent: false };
      if (op === 'delete' && (item.state === 'proposed' || !confirmed || deleteBlocked(id))) return { sent: false };
      const ticket = epoch;
      if (op === 'delete') {
        // An irreversible write is only sent after the server names the same account (F5).
        const account = await confirmAccount();
        if (ticket !== epoch) return { sent: false };
        if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
        if (account !== 'same') { notice(id, ACCOUNT_UNKNOWN); announce(ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
        if (state.markers[id] || deleteBlocked(id)) return { sent: false };
      }
      const marker = { op, subject: subject(), sent_at: wall(), state: 'sending', reason: '', after: Infinity };
      try { persist(id, marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado de la operación; no se envió.';
        notice(id, text); announce(text); emit(); return { sent: false };
      }
      state.markers[id] = marker; notice(id, null); emit();
      return send(id, marker, true);
    }
    /** Resend the same operation: idempotent per fact id, so its answer settles the marker. */
    async function retry(id) {
      const marker = state.markers[id];
      if (!opened || !capabilities().review || !marker || marker.state === 'sending') return { sent: false };
      if (marker.subject !== subject()) {
        dropMarker(id); notice(id, 'Esta operación pertenece a otra cuenta y se descartó sin reenviarla.'); emit();
        return { sent: false };
      }
      const ticket = epoch, previous = { state: marker.state, reason: marker.reason };
      marker.state = 'sending'; emit();
      const account = await confirmAccount();
      if (ticket !== epoch || state.markers[id] !== marker) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      // Not asked: nothing sent, and the marker keeps what it knew (a provisional server reason included).
      if (account !== 'same') { Object.assign(marker, previous); notice(id, ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      marker.after = Infinity; setMarker(id, marker); notice(id, null); emit();
      return send(id, marker, false);
    }

    /** Every activation (also the return to a hidden tab) confirms the account before reading (F5). */
    async function open() {
      opened = true; visit++; restore();
      try { if (storage?.getItem(ENDED_KEY)) { remove(ENDED_KEY); notice('session', 'Una operación de Memoria quedó sin confirmar al terminar la sesión anterior. Revisá la lista antes de repetirla.'); } }
      catch (_) { /* Reminder is best effort. */ }
      emit();
      const ticket = epoch, seen = visit, account = await confirmAccount();
      // A whoami of an earlier visit (open -> leave -> open) never speaks for this one.
      if (ticket !== epoch || !opened || seen !== visit || account === 'aborted') return true;
      if (account === 'changed') { accountChanged(); return true; }
      if (account === 'unknown') { state.read.error = ACCOUNT_UNKNOWN; emit(); return true; }
      void refresh();
      return true;
    }
    /** Stop reads, drop private content; writes and markers continue. Resolves immediately. */
    function leave() {
      opened = false; again = false; stopTimer();
      for (const controller of reads) controller.abort();
      reads.clear(); refreshing = null;
      dropContent(); state.read = { loading: false, loaded: false, error: null, observedWall: state.read.observedWall };
      return Promise.resolve(true);
    }
    /** Session boundary: forget everything; wipe stored markers unless the page is only being hidden. */
    function clear({ wipe = true } = {}) {
      const pending = Object.keys(state.markers).length > 0;
      epoch++; leave(); state = initial(); readFloor = readSeq;
      if (wipe) {
        let keys = [];
        try { keys = storageKeys(); } catch (_) { /* Nothing to wipe. */ }
        keys.forEach(remove);
        // Every marker is unresolved by definition (F11): leave only a content-free reminder.
        if (pending || keys.length) { try { storage.setItem(ENDED_KEY, '1'); } catch (_) { /* Best effort. */ } }
      }
      emit();
      return Promise.resolve(true);
    }
    function getState() {
      const view = visible(), snapshot = state.snapshot;
      return { facts: view.facts, proposals: view.proposals, kept: view.kept, loaded: Boolean(snapshot), read: { ...state.read },
        factsFull: Boolean(snapshot && snapshot.facts.length >= CAP), proposalsFull: Boolean(snapshot && snapshot.proposals.length >= CAP),
        pendingIndex: snapshot?.pending_index ?? null, pendingCleanup: snapshot?.pending_index_cleanup ?? null, legacyPolicy: snapshot?.legacy_policy ?? null,
        markers: Object.fromEntries(Object.entries(state.markers).map(([id, marker]) => [id, { op: marker.op, state: marker.state }])),
        notices: { ...state.notices }, announcement: state.announcement, capabilities: capabilities(), subject: subject() };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, reload, subscribe, getState, capabilities, dependents: affected, deleteBlocked,
      confirm: id => act('confirm', id), reject: id => act('reject', id), remove: (id, options) => act('delete', id, options), retry,
      isOpen: () => opened };
  }

  const exported = { createMemoryOwner, parseSnapshot, classifyWrite, writeDisposition, resolveMarker, observe, dependents, describeReadError, PREFIX, ENDED_KEY, CAP };
  root.NexoMemoryOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
