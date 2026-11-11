/* NEXO Memoria owner (no DOM). Reads may be aborted; writes never are. A pending
 * operation is bound to the account that sent it, survives reload without any
 * fact content, and is settled only by evidence: a visible effect in a read that
 * started after the loss, or the answer to a resend of the same operation.
 *
 * U3b-3: a read walks every keyset page of GET /api/memory (proposals first, then
 * facts, so a proposal confirmed mid-walk is seen in one list or the other), plus
 * the 'Devueltos' queue, owner counts and the people catalog for assignment. */
(function (root) {
  'use strict';
  const PREFIX = 'kza:memory-op:';
  // Outside the prefix so a session-end wipe can leave a content-free reminder.
  const ENDED_KEY = 'kza:memory-op-ended';
  // Server page size (MEMORY_PAGE_LIMIT): no page may carry more items.
  const CAP = 100;
  // Pages walked per list before the read is reported as incomplete (100 x 100 = 10 000 items).
  const MAX_PAGES = 100;
  const OPS = ['confirm', 'reject', 'delete', 'assign', 'unassign'];
  const FACT_ID = /^fact_[0-9a-f]{64}$/;
  const CURSOR = /^k1\.[0-9a-f]{16}\.fact_[0-9a-f]{64}$/;
  const FACT_STATES = ['automatic', 'confirmed'];
  const ALL_STATES = ['automatic', 'confirmed', 'proposed'];
  const ORIGINS = ['ambient', 'voice', 'interaction'];
  const OWNER_METHODS = ['operator_assignment', 'person_session', 'voice_proposal'];
  const OWNER_CHANGES = ['assigned', 'reassigned', 'owner_returned', 'operator_unassigned', 'voice_proposed', 'owner_confirmed'];
  // Pre-handler or pre-commit answers that say "not applied"; still checked by a read (F3).
  const NOT_APPLIED_503 = ['memory_unavailable', 'dashboard_authority_unavailable', 'dashboard_auth_not_configured',
    'memory_owner_authority_missing', 'memory_owner_authority_unavailable'];
  const string = value => typeof value === 'string';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const revision = (value, minimum) => Number.isSafeInteger(value) && value >= minimum;
  // Mirrors fact_ledger._valid_person: an account id, never a voice label.
  const person = value => string(value) && value.length >= 1 && value.length <= 128 && ![...value].some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);

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
  function parseOwnerFields(value) {
    const owner = value.owner;
    if (owner !== null && (!owner || typeof owner !== 'object' || !person(owner.person_id) || !OWNER_METHODS.includes(owner.method)
        || !revision(owner.revision, 1) || !finite(owner.assigned_at))) throw Error('invalid_memory_fact');
    if (!revision(value.owner_revision, 0) || !(value.owner_change === null || OWNER_CHANGES.includes(value.owner_change))
        || (owner !== null && value.owner_revision < 1) || (value.owner_revision === 0) !== (value.owner_change === null)) throw Error('invalid_memory_fact');
    return { owner: owner === null ? null : { person_id: owner.person_id, method: owner.method, revision: owner.revision, assigned_at: owner.assigned_at },
      owner_revision: value.owner_revision, owner_change: value.owner_change };
  }
  /** kind: 'fact' (automatic/confirmed), 'proposal', or 'any'; owner fields are required on list items. */
  function parseFact(value, kind, withOwner = true) {
    const states = kind === 'proposal' ? ['proposed'] : kind === 'any' ? ALL_STATES : FACT_STATES;
    if (!value || typeof value !== 'object' || !string(value.id) || !FACT_ID.test(value.id) || !string(value.content)
        || !string(value.category) || !states.includes(value.state)
        || !(value.expires_at === null || finite(value.expires_at)) || (value.state === 'confirmed') !== (value.expires_at === null)
        || !revision(value.revision, 1)
        || !(value.confirmed_at === null || value.confirmed_at === undefined || finite(value.confirmed_at))
        || !(value.confirmed_by === null || value.confirmed_by === undefined || string(value.confirmed_by))
        || !Array.isArray(value.sources)) throw Error('invalid_memory_fact');
    const fact = { id: value.id, content: value.content, category: value.category, state: value.state, expires_at: value.expires_at,
      revision: value.revision, confirmed_at: value.confirmed_at ?? null, confirmed_by: value.confirmed_by ?? null,
      sources: value.sources.map(parseSource) };
    return withOwner ? { ...fact, ...parseOwnerFields(value) } : fact;
  }
  const count = value => value === undefined || value === null ? null : Number.isSafeInteger(value) && value >= 0 ? value : (() => { throw Error('invalid_memory_snapshot'); })();
  const cursor = value => value === null ? null : string(value) && CURSOR.test(value) ? value : (() => { throw Error('invalid_memory_snapshot'); })();
  /** One page of GET /api/memory. Fail closed: one malformed item makes the whole page unusable. */
  function parsePage(value) {
    if (!value || typeof value !== 'object' || !finite(value.observed_at) || !Array.isArray(value.facts) || !Array.isArray(value.proposals)
        || value.facts.length > CAP || value.proposals.length > CAP) throw Error('invalid_memory_snapshot');
    const facts = value.facts.map(item => parseFact(item, 'fact')), proposals = value.proposals.map(item => parseFact(item, 'proposal'));
    const ids = new Set();
    for (const item of [...facts, ...proposals]) { if (ids.has(item.id)) throw Error('invalid_memory_snapshot'); ids.add(item.id); }
    return { observed_at: value.observed_at, facts, proposals, pending_index: count(value.pending_index),
      pending_index_cleanup: count(value.pending_index_cleanup), legacy_policy: string(value.legacy_policy) ? value.legacy_policy : null,
      facts_next: cursor(value.facts_next), proposals_next: cursor(value.proposals_next),
      assign_notice: string(value.assign_notice) && value.assign_notice ? value.assign_notice : null };
  }
  /** One page of GET /api/memory/returned ('Devueltos'). */
  function parseReturnedPage(value) {
    if (!value || typeof value !== 'object' || !Array.isArray(value.facts) || value.facts.length > CAP) throw Error('invalid_memory_returned');
    const facts = value.facts.map(item => parseFact(item, 'any'));
    if (new Set(facts.map(item => item.id)).size !== facts.length || facts.some(item => item.owner !== null)) throw Error('invalid_memory_returned');
    return { facts, next: cursor(value.next) };
  }
  function parseCounts(value) {
    const keys = ['currently_owned', 'returned', 'ever_assigned_live', 'owned_proposals'];
    if (!value || typeof value !== 'object' || keys.some(key => !revision(value[key], 0)) || typeof value.retired_owner_prune_pending !== 'boolean') throw Error('invalid_memory_counts');
    return { currently_owned: value.currently_owned, returned: value.returned, ever_assigned_live: value.ever_assigned_live,
      owned_proposals: value.owned_proposals, prune_pending: value.retired_owner_prune_pending };
  }
  /** GET /api/admin/users rows; only id, name and whether the account is active are kept. */
  function parsePeople(value) {
    if (!Array.isArray(value)) throw Error('invalid_people');
    const ids = new Set();
    return value.map(row => {
      if (!row || typeof row !== 'object' || !person(row.id) || !string(row.name) || typeof row.is_active !== 'boolean' || ids.has(row.id)) throw Error('invalid_people');
      ids.add(row.id);
      return { id: row.id, name: row.name, active: row.is_active };
    });
  }

  const APPLIED_TEXT = { confirm: 'Recuerdo conservado sin vencimiento.', reject: 'Propuesta descartada. No se puede recuperar.', delete: 'Recuerdo eliminado.',
    assign: 'Recuerdo asignado. Quedó confirmado y no vence.', unassign: 'El recuerdo quedó sin asignar. Sigue confirmado y no vence.' };
  const ALREADY_TEXT = { reject: 'Ya no estaba guardada.', delete: 'Ya no estaba guardado.' };
  const UNCONFIRMED_RECEIPT = 'La respuesta del servidor no confirma el resultado.';
  // Assignment refusals the ledger raises before writing; a read still has to agree (F3), except
  // has_dependents, which follows a matching CAS and therefore proves nothing changed.
  const OWNER_REFUSALS = {
    409: { memory_owner_conflict: 'Otra sesión cambió este recuerdo.', memory_owner_derived_fact: 'Este recuerdo deriva de otro y no se puede asignar.' },
    404: { memory_fact_unavailable: 'El recuerdo ya no está disponible.' },
    422: { memory_owner_dependents_not_applicable: 'Este recuerdo ya está confirmado: no tiene derivados que eliminar.',
      memory_owner_invalid_person: 'La persona elegida no es válida.', memory_owner_retired: 'La cuenta de esa persona fue retirada.',
      memory_owner_inactive: 'La cuenta de esa persona no está activa.', memory_owner_invalid_request: 'El servidor rechazó los datos de la asignación.',
      memory_fact_invalid: 'El servidor rechazó el identificador del recuerdo.' },
  };
  function parseDependents(detail) {
    const list = detail?.dependents;
    if (!Array.isArray(list) || !list.length || list.some(item => !item || !string(item.id) || !FACT_ID.test(item.id) || !ALL_STATES.includes(item.state))) return null;
    return list.map(item => ({ id: item.id, state: item.state }));
  }
  /**
   * Classify one write answer. applied: a valid 200; not_applied: never sent;
   * provisional: sent and the server says "not applied" (a read must agree, F3);
   * dependents: assignment refused because derived memories exist (nothing changed);
   * uncertain: anything else, including every 401/403 (a 403 can follow a commit).
   */
  function classifyWrite(op, id, outcome) {
    if (!OPS.includes(op)) throw Error('invalid_memory_operation');
    if (outcome && 'value' in outcome) {
      const value = outcome.value;
      if (op === 'confirm') {
        let fact = null;
        try { fact = parseFact(value?.fact, 'fact', false); } catch (_) { /* Malformed receipt. */ }
        return fact && fact.id === id && fact.state === 'confirmed' ? { kind: 'applied', text: APPLIED_TEXT.confirm, fact }
          : { kind: 'uncertain', text: UNCONFIRMED_RECEIPT };
      }
      if (op === 'assign') {
        if (!value || !revision(value.fact_revision, 1) || !revision(value.owner_revision, 1) || !revision(value.deleted_dependents, 0)
            || typeof value.confirmed_now !== 'boolean' || !(value.notice === null || string(value.notice))) return { kind: 'uncertain', text: UNCONFIRMED_RECEIPT };
        const removed = value.deleted_dependents > 0 ? ` Se eliminaron ${value.deleted_dependents} ${value.deleted_dependents === 1 ? 'recuerdo derivado' : 'recuerdos derivados'}.` : '';
        return { kind: 'applied', text: APPLIED_TEXT.assign + removed + (value.notice ? ' ' + value.notice : ''),
          receipt: { fact_revision: value.fact_revision, owner_revision: value.owner_revision, deleted_dependents: value.deleted_dependents, confirmed_now: value.confirmed_now } };
      }
      if (op === 'unassign') {
        if (!value || !revision(value.fact_revision, 1) || !revision(value.owner_revision, 0)) return { kind: 'uncertain', text: UNCONFIRMED_RECEIPT };
        return { kind: 'applied', text: APPLIED_TEXT.unassign, receipt: { fact_revision: value.fact_revision, owner_revision: value.owner_revision } };
      }
      if (typeof value?.deleted !== 'boolean') return { kind: 'uncertain', text: UNCONFIRMED_RECEIPT };
      return { kind: 'applied', text: value.deleted ? APPLIED_TEXT[op] : ALREADY_TEXT[op], deleted: value.deleted };
    }
    const error = outcome?.error;
    if (error?.sent === false) return { kind: 'not_applied', text: 'No se envió: este acceso no permite la operación.' };
    if (error?.answered === true) return { kind: 'uncertain', text: 'La sesión dejó de estar vigente y no pudimos leer la respuesta.' };
    const status = error?.status, code = detailCode(error);
    if (!Number.isInteger(status)) return { kind: 'uncertain', text: 'No se pudo contactar al servidor.' };
    if (status === 401 || status === 403) return { kind: 'uncertain', text: 'La sesión no permitió confirmar el resultado.' };
    if (op === 'assign' || op === 'unassign') {
      if (op === 'assign' && status === 409 && code === 'memory_owner_has_dependents') {
        const dependents = parseDependents(error.detail);
        if (dependents) return { kind: 'dependents', text: string(error.detail?.notice) && error.detail.notice ? error.detail.notice : 'Este recuerdo tiene recuerdos derivados.', dependents };
        return { kind: 'provisional', text: 'El servidor informó recuerdos derivados con un formato desconocido.' };
      }
      const text = OWNER_REFUSALS[status]?.[code];
      if (text) return { kind: 'provisional', text };
    } else {
      if (op === 'confirm' && status === 404 && code === 'memory_proposal_unavailable') return { kind: 'provisional', text: 'La propuesta venció o ya no está disponible.' };
      if (op === 'confirm' && status === 409 && code === 'memory_proposal_required') return { kind: 'provisional', text: 'Este recuerdo no es una propuesta.' };
      if (op === 'reject' && status === 409 && code === 'memory_proposal_state_changed') return { kind: 'provisional', text: 'Otra sesión ya la confirmó; no se descartó.' };
      if (status === 422) return { kind: 'provisional', text: 'El servidor rechazó el identificador del recuerdo.' };
    }
    if (status === 503 && NOT_APPLIED_503.includes(code)) return { kind: 'provisional', text: 'La memoria no está disponible en este momento.' };
    return { kind: 'uncertain', text: status >= 500 ? 'El servidor no pudo completar la operación.' : 'El servidor respondió de forma inesperada.' };
  }

  /**
   * What a write answer does to its marker. A local refusal is final only on the first send:
   * on a retry the earlier send may already have committed, so the marker stays unresolved.
   * 'dependents' is final on any send: the ledger raises it only after the CAS matched.
   */
  function writeDisposition(kind, first) {
    if (kind === 'applied') return 'drop';
    if (kind === 'dependents') return 'final';
    if (kind === 'not_applied') return first ? 'final' : 'unresolved';
    return kind === 'provisional' ? 'provisional' : 'unresolved';
  }
  /** Where an id is visible in a walked snapshot and whether each list was walked completely. */
  function observe(snapshot, id) {
    const fact = snapshot.facts.find(item => item.id === id), proposal = snapshot.proposals.find(item => item.id === id);
    const item = fact || proposal || null;
    return { where: fact ? 'facts' : proposal ? 'proposals' : null, state: item?.state ?? null, item,
      factsComplete: snapshot.factsComplete !== false, proposalsComplete: snapshot.proposalsComplete !== false };
  }
  const STILL = { confirm: 'Todavía figura como propuesta; puede que el cambio no se haya aplicado.',
    reject: 'Todavía figura como propuesta; puede que el cambio no se haya aplicado.', delete: 'Todavía figura guardado; puede que el cambio no se haya aplicado.',
    assign: 'Todavía figura sin esa asignación; puede que el cambio no se haya aplicado.', unassign: 'Todavía figura asignado; puede que el cambio no se haya aplicado.' };
  const INCONCLUSIVE = 'No pudimos comprobarlo: la lista de recuerdos no se pudo recorrer completa.';
  const GONE = 'Ya no está guardado: venció o se eliminó.';
  const OWNER_MOVED = 'La asignación de este recuerdo cambió desde otra sesión. Revisá a quién quedó asignado.';
  /**
   * Resolve a marker from a read that started after the loss (or after the provisional answer).
   * A read never proves "not applied" for an uncertain write (F2): it may still commit later,
   * unless the read shows the operation became impossible (an assignment CAS that moved on).
   * expected: {person_id, expected_fact_revision, expected_owner_revision} for owner changes.
   * Returns {result: applied|final|pending|inconclusive, text}.
   */
  function resolveMarker(op, observation, provisional = false, reason = '', expected = {}) {
    const { where, state, item, factsComplete, proposalsComplete } = observation;
    const complete = factsComplete && proposalsComplete;
    const agree = () => provisional ? { result: 'final', text: reason || STILL[op] } : { result: 'pending', text: STILL[op] };
    if (op === 'assign' || op === 'unassign') {
      if (!item) return complete ? { result: 'final', text: GONE } : { result: 'inconclusive', text: INCONCLUSIVE };
      if (op === 'assign' && item.owner?.person_id === expected.person_id) return { result: 'applied', text: 'Comprobado: el recuerdo quedó asignado y confirmado.' };
      if (op === 'unassign' && item.owner === null) return { result: 'applied', text: 'Comprobado: el recuerdo quedó sin asignar.' };
      // The CAS can no longer match: the write is impossible from now on, whatever happened before.
      if (item.owner_revision > expected.expected_owner_revision || (op === 'assign' && item.revision > expected.expected_fact_revision)) return { result: 'final', text: OWNER_MOVED };
      return agree();
    }
    if (op === 'confirm') {
      if (where === 'facts' && state === 'confirmed') return { result: 'applied', text: 'Comprobado: el recuerdo quedó conservado sin vencimiento.' };
      // An automatic fact can never be confirmed: the operation is impossible from now on.
      if (where === 'facts') return { result: 'final', text: reason || 'Este recuerdo no es una propuesta; no se conservó.' };
      if (where === 'proposals') return agree();
      if (!factsComplete) return { result: 'inconclusive', text: INCONCLUSIVE };
      // With the facts list complete a confirmed fact would be listed; an absent proposal cannot become
      // confirmed later unless it lies beyond an incomplete proposals list.
      if (!proposalsComplete) return provisional ? { result: 'final', text: reason } : { result: 'inconclusive', text: INCONCLUSIVE };
      return { result: 'final', text: 'Ya no está guardada: venció o se eliminó.' };
    }
    if (where === null) {
      if (!complete) return { result: 'inconclusive', text: INCONCLUSIVE };
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
  const derived = item => item.sources.some(source => source.depends_on.length > 0);

  const aborted = error => error?.name === 'AbortError';
  const statusless = error => !Number.isInteger(error?.status) && error?.sent !== false && error?.answered !== true;
  const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.';
  const RETRY_DEPENDENTS = 'No se asignó. Usá «Asignar» para revisar los recuerdos derivados antes de decidir.';

  function createMemoryOwner({ api, session, storage = root.sessionStorage, schedule = setTimeout, cancel = clearTimeout,
    monotonic = () => root.performance.now(), wall = Date.now, onAccountChanged = () => {}, maxPages = MAX_PAGES }) {
    let epoch = 0, visit = 0, opened = false, timer, readSeq = 0, readFloor = 0, refreshing = null, again = false, serial = 0;
    const reads = new Set(), listeners = new Set();
    let state = initial();
    function initial() {
      return { snapshot: null, received: null, read: { loading: false, loaded: false, error: null, observedWall: null },
        markers: {}, kept: {}, notices: {}, announcement: null, extras: emptyExtras() };
    }
    function emptyExtras() { return { returned: null, returnedError: null, counts: null, countsError: null, people: null, peopleError: null }; }
    function capabilities() {
      return { read: session.can('memory:read'), review: session.can('memory:review'),
        assign: session.can('memory:assign') && session.can('memory:read'), people: session.can('admin:manage') };
    }
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
      if (!snapshot) return { facts: [], proposals: [], kept: [], returned: [] };
      const now = serverNow(), marked = id => Boolean(state.markers[id]);
      const facts = snapshot.facts.filter(item => marked(item.id) || live(item, now));
      const proposals = snapshot.proposals.filter(item => marked(item.id) || live(item, now));
      const listed = new Set(facts.map(item => item.id));
      const returned = (state.extras.returned?.items || []).filter(item => marked(item.id) || live(item, now));
      return { facts, proposals, kept: Object.values(state.kept).filter(item => !listed.has(item.id)), returned };
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

    // Markers: one per fact id, never with content (an account id and revisions at most).
    function storageKeys() {
      const keys = [];
      if (!storage) return keys;
      for (let index = 0; index < storage.length; index++) { const key = storage.key(index); if (key?.startsWith(PREFIX)) keys.push(key); }
      return keys;
    }
    function remove(key) { try { storage?.removeItem(key); } catch (_) { /* Best effort. */ } }
    function stored(marker) {
      const value = { op: marker.op, subject: marker.subject, sent_at: marker.sent_at, state: marker.state, ...(marker.reason ? { reason: marker.reason } : {}) };
      if (marker.op === 'assign') Object.assign(value, { person_id: marker.person_id, expected_fact_revision: marker.expected_fact_revision,
        expected_owner_revision: marker.expected_owner_revision, delete_dependents: marker.delete_dependents,
        ...(Number.isInteger(marker.expected_dependents) ? { expected_dependents: marker.expected_dependents } : {}) });
      if (marker.op === 'unassign') value.expected_owner_revision = marker.expected_owner_revision;
      return value;
    }
    function persist(id, marker) {
      if (!storage) throw Error('storage_unavailable');
      if (!marker.subject) throw Error('subject_unavailable');
      storage.setItem(PREFIX + id, JSON.stringify(stored(marker)));
    }
    function setMarker(id, marker) {
      state.markers[id] = marker;
      try { persist(id, marker); } catch (_) { /* The identity was stored before sending. */ }
    }
    function dropMarker(id) { delete state.markers[id]; remove(PREFIX + id); }
    function ownerBody(value) {
      if (value.op === 'assign') {
        if (!person(value.person_id) || !revision(value.expected_fact_revision, 1) || !revision(value.expected_owner_revision, 0) || typeof value.delete_dependents !== 'boolean') throw Error('invalid');
        if (value.expected_dependents !== undefined && !revision(value.expected_dependents, 0)) throw Error('invalid');
        return { person_id: value.person_id, expected_fact_revision: value.expected_fact_revision, expected_owner_revision: value.expected_owner_revision, delete_dependents: value.delete_dependents,
          ...(value.expected_dependents !== undefined ? { expected_dependents: value.expected_dependents } : {}) };
      }
      if (value.op === 'unassign') {
        if (!revision(value.expected_owner_revision, 0)) throw Error('invalid');
        return { expected_owner_revision: value.expected_owner_revision };
      }
      return {};
    }
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
            state: value.state === 'provisional' ? 'provisional' : 'unresolved', reason: string(value.reason) ? value.reason : '', after: -1, ...ownerBody(value) };
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
    function dropContent() { state.snapshot = null; state.received = null; state.kept = {}; state.extras = emptyExtras(); stopTimer(); }

    function settle(id, marker, verdict) {
      if (verdict.result === 'applied' || verdict.result === 'final') {
        dropMarker(id); notice(id, verdict.text); announce(verdict.text);
        if (verdict.result === 'applied' && marker.op !== 'confirm' && marker.op !== 'assign') delete state.kept[id];
        return;
      }
      marker.state = 'unresolved'; marker.reason = marker.reason || '';
      setMarker(id, marker);
      notice(id, verdict.text);
    }
    function resolveAll(started) {
      for (const [id, marker] of Object.entries(state.markers)) {
        if (marker.state === 'sending' || marker.after >= started) continue;
        const verdict = resolveMarker(marker.op, observe(state.snapshot, id), marker.state === 'provisional', marker.reason, marker);
        settle(id, marker, verdict);
      }
      // A kept card stays visible only while an incomplete facts list can hide it (F1).
      for (const id of Object.keys(state.kept)) {
        const listed = state.snapshot.facts.some(item => item.id === id);
        if (listed || state.snapshot.factsComplete) delete state.kept[id];
      }
    }

    /**
     * Walk every page: proposals first, then facts starting from the facts page read together with
     * the last proposals page. Keyset positions never change, so an item that exists during the
     * whole walk is seen at least once; a proposal confirmed mid-walk can be seen twice (facts wins).
     */
    async function walk(current) {
      let page = parsePage(await read('/api/memory'));
      if (!current()) return null;
      const proposals = [...page.proposals], proposalCursors = new Set(), factCursors = new Set();
      let latest = page, proposalsComplete = true, pages = 1;
      while (latest.proposals_next) {
        if (pages >= maxPages) { proposalsComplete = false; break; }
        // Each list has its own keyset: a proposal confirmed mid-walk keeps its cursor in facts.
        if (proposalCursors.has(latest.proposals_next)) throw Error('invalid_memory_snapshot');
        proposalCursors.add(latest.proposals_next);
        page = parsePage(await read('/api/memory?proposals_after=' + encodeURIComponent(latest.proposals_next)));
        if (!current()) return null;
        proposals.push(...page.proposals); latest = page; pages++;
      }
      const facts = [...latest.facts];
      let last = latest, factsComplete = true;
      pages = 1;
      while (last.facts_next) {
        if (pages >= maxPages) { factsComplete = false; break; }
        if (factCursors.has(last.facts_next)) throw Error('invalid_memory_snapshot');
        factCursors.add(last.facts_next);
        page = parsePage(await read('/api/memory?facts_after=' + encodeURIComponent(last.facts_next)));
        if (!current()) return null;
        facts.push(...page.facts); last = page; pages++;
      }
      const factIds = new Set(), proposalIds = new Set();
      for (const item of facts) { if (factIds.has(item.id)) throw Error('invalid_memory_snapshot'); factIds.add(item.id); }
      for (const item of proposals) { if (proposalIds.has(item.id)) throw Error('invalid_memory_snapshot'); proposalIds.add(item.id); }
      return { observed_at: last.observed_at, facts, proposals: proposals.filter(item => !factIds.has(item.id)), factsComplete, proposalsComplete,
        pending_index: last.pending_index, pending_index_cleanup: last.pending_index_cleanup, legacy_policy: last.legacy_policy,
        assign_notice: last.assign_notice ?? page.assign_notice };
    }
    /** 'Devueltos', counts and people: presentation only, each with its own honest error. */
    async function readExtras(current) {
      const caps = capabilities(), extras = emptyExtras();
      if (caps.assign) {
        try {
          const items = [], seen = new Set(), cursors = new Set();
          let next = null, pages = 0, complete = true;
          do {
            if (pages >= maxPages) { complete = false; break; }
            const page = parseReturnedPage(await read('/api/memory/returned' + (next ? '?after=' + encodeURIComponent(next) : '')));
            if (!current()) return null;
            for (const item of page.facts) { if (seen.has(item.id)) throw Error('invalid_memory_returned'); seen.add(item.id); items.push(item); }
            // A cursor seen before (one step back or many) is a cycle, never more pages.
            if (page.next && cursors.has(page.next)) throw Error('invalid_memory_returned');
            if (page.next) cursors.add(page.next);
            next = page.next; pages++;
          } while (next);
          extras.returned = { items, complete };
        } catch (error) { if (!current()) return null; if (aborted(error)) throw error; extras.returnedError = 'No pudimos consultar los recuerdos devueltos.'; }
        try { extras.counts = parseCounts(await read('/api/memory/owners/counts')); }
        catch (error) { if (!current()) return null; if (aborted(error)) throw error; extras.countsError = 'No pudimos consultar los totales de asignación.'; }
      }
      if (caps.assign && caps.people) {
        try { extras.people = parsePeople(await read('/api/admin/users')); }
        catch (error) { if (!current()) return null; if (aborted(error)) throw error; extras.peopleError = 'No pudimos consultar las personas del hogar; no se puede asignar por ahora.'; }
      }
      return current() ? extras : null;
    }

    async function readOnce(ticket) {
      const started = ++readSeq, seen = visit;
      const current = () => ticket === epoch && opened && seen === visit;
      state.read.loading = true; emit();
      let snapshot = null, extras = null, failure = null, malformed = false;
      try {
        snapshot = await walk(current);
        if (snapshot && current()) extras = await readExtras(current);
      } catch (error) {
        if (/^invalid_memory_snapshot$|^invalid_memory_fact$|^invalid_memory_source$/.test(error?.message || '') && !Number.isInteger(error?.status)) malformed = true;
        else failure = error;
      }
      // Reads aborted by leave() belong to a finished visit; an abort seen here is the client timeout.
      if (!current()) return;
      state.read.loading = false;
      if (malformed) { state.read.error = 'El servidor devolvió la memoria con un formato desconocido.'; emit(); return; }
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
      if (!snapshot || !extras) return;
      // A write applied after this read started is newer than it: read again.
      if (started <= readFloor) { again = true; emit(); return; }
      state.snapshot = snapshot; state.extras = extras; state.received = { observed: snapshot.observed_at, mono: monotonic(), wall: wall() };
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
      return view.proposals.find(item => item.id === id) || view.facts.find(item => item.id === id) || view.kept.find(item => item.id === id)
        || view.returned.find(item => item.id === id) || null;
    }
    function affected(id) {
      const view = visible();
      return dependents([...view.facts, ...view.proposals, ...view.kept], id);
    }
    function deleteBlocked(id) {
      if (!Object.keys(state.markers).length) return false;
      return [id, ...affected(id).map(item => item.id)].some(item => state.markers[item]);
    }
    function activePeople() { return (state.extras.people || []).filter(entry => entry.active); }
    function assignable(item) {
      const caps = capabilities();
      return Boolean(item) && caps.assign && caps.people && Boolean(state.snapshot?.assign_notice) && !derived(item) && Array.isArray(state.extras.people);
    }

    function writeRequest(id, marker) {
      if (marker.op === 'delete') return ['/api/memory/facts/' + id, { method: 'DELETE' }];
      if (marker.op === 'assign') return ['/api/memory/facts/' + id + '/owner', { method: 'POST', body: { person_id: marker.person_id,
        expected_fact_revision: marker.expected_fact_revision, expected_owner_revision: marker.expected_owner_revision,
        ...(marker.delete_dependents ? { delete_dependents: true } : {}) } }];
      if (marker.op === 'unassign') return ['/api/memory/facts/' + id + '/owner', { method: 'POST', body: { person_id: null, expected_owner_revision: marker.expected_owner_revision } }];
      return ['/api/memory/proposals/' + id + '/' + marker.op, { method: 'POST' }];
    }
    function patchApplied(id, marker, verdict) {
      if (!state.snapshot || !opened) return;
      const snapshot = state.snapshot;
      if (marker.op === 'confirm' || marker.op === 'reject' || marker.op === 'delete') {
        const previous = [...snapshot.facts, ...snapshot.proposals].find(item => item.id === id);
        const proposals = snapshot.proposals.filter(item => item.id !== id);
        let facts = snapshot.facts.filter(item => item.id !== id);
        if (marker.op === 'confirm') {
          const fact = { ...verdict.fact, owner: previous?.owner ?? null, owner_revision: previous?.owner_revision ?? 0, owner_change: previous?.owner_change ?? null };
          facts = [fact, ...facts]; state.kept[id] = fact;
        }
        state.snapshot = { ...snapshot, facts, proposals };
        if (state.extras.returned && marker.op === 'delete') state.extras.returned = { ...state.extras.returned, items: state.extras.returned.items.filter(item => item.id !== id) };
        return;
      }
      const change = item => {
        if (item.id !== id) return item;
        if (marker.op === 'unassign') return { ...item, owner: null, owner_revision: verdict.receipt.owner_revision, owner_change: item.owner ? 'operator_unassigned' : item.owner_change };
        return { ...item, state: 'confirmed', expires_at: null, revision: verdict.receipt.fact_revision,
          owner: { person_id: marker.person_id, method: 'operator_assignment', revision: verdict.receipt.owner_revision, assigned_at: wall() / 1000 },
          owner_revision: verdict.receipt.owner_revision, owner_change: item.owner ? 'reassigned' : 'assigned' };
      };
      const moved = snapshot.proposals.find(item => item.id === id);
      let facts = snapshot.facts.map(change), proposals = snapshot.proposals.filter(item => item.id !== id || marker.op === 'unassign').map(change);
      if (moved && marker.op === 'assign') { facts = [change(moved), ...facts]; state.kept[id] = facts[0]; }
      state.snapshot = { ...snapshot, facts, proposals };
      if (state.extras.returned && marker.op === 'assign') state.extras.returned = { ...state.extras.returned, items: state.extras.returned.items.filter(item => item.id !== id) };
    }
    /**
     * The receipt reports how many derived memories were deleted, not which ones. Compare the count
     * with the list the operator confirmed and say so on the card when they differ (R1).
     */
    function dependentsMismatch(marker, receipt) {
      if (marker.op !== 'assign' || !marker.delete_dependents || !Number.isInteger(marker.expected_dependents) || !receipt) return '';
      if (receipt.deleted_dependents === marker.expected_dependents) return '';
      return ` Confirmaste ${marker.expected_dependents} ${marker.expected_dependents === 1 ? 'recuerdo derivado' : 'recuerdos derivados'} y se eliminaron ${receipt.deleted_dependents}: la lista cambió mientras decidías. El servidor no informa cuáles se eliminaron.`;
    }
    async function send(id, marker, first) {
      const ticket = epoch;
      const [path, options] = writeRequest(id, marker);
      let outcome;
      // Never aborted: leaving the view must not cut a write the server may commit.
      try { outcome = { value: await api(path, options) }; } catch (error) { outcome = { error }; }
      if (ticket !== epoch || state.markers[id] !== marker) return { stale: true };
      const verdict = classifyWrite(marker.op, id, outcome);
      if (verdict.kind === 'applied') {
        dropMarker(id); readFloor = readSeq; again = again || Boolean(refreshing);
        patchApplied(id, marker, verdict);
        if (marker.op !== 'confirm' && marker.op !== 'assign') delete state.kept[id];
        const text = verdict.text + dependentsMismatch(marker, verdict.receipt);
        notice(id, text); announce(text); emit();
        void refresh();
        return { applied: true, receipt: verdict.receipt };
      }
      const disposition = writeDisposition(verdict.kind, first);
      if (disposition === 'final') {
        // A first send leaves the dependents question to the dialog; a retry (no dialog) says it on the card (R5).
        const text = verdict.kind === 'dependents' && !first ? verdict.text + ' ' + RETRY_DEPENDENTS : verdict.text;
        dropMarker(id); notice(id, verdict.kind === 'dependents' && first ? null : text); announce(text); emit();
        return verdict.kind === 'dependents'
          ? { applied: false, dependents: verdict.dependents, notice: verdict.text, expected: { person_id: marker.person_id, expected_fact_revision: marker.expected_fact_revision, expected_owner_revision: marker.expected_owner_revision } }
          : { applied: false };
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
    async function act(op, id, { confirmed = false, personId = null, deleteDependents = false, expected = null, expectedDependents = null } = {}) {
      const caps = capabilities();
      if (!opened || !OPS.includes(op) || state.markers[id]) return { sent: false };
      if (['confirm', 'reject', 'delete'].includes(op) ? !caps.review : !caps.assign) return { sent: false };
      const item = find(id);
      if (!item) return { sent: false };
      if ((op === 'confirm' || op === 'reject') && item.state !== 'proposed') return { sent: false };
      if (op === 'delete' && (item.state === 'proposed' || !confirmed || deleteBlocked(id))) return { sent: false };
      const extra = {};
      if (op === 'assign') {
        if (!confirmed || !assignable(item) || deleteBlocked(id) || !activePeople().some(entry => entry.id === personId)) return { sent: false };
        // A dependents confirmation resends exactly the revisions the server just refused on (nothing changed).
        const revisions = expected && expected.person_id === personId ? expected : { expected_fact_revision: item.revision, expected_owner_revision: item.owner_revision };
        Object.assign(extra, { person_id: personId, expected_fact_revision: revisions.expected_fact_revision,
          expected_owner_revision: revisions.expected_owner_revision, delete_dependents: deleteDependents === true });
        if (extra.delete_dependents && Number.isInteger(expectedDependents) && expectedDependents >= 0) extra.expected_dependents = expectedDependents;
      }
      if (op === 'unassign') {
        if (!confirmed || !item.owner) return { sent: false };
        extra.expected_owner_revision = item.owner_revision;
      }
      const ticket = epoch;
      if (op === 'delete' || op === 'assign' || op === 'unassign') {
        // An irreversible or ownership write is only sent after the server names the same account (F5).
        const account = await confirmAccount();
        if (ticket !== epoch) return { sent: false };
        if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
        if (account !== 'same') { notice(id, ACCOUNT_UNKNOWN); announce(ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
        if (state.markers[id] || ((op === 'delete' || op === 'assign') && deleteBlocked(id))) return { sent: false };
      }
      const marker = { op, subject: subject(), sent_at: wall(), state: 'sending', reason: '', after: Infinity, ...extra };
      try { persist(id, marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado de la operación; no se envió.';
        notice(id, text); announce(text); emit(); return { sent: false };
      }
      state.markers[id] = marker; notice(id, null); emit();
      return send(id, marker, true);
    }
    /** Resend the same operation (same expected revisions): its answer settles the marker. */
    async function retry(id) {
      const marker = state.markers[id];
      if (!opened || !marker || marker.state === 'sending') return { sent: false };
      if (['confirm', 'reject', 'delete'].includes(marker.op) ? !capabilities().review : !capabilities().assign) return { sent: false };
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
      const view = visible(), snapshot = state.snapshot, extras = state.extras;
      return { facts: view.facts, proposals: view.proposals, kept: view.kept, returned: view.returned, loaded: Boolean(snapshot), read: { ...state.read },
        factsComplete: !snapshot || snapshot.factsComplete, proposalsComplete: !snapshot || snapshot.proposalsComplete,
        // Kept for the U3 view contract: 'full' now means the walk stopped before the end.
        factsFull: Boolean(snapshot && !snapshot.factsComplete), proposalsFull: Boolean(snapshot && !snapshot.proposalsComplete),
        pendingIndex: snapshot?.pending_index ?? null, pendingCleanup: snapshot?.pending_index_cleanup ?? null, legacyPolicy: snapshot?.legacy_policy ?? null,
        assignNotice: snapshot?.assign_notice ?? null,
        returnedLoaded: Boolean(extras.returned), returnedComplete: extras.returned?.complete !== false, returnedError: extras.returnedError,
        counts: extras.counts, countsError: extras.countsError, people: extras.people ? extras.people.map(entry => ({ ...entry })) : null, peopleError: extras.peopleError,
        markers: Object.fromEntries(Object.entries(state.markers).map(([id, marker]) => [id, { op: marker.op, state: marker.state }])),
        notices: { ...state.notices }, announcement: state.announcement, capabilities: capabilities(), subject: subject() };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, reload, subscribe, getState, capabilities, dependents: affected, deleteBlocked,
      assignable: id => assignable(find(id)),
      confirm: id => act('confirm', id), reject: id => act('reject', id), remove: (id, options) => act('delete', id, options),
      assign: (id, personId, options = {}) => act('assign', id, { ...options, personId }), unassign: (id, options) => act('unassign', id, options), retry,
      isOpen: () => opened };
  }

  const exported = { createMemoryOwner, parsePage, parseSnapshot: parsePage, parseReturnedPage, parseCounts, parsePeople, classifyWrite, writeDisposition,
    resolveMarker, observe, dependents, describeReadError, PREFIX, ENDED_KEY, CAP, MAX_PAGES, RETRY_DEPENDENTS };
  root.NexoMemoryOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
