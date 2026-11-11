/* NEXO administration of the voice stage (no DOM), U3b-4a-3: the G1 switch, the consents list with
 * the verified profile, multi-profile people, suspended profiles, reopen and resume. Same rules as
 * the Memoria owner: reads may be aborted, writes never are; a pending operation is bound to the
 * account that sent it and stored with identifiers and revisions only (no names); only a read that
 * started after a loss settles it. No scores or attempt details exist in any answer it reads. */
(function (root) {
  'use strict';
  const PREFIX = 'kza:voice-admin-op:';
  const OPS = ['switch', 'reopen', 'resume'];
  const STATES = ['pending_verification', 'active', 'verification_locked', 'revoked', 'invalidated'];
  const PERSON = /^[A-Za-z0-9_.:@-]{1,128}$/;
  const PROFILE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const MAX_CONSENTS = 256;
  // memory_voice_api.py copy; the parity test pins each string.
  const VOICES_DISCLOSURE = "Para evitar confusiones, KZA compara cada '¿acordate que…?' con todas las voces registradas en el hogar, incluidas las de personas no inscriptas, sólo para descartar coincidencias. Nunca asigna un recuerdo a alguien que no se haya inscripto y verificado. Vincular una voz a una persona no le da permisos ni la inscribe.";
  const MULTI_PROFILE_NOTICE = 'Esta persona tiene más de una voz vinculada: no puede verificarse ni recibir propuestas por voz hasta que quede una sola';
  const STATE_LABELS = { pending_verification: 'Falta verificar', active: 'Activa', verification_locked: 'Verificación bloqueada', revoked: 'Dada de baja', invalidated: 'Sin efecto (cambió la voz)' };
  const REFUSALS = {
    memory_voice_disabled: 'La memoria por voz está desactivada en la configuración. No se cambió nada.',
    memory_voice_not_configured: 'La configuración del servidor no habilita la memoria por voz: el interruptor no se puede encender. No se cambió nada.',
    memory_voice_conflict: 'El estado cambió mientras decidías. Revisalo y volvé a intentarlo.',
    memory_voice_consent_state: 'Esa inscripción ya no está bloqueada. No se cambió nada.',
    memory_voice_invalid_request: 'El servidor rechazó el pedido. No se cambió nada.',
  };
  const string = value => typeof value === 'string';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const revision = value => Number.isSafeInteger(value) && value >= 0;
  const keys = value => Object.keys(value).sort().join(',');
  function detailCode(error) {
    const detail = error?.detail;
    return string(detail) ? detail : string(detail?.code) ? detail.code : null;
  }
  function parseSwitch(value) {
    if (!value || typeof value !== 'object' || keys(value) !== 'configured,effective,enabled,revision' || typeof value.configured !== 'boolean'
        || typeof value.enabled !== 'boolean' || typeof value.effective !== 'boolean' || !revision(value.revision)
        || value.effective !== (value.configured && value.enabled)) throw Error('invalid_voice_admin');
    return { configured: value.configured, enabled: value.enabled, effective: value.effective, revision: value.revision };
  }
  function parseConsents(value) {
    if (!value || typeof value !== 'object' || keys(value) !== 'consents,disclosure,multiple_profiles,multiple_profiles_notice,suspended_profiles,truncated'
        || !Array.isArray(value.consents) || value.consents.length > MAX_CONSENTS || typeof value.truncated !== 'boolean' || !string(value.disclosure)
        || !string(value.multiple_profiles_notice) || !Array.isArray(value.suspended_profiles) || !Array.isArray(value.multiple_profiles)) throw Error('invalid_voice_admin');
    const people = new Set(), consents = value.consents.map(row => {
      if (!row || keys(row) !== 'changed_at,linked_live_profiles,person_id,revision,state,verified_profile' || !string(row.person_id) || !PERSON.test(row.person_id)
          || people.has(row.person_id) || !STATES.includes(row.state) || !revision(row.revision) || !finite(row.changed_at)
          || !Number.isSafeInteger(row.linked_live_profiles) || row.linked_live_profiles < 0) throw Error('invalid_voice_admin');
      const verified = row.verified_profile;
      if (!(verified === null || (verified && keys(verified) === 'id,name' && string(verified.id) && PROFILE.test(verified.id) && (verified.name === null || string(verified.name))))) throw Error('invalid_voice_admin');
      people.add(row.person_id);
      return { person_id: row.person_id, state: row.state, revision: row.revision, changed_at: row.changed_at, linked: row.linked_live_profiles,
        verified: verified === null ? null : { id: verified.id, name: verified.name } };
    });
    const profiles = new Set(), suspended = value.suspended_profiles.map(row => {
      if (!row || keys(row) !== 'profile_id,revision' || !string(row.profile_id) || !PROFILE.test(row.profile_id) || profiles.has(row.profile_id)
          || !Number.isSafeInteger(row.revision) || row.revision < 1) throw Error('invalid_voice_admin');
      profiles.add(row.profile_id); return { profile_id: row.profile_id, revision: row.revision };
    });
    const multiple = value.multiple_profiles.map(row => {
      if (!row || keys(row) !== 'linked_live_profiles,person_id' || !string(row.person_id) || !PERSON.test(row.person_id)
          || !Number.isSafeInteger(row.linked_live_profiles) || row.linked_live_profiles < 2) throw Error('invalid_voice_admin');
      return { person_id: row.person_id, linked: row.linked_live_profiles };
    });
    return { consents, suspended, multiple, truncated: value.truncated };
  }
  /** GET /api/admin/users: only id and name are kept, to name people in this view. */
  function parseNames(value) {
    if (!Array.isArray(value)) throw Error('invalid_names');
    const names = {};
    for (const row of value) { if (row && string(row.id) && PERSON.test(row.id) && string(row.name)) names[row.id] = row.name; }
    return names;
  }

  const UNCONFIRMED = 'La respuesta del servidor no confirma el resultado.';
  /** applied | refused | not_applied | uncertain; 401/403 on a write are uncertain. */
  function classifyWrite(op, outcome) {
    if (!OPS.includes(op)) throw Error('invalid_voice_admin_operation');
    if (outcome && 'value' in outcome) {
      const value = outcome.value;
      if (op === 'switch') { try { return { kind: 'applied', switch: parseSwitch(value), text: value.enabled ? 'Memoria por voz encendida.' : 'Memoria por voz apagada.' }; } catch (_) { return { kind: 'uncertain', text: UNCONFIRMED }; } }
      if (op === 'resume') return value && keys(value) === 'resumed' && value.resumed === true ? { kind: 'applied', text: 'Se reanudaron las propuestas por voz de ese perfil.' } : { kind: 'uncertain', text: UNCONFIRMED };
      return value && keys(value) === 'revision,state' && STATES.includes(value.state) && revision(value.revision)
        ? { kind: 'applied', text: value.state === 'revoked' ? 'Se desbloqueó: la persona se había dado de baja y sigue de baja.' : 'Se habilitó un nuevo intento de verificación.' }
        : { kind: 'uncertain', text: UNCONFIRMED };
    }
    const error = outcome?.error;
    if (error?.sent === false) return { kind: 'not_applied', text: 'No se envió: este acceso no permite la operación.' };
    if (error?.answered === true) return { kind: 'uncertain', text: 'La sesión dejó de estar vigente y no pudimos leer la respuesta.' };
    const status = error?.status, code = detailCode(error);
    if (!Number.isInteger(status)) return { kind: 'uncertain', text: 'No se pudo contactar al servidor.' };
    if (status === 401 || status === 403) return { kind: 'uncertain', text: 'La sesión no permitió confirmar el resultado.' };
    if ([409, 422].includes(status)) return { kind: 'refused', code, text: REFUSALS[code] || 'El servidor rechazó el pedido. No se cambió nada.' };
    return { kind: 'uncertain', text: status >= 500 ? 'El servidor no pudo completar la operación.' : 'El servidor respondió de forma inesperada.' };
  }
  const STILL = 'Todavía no cambió; puede que el pedido no se haya aplicado. Podés volver a intentarlo.';
  const CHANGED = 'El estado cambió, pero no podemos confirmar que haya sido por este pedido. Revisalo.';
  const INCONCLUSIVE = 'No pudimos comprobarlo: la lista de inscripciones no está disponible.';
  /** Judge a lost write from a read that started after the loss; restored markers are never 'Comprobado'. */
  function resolveMarker(marker, observed) {
    if (marker.op === 'switch') {
      if (!observed.switch) return { result: 'pending', text: INCONCLUSIVE };
      if (observed.switch.revision === marker.expected_revision) return { result: 'final', text: STILL };
      if (!marker.restored && observed.switch.revision === marker.expected_revision + 1 && observed.switch.enabled === marker.enabled) return { result: 'applied', text: marker.enabled ? 'Comprobado: quedó encendida.' : 'Comprobado: quedó apagada.' };
      return { result: 'final', text: CHANGED };
    }
    if (!observed.list) return { result: 'pending', text: INCONCLUSIVE };
    if (marker.op === 'reopen') {
      const row = observed.list.consents.find(item => item.person_id === marker.target);
      if (row && row.revision === marker.expected_revision) return { result: 'final', text: STILL };
      if (!marker.restored && row && row.revision === marker.expected_revision + 1 && ['pending_verification', 'revoked'].includes(row.state)) return { result: 'applied', text: 'Comprobado: se habilitó un nuevo intento.' };
      if (!row && observed.list.truncated) return { result: 'pending', text: INCONCLUSIVE };
      return { result: 'final', text: CHANGED };
    }
    const row = observed.list.suspended.find(item => item.profile_id === marker.target);
    if (row && row.revision === marker.expected_revision) return { result: 'final', text: STILL };
    if (!row && !marker.restored) return { result: 'applied', text: 'Comprobado: ese perfil ya no está en pausa.' };
    return { result: 'final', text: CHANGED };
  }
  function describeReadError(error) {
    const status = error?.status;
    if (error?.sent === false || status === 403) return 'Este acceso no permite administrar la memoria por voz.';
    if (status === 401) return 'Iniciá sesión para administrar la memoria por voz.';
    if (status === 503) return 'La memoria por voz no está disponible en este momento.';
    if (Number.isInteger(status)) return 'El servidor no pudo completar la consulta.';
    return 'No pudimos consultar la memoria por voz.';
  }
  const aborted = error => error?.name === 'AbortError';
  const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.';

  const ENDED_KEY = 'kza:voice-admin-op-ended';
  const ENDED_NOTICE = 'Una operación de memoria por voz quedó sin confirmar al terminar la sesión anterior. Revisá el estado antes de repetirla.';

  function createVoiceConsentsOwner({ api, session, storage = root.sessionStorage, wall = Date.now, onAccountChanged = () => {} }) {
    let epoch = 0, visit = 0, opened = false, readSeq = 0, refreshing = null, again = false, serial = 0;
    const reads = new Set(), listeners = new Set();
    let state = initial();
    function initial() {
      return { switch: null, list: null, listReason: null, names: {}, read: { loading: false, loaded: false, error: null, observedWall: null },
        markers: {}, notices: {}, announcement: null };
    }
    function capabilities() {
      return { read: session.can('memory:assign') && session.can('memory:read'), toggle: session.can('memory:assign') && session.can('admin:manage'),
        reopen: session.can('memory:assign'), resume: ['memory:assign', 'admin:manage', 'voices:review'].every(permission => session.can(permission)), names: session.can('admin:manage') };
    }
    function emit() { for (const listener of [...listeners]) { try { listener(); } catch (_) { /* A view must not break the owner. */ } } }
    function announce(text) { state.announcement = { serial: ++serial, text }; }
    function notice(key, text) { if (text) state.notices[key] = text; else delete state.notices[key]; }
    function subject() { return typeof session.subject === 'function' ? session.subject() : null; }
    async function read(path) {
      const controller = new AbortController(); reads.add(controller);
      try { return await api(path, { signal: controller.signal }); } finally { reads.delete(controller); }
    }
    const markerKey = (op, target) => op + ':' + target;
    function storageKeys() {
      const found = [];
      if (!storage) return found;
      for (let index = 0; index < storage.length; index++) { const key = storage.key(index); if (key?.startsWith(PREFIX)) found.push(key); }
      return found;
    }
    function remove(key) { try { storage?.removeItem(PREFIX + key); } catch (_) { /* Best effort. */ } }
    function persist(key, marker) {
      if (!storage) throw Error('storage_unavailable');
      if (!marker.subject) throw Error('subject_unavailable');
      storage.setItem(PREFIX + key, JSON.stringify({ op: marker.op, subject: marker.subject, sent_at: marker.sent_at, expected_revision: marker.expected_revision,
        ...(marker.op === 'switch' ? { enabled: marker.enabled } : {}) }));
    }
    function dropMarker(key) { delete state.markers[key]; remove(key); }
    function validTarget(op, target) { return op === 'switch' ? target === 'switch' : op === 'reopen' ? PERSON.test(target) : PROFILE.test(target); }
    function restore() {
      let found;
      try { found = storageKeys(); } catch (_) { return; }
      const current = subject();
      for (const full of found) {
        const key = full.slice(PREFIX.length), [op, ...rest] = key.split(':'), target = rest.join(':');
        try {
          const value = JSON.parse(storage.getItem(full));
          if (!OPS.includes(op) || value?.op !== op || !validTarget(op, target) || !string(value.subject) || !revision(value.expected_revision)
              || (op === 'switch' && typeof value.enabled !== 'boolean')) throw Error('invalid');
          if (!current || value.subject !== current) { notice('session', 'Una operación de otra cuenta quedó sin confirmar; no se reintentó.'); throw Error('foreign'); }
          if (state.markers[key]) continue;
          state.markers[key] = { op, target, subject: value.subject, sent_at: finite(value.sent_at) ? value.sent_at : null, expected_revision: value.expected_revision,
            enabled: value.enabled, state: 'unresolved', after: -1, restored: true };
          notice(key, 'No sabemos si se aplicó la última operación. Consultando…');
        } catch (_) { try { storage.removeItem(full); } catch (_) { /* Best effort. */ } }
      }
    }
    /** 'same' needs the cached subject, the operator role and memory:assign plus memory:read in the server's answer. */
    async function confirmAccount() {
      const expected = subject(), seen = visit, wasOpen = opened;
      if (!expected) return 'changed';
      try {
        const identity = await read('/api/admin/auth/whoami');
        return identity?.ok === true && identity.subject === expected && identity.role === 'operator' && session.role() === 'operator'
          && Array.isArray(identity.permissions) && ['memory:assign', 'memory:read'].every(permission => identity.permissions.includes(permission)) ? 'same' : 'changed';
      } catch (error) {
        if (aborted(error) && (wasOpen && (!opened || seen !== visit))) return 'aborted';
        return [401, 403].includes(error?.status) ? 'changed' : 'unknown';
      }
    }
    function accountChanged() {
      clear();
      notice('session', 'La cuenta de esta pestaña cambió en otra pestaña. No se consultó ni se envió nada.');
      emit();
      try { onAccountChanged(); } catch (_) { /* The page re-reads the session on its own. */ }
    }
    async function readOnce(ticket) {
      const started = ++readSeq, seen = visit;
      const current = () => ticket === epoch && opened && seen === visit;
      state.read.loading = true; emit();
      let switchValue = null, list = null, listReason = null, names = state.names, failure = null, malformed = false;
      try {
        switchValue = parseSwitch(await read('/api/memory/voice/switch'));
        try { list = parseConsents(await read('/api/memory/voice/consents')); }
        catch (error) {
          if (error?.message === 'invalid_voice_admin') throw error;
          if (error?.status === 409 && detailCode(error) === 'memory_voice_disabled') listReason = 'disabled'; else throw error;
        }
        if (capabilities().names) { try { names = parseNames(await read('/api/admin/users')); } catch (error) { if (aborted(error)) throw error; } }
      } catch (error) { if (error?.message === 'invalid_voice_admin') malformed = true; else failure = error; }
      if (!current()) return;
      state.read.loading = false;
      if (malformed || failure) {
        state.switch = null; state.list = null; state.listReason = null;
        state.read.error = malformed ? 'El servidor devolvió la memoria por voz con un formato desconocido.' : describeReadError(failure);
        emit(); return;
      }
      state.switch = switchValue; state.list = list; state.listReason = listReason; state.names = names;
      state.read = { loading: false, loaded: true, error: null, observedWall: wall() };
      for (const [key, marker] of Object.entries(state.markers)) {
        if (marker.state === 'sending' || marker.after >= started) continue;
        const verdict = resolveMarker(marker, { switch: switchValue, list });
        if (verdict.result === 'pending') { notice(key, verdict.text); continue; }
        dropMarker(key); notice(key, verdict.text); announce(verdict.text);
      }
      emit();
    }
    function refresh() {
      if (!opened || !capabilities().read) return Promise.resolve();
      if (refreshing) { again = true; return refreshing; }
      const ticket = epoch;
      const running = (async () => {
        try { do { again = false; await readOnce(ticket); } while (again && ticket === epoch && opened); }
        finally { if (refreshing === running) refreshing = null; }
      })();
      refreshing = running;
      return running;
    }
    let reloading = null;
    function reload() {
      if (!opened) return Promise.resolve();
      if (!reloading) reloading = (async () => {
        const ticket = epoch, seen = visit, account = await confirmAccount();
        if (ticket !== epoch || !opened || seen !== visit || account === 'aborted') return;
        if (account === 'changed') { accountChanged(); return; }
        if (account === 'unknown') { state.read.error = ACCOUNT_UNKNOWN; emit(); return; }
        await refresh();
      })().finally(() => { reloading = null; });
      return reloading;
    }
    function request(marker) {
      if (marker.op === 'switch') return ['/api/memory/voice/switch', { method: 'POST', body: { enabled: marker.enabled, expected_revision: marker.expected_revision } }];
      if (marker.op === 'reopen') return ['/api/memory/voice/consents/' + encodeURIComponent(marker.target) + '/reopen', { method: 'POST', body: { expected_revision: marker.expected_revision } }];
      return ['/api/voices/profiles/' + marker.target + '/attribution/resume', { method: 'POST', body: { expected_revision: marker.expected_revision } }];
    }
    async function send(key, marker) {
      const ticket = epoch, [path, options] = request(marker);
      let outcome;
      // Never aborted: leaving the view must not cut a write the server may commit.
      try { outcome = { value: await api(path, options) }; } catch (error) { outcome = { error }; }
      if (ticket !== epoch || state.markers[key] !== marker) return { stale: true };
      const verdict = classifyWrite(marker.op, outcome);
      if (verdict.kind !== 'uncertain') {
        dropMarker(key); notice(key, verdict.text); announce(verdict.text);
        if (verdict.switch) state.switch = verdict.switch;
        emit();
        if (verdict.kind !== 'not_applied') { again = again || Boolean(refreshing); void refresh(); }
        return { applied: verdict.kind === 'applied', refused: verdict.kind === 'refused' };
      }
      marker.state = 'unresolved'; marker.after = readSeq;
      try { persist(key, marker); } catch (_) { /* Stored before sending. */ }
      const text = 'Sin confirmar. ' + verdict.text + ' Consultando…';
      notice(key, text); announce(text); emit();
      again = again || Boolean(refreshing); void refresh();
      return { uncertain: true };
    }
    function current(op, target) {
      if (op === 'switch') return state.switch ? { revision: state.switch.revision } : null;
      if (op === 'reopen') { const row = state.list?.consents.find(item => item.person_id === target); return row && row.state === 'verification_locked' ? { revision: row.revision } : null; }
      const row = state.list?.suspended.find(item => item.profile_id === target); return row ? { revision: row.revision } : null;
    }
    function permitted(op) { const caps = capabilities(); return op === 'switch' ? caps.toggle : op === 'reopen' ? caps.reopen : caps.resume; }
    /** Every write is confirmed by a dialog and a whoami; the CAS revision is the one the operator saw. */
    async function act(op, target, { confirmed = false, enabled } = {}) {
      const key = markerKey(op, target), seenRow = current(op, target);
      if (!opened || !OPS.includes(op) || !validTarget(op, target) || !permitted(op) || !confirmed || !seenRow || state.markers[key]?.state === 'sending') return { sent: false };
      if (op === 'switch' && (typeof enabled !== 'boolean' || (enabled && !state.switch.configured))) return { sent: false };
      const ticket = epoch, account = await confirmAccount();
      if (ticket !== epoch) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      if (account !== 'same') { notice(key, ACCOUNT_UNKNOWN); announce(ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      if (!opened || state.markers[key]?.state === 'sending') return { sent: false };
      if (current(op, target)?.revision !== seenRow.revision) { notice(key, REFUSALS.memory_voice_conflict); announce(REFUSALS.memory_voice_conflict); emit(); return { sent: false }; }
      const marker = { op, target, subject: subject(), sent_at: wall(), expected_revision: seenRow.revision, enabled, state: 'sending', after: Infinity };
      try { persist(key, marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado de la operación; no se envió.';
        notice(key, text); announce(text); emit(); return { sent: false };
      }
      state.markers[key] = marker; notice(key, null); emit();
      return send(key, marker);
    }
    async function open() {
      if (!capabilities().read) return true;
      opened = true; visit++; restore();
      try { if (storage?.getItem(ENDED_KEY)) { storage.removeItem(ENDED_KEY); notice('session', ENDED_NOTICE); } }
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
      opened = false; again = false;
      for (const controller of reads) controller.abort();
      reads.clear(); refreshing = null;
      state.switch = null; state.list = null; state.listReason = null; state.names = {};
      state.read = { loading: false, loaded: false, error: null, observedWall: state.read.observedWall };
      emit();
      return Promise.resolve(true);
    }
    /** wipe=false (pagehide, a failed bootstrap) keeps the stored markers for the next load; only the end of the session removes them. */
    function clear({ wipe = true } = {}) {
      const pending = Object.keys(state.markers).length > 0;
      epoch++; leave(); state = initial();
      if (wipe) {
        let found = [];
        try { found = storageKeys(); } catch (_) { /* Nothing to wipe. */ }
        for (const full of found) { try { storage.removeItem(full); } catch (_) { /* Best effort. */ } }
        if (pending || found.length) { try { storage.setItem(ENDED_KEY, '1'); } catch (_) { /* Best effort. */ } }
      }
      emit();
      return Promise.resolve(true);
    }
    function personName(personId) { return state.names[personId] || 'Persona sin nombre visible'; }
    /** Voces: the line next to a profile ('Verificada por …' or the pause), or null. */
    function profileNote(profileId) {
      if (!state.list) return null;
      const verified = state.list.consents.filter(row => row.state === 'active' && row.verified?.id === profileId).map(row => personName(row.person_id));
      const parts = verified.length ? ['Verificada por ' + verified.join(', ')] : [];
      if (state.list.suspended.some(row => row.profile_id === profileId)) parts.push('Propuestas por voz en pausa');
      return parts.length ? parts.join(' · ') : null;
    }
    function getState() {
      return { switch: state.switch ? { ...state.switch } : null,
        list: state.list ? { truncated: state.list.truncated, consents: state.list.consents.map(row => ({ ...row, verified: row.verified ? { ...row.verified } : null, name: personName(row.person_id) })),
          suspended: state.list.suspended.map(row => ({ ...row })), multiple: state.list.multiple.map(row => ({ ...row, name: personName(row.person_id) })) } : null,
        listReason: state.listReason, read: { ...state.read },
        markers: Object.fromEntries(Object.entries(state.markers).map(([key, marker]) => [key, { op: marker.op, state: marker.state, restored: marker.restored === true }])),
        notices: { ...state.notices }, announcement: state.announcement, capabilities: capabilities() };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, reload, subscribe, getState, capabilities, profileNote, isOpen: () => opened,
      setSwitch: (enabled, options = {}) => act('switch', 'switch', { ...options, enabled }),
      reopen: (personId, options) => act('reopen', personId, options), resume: (profileId, options) => act('resume', profileId, options) };
  }
  const exported = { createVoiceConsentsOwner, parseSwitch, parseConsents, parseNames, classifyWrite, resolveMarker, describeReadError,
    PREFIX, ENDED_KEY, VOICES_DISCLOSURE, MULTI_PROFILE_NOTICE, STATE_LABELS, REFUSALS, MAX_CONSENTS };
  root.NexoVoiceConsentsOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
