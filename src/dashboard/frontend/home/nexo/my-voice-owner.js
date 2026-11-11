/* NEXO 'Memoria por voz' owner (no DOM), for a household account bound to a person (U3b-4a-3).
 * Same rules as the 'Mis recuerdos' owner: reads may be aborted, writes never are; a pending
 * operation is bound to the account that sent it and stored without the room, the phrase or any
 * other content; only a status read that started after a loss settles it. With the settings flag
 * or the switch off only the status and the revocation are offered (design v4, 4a-2 routes). */
(function (root) {
  'use strict';
  const PREFIX = 'kza:my-voice-op:';
  const OPS = ['request', 'revoke', 'verify'];
  const STATES = ['pending_verification', 'active', 'verification_locked', 'revoked', 'invalidated'];
  const ROOM = /^[a-z0-9][a-z0-9_-]{0,63}$/;
  // whoami's opaque per-grant tag (home + person only): it tells grants apart and authorizes nothing.
  const BINDING = /^[0-9a-f]{32}$/;
  const OFF_KEYS = 'enabled,revision,state,window';
  const ON_KEYS = 'disclosure,enabled,failures_left,message,revision,state,window';
  // memory_voice_api.py copy (design v4 disclosure_copy); the parity test pins each string.
  const OPT_IN_DISCLOSURE = "Si te inscribís, lo que pidas recordar con tu voz ('acordate que…') puede quedar como propuesta tuya. Hasta que la administración la confirme vas a ver sólo un aviso sin el texto: podés descartarla, o decir 'No es mío' si no fuiste vos; si nadie la confirma, vence sola. Para inscribirte tenés que decir en voz alta, en una habitación, una frase que te mostramos. Como cualquier cosa que se dice en casa, esa frase puede quedar unos segundos en el registro de voces que revisa la administración o un evaluador autorizado, hasta 48 horas en la transcripción ambiental de esa habitación, y el audio de las primeras palabras puede quedar en los clips de activación ('Nexa') que se guardan para mejorar el reconocimiento. KZA compara cada pedido con todas las voces registradas en el hogar sólo para descartar confusiones: si no está seguro, la propuesta queda para la administración. Tu voz no te da permisos ni acceso. Podés darte de baja en cualquier momento.";
  const REENROLMENT_NOTICE = 'Tenés más de una voz registrada a tu nombre. Para inscribirte, pedile a la administración que deje una sola.';
  const INSUFFICIENT_PROFILES_MESSAGE = 'Todavía no se puede verificar tu voz en este hogar. Podés volver a intentarlo más adelante.';
  const STATE_MESSAGES = {
    active: 'Listo: verificamos tu voz. Desde ahora, lo que pidas recordar con tu voz puede quedar como propuesta tuya.',
    verification_locked: 'Hubo varios intentos sin éxito (también cuenta si otra persona dijo tu frase). Pedile a la administración que habilite uno nuevo.',
  };
  const STATE_LABELS = { pending_verification: 'Inscripción pedida: falta verificar tu voz.', active: 'Inscripción activa.',
    verification_locked: 'Verificación bloqueada.', revoked: 'Te diste de baja.', invalidated: 'Tu inscripción quedó sin efecto porque cambió tu voz registrada.' };
  const NOT_ENROLLED = 'No estás inscripto en la memoria por voz.';
  const OFF = 'La memoria por voz está desactivada en este hogar.';
  // K: the revocation line only when there is a consent to revoke.
  const OFF_REVOCABLE = 'Podés darte de baja.';
  // Refusals the server answers before changing anything (409/422/429), in Spanish.
  const REFUSALS = {
    memory_voice_disabled: 'La memoria por voz está desactivada en este hogar. No se cambió nada.',
    memory_voice_conflict: 'Tu estado cambió mientras decidías. Revisalo y volvé a intentarlo.',
    memory_voice_consent_state: 'Esta acción ya no corresponde a tu estado actual. Revisalo.',
    memory_voice_consent_locked: STATE_MESSAGES.verification_locked,
    memory_voice_no_profile: 'No tenés una voz registrada a tu nombre. Pedile a la administración que registre y vincule tu voz.',
    memory_voice_profile_ambiguous: REENROLMENT_NOTICE,
    memory_voice_insufficient_profiles: INSUFFICIENT_PROFILES_MESSAGE,
    memory_voice_scoring_incomplete: INSUFFICIENT_PROFILES_MESSAGE,
    memory_voice_profile_suspended: 'Tu voz está en pausa para propuestas porque varias no eran tuyas. La administración puede reanudarla.',
    memory_voice_room_unavailable: 'Esa habitación no está escuchando ahora. Elegí otra o probá más tarde. Cuenta como uno de los intentos de hoy.',
    memory_voice_rate_limited: 'Ya abriste las 6 ventanas de verificación de hoy. Probá de nuevo mañana.',
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
  function parseWindow(value) {
    if (value === null) return null;
    if (!value || typeof value !== 'object' || keys(value) !== 'expires_at,phrase,room_id' || !string(value.room_id) || !ROOM.test(value.room_id)
        || !(value.phrase === null || (string(value.phrase) && value.phrase.length > 0 && value.phrase.length <= 200)) || !finite(value.expires_at)) throw Error('invalid_my_voice');
    return { room_id: value.room_id, phrase: value.phrase, expires_at: value.expires_at };
  }
  /** GET /api/memory/mine/voice: the flag-off shape (4 keys) or the full shape (7 keys), nothing else. */
  function parseStatus(value) {
    if (!value || typeof value !== 'object') throw Error('invalid_my_voice');
    const shape = keys(value);
    if (shape !== OFF_KEYS && shape !== ON_KEYS) throw Error('invalid_my_voice');
    if (!(value.state === null || STATES.includes(value.state)) || !revision(value.revision) || typeof value.enabled !== 'boolean') throw Error('invalid_my_voice');
    if (shape === OFF_KEYS) {
      if (value.enabled !== false || value.window !== null) throw Error('invalid_my_voice');
      return { state: value.state, revision: value.revision, enabled: false, configured: false, window: null, failuresLeft: null };
    }
    const failures = value.failures_left;
    if (!(failures === null || (Number.isSafeInteger(failures) && failures >= 0 && failures <= 100)) || (failures !== null) !== (value.state === 'pending_verification')
        || !(value.message === null || string(value.message)) || !string(value.disclosure)) throw Error('invalid_my_voice');
    const window = parseWindow(value.window);
    if (window && value.state !== 'pending_verification') throw Error('invalid_my_voice');
    return { state: value.state, revision: value.revision, enabled: value.enabled, configured: true, window, failuresLeft: failures };
  }
  /** GET /api/home/catalog rooms: only id and name are kept. */
  function parseRooms(value) {
    if (!value || typeof value !== 'object' || !Array.isArray(value.rooms)) throw Error('invalid_rooms');
    const rooms = [], seen = new Set();
    for (const room of value.rooms) {
      if (!room || !string(room.room_id) || !ROOM.test(room.room_id) || !string(room.name) || seen.has(room.room_id)) throw Error('invalid_rooms');
      seen.add(room.room_id); rooms.push({ room_id: room.room_id, name: room.name });
    }
    return rooms;
  }

  const UNCONFIRMED = 'La respuesta del servidor no confirma el resultado.';
  /** applied | refused | not_applied | uncertain. 401/403 on a write are uncertain (the server may have acted). */
  const WINDOW_OPENED = 'Ventana de verificación abierta.';
  const ENDED_KEY = 'kza:my-voice-op-ended';
  const ENDED_NOTICE = 'Una operación sobre tu inscripción por voz quedó sin confirmar al terminar la sesión anterior. Revisá el estado antes de repetirla.';
  function classifyWrite(op, outcome) {
    if (!OPS.includes(op)) throw Error('invalid_my_voice_operation');
    if (outcome && 'value' in outcome) {
      const value = outcome.value;
      if (op === 'verify') {
        try { const window = value && keys(value) === 'window' ? parseWindow(value.window) : null; if (window) return { kind: 'applied', window, text: WINDOW_OPENED }; }
        catch (_) { /* Falls through to uncertain. */ }
        return { kind: 'uncertain', text: UNCONFIRMED };
      }
      if (!value || keys(value) !== 'revision,state' || !STATES.includes(value.state) || !revision(value.revision)) return { kind: 'uncertain', text: UNCONFIRMED };
      return { kind: 'applied', state: value.state, revision: value.revision,
        text: op === 'request' ? 'Inscripción pedida. Falta verificar tu voz.' : value.state === 'verification_locked' ? 'Registramos tu baja. La verificación sigue bloqueada hasta que la administración la habilite.' : 'Te diste de baja de la memoria por voz.' };
    }
    const error = outcome?.error;
    if (error?.sent === false) return { kind: 'not_applied', text: 'No se envió: este acceso no permite la operación.' };
    if (error?.answered === true) return { kind: 'uncertain', text: 'La sesión dejó de estar vigente y no pudimos leer la respuesta.' };
    const status = error?.status, code = detailCode(error);
    if (!Number.isInteger(status)) return { kind: 'uncertain', text: 'No se pudo contactar al servidor.' };
    if (status === 401 || status === 403) return { kind: 'uncertain', text: 'La sesión no permitió confirmar el resultado.' };
    if ([409, 422, 429].includes(status)) return { kind: 'refused', code, text: REFUSALS[code] || 'El servidor rechazó el pedido. No se cambió nada.' };
    return { kind: 'uncertain', text: status >= 500 ? 'El servidor no pudo completar la operación.' : 'El servidor respondió de forma inesperada.' };
  }
  const STILL = 'Tu estado todavía no cambió; puede que el pedido no se haya aplicado. Podés volver a intentarlo.';
  const CHANGED = 'Tu estado cambió, pero no podemos confirmar que haya sido por este pedido. Revisalo.';
  /**
   * Judge a lost write from a status read that started after the loss. A marker stored by an earlier page
   * is never 'Comprobado' unless whoami's account_binding equals the one stored with it (R3).
   */
  function resolveMarker(marker, status) {
    if (marker.op === 'verify') {
      if (status.window && !marker.restored) return { result: 'applied', text: 'Comprobado: tenés una ventana de verificación abierta.' };
      if (status.revision !== marker.expected_revision || status.state !== 'pending_verification') return { result: 'final', text: CHANGED };
      return { result: 'final', text: status.window ? 'Tenés una ventana de verificación abierta. La operación quedó de una sesión anterior.' : 'No figura una ventana abierta. Podés volver a intentarlo; cada intento cuenta para el límite diario.' };
    }
    if (status.revision === marker.expected_revision) return { result: 'final', text: STILL };
    const expected = marker.op === 'request' ? ['pending_verification'] : ['revoked', 'verification_locked'];
    if (!marker.restored && status.revision === marker.expected_revision + 1 && expected.includes(status.state)) {
      return { result: 'applied', text: marker.op === 'request' ? 'Comprobado: tu inscripción quedó pedida.' : 'Comprobado: te diste de baja.' };
    }
    return { result: 'final', text: CHANGED };
  }
  function describeReadError(error) {
    const status = error?.status;
    if (error?.sent === false || status === 403) return 'Este acceso no permite consultar tu inscripción por voz.';
    if (status === 401) return 'Iniciá sesión para consultar tu inscripción por voz.';
    if (status === 503) return 'La memoria por voz no está disponible en este momento.';
    if (Number.isInteger(status)) return 'El servidor no pudo completar la consulta.';
    return 'No pudimos consultar tu inscripción por voz.';
  }
  /** What the person may do now: the flag or switch off leaves only the revocation. */
  function allowed(status) {
    if (!status) return { request: false, revoke: false, verify: false };
    const revocable = status.state !== null && status.state !== 'revoked';
    if (!status.enabled) return { request: false, revoke: revocable, verify: false };
    return { request: [null, 'revoked', 'invalidated'].includes(status.state), revoke: revocable, verify: status.state === 'pending_verification' };
  }
  const aborted = error => error?.name === 'AbortError';
  const ACCOUNT_UNKNOWN = 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.';

  function createMyVoiceOwner({ api, session, storage = root.sessionStorage, schedule = setTimeout, cancel = clearTimeout,
    wall = Date.now, onAccountChanged = () => {} }) {
    let epoch = 0, visit = 0, opened = false, timer, readSeq = 0, refreshing = null, again = false, serial = 0;
    // The account_binding the first whoami of this epoch named; null when the server names none.
    let binding = null;
    const reads = new Set(), listeners = new Set();
    let state = initial();
    function initial() {
      return { status: null, rooms: null, read: { loading: false, loaded: false, error: null, observedWall: null }, marker: null, notice: null, announcement: null };
    }
    function capabilities() { return { voice: session.can('memory:own:voice'), rooms: session.can('home:read') }; }
    function emit() { for (const listener of [...listeners]) { try { listener(); } catch (_) { /* A view must not break the owner. */ } } }
    function announce(text) { state.announcement = { serial: ++serial, text }; }
    function subject() { return typeof session.subject === 'function' ? session.subject() : null; }
    function stopTimer() { if (timer !== undefined) cancel(timer); timer = undefined; }
    async function read(path) {
      const controller = new AbortController(); reads.add(controller);
      try { return await api(path, { signal: controller.signal }); } finally { reads.delete(controller); }
    }
    // The window closes at the server's deadline: re-read then (the phrase is never kept after it).
    function plan() {
      stopTimer();
      const window = state.status?.window;
      if (!opened || !window) return;
      const ticket = epoch;
      timer = schedule(() => { timer = undefined; if (ticket === epoch && opened) void refresh(); }, Math.max(50, Math.min(60000, window.expires_at * 1000 - wall() + 50)));
    }
    function remove() { try { storage?.removeItem(PREFIX + 'consent'); } catch (_) { /* Best effort. */ } }
    function persist(marker) {
      if (!storage) throw Error('storage_unavailable');
      if (!marker.subject) throw Error('subject_unavailable');
      // Only the operation, the account (subject and opaque binding), the time and the revision: never the room or the phrase.
      storage.setItem(PREFIX + 'consent', JSON.stringify({ op: marker.op, subject: marker.subject, ...(marker.binding ? { binding: marker.binding } : {}),
        sent_at: marker.sent_at, expected_revision: marker.expected_revision }));
    }
    function dropMarker() { state.marker = null; remove(); }
    function restore() {
      let raw;
      try { raw = storage?.getItem(PREFIX + 'consent'); } catch (_) { return; }
      if (!raw) return;
      try {
        const value = JSON.parse(raw), current = subject();
        if (!value || !OPS.includes(value.op) || !string(value.subject) || !revision(value.expected_revision)) throw Error('invalid');
        if (!current || value.subject !== current) { state.notice = 'Una operación de otra cuenta quedó sin confirmar; no se reintentó.'; throw Error('foreign'); }
        if (!state.marker) {
          state.marker = { op: value.op, subject: value.subject, sent_at: finite(value.sent_at) ? value.sent_at : null, expected_revision: value.expected_revision, state: 'unresolved', after: -1, restored: true,
            binding: string(value.binding) && BINDING.test(value.binding) ? value.binding : null };
          state.notice = 'No sabemos si se aplicó la última operación. Consultando…';
        }
      } catch (_) { remove(); }
    }
    /** 'same' needs the cached subject, the home role and memory:own:voice in the server's answer. */
    async function confirmAccount() {
      const expected = subject(), seen = visit, wasOpen = opened, ticket = epoch;
      if (!expected) return 'changed';
      try {
        const identity = await read('/api/admin/auth/whoami');
        if (!(identity?.ok === true && identity.subject === expected && identity.role === 'home' && session.role() === 'home'
          && Array.isArray(identity.permissions) && identity.permissions.includes('memory:own:voice'))) return 'changed';
        // Another credential or person behind the same subject is a change too (R3).
        const named = identity.account_binding ?? null;
        if (named !== null && !(string(named) && BINDING.test(named))) return 'changed';
        if (binding !== null && named !== binding) return 'changed';
        if (ticket === epoch) binding = named;
        return 'same';
      } catch (error) {
        if (aborted(error) && (wasOpen && (!opened || seen !== visit))) return 'aborted';
        return [401, 403].includes(error?.status) ? 'changed' : 'unknown';
      }
    }
    function accountChanged() {
      clear();
      state.notice = 'La cuenta de esta pestaña cambió en otra pestaña. No se consultó ni se envió nada.';
      emit();
      try { onAccountChanged(); } catch (_) { /* The page re-reads the session on its own. */ }
    }
    async function readOnce(ticket) {
      const started = ++readSeq, seen = visit;
      const current = () => ticket === epoch && opened && seen === visit;
      state.read.loading = true; emit();
      let status = null, failure = null, malformed = false;
      try { status = parseStatus(await read('/api/memory/mine/voice')); } catch (error) { if (error?.message === 'invalid_my_voice') malformed = true; else failure = error; }
      if (!current()) return;
      state.read.loading = false;
      if (malformed || failure) {
        state.status = null; state.rooms = null; stopTimer();
        state.read.error = malformed ? 'El servidor devolvió tu inscripción por voz con un formato desconocido.' : describeReadError(failure);
        emit(); return;
      }
      state.status = status;
      state.read = { loading: false, loaded: true, error: null, observedWall: wall() };
      // A window notice outlives its window only until the next read shows it closed.
      if (state.notice === WINDOW_OPENED && !status.window) state.notice = null;
      const marker = state.marker;
      if (marker && marker.state !== 'sending' && marker.after < started) {
        const verdict = resolveMarker(marker, status);
        dropMarker(); state.notice = verdict.text; announce(verdict.text);
      }
      if (allowed(status).verify && capabilities().rooms && state.rooms === null) {
        try { const rooms = parseRooms(await read('/api/home/catalog')); if (current()) state.rooms = rooms; }
        catch (_) { if (current()) state.rooms = []; }
        if (!current()) return;
      }
      emit(); plan();
    }
    function refresh() {
      if (!opened || !capabilities().voice) return Promise.resolve();
      if (refreshing) { again = true; return refreshing; }
      const ticket = epoch;
      const running = (async () => {
        try { do { again = false; await readOnce(ticket); } while (again && ticket === epoch && opened); }
        finally { if (refreshing === running) refreshing = null; }
      })();
      refreshing = running;
      return running;
    }
    function request(op, marker) {
      if (op === 'verify') return ['/api/memory/mine/voice/verification', { method: 'POST', body: { room_id: marker.room_id, expected_revision: marker.expected_revision } }];
      return ['/api/memory/mine/voice/consent', { method: op === 'request' ? 'POST' : 'DELETE', body: { expected_revision: marker.expected_revision } }];
    }
    async function send(marker) {
      const ticket = epoch, [path, options] = request(marker.op, marker);
      delete marker.room_id;  // Only the request carried it.
      let outcome;
      // Never aborted: leaving the view must not cut a write the server may commit.
      try { outcome = { value: await api(path, options) }; } catch (error) { outcome = { error }; }
      if (ticket !== epoch || state.marker !== marker) return { stale: true };
      const verdict = classifyWrite(marker.op, outcome);
      if (verdict.kind === 'applied' || verdict.kind === 'refused' || verdict.kind === 'not_applied') {
        dropMarker(); state.notice = verdict.text; announce(verdict.text);
        if (verdict.kind === 'applied' && marker.op === 'verify' && state.status) state.status = { ...state.status, window: verdict.window };
        else if (verdict.kind === 'applied' && state.status) state.status = { ...state.status, state: verdict.state, revision: verdict.revision, window: null };
        emit(); plan();
        if (verdict.kind !== 'not_applied') { again = again || Boolean(refreshing); void refresh(); }
        return { applied: verdict.kind === 'applied', refused: verdict.kind === 'refused', code: verdict.code || null };
      }
      marker.state = 'unresolved'; marker.after = readSeq;
      try { persist(marker); } catch (_) { /* Stored before sending. */ }
      const text = 'Sin confirmar. ' + verdict.text + ' Consultando…';
      state.notice = text; announce(text); emit();
      again = again || Boolean(refreshing); void refresh();
      return { uncertain: true };
    }
    /** Every write is confirmed by the person (a dialog or an explicit form button) and by a whoami. */
    async function act(op, { confirmed = false, roomId = null } = {}) {
      const status = state.status;
      if (!opened || !OPS.includes(op) || !capabilities().voice || !confirmed || !status || state.marker?.state === 'sending' || !allowed(status)[op]) return { sent: false };
      if (op === 'verify' && (!string(roomId) || !ROOM.test(roomId) || !(state.rooms || []).some(room => room.room_id === roomId))) return { sent: false };
      const ticket = epoch, expected = status.revision, account = await confirmAccount();
      if (ticket !== epoch) return { sent: false };
      if (account === 'changed') { accountChanged(); return { sent: false, accountChanged: true }; }
      if (account !== 'same') { state.notice = ACCOUNT_UNKNOWN; announce(ACCOUNT_UNKNOWN); emit(); return { sent: false }; }
      if (state.marker?.state === 'sending' || !opened) return { sent: false };
      if (state.status?.revision !== expected || !allowed(state.status)[op]) {
        state.notice = REFUSALS.memory_voice_conflict; announce(state.notice); emit(); return { sent: false };
      }
      const marker = { op, subject: subject(), binding, sent_at: wall(), expected_revision: expected, state: 'sending', after: Infinity, room_id: roomId };
      try { persist(marker); } catch (error) {
        const text = error?.message === 'subject_unavailable' ? 'No se pudo identificar la cuenta de esta sesión; no se envió.' : 'El navegador no permite guardar el estado de la operación; no se envió.';
        state.notice = text; announce(text); emit(); return { sent: false };
      }
      state.marker = marker; state.notice = null; emit();
      return send(marker);
    }
    /**
     * After whoami confirmed the account: a restored marker with this binding was sent by this same
     * grant and person and resolves like this page's own; one with another binding belongs to another
     * account and is dropped unsent. A marker without a binding stays restored.
     */
    function adoptRestored() {
      const marker = state.marker;
      if (binding === null || !marker?.restored || marker.binding === null || marker.binding === undefined) return;
      if (marker.binding === binding) { marker.restored = false; return; }
      dropMarker(); state.notice = 'Una operación de otra cuenta quedó sin confirmar; no se reintentó.';
    }
    async function open() {
      if (!capabilities().voice) return true;
      opened = true; visit++; restore();
      try { if (storage?.getItem(ENDED_KEY)) { storage.removeItem(ENDED_KEY); if (!state.marker) state.notice = ENDED_NOTICE; } }
      catch (_) { /* Reminder is best effort. */ }
      emit();
      const ticket = epoch, seen = visit, account = await confirmAccount();
      if (ticket !== epoch || !opened || seen !== visit || account === 'aborted') return true;
      if (account === 'changed') { accountChanged(); return true; }
      if (account === 'unknown') { state.read.error = ACCOUNT_UNKNOWN; emit(); return true; }
      adoptRestored(); emit();
      void refresh();
      return true;
    }
    function leave() {
      opened = false; again = false; stopTimer();
      for (const controller of reads) controller.abort();
      reads.clear(); refreshing = null;
      // The phrase and the rooms live only while the view is open.
      state.status = null; state.rooms = null;
      // An outcome notice belongs to this visit; an unresolved marker keeps its own.
      if (!state.marker) state.notice = null;
      state.read = { loading: false, loaded: false, error: null, observedWall: state.read.observedWall };
      return Promise.resolve(true);
    }
    /** wipe=false (pagehide, a failed bootstrap) keeps the stored marker for the next load; only the end of the session removes it. */
    function clear({ wipe = true } = {}) {
      const pending = Boolean(state.marker);
      let stored = false;
      try { stored = Boolean(storage?.getItem(PREFIX + 'consent')); } catch (_) { /* Nothing to wipe. */ }
      epoch++; binding = null; leave(); state = initial();
      if (wipe) {
        remove();
        if (pending || stored) { try { storage?.setItem(ENDED_KEY, '1'); } catch (_) { /* Best effort. */ } }
      }
      emit();
      return Promise.resolve(true);
    }
    function getState() {
      const status = state.status;
      return { status: status ? { ...status, window: status.window ? { ...status.window } : null } : null, rooms: state.rooms ? state.rooms.map(room => ({ ...room })) : null,
        read: { ...state.read }, marker: state.marker ? { op: state.marker.op, state: state.marker.state, restored: state.marker.restored === true } : null,
        notice: state.notice, announcement: state.announcement, capabilities: capabilities(), allowed: allowed(status),
        message: status && status.enabled ? STATE_MESSAGES[status.state] || null : null };
    }
    function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
    return { open, leave, clear, refresh, subscribe, getState, capabilities, isOpen: () => opened,
      requestConsent: options => act('request', options), revoke: options => act('revoke', options), openWindow: (roomId, options = {}) => act('verify', { ...options, roomId }) };
  }

  const exported = { createMyVoiceOwner, parseStatus, parseWindow, parseRooms, classifyWrite, resolveMarker, allowed, describeReadError,
    PREFIX, ENDED_KEY, OPT_IN_DISCLOSURE, REENROLMENT_NOTICE, INSUFFICIENT_PROFILES_MESSAGE, STATE_MESSAGES, STATE_LABELS, NOT_ENROLLED, OFF, OFF_REVOCABLE, REFUSALS };
  root.NexoMyVoiceOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
