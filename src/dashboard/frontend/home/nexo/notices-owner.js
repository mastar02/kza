/* The shared Home notice channel. No notice or receipt survives its session. */
(function (root) {
  'use strict';
  const isNode = typeof module !== 'undefined' && module.exports;
  const core = isNode ? require('../home.js') : root.HomeUI;
  const { NOTICE_LIST, NOTICE_DISMISS, canReadNotices } = isNode ? require('./state.js') : root.NexoState;
  const SOURCE = 'reminder-store';
  const LABELS = {
    pending: { pendiente: 'Pendiente', 'no entregado': 'No entregado' },
    sending: { anunciando: 'Anunciando' },
    uncertain: { cortado: 'Entrega incierta · Cortado', incompleto: 'Entrega incierta · Incompleto', incierto: 'Entrega incierta' }
  };

  async function decodeNotices(response) {
    const data = await response.json();
    const state = response.headers.get('X-KZA-Observation-State');
    const observed_at = response.headers.get('X-KZA-Observed-At');
    const expires_at = response.headers.get('X-KZA-Expires-At');
    const observed = core.observationMillis(observed_at), expires = core.observationMillis(expires_at);
    if (data?.source !== SOURCE || response.headers.get('X-KZA-Source-Id') !== SOURCE ||
        !['fresh', 'stale'].includes(state) || observed === null || expires === null || expires <= observed ||
        core.observationMillis(data.observed_at) !== observed || !Array.isArray(data.notices)) {
      throw new Error('invalid_notice_observation');
    }
    const notices = new Map();
    for (const entry of data.notices) {
      if (entry?.status === 'confirmed') continue; // The backend keeps finished deliveries in its history.
      const labels = Object.hasOwn(LABELS, entry?.status) && LABELS[entry.status];
      if (!labels || !Object.hasOwn(labels, entry.label) || typeof entry.reminder_id !== 'string' ||
          entry.reminder_id.length < 1 || entry.reminder_id.length > 64 ||
          !Number.isFinite(entry.trigger_at) || !Number.isFinite(new Date(entry.trigger_at * 1000).getTime()) ||
          !(entry.text === null || typeof entry.text === 'string')) throw new Error('invalid_notice_entry');
      const { reminder_id, trigger_at, text, status } = entry;
      const key = JSON.stringify([reminder_id, trigger_at]);
      if (notices.has(key)) throw new Error('duplicate_notice_occurrence');
      notices.set(key, { key, reminder_id, trigger_at, text, status, label: labels[entry.label] });
    }
    return { notices, observation: { state, observed_at, expires_at } };
  }

  function readError(error) {
    if (error?.status === 401) return 'Iniciá sesión para consultar los avisos.';
    if (error?.status === 403) return 'Esta sesión no tiene permiso para consultar los avisos.';
    if (error?.status === 503) return 'El servicio de avisos no está disponible.';
    if (error?.name === 'AbortError' || error instanceof TypeError) return 'No se pudo contactar al servicio de avisos.';
    return 'No se pudo verificar la lista de avisos.';
  }

  function dismissError(error) {
    if (error?.status === 401) return { kind: 'error', text: 'Iniciá sesión para descartar avisos.' };
    if (error?.status === 403) return { kind: 'error', text: 'Esta sesión no tiene permiso para descartar avisos.' };
    if (error?.status === 409) return { kind: 'error', text: 'El aviso se está anunciando. Volvé a consultar antes de descartarlo.' };
    if (error?.status === 404) return { kind: 'error', text: 'El aviso ya no está disponible para descartar.' };
    if (error?.status === 422) return { kind: 'error', text: 'El servicio rechazó el descarte del aviso.' };
    if (error?.status === 503 && error.detail === 'reminder_notices_unavailable') {
      return { kind: 'error', text: 'El servicio de avisos no está disponible.' };
    }
    return { kind: 'uncertain', text: 'No se pudo confirmar el descarte. Consultá los avisos antes de volver a intentarlo.' };
  }

  function createNoticesOwner({ api, session, onAccountChanged = () => {}, clockOptions, schedule, cancel, now }) {
    const listeners = new Set();
    let opened = false, version = 0, generation = -1;
    let snapshot = null, clock = null, error = null, needsRefresh = false, mayWrite = false;
    let writing = null, feedback = null;
    const currentAccount = () => generation === session.generation() && canReadNotices(session) && !!session.subject();
    const currentView = () => opened && currentAccount();
    const stale = () => !clock || clock.read().stale || needsRefresh || !!error;
    const emit = () => { for (const listener of listeners) listener(); };

    function getState() {
      if (!currentView()) return { visible: false };
      const notices = snapshot ? [...snapshot.values()] : [];
      return {
        visible: true, notices, error, feedback,
        status: snapshot ? stale() ? 'stale' : notices.length ? 'ready' : 'empty' : error ? 'error' : 'loading',
        canWrite: mayWrite && session.can('tasks:write'), locked: stale() || writing !== null
      };
    }

    function leave() {
      opened = false; version++;
      if (writing?.preflight) { writing.preflight.abort(); writing = null; }
      poller.stop(); ticker.stop();
      snapshot = clock = error = feedback = null;
      needsRefresh = mayWrite = false;
      emit();
    }

    function clear() {
      generation = -1;
      leave();
      writing = null;
    }

    function forgetObservation() {
      snapshot = clock = feedback = null;
      mayWrite = false;
    }

    async function verify(signal, stillCurrent) {
      const identity = await api('/api/admin/auth/whoami', { signal });
      if (!stillCurrent()) return false;
      if (identity?.ok !== true || identity.subject !== session.subject() || identity.role !== session.role() ||
          (identity.account_binding ?? null) !== session.accountBinding()) {
        const previousGeneration = generation;
        clear(); onAccountChanged(previousGeneration);
        return false;
      }
      if (!Array.isArray(identity.permissions) || !['home:read', 'tasks:read'].every(p => identity.permissions.includes(p))) {
        const denied = Object.assign(new Error('notice_read_denied'), { status: 403 });
        forgetObservation(); error = readError(denied);
        throw denied;
      }
      return identity;
    }

    const poller = core.createPoller({
      schedule, cancel, now,
      deadline: () => clock?.read().remainingMs ?? Infinity,
      async load(signal) {
        const started = version;
        const current = () => !signal.aborted && started === version && currentView();
        if (!current() || !await verify(signal, current) || !current()) return null;
        const data = await api(NOTICE_LIST, { signal, decode: decodeNotices });
        if (!current()) return null;
        const identity = await verify(signal, current);
        if (!identity || !current()) return null;
        return { ...data, mayWrite: identity.permissions.includes('tasks:write') };
      },
      onData(data) {
        if (!data || !currentView()) return;
        snapshot = data.notices; clock = core.createSnapshotClock(data, clockOptions);
        error = null; needsRefresh = false; mayWrite = data.mayWrite;
        ticker.refresh();
      },
      onError(failure) {
        if (!currentView()) return;
        if ([401, 403].includes(failure?.status)) forgetObservation();
        error = readError(failure);
        emit();
      }
    });
    const ticker = core.createFreshnessTicker({
      schedule, cancel, read: () => clock?.read() || {}, onTick: emit
    });

    function open() {
      if (!canReadNotices(session) || !session.subject()) { clear(); return false; }
      if (generation !== session.generation()) clear();
      if (!opened) {
        generation = session.generation(); opened = true;
        ticker.start(); poller.start();
      }
      return true;
    }

    function refresh() { if (currentView()) return poller.refresh(); }

    async function dismiss(key) {
      const entry = snapshot?.get(key);
      if (!currentView() || stale() || !mayWrite || !session.can('tasks:write') ||
          writing || !entry || entry.status === 'sending') return false;
      const started = version, operation = { preflight: new AbortController() };
      const current = () => started === version && currentView() && writing === operation;
      writing = operation; feedback = null; emit();
      let sent = false;
      try {
        const identity = await verify(operation.preflight.signal, current);
        if (!identity || !current()) return false;
        mayWrite = identity.permissions.includes('tasks:write');
        if (!mayWrite || !session.can('tasks:write')) {
          feedback = dismissError({ status: 403 }); return false;
        }
        if (stale() || snapshot.get(key)?.status !== entry.status) {
          feedback = { kind: 'error', text: 'La lista cambió o perdió vigencia. Consultá el aviso actualizado.' };
          return false;
        }
        sent = true; operation.preflight = null;
        const result = await api(NOTICE_DISMISS, { method: 'POST',
          body: { reminder_id: entry.reminder_id, trigger_at: entry.trigger_at } });
        if (!currentAccount() || writing !== operation) return false;
        const outcome = result?.outcome;
        if (!['dismissed', 'purged'].includes(result?.status) || outcome?.status !== 'completed' ||
            outcome.persistence !== 'committed' || outcome.effect !== 'observed' ||
            outcome.observation_scope !== 'software_state') throw new Error('invalid_notice_receipt');
        snapshot?.delete(key);
        feedback = { kind: 'success', text: 'Aviso descartado.' };
        return true;
      } catch (failure) {
        if (currentAccount() && writing === operation && (sent || current())) {
          if (failure?.status === 403) mayWrite = false;
          feedback = sent || [401, 403].includes(failure?.status)
            ? dismissError(failure) : { kind: 'error', text: 'No se pudo verificar la sesión. El descarte no se envió.' };
        }
        return false;
      } finally {
        if (writing === operation) {
          writing = null;
          if (sent && currentView()) {
            // A list read started before the POST can still contain the dismissed occurrence.
            needsRefresh = true; poller.stop(); poller.start();
          }
          emit();
        }
      }
    }

    return { open, leave, clear, refresh, dismiss, getState,
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); } };
  }

  const exported = { createNoticesOwner };
  root.NexoNoticesOwner = exported;
  if (isNode) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
