/* NEXO evaluator owner with scoped payload and native media retirement. */
(function (root) {
  'use strict';
  const liveAPI = root.KzaLivePrivacy || require('../../obs/src/live-privacy.js');
  const channelAPI = root.KzaConversationChannel || require('../../obs/src/conversation-channel.js');
  const voiceAPI = root.KzaVoicePrivacy || require('../../obs/src/voice-privacy.js');
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const VF = /^vf_[0-9a-f]{64}$/;
  const MODES = ['audio_preview', 'transcript_full'];
  const CHANNEL = 'evaluation:sample';
  const CATALOG = '/api/evaluation/grants';
  const SAMPLE = '/api/evaluation/samples/';
  const keys = (value, names) => !!value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).sort().join(',') === [...names].sort().join(',');
  const finiteTime = value => Number.isFinite(value) && value >= 0 && Number.isFinite(value * 1000);
  function clocks(monotonic, wall) {
    let lastMono = -Infinity, lastWall = -Infinity;
    return {
      mono() { const value = monotonic(); if (!Number.isFinite(value)) throw Error('invalid evaluator clock'); return lastMono = Math.max(lastMono, value); },
      wall() { const value = wall(); if (!Number.isFinite(value)) throw Error('invalid evaluator clock'); return lastWall = Math.max(lastWall, value); },
    };
  }
  function permittedPath(path, method = 'GET') {
    if (method !== 'GET' || typeof path !== 'string') return false;
    return path === CATALOG || /^\/api\/evaluation\/samples\/vf_[0-9a-f]{64}\?modality=(audio_preview|transcript_full)$/.test(path);
  }
  function parseCatalog(value, started, clock, remaining) {
    if (!keys(value, ['observed_at', 'expires_at', 'grants']) || !finiteTime(value.observed_at)
        || !finiteTime(value.expires_at) || value.expires_at <= value.observed_at
        || !Array.isArray(value.grants) || value.grants.length > 64) throw Error('invalid evaluation catalogue');
    const seen = new Set();
    const grants = value.grants.map(row => {
      if (!keys(row, ['fingerprint_id', 'modalities', 'expires_at']) || !VF.test(row.fingerprint_id)
          || seen.has(row.fingerprint_id) || !finiteTime(row.expires_at) || row.expires_at <= value.observed_at
          || !Array.isArray(row.modalities) || !row.modalities.length || row.modalities.length > 2
          || !row.modalities.every(mode => MODES.includes(mode)) || new Set(row.modalities).size !== row.modalities.length) throw Error('invalid evaluation choice');
      seen.add(row.fingerprint_id);
      return Object.freeze({ fingerprint_id: row.fingerprint_id, modalities: Object.freeze([...row.modalities].sort()), expires_at: row.expires_at });
    });
    if (grants.length && value.expires_at !== Math.min(...grants.map(row => row.expires_at))) throw Error('invalid catalogue deadline');
    const nowMono = clock.mono(), nowWall = clock.wall();
    const elapsed = Math.max(0, nowMono - started.mono, nowWall - started.wall);
    const budget = Math.min((value.expires_at - value.observed_at) * 1000 - elapsed,
      value.expires_at * 1000 - nowWall, remaining());
    if (!Number.isFinite(budget) || budget <= 0) throw Error('expired evaluation catalogue');
    return Object.freeze({ grants: Object.freeze(grants), observed_at: value.observed_at,
      expires_at: value.expires_at, mono: nowMono + budget, wall: nowWall + budget });
  }
  function validText(value) {
    if (typeof value !== 'string' || value.length > 16384) return false;
    let count = 0;
    for (const character of value) { const code = character.codePointAt(0); if (++count > 8192 || code >= 0xd800 && code <= 0xdfff) return false; }
    return new TextEncoder().encode(value).length <= 32768;
  }
  function audioBytes(row) {
    if (row.audio_mime !== 'audio/wav' || !Number.isFinite(row.clip_duration_s)
        || row.clip_duration_s < .8 || row.clip_duration_s > 6 || typeof row.audio_base64 !== 'string'
        || row.audio_base64.length > 256060 || !/^[A-Za-z0-9+/]+={0,2}$/.test(row.audio_base64)) throw Error('invalid evaluation audio');
    const raw = atob(row.audio_base64), bytes = Uint8Array.from(raw, character => character.charCodeAt(0));
    const view = new DataView(bytes.buffer), text = (offset, count) => String.fromCharCode(...bytes.slice(offset, offset + count));
    if (bytes.length < 45 || bytes.length > 192044 || text(0, 4) !== 'RIFF' || view.getUint32(4, true) !== bytes.length - 8 || text(8, 4) !== 'WAVE'
        || text(12, 4) !== 'fmt ' || view.getUint32(16, true) !== 16 || view.getUint16(20, true) !== 1
        || view.getUint16(22, true) !== 1 || view.getUint32(24, true) !== 16000
        || view.getUint32(28, true) !== 32000 || view.getUint16(32, true) !== 2 || view.getUint16(34, true) !== 16
        || text(36, 4) !== 'data' || view.getUint32(40, true) !== bytes.length - 44
        || bytes.length - 44 !== Math.round(row.clip_duration_s * 16000) * 2) throw Error('invalid evaluation WAV');
    return bytes;
  }
  function validateRow(row, choice, augmented = false) {
    const fields = choice.modality === 'audio_preview'
      ? ['id', 'modality', 'audio_mime', 'clip_duration_s', 'audio_base64']
      : ['id', 'modality', 'transcript_scope', 'transcript'];
    if (augmented) fields.push('observed_at', 'expires_at');
    if (!keys(row, fields) || row.id !== choice.fingerprint_id || row.modality !== choice.modality) throw Error('invalid evaluation row');
    if (choice.modality === 'transcript_full' && (row.transcript_scope !== 'full_segment' || !validText(row.transcript))) throw Error('invalid evaluation text');
  }
  function validateEnvelope(document, choice) {
    if (!keys(document, ['connection_id', 'snapshot', 'channel']) || !UUID.test(document.connection_id)
        || document.channel !== CHANNEL) throw Error('invalid evaluation envelope');
    const batch = document.snapshot;
    if (batch === null) return document;
    if (!keys(batch, ['event_id', 'expires_at', 'observed_at', 'receipt', 'channel', 'rows'])
        || !UUID.test(batch.event_id) || batch.channel !== CHANNEL || !finiteTime(batch.observed_at)
        || !finiteTime(batch.expires_at) || batch.expires_at <= batch.observed_at || batch.expires_at > choice.expires_at
        || !keys(batch.receipt, ['ticket', 'nonce']) || !UUID.test(batch.receipt.ticket) || !UUID.test(batch.receipt.nonce)
        || !Array.isArray(batch.rows) || batch.rows.length !== 1) throw Error('invalid evaluation snapshot');
    validateRow(batch.rows[0], choice);
    return document;
  }
  function createEvaluationStore({ choice, clock, schedule = setTimeout, cancel = clearTimeout,
      url = root.URL, createMediaStore = voiceAPI.createVoiceStore, onExpired = () => {} }) {
    // Reuse the real native-player retirement contract; no audio timeout can authorize ACK.
    const media = createMediaStore('profiles', { monotonic: clock.mono, wall: clock.wall, schedule, cancel, url });
    let entry = null, timer = null, privacy = null, status = 'waiting';
    const listeners = new Set();
    function notify() { for (const listener of listeners) { try { listener(); } catch (_) { /* A renderer cannot skip retirement. */ } } }
    function clear() {
      media.clear();
      if (entry?.row.audio_url) url.revokeObjectURL(entry.row.audio_url);
      entry = null;
      if (timer !== null) cancel(timer); timer = null; notify();
    }
    function check() {
      if (entry && (clock.mono() >= entry.mono || clock.wall() >= entry.wall)) {
        clear(); status = 'expired'; notify(); onExpired();
      }
    }
    function arm() {
      if (timer !== null) cancel(timer); timer = null;
      if (!entry) return;
      const delay = Math.min(entry.mono - clock.mono(), entry.wall - clock.wall());
      timer = schedule(() => { timer = null; check(); arm(); }, Math.max(1, Math.min(delay, 2147483647)));
    }
    return {
      receive(rows, started) {
        let objectURL = null;
        try {
          const selected = choice();
          if (!selected || !Array.isArray(rows) || rows.length !== 1) throw Error('missing evaluation choice');
          const row = rows[0]; validateRow(row, selected, true);
          if (!finiteTime(row.observed_at) || !finiteTime(row.expires_at) || row.expires_at <= row.observed_at
              || row.expires_at > selected.expires_at) throw Error('invalid evaluation lifetime');
          const nowMono = clock.mono(), nowWall = clock.wall();
          const elapsed = Math.max(0, nowMono - started.mono, nowWall - started.wall);
          const budget = Math.min((row.expires_at - row.observed_at) * 1000 - elapsed, row.expires_at * 1000 - nowWall);
          if (!Number.isFinite(budget) || budget <= 0) throw Error('expired evaluation row');
          let projected;
          if (selected.modality === 'audio_preview') {
            objectURL = url.createObjectURL(new Blob([audioBytes(row)], { type: 'audio/wav' }));
            projected = { id: row.id, modality: row.modality, expires_at: row.expires_at, audio_url: objectURL, clip_duration_s: row.clip_duration_s };
          } else projected = { id: row.id, modality: row.modality, expires_at: row.expires_at, transcript: row.transcript };
          clear(); entry = { row: Object.freeze(projected), mono: nowMono + budget, wall: nowWall + budget }; objectURL = null;
          status = 'real'; arm(); notify(); return true;
        } catch (_) {
          if (objectURL) url.revokeObjectURL(objectURL);
          clear(); status = 'unavailable'; notify(); return false;
        }
      },
      clear, check, snapshot() { check(); return entry ? [entry.row] : []; }, status: () => status,
      setStatus(value) { status = value; notify(); },
      attachMedia: element => media.attachMedia(element), detachMedia: () => media.detachMedia(),
      pendingRetirements: () => media.pendingRetirements(), retired: () => media.retired(),
      retirementStatus: () => media.retirementStatus(),
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      bindPrivacy(value) { privacy = value; }, committed(token) { privacy?.committed(token); },
      deliveryToken: () => privacy?.token() ?? null,
    };
  }
  function createEvaluationOwner({ session, mount, fetch: rawFetch, connect, socketUrl,
      onStatus = () => {}, onUnauthorized = () => {}, verifyIdentity = async () => null,
      monotonic = () => root.performance.now(), wall = Date.now, schedule = setTimeout, cancel = clearTimeout,
      retryMs = 1000, url = root.URL, createLive = liveAPI.createLiveState,
      createChannel = channelAPI.createConversationChannel, createMediaStore = voiceAPI.createVoiceStore,
      pageId = liveAPI.pageId }) {
    let wanted = false, generation = null, current = null, closing = null, blocked = null, retry = null;
    const allowed = () => session.role() === 'evaluator' && session.can('evaluation:read');
    const owns = context => current === context && !context.sealed && wanted && allowed()
      && generation === context.generation && session.generation() === context.generation;
    const ready = context => owns(context) && context.phase === 'ready';
    function report(context, message) { if (owns(context)) onStatus(message); }
    function notify(context) { for (const listener of context.listeners) { try { listener(); } catch (_) { /* Keep ownership independent of rendering. */ } } }
    function track(context, work) {
      const task = Promise.resolve().then(work); context.checks.add(task);
      const finish = () => { context.checks.delete(task); context.channel?.flush(); notify(context); };
      task.then(finish, finish); return task;
    }
    function clearSelection(context, status = 'waiting') {
      context.selected = null; context.channel?.clearView(); context.store?.setStatus(status); notify(context);
    }
    function clearCatalog(context, status = 'waiting') {
      if (context.catalogTimer !== null) cancel(context.catalogTimer); context.catalogTimer = null;
      context.catalog = null; context.catalogStatus = status; clearSelection(context, status); notify(context);
    }
    function catalogueCurrent(context) {
      const value = context.catalog;
      if (value && (context.clock.mono() >= value.mono || context.clock.wall() >= value.wall)) clearCatalog(context, 'expired');
      return ready(context) && context.catalogStatus === 'real' && !!context.catalog;
    }
    function verifyOriginal(context, refresh = false) {
      if (!ready(context) || context.verifying) return;
      context.verifying = true;
      track(context, async () => {
        try {
          const valid = await verifyIdentity({ generation: context.generation });
          if (!ready(context)) return;
          if (valid === false) onUnauthorized();
          else if (valid === true) {
            report(context, 'La concesión cambió o la muestra dejó de estar disponible.');
            if (refresh) await loadCatalog(context);
          } else report(context, 'No se pudo verificar la sesión. Actualizá las concesiones.');
        } catch (_) { report(context, 'No se pudo verificar la sesión. Actualizá las concesiones.'); }
        finally { context.verifying = false; }
      });
    }
    function loadCatalog(context) {
      if (!ready(context) || context.catalogBusy) return Promise.resolve(false);
      const prior = context.selected, version = ++context.catalogVersion;
      context.catalogAbort?.abort(); const abort = new AbortController(); context.catalogAbort = abort;
      clearCatalog(context, 'loading'); context.catalogBusy = true; notify(context);
      const started = { mono: context.clock.mono(), wall: context.clock.wall() };
      return track(context, async () => {
        let response = null, document = null;
        try {
          response = await rawFetch(CATALOG, { method: 'GET', credentials: 'same-origin', cache: 'no-store', signal: abort.signal });
          if (!ready(context) || abort.signal.aborted || version !== context.catalogVersion) { await response.body?.cancel(); return false; }
          if (!response.ok) {
            await response.body?.cancel();
            if (response.status === 401 || response.status === 403) verifyOriginal(context);
            throw Error('evaluation catalogue unavailable');
          }
          document = await response.json();
          await context.store.retired();
          if (!ready(context) || abort.signal.aborted || version !== context.catalogVersion) return false;
          const catalog = parseCatalog(document, started, context.clock, () => session.remaining());
          context.catalog = catalog; context.catalogStatus = 'real';
          const selected = catalog.grants.find(row => row.fingerprint_id === prior?.fingerprint_id && row.modalities.includes(prior.modality))
            || catalog.grants[0];
          context.selected = selected ? Object.freeze({ fingerprint_id: selected.fingerprint_id,
            modality: selected === catalog.grants[0] && !selected.modalities.includes(prior?.modality) ? selected.modalities[0] : prior?.modality || selected.modalities[0],
            expires_at: selected.expires_at }) : null;
          context.store.setStatus('waiting');
          context.catalogTimer = schedule(() => {
            context.catalogTimer = null;
            if (!ready(context) || context.catalog !== catalog) return;
            clearCatalog(context, 'expired'); void loadCatalog(context);
          }, Math.max(1, Math.min(catalog.mono - context.clock.mono(), catalog.wall - context.clock.wall(), 2147483647)));
          report(context, catalog.grants.length ? 'Concesiones vigentes. Elegí una muestra y una modalidad.' : 'No hay concesiones vigentes para esta sesión.');
          notify(context); return true;
        } catch (_) {
          if (ready(context) && version === context.catalogVersion) { clearCatalog(context, 'unavailable'); report(context, 'No se pudieron consultar las concesiones.'); }
          return false;
        } finally {
          response = document = null;
          if (version === context.catalogVersion) { context.catalogBusy = false; context.catalogAbort = null; notify(context); }
        }
      });
    }
    function unmount(context) {
      const stop = context.unmount; context.unmount = null;
      try { if (stop) stop(); } catch (error) { context.cleanupFailed = true; throw error; }
    }
    function seal(context) {
      if (context.sealed) return;
      context.sealed = true; context.phase = 'closed'; context.catalogVersion++;
      const attempt = operation => { try { operation(); } catch (_) { context.cleanupFailed = true; } };
      attempt(() => context.catalogAbort?.abort());
      if (context.catalogTimer !== null) attempt(() => cancel(context.catalogTimer));
      context.catalogTimer = null; context.catalog = context.selected = null;
      attempt(() => unmount(context)); attempt(() => context.channel?.clearView());
      attempt(() => context.store?.clear()); attempt(() => context.socket?.close());
    }
    function drain(context) {
      if (closing) return closing;
      seal(context); if (current === context) current = null;
      closing = (async () => {
        const attempt = async operation => { try { await operation(); } catch (_) { context.cleanupFailed = true; } };
        await attempt(() => context.frames);
        await attempt(async () => { if (context.channel && !await context.channel.reset()) throw Error('evaluation retirement incomplete'); });
        await attempt(() => Promise.allSettled([...context.checks]));
        await attempt(() => context.store?.retired());
        await attempt(() => context.channel?.dispose()); await attempt(() => context.live?.dispose());
        await attempt(() => context.store?.retired());
        if (context.cleanupFailed) { blocked = context; return false; }
        return true;
      })();
      const pending = closing; pending.then(() => { if (closing === pending) closing = null; }); return pending;
    }
    function retryAfter(context) {
      if (!owns(context)) return;
      const original = context.generation; report(context, 'Se perdió la conexión. Retirando la muestra…');
      drain(context).then(done => {
        if (!done || !wanted || !allowed() || generation !== original || session.generation() !== original) return;
        const timer = { handle: null }; retry = timer;
        timer.handle = schedule(() => { if (retry !== timer) return; retry = null; void start(original); }, retryMs);
      });
    }
    async function start(expected = generation) {
      if (closing && !await closing) return false;
      if (blocked || !wanted || !allowed() || generation !== expected || session.generation() !== expected) return false;
      if (current) return true;
      const context = { generation: expected, sealed: false, phase: 'hello', frames: Promise.resolve(),
        checks: new Set(), listeners: new Set(), unmount: null, catalog: null, selected: null,
        catalogStatus: 'waiting', catalogBusy: false, catalogVersion: 0, catalogTimer: null,
        catalogAbort: null, verifying: false, clock: clocks(monotonic, wall) };
      current = context;
      try {
        context.store = createEvaluationStore({ choice: () => context.selected, clock: context.clock, schedule, cancel, url, createMediaStore,
          onExpired() {
            if (!ready(context)) return;
            clearSelection(context, 'expired'); report(context, 'La consulta venció. Actualizando las concesiones…'); void loadCatalog(context);
          } });
        function emitAck(message) {
          if (!owns(context) || context.socket?.readyState !== 1 || context.checks.size || context.store.pendingRetirements()) return false;
          if (message.type === 'live_reset_ack' ? context.phase !== 'waiting_ready' : context.phase !== 'ready') return false;
          context.socket.send(JSON.stringify(message));
          if (message.type === 'live_reset_ack') context.resetAcknowledged = true;
          return true;
        }
        async function boundFetch(path, options = {}) {
          const selected = context.selected;
          if (!ready(context) || !catalogueCurrent(context) || !selected || !permittedPath(path, options.method || 'GET')
              || path !== `${SAMPLE}${selected.fingerprint_id}?modality=${selected.modality}`) throw Error('evaluation reader retired');
          let response = await rawFetch(path, { ...options, method: 'GET', credentials: 'same-origin', cache: 'no-store' });
          if (!ready(context) || context.selected !== selected) { await response.body?.cancel(); throw Error('evaluation reader retired'); }
          if (!response.ok) await response.body?.cancel();
          const ok = response.ok, status = response.status;
          return { ok, status, async json() {
            let document = null;
            try {
              document = await response.json();
              if (!ready(context) || !catalogueCurrent(context) || context.selected !== selected) throw Error('evaluation reader retired');
              return validateEnvelope(document, selected);
            } finally { document = response = null; }
          } };
        }
        context.live = createLive({ onChange() {}, emitAck, now: context.clock.wall, schedule, cancel });
        context.channel = createChannel({ store: context.store, fetch: boundFetch, emitAck,
          hasView: () => ready(context) && !!context.unmount, path: SAMPLE, channel: CHANNEL, loadOnOpen: false,
          monotonic: context.clock.mono, wall: context.clock.wall,
          onUnauthorized: () => verifyOriginal(context, true), onState: value => context.store.setStatus(value) });
        context.controller = {
          store: context.store,
          snapshot() { catalogueCurrent(context); return { status: context.catalogStatus, grants: context.catalog?.grants || [],
            selected: context.selected, ready: ready(context), busy: context.catalogBusy || context.checks.size > 0 || context.store.status() === 'loading' }; },
          subscribe(listener) { context.listeners.add(listener); return () => context.listeners.delete(listener); },
          refresh: () => loadCatalog(context),
          choose(fingerprint, modality) {
            if (!catalogueCurrent(context)) return Promise.resolve(false);
            const grant = context.catalog.grants.find(row => row.fingerprint_id === fingerprint && row.modalities.includes(modality));
            if (!grant) return Promise.resolve(false);
            clearSelection(context);
            context.selected = Object.freeze({ fingerprint_id: fingerprint, modality, expires_at: grant.expires_at }); notify(context);
            return track(context, async () => { await context.store.retired(); return ready(context) && context.selected?.fingerprint_id === fingerprint && context.selected.modality === modality; });
          },
          load() {
            if (!catalogueCurrent(context) || !context.selected || context.checks.size) return false;
            return context.channel.refresh(`${SAMPLE}${context.selected.fingerprint_id}?modality=${context.selected.modality}`);
          },
          clearSample() { context.channel.clearView(); context.store.setStatus('waiting'); },
        };
        context.unmount = mount(context.controller);
        if (typeof context.unmount !== 'function') throw Error('missing evaluator view cleanup');
        context.socket = connect(socketUrl());
        context.socket.addEventListener('open', () => {
          if (!owns(context)) return;
          context.socket.send(JSON.stringify({ type: 'live_hello', version: 3, client_id: pageId() }));
        });
        context.socket.addEventListener('message', event => {
          if (!owns(context) || typeof event.data !== 'string' || event.data.length > 16384) return;
          let frame; try { frame = JSON.parse(event.data); } catch (_) { return; }
          context.frames = context.frames.then(async () => {
            if (!owns(context)) return;
            if (keys(frame, ['type', 'payload', 'ts']) && frame.type === 'live_reset' && frame.ts === ''
                && keys(frame.payload, ['id', 'nonce']) && UUID.test(frame.payload.id) && UUID.test(frame.payload.nonce)) {
              context.phase = 'resetting'; context.resetAcknowledged = false; context.catalogVersion++;
              context.catalogAbort?.abort(); clearCatalog(context); unmount(context);
              await Promise.allSettled([...context.checks]);
              context.catalogBusy = false; context.catalogAbort = null;
              if (!await context.channel.reset()) throw Error('evaluation reset incomplete');
              await context.store.retired(); if (!owns(context)) return;
              context.unmount = mount(context.controller);
              if (typeof context.unmount !== 'function') throw Error('missing evaluator view cleanup');
              context.phase = 'waiting_ready'; context.live.receive(frame); context.live.committed(null);
            } else if (keys(frame, ['type', 'connection_id', 'channels', 'ts']) && frame.type === 'rest_ready'
                && frame.ts === '' && UUID.test(frame.connection_id) && context.phase === 'waiting_ready'
                && context.resetAcknowledged && Array.isArray(frame.channels) && frame.channels.length === 1 && frame.channels[0] === CHANNEL) {
              context.phase = 'ready'; context.channel.open(frame.connection_id); void loadCatalog(context);
            } else if (keys(frame, ['type', 'payload', 'channel', 'ts']) && frame.type === 'rest_retire'
                && frame.ts === '' && frame.channel === CHANNEL && keys(frame.payload, ['event_ids', 'receipts'])) {
              const affected = Array.isArray(frame.payload.event_ids) && frame.payload.event_ids.includes(context.store.deliveryToken());
              if (ready(context) && context.channel.retire(frame) && affected) {
                clearSelection(context, 'expired'); report(context, 'La muestra fue retirada. Actualizando las concesiones…'); void loadCatalog(context);
              }
            }
          }).catch(() => { retryAfter(context); });
        });
        context.socket.addEventListener('close', () => retryAfter(context));
        context.socket.addEventListener('error', () => retryAfter(context));
        return true;
      } catch (_) { await drain(context); return false; }
    }
    function leave() {
      wanted = false; generation = null;
      if (retry !== null) { cancel(retry.handle); retry = null; }
      if (current) return drain(current);
      return closing || Promise.resolve(blocked === null);
    }
    return {
      open() {
        wanted = true; generation = session.generation();
        if (current && (current.generation !== generation || !allowed())) drain(current);
        return start(generation);
      },
      leave, clear: leave, settled: () => closing || Promise.resolve(blocked === null),
    };
  }
  const api = { createEvaluationOwner, createEvaluationStore, permittedPath, parseCatalog, validateEnvelope };
  root.NexoEvaluationOwner = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
