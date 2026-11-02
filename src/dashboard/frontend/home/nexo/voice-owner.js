/* A voice connection belongs to one browser login and one mounted view. */
(function (root) {
  'use strict';
  const liveAPI = root.KzaLivePrivacy || require('../../obs/src/live-privacy.js');
  const voiceAPI = root.KzaVoicePrivacy || require('../../obs/src/voice-privacy.js');
  const channelAPI = root.KzaVoiceChannel || require('../../obs/src/voice-channel.js');
  const requestAPI = root.HomeRequestIdentity || require('../request-id.js');
  const peopleAPI = root.NexoPeople || require('./people.js');
  const CHANNELS = ['voices:captures', 'voices:profiles', 'voices:review'];
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  function confirmationId(seed, identities, name) {
    // Retry grouping only: the server compares the complete original command.
    // A collision rejects the command; this identifier never grants authority.
    const value = JSON.stringify([[...identities].sort(), name.trim()]);
    const words = seed.map((word, stream) => {
      for (let index = 0; index < value.length; index++) word = Math.imul(word ^ value.charCodeAt(index) ^ stream, 16777619);
      return word >>> 0;
    });
    const bytes = words.flatMap(word => [word >>> 24, (word >>> 16) & 255, (word >>> 8) & 255, word & 255]);
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = bytes.map(byte => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  function permittedPath(path, method, includePerson = false) {
    if (typeof path !== 'string' || path.length > 8192 || path.split('?').length > 2) return false;
    const [pathname, query] = path.split('?'), params = new URLSearchParams(query || '');
    if (query !== undefined && !query || [...params.keys()].some(key => !['cursor','include_person'].includes(key))
        || [...params.keys()].some(key => params.getAll(key).length !== 1)
        || params.has('cursor') && !/(?:^|&)cursor=[A-Za-z0-9%_.~-]+(?:&|$)/.test(query)
        || params.has('include_person') && (!includePerson || pathname !== '/api/voices/profiles' || params.get('include_person') !== 'true')) return false;
    if (method === 'GET') return ['/api/voices', '/api/voices/profiles'].includes(pathname)
      || query === undefined && /^\/api\/voices\/captures\/vf_[a-f0-9]{64}\/review$/.test(pathname);
    if (query !== undefined) return false;
    return method === 'POST' && ['/api/voices/profiles', '/api/voices/groups/corrections'].includes(pathname)
      || method === 'DELETE' && pathname.startsWith('/api/voices/profiles/') && UUID.test(pathname.slice('/api/voices/profiles/'.length))
      || includePerson && method === 'PUT' && /^\/api\/voices\/profiles\/([a-f0-9-]+)\/person$/.test(pathname)
        && UUID.test(pathname.split('/')[4]);
  }
  function createVoiceOwner({ session, mount, fetch: rawFetch, connect, socketUrl,
      onStatus = () => {}, onUnauthorized = () => {}, verifyIdentity = async () => true,
      schedule = setTimeout, cancel = clearTimeout, retryMs = 1000,
      createState = voiceAPI.createVoiceState, createController = channelAPI.createVoiceController,
      createLive = liveAPI.createLiveState, pageId = liveAPI.pageId }) {
    let wanted = false, generation = null, current = null, closing = null, retry = null, blocked = null;
    let intentGeneration = null, intentSeed = null;
    const uncertainConfirmations = new Map();
    const allowed = () => session.can('voices:read') && session.can('voices:review');
    const owns = context => current === context && !context.sealed && wanted && allowed()
      && session.generation() === context.generation && generation === context.generation;
    function unmount(context) {
      const stop = context.unmount; context.unmount = null;
      if (stop) stop();
    }
    function seal(context) {
      if (context.sealed) return;
      context.sealed = true; context.phase = 'closed';
      for (const abort of context.auxAborts || []) abort.abort();
      unmount(context); context.controller?.leave();
      context.socket?.close();
    }
    function drain(context) {
      if (closing) return closing;
      seal(context);
      if (current === context) current = null;
      closing = (async () => {
        try {
          await context.frames;
          if (context.controller && !await context.controller.reset()) throw Error('voice_retirement_incomplete');
          await context.state?.retired();
          await Promise.allSettled([...(context.checks || [])]);
          context.controller?.dispose(); context.live?.dispose();
          await context.state?.retired();
          return true;
        } catch {
          // Retain the actual owner if withdrawal could not be established.
          blocked = context; onStatus('No se pudo confirmar la retirada de la vista anterior.');
          return false;
        }
      })();
      const pending = closing;
      pending.then(() => { if (closing === pending) closing = null; });
      return pending;
    }
    function retryAfter(context) {
      if (!owns(context)) return;
      const originalGeneration = context.generation;
      onStatus('Se perdió la conexión. Retirando la vista…');
      drain(context).then(done => {
        if (!done || !wanted || !allowed() || session.generation() !== originalGeneration) return;
        retry = schedule(() => { retry = null; start(); }, retryMs);
      });
    }
    async function start() {
      if (retry !== null) { cancel(retry); retry = null; }
      const attempt = generation;
      if (closing && !await closing) return false;
      if (blocked || !wanted || !allowed() || session.generation() !== attempt || generation !== attempt) return false;
      if (current) return true;
      const context = { generation: attempt, sealed: false, phase: 'hello', frames: Promise.resolve(), unmount: null,
        checks:new Set(), auxAborts:new Set(), includePerson:session.can('admin:manage') };
      current = context;
      try {
      context.state = createState({includePerson:context.includePerson});
      function track(work) {
        const task = Promise.resolve().then(work);
        context.checks.add(task);
        const finish = () => { context.checks.delete(task); context.controller?.flush(); };
        task.then(finish,finish); return task;
      }
      function verifyOriginal() {
        if (!owns(context) || context.phase !== 'ready') return;
        track(async () => {
          if (!owns(context) || context.phase !== 'ready') return;
          try { const valid = await verifyIdentity(); if (owns(context) && context.phase === 'ready' && !valid) onUnauthorized(); }
          catch { if (owns(context)) onStatus('No se pudo verificar la sesión.'); }
        });
      }
      function emitAck(message) {
        if (!owns(context) || context.socket.readyState !== 1 || context.checks.size) return false;
        if (message.type === 'live_reset_ack' ? context.phase !== 'waiting_ready' : context.phase !== 'ready') return false;
        context.socket.send(JSON.stringify(message));
        if (message.type === 'live_reset_ack') context.resetAcknowledged = true;
        return true;
      }
      async function boundFetch(path, options = {}) {
        if (!owns(context) || context.phase !== 'ready' || !permittedPath(path, options.method || 'GET', context.includePerson && session.can('admin:manage'))) throw Error('voice_reader_retired');
        const response = await rawFetch(path, { ...options, credentials: 'same-origin', cache: 'no-store' });
        if (!owns(context) || context.phase !== 'ready') {
          // The channel still owns this fetch until cancellation of its body settles.
          await response.body?.cancel();
          throw Error('voice_reader_retired');
        }
        if (!response.ok) await response.body?.cancel();
        async function read(method) {
          const value = await response[method]();
          if (!owns(context) || context.phase !== 'ready') throw Error('voice_reader_retired');
          return value;
        }
        return { ok: response.ok, status: response.status,
          json: () => read('json'), arrayBuffer: () => read('arrayBuffer') };
      }
      context.live = createLive({ onChange() {}, emitAck });
      context.controller = createController({ state: context.state, fetch: boundFetch, emitAck, includePerson:context.includePerson,
        hasView: () => owns(context) && context.phase === 'ready' && !!context.unmount,
        onUnauthorized:verifyOriginal, onStatus() {} });
      context.controller.loadPeople = accept => {
        const valid = () => owns(context) && context.phase === 'ready' && context.includePerson && session.can('admin:manage');
        if (!valid() || context.auxAborts.size || typeof accept !== 'function') return Promise.resolve(false);
        const abort = new AbortController(); context.auxAborts.add(abort);
        let consumer = accept; accept = null;
        return track(async () => {
          let response = null, document = null, accounts = null;
          try {
            if (!valid() || abort.signal.aborted) return false;
            response = await rawFetch('/api/admin/users', {credentials:'same-origin',cache:'no-store',signal:abort.signal});
            if (!valid() || abort.signal.aborted) { await response.body?.cancel(); return false; }
            if (!response.ok) {
              await response.body?.cancel();
              if (response.status === 401 || response.status === 403) verifyOriginal();
              return false;
            }
            document = await response.json();
            if (!valid() || abort.signal.aborted || !Array.isArray(document) || document.length > 256) return false;
            accounts = peopleAPI.parseAccounts(document).map(row => {
              if (row.id.length > 128 || /[\x00-\x1f\x7f]/.test(row.id) || !row.name.trim() || row.name.length > 128) throw Error('invalid person catalogue');
              return Object.freeze({id:row.id,name:row.name,is_active:row.is_active});
            });
            await consumer(Object.freeze(accounts));
            return valid() && !abort.signal.aborted;
          } catch { return false; }
          finally { response = document = accounts = consumer = null; context.auxAborts.delete(abort); }
        });
      };
      const confirm = context.controller.confirm.bind(context.controller);
      context.controller.confirm = async (identities, name) => {
        if (!owns(context) || context.phase !== 'ready') return false;
        const key = confirmationId(intentSeed, identities, name);
        let requestId = uncertainConfirmations.get(key);
        if (!requestId) {
          if (uncertainConfirmations.size >= 64) { onStatus('Volvé a iniciar sesión para revisar las operaciones pendientes.'); return false; }
          requestId = requestAPI.newRequestId(); uncertainConfirmations.set(key, requestId);
        }
        const accepted = await confirm(identities, name, requestId);
        if (accepted && intentGeneration === context.generation && uncertainConfirmations.get(key) === requestId) uncertainConfirmations.delete(key);
        return accepted;
      };
      context.socket = connect(socketUrl());
      context.socket.addEventListener('open', () => {
        if (!owns(context)) return;
        context.socket.send(JSON.stringify({ type: 'live_hello', version: 3, client_id: pageId() }));
        onStatus('Preparando la consulta privada…');
      });
      context.socket.addEventListener('message', event => {
        if (!owns(context) || typeof event.data !== 'string' || event.data.length > 16384) return;
        let frame; try { frame = JSON.parse(event.data); } catch { return; }
        context.frames = context.frames.then(async () => {
          if (!owns(context)) return;
          if (frame?.type === 'live_reset' && UUID.test(frame.payload?.id) && UUID.test(frame.payload?.nonce)) {
            context.phase = 'resetting'; context.resetAcknowledged = false;
            for (const abort of context.auxAborts) abort.abort();
            unmount(context);
            await Promise.allSettled([...context.checks]);
            if (!await context.controller.reset()) throw Error('voice_reset_incomplete');
            await context.state.retired();
            if (!owns(context)) return;
            context.unmount = mount(context.controller);
            context.phase = 'waiting_ready';
            context.live.receive(frame); context.live.committed(null);
          } else if (frame?.type === 'rest_ready' && context.phase === 'waiting_ready' && context.resetAcknowledged && UUID.test(frame.connection_id)
              && Array.isArray(frame.channels) && frame.channels.length === 3
              && [...frame.channels].sort().every((channel, index) => channel === CHANNELS[index])) {
            context.phase = 'ready'; onStatus('Conexión privada vigente.'); context.controller.open(frame.connection_id);
          } else if (frame?.type === 'rest_retire' && context.phase === 'ready' && CHANNELS.includes(frame.channel)) {
            context.controller.retire(frame);
          }
        }).catch(() => { retryAfter(context); });
      });
      context.socket.addEventListener('close', () => retryAfter(context));
      context.socket.addEventListener('error', () => retryAfter(context));
      return true;
      } catch {
        await drain(context); onStatus('No se pudo abrir la conexión de Voces.'); return false;
      }
    }
    function leave() {
      wanted = false; generation = null;
      if (retry !== null) { cancel(retry); retry = null; }
      if (current) return drain(current);
      return closing || Promise.resolve(blocked === null);
    }
    return {
      open() {
        wanted = true; generation = session.generation();
        if (intentGeneration !== generation) {
          uncertainConfirmations.clear();
          intentGeneration = generation; intentSeed = [...root.crypto.getRandomValues(new Uint32Array(4))];
        }
        if (current && (current.generation !== generation || !allowed())) drain(current);
        return start();
      }, leave,
      clear() { uncertainConfirmations.clear(); intentGeneration = intentSeed = null; return leave(); },
      settled: () => closing || Promise.resolve(blocked === null),
    };
  }
  const api = { createVoiceOwner, permittedPath, confirmationId };
  root.NexoVoiceOwner = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
