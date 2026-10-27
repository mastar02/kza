(function (root) {
  'use strict';
  const livePrivacy = root?.KzaLivePrivacy || require('./live-privacy.js');
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  function createConversationChannel({ store, fetch, emitAck, hasView = () => false, onUnauthorized = () => {}, onState = () => {},
      monotonic = () => performance.now(), wall = Date.now }) {
    let connection = null, generation = 0, closed = false, rendered = null, metadata, accepting = false;
    const inflight = new Set(), controllers = new Set(), retired = new Set();
    function fresh() {
      metadata = livePrivacy.createLiveState({
        emitAck: message => connection !== null && inflight.size === 0 && emitAck(message),
        onChange() { store.clear(); if (!hasView()) { rendered = null; metadata.committed(null); } },
        now: wall,
      });
      metadata.committed(null);
    }
    fresh();
    const unsubscribe = store.subscribe(() => {
      if (!accepting && !store.snapshot().length && metadata.snapshot()) metadata.clear();
    });
    store.bindPrivacy({ committed(token) { rendered = token; metadata.committed(token); },
      token: () => metadata.snapshot()?.event_id ?? null });
    async function reset() {
      const attempt = ++generation;
      onState('waiting');
      connection = null;
      for (const controller of controllers) controller.abort();
      metadata.clear(); store.clear();
      await Promise.allSettled([...inflight]);
      if (attempt !== generation || closed) return false;
      metadata.dispose(); retired.clear(); rendered = null; fresh();
      return true;
    }
    function open(identity) {
      if (closed || !ID.test(identity) || connection !== null) return false;
      connection = identity;
      onState('loading');
      const attempt = generation, controller = new AbortController();
      controllers.add(controller);
      const started = { mono: monotonic(), wall: wall() };
      let loaded = false;
      const loading = (async () => {
        let response = null, document = null;
        try {
          response = await fetch('/api/conversations', { credentials:'same-origin', signal:controller.signal,
            headers:{ 'X-KZA-Live-Connection':identity } });
          if (closed || generation !== attempt || controller.signal.aborted) return;
          if (!response.ok) {
            store.clear();
            if (response.status === 401 || response.status === 403) onUnauthorized();
            return;
          }
          document = await response.json();
          if (closed || generation !== attempt || controller.signal.aborted || document?.connection_id !== connection) return;
          const batch = document.snapshot;
          if (batch === null) { metadata.clear(); loaded = true; return; }
          if (!batch || !ID.test(batch.event_id) || !Array.isArray(batch.rows)
              || !Number.isFinite(batch.observed_at) || !Number.isFinite(batch.expires_at)) { metadata.clear(); return; }
          if (retired.has(batch.event_id)) { loaded = true; return; }
          // One ticket covers the whole batch, so the earliest source retires it all.
          const expiry = batch.expires_at - Math.max(0,monotonic()-started.mono,wall()-started.wall)/1000;
          accepting = true;
          try {
            if (!metadata.receive({ type:'turn', event_id:batch.event_id, expires_at:expiry,
                receipt:batch.receipt, ts:'', payload:{} })) return;
            const rows = batch.rows.map(row => ({ ...row, expires_at:Math.min(row.expires_at,batch.expires_at) }));
            loaded = store.receive(rows,started);
            if (!loaded) metadata.clear();
          } finally { accepting = false; }
          if (!store.snapshot().length) metadata.clear();
          if (!hasView()) { rendered = null; metadata.committed(null); }
        } catch {
          if (generation === attempt) metadata.clear();
        } finally { response = document = null; }
      })();
      inflight.add(loading);
      loading.finally(() => {
        inflight.delete(loading); controllers.delete(controller);
        if (!closed) metadata.committed(rendered);
        if (!closed && generation === attempt) onState(loaded ? 'real' : 'unavailable');
      });
      return true;
    }
    function retire(frame) {
      const tokens = frame?.payload?.event_ids;
      if (!Array.isArray(tokens) || tokens.length > 32 || !tokens.every(token => ID.test(token))
          || retired.size + tokens.filter(token => !retired.has(token)).length > 64) return false;
      for (const token of tokens) retired.add(token);
      return metadata.receive({ ...frame, type:'retire' });
    }
    return { open, reset, retire,
      dispose() {
        closed = true; generation++;
        onState('closed');
        for (const controller of controllers) controller.abort();
        metadata.dispose(); store.clear(); unsubscribe(); store.bindPrivacy(null);
        queueMicrotask(() => { connection = null; });
      },
    };
  }
  const api = { createConversationChannel };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaConversationChannel = api;
})(typeof window === 'undefined' ? null : window);
