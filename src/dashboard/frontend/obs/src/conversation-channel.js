(function (root) {
  'use strict';
  const livePrivacy = root?.KzaLivePrivacy || require('./live-privacy.js');
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  function createConversationChannel({ store, fetch, emitAck, hasView = () => false, onUnauthorized = () => {}, onState = () => {},
      path = '/api/conversations', channel = 'conversations', onPage = () => {}, loadOnOpen = true,
      onRetire = () => {}, retains = () => false, onHttpError = () => false,
      monotonic = () => performance.now(), wall = Date.now }) {
    let connection = null, generation = 0, closed = false, rendered = null, metadata, accepting = false;
    const inflight = new Set(), controllers = new Set(), retired = new Set();
    function fresh() {
      metadata = livePrivacy.createLiveState({
        emitAck: message => connection !== null && inflight.size === 0 && !(store.pendingRetirements?.()) && emitAck(message),
        onChange() { store.clear(); if (!hasView()) { rendered = null; commit(null); } },
        now: wall, retains,
      });
      metadata.committed(null);
    }
    fresh();
    function commit(token) {
      rendered = token;
      if (store.pendingRetirements?.()) store.retired().then(() => { if (!closed) metadata.committed(rendered); });
      else metadata.committed(token);
    }
    const unsubscribe = store.subscribe(() => {
      if (!accepting && !store.snapshot().length && metadata.snapshot()) metadata.clear();
    });
    store.bindPrivacy({ committed:commit,
      token: () => metadata.snapshot()?.event_id ?? null });
    async function reset() {
      const attempt = ++generation;
      onState('waiting');
      connection = null;
      for (const controller of controllers) controller.abort();
      metadata.clear(); store.clear();
      await Promise.allSettled([...inflight]);
      await store.retired?.();
      if (attempt !== generation || closed) return false;
      metadata.dispose(); retired.clear(); rendered = null; fresh();
      return true;
    }
    function open(identity) {
      if (closed || !ID.test(identity) || connection !== null) return false;
      connection = identity;
      if (loadOnOpen) refresh(path);
      return true;
    }
    function refresh(requestPath = path, { method, body, variant, rowId } = {}) {
      if (closed || connection === null || inflight.size || typeof requestPath !== 'string' || !requestPath.startsWith(path)) return false;
      const identity = connection;
      generation++;
      for (const controller of controllers) controller.abort();
      metadata.clear(); store.clear();
      onState('loading');
      const attempt = generation, controller = new AbortController();
      controllers.add(controller);
      const started = { mono: monotonic(), wall: wall() };
      let status = 'unavailable';
      const loading = (async () => {
        let response = null, document = null;
        try {
          // A paused, detached player can still own its native resource. Keep
          // one admitted request until that retirement actually completes.
          if (store.pendingRetirements?.()) await store.retired();
          if (closed || generation !== attempt || controller.signal.aborted) return;
          response = await fetch(requestPath, { credentials:'same-origin', cache:'no-store', signal:controller.signal,
            ...(method === undefined ? {} : {method}), ...(body === undefined ? {} : {body}),
            headers:{ 'X-KZA-Live-Connection':identity, ...(body === undefined ? {} : {'Content-Type':'application/json'}) } });
          if (closed || generation !== attempt || controller.signal.aborted) return;
          if (!response.ok) {
            store.clear();
            if (onHttpError(response.status) === true) status = 'forbidden';
            else if (response.status === 401 || response.status === 403) onUnauthorized();
            return;
          }
          document = await response.json();
          if (closed || generation !== attempt || controller.signal.aborted || document?.connection_id !== connection) return;
          if (document.channel && document.channel !== channel) return;
          if (channel!=='conversations' && (document.channel!==channel || document.snapshot && document.snapshot.channel!==channel)) return;
          const batch = document.snapshot;
          const publishPage = snapshot => {
            let page;
            if (snapshot) {
              const {rows, receipt, ...fields} = snapshot;
              page = fields;
            } else if (document.empty_page) {
              const {matching_count,scan_complete,count_observed_at} = document.empty_page;
              page = {matching_count,scan_complete,count_observed_at};
            } else page = null;
            onPage(snapshot?.next_cursor ?? document.next_cursor ?? null,page,started);
          };
          if (batch === null) { metadata.clear(); status = 'real'; publishPage(null); return; }
          if (!batch || !ID.test(batch.event_id) || !Array.isArray(batch.rows)
              || !Number.isFinite(batch.observed_at) || !Number.isFinite(batch.expires_at)) { metadata.clear(); return; }
          if (retired.has(batch.event_id)) { status = 'real'; return; }
          // One ticket covers the whole batch, so the earliest source retires it all.
          const expiry = batch.expires_at - Math.max(0,monotonic()-started.mono,wall()-started.wall)/1000;
          accepting = true;
          try {
            if (!metadata.receive({ type:'turn', event_id:batch.event_id, expires_at:expiry,
                receipt:batch.receipt, ts:'', payload:{} })) return;
            const rows = batch.rows.map(row => ({ ...row, observed_at:batch.observed_at,
              expires_at:Math.min(row.expires_at ?? batch.expires_at,batch.expires_at) }));
            if (store.receive(rows,started,{variant,rowId})) status = 'real';
            else metadata.clear();
          } finally { accepting = false; }
          if (status === 'real' && store.snapshot().length) publishPage(batch);
          else metadata.clear();
          if (!hasView()) commit(null);
        } catch {
          if (generation === attempt) metadata.clear();
        } finally { response = document = body = null; }
      })();
      inflight.add(loading);
      loading.finally(() => {
        inflight.delete(loading); controllers.delete(controller);
        if (!closed) commit(rendered);
        if (!closed && generation === attempt) onState(status);
      });
      return true;
    }
    function retire(frame) {
      if (frame?.channel !== channel && !(channel==='conversations' && !frame?.channel)) return false;
      const tokens = frame?.payload?.event_ids;
      if (!Array.isArray(tokens) || tokens.length > 32 || !tokens.every(token => ID.test(token))
          || retired.size + tokens.filter(token => !retired.has(token)).length > 64) return false;
      for (const token of tokens) retired.add(token);
      const accepted = metadata.receive({ ...frame, type:'retire' });
      if (accepted) onRetire();
      return accepted;
    }
    function clearView() {
      generation++; for (const controller of controllers) controller.abort();
      metadata.clear(); store.clear();
    }
    return { open, refresh, reset, retire, clearView, busy:() => inflight.size > 0,
      flush() { if (!closed) commit(rendered); }, settled:() => Promise.allSettled([...inflight]),
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
