(function (root) {
  'use strict';
  const TYPES = new Set(['turn', 'wake', 'tts', 'alert', 'cooldown']);
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const FIELDS = new Set(['id', 'user', 'zone', 'stt', 'tts', 'intent', 'path', 'latency_ms',
    'success', 'confidence', 'text', 'title', 'body', 'endpoint_id', 'until', 'priority', 'type', 'duration_ms', 'step']);

  function createLiveState({ onChange, now = Date.now, schedule = setTimeout, cancel = clearTimeout }) {
    let current = null, timer = null, disposed = false;
    function clear() {
      if (timer !== null) cancel(timer);
      timer = null;
      current = null;
      onChange(null);
    }
    function expire() {
      if (current && current.expires_at * 1000 <= now()) clear();
    }
    function arm() {
      if (timer !== null) cancel(timer);
      timer = null;
      if (!current) return;
      const token = current.event_id;
      const delay = Math.max(0, current.expires_at * 1000 - now());
      timer = schedule(() => {
        timer = null;
        if (!current || current.event_id !== token) return;
        expire();
        if (current) arm();
      }, Math.min(delay, 2147483647));
    }
    function receive(frame) {
      if (disposed) return false;
      expire();
      if (!frame || typeof frame !== 'object') return false;
      if (frame.type === 'retire') {
        const ids = frame.payload?.event_ids;
        if (!Array.isArray(ids) || ids.length > 32 || !ids.every(id => typeof id === 'string' && ID.test(id))) return false;
        if (current && ids.includes(current.event_id)) clear();
        return true;
      }
      if (!TYPES.has(frame.type) || typeof frame.event_id !== 'string' || !ID.test(frame.event_id)
          || !Number.isFinite(frame.expires_at) || !Number.isFinite(frame.expires_at * 1000) || frame.expires_at * 1000 <= now()
          || typeof frame.ts !== 'string' || frame.ts.length > 32
          || !frame.payload || typeof frame.payload !== 'object' || Array.isArray(frame.payload)) return false;
      const payload = {};
      for (const [key, value] of Object.entries(frame.payload)) {
        if (!FIELDS.has(key)) continue;
        if (!(value === null || typeof value === 'boolean' || (typeof value === 'string' && value.length <= 8192)
            || (typeof value === 'number' && Number.isFinite(value)))) return false;
        payload[key] = value;
      }
      const expiry = current?.event_id === frame.event_id ? Math.min(current.expires_at, frame.expires_at) : frame.expires_at;
      current = Object.freeze({ type: frame.type, event_id: frame.event_id, expires_at: expiry, ts: frame.ts, payload: Object.freeze(payload) });
      arm();
      onChange(current);
      return true;
    }
    function snapshot() { expire(); return current; }
    function dispose() { disposed = true; clear(); }
    return { receive, snapshot, clear, expire, dispose };
  }
  const api = { createLiveState };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaLivePrivacy = api;
})(typeof window === 'undefined' ? null : window);
