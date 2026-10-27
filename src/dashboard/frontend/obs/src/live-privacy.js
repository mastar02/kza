(function (root) {
  'use strict';
  const TYPES = new Set(['turn', 'wake', 'tts', 'alert', 'cooldown']);
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const FIELDS = new Set(['id', 'user', 'zone', 'stt', 'tts', 'intent', 'path', 'latency_ms',
    'success', 'confidence', 'text', 'title', 'body', 'endpoint_id', 'until', 'priority', 'type', 'duration_ms', 'step']);
  let pageIdentity = null;
  function pageId() {
    if (!pageIdentity) {
      const bytes = new Uint8Array(16);
      (root?.crypto || globalThis.crypto).getRandomValues(bytes);
      bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
      const hex = [...bytes].map(b => b.toString(16).padStart(2, '0')).join('');
      pageIdentity = `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
    }
    return pageIdentity;
  }
  function validId(value) { return typeof value === 'string' && ID.test(value); }
  function receipt(value) {
    return value && validId(value.ticket) && validId(value.nonce)
      ? Object.freeze({ ticket: value.ticket, nonce: value.nonce }) : null;
  }

  function createLiveState({ onChange, emitAck = () => false, now = Date.now,
      schedule = setTimeout, cancel = clearTimeout, defer = queueMicrotask }) {
    let current = null, timer = null, disposed = false, rendered;
    let reset = null, queued = false;
    const pending = new Map();
    function flush() {
      queued = false;
      if (rendered === undefined) return;
      if (reset) {
        if (current === null && rendered === null && emitAck({ type: 'live_reset_ack', ...reset }) === true) reset = null;
        return;
      }
      const retired = [...pending.values()].filter(row => row.eventId !== current?.event_id && row.eventId !== rendered);
      for (let index = 0; index < retired.length; index += 16) {
        const batch = retired.slice(index, index + 16);
        if (emitAck({ type:'live_ack', receipts:batch.map(row => ({ ticket:row.ticket, nonce:row.nonce })) }) !== true) return;
        for (const row of batch) pending.delete(row.ticket);
      }
    }
    function requestAck() {
      if (!queued) { queued = true; defer(flush); }
    }
    function remember(capability, eventId) {
      if (!capability || (!pending.has(capability.ticket) && pending.size >= 64)) return false;
      pending.set(capability.ticket, { ...capability, eventId });
      requestAck();
      return true;
    }
    function clear(keepReceipt = true) {
      if (timer !== null) cancel(timer);
      timer = null;
      if (keepReceipt && current) remember(current.receipt, current.event_id);
      current = null;
      onChange(null);
      requestAck();
    }
    function committed(eventId) {
      if (eventId !== null && !validId(eventId)) return;
      rendered = eventId;
      requestAck();
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
      if (frame.type === 'live_reset') {
        if (!validId(frame.payload?.id) || !validId(frame.payload?.nonce)) return false;
        pending.clear();
        reset = Object.freeze({ id:frame.payload.id, nonce:frame.payload.nonce });
        clear(false);
        return true;
      }
      if (frame.type === 'retire') {
        const ids = frame.payload?.event_ids, rows = frame.payload?.receipts || [];
        if (!Array.isArray(ids) || ids.length > 32 || !ids.every(validId)
            || !Array.isArray(rows) || rows.length > 32
            || !rows.every(row => ids.includes(row.event_id) && receipt(row))) return false;
        for (const row of rows) remember(receipt(row), row.event_id);
        if (current && ids.includes(current.event_id)) clear();
        requestAck();
        return true;
      }
      const capability = receipt(frame.receipt);
      if (reset || !TYPES.has(frame.type) || !validId(frame.event_id) || !capability
          || !Number.isFinite(frame.expires_at) || !Number.isFinite(frame.expires_at * 1000)
          || typeof frame.ts !== 'string' || frame.ts.length > 32
          || !frame.payload || typeof frame.payload !== 'object' || Array.isArray(frame.payload)) return false;
      if (frame.expires_at * 1000 <= now()) {
        remember(capability, frame.event_id);
        return false;
      }
      const payload = {};
      for (const [key, value] of Object.entries(frame.payload)) {
        if (!FIELDS.has(key)) continue;
        if (!(value === null || typeof value === 'boolean' || (typeof value === 'string' && value.length <= 8192)
            || (typeof value === 'number' && Number.isFinite(value)))) return false;
        payload[key] = value;
      }
      const expiry = current?.event_id === frame.event_id ? Math.min(current.expires_at, frame.expires_at) : frame.expires_at;
      if (current && current.receipt.ticket !== capability.ticket) remember(current.receipt, current.event_id);
      current = Object.freeze({ type:frame.type, event_id:frame.event_id, expires_at:expiry,
        ts:frame.ts, payload:Object.freeze(payload), receipt:capability });
      arm();
      onChange(current.event_id);
      requestAck();
      return true;
    }
    function snapshot() { expire(); return current; }
    function dispose() { disposed = true; clear(); }
    return { receive, snapshot, clear, expire, dispose, committed };
  }
  const api = { createLiveState, pageId };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaLivePrivacy = api;
})(typeof window === 'undefined' ? null : window);
