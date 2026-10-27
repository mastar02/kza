(function (root) {
  'use strict';
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const VF = /^vf_[0-9a-f]{64}$/;
  function createVoiceStore(kind, { monotonic = () => performance.now(), wall = Date.now,
      schedule = setTimeout, cancel = clearTimeout, url = root?.URL || URL } = {}) {
    let entries = [], timer = null, privacy = null, media = null, status = 'waiting';
    const listeners = new Set(), retiring = new Map();
    function notify() { for (const listener of listeners) listener(); }
    function retireMedia() {
      if (!media) return;
      const element = media; media = null;
      const needsEvent = element.networkState !== 0 || element.readyState !== 0 || !!element.currentSrc;
      let emptied = false, finished = false, verify = null, finish;
      const pending = new Promise(resolve => { finish = resolve; });
      function check() {
        if (finished) return;
        // Chromium can retain the revoked currentSrc string after load(). The
        // load/emptied contract resets the actual resource, tracks and ranges:
        // https://html.spec.whatwg.org/multipage/media.html#media-element-load-algorithm
        if ((!needsEvent || emptied) && !element.isConnected && element.paused && !element.hasAttribute('src')
            && element.srcObject === null && !element.querySelector('source')
            && element.readyState === 0 && element.networkState === 0
            && element.buffered.length === 0 && Number.isNaN(element.duration)) {
          finished = true;
          if (verify !== null) clearTimeout(verify);
          element.removeEventListener('emptied', onEmpty); finish();
        } else if ((!needsEvent || emptied) && verify === null) {
          // The event and final public media state can settle on different
          // native turns. Observe progress; elapsed time never authorizes ACK.
          verify = setTimeout(() => { verify = null; check(); },5);
        }
      }
      function onEmpty() { emptied = true; check(); }
      element.addEventListener('emptied', onEmpty);
      retiring.set(pending,() => ({needs_event:needsEvent,emptied,connected:element.isConnected,paused:element.paused,
        has_source:element.hasAttribute('src'),has_provider:element.srcObject!==null,has_child:!!element.querySelector('source'),
        ready:element.readyState,network:element.networkState,buffers:element.buffered.length,empty_duration:Number.isNaN(element.duration)}));
      pending.then(() => { retiring.delete(pending); });
      element.pause(); element.removeAttribute('src'); element.remove(); element.load();
      queueMicrotask(check);
    }
    function release(rows) { for (const entry of rows) if (entry.row.audio_url) url.revokeObjectURL(entry.row.audio_url); }
    function clear() {
      retireMedia(); release(entries); entries = [];
      if (timer !== null) cancel(timer); timer = null; notify();
    }
    function prune() {
      const now = monotonic(), clock = wall();
      const expired = entries.filter(entry => now >= entry.mono || clock >= entry.wall);
      if (expired.length) {
        retireMedia(); release(expired);
        entries = entries.filter(entry => !expired.includes(entry)); status = 'expired'; notify();
      }
    }
    function arm() {
      if (timer !== null) cancel(timer); timer = null;
      if (!entries.length) return;
      const delay = Math.min(...entries.map(entry => Math.min(entry.mono-monotonic(),entry.wall-wall())));
      timer = schedule(() => { timer = null; prune(); arm(); },Math.max(1,Math.min(delay,2147483647)));
    }
    function copy(value) {
      if (kind === 'profiles') {
        if (!ID.test(value.id) || typeof value.name !== 'string' || !value.name.trim() || value.name.length>64
            || !Number.isInteger(value.revision) || value.revision<1 || !Number.isFinite(value.confirmed_at)) throw Error('invalid voice profile');
        return { id:value.id, name:value.name, revision:value.revision, confirmed_at:value.confirmed_at };
      }
      if (!VF.test(value.id)) throw Error('invalid voice capture');
      if (kind === 'captures') {
        if (!Number.isFinite(value.duration_s) || value.duration_s<.8 || value.duration_s>30
            || !Number.isFinite(value.created_at) || typeof value.review_available !== 'boolean'
            || value.suggested_profile !== null && !ID.test(value.suggested_profile)
            || value.similarity !== null && (!Number.isFinite(value.similarity) || Math.abs(value.similarity)>1)) throw Error('invalid capture metadata');
        return { id:value.id, duration_s:value.duration_s, created_at:value.created_at, review_available:value.review_available,
          suggested_profile:value.suggested_profile, similarity:value.similarity };
      }
      if (kind !== 'review' || typeof value.transcript !== 'string' || value.transcript.length>8192
          || value.room_id !== null && (typeof value.room_id !== 'string' || value.room_id.length>128)
          || value.audio_mime !== 'audio/wav' || value.transcript_scope !== 'full_segment'
          || !Number.isFinite(value.duration_s) || value.duration_s<.8 || value.duration_s>30
          || !Number.isFinite(value.clip_duration_s) || value.clip_duration_s<.8 || value.clip_duration_s>Math.min(value.duration_s,6)
          || typeof value.audio_base64 !== 'string' || value.audio_base64.length>256060
          || !/^[A-Za-z0-9+/]+={0,2}$/.test(value.audio_base64)) throw Error('invalid listening sample');
      const raw = atob(value.audio_base64), bytes = Uint8Array.from(raw,c => c.charCodeAt(0));
      const view = new DataView(bytes.buffer), text = (offset,size) => String.fromCharCode(...bytes.slice(offset,offset+size));
      if (bytes.length<44 || bytes.length>192044 || text(0,4)!=='RIFF' || text(8,4)!=='WAVE'
          || text(12,4)!=='fmt ' || view.getUint32(16,true)!==16 || view.getUint16(20,true)!==1
          || view.getUint16(22,true)!==1 || view.getUint32(24,true)!==16000 || view.getUint16(34,true)!==16
          || text(36,4)!=='data' || view.getUint32(40,true)!==bytes.length-44
          || bytes.length-44!==Math.round(value.clip_duration_s*16000)*2) throw Error('invalid listening WAV');
      return { id:value.id, transcript:value.transcript, room_id:value.room_id, duration_s:value.duration_s,
        clip_duration_s:value.clip_duration_s, clip_truncated:value.clip_duration_s<value.duration_s,
        audio_url:url.createObjectURL(new Blob([bytes],{type:'audio/wav'})) };
    }
    return {
      receive(rows,started) {
        const next = [];
        try {
          const elapsed = started && Math.max(0,monotonic()-started.mono,wall()-started.wall);
          if (!Array.isArray(rows) || rows.length>(kind==='review'?1:50) || !Number.isFinite(elapsed)) throw Error('invalid voice snapshot');
          const ids = new Set();
          for (const value of rows) {
            if (!value || !Number.isFinite(value.observed_at) || !Number.isFinite(value.expires_at)
                || !Number.isFinite(value.expires_at*1000) || value.expires_at<=value.observed_at
                || value.expires_at-value.observed_at>(kind==='profiles'?300:172800) || ids.has(value.id)) throw Error('invalid voice deadline');
            const budget = (value.expires_at-value.observed_at)*1000-elapsed;
            if (budget<=0 || value.expires_at*1000<=wall()) continue;
            const row = copy(value); ids.add(row.id);
            next.push({ row:Object.freeze(row), mono:monotonic()+budget, wall:Math.min(value.expires_at*1000,wall()+budget) });
          }
          clear(); entries = next; status = 'real'; prune(); arm(); notify(); return true;
        } catch { release(next); clear(); status = 'unavailable'; notify(); return false; }
      },
      clear, check() { prune(); arm(); }, snapshot() { prune(); return entries.map(entry => entry.row); },
      status:() => status, setStatus(value) { status = value; notify(); },
      attachMedia(element) { retireMedia(); media = element; }, detachMedia:retireMedia,
      pendingRetirements:() => retiring.size, retired:() => Promise.allSettled([...retiring.keys()]),
      retirementStatus:() => [...retiring.values()].map(snapshot => snapshot()),
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      bindPrivacy(value) { privacy = value; }, committed(token) { privacy?.committed(token); },
      deliveryToken:() => privacy?.token() ?? null,
    };
  }
  function createVoiceState(options) {
    const state = Object.fromEntries(['profiles','captures','review'].map(kind => [kind,createVoiceStore(kind,options)]));
    return Object.assign(state,{ clear() { for (const key of ['profiles','captures','review']) state[key].clear(); },
      retired:() => Promise.all(['profiles','captures','review'].map(key => state[key].retired())) });
  }
  const api = { createVoiceStore,createVoiceState };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaVoicePrivacy = api;
})(typeof window === 'undefined' ? null : window);
