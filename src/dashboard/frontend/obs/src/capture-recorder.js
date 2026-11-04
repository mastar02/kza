/* A modal owns acquisition, PCM, upload and explicit server retirement. */
(function (root) {
  'use strict';
  const owners = new Set();
  const endpoint = '/api/capture/recorders/';

  function createModalOwner(options = {}) {
    const fetcher = options.fetch || root.fetch.bind(root);
    const media = options.getUserMedia || (value => root.navigator.mediaDevices.getUserMedia(value));
    const Audio = options.AudioContext || root.AudioContext || root.webkitAudioContext;
    const later = options.setTimeout || root.setTimeout.bind(root);
    const cancelTimer = options.clearTimeout || root.clearTimeout.bind(root);
    let token = null, sealed = false, stopPromise = null, heartbeat = null;
    let activation = null, cleanupFailed = false;
    const operations = new Set(), resources = new Set(), uploads = new Set(), blobs = [];
    let owner;

    async function json(action, body, keepalive = false) {
      const abort = new AbortController();
      const timeout = later(() => abort.abort(), 5000);
      try {
        const response = await fetcher(endpoint + action, {
          method: 'POST', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body || {}), signal: abort.signal, keepalive,
        });
        if (!response.ok) {
          if (response.status === 401) root.dispatchEvent?.(new Event('kza:auth-required'));
          throw Error('capture_recorder_unavailable');
        }
        return await response.json();
      } finally { cancelTimer(timeout); }
    }

    const registration = json('register').then(value => {
      if (typeof value.token !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(value.token)
          || value.state !== 'reserved' || value.capture_allowed !== false) {
        throw Error('capture_recorder_unavailable');
      }
      token = value.token;
    });
    registration.catch(() => {});

    function current() { return !sealed; }

    function scheduleHeartbeat() {
      if (sealed) return;
      heartbeat = later(async () => {
        heartbeat = null;
        try {
          const value = await json('poll', { token });
          if (value.capture_allowed !== true) throw Error('capture_muted');
          scheduleHeartbeat();
        } catch (_) { stop().catch(() => {}); }
      }, options.heartbeatMs || 500);
    }

    function active() {
      if (!activation) activation = (async () => {
        await registration;
        if (sealed) throw Error('capture_muted');
        const value = await json('activate', { token });
        if (sealed || value.capture_allowed !== true || value.state !== 'active') {
          throw Error('capture_muted');
        }
        scheduleHeartbeat();
      })();
      return activation;
    }

    function retain(operation) {
      operations.add(operation);
      operation.then(() => operations.delete(operation), () => operations.delete(operation));
      return operation;
    }

    function interrupt(state) {
      state.cancelled = true;
      if (state.timer !== null) cancelTimer(state.timer);
      state.timer = null;
      state.wake?.();
      state.wake = null;
      if (state.proc) state.proc.onaudioprocess = null;
      if (state.stream) for (const track of state.stream.getTracks()) {
        try { track.stop(); } catch (_) { state.failed = cleanupFailed = true; }
      }
    }

    async function cleanup(state) {
      if (state.proc) state.proc.onaudioprocess = null;
      for (const node of [state.proc, state.source]) if (node) {
        try { node.disconnect(); } catch (_) { state.failed = cleanupFailed = true; }
      }
      if (state.stream) for (const track of state.stream.getTracks()) {
        try {
          track.stop();
          if (track.readyState !== 'ended') throw Error('capture_track_unconfirmed');
        } catch (_) { state.failed = cleanupFailed = true; }
      }
      if (state.ctx) {
        try {
          await state.ctx.close();
          if (state.ctx.state !== 'closed') throw Error('capture_context_unconfirmed');
        } catch (_) { state.failed = cleanupFailed = true; }
      }
      if (state.failed) throw Error('capture_retirement_unconfirmed');
      state.proc = state.source = state.stream = state.ctx = null;
      resources.delete(state);
    }

    function recordWavSample(durationSec, encodeWav) {
      const operation = (async () => {
        const state = { stream: null, ctx: null, source: null, proc: null,
          timer: null, wake: null, failed: false, cancelled: false };
        const chunks = [];
        resources.add(state);
        try {
          await active();
          if (sealed) throw Error('capture_muted');
          state.stream = await media({
            audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: true },
          });
          if (sealed || state.cancelled) throw Error('capture_muted');
          state.ctx = new Audio({ sampleRate: 48000 });
          if (sealed) throw Error('capture_muted');
          state.source = state.ctx.createMediaStreamSource(state.stream);
          state.proc = state.ctx.createScriptProcessor(4096, 1, 1);
          state.proc.onaudioprocess = event => {
            if (!sealed && !state.cancelled) {
              chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
            }
          };
          if (sealed) throw Error('capture_muted');
          state.source.connect(state.proc);
          state.proc.connect(state.ctx.destination);
          await new Promise(resolve => {
            state.wake = resolve;
            state.timer = later(resolve, durationSec * 1000);
          });
        } catch (error) {
          chunks.length = 0;
          throw error;
        } finally {
          if (state.timer !== null) cancelTimer(state.timer);
          state.timer = null; state.wake = null;
          try { await cleanup(state); }
          catch (error) { chunks.length = 0; throw error; }
        }
        try {
          if (sealed || state.cancelled) throw Error('capture_muted');
          const total = chunks.reduce((n, chunk) => n + chunk.length, 0);
          const merged = new Float32Array(total);
          let offset = 0;
          for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length; }
          const downsampled = new Int16Array(Math.floor(merged.length / 3));
          for (let i = 0; i < downsampled.length; i++) {
            const average = (merged[i * 3] + merged[i * 3 + 1] + merged[i * 3 + 2]) / 3;
            downsampled[i] = Math.max(-32768, Math.min(32767, Math.round(average * 32767)));
          }
          if (sealed) throw Error('capture_muted');
          const blob = encodeWav(downsampled, 16000);
          blobs.push(blob);
          return blob;
        } finally { chunks.length = 0; }
      })();
      retain(operation);
      operation.catch(() => { stop().catch(() => {}); });
      return operation;
    }

    function upload(path, body) {
      const abort = new AbortController();
      uploads.add(abort);
      const operation = (async () => {
        try {
          await active();
          if (sealed) throw Error('capture_muted');
          const response = await fetcher(path, { method: 'POST', credentials: 'same-origin',
            body, signal: abort.signal });
          if (sealed) throw Error('capture_muted');
          if (response.status === 401) root.dispatchEvent?.(new Event('kza:auth-required'));
          return response;
        } finally { body = null; uploads.delete(abort); }
      })();
      retain(operation);
      return operation;
    }

    function appendSamples(body) {
      if (sealed) throw Error('capture_muted');
      blobs.forEach((blob, index) => body.append('samples', blob, `sample_${index}.wav`));
    }

    function uploadSamples(path) {
      const Form = options.FormData || root.FormData;
      const body = new Form();
      appendSamples(body);
      return upload(path, body);
    }

    function stop() {
      if (stopPromise) return stopPromise;
      if (!sealed) {
        sealed = true;
        if (heartbeat !== null) cancelTimer(heartbeat);
        heartbeat = null;
        for (const state of resources) interrupt(state);
        for (const abort of uploads) abort.abort();
        blobs.length = 0;
        try { options.onRetiring?.(); } catch (_) { /* UI cannot skip disposal. */ }
      }
      stopPromise = (async () => {
        await Promise.allSettled([...operations]);
        await registration.catch(() => {});
        if (activation) await activation.catch(() => {});
        if (cleanupFailed || resources.size) throw Error('capture_retirement_unconfirmed');
        if (token) {
          const value = await json('retire', { token }, true);
          if (value.retired !== true) throw Error('capture_retirement_unconfirmed');
          token = null;
        }
        owners.delete(owner);
      })();
      stopPromise.catch(() => { stopPromise = null; });
      return stopPromise;
    }

    owner = { recordWavSample, uploadSamples, upload, stop, current,
      sampleCount: () => blobs.length };
    owners.add(owner);
    return owner;
  }

  function stopAll() { for (const owner of owners) owner.stop().catch(() => {}); }
  root.addEventListener?.('pagehide', stopAll);
  root.addEventListener?.('kza:auth-required', stopAll);
  const api = { createModalOwner, stopAll };
  root.KzaCaptureRecorder = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
