/* Session ownership for the private observability UI, with no framework state. */
(function (root) {
  'use strict';

  const ACCESS_TIMEOUT_MS = 12000;
  const ACCESS_FAILURES = Object.freeze({
    401: ['login', 'La credencial es inválida o la sesión venció.'],
    403: ['forbidden', 'No tenés permiso para este acceso.'],
    503: ['unavailable', 'La autoridad de acceso no está disponible.'],
    TypeError: ['network', 'No se pudo conectar con el servidor.'],
    TimeoutError: ['timeout', 'El servidor no respondió a tiempo.'],
    AbortError: ['aborted', 'Se canceló la solicitud.'],
  });

  class AccessError extends Error {
    constructor(kind, message) {
      super(message);
      this.kind = kind;
    }
  }

  class SessionRequired extends AccessError {
    constructor() {
      super(...ACCESS_FAILURES[401]);
    }
  }

  function classifyAccessFailure(failure) {
    if (failure instanceof AccessError) return failure;
    if (failure === 401) return new SessionRequired();
    const key = typeof failure === 'number' ? failure : failure?.name;
    return new AccessError(...(Object.hasOwn(ACCESS_FAILURES, key)
      ? ACCESS_FAILURES[key] : ['error', 'No se pudo completar la solicitud.']));
  }

  async function requestAccess(fetch, path, { signal: owner, ...options } = {}, read = response => response) {
    const deadline = new AbortController();
    const signal = owner ? AbortSignal.any([owner, deadline.signal]) : deadline.signal;
    const timer = setTimeout(() => deadline.abort(new DOMException('Access timed out', 'TimeoutError')),
      ACCESS_TIMEOUT_MS);
    try {
      signal.throwIfAborted();
      const response = await fetch(path, { credentials: 'same-origin', cache: 'no-store', ...options, signal });
      signal.throwIfAborted();
      if (!response.ok) throw classifyAccessFailure(response.status);
      const result = await read(response);
      signal.throwIfAborted();
      return result;
    } catch (error) {
      throw classifyAccessFailure(signal.aborted ? signal.reason : error);
    } finally {
      clearTimeout(timer);
    }
  }

  function checkAccess(fetch, signal, read) {
    return requestAccess(fetch, '/api/admin/auth/whoami', { signal }, read);
  }

  function loginAccess(fetch, token, signal) {
    return requestAccess(fetch, '/api/admin/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }), signal,
    });
  }

  async function liveCloseStatus(fetch, code, signal) {
    let identity;
    try {
      identity = await checkAccess(fetch, signal, code === 1006 ? response => response.json() : undefined);
    } catch (error) {
      if (code === 1006 && ['network', 'timeout'].includes(error.kind)) {
        return { kind: error.kind, message: `Reconectando. ${error.message}`, retry: true };
      }
      return { failure: error };
    }
    // Chromium reports a rejection before the handshake completes as 1006.
    const permissions = identity?.permissions;
    const denied = code === 1008 || (code === 1006 &&
      !(Array.isArray(permissions) && permissions.includes('conversations:read')));
    if (denied || code === 1013) {
      const { kind, message } = classifyAccessFailure(denied ? 403 : 503);
      return { kind, message, retry: !denied };
    }
    return { kind: 'disconnected', message: 'desconectado', retry: true };
  }

  function createSession({ fetch, hydrate, connect, clear, onState }) {
    let generation = 0, abort = null, dispose = null;
    let initialCheck = true;
    function reset() {
      generation += 1;
      abort?.abort(); abort = null;
      dispose?.(); dispose = null;
      clear();
    }
    function fail(error) {
      initialCheck = false;
      reset();
      const failure = classifyAccessFailure(error);
      const state = ['login', 'forbidden'].includes(failure.kind) ? failure.kind : 'unavailable';
      onState(state, failure);
    }
    function expire() {
      initialCheck = false;
      reset();
      onState('login');
    }
    async function start() {
      reset();
      const attempt = generation;
      const controller = new AbortController();
      abort = controller;
      onState('checking');
      try {
        await checkAccess(fetch, controller.signal);
        if (attempt !== generation) return;
        initialCheck = false;
        onState('loading');
        await hydrate({ signal: controller.signal });
        if (attempt !== generation || controller.signal.aborted) return;
        dispose = connect();
        onState('ready', undefined, controller.signal);
      } catch (error) {
        if (attempt !== generation || controller.signal.aborted) return;
        if (initialCheck && error instanceof SessionRequired) expire();
        else fail(error);
      }
    }
    return {
      start,
      fail,
      expire,
      stop: reset,
    };
  }

  const api = { createSession, SessionRequired, requestAccess, checkAccess, loginAccess, liveCloseStatus };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KzaDashboardSession = api;
})(typeof window === 'undefined' ? globalThis : window);
