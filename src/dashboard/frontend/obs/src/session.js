/* Session ownership for the private observability UI, with no framework state. */
(function (root) {
  'use strict';

  class SessionRequired extends Error {}

  function createSession({ fetch, hydrate, connect, clear, onState }) {
    let generation = 0, abort = null, dispose = null;
    function reset() {
      generation += 1;
      abort?.abort(); abort = null;
      dispose?.(); dispose = null;
      clear();
    }
    async function start() {
      reset();
      const attempt = generation;
      const controller = new AbortController();
      abort = controller;
      onState('checking');
      try {
        const response = await fetch('/api/admin/auth/whoami', {
          credentials: 'same-origin', signal: controller.signal,
        });
        if (attempt !== generation) return;
        if (!response.ok) {
          onState(response.status === 401 ? 'login' : 'unavailable');
          return;
        }
        onState('loading');
        await hydrate({ signal: controller.signal });
        if (attempt !== generation || controller.signal.aborted) return;
        dispose = connect();
        onState('ready');
      } catch (error) {
        if (attempt !== generation || controller.signal.aborted) return;
        reset();
        onState(error instanceof SessionRequired ? 'login' : 'unavailable');
      }
    }
    return {
      start,
      expire() { reset(); onState('login'); },
      stop: reset,
    };
  }

  const api = { createSession, SessionRequired };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KzaDashboardSession = api;
})(typeof window === 'undefined' ? globalThis : window);
