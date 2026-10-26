/* Private review: no mock fallback, token storage or automatic confirmation. */
(function (host) {
  'use strict';
  function createMemoryController({ apiFetch, onChange = () => {}, monotonic = () => host.performance.now(), wall = Date.now, schedule = setTimeout, cancel = clearTimeout }) {
    let alive = true, version = 0, timer, observed = 0, receivedMono = 0, receivedWall = 0;
    const requests = new Set(), busy = new Set();
    let state = { ready: false, loading: false, facts: [], proposals: [], error: null, pending_index: 0 };
    const now = () => observed + Math.max(0, monotonic() - receivedMono, wall() - receivedWall) / 1000;
    const live = item => item.expires_at === null || (Number.isFinite(item.expires_at) && item.expires_at > now());
    function emit() { if (alive) onChange({ ...state, busy: [...busy] }); }
    function prune() { state.facts = state.facts.filter(live); state.proposals = state.proposals.filter(live); }
    function tick() {
      if (!alive) return;
      if (timer !== undefined) cancel(timer);
      prune(); emit();
      const deadlines = [...state.facts, ...state.proposals].map(item => item.expires_at).filter(Number.isFinite);
      if (deadlines.length) timer = schedule(tick, Math.max(1, Math.min(1000, (Math.min(...deadlines) - now()) * 1000)));
    }
    async function request(url, init = {}) {
      const controller = new AbortController(); requests.add(controller);
      try { return await apiFetch(url, { ...init, signal: controller.signal }); }
      finally { requests.delete(controller); }
    }
    async function refresh() {
      if (!alive) return;
      const attempt = ++version;
      state.loading = true; emit();
      try {
        const data = await request('/api/memory');
        if (!alive || attempt !== version) return;
        if (!Number.isFinite(data.observed_at) || !Array.isArray(data.facts) || !Array.isArray(data.proposals)) throw new Error('invalid memory snapshot');
        observed = data.observed_at; receivedMono = monotonic(); receivedWall = wall();
        state = { ...data, ready: true, loading: false, error: null }; tick();
      } catch (error) {
        if (!alive || attempt !== version) return;
        state = { ready: false, loading: false, facts: [], proposals: [], pending_index: 0,
          error: error.status === 503 ? 'La memoria no está disponible.' : 'No pudimos consultar la memoria. Actualizá para volver a intentar.' };
        emit();
      }
    }
    async function act(id, operation) {
      if (!alive || busy.has(id)) return;
      prune();
      const collection = operation === 'delete' ? state.facts : state.proposals;
      if (!collection.some(item => item.id === id)) { emit(); return; }
      busy.add(id); emit();
      const url = operation === 'delete' ? `/api/memory/facts/${id}` : `/api/memory/proposals/${id}/${operation}`;
      try {
        await request(url, { method: operation === 'delete' ? 'DELETE' : 'POST' });
        if (alive) await refresh();
      } catch (error) {
        if (!alive) return;
        version++;
        state = { ready: false, loading: false, facts: [], proposals: [], pending_index: 0,
          error: error.status === 409 || error.status === 404 ? 'El recuerdo cambió o venció. Actualizá para consultar su estado.' : 'El resultado quedó sin comprobar. Actualizá antes de volver a actuar.' };
      } finally { busy.delete(id); emit(); }
    }
    return { refresh, act, tick, getState: () => ({ ...state, busy: [...busy] }),
      destroy() { alive = false; version++; if (timer !== undefined) cancel(timer); requests.forEach(request => request.abort()); requests.clear(); state = { ready: false, facts: [], proposals: [] }; busy.clear(); } };
  }

  function mountMemory(container, { apiFetch } = {}) {
    const doc = container.ownerDocument;
    let controller;
    function node(tag, text, className = '') { const element = doc.createElement(tag); element.textContent = text; element.className = className; return element; }
    function date(value) { return new Date(value * 1000).toLocaleString('es-UY', { timeZone: 'America/Montevideo' }); }
    function render(state) {
      container.replaceChildren();
      const intro = node('section', '', 'guide');
      intro.append(node('h3', 'Recuerdos y propuestas'), node('p', 'La voz puede proponer un recuerdo. Sólo tu confirmación en este panel lo conserva sin vencimiento.'));
      intro.append(node('p', 'Los registros anteriores están en cuarentena hasta su revisión manual y no se usan en las respuestas.'));
      container.append(intro);
      if (state.error) { const error = node('p', state.error, 'connection-status warning'); error.setAttribute('role', 'alert'); container.append(error); }
      if (!state.ready) { if (!state.error) container.append(node('p', 'Consultando la memoria…')); return; }
      if (state.pending_index) container.append(node('p', 'Hay cambios pendientes de sincronización. Los recuerdos vencidos o borrados ya no se usan en las respuestas.', 'connection-status warning'));
      function section(title, items, proposed) {
        container.append(node('h3', title));
        if (!items.length) { container.append(node('p', proposed ? 'No hay propuestas pendientes.' : 'No hay recuerdos vigentes.')); return; }
        const list = node('div', '', 'room-grid');
        for (const item of items) {
          const card = node('article', '', 'room-card'); card.dataset.memoryId = item.id;
          card.append(node('p', item.content), node('p', item.state === 'confirmed' ? 'Confirmado manualmente · sin vencimiento' : (proposed ? 'Propuesta por revisar' : 'Hecho automático') + ' · vence ' + date(item.expires_at), 'badge'));
          if (item.confirmed_at) card.append(node('p', 'Confirmación: ' + date(item.confirmed_at), 'fine-print'));
          const details = doc.createElement('details'); details.append(node('summary', 'Ver procedencia'));
          for (const source of item.sources || []) details.append(node('p', (source.origin === 'ambient' ? 'Captura ambiental' : 'Interacción de voz') + ' · ' + date(source.captured_at), 'fine-print'));
          card.append(details);
          for (const [operation, label] of proposed ? [['confirm', 'Conservar sin vencimiento'], ['reject', 'Descartar propuesta']] : [['delete', 'Eliminar recuerdo']]) {
            const button = node('button', label, 'button'); button.type = 'button'; button.disabled = state.busy.includes(item.id) || state.loading;
            button.addEventListener('click', () => controller.act(item.id, operation)); card.append(button);
          }
          list.append(card);
        }
        container.append(list);
      }
      section('Propuestas pendientes', state.proposals, true); section('Recuerdos vigentes', state.facts, false);
    }
    controller = createMemoryController({ apiFetch, onChange: render });
    const visible = () => { if (!doc.hidden) controller.tick(); };
    doc.addEventListener('visibilitychange', visible); controller.refresh();
    return { refresh: controller.refresh, destroy() { doc.removeEventListener('visibilitychange', visible); controller.destroy(); container.replaceChildren(); } };
  }
  const exported = { createMemoryController, mountMemory };
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  else host.MemoryUI = exported;
})(typeof window === 'undefined' ? globalThis : window);
