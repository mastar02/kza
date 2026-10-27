(function (root) {
  'use strict';
  const TEXT = ['id', 'ts', 'user', 'zone', 'path', 'stt', 'tts', 'intent', 'target'];
  function createConversationState({ monotonic = () => performance.now(), wall = Date.now,
      schedule = setTimeout, cancel = clearTimeout } = {}) {
    let entries = [], timer = null, revision = 0, disposed = false, privacy = null;
    const listeners = new Set();
    function changed() { revision++; for (const listener of listeners) listener(revision); }
    function prune() {
      const mono = monotonic(), clock = wall();
      const remaining = entries.filter(entry => mono < entry.monoDeadline && clock < entry.wallDeadline);
      if (remaining.length !== entries.length) { entries = remaining; changed(); }
    }
    function arm() {
      if (timer !== null) cancel(timer);
      timer = null;
      if (!entries.length) return;
      const delay = Math.min(...entries.map(entry => Math.min(entry.monoDeadline - monotonic(), entry.wallDeadline - wall())));
      timer = schedule(() => { timer = null; prune(); arm(); }, Math.max(1, Math.min(delay, 2147483647)));
    }
    function clear() { entries = []; if (timer !== null) cancel(timer); timer = null; changed(); }
    function receive(rows, started) {
      if (disposed) return false;
      // Include the entire request duration: delay must consume, never renew, TTL.
      const elapsed = started && Math.max(0, monotonic() - started.mono, wall() - started.wall);
      if (!Array.isArray(rows) || rows.length > 50 || !Number.isFinite(elapsed)) { clear(); return false; }
      const previous = new Map(entries.map(entry => [entry.row.id, entry]));
      const next = [], ids = new Set();
      for (const value of rows) {
        if (!value || !TEXT.every(key => typeof value[key] === 'string' && value[key].length <= (key === 'stt' || key === 'tts' ? 8192 : 512))
            || !Number.isFinite(value.observed_at) || !Number.isFinite(value.expires_at)
            || !Number.isFinite(value.expires_at * 1000) || value.expires_at <= value.observed_at
            || !/^(turn_)[0-9]+_[0-9a-f-]{36}$/.test(value.id) || ids.has(value.id)) { clear(); return false; }
        const budget = (value.expires_at - value.observed_at) * 1000 - elapsed;
        if (budget <= 0 || value.expires_at * 1000 <= wall()) continue;
        const old = previous.get(value.id), row = {};
        for (const key of TEXT) row[key] = value[key];
        row.latency_ms = Number.isFinite(value.latency_ms) && value.latency_ms >= 0 ? value.latency_ms : null;
        row.success = typeof value.success === 'boolean' ? value.success : null;
        row.expires_at = value.expires_at;
        next.push({ row: Object.freeze(row),
          monoDeadline: Math.min(monotonic() + budget, old?.monoDeadline ?? Infinity),
          wallDeadline: Math.min(value.expires_at * 1000, wall() + budget, old?.wallDeadline ?? Infinity) });
        ids.add(value.id);
      }
      entries = next; prune(); changed(); arm(); return true;
    }
    return {
      receive, clear, check() { prune(); arm(); },
      snapshot() { prune(); return entries.map(entry => entry.row); },
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      bindPrivacy(callbacks) { privacy = callbacks; },
      committed(token) { privacy?.committed(token); },
      deliveryToken() { return privacy?.token() ?? null; },
      dispose() { clear(); disposed = true; listeners.clear(); },
    };
  }

  function mountConversations(container, store) {
    const doc = container.ownerDocument;
    let selected = null, user = 'all', path = 'all';
    function node(tag, text = '', className = '') {
      const element = doc.createElement(tag); element.textContent = text; element.className = className; return element;
    }
    function render() {
      const rows = store.snapshot();
      if (!rows.some(row => row.user === user)) user = 'all';
      if (!rows.some(row => row.path === path)) path = 'all';
      container.replaceChildren();
      const header = node('div', '', 'page-head');
      header.append(node('h1', 'Conversaciones'), node('p', 'Turnos vigentes · se retiran al vencer la captura original.', 'mute'));
      container.append(header);
      if (!rows.length) {
        selected = null; container.append(node('p', 'No hay turnos vigentes disponibles.'));
        store.committed(null); return;
      }
      const filters = node('div', '', 'card');
      function selectFilter(values, current, change, label) {
        const select = doc.createElement('select'); select.className = 'btn'; select.setAttribute('aria-label', label);
        for (const value of values) { const option = node('option', value === 'all' ? label : value); option.value = value; select.append(option); }
        select.value = current;
        select.addEventListener('change', () => { change(select.value); render(); }); filters.append(select);
      }
      selectFilter(['all', ...new Set(rows.map(row => row.user))], user, value => { user = value; }, 'Todas las voces');
      selectFilter(['all', ...new Set(rows.map(row => row.path))], path, value => { path = value; }, 'Todos los recorridos');
      container.append(filters);
      const filtered = rows.filter(row => (user === 'all' || row.user === user) && (path === 'all' || row.path === path));
      if (!filtered.some(row => row.id === selected)) selected = filtered[0]?.id || null;
      const table = node('table', '', 'k');
      const head = doc.createElement('thead'), headings = doc.createElement('tr');
      for (const title of ['Hora', 'Voz', 'Zona', 'Recorrido', 'Frase', 'Latencia']) headings.append(node('th', title));
      head.append(headings); table.append(head);
      const body = doc.createElement('tbody');
      for (const row of filtered) {
        const id = row.id, tr = doc.createElement('tr');
        const button = node('button', row.stt, 'btn'); button.type = 'button';
        button.addEventListener('click', () => { selected = id; render(); });
        for (const value of [row.ts, row.user, row.zone, row.path]) tr.append(node('td', value));
        const phrase = doc.createElement('td'); phrase.append(button); tr.append(phrase, node('td', row.latency_ms === null ? 'Sin medición' : `${row.latency_ms} ms`)); body.append(tr);
      }
      table.append(body); container.append(table);
      const detail = filtered.find(row => row.id === selected);
      if (detail) {
        const card = node('section', '', 'card'); card.setAttribute('aria-label', 'Detalle del turno');
        card.append(node('h2', 'Detalle del turno'));
        for (const [title, text] of [['Transcripción', detail.stt], ['Respuesta', detail.tts || 'Sin respuesta registrada'],
          ['Intención', detail.intent], ['Destino', detail.target],
          ['Resultado registrado', detail.success === true ? 'Correcto' : detail.success === false ? 'Fallido' : 'Sin confirmar']]) {
          card.append(node('h3', title), node('p', text));
        }
        container.append(card);
      }
      store.committed(store.deliveryToken());
    }
    // Notifications carry only a revision. Private rows live in this stack and DOM.
    const unsubscribe = store.subscribe(render);
    const check = () => store.check();
    doc.addEventListener('visibilitychange', check); root?.addEventListener('focus', check);
    render();
    return () => {
      unsubscribe(); doc.removeEventListener('visibilitychange', check); root?.removeEventListener('focus', check);
      selected = null; user = 'all'; path = 'all'; container.replaceChildren();
      store.committed(null);
    };
  }
  const api = { createConversationState, mountConversations };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaConversationPrivacy = api;
})(typeof window === 'undefined' ? null : window);
