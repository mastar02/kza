// =============================================================
// App shell — KZA technical panel (direct URL /obs/)
// =============================================================

const { useState, useEffect } = React;

const NAV = [
  { id: 'live',          label: 'Live',           group: 'Operación' },
  { id: 'conversations', label: 'Conversaciones', group: 'Operación' },
  { id: 'privacy',       label: 'Privacidad',     group: 'Operación' },
  { id: 'alerts',        label: 'Alertas',        group: 'Operación' },
  { id: 'ha',            label: 'Home Assistant', group: 'Plataforma' },
  { id: 'llm',           label: 'LLM Router',     group: 'Plataforma' },
  { id: 'users',         label: 'Voces',          group: 'Plataforma' },
  { id: 'system',        label: 'Sistema',        group: 'Infraestructura' },
  { id: 'admin',         label: 'Admin',          group: 'Infraestructura' },
];

function navBadge(obs, id) {
  if (id !== 'alerts' || !obs.available('ALERTS')) return null;
  const active = obs.get('ALERTS').filter(alert => !alert.acked).length;
  return active ? String(active) : null;
}

function Sidebar({ active, onNav }) {
  const obs = useObs();
  const groups = [...new Set(NAV.map(n => n.group))];
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">K</div>
        <div>
          <div className="brand-name">KZA</div>
          <div className="brand-sub">panel técnico</div>
        </div>
      </div>
      <nav className="nav" aria-label="Secciones del panel técnico">
        {groups.map(g => (
          <React.Fragment key={g}>
            <div className="nav-section">{g}</div>
            {NAV.filter(n => n.group === g).map(n => {
              const badge = navBadge(obs, n.id);
              return (
                <button type="button" key={n.id}
                     className={`nav-item ${active === n.id ? 'active' : ''}`}
                     aria-current={active === n.id ? 'page' : undefined}
                     onClick={() => onNav(n.id)}>
                  <span className="dot" />
                  <span>{n.label}</span>
                  {badge && <span className="badge" aria-label={`${badge} activas`}>{badge}</span>}
                </button>
              );
            })}
          </React.Fragment>
        ))}
      </nav>
      <div className="sidebar-foot">
        <div className="row"><span>host</span><b style={{ color: 'var(--ink)' }}>{window.location.host}</b></div>
        <div className="row"><span>uptime</span><b className="mute">no disponible</b></div>
        <div className="row"><span>build</span><b className="mute">no disponible</b></div>
      </div>
    </aside>
  );
}

function Topbar({ activeNav, onRefresh, refreshing }) {
  const obs = useObs();
  const item = NAV.find(n => n.id === activeNav);
  const summary = obs.summary();
  const healthy = summary.degraded.length === 0 && summary.pending.length === 0;
  const ws = obs.channel('ws');
  return (
    <header className="topbar">
      <div className="crumbs">kza · <b>{item.label}</b></div>
      <div className="topbar-spacer" />
      <div className="pulse-group">
        <div className={`pulse ${healthy ? '' : 'warn'}`}><i className="led" /> datos · {summary.real.length}/{summary.total} reales</div>
        <div className={`pulse ${ws === 'conectado' ? '' : 'warn'}`}><i className="led" /> WS · {ws}</div>
      </div>
      <button type="button" className="btn" disabled={refreshing} onClick={onRefresh}>{refreshing ? 'Actualizando…' : 'Actualizar datos'}</button>
    </header>
  );
}

function SourceBanner() {
  const obs = useObs();
  const summary = obs.summary();
  const label = key => window.KzaObsData.COLLECTIONS.find(([name]) => name === key)?.[2] || key;
  if (summary.demo.length) {
    return (
      <section role="status" className="source-banner warn">
        <strong>Modo demostración</strong>
        <div>El servidor está configurado con datos de ejemplo (dashboard.observability_use_mocks). No describen el sistema real: {summary.demo.map(label).join(', ')}.</div>
      </section>
    );
  }
  if (summary.degraded.length) {
    return (
      <section role="status" className="source-banner warn">
        <strong>Fuentes no disponibles</strong>
        <div>No se pudieron leer: {summary.degraded.map(label).join(', ')}. Esas secciones se muestran como no disponibles, sin datos de reemplazo.</div>
      </section>
    );
  }
  return null;
}

function App() {
  const [active, setActive] = useState(() => {
    const requested = window.location.hash.slice(1);
    return NAV.some(item => item.id === requested) ? requested : 'live';
  });
  const [refreshing, setRefreshing] = useState(false);
  async function refresh() {
    setRefreshing(true);
    try {
      await window.KzaObsData.hydrate({ fetch: window.fetch.bind(window), store: window.__kzaObs });
    } catch (error) {
      if (error instanceof window.KzaObsData.AuthRequired) window.dispatchEvent(new CustomEvent('kza:auth-required'));
    } finally {
      setRefreshing(false);
    }
  }

  const Views = {
    live:          window.LiveView,
    conversations: window.ConversationsView,
    privacy:       window.PrivacyView,
    ha:            window.HAView,
    llm:           window.LLMView,
    users:         window.UsersView,
    alerts:        window.AlertsView,
    system:        window.SystemView,
    admin:         window.AdminView,
  };
  const ViewComp = Views[active];

  return (
    <div className="app" data-screen-label={`${NAV.findIndex(n=>n.id===active)+1} ${NAV.find(n=>n.id===active).label}`}>
      <Sidebar active={active} onNav={id => { window.__kzaObs.setNotice(null); setActive(id); }} />
      <main className="main">
        <Topbar activeNav={active} onRefresh={refresh} refreshing={refreshing} />
        <SourceBanner />
        <OperationNotice />
        <div className="content">
          {ViewComp && <ViewComp />}
        </div>
      </main>
    </div>
  );
}

// =============================================================
// Bootstrap: read /api/* before mounting React. A failed read leaves the
// collection empty and marked unavailable; there are no local examples.
// =============================================================

window.__kzaObs = window.KzaObsData.createObservationStore();
window.__kzaSources = window.__kzaObs.sources;
window.__kzaConversationState = window.KzaConversationPrivacy.createConversationState();
window.__kzaVoiceState = window.KzaVoicePrivacy.createVoiceState();

function clearPrivateData() {
  window.__kzaLiveState?.clear();
  window.__kzaConversationState.clear();
  window.__kzaVoiceState.clear();
  window.__kzaObs.clear();
}

async function hydrateObservations({ signal }) {
  try {
    await window.KzaObsData.hydrate({ fetch: window.fetch.bind(window), store: window.__kzaObs, signal });
  } catch (error) {
    if (error instanceof window.KzaObsData.AuthRequired) throw new window.KzaDashboardSession.SessionRequired();
    throw error;
  }
}

function mountWsLiveBridge() {
  let retired = false, retry = null, socket = null;
  function emitAck(message) {
    if (!socket || socket.readyState !== 1) return false;
    socket.send(JSON.stringify(message)); return true;
  }
  const conversations = window.KzaConversationChannel.createConversationChannel({
    store:window.__kzaConversationState, fetch:window.fetch.bind(window), emitAck,
    hasView:() => !!document.querySelector('#view-conversations'),
    onUnauthorized:() => window.dispatchEvent(new CustomEvent('kza:auth-required')),
    onState:state => { window.__kzaObs.setSource('CONVERSATIONS', state === 'real' ? 'real' :
      ({ waiting:'sin confirmar', loading:'consultando', unavailable:'degradada', closed:'sin sesión' })[state]); },
  });
  const voices = window.KzaVoiceChannel.createVoiceController({ state:window.__kzaVoiceState,
    fetch:window.fetch.bind(window), emitAck, hasView:() => !!document.querySelector('#view-users'),
    onUnauthorized:() => window.dispatchEvent(new CustomEvent('kza:auth-required')),
    onStatus:status => { window.__kzaObs.setSource('VOICES', status); },
  });
  window.__kzaVoiceController = voices;
  const live = window.KzaLivePrivacy.createLiveState({
    emitAck,
    onChange: eventId => {
      window.dispatchEvent(new CustomEvent('kza:live', { detail:{ event_id:eventId } }));
      if (!document.querySelector('[data-live-ticker]')) live.committed(null);
    },
  });
  window.__kzaLiveState = live;
  const checkExpiry = () => live.expire();
  window.addEventListener('focus', checkExpiry);
  document.addEventListener('visibilitychange', checkExpiry);
  function connect() {
    if (retired) return;
    try {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/live`);
    socket = ws;
    window.__kzaObs.setChannel('ws', 'conectando');
    ws.onopen = () => {
      if (!retired && socket === ws) window.__kzaObs.setChannel('ws', 'conectado');
      if (!retired && socket === ws) ws.send(JSON.stringify({ type:'live_hello', version:3, client_id:window.KzaLivePrivacy.pageId() }));
    };
    ws.onmessage = (ev) => {
      if (socket !== ws || retired) return;
      try {
        const frame = JSON.parse(ev.data);
        if (frame.type === 'live_reset') {
          live.clear();
          Promise.all([conversations.reset(),voices.reset()]).then(cleared => {
            if (cleared.every(Boolean) && socket === ws && !retired) live.receive(frame);
          });
        } else if (frame.type === 'rest_ready') { conversations.open(frame.connection_id); voices.open(frame.connection_id); }
        else if (frame.type === 'rest_retire') {
          if (frame.channel?.startsWith('voices:')) voices.retire(frame);
          else conversations.retire(frame);
        }
        else live.receive(frame);
      } catch {}
    };
    ws.onclose = async (event) => {
      if (socket !== ws) return;
      window.__kzaObs.setChannel('ws', retired ? 'cerrado' : 'desconectado');
      live.clear();
      conversations.reset();
      voices.reset();
      if (retired) return;
      let needsLogin = event.code === 1008;
      try {
        const response = await fetch('/api/admin/auth/whoami', { credentials: 'same-origin' });
        needsLogin = needsLogin || response.status === 401 || response.status === 503;
      } catch {}
      if (retired) return;
      if (needsLogin) window.dispatchEvent(new CustomEvent('kza:auth-required'));
      else retry = setTimeout(connect, 3000);
    };
    window.__kzaLiveWs = ws;
  } catch (e) {
    console.warn('[ws] could not connect:', e);
  }
  }
  connect();
  return () => {
    retired = true; clearTimeout(retry);
    window.removeEventListener('focus', checkExpiry);
    document.removeEventListener('visibilitychange', checkExpiry);
    live.dispose();
    conversations.dispose();
    voices.dispose();
    queueMicrotask(() => socket?.close());
    if (window.__kzaLiveState === live) window.__kzaLiveState = null;
    if (window.__kzaLiveWs === socket) window.__kzaLiveWs = null;
  };
}

const appRoot = ReactDOM.createRoot(document.getElementById('root'));
const session = window.KzaDashboardSession.createSession({
  fetch: window.fetch.bind(window), hydrate: hydrateObservations,
  connect: mountWsLiveBridge, clear: clearPrivateData,
  onState(state) {
    if (state === 'ready') appRoot.render(<App />);
    else if (state === 'login') appRoot.render(<LoginModal onLogin={() => session.start()} />);
    else if (state === 'unavailable') appRoot.render(<div className="content" role="alert">El acceso no está disponible. <button className="btn" onClick={() => session.start()}>Reintentar</button></div>);
    else appRoot.render(<div className="content">Cargando…</div>);
  },
});
window.addEventListener('kza:auth-required', () => session.expire());
window.addEventListener('pagehide', () => session.stop());
session.start();
