// =============================================================
// App shell — KZA technical panel (direct URL /obs/)
// =============================================================

const { useState, useEffect, useRef } = React;

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
const NAV_IDS = NAV.map(n => n.id);
const DEFAULT_VIEW = 'live';

function navBadge(obs, id) {
  if (id !== 'alerts' || !obs.available('ALERTS')) return null;
  const active = obs.get('ALERTS').filter(alert => !alert.acked).length;
  return active ? String(active) : null;
}

function Sidebar({ active, onNav, menuOpen, menuButtonRef, onToggleMenu }) {
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
        <button type="button" className="btn menu-toggle" ref={menuButtonRef}
                aria-expanded={menuOpen} aria-controls="obs-nav" onClick={onToggleMenu}>
          Menú
        </button>
      </div>
      <nav id="obs-nav" className="nav" data-open={menuOpen} aria-label="Secciones del panel técnico">
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
  const healthy = !summary.degraded.length && !summary.forbidden.length && !summary.pending.length;
  const ws = obs.channel('ws');
  return (
    <header className="topbar">
      <div className="crumbs">kza · <b>{item.label}</b></div>
      <div className="topbar-spacer" />
      <div className="pulse-group">
        <div className={`pulse ${healthy ? '' : 'warn'}`}><i className="led" /> datos REST · {summary.real.length}/{summary.total} reales</div>
        <div className={`pulse ${ws === 'conectado' ? '' : 'warn'}`}
             title="WebSocket /ws/live solo para eventos en vivo, conversaciones y voces; los inventarios REST se renuevan por vencimiento">
          <i className="led" /> WS en vivo (live/conv/voces) · {ws}
        </div>
      </div>
      <button type="button" className="btn" disabled={refreshing} onClick={onRefresh}>{refreshing ? 'Actualizando…' : 'Actualizar datos'}</button>
    </header>
  );
}

function SourceBanner() {
  const obs = useObs();
  const summary = obs.summary();
  const label = key => window.KzaObsData.COLLECTIONS.find(([name]) => name === key)?.[2] || key;
  const degraded = summary.degraded.map(key => {
    const { reason } = obs.entry(key);
    return reason ? `${label(key)} (${reason})` : label(key);
  });
  return (
    <>
      {!!summary.demo.length && <section role="status" className="source-banner warn">
        <strong>Modo demostración</strong>
        <div>El servidor está configurado con datos de ejemplo (dashboard.observability_use_mocks). No describen el sistema real: {summary.demo.map(label).join(', ')}.</div>
      </section>}
      {!!degraded.length && <section role="status" className="source-banner warn">
        <strong>Fuentes no disponibles</strong>
        <div>No se pudieron leer: {degraded.join(', ')}. Esas secciones se muestran como no disponibles, sin datos de reemplazo.</div>
      </section>}
      {!!summary.forbidden.length && <section role="status" className="source-banner warn">
        <strong>Fuentes sin permiso</strong>
        <div>{summary.forbidden.map(label).join(', ')}.</div>
      </section>}
    </>
  );
}

function App({ afterLogin, sessionSignal }) {
  const nav = window.KzaObsNav;
  const [active, setActive] = useState(() => nav.viewFromHash(window.location.hash, NAV_IDS, DEFAULT_VIEW));
  const [announcement, setAnnouncement] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const activeRef = useRef(active);
  const mainRef = useRef(null);
  const refreshRequest = useRef(null);
  const menuButtonRef = useRef(null);
  const focusView = useRef(false);

  function closeMenu() {
    setMenuOpen(false);
    menuButtonRef.current?.focus();
  }

  useEffect(() => {
    if (!menuOpen) return;
    const desktop = window.matchMedia('(min-width: 768px)');
    function onEscape(event) {
      if (event.key !== 'Escape' || event.defaultPrevented || desktop.matches) return;
      event.preventDefault();
      closeMenu();
    }
    function onBreakpoint(event) {
      if (event.matches) setMenuOpen(false);
    }
    window.addEventListener('keydown', onEscape);
    desktop.addEventListener('change', onBreakpoint);
    return () => {
      window.removeEventListener('keydown', onEscape);
      desktop.removeEventListener('change', onBreakpoint);
    };
  }, [menuOpen]);

  // A view change by navigation (not the first render) clears the operation
  // notice, announces the view and moves focus to the view region.
  function showView(id) {
    activeRef.current = id;
    focusView.current = true;
    setMenuOpen(false);
    window.__kzaObs.setNotice(null);
    setAnnouncement(nav.announcement(NAV.find(n => n.id === id).label));
    setActive(id);
  }
  function onNav(id) {
    const step = nav.planNavigation(activeRef.current, id, window.location.hash, NAV_IDS);
    if (step.show) showView(id);
    else {
      window.__kzaObs.setNotice(null);
      if (menuOpen) {
        setMenuOpen(false);
        mainRef.current?.focus();
      }
    }
    if (step.write) window.location.hash = step.write;
  }
  useEffect(() => {
    function onHashChange() {
      const step = nav.planHashChange(activeRef.current, window.location.hash, NAV_IDS, DEFAULT_VIEW);
      if (step.changed) showView(step.view);
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
  useEffect(() => {
    if (!focusView.current) return;
    focusView.current = false;
    mainRef.current?.focus();
  }, [active]);
  const [refreshing, setRefreshing] = useState(false);
  const pollerRef = useRef(null);
  useEffect(() => {
    const poller = window.KzaObsData.createObservationPoller({
      fetch: window.fetch.bind(window),
      store: window.__kzaObs,
      view: activeRef.current,
      signal: sessionSignal,
      onUnauthorized: failure => {
        if (!sessionSignal.aborted) session.fail(failure);
      },
    });
    pollerRef.current = poller;
    const onVisibility = () => poller.sync();
    const onAuthRequired = () => poller.stop();
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('kza:auth-required', onAuthRequired);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('kza:auth-required', onAuthRequired);
      poller.stop();
      if (pollerRef.current === poller) pollerRef.current = null;
    };
  }, [sessionSignal]);
  useEffect(() => {
    pollerRef.current?.setView(active);
  }, [active]);
  React.useLayoutEffect(() => {
    if (afterLogin) mainRef.current?.focus();
    return () => refreshRequest.current?.abort();
  }, [afterLogin]);
  async function refresh() {
    if (refreshRequest.current) return;
    const controller = new AbortController();
    refreshRequest.current = controller;
    setRefreshing(true);
    try {
      await window.KzaObsData.hydrate({
        fetch: window.fetch.bind(window), store: window.__kzaObs,
        signal: AbortSignal.any([sessionSignal, controller.signal]),
      });
    } catch (failure) {
      if (!sessionSignal.aborted && !controller.signal.aborted) {
        controller.abort();
        session.fail(failure);
      }
    } finally {
      refreshRequest.current = null;
      if (!controller.signal.aborted) setRefreshing(false);
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
      <Sidebar active={active} onNav={onNav} menuOpen={menuOpen} menuButtonRef={menuButtonRef}
               onToggleMenu={() => menuOpen ? closeMenu() : setMenuOpen(true)} />
      <main className="main" ref={mainRef} tabIndex={-1} aria-label={NAV.find(n => n.id === active).label}>
        <Topbar activeNav={active} onRefresh={refresh} refreshing={refreshing} />
        <SourceBanner />
        <OperationNotice />
        <div className="content">
          {ViewComp && <ViewComp />}
        </div>
      </main>
      <p className="sr-only" role="status" aria-live="polite">{announcement}</p>
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
  window.KzaCaptureRecorder.stopAll();
  window.__kzaLiveState?.clear();
  window.__kzaConversationState.clear();
  window.__kzaVoiceState.clear();
  window.__kzaObs.clear();
}

function hydrateObservations({ signal }) {
  return window.KzaObsData.hydrate({ fetch: window.fetch.bind(window), store: window.__kzaObs, signal });
}

function mountWsLiveBridge() {
  let retired = false, retry = null, socket = null, closeCheck = null;
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
    canSearch:() => window.__kzaObs.can('admin:manage'),
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
      if (socket !== ws || retired) return;
      window.__kzaObs.setChannel('ws', 'desconectado');
      live.clear();
      conversations.reset();
      voices.reset();
      closeCheck = new AbortController();
      const { failure, message, retry: shouldRetry } = await window.KzaDashboardSession.liveCloseStatus(
        window.fetch.bind(window), event.code, closeCheck.signal,
      );
      if (retired || socket !== ws) return;
      closeCheck = null;
      if (failure) {
        session.fail(failure);
        return;
      }
      window.__kzaObs.setChannel('ws', message);
      if (shouldRetry) retry = setTimeout(connect, 3000);
    };
    window.__kzaLiveWs = ws;
  } catch (e) {
    console.warn('[ws] could not connect:', e);
  }
  }
  connect();
  return () => {
    retired = true; clearTimeout(retry);
    closeCheck?.abort();
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
let afterLogin = false;
const session = window.KzaDashboardSession.createSession({
  fetch: window.fetch.bind(window), hydrate: hydrateObservations,
  connect: mountWsLiveBridge, clear: clearPrivateData,
  onState(state, failure, sessionSignal) {
    if (state === 'ready') {
      appRoot.render(<App afterLogin={afterLogin} sessionSignal={sessionSignal} />);
      afterLogin = false;
    } else if (state === 'login') {
      afterLogin = false;
      appRoot.render(<LoginGate key={failure?.kind ?? 'signed-out'} failure={failure} onLogin={() => {
        afterLogin = true;
        return session.start();
      }} />);
    } else if (state === 'unavailable' || state === 'forbidden') {
      appRoot.render(<AccessNotice failure={failure} onRetry={() => session.start()} />);
    } else appRoot.render(<div className="content">Cargando…</div>);
  },
});
window.addEventListener('kza:auth-required', event => {
  if (event.detail) session.fail(event.detail);
  else session.expire();
});
window.addEventListener('pagehide', () => session.stop());
window.addEventListener('pageshow', event => { if (event.persisted) session.start(); });
session.start();
