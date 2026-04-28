// =============================================================
// App shell — KZA dashboard
// =============================================================

const { useState, useEffect } = React;

const NAV = [
  { id: 'live',          label: 'Live',           group: 'Operación', icon: '◉', view: 'LiveView',          badge: 'WS' },
  { id: 'conversations', label: 'Conversaciones', group: 'Operación', icon: '◰', view: 'ConversationsView' },
  { id: 'alerts',        label: 'Alertas',        group: 'Operación', icon: '◬', view: 'AlertsView',        badge: '3' },
  { id: 'ha',            label: 'Home Assistant', group: 'Plataforma', icon: '◧', view: 'HAView' },
  { id: 'llm',           label: 'LLM Router',     group: 'Plataforma', icon: '◆', view: 'LLMView',           badge: '!' },
  { id: 'users',         label: 'Usuarios',       group: 'Plataforma', icon: '◐', view: 'UsersView' },
  { id: 'system',        label: 'Sistema',        group: 'Infraestructura', icon: '▦', view: 'SystemView' },
];

const ACCENTS = {
  violet: { color: '#7c3aed', soft: 'rgba(124,58,237,0.16)', line: 'rgba(124,58,237,0.4)' },
  cyan:   { color: '#06b6d4', soft: 'rgba(6,182,212,0.16)',  line: 'rgba(6,182,212,0.4)' },
  lime:   { color: '#84cc16', soft: 'rgba(132,204,22,0.16)', line: 'rgba(132,204,22,0.4)' },
  orange: { color: '#f97316', soft: 'rgba(249,115,22,0.16)', line: 'rgba(249,115,22,0.4)' },
};

function Sidebar({ active, onNav }) {
  const groups = [...new Set(NAV.map(n => n.group))];
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">K</div>
        <div>
          <div className="brand-name">KZA</div>
          <div className="brand-sub">v0.4 · on-prem</div>
        </div>
      </div>
      <nav className="nav">
        {groups.map(g => (
          <React.Fragment key={g}>
            <div className="nav-section">{g}</div>
            {NAV.filter(n => n.group === g).map(n => (
              <div key={n.id}
                   className={`nav-item ${active === n.id ? 'active' : ''}`}
                   onClick={() => onNav(n.id)}>
                <span className="dot" />
                <span>{n.label}</span>
                {n.badge && <span className="badge">{n.badge}</span>}
              </div>
            ))}
          </React.Fragment>
        ))}
      </nav>
      <div className="sidebar-foot">
        <div className="row"><span>host</span><b style={{ color: 'var(--ink)' }}>192.168.1.2</b></div>
        <div className="row"><span>uptime</span><b style={{ color: 'var(--ink)' }}>21d 08h</b></div>
        <div className="row"><span>build</span><b style={{ color: 'var(--ink)' }}>kza@a3f9c2e</b></div>
      </div>
    </aside>
  );
}

function Topbar({ activeNav }) {
  const item = NAV.find(n => n.id === activeNav);
  return (
    <header className="topbar">
      <div className="crumbs">kza · <b>{item.label}</b></div>
      <div className="topbar-spacer" />
      <input type="search" placeholder="buscar entidades, turns, users… (⌘K)" />
      <div className="pulse-group">
        <div className="pulse"><i className="led" /> sistema</div>
        <div className="pulse warn"><i className="led" /> 1 cooldown</div>
        <div className="pulse"><i className="led" /> WS live</div>
      </div>
      <span className="kbd">⌘K</span>
    </header>
  );
}

function App() {
  const [active, setActive] = useState('live');
  const tweaks = window.useTweaks ? window.useTweaks({
    accent: 'violet',
    density: 'balanced',
    homeLayout: 'zones-grid',
    showAnnotations: true,
  }) : { tweaks: { accent: 'violet', density: 'balanced', homeLayout: 'zones-grid', showAnnotations: true }, setTweak: () => {} };
  const t = tweaks.tweaks || tweaks;

  // Apply density + accent
  useEffect(() => {
    document.documentElement.setAttribute('data-density', t.density);
    const a = ACCENTS[t.accent] || ACCENTS.violet;
    document.documentElement.style.setProperty('--accent', a.color);
    document.documentElement.style.setProperty('--accent-soft', a.soft);
    document.documentElement.style.setProperty('--accent-line', a.line);
    window.__showAnnotations = t.showAnnotations;
  }, [t.density, t.accent, t.showAnnotations]);

  const Views = {
    live:          window.LiveView,
    conversations: window.ConversationsView,
    ha:            window.HAView,
    llm:           window.LLMView,
    users:         window.UsersView,
    alerts:        window.AlertsView,
    system:        window.SystemView,
  };
  const ViewComp = Views[active];

  return (
    <div className="app" data-screen-label={`${NAV.findIndex(n=>n.id===active)+1} ${NAV.find(n=>n.id===active).label}`}>
      <Sidebar active={active} onNav={setActive} />
      <main className="main">
        <Topbar activeNav={active} />
        <div className="content">
          {ViewComp && <ViewComp homeLayout={t.homeLayout} />}
        </div>
      </main>

      {window.TweaksPanel && (
        <window.TweaksPanel title="Tweaks">
          <window.TweakSection title="Apariencia">
            <window.TweakRadio
              label="Acento"
              value={t.accent}
              onChange={v => tweaks.setTweak('accent', v)}
              options={[
                { value: 'violet', label: 'Violeta' },
                { value: 'cyan',   label: 'Cyan' },
                { value: 'lime',   label: 'Lime' },
                { value: 'orange', label: 'Naranja' },
              ]}
            />
            <window.TweakRadio
              label="Densidad"
              value={t.density}
              onChange={v => tweaks.setTweak('density', v)}
              options={[
                { value: 'cozy',     label: 'Cozy' },
                { value: 'balanced', label: 'Balanced' },
                { value: 'dense',    label: 'Dense' },
              ]}
            />
          </window.TweakSection>
          <window.TweakSection title="Wireframe">
            <window.TweakToggle
              label="Anotaciones a mano"
              value={t.showAnnotations}
              onChange={v => tweaks.setTweak('showAnnotations', v)}
            />
          </window.TweakSection>
        </window.TweaksPanel>
      )}
    </div>
  );
}

// =============================================================
// Bootstrap: fetch real data desde /api/* antes de montar React.
// Si un endpoint falla, se conserva el mock literal de mocks.jsx (fallback).
// =============================================================

const API_BASE = window.location.origin;

const ENDPOINT_MAP = [
  ['ZONES',         '/api/zones'],
  ['LLM_ENDPOINTS', '/api/llm/endpoints'],
  ['CONVERSATIONS', '/api/conversations'],
  ['HA_ENTITIES',   '/api/ha/entities'],
  ['HA_ACTIONS',    '/api/ha/actions'],
  ['USERS',         '/api/users'],
  ['ALERTS',        '/api/alerts'],
  ['GPUS',          '/api/system/gpus'],
  ['SERVICES',      '/api/system/services'],
];

async function hydrateMocks() {
  await Promise.all(ENDPOINT_MAP.map(async ([key, path]) => {
    try {
      const r = await fetch(API_BASE + path);
      if (!r.ok) return;
      const data = await r.json();
      if (Array.isArray(data) ? data.length > 0 : !!data) {
        window.MOCKS[key] = data;
      }
    } catch (e) {
      console.warn('[bootstrap] endpoint failed, keeping mock:', path, e);
    }
  }));
}

function mountWsLiveBridge() {
  try {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/live`);
    ws.onmessage = (ev) => {
      try {
        const frame = JSON.parse(ev.data);
        window.dispatchEvent(new CustomEvent('kza:live', { detail: frame }));
      } catch {}
    };
    ws.onclose = () => setTimeout(mountWsLiveBridge, 3000);
    window.__kzaLiveWs = ws;
  } catch (e) {
    console.warn('[ws] could not connect:', e);
  }
}

(async () => {
  await hydrateMocks();
  mountWsLiveBridge();
  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
})();
