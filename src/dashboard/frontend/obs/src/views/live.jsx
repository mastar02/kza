// =============================================================
// Live / Home view — KZA technical panel
// =============================================================

function ZoneCard({ z, userObserved, utteranceObserved, playbackObserved }) {
  const stateMap = {
    idle:      { kind: '',     label: 'idle' },
    listening: { kind: 'info', label: 'listening' },
    speaking:  { kind: 'accent', label: 'speaking' },
  };
  const s = stateMap[z.audioState] || { kind: '', label: z.audioState || 'desconocido' };
  return (
    <div className="card" style={{ position: 'relative' }}>
      <div className="card-h">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: 'var(--mono)', color: 'var(--ink-mute)' }} aria-hidden="true">{z.icon}</span>
          <h3>{z.name}</h3>
          <span className="mono mute" style={{ fontSize: 10 }}>zona {z.ma1260_zone ?? '—'}</span>
        </div>
        <Pill kind={s.kind}>
          {z.audioState === 'speaking' ? <Waveform live bars={6} height={10} /> : <span className="mono" aria-hidden="true">●</span>}
          {s.label}
        </Pill>
      </div>
      <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="mute mono" style={{ fontSize: 10, marginBottom: 2 }}>USUARIO</div>
            {z.user ? (
              <div>
                <div style={{ fontWeight: 500 }}>{z.user.name}</div>
                <div className="mono mute" style={{ fontSize: 10 }}>
                  {z.user.present ? (typeof z.user.ble_rssi === 'number' ? `BLE ${z.user.ble_rssi}dBm` : 'presente') : 'ausente'}
                </div>
              </div>
            ) : (
              <div className="mute">{userObserved ? 'nadie presente' : 'presencia no observada'}</div>
            )}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="mute mono" style={{ fontSize: 10, marginBottom: 2 }}>VOL</div>
            <div className="mono">{typeof z.volume === 'number' ? `${z.volume}%` : '—'}</div>
          </div>
        </div>

        <div className="wire" style={{ padding: 10, background: 'var(--bg-elev-2)' }}>
          <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>ÚLTIMA FRASE</div>
          {z.lastUtterance ? (
            <>
              <div style={{ fontStyle: 'italic' }}>"{z.lastUtterance.text}"</div>
              <div className="mono mute" style={{ fontSize: 10, marginTop: 4 }}>
                {z.lastUtterance.user} · {z.lastUtterance.ts}
              </div>
            </>
          ) : (
            <div className="mute">{utteranceObserved ? '— sin actividad reciente —' : 'no observada'}</div>
          )}
        </div>

        {z.spotify?.playing === true && (
          <div style={{ padding: 8, borderRadius: 4, border: '1px solid var(--line-soft)', minWidth: 0 }}>
            <div style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{z.spotify.song}</div>
            <div className="mute mono" style={{ fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{z.spotify.artist}</div>
          </div>
        )}
        {z.spotify?.playing !== true && !playbackObserved && (
          <div className="mute mono" style={{ fontSize: 10 }}>reproducción no observada</div>
        )}
      </div>
    </div>
  );
}

function RouterSummary() {
  const obs = useObs();
  const endpoints = obs.get('LLM_ENDPOINTS');
  if (!obs.available('LLM_ENDPOINTS')) return <Unavailable>Estado del router no disponible.</Unavailable>;
  if (endpoints.length === 0) return <Unavailable>Sin endpoints configurados.</Unavailable>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {endpoints.map(e => (
        <div key={e.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
          <span className="mono">{e.id}</span>
          {e.cooldown_ends || e.cooldown_until ? <Pill kind="warn">cooldown <Countdown until={e.cooldown_until} fallback={e.cooldown_ends} /></Pill>
            : e.state === 'healthy' ? <Pill kind="ok">saludable</Pill> : <Pill>sin verificar</Pill>}
        </div>
      ))}
    </div>
  );
}

function LiveTicker() {
  const container = React.useRef(null);
  React.useLayoutEffect(() => {
    const root = container.current;
    function render() {
      const store = window.__kzaLiveState;
      const frame = store?.snapshot();
      root.replaceChildren();
      root.hidden = !frame;
      if (frame) {
        const row = document.createElement('div');
        row.className = 'card-b';
        Object.assign(row.style, { display:'flex', gap:'14px', alignItems:'center' });
        const add = (text, className) => {
          const span = document.createElement('span');
          span.className = className;
          span.textContent = text;
          row.append(span);
        };
        const p = frame.payload;
        add(frame.type.toUpperCase(), 'pill accent');
        add(frame.ts, 'mono mute');
        if (p.zone) add(`zone=${p.zone}`, 'mono');
        if (p.user) add(`user=${p.user}`, 'mono');
        if (p.stt) add(`"${p.stt}"`, 'dim');
        if (typeof p.latency_ms === 'number') add(`${p.latency_ms}ms`, 'mono');
        root.append(row);
      }
      // React owns only this empty container, never private props or state.
      // ACK emission is deferred until this render stack has also returned.
      store?.committed(frame?.event_id || null);
    }
    render();
    window.addEventListener('kza:live', render);
    return () => {
      window.removeEventListener('kza:live', render);
      root.replaceChildren();
      window.__kzaLiveState?.committed(null);
    };
  }, []);
  return <div ref={container} data-live-ticker className="card" hidden
    style={{ marginBottom:12, borderColor:'var(--accent-line)', background:'var(--accent-soft)' }} />;
}

function LiveView() {
  const obs = useObs();
  const zones = obs.get('ZONES');
  return (
    <div className="view" data-active id="view-live">
      <PageHead
        title="Live"
        subtitle="Último evento en vivo · estado de zonas · router LLM"
        meta={<span>WebSocket {obs.channel('ws')}</span>}
      />

      <LiveTicker />

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Card title="Latencia fast path">
          <Unavailable>No disponible en este panel: no hay una fuente de latencia con procedencia verificable.</Unavailable>
        </Card>
        <Card title="Wake-word 24h">
          <Unavailable>No disponible: el servidor no publica conteos de activación.</Unavailable>
        </Card>
        <Card title="LLM router" meta={<SourceNote k="LLM_ENDPOINTS" />}>
          <RouterSummary />
        </Card>
      </div>

      <SectionBar title="Zonas" meta={<SourceNote k="ZONES" />} />
      {obs.available('ZONES') && zones.length > 0 ? (
        <div className="grid grid-3">
          {zones.map(z => <ZoneCard key={z.id} z={z}
            userObserved={!obs.unobserved('ZONES', 'user')}
            utteranceObserved={!obs.unobserved('ZONES', 'last_utterance')}
            playbackObserved={!obs.unobserved('ZONES', 'spotify_playback')} />)}
        </div>
      ) : (
        <Card><Unavailable>{obs.available('ZONES') ? 'No hay zonas configuradas.' : 'Zonas no disponibles: no se pudo leer el gestor de zonas.'}</Unavailable></Card>
      )}
    </div>
  );
}

window.LiveView = LiveView;
