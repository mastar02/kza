// =============================================================
// Live / Home view — KZA dashboard
// =============================================================

const { ZONES, LATENCY_FAST, LATENCY_BREAKDOWN, WAKE_EVENTS_24H, LLM_ENDPOINTS, PIPELINE_STAGES } = window.MOCKS;

function ZoneCard({ z }) {
  const stateMap = {
    idle:      { kind: '',     label: 'idle' },
    listening: { kind: 'info', label: 'listening' },
    speaking:  { kind: 'accent', label: 'speaking' },
  };
  const s = stateMap[z.audioState];
  return (
    <div className="card" style={{ position: 'relative' }}>
      <div className="card-h">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: 'var(--mono)', color: 'var(--ink-mute)' }}>{z.icon}</span>
          <h3>{z.name}</h3>
          <span className="mono mute" style={{ fontSize: 10 }}>zona {z.ma1260_zone}</span>
        </div>
        <Pill kind={s.kind}>
          {z.audioState === 'speaking' ? <Waveform live bars={6} height={10} /> : <span className="mono">●</span>}
          {s.label}
        </Pill>
      </div>
      <div className="card-b" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="mute mono" style={{ fontSize: 10, marginBottom: 2 }}>USUARIO</div>
            {z.user ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  width: 22, height: 22, borderRadius: '50%',
                  background: z.user.present ? 'var(--accent-soft)' : 'var(--bg-elev-2)',
                  border: `1px solid ${z.user.present ? 'var(--accent-line)' : 'var(--line)'}`,
                  display: 'grid', placeItems: 'center',
                  fontSize: 10, fontWeight: 600, color: z.user.present ? 'var(--accent)' : 'var(--ink-mute)',
                }}>{z.user.name[0]}</div>
                <div>
                  <div style={{ fontWeight: 500 }}>{z.user.name}</div>
                  <div className="mono mute" style={{ fontSize: 10 }}>
                    {z.user.present ? `BLE ${z.user.ble_rssi}dBm` : 'ausente'}
                  </div>
                </div>
              </div>
            ) : (
              <div className="mute" style={{ fontStyle: 'italic' }}>nadie presente</div>
            )}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="mute mono" style={{ fontSize: 10, marginBottom: 2 }}>VOL</div>
            <div className="mono">{z.volume}%</div>
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
            <div className="mute">— sin actividad reciente —</div>
          )}
        </div>

        {z.spotify.playing && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 8, borderRadius: 4, border: '1px solid var(--line-soft)' }}>
            <div style={{ width: 32, height: 32, background: 'linear-gradient(135deg, #7c3aed, #1e293b)', borderRadius: 4, flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{z.spotify.song}</div>
              <div className="mute mono" style={{ fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{z.spotify.artist}</div>
              <div style={{ height: 2, background: 'var(--line)', borderRadius: 1, marginTop: 4 }}>
                <div style={{ height: '100%', width: `${(z.spotify.progress/z.spotify.total)*100}%`, background: 'var(--accent)', borderRadius: 1 }} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function PipelineWidget() {
  return (
    <div className="pipe">
      {PIPELINE_STAGES.map((s, i) => (
        <div key={s.id} className={`stage ${i === 1 ? 'active' : ''}`}>
          <h5>{s.label}</h5>
          <div className="mono">{s.hw}</div>
          <div className="mono" style={{ color: 'var(--accent)', marginTop: 4 }}>{s.ms}ms</div>
          <span className="arrow">→</span>
        </div>
      ))}
    </div>
  );
}

function useLiveStream() {
  const [last, setLast] = React.useState(null);
  React.useEffect(() => {
    const onLive = (ev) => setLast(ev.detail);
    window.addEventListener('kza:live', onLive);
    return () => window.removeEventListener('kza:live', onLive);
  }, []);
  return last;
}

function LiveTicker({ frame }) {
  if (!frame) return null;
  const p = frame.payload || {};
  return (
    <div className="card" style={{ marginBottom: 12, borderColor: 'var(--accent-line)', background: 'var(--accent-soft)' }}>
      <div className="card-b" style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
        <span className="pill accent">{frame.type.toUpperCase()}</span>
        <span className="mono mute">{frame.ts}</span>
        {p.zone && <span className="mono">zone={p.zone}</span>}
        {p.user && <span className="mono">user={p.user}</span>}
        {p.stt && <span className="dim" style={{ flex: 1 }}>"{p.stt}"</span>}
        {typeof p.latency_ms === 'number' && <span className="mono">{p.latency_ms}ms</span>}
      </div>
    </div>
  );
}

function LiveView() {
  const liveFrame = useLiveStream();
  return (
    <div className="view" data-active id="view-live">
      <PageHead
        title="Live"
        subtitle="Estado en tiempo real por zona · pipeline activo · latencias fast path"
        meta={<><span>WS conectado</span> · <span>tick 1.0s</span></>}
        actions={<button className="btn"><span className="mono">⏸</span> pausar stream</button>}
      />

      <LiveTicker frame={liveFrame} />

      {/* Top metrics row */}
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Card title="Latencia fast" meta="p50 / p95 / p99 · 5min">
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
            <div><div className="mono" style={{ fontSize: 22, color: 'var(--ok)' }}>192</div><div className="mute mono" style={{ fontSize: 10 }}>p50 ms</div></div>
            <div><div className="mono" style={{ fontSize: 16 }}>262</div><div className="mute mono" style={{ fontSize: 10 }}>p95</div></div>
            <div><div className="mono" style={{ fontSize: 16, color: 'var(--warn)' }}>308</div><div className="mute mono" style={{ fontSize: 10 }}>p99</div></div>
          </div>
          <div style={{ marginTop: 8 }}>
            <Sparkline data={LATENCY_FAST.p95} stroke="var(--accent)" />
          </div>
        </Card>

        <Card title="Wake-word 24h" meta="ReSpeaker XVF3800">
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
            <div><div className="mono" style={{ fontSize: 22 }}>{WAKE_EVENTS_24H.total}</div><div className="mute mono" style={{ fontSize: 10 }}>triggers</div></div>
            <div><div className="mono" style={{ fontSize: 16, color: 'var(--warn)' }}>{WAKE_EVENTS_24H.false_positive}</div><div className="mute mono" style={{ fontSize: 10 }}>false positives</div></div>
          </div>
          <div style={{ marginTop: 8 }}>
            <BarChart data={WAKE_EVENTS_24H.hourly} height={28} />
          </div>
          <div className="mono mute" style={{ fontSize: 10, marginTop: 4 }}>
            último: {WAKE_EVENTS_24H.last_trigger.ts} · conf {WAKE_EVENTS_24H.last_trigger.confidence} · {WAKE_EVENTS_24H.last_trigger.zone}
          </div>
        </Card>

        <Card title="LLM router" meta="endpoint activo">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="mono">vllm-7b</span>
              <Pill kind="ok">healthy · 82 tok/s</Pill>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="mono">ik-30b</span>
              <Pill kind="warn"><Countdown until="14:35:02" /></Pill>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="mono">bge-m3</span>
              <Pill kind="ok">healthy</Pill>
            </div>
          </div>
        </Card>

        <Card title="Pipeline breakdown" meta="último turn · 192ms">
          <LatencyBreakdown stages={LATENCY_BREAKDOWN} />
        </Card>
      </div>

      {/* Pipeline live diagram */}
      <SectionBar title="Pipeline en vivo" meta="turn_8821 · sala · u_marco" />
      <Card>
        <PipelineWidget />
      </Card>

      {/* Zones */}
      <SectionBar title="Zonas" meta={`${ZONES.length} habitaciones · ${ZONES.filter(z => z.user?.present).length} con usuario presente`} />
      <div className="grid grid-3">
        {ZONES.map(z => <ZoneCard key={z.id} z={z} />)}
      </div>
    </div>
  );
}

window.LiveView = LiveView;
