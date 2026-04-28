// =============================================================
// Conversations view
// =============================================================

const { CONVERSATIONS: CONVS } = window.MOCKS;

function PathPill({ path }) {
  const map = {
    fast: { kind: 'ok', label: 'fast' },
    slow: { kind: 'info', label: 'slow' },
    music: { kind: 'accent', label: 'music' },
  };
  const m = map[path];
  return <Pill kind={m.kind}>{m.label}</Pill>;
}

function ConversationsView() {
  const [selectedId, setSelectedId] = React.useState(CONVS[0].id);
  const selected = CONVS.find(c => c.id === selectedId);
  const [filterPath, setFilterPath] = React.useState('all');
  const [filterUser, setFilterUser] = React.useState('all');
  const filtered = CONVS.filter(c =>
    (filterPath === 'all' || c.path === filterPath) &&
    (filterUser === 'all' || c.user === filterUser)
  );
  return (
    <div className="view" id="view-conversations">
      <PageHead
        title="Conversaciones"
        subtitle="Transcript browser · turnos completos con desglose por etapa"
        meta={<><span>{CONVS.length} turnos</span> · <span>últimas 24h</span></>}
        actions={<button className="btn">exportar JSONL</button>}
      />

      <Card>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className="mono mute" style={{ fontSize: 10 }}>FILTROS</span>
          <select className="btn" value={filterUser} onChange={e => setFilterUser(e.target.value)}>
            <option value="all">todos los users</option>
            <option value="Marco">Marco</option>
            <option value="Lucía">Lucía</option>
          </select>
          <select className="btn" value={filterPath} onChange={e => setFilterPath(e.target.value)}>
            <option value="all">todos los paths</option>
            <option value="fast">fast</option>
            <option value="slow">slow</option>
            <option value="music">music</option>
          </select>
          <input type="search" placeholder="buscar en transcript…" style={{ background: 'var(--bg-elev-2)', border: '1px solid var(--line)', color: 'var(--ink)', borderRadius: 5, padding: '5px 10px', fontFamily: 'var(--mono)', fontSize: 11, flex: 1, minWidth: 200 }} />
          <span className="mono mute" style={{ fontSize: 10 }}>{filtered.length} resultados</span>
        </div>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: '1.4fr 1fr', marginTop: 16 }}>
        <Card title="Turnos" meta="ordenados por timestamp" >
          <table className="k">
            <thead>
              <tr><th>ts</th><th>user</th><th>zona</th><th>path</th><th>frase</th><th style={{ textAlign: 'right' }}>lat</th></tr>
            </thead>
            <tbody>
              {filtered.map(c => (
                <tr key={c.id} onClick={() => setSelectedId(c.id)}
                    style={{ cursor: 'pointer', background: c.id === selectedId ? 'var(--accent-soft)' : 'transparent' }}>
                  <td className="mono">{c.ts}</td>
                  <td>{c.user}</td>
                  <td className="mute">{c.zone}</td>
                  <td><PathPill path={c.path} /></td>
                  <td style={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>"{c.stt}"</td>
                  <td className="num" style={{ color: c.success ? 'var(--ink)' : 'var(--err)' }}>{c.latency_ms}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <div className="col">
          <Card title={`Turn ${selected.id}`} meta={`${selected.ts} · ${selected.zone}`} action={<PathPill path={selected.path} />}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>AUDIO</div>
                <Waveform bars={48} height={32} />
              </div>
              <div>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>STT (whisper-v3-turbo)</div>
                <div style={{ fontStyle: 'italic' }}>"{selected.stt}"</div>
              </div>
              <div>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>INTENT (router 7B)</div>
                <div className="mono">{selected.intent}</div>
                {selected.target && <div className="mono mute" style={{ fontSize: 11 }}>→ {selected.target}</div>}
                {Object.keys(selected.args).length > 0 && (
                  <div className="mono mute" style={{ fontSize: 11 }}>args: {JSON.stringify(selected.args)}</div>
                )}
              </div>
              <div>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>ACCIÓN HA</div>
                {selected.success ? (
                  <Pill kind="ok">✓ ejecutada</Pill>
                ) : (
                  <div>
                    <Pill kind="err">✗ fallo</Pill>
                    <div className="mono" style={{ color: 'var(--err)', fontSize: 11, marginTop: 4 }}>{selected.error}</div>
                  </div>
                )}
              </div>
              <div>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>TTS (kokoro)</div>
                <div>{selected.tts || <span className="mute">— ack only —</span>}</div>
              </div>
            </div>
          </Card>

          <Card title="Latencia desglosada" meta={`total ${selected.latency_ms}ms`}>
            <LatencyBreakdown stages={selected.path === 'fast' ? window.MOCKS.LATENCY_BREAKDOWN : [
              { stage: 'Wake', ms: 12 },
              { stage: 'STT', ms: 82 },
              { stage: 'Router', ms: 28 },
              { stage: '30B reasoning', ms: Math.round(selected.latency_ms * 0.85) },
              { stage: 'TTS', ms: 32 },
            ]} />
          </Card>
        </div>
      </div>
    </div>
  );
}

window.ConversationsView = ConversationsView;
