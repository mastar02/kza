// =============================================================
// Home Assistant view
// =============================================================

const { HA_ENTITIES, HA_ACTIONS } = window.MOCKS;

function ScoreBar({ score }) {
  const pct = score * 100;
  const color = score > 0.9 ? 'var(--ok)' : score > 0.8 ? 'var(--warn)' : 'var(--ink-mute)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 80 }}>
      <div style={{ flex: 1, height: 4, background: 'var(--bg-elev-2)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color }} />
      </div>
      <span className="mono" style={{ fontSize: 10, color, minWidth: 28, textAlign: 'right' }}>{score.toFixed(2)}</span>
    </div>
  );
}

function StatePill({ entity }) {
  const s = entity.state;
  if (s === 'unavailable') return <Pill kind="err">unavailable</Pill>;
  if (s === 'on' || s === 'playing' || s === 'heat' || s === 'open') return <Pill kind="ok">{s}</Pill>;
  if (s === 'off' || s === 'closed' || s === 'locked') return <Pill>{s}</Pill>;
  return <Pill kind="info">{s}</Pill>;
}

function HAView() {
  const [domainFilter, setDomainFilter] = React.useState('all');
  const domains = ['all', ...Array.from(new Set(HA_ENTITIES.map(e => e.domain)))];
  const filtered = domainFilter === 'all' ? HA_ENTITIES : HA_ENTITIES.filter(e => e.domain === domainFilter);
  return (
    <div className="view" id="view-ha">
      <PageHead
        title="Home Assistant"
        subtitle="Entidades indexadas en ChromaDB · log de acciones con idempotency keys"
        meta={<><span>{HA_ENTITIES.length} entidades</span> · <span>BGE-M3 embeddings</span></>}
      />

      <SectionBar title="Entidades indexadas" meta={`${filtered.length} de ${HA_ENTITIES.length}`} />
      <Card>
        <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
          {domains.map(d => (
            <button key={d} className={`btn ${domainFilter === d ? 'primary' : ''}`} onClick={() => setDomainFilter(d)}>
              {d}
            </button>
          ))}
        </div>
        <table className="k">
          <thead>
            <tr>
              <th>entity_id</th><th>domain</th><th>nombre</th><th>state</th>
              <th>similarity</th><th>last seen</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(e => (
              <tr key={e.id}>
                <td className="mono" style={{ fontSize: 11 }}>{e.id}</td>
                <td><span className="mono mute">{e.domain}</span></td>
                <td>{e.name}</td>
                <td><StatePill entity={e} /></td>
                <td><ScoreBar score={e.score} /></td>
                <td className="mono mute" style={{ fontSize: 10 }}>{e.lastSeen}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <SectionBar title="Acciones disparadas" meta="con idempotency key" action={<button className="btn">tail en vivo</button>} />
      <Card>
        <table className="k">
          <thead>
            <tr>
              <th>ts</th><th>idempotency</th><th>user</th><th>service</th>
              <th>target</th><th>args</th><th>resultado</th><th style={{ textAlign:'right' }}>lat</th>
            </tr>
          </thead>
          <tbody>
            {HA_ACTIONS.map(a => (
              <tr key={a.id}>
                <td className="mono">{a.ts}</td>
                <td className="mono" style={{ fontSize: 10, color: 'var(--accent)' }}>{a.idem}</td>
                <td>{a.user}</td>
                <td className="mono" style={{ fontSize: 11 }}>{a.service}</td>
                <td className="mono mute" style={{ fontSize: 10 }}>{a.target}</td>
                <td className="mono mute" style={{ fontSize: 10 }}>{a.args}</td>
                <td>
                  {a.ok ? <Pill kind="ok">✓ ok</Pill> : (
                    <div>
                      <Pill kind="err">✗ fail</Pill>
                      <div className="mono" style={{ fontSize: 9, color: 'var(--err)', marginTop: 2 }}>{a.err}</div>
                    </div>
                  )}
                </td>
                <td className="num">{a.lat_ms}ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

window.HAView = HAView;
