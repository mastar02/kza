// =============================================================
// Home Assistant view
// =============================================================

function ScoreBar({ score }) {
  if (typeof score !== 'number') return <span className="mono mute" style={{ fontSize: 10 }}>no observado</span>;
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

function ActionResult({ a }) {
  if (a.ok === true) return <Pill kind="ok">✓ ok</Pill>;
  if (a.ok === false) {
    return (
      <div>
        <Pill kind="err">✗ falló</Pill>
        {a.err && <div className="mono" style={{ fontSize: 9, color: 'var(--err)', marginTop: 2 }}>{a.err}</div>}
      </div>
    );
  }
  return <Pill>sin confirmar</Pill>;
}

function HAView() {
  const obs = useObs();
  const entities = obs.get('HA_ENTITIES');
  const actions = obs.get('HA_ACTIONS');
  const [domainFilter, setDomainFilter] = React.useState('all');
  const domains = ['all', ...Array.from(new Set(entities.map(e => e.domain)))];
  const filtered = domainFilter === 'all' ? entities : entities.filter(e => e.domain === domainFilter);
  return (
    <div className="view" id="view-ha">
      <PageHead
        title="Home Assistant"
        subtitle="Estados leídos de Home Assistant · registro de acciones"
        meta={obs.available('HA_ENTITIES') ? <span>{entities.length} entidades</span> : null}
      />

      <SectionBar title="Entidades" meta={<SourceNote k="HA_ENTITIES" />} />
      <Card>
        {obs.available('HA_ENTITIES') && entities.length > 0 ? (
          <>
            <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
              {domains.map(d => (
                <button type="button" key={d} className={`btn ${domainFilter === d ? 'primary' : ''}`} aria-pressed={domainFilter === d} onClick={() => setDomainFilter(d)}>
                  {d}
                </button>
              ))}
            </div>
            <table className="k">
              <thead>
                <tr>
                  <th>entity_id</th><th>dominio</th><th>nombre</th><th>estado</th>
                  <th>similitud</th><th>último cambio</th>
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
                    <td className="mono mute" style={{ fontSize: 10 }}>{e.state_changed_at ? window.KzaObsData.formatInstant(e.state_changed_at) : 'no observado'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <Unavailable>{obs.available('HA_ENTITIES') ? 'Home Assistant no informó entidades.' : 'Entidades no disponibles: no se pudo leer Home Assistant.'}</Unavailable>
        )}
      </Card>

      <SectionBar title="Acciones disparadas" meta={<SourceNote k="HA_ACTIONS" />} />
      <Card>
        {obs.available('HA_ACTIONS') && actions.length > 0 ? (
          <table className="k">
            <thead>
              <tr>
                <th>ts</th><th>idempotency</th><th>usuario</th><th>servicio</th>
                <th>destino</th><th>args</th><th>resultado</th><th style={{ textAlign:'right' }}>lat</th>
              </tr>
            </thead>
            <tbody>
              {actions.map(a => (
                <tr key={a.id}>
                  <td><EventTime at={a.at} ts={a.ts} /></td>
                  <td className="mono" style={{ fontSize: 10, color: 'var(--accent)' }}>{a.idem || '—'}</td>
                  <td>{a.user || '—'}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{a.service}</td>
                  <td className="mono mute" style={{ fontSize: 10 }}>{a.target}</td>
                  <td className="mono mute" style={{ fontSize: 10 }}>{a.args ?? '—'}</td>
                  <td><ActionResult a={a} /></td>
                  <td className="num">{typeof a.lat_ms === 'number' ? `${a.lat_ms}ms` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Unavailable>{obs.available('HA_ACTIONS') ? 'Sin acciones registradas.' : 'Acciones no disponibles: no se pudo leer el registro.'}</Unavailable>
        )}
      </Card>
    </div>
  );
}

window.HAView = HAView;
