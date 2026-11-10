// =============================================================
// Alerts view
// =============================================================

function AlertCard({ a }) {
  const kindMap = { info: '', warn: 'warn', critical: 'err' };
  const k = kindMap[a.priority];
  const accentColor = a.priority === 'critical' ? 'var(--err)' : a.priority === 'warn' ? 'var(--warn)' : 'var(--info)';
  return (
    <div className="card" style={{ borderLeft: `2px solid ${accentColor}`, opacity: a.acked ? 0.55 : 1 }}>
      <div className="card-b">
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
              <Pill kind={k}>{a.priority}</Pill>
              <Pill>{a.type}</Pill>
              <span className="mono mute" style={{ fontSize: 10 }}>{a.zone}</span>
              <EventTime at={a.at} ts={a.ts} />
              {a.acked && <span className="mono mute" style={{ fontSize: 10 }}>· atendida</span>}
            </div>
            <div style={{ fontWeight: 500, marginBottom: 4 }}>{a.title}</div>
            <div className="mute" style={{ fontSize: 12 }}>{a.body}</div>
          </div>
          {!a.acked && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <OperationButton kind="ackAlert" target={a.id} label="Marcar atendida" />
              <OperationButton kind="dismissAlert" target={a.id} label="Descartar" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AlertsView() {
  const obs = useObs();
  const [tab, setTab] = React.useState('active');
  const alerts = obs.get('ALERTS');
  const available = obs.available('ALERTS');
  const active = alerts.filter(a => !a.acked);
  const acked = alerts.filter(a => a.acked);
  const list = tab === 'active' ? active : acked;
  return (
    <div className="view" id="view-alerts">
      <PageHead
        title="Alertas"
        subtitle="Alertas proactivas · seguridad / patrones / dispositivos"
        meta={available ? <><span>{active.length} activas</span> · <span>{acked.length} atendidas</span> · <SourceNote k="ALERTS" /></> : <SourceNote k="ALERTS" />}
      />

      <Card>
        <div style={{ display: 'flex', gap: 4 }}>
          <button type="button" className={`btn ${tab === 'active' ? 'primary' : ''}`} aria-pressed={tab === 'active'} onClick={() => setTab('active')}>activas · {active.length}</button>
          <button type="button" className={`btn ${tab === 'acked' ? 'primary' : ''}`} aria-pressed={tab === 'acked'} onClick={() => setTab('acked')}>atendidas · {acked.length}</button>
        </div>
      </Card>

      <div className="col" style={{ marginTop: 16 }}>
        {list.map(a => <AlertCard key={a.id} a={a} />)}
        {list.length === 0 && (
          <Card>
            <div className="mute" style={{ textAlign: 'center', padding: 30 }}>
              {available ? 'sin alertas' : 'Alertas no disponibles: no se pudo leer la fuente.'}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

window.AlertsView = AlertsView;
