// =============================================================
// Alerts view
// =============================================================

const { ALERTS } = window.MOCKS;

function AlertCard({ a }) {
  const kindMap = { info: '', warn: 'warn', critical: 'err' };
  const k = kindMap[a.priority];
  const accentColor = a.priority === 'critical' ? 'var(--err)' : a.priority === 'warn' ? 'var(--warn)' : 'var(--info)';
  return (
    <div className="card" style={{ borderLeft: `2px solid ${accentColor}`, opacity: a.acked ? 0.55 : 1 }}>
      <div className="card-b">
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
              <Pill kind={k}>{a.priority}</Pill>
              <Pill>{a.type}</Pill>
              <span className="mono mute" style={{ fontSize: 10 }}>{a.zone}</span>
              <span className="mono mute" style={{ fontSize: 10 }}>{a.ts}</span>
              {a.acked && <span className="mono mute" style={{ fontSize: 10 }}>· acknowledged</span>}
            </div>
            <div style={{ fontWeight: 500, marginBottom: 4 }}>{a.title}</div>
            <div className="mute" style={{ fontSize: 12 }}>{a.body}</div>
          </div>
          {!a.acked && (
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn">ack</button>
              <button className="btn">dismiss</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AlertsView() {
  const [tab, setTab] = React.useState('active');
  const active = ALERTS.filter(a => !a.acked);
  const acked = ALERTS.filter(a => a.acked);
  const list = tab === 'active' ? active : acked;
  return (
    <div className="view" id="view-alerts">
      <PageHead
        title="Alertas"
        subtitle="Stream proactivo · security / pattern / device"
        meta={<><span>{active.length} activas</span> · <span>{acked.length} acked</span></>}
      />

      <Card>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className={`btn ${tab === 'active' ? 'primary' : ''}`} onClick={() => setTab('active')}>activas · {active.length}</button>
          <button className={`btn ${tab === 'acked' ? 'primary' : ''}`} onClick={() => setTab('acked')}>acknowledged · {acked.length}</button>
        </div>
      </Card>

      <div className="col" style={{ marginTop: 16 }}>
        {list.map(a => <AlertCard key={a.id} a={a} />)}
        {list.length === 0 && (
          <Card>
            <div className="mute" style={{ textAlign: 'center', padding: 30 }}>sin alertas</div>
          </Card>
        )}
      </div>
    </div>
  );
}

window.AlertsView = AlertsView;
