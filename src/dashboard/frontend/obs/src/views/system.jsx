// =============================================================
// System view — health
// =============================================================

const { GPUS, SERVICES, LOG_LINES } = window.MOCKS;

function GpuCard({ g }) {
  const vramPct = (g.vramUsed / g.vramTotal) * 100;
  const tempColor = g.temp > 80 ? 'var(--err)' : g.temp > 70 ? 'var(--warn)' : 'var(--ok)';
  return (
    <Card title={g.name} meta={g.role}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 14 }}>
        <div><div className="mute mono" style={{ fontSize: 10 }}>UTIL</div><div className="mono" style={{ fontSize: 18 }}>{g.util}%</div></div>
        <div><div className="mute mono" style={{ fontSize: 10 }}>VRAM</div><div className="mono" style={{ fontSize: 18 }}>{g.vramUsed}<span className="mute" style={{ fontSize: 12 }}>/{g.vramTotal}GB</span></div></div>
        <div><div className="mute mono" style={{ fontSize: 10 }}>TEMP</div><div className="mono" style={{ fontSize: 18, color: tempColor }}>{g.temp}°C</div></div>
      </div>
      <GpuMeter label="util" value={g.util} unit="%" />
      <GpuMeter label="vram" value={g.vramUsed} max={g.vramTotal} unit={`/${g.vramTotal}GB`} color={vramPct > 90 ? 'var(--warn)' : 'var(--accent)'} />
      <GpuMeter label="power" value={g.power} max={220} unit="W" />
      <SectionBar title="Procesos" />
      <table className="k">
        <tbody>
          {g.procs.map(p => (
            <tr key={p.name}>
              <td className="mono" style={{ fontSize: 11 }}>{p.name}</td>
              <td className="num">{p.vram} GB</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function SystemView() {
  const [logFilter, setLogFilter] = React.useState(null);
  const services = ['all', ...Array.from(new Set(LOG_LINES.map(l => l.svc)))];
  return (
    <div className="view" id="view-system">
      <PageHead
        title="Sistema"
        subtitle="GPU · servicios · logs · Threadripper 7965WX (24c/48t · 128GB)"
        meta={<><span>uptime 21d</span> · <span className="mono">192.168.1.2</span></>}
      />

      <SectionBar title="GPUs" meta="2x RTX 3070 8GB" />
      <div className="grid grid-2">
        {GPUS.map(g => <GpuCard key={g.id} g={g} />)}
      </div>

      <SectionBar title="Servicios systemd" />
      <Card>
        <table className="k">
          <thead>
            <tr><th>servicio</th><th>status</th><th>uptime</th><th style={{ textAlign:'right' }}>mem</th><th style={{ textAlign:'right' }}>cpu</th><th>pid</th><th></th></tr>
          </thead>
          <tbody>
            {SERVICES.map(s => (
              <tr key={s.name}>
                <td className="mono">{s.name}</td>
                <td>{s.status === 'active' ? <Pill kind="ok">● active</Pill> : <Pill kind="warn">● {s.status}</Pill>}</td>
                <td className="mono mute" style={{ fontSize: 11 }}>{s.uptime}</td>
                <td className="num">{s.mem}</td>
                <td className="num">{s.cpu}</td>
                <td className="mono mute" style={{ fontSize: 10 }}>{s.pid}</td>
                <td><button className="btn">restart</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <SectionBar title="Logs (tail)" meta="multiplexed · journalctl-style" action={<button className="btn">pausar</button>} />
      <Card>
        <div style={{ display: 'flex', gap: 4, marginBottom: 10, flexWrap: 'wrap' }}>
          {services.map(s => (
            <button key={s} className={`btn ${(s === 'all' && !logFilter) || logFilter === s ? 'primary' : ''}`}
                    onClick={() => setLogFilter(s === 'all' ? null : s)}>{s}</button>
          ))}
        </div>
        <LogViewer lines={LOG_LINES} filter={logFilter} height={420} />
      </Card>
    </div>
  );
}

window.SystemView = SystemView;
