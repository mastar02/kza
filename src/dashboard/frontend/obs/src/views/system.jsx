// =============================================================
// System view — GPUs and services observed by the server probes
// =============================================================

function GpuCard({ g, roleObserved }) {
  const vramPct = typeof g.vramUsed === 'number' && g.vramTotal ? (g.vramUsed / g.vramTotal) * 100 : 0;
  const tempColor = g.temp > 80 ? 'var(--err)' : g.temp > 70 ? 'var(--warn)' : 'var(--ok)';
  const show = (value, unit = '') => (typeof value === 'number' ? `${value}${unit}` : '—');
  return (
    <Card title={g.name} meta={roleObserved ? g.role : `${g.role || '—'} (asignación declarada)`}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 14 }}>
        <div><div className="mute mono" style={{ fontSize: 10 }}>UTIL</div><div className="mono" style={{ fontSize: 18 }}>{show(g.util, '%')}</div></div>
        <div><div className="mute mono" style={{ fontSize: 10 }}>VRAM</div><div className="mono" style={{ fontSize: 18 }}>{show(g.vramUsed)}<span className="mute" style={{ fontSize: 12 }}>/{show(g.vramTotal)}GB</span></div></div>
        <div><div className="mute mono" style={{ fontSize: 10 }}>TEMP</div><div className="mono" style={{ fontSize: 18, color: tempColor }}>{show(g.temp, '°C')}</div></div>
      </div>
      <GpuMeter label="util" value={g.util} unit="%" />
      <GpuMeter label="vram" value={g.vramUsed} max={g.vramTotal || 1} unit={`/${show(g.vramTotal)}GB`} color={vramPct > 90 ? 'var(--warn)' : 'var(--accent)'} />
      <GpuMeter label="power" value={g.power} max={220} unit="W" />
      <SectionBar title="Procesos" />
      {Array.isArray(g.procs) && g.procs.length > 0 ? (
        <table className="k">
          <tbody>
            {g.procs.map(p => (
              <tr key={p.name}>
                <td className="mono" style={{ fontSize: 11 }}>{p.name}</td>
                <td className="num">{show(p.vram)} GB</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <Unavailable>Sin procesos observados.</Unavailable>}
    </Card>
  );
}

function ServiceStatus({ status }) {
  if (status === 'active') return <Pill kind="ok">● activo</Pill>;
  if (status === 'missing') return <Pill>● no encontrado</Pill>;
  if (status === 'unreachable') return <Pill kind="err">● sin respuesta</Pill>;
  return <Pill kind="warn">● {status || 'desconocido'}</Pill>;
}

function SystemView() {
  const obs = useObs();
  const gpus = obs.get('GPUS');
  const services = obs.get('SERVICES');
  return (
    <div className="view" id="view-system">
      <PageHead
        title="Sistema"
        subtitle="GPUs y servicios según las sondas del servidor"
        meta={<SourceNote k="SERVICES" />}
      />

      <SectionBar title="GPUs" meta={obs.available('GPUS') ? `${gpus.length} observadas` : null} />
      {obs.available('GPUS') && gpus.length > 0 ? (
        <div className="grid grid-2">
          {gpus.map(g => <GpuCard key={g.id} g={g} roleObserved={!obs.unobserved('GPUS', 'role_assignment')} />)}
        </div>
      ) : (
        <Card><Unavailable>{obs.available('GPUS') ? 'No se observaron GPUs.' : 'GPUs no disponibles: no se pudo leer la sonda.'}</Unavailable></Card>
      )}

      <SectionBar title="Servicios" meta={<SourceNote k="SERVICES" />} />
      <Card>
        {obs.available('SERVICES') && services.length > 0 ? (
          <table className="k">
            <thead>
              <tr><th>servicio</th><th>estado</th><th>inicio</th><th style={{ textAlign:'right' }}>mem</th><th style={{ textAlign:'right' }}>cpu</th><th>pid</th><th>acción</th></tr>
            </thead>
            <tbody>
              {services.map(s => (
                <tr key={s.name}>
                  <td className="mono">{s.name}</td>
                  <td><ServiceStatus status={s.status} /></td>
                  <td className="mono mute" style={{ fontSize: 11 }}>{s.uptime || '—'}</td>
                  <td className="num">{s.mem || '—'}</td>
                  <td className="num">{s.cpu || '—'}</td>
                  <td className="mono mute" style={{ fontSize: 10 }}>{s.pid || '—'}</td>
                  <td><OperationButton kind="restartService" target={s.name} label="Reiniciar" confirmLabel={`Confirmar reinicio de ${s.name}`} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Unavailable>{obs.available('SERVICES') ? 'No se observaron servicios.' : 'Servicios no disponibles: no se pudo leer la sonda.'}</Unavailable>
        )}
      </Card>

      <SectionBar title="Registros" />
      <Card>
        <Unavailable>Registros no disponibles en este panel: no hay una fuente autenticada de logs. Consultalos con journalctl en el servidor.</Unavailable>
      </Card>
    </div>
  );
}

window.SystemView = SystemView;
