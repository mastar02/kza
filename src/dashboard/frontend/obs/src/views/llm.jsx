// =============================================================
// LLM Router view
// =============================================================

function StateBadge({ state }) {
  if (state === 'healthy') return <Pill kind="ok">● saludable</Pill>;
  if (state === 'cooldown') return <Pill kind="warn">● cooldown</Pill>;
  if (state === 'disabled') return <Pill kind="err">● deshabilitado</Pill>;
  if (state === 'unknown' || !state) return <Pill>sin verificar</Pill>;
  return <Pill>{state}</Pill>;
}

function Failures({ value, observed }) {
  if (!observed || !value || typeof value !== 'object' || Object.keys(value).length === 0) {
    return <span className="mute">no observado</span>;
  }
  const total = Object.values(value).reduce((s, n) => s + (Number(n) || 0), 0);
  if (total === 0) return <span className="mute">0</span>;
  return (
    <span className="mono" style={{ fontSize: 10 }}>
      {value.timeout > 0 && <Pill kind="warn">to:{value.timeout}</Pill>}
      {value.idle > 0 && <Pill>idle:{value.idle}</Pill>}
      {value.rate_limit > 0 && <Pill kind="warn">429:{value.rate_limit}</Pill>}
      {value.billing > 0 && <Pill kind="err">$:{value.billing}</Pill>}
    </span>
  );
}

function LLMView() {
  const obs = useObs();
  const endpoints = obs.get('LLM_ENDPOINTS');
  const available = obs.available('LLM_ENDPOINTS');
  const inCooldown = endpoints.filter(e => e.cooldown_ends);
  const verified = endpoints.filter(e => e.state === 'healthy');
  const failuresObserved = !obs.unobserved('LLM_ENDPOINTS', 'failures_7d');
  return (
    <div className="view" id="view-llm">
      <PageHead
        title="LLM Router"
        subtitle="Failover entre endpoints · escalones de cooldown por defecto del código: 1m → 5m → 25m → 1h"
        meta={<>ruta de persistencia por defecto <span className="mono">data/llm_cooldowns.json</span> · <SourceNote k="LLM_ENDPOINTS" /></>}
        actions={<UnavailableButton id="llm-health-reason" label="Forzar chequeo de salud"
          reason="Chequeo de salud no disponible: no existe una operación autenticada para forzarlo." />}
      />

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Card title="Endpoints configurados">
          <div className="mono" style={{ fontSize: 18 }}>{available ? endpoints.length : '—'}</div>
          <div className="mute mono" style={{ fontSize: 11 }}>{available ? 'según el estado del router' : 'no disponible'}</div>
        </Card>
        <Card title="En cooldown">
          <div className="mono" style={{ fontSize: 18, color: inCooldown.length ? 'var(--warn)' : 'var(--ink)' }}>{available ? inCooldown.length : '—'}</div>
          <div className="mute mono" style={{ fontSize: 11 }}>{available ? inCooldown.map(e => e.id).join(', ') || 'ninguno' : 'no disponible'}</div>
        </Card>
        <Card title="Salud verificada">
          <div className="mono" style={{ fontSize: 18 }}>{available ? verified.length : '—'}</div>
          <div className="mute mono" style={{ fontSize: 11 }}>
            {available ? (obs.unobserved('LLM_ENDPOINTS', 'backend_health') ? 'el servidor no observa la salud de los backends' : 'endpoints saludables') : 'no disponible'}
          </div>
        </Card>
      </div>

      <SectionBar title="Endpoints" meta={available ? `${endpoints.length} configurados` : null} />
      <Card>
        {available && endpoints.length > 0 ? (
          <table className="k">
            <thead>
              <tr>
                <th>id</th><th>rol</th><th>url</th><th>prio</th><th>estado</th>
                <th>tok/s</th><th>ttft</th>
                <th>fallos 7d</th><th>cooldown</th><th>última llamada</th><th>acción</th>
              </tr>
            </thead>
            <tbody>
              {endpoints.map(e => (
                <tr key={e.id}>
                  <td className="mono">{e.id}</td>
                  <td className="mute">{e.role || '—'}</td>
                  <td className="mono" style={{ fontSize: 10 }}>{e.url || '—'}</td>
                  <td className="mono">{e.priority ?? '—'}</td>
                  <td><StateBadge state={e.state} /></td>
                  <td className="num">{typeof e.tps === 'number' ? e.tps : '—'}</td>
                  <td className="num">{typeof e.ttft_ms === 'number' ? `${e.ttft_ms}ms` : '—'}</td>
                  <td><Failures value={e.failures_7d} observed={failuresObserved} /></td>
                  <td>{e.cooldown_ends || e.cooldown_until ? <Countdown until={e.cooldown_until} fallback={e.cooldown_ends} /> : <span className="mute">—</span>}</td>
                  <td className="mono mute" style={{ fontSize: 10 }}>{e.last_call_at ? window.KzaObsData.formatInstant(e.last_call_at) : e.last_check || '—'}</td>
                  <td>
                    {e.cooldown_ends ? (
                      <OperationButton kind="clearCooldown" target={e.id} label="Quitar cooldown" confirmLabel={`Confirmar: quitar cooldown de ${e.id}`} />
                    ) : (
                      <UnavailableButton id={`probe-reason-${String(e.id).replace(/[^a-zA-Z0-9_-]/g, '_')}`} label="Probar"
                        reason="Prueba no disponible: no existe una operación autenticada." />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Unavailable>{available ? 'No hay endpoints configurados.' : 'Endpoints no disponibles: no se pudo leer el estado del router.'}</Unavailable>
        )}
      </Card>

      <SectionBar title="Historial de cooldowns y decisiones del router" />
      <Card>
        <Unavailable>No disponible: el servidor no publica historial de cooldowns ni de decisiones del router.</Unavailable>
      </Card>
    </div>
  );
}

window.LLMView = LLMView;
