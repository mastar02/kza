// =============================================================
// LLM Router view
// =============================================================

const { LLM_ENDPOINTS, COOLDOWN_HISTORY_7D } = window.MOCKS;

function StateBadge({ state }) {
  if (state === 'healthy') return <Pill kind="ok">● healthy</Pill>;
  if (state === 'cooldown') return <Pill kind="warn">● cooldown</Pill>;
  if (state === 'disabled') return <Pill kind="err">● disabled</Pill>;
  return <Pill>{state}</Pill>;
}

function LLMView() {
  const [confirmId, setConfirmId] = React.useState(null);
  return (
    <div className="view" id="view-llm">
      <PageHead
        title="LLM Router"
        subtitle="Failover entre endpoints · cooldown exponencial 1m → 5m → 25m → 1h"
        meta={<>persistido en <span className="mono">data/llm_cooldowns.json</span></>}
        actions={<button className="btn">force health check</button>}
      />

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Card title="Endpoint activo" meta="fast path">
          <div className="mono" style={{ fontSize: 18, color: 'var(--accent)' }}>vllm-7b</div>
          <div className="mute mono" style={{ fontSize: 11 }}>:8100 · cuda:1 · prio 1</div>
          <div style={{ display: 'flex', gap: 16, marginTop: 12 }}>
            <div><div className="mono" style={{ fontSize: 16 }}>82</div><div className="mute mono" style={{ fontSize: 10 }}>tok/s</div></div>
            <div><div className="mono" style={{ fontSize: 16 }}>142</div><div className="mute mono" style={{ fontSize: 10 }}>ttft ms</div></div>
          </div>
        </Card>

        <Card title="Slow path" meta="reasoning">
          <div className="mono" style={{ fontSize: 18, color: 'var(--warn)' }}>ik-30b-cpu</div>
          <div className="mute mono" style={{ fontSize: 11 }}>:8200 · CPU · prio 2</div>
          <div style={{ display: 'flex', gap: 16, marginTop: 12, alignItems: 'center' }}>
            <Pill kind="warn">cooldown 5m</Pill>
            <Countdown until="14:35:02" />
          </div>
        </Card>

        <Card title="Cooldowns últimos 7 días" meta={`${COOLDOWN_HISTORY_7D.reduce((s,d)=>s+d.count,0)} eventos`}>
          <BarChart data={COOLDOWN_HISTORY_7D} height={56} labels />
        </Card>
      </div>

      <SectionBar title="Endpoints" meta={`${LLM_ENDPOINTS.length} configurados`} />
      <Card>
        <table className="k">
          <thead>
            <tr>
              <th>id</th><th>rol</th><th>url</th><th>prio</th><th>state</th>
              <th>tok/s</th><th>ttft</th>
              <th>fallos 7d</th><th>cooldown</th><th>last check</th><th></th>
            </tr>
          </thead>
          <tbody>
            {LLM_ENDPOINTS.map(e => {
              const totalFails = Object.values(e.failures_7d).reduce((s,n)=>s+n,0);
              return (
                <tr key={e.id}>
                  <td className="mono">{e.id}</td>
                  <td className="mute">{e.role}</td>
                  <td className="mono" style={{ fontSize: 10 }}>{e.url}</td>
                  <td className="mono">{e.priority}</td>
                  <td><StateBadge state={e.state} /></td>
                  <td className="num">{e.tps ?? '—'}</td>
                  <td className="num">{e.ttft_ms}ms</td>
                  <td>
                    <span className="mono" style={{ fontSize: 10 }}>
                      {totalFails === 0 ? <span className="mute">0</span> : (
                        <>
                          {e.failures_7d.timeout    > 0 && <Pill kind="warn">to:{e.failures_7d.timeout}</Pill>}
                          {e.failures_7d.idle       > 0 && <Pill>idle:{e.failures_7d.idle}</Pill>}
                          {e.failures_7d.rate_limit > 0 && <Pill kind="warn">429:{e.failures_7d.rate_limit}</Pill>}
                          {e.failures_7d.billing    > 0 && <Pill kind="err">$:{e.failures_7d.billing}</Pill>}
                        </>
                      )}
                    </span>
                  </td>
                  <td>{e.cooldown_ends ? <Countdown until={e.cooldown_ends} /> : <span className="mute">—</span>}</td>
                  <td className="mono mute" style={{ fontSize: 10 }}>{e.last_check}</td>
                  <td>
                    {e.cooldown_ends ? (
                      confirmId === e.id ? (
                        <div style={{ display: 'flex', gap: 4 }}>
                          <button className="btn danger" onClick={() => setConfirmId(null)}>confirmar</button>
                          <button className="btn" onClick={() => setConfirmId(null)}>×</button>
                        </div>
                      ) : (
                        <button className="btn" onClick={() => setConfirmId(e.id)}>clear cooldown</button>
                      )
                    ) : <button className="btn">probe</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <SectionBar title="Decisión del router · últimas 30 requests" />
      <Card>
        <div style={{ display: 'flex', gap: 2, height: 44, alignItems: 'flex-end' }}>
          {Array.from({ length: 30 }, (_, i) => {
            const v = (i % 7 === 6) ? 'slow' : (i % 5 === 4 ? 'music' : 'fast');
            const c = v === 'fast' ? 'var(--ok)' : v === 'slow' ? 'var(--info)' : 'var(--accent)';
            const h = v === 'fast' ? 16 : v === 'music' ? 26 : 40;
            return <div key={i} style={{ flex: 1, height: h, background: c, opacity: 0.75, borderRadius: 2 }} title={v} />;
          })}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 8, fontFamily: 'var(--mono)', fontSize: 10 }}>
          <span><span style={{ display: 'inline-block', width: 8, height: 8, background: 'var(--ok)', borderRadius: 2, marginRight: 4 }} />fast</span>
          <span><span style={{ display: 'inline-block', width: 8, height: 8, background: 'var(--accent)', borderRadius: 2, marginRight: 4 }} />music</span>
          <span><span style={{ display: 'inline-block', width: 8, height: 8, background: 'var(--info)', borderRadius: 2, marginRight: 4 }} />slow</span>
        </div>
      </Card>
    </div>
  );
}

window.LLMView = LLMView;
