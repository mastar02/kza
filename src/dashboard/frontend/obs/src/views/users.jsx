// =============================================================
// Users view
// =============================================================

const { USERS } = window.MOCKS;

function PermToggle({ on, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--line-soft)' }}>
      <span style={{ fontSize: 12 }}>{label}</span>
      <div style={{
        width: 30, height: 16, borderRadius: 999,
        background: on ? 'var(--accent)' : 'var(--bg-elev-2)',
        border: `1px solid ${on ? 'var(--accent)' : 'var(--line)'}`,
        position: 'relative', cursor: 'pointer',
      }}>
        <div style={{
          position: 'absolute', top: 1, left: on ? 15 : 1,
          width: 12, height: 12, borderRadius: '50%',
          background: 'white', transition: 'left 0.15s',
        }} />
      </div>
    </div>
  );
}

function EmotionBar({ dist }) {
  const colors = { neutral: 'var(--ink-mute)', happy: 'var(--ok)', focused: 'var(--info)', frustrated: 'var(--err)', tired: 'var(--warn)', calm: '#a78bfa' };
  return (
    <>
      <div style={{ display: 'flex', height: 14, borderRadius: 4, overflow: 'hidden', border: '1px solid var(--line)' }}>
        {Object.entries(dist).map(([k, v]) => (
          <div key={k} style={{ flex: v, background: colors[k] || 'var(--ink-mute)' }} title={`${k}: ${v}%`} />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 10, marginTop: 6, flexWrap: 'wrap' }}>
        {Object.entries(dist).map(([k, v]) => (
          <span key={k} className="mono" style={{ fontSize: 10, color: 'var(--ink-dim)' }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, background: colors[k] || 'var(--ink-mute)', marginRight: 4, borderRadius: 1 }} />
            {k} <span style={{ color: 'var(--ink)' }}>{v}%</span>
          </span>
        ))}
      </div>
    </>
  );
}

function UserCard({ u }) {
  return (
    <Card title={u.name} meta={`${u.samples} samples · enrolled ${u.lastEnroll}`} action={<button className="btn">re-enroll</button>}>
      <SectionBar title="Permisos por dominio" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0 24px' }}>
        {Object.entries(u.permissions).map(([k, v]) => <PermToggle key={k} on={v} label={k} />)}
      </div>

      <SectionBar title="Distribución emocional" meta="wav2vec2" />
      <EmotionBar dist={u.emotions} />

      <SectionBar title="Comandos top" />
      <table className="k">
        <tbody>
          {u.topCommands.map(c => (
            <tr key={c.cmd}>
              <td className="mono" style={{ fontSize: 11 }}>{c.cmd}</td>
              <td className="num">{c.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function UsersView() {
  return (
    <div className="view" id="view-users">
      <PageHead
        title="Usuarios"
        subtitle="Perfiles de voz ECAPA · permisos por dominio · análisis de uso"
        meta={<><span>{USERS.length} enrolled</span> · <span>SpeakerID activo</span></>}
        actions={<button className="btn primary">+ enroll usuario</button>}
      />

      <SectionBar title="Embedding space (PCA 2D)" meta="ECAPA-TDNN · proyectado a 2D para visualización" />
      <Card>
        <EmbeddingScatter users={USERS} />
        <div className="mute mono" style={{ fontSize: 10, marginTop: 4 }}>
          clusters separados → buena discriminación · superposición = posible confusión de SpeakerID
        </div>
      </Card>

      <SectionBar title="Perfiles" />
      <div className="grid grid-3">
        {USERS.map(u => <UserCard key={u.id} u={u} />)}
      </div>
    </div>
  );
}

window.UsersView = UsersView;
