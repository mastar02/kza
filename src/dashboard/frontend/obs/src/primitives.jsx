// =============================================================
// Reusable primitives for the KZA technical panel
// =============================================================

const { useState, useEffect, useRef, useMemo } = React;

// ---- Observed data subscription ----
function useObs() {
  const [, setVersion] = useState(0);
  useEffect(() => window.__kzaObs.subscribe(() => setVersion(value => value + 1)), []);
  return window.__kzaObs;
}

// ---- Waveform (decorative animation for a real "speaking" state) ----
function Waveform({ live = false, bars = 32, height = 24 }) {
  const heights = useMemo(() => Array.from({ length: bars }, (_, i) => {
    const seed = (Math.sin(i * 1.7) + Math.cos(i * 0.4)) * 0.5 + 0.5;
    return 0.3 + seed * 0.7;
  }), [bars]);
  return (
    <div className={`wf ${live ? 'live' : ''}`} style={{ height }} aria-hidden="true">
      {heights.map((h, i) => (
        <span key={i} style={{ height: `${h * 100}%`, animationDelay: `${i * 0.04}s` }} />
      ))}
    </div>
  );
}

// ---- Pill ----
function Pill({ kind = '', children }) {
  return <span className={`pill ${kind}`}>{children}</span>;
}

// ---- Card ----
function Card({ title, meta, children, action, style, className = '' }) {
  return (
    <div className={`card ${className}`} style={style}>
      {(title || meta || action) && (
        <div className="card-h">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {title && <h3>{title}</h3>}
            {meta && <span className="h-meta">{meta}</span>}
          </div>
          {action}
        </div>
      )}
      <div className="card-b">{children}</div>
    </div>
  );
}

// ---- Explicit unavailable state ----
function Unavailable({ children }) {
  return <p className="mute" role="status" style={{ margin: 0 }}>{children}</p>;
}

// ---- Section heading bar ----
function SectionBar({ title, meta, action }) {
  return (
    <div className="h-bar">
      <h2>{title}</h2>
      <div className="line" />
      {meta && <span className="meta">{meta}</span>}
      {action}
    </div>
  );
}

// ---- Source provenance for one collection ----
function SourceNote({ k }) {
  const obs = useObs();
  const entry = obs.entry(k);
  const data = window.KzaObsData;
  const status = obs.status(k);
  const parts = [`fuente ${entry.sourceId || 'no declarada'}`, status];
  if (status === data.STATUS.REAL) {
    // Re-rendered by the store at expiry, so 'vigente' turns 'vencida' without a new read.
    parts.push(`observado ${data.formatInstant(entry.observedAt)}`, obs.freshness(k));
  }
  return <span className="mono mute" style={{ fontSize: 10 }}>{parts.join(' · ')}</span>;
}

// ---- Authorized operation with pending state and server result ----
function OperationButton({ kind, target, label, confirmLabel = null, className = 'btn' }) {
  const obs = useObs();
  const [phase, setPhase] = useState('idle');
  const live = useRef(true);
  useEffect(() => () => { live.current = false; }, []);
  const allowed = window.KzaObsData.availability(obs, kind, target);
  const reasonId = useMemo(() => `op-${kind}-${String(target).replace(/[^a-zA-Z0-9_-]/g, '_')}`, [kind, target]);
  async function run() {
    setPhase('pending'); obs.setNotice(null);
    const outcome = await window.KzaObsData.runOperation({ fetch: window.fetch.bind(window), store: obs, kind, target });
    // The row may disappear after the fresh read, so the result lives in the shell.
    obs.setNotice(outcome);
    if (live.current) setPhase('idle');
    // After an applied operation with an expired session, keep its result visible.
    if (outcome.state === 'auth') window.dispatchEvent(new CustomEvent('kza:auth-required'));
  }
  const describedBy = !allowed.enabled ? reasonId : undefined;
  return (
    <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start' }}>
      <span style={{ display: 'inline-flex', gap: 4 }}>
        {phase === 'confirm' ? (
          <>
            <button type="button" className="btn danger" onClick={run}>{confirmLabel}</button>
            <button type="button" className="btn" onClick={() => setPhase('idle')}>Cancelar</button>
          </>
        ) : (
          <button type="button" className={className} disabled={!allowed.enabled || phase === 'pending'}
                  aria-describedby={describedBy}
                  onClick={() => (confirmLabel ? setPhase('confirm') : run())}>
            {phase === 'pending' ? 'Enviando…' : label}
          </button>
        )}
      </span>
      {!allowed.enabled && <span id={reasonId} className="mono mute" style={{ fontSize: 10 }}>{allowed.reason}</span>}
    </span>
  );
}

// ---- Latest operation result, kept outside the row that started it ----
function OperationNotice() {
  const obs = useObs();
  const notice = obs.notice();
  const kind = !notice ? '' : notice.state === 'confirmed' ? 'ok' : notice.state === 'uncertain' ? 'warn' : 'err';
  return (
    <div role="status" aria-live="polite" style={{ margin: notice ? '12px 24px 0' : 0 }}>
      {notice && <span className={`pill ${kind}`} style={{ whiteSpace: 'normal' }}>{notice.message}</span>}
    </div>
  );
}

// ---- Control without a backing operation ----
function UnavailableButton({ label, reason, id }) {
  return (
    <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start' }}>
      <button type="button" className="btn" disabled aria-describedby={id}>{label}</button>
      <span id={id} className="mono mute" style={{ fontSize: 10 }}>{reason}</span>
    </span>
  );
}

// ---- GPU Meter ----
function GpuMeter({ label, value, max = 100, unit = '%', color }) {
  if (typeof value !== 'number') {
    return <div className="meter"><div className="label">{label}</div><div className="mute mono">no disponible</div><div /></div>;
  }
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="meter">
      <div className="label">{label}</div>
      <div className="bar"><i style={{ width: `${pct}%`, background: color || 'var(--accent)' }} /></div>
      <div className="val">{value}{unit ? ` ${unit}` : ''}</div>
    </div>
  );
}

// ---- Page header ----
function PageHead({ title, subtitle, meta, actions }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {subtitle && <div className="subtitle">{subtitle}</div>}
      </div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        {meta && <div className="meta">{meta}</div>}
        {actions}
      </div>
    </div>
  );
}

// ---- Cooldown end: countdown only from a zone-explicit instant ----
function Countdown({ until, fallback }) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const delta = window.KzaObsData.remainingUntil(until, now);
  if (delta === null) {
    // Server-local clock time without a declared zone: show it as such, no countdown.
    return fallback ? <span className="mono mute">hasta {fallback} (hora del servidor, zona no declarada)</span>
      : <span className="mono mute">no disponible</span>;
  }
  if (delta <= 0) return <span className="mono mute">vencido</span>;
  const m = Math.floor(delta / 60), s = delta % 60;
  return <span className="mono" style={{ color: 'var(--warn)' }}
    title={`hasta ${window.KzaObsData.formatInstant(until)}`}>{m}m {String(s).padStart(2, '0')}s</span>;
}

// ---- Event time with an explicit zone ----
function EventTime({ at, ts }) {
  if (at) return <span className="mono mute" style={{ fontSize: 10 }}>{window.KzaObsData.formatInstant(at)}</span>;
  return <span className="mono mute" style={{ fontSize: 10 }}>{ts ? `${ts} (zona no declarada)` : '—'}</span>;
}

Object.assign(window, {
  useObs, Waveform, Pill, Card, Unavailable, SectionBar, SourceNote,
  OperationButton, OperationNotice, UnavailableButton, GpuMeter, PageHead, Countdown, EventTime,
});
