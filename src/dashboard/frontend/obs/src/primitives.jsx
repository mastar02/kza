// =============================================================
// Reusable primitives for KZA dashboard
// =============================================================

const { useState, useEffect, useRef, useMemo } = React;

// ---- Sparkline ----
function Sparkline({ data, height = 28, stroke = 'currentColor', fill = true, strokeWidth = 1.5 }) {
  if (!data || !data.length) return null;
  const w = 100, h = 28;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 2) - 1;
    return [x, y];
  });
  const d = pts.map((p, i) => (i === 0 ? `M${p[0].toFixed(1)},${p[1].toFixed(1)}` : `L${p[0].toFixed(1)},${p[1].toFixed(1)}`)).join(' ');
  const fillD = fill ? `${d} L${w},${h} L0,${h} Z` : null;
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ height, color: stroke }}>
      {fill && <path d={fillD} fill="currentColor" opacity="0.12" />}
      <path d={d} fill="none" stroke="currentColor" strokeWidth={strokeWidth} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// ---- Tiny bar chart ----
function BarChart({ data, height = 60, color = 'var(--accent)', labels = false }) {
  const max = Math.max(...data.map(d => typeof d === 'object' ? d.count : d), 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height, width: '100%' }}>
      {data.map((d, i) => {
        const v = typeof d === 'object' ? d.count : d;
        const lbl = typeof d === 'object' ? d.day : '';
        return (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
            <div style={{
              width: '100%',
              height: `${(v / max) * (height - (labels ? 14 : 0))}px`,
              minHeight: 2,
              background: v ? color : 'var(--line)',
              borderRadius: 2,
              opacity: v ? 0.85 : 0.4,
            }} title={`${lbl}: ${v}`}/>
            {labels && <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--ink-mute)' }}>{lbl}</div>}
          </div>
        );
      })}
    </div>
  );
}

// ---- Waveform ----
function Waveform({ live = false, bars = 32, height = 24 }) {
  const heights = useMemo(() => Array.from({ length: bars }, (_, i) => {
    const seed = (Math.sin(i * 1.7) + Math.cos(i * 0.4)) * 0.5 + 0.5;
    return 0.3 + seed * 0.7;
  }), [bars]);
  return (
    <div className={`wf ${live ? 'live' : ''}`} style={{ height }}>
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

// ---- LogViewer ----
function LogViewer({ lines, filter = null, height = 380 }) {
  const filtered = filter ? lines.filter(l => l.svc === filter || l.lvl === filter) : lines;
  return (
    <div className="logs" style={{ maxHeight: height }}>
      {filtered.map((l, i) => (
        <div className="ll" key={i}>
          <span className="ts">{l.ts}</span>
          <span className={`lvl ${l.lvl}`}>{l.lvl}</span>
          <span className="svc">{l.svc}</span>
          <span className="msg">{l.msg}</span>
        </div>
      ))}
    </div>
  );
}

// ---- GPU Meter ----
function GpuMeter({ label, value, max = 100, unit = '%', color }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="meter">
      <div className="label">{label}</div>
      <div className="bar"><i style={{ width: `${pct}%`, background: color || 'var(--accent)' }} /></div>
      <div className="val">{value}{unit ? ` ${unit}` : ''}</div>
    </div>
  );
}

// ---- Embedding scatter (PCA 2D) ----
function EmbeddingScatter({ users, height = 220 }) {
  const w = 400, h = height;
  const colors = ['var(--accent)', '#22c55e', '#38bdf8', '#f59e0b', '#ec4899'];
  // Map [-0.5..0.5] → padded plot
  const map = (x, y) => [50 + (x + 0.5) * (w - 100), h - 30 - (y + 0.5) * (h - 60)];
  return (
    <svg className="scatter" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {/* Axes */}
      <line x1="40" y1={h-25} x2={w-20} y2={h-25} stroke="var(--line)" strokeWidth="1" />
      <line x1="40" y1="20" x2="40" y2={h-25} stroke="var(--line)" strokeWidth="1" />
      <text x="44" y="16" fontSize="9" fill="var(--ink-mute)" fontFamily="var(--mono)">PC2</text>
      <text x={w-30} y={h-12} fontSize="9" fill="var(--ink-mute)" fontFamily="var(--mono)">PC1</text>
      {/* Cluster centers + points */}
      {users.map((u, ui) => {
        const c = colors[ui % colors.length];
        const cx = u.pca.reduce((s, p) => s + p[0], 0) / u.pca.length;
        const cy = u.pca.reduce((s, p) => s + p[1], 0) / u.pca.length;
        const [ccx, ccy] = map(cx, cy);
        return (
          <g key={u.id}>
            {/* cluster halo */}
            <circle cx={ccx} cy={ccy} r="34" fill={c} opacity="0.07" />
            {/* points */}
            {u.pca.map((p, i) => {
              const [x, y] = map(p[0], p[1]);
              return <circle key={i} cx={x} cy={y} r="3.5" fill={c} opacity="0.8" />;
            })}
            <text x={ccx + 24} y={ccy - 8} fontSize="11" fill="var(--ink)" fontFamily="var(--sans)" fontWeight="500">{u.name}</text>
            <text x={ccx + 24} y={ccy + 4} fontSize="9" fill="var(--ink-mute)" fontFamily="var(--mono)">{u.samples} samples</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---- Latency histogram (ms breakdown, stacked horiz bar) ----
function LatencyBreakdown({ stages }) {
  const total = stages.reduce((s, x) => s + x.ms, 0);
  const colors = ['#7c3aed', '#a855f7', '#c084fc', '#d8b4fe', '#e9d5ff', '#f3e8ff'];
  return (
    <div>
      <div style={{ display: 'flex', height: 18, borderRadius: 4, overflow: 'hidden', border: '1px solid var(--line)' }}>
        {stages.map((s, i) => (
          <div key={s.stage} style={{
            flex: s.ms,
            background: colors[i % colors.length],
            display: 'grid', placeItems: 'center',
            fontFamily: 'var(--mono)', fontSize: 9, color: '#0b0b0d', fontWeight: 600,
          }} title={`${s.stage}: ${s.ms}ms`}>
            {(s.ms/total*100) > 8 ? `${s.stage}` : ''}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
        {stages.map((s, i) => (
          <div key={s.stage} style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'var(--mono)', fontSize: 10 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: colors[i % colors.length] }} />
            <span style={{ color: 'var(--ink-dim)' }}>{s.stage}</span>
            <span style={{ color: 'var(--ink)' }}>{s.ms}ms</span>
          </div>
        ))}
        <div style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--ink-mute)' }}>
          total <b style={{ color: 'var(--ink)' }}>{total}ms</b>
        </div>
      </div>
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
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        {meta && <div className="meta">{meta}</div>}
        {actions}
      </div>
    </div>
  );
}

// ---- Annotation note (sketchy hand-drawn) ----
function HandNote({ children, top, left, right, bottom, rotate = -3 }) {
  if (!window.__showAnnotations) return null;
  return (
    <div className="hand" style={{
      position: 'absolute',
      top, left, right, bottom,
      transform: `rotate(${rotate}deg)`,
      fontSize: 15,
      color: 'var(--accent)',
      pointerEvents: 'none',
      whiteSpace: 'nowrap',
      zIndex: 5,
    }}>
      {children}
    </div>
  );
}

// ---- Cooldown countdown ----
function Countdown({ until }) {
  // until is "HH:MM:SS"; we just display delta from a fake "now" 14:32:31
  const now = '14:32:31';
  const [hh, mm, ss] = until.split(':').map(Number);
  const [nh, nm, ns] = now.split(':').map(Number);
  const delta = (hh*3600+mm*60+ss) - (nh*3600+nm*60+ns);
  if (delta <= 0) return <span className="mono mute">expirado</span>;
  const m = Math.floor(delta/60), s = delta % 60;
  return <span className="mono" style={{ color: 'var(--warn)' }}>{m}m {String(s).padStart(2,'0')}s</span>;
}

Object.assign(window, {
  Sparkline, BarChart, Waveform, Pill, Card, SectionBar,
  LogViewer, GpuMeter, EmbeddingScatter, LatencyBreakdown,
  PageHead, HandNote, Countdown,
});
