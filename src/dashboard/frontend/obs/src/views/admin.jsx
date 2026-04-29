// =============================================================
// Admin view — centro de administración
// CRUD usuarios + voice enrollment + alerts ack + service restart
// =============================================================

const { Card, Pill } = window.KZA_PRIMS || {};

const PERMISSION_LEVELS = ["guest", "child", "teen", "adult", "admin"];
const RESTARTABLE = ["kza-voice", "kza-llm-ik"];
const ENROLL_SAMPLES = 3;
const ENROLL_PHRASES = [
  "Hola, soy yo grabando una muestra para KZA",
  "Encendé las luces del living",
  "Cuál es la temperatura de afuera",
];

// ---------------- Auth helpers ----------------

async function authedFetch(path, opts = {}) {
  const r = await fetch(path, { credentials: "same-origin", ...opts });
  if (r.status === 401) {
    window.dispatchEvent(new CustomEvent("kza:auth-required"));
    throw new Error("auth required");
  }
  return r;
}

function useAuth() {
  const [authed, setAuthed] = React.useState(null);
  const check = React.useCallback(async () => {
    try {
      const r = await fetch("/api/admin/auth/whoami", { credentials: "same-origin" });
      setAuthed(r.ok);
    } catch {
      setAuthed(false);
    }
  }, []);
  React.useEffect(() => { check(); }, [check]);
  React.useEffect(() => {
    const onReq = () => setAuthed(false);
    window.addEventListener("kza:auth-required", onReq);
    return () => window.removeEventListener("kza:auth-required", onReq);
  }, []);
  return { authed, refresh: check };
}

function LoginModal({ onLogin }) {
  const [token, setToken] = React.useState("");
  const [err, setErr] = React.useState(null);
  const submit = async (e) => {
    e.preventDefault();
    const r = await fetch("/api/admin/auth/login", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (r.ok) { onLogin(); }
    else { setErr("Token inválido"); }
  };
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)",
      display: "grid", placeItems: "center", zIndex: 100,
    }}>
      <form onSubmit={submit} className="card" style={{ width: 380, padding: 0 }}>
        <div className="card-h"><h3>Acceso admin</h3></div>
        <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <p className="dim" style={{ fontSize: 12, margin: 0 }}>
            Token configurado en <span className="mono">KZA_DASHBOARD_TOKEN</span>.
          </p>
          <input type="password" value={token} onChange={e => setToken(e.target.value)}
                 autoFocus placeholder="paste token"
                 style={{ background: "var(--bg-elev-2)", border: "1px solid var(--line)",
                          color: "var(--ink)", borderRadius: 5, padding: "8px 10px",
                          fontFamily: "var(--mono)", fontSize: 12 }} />
          {err && <div className="pill err">{err}</div>}
          <button type="submit" className="btn primary">Entrar</button>
        </div>
      </form>
    </div>
  );
}

// ---------------- Audio recorder ----------------

async function recordWavSample(durationSec = 3) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(proc); proc.connect(ctx.destination);
  await new Promise((res) => setTimeout(res, durationSec * 1000));
  proc.disconnect(); src.disconnect();
  stream.getTracks().forEach(t => t.stop());
  await ctx.close();

  // concat + downsample 48k → 16k (factor 3, average)
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.length; }
  const targetLen = Math.floor(merged.length / 3);
  const ds = new Int16Array(targetLen);
  for (let i = 0; i < targetLen; i++) {
    const a = (merged[i*3] + merged[i*3+1] + merged[i*3+2]) / 3;
    ds[i] = Math.max(-32768, Math.min(32767, Math.round(a * 32767)));
  }
  return encodeWav(ds, 16000);
}

function encodeWav(int16Samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + int16Samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + int16Samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);  // PCM
  view.setUint16(22, 1, true);  // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, int16Samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < int16Samples.length; i++, off += 2) {
    view.setInt16(off, int16Samples[i], true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------- Users panel ----------------

function UsersPanel({ users, onChanged }) {
  const [name, setName] = React.useState("");
  const [level, setLevel] = React.useState("adult");
  const [err, setErr] = React.useState(null);
  const create = async (e) => {
    e.preventDefault();
    setErr(null);
    try {
      const r = await authedFetch("/api/admin/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, permission_level: level }),
      });
      if (!r.ok) { setErr((await r.json()).detail || `HTTP ${r.status}`); return; }
      setName(""); onChanged();
    } catch (e) { setErr(String(e)); }
  };
  const del = async (uid) => {
    if (!confirm(`¿Eliminar ${uid}?`)) return;
    await authedFetch(`/api/admin/users/${uid}`, { method: "DELETE" });
    onChanged();
  };

  return (
    <div className="card">
      <div className="card-h"><h3>Usuarios</h3>
        <span className="h-meta"><span>{users.length} enrolled</span></span>
      </div>
      <div className="card-b">
        <form onSubmit={create} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <input value={name} onChange={e => setName(e.target.value)}
                 placeholder="nombre" required
                 style={{ flex: 1, background: "var(--bg-elev-2)", border: "1px solid var(--line)",
                          color: "var(--ink)", borderRadius: 5, padding: "6px 10px",
                          fontFamily: "var(--mono)", fontSize: 12 }} />
          <select value={level} onChange={e => setLevel(e.target.value)}
                  style={{ background: "var(--bg-elev-2)", border: "1px solid var(--line)",
                           color: "var(--ink)", borderRadius: 5, padding: "6px 10px",
                           fontFamily: "var(--mono)", fontSize: 12 }}>
            {PERMISSION_LEVELS.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
          <button className="btn primary" type="submit">+ crear</button>
        </form>
        {err && <div className="pill err" style={{ marginBottom: 8 }}>{err}</div>}
        <table className="k">
          <thead><tr><th>id</th><th>nombre</th><th>permiso</th><th>voz</th><th></th></tr></thead>
          <tbody>
            {users.map(u => (
              <tr key={u.id}>
                <td className="mono">{u.id}</td>
                <td>{u.name}</td>
                <td>{u.permissions ? Object.entries(u.permissions).filter(([,v]) => v).map(([k]) => k).join(", ") : "—"}</td>
                <td>{u.samples > 0 ? <Pill kind="ok">enrolado</Pill> : <Pill>sin voz</Pill>}</td>
                <td style={{ display: "flex", gap: 6 }}>
                  <EnrollButton userId={u.id} onDone={onChanged} />
                  <button className="btn danger" onClick={() => del(u.id)}>×</button>
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr><td colSpan={5} className="mute" style={{ textAlign: "center", padding: 24 }}>
                Sin usuarios — creá uno arriba
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EnrollButton({ userId, onDone }) {
  const [open, setOpen] = React.useState(false);
  const [step, setStep] = React.useState(0);  // 0..ENROLL_SAMPLES
  const [recording, setRecording] = React.useState(false);
  const [samples, setSamples] = React.useState([]);
  const [err, setErr] = React.useState(null);
  const [submitting, setSubmitting] = React.useState(false);

  const reset = () => { setStep(0); setRecording(false); setSamples([]); setErr(null); setSubmitting(false); };

  const captureOne = async () => {
    setErr(null); setRecording(true);
    try {
      const blob = await recordWavSample(3);
      setSamples(s => [...s, blob]);
      setStep(s => s + 1);
    } catch (e) { setErr(String(e.message || e)); }
    finally { setRecording(false); }
  };

  const submit = async () => {
    setSubmitting(true); setErr(null);
    const fd = new FormData();
    samples.forEach((b, i) => fd.append("samples", b, `sample_${i}.wav`));
    try {
      const r = await authedFetch(`/api/admin/users/${userId}/enroll`, {
        method: "POST", body: fd,
      });
      if (!r.ok) { setErr((await r.json()).detail || `HTTP ${r.status}`); return; }
      setOpen(false); reset(); onDone();
    } catch (e) { setErr(String(e)); }
    finally { setSubmitting(false); }
  };

  if (!open) {
    return <button className="btn" onClick={() => { reset(); setOpen(true); }}>🎙 enrolar</button>;
  }
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)",
      display: "grid", placeItems: "center", zIndex: 90,
    }}>
      <div className="card" style={{ width: 480 }}>
        <div className="card-h">
          <h3>Enrolar voz · {userId}</h3>
          <button className="btn" onClick={() => setOpen(false)}>×</button>
        </div>
        <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="mute mono" style={{ fontSize: 11 }}>
            Muestra {Math.min(step + 1, ENROLL_SAMPLES)} de {ENROLL_SAMPLES}
          </div>
          {step < ENROLL_SAMPLES ? (
            <>
              <div className="wire" style={{ padding: 12, background: "var(--bg-elev-2)" }}>
                <div className="mute mono" style={{ fontSize: 10, marginBottom: 4 }}>FRASE</div>
                <div style={{ fontSize: 14 }}>"{ENROLL_PHRASES[step]}"</div>
              </div>
              <button className="btn primary" disabled={recording} onClick={captureOne}>
                {recording ? "🔴 grabando 3s…" : `🎙 grabar muestra ${step + 1}`}
              </button>
            </>
          ) : (
            <>
              <div className="pill ok">{ENROLL_SAMPLES} muestras capturadas</div>
              <button className="btn primary" disabled={submitting} onClick={submit}>
                {submitting ? "subiendo…" : "guardar embedding"}
              </button>
            </>
          )}
          {err && <div className="pill err">{err}</div>}
        </div>
      </div>
    </div>
  );
}

// ---------------- Services panel ----------------

function ServicesPanel() {
  const [busy, setBusy] = React.useState(null);
  const [msg, setMsg] = React.useState(null);
  const restart = async (name) => {
    if (!confirm(`Reiniciar ${name}?`)) return;
    setBusy(name); setMsg(null);
    try {
      const r = await authedFetch(`/api/admin/services/${name}/restart`, { method: "POST" });
      const j = await r.json();
      setMsg(r.ok ? { kind: "ok", text: `${name} reiniciado` }
                  : { kind: "err", text: j.detail || `HTTP ${r.status}` });
    } catch (e) { setMsg({ kind: "err", text: String(e) }); }
    finally { setBusy(null); }
  };
  return (
    <div className="card">
      <div className="card-h"><h3>Acciones de servicio</h3></div>
      <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {RESTARTABLE.map(svc => (
          <div key={svc} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="mono">{svc}</span>
            <button className="btn" disabled={busy === svc} onClick={() => restart(svc)}>
              {busy === svc ? "reiniciando…" : "↻ restart"}
            </button>
          </div>
        ))}
        {msg && <div className={`pill ${msg.kind}`}>{msg.text}</div>}
      </div>
    </div>
  );
}

// ---------------- AdminView (entry point) ----------------

function AdminView() {
  const { authed, refresh } = useAuth();
  const [users, setUsers] = React.useState([]);
  const loadUsers = React.useCallback(async () => {
    try {
      const r = await fetch("/api/users");
      if (r.ok) setUsers(await r.json());
    } catch {}
  }, []);
  React.useEffect(() => { if (authed) loadUsers(); }, [authed, loadUsers]);

  if (authed === null) return <div className="content mute">cargando…</div>;
  if (authed === false) return <LoginModal onLogin={refresh} />;

  return (
    <div className="view">
      <div className="page-head">
        <div>
          <h1>Admin</h1>
          <div className="subtitle">Gestión de usuarios, voz, servicios — endpoints autenticados</div>
        </div>
        <button className="btn" onClick={async () => {
          await fetch("/api/admin/auth/logout", { method: "POST", credentials: "same-origin" });
          refresh();
        }}>cerrar sesión</button>
      </div>
      <div className="grid grid-2">
        <UsersPanel users={users} onChanged={loadUsers} />
        <ServicesPanel />
      </div>
    </div>
  );
}

window.AdminView = AdminView;
