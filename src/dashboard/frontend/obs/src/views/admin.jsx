// =============================================================
// Admin view — centro de administración
// CRUD usuarios + voice enrollment + alerts ack + service restart
// Card/Pill son globals (declarados en primitives.jsx — Babel inline scripts
// comparten scope global). No destructuro de window.* para no shadowearlos.
// =============================================================

const PERMISSION_LEVELS = ["guest", "child", "teen", "adult", "admin"];
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

async function recordWavSample(durationSec = 3, owner) {
  if (!owner) throw new Error('La captura no está disponible.');
  return await owner.recordWavSample(durationSec, encodeWav);
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

function UsersPanel({ users, loadError, onChanged }) {
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
    // U3b-1b/D7: the removal also deletes the memories assigned to that person.
    if (!confirm(`¿Eliminar ${uid}? También se eliminan, sin posibilidad de recuperarlos, los recuerdos asignados a esta persona en Memoria, junto con los recuerdos que dependen de ellos.`)) return;
    setErr(null);
    try {
      const r = await authedFetch(`/api/admin/users/${encodeURIComponent(uid)}`, { method: "DELETE" });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        setErr(typeof body.detail === "string" ? body.detail : `No se pudo eliminar (HTTP ${r.status}).`);
      }
    } catch (e) { setErr(String(e)); }
    onChanged();
  };

  return (
    <div className="card">
      <div className="card-h"><h3>Usuarios</h3>
        <span className="h-meta"><span>{loadError ? "no disponible" : `${users.length} usuarios`}</span></span>
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
                <td>{u.permissions && Object.values(u.permissions).some(Boolean) ? Object.entries(u.permissions).filter(([,v]) => v).map(([k]) => k).join(", ") : <span className="mute">no observado</span>}</td>
                <td>{typeof u.samples !== "number" ? <span className="mute">no observado</span> : u.samples > 0 ? <Pill kind="ok">enrolado</Pill> : <Pill>sin voz</Pill>}</td>
                <td style={{ display: "flex", gap: 6 }}>
                  <EnrollButton userId={u.id} onDone={onChanged} />
                  <button className="btn danger" onClick={() => del(u.id)}>×</button>
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr><td colSpan={5} className="mute" style={{ textAlign: "center", padding: 24 }}>
                {loadError ? `Usuarios no disponibles: ${loadError}` : "Sin usuarios — creá uno arriba"}
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
  const [err, setErr] = React.useState(null);
  const [submitting, setSubmitting] = React.useState(false);
  const ownerRef = React.useRef(null);
  const liveRef = React.useRef(true);

  const reset = () => { setStep(0); setRecording(false); setErr(null); setSubmitting(false); };

  React.useEffect(() => {
    liveRef.current = true;
    return () => {
      liveRef.current = false;
      const owner = ownerRef.current;
      ownerRef.current = null;
      owner?.stop().catch(() => {});
    };
  }, []);

  const close = () => {
    const owner = ownerRef.current;
    ownerRef.current = null;
    setOpen(false); reset();
    owner?.stop().catch(() => {});
  };

  const begin = () => {
    reset();
    if (!window.KzaCaptureRecorder) {
      setErr('La captura no está disponible.'); setOpen(true); return;
    }
    const owner = window.KzaCaptureRecorder.createModalOwner({
      onRetiring() {
        if (liveRef.current && ownerRef.current === owner) {
          ownerRef.current = null; setOpen(false); reset();
        }
      },
    });
    ownerRef.current = owner;
    setOpen(true);
  };

  const captureOne = async () => {
    const owner = ownerRef.current;
    if (!owner) { setErr('La captura no está disponible.'); return; }
    setErr(null); setRecording(true);
    try {
      await recordWavSample(3, owner);
      if (!liveRef.current || ownerRef.current !== owner || !owner.current()) return;
      setStep(owner.sampleCount());
    } catch (_) {
      if (liveRef.current && ownerRef.current === owner) setErr('La captura no está disponible.');
    } finally {
      if (liveRef.current && ownerRef.current === owner) setRecording(false);
    }
  };

  const submit = async () => {
    const owner = ownerRef.current;
    if (!owner || !owner.current()) return;
    setSubmitting(true); setErr(null);
    try {
      const r = await owner.uploadSamples(`/api/admin/users/${userId}/enroll`);
      if (!liveRef.current || ownerRef.current !== owner || !owner.current()) return;
      if (!r.ok) {
        const body = await r.json();
        if (liveRef.current && ownerRef.current === owner && owner.current()) {
          setErr(body.detail || `HTTP ${r.status}`);
        }
        return;
      }
      await owner.stop();
      if (liveRef.current) onDone();
    } catch (_) {
      if (liveRef.current && ownerRef.current === owner) setErr('No se pudo completar la inscripción.');
    } finally {
      if (liveRef.current && ownerRef.current === owner) setSubmitting(false);
    }
  };

  if (!open) {
    return <button className="btn" onClick={begin}>🎙 enrolar</button>;
  }
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)",
      display: "grid", placeItems: "center", zIndex: 90,
    }}>
      <div className="card" style={{ width: 480 }}>
        <div className="card-h">
          <h3>Enrolar voz · {userId}</h3>
          <button className="btn" onClick={close}>×</button>
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
  return (
    <div className="card">
      <div className="card-h"><h3>Acciones de servicio</h3></div>
      <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {window.KzaObsData.RESTARTABLE_SERVICES.map(svc => (
          <div key={svc} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
            <span className="mono">{svc}</span>
            <OperationButton kind="restartService" target={svc} label="↻ Reiniciar" confirmLabel={`Confirmar reinicio de ${svc}`} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------- AdminView (entry point) ----------------

function AdminView() {
  const { authed, refresh } = useAuth();
  const [users, setUsers] = React.useState([]);
  const [loadError, setLoadError] = React.useState(null);
  const loadUsers = React.useCallback(async () => {
    try {
      const r = await fetch("/api/users", { credentials: "same-origin", cache: "no-store" });
      const source = (r.headers.get("x-kza-source") || "").toLowerCase();
      if (r.ok && (source === "real" || source === "mock")) { setUsers(await r.json()); setLoadError(null); }
      else { setUsers([]); setLoadError(r.ok ? "la fuente no está disponible" : `HTTP ${r.status}`); }
    } catch {
      setUsers([]); setLoadError("sin respuesta del servidor");
    }
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
          window.dispatchEvent(new CustomEvent("kza:auth-required"));
          refresh();
        }}>cerrar sesión</button>
      </div>
      <div className="grid grid-2">
        <UsersPanel users={users} loadError={loadError} onChanged={loadUsers} />
        <ServicesPanel />
      </div>
    </div>
  );
}

window.AdminView = AdminView;
