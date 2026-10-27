// Real ambient control: no sample data and no dependency on a spoken command.
function PrivacyView() {
  const [status,setStatus] = React.useState(null);
  const [message,setMessage] = React.useState('Consultando captura ambiental…');
  const action = React.useRef(null);
  React.useEffect(() => {
    let active = true, pending = false, sequence = 0;
    const requests = new Set();
    async function readOrChange(paused,revision) {
      if (!active || pending) return;
      pending = true;
      const attempt = ++sequence, controller = new AbortController(); requests.add(controller);
      try {
        const change = typeof paused === 'boolean';
        const response = await fetch('/api/ambient/control',{credentials:'same-origin',cache:'no-store',signal:controller.signal,
          ...(change ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paused,expected_revision:revision})} : {})});
        if (!response.ok) throw Error(response.status===409 ? 'La revisión cambió. Actualizá el estado antes de reintentar.' : 'La captura ambiental no está disponible para esta sesión.');
        const data = await response.json();
        if (!data || !Number.isInteger(data.revision) || data.revision<0 || typeof data.admission_paused!=='boolean'
            || typeof data.persisted_paused!=='boolean'
            || !['initializing','running','pausing','paused','resuming','failed','closed'].includes(data.phase)
            || !Number.isInteger(data.pending_operations) || data.pending_operations<0) throw Error('No se pudo verificar el estado de captura.');
        if (active && sequence===attempt) { setStatus(data); setMessage(''); }
      } catch (error) {
        if (active && sequence===attempt) { setStatus(null); setMessage(error.name==='AbortError' ? 'Consulta cancelada.' : error.message); }
      } finally { requests.delete(controller); pending = false; }
    }
    action.current = readOrChange;
    readOrChange();
    const timer = setInterval(() => readOrChange(),1000);
    return () => {active=false;sequence++;clearInterval(timer);for (const request of requests) request.abort();requests.clear();action.current=null;};
  },[]);
  const labels = {initializing:'Preparando captura',running:'Captura ambiental habilitada',pausing:'Pausando: esperando el trabajo iniciado',
    paused:'Captura ambiental pausada',resuming:'Reactivando captura ambiental',failed:'No se pudo completar el cambio; la entrada ambiental sigue bloqueada',closed:'Control cerrado'};
  const waiting = !status || ['initializing','pausing','resuming','closed'].includes(status.phase);
  return <div id="view-privacy">
    <header className="page-head"><div><h1>Privacidad</h1><p className="mute">Control local de la captura ambiental.</p></div></header>
    <section className="card" aria-label="Captura ambiental">
      <h2>Captura ambiental</h2>
      <p role="status">{message || labels[status?.phase]}</p>
      {status?.pending_operations>0 && <p className="mute">Hay procesamiento anterior que todavía debe terminar.</p>}
      <p>La pausa detiene la transcripción ambiental y el aprendizaje que depende de ella. Podés seguir usando la palabra de activación y los comandos de voz.</p>
      <p>Los datos anteriores siguen sus vencimientos. Pausar no equivale a borrarlos.</p>
      <div className="row">
        <button type="button" className="btn" disabled={waiting || status?.phase==='paused'} onClick={() => action.current?.(true,status.revision)}>Pausar captura ambiental</button>
        <button type="button" className="btn" disabled={waiting || status?.phase==='running'} onClick={() => action.current?.(false,status.revision)}>Reactivar captura ambiental</button>
      </div>
      {status?.phase==='failed' && !status.persisted_paused && <p>No se pudo guardar la pausa. La entrada está bloqueada en esta ejecución; actualizá y reintentá.</p>}
      <p className="mute">La pausa completada se conserva para el próximo inicio. La reactivación se hace desde este panel autenticado, sin hablarle al micrófono.</p>
    </section>
    <section className="card"><h2>Micrófonos</h2><p>El mute total todavía no está disponible en esta versión de desarrollo. La pausa ambiental permite seguir usando comandos de voz.</p></section>
  </div>;
}
window.PrivacyView = PrivacyView;
