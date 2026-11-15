// Capture controls use server state and the current authenticated session.
function lostRecordings(count) {
  return count===1 ? '1 grabación perdida' : `${count} grabaciones perdidas`;
}

function CaptureControlCard() {
  const [status,setStatus] = React.useState(null);
  const [message,setMessage] = React.useState('Consultando captura global…');
  const [notice,setNotice] = React.useState('');
  const [busy,setBusy] = React.useState(false);
  // R1-B2: the carried count the administrator is confirming, or null.
  const [forgetShown,setForgetShown] = React.useState(null);
  const [affirmed,setAffirmed] = React.useState(false);
  const action = React.useRef(null);
  React.useEffect(() => {
    let active = true, pending = false, denied = false, sequence = 0;
    const requests = new Set();
    const phases = ['initializing','running','muting','muted','resuming','failed','closed'];
    function valid(data) {
      if (!data || !Number.isSafeInteger(data.revision) || data.revision<0
          || !phases.includes(data.phase)
          || !['target_muted','persisted_muted','ingress_sealed','supported','total_mute_confirmed'].every(key => typeof data[key]==='boolean')
          || !['pending_operations','pending_browser_recorders','pending_native_clients','live_capture_owners','carried_capture_owners'].every(key => Number.isSafeInteger(data[key]) && data[key]>=0)
          || !(data.error_code===null || typeof data.error_code==='string' && /^[a-z][a-z0-9_]{0,79}$/.test(data.error_code))) return false;
      if (data.total_mute_confirmed && (data.phase!=='muted' || !data.supported || !data.target_muted
          || !data.persisted_muted || !data.ingress_sealed || data.pending_operations!==0 || data.pending_browser_recorders!==0 || data.pending_native_clients!==0
          || data.live_capture_owners!==0 || data.carried_capture_owners!==0)) return false;
      if (data.forgotten_capture_owners!==undefined && !(Number.isSafeInteger(data.forgotten_capture_owners) && data.forgotten_capture_owners>=0)) return false;
      if (data.phase==='running' && (data.target_muted || data.persisted_muted || data.ingress_sealed || data.total_mute_confirmed)) return false;
      return true;
    }
    // request: undefined reads; {muted,revision} changes the policy; {forget}
    // forgets the lost recordings whose count the administrator confirmed.
    async function readOrChange(request) {
      if (!active || pending || denied) return;
      const forget = request!==undefined && Number.isSafeInteger(request.forget) ? request.forget : null;
      const change = forget!==null || typeof request?.muted==='boolean';
      if (request!==undefined && !change) return;
      pending = true; setBusy(true);
      if (change) {
        setStatus(null); setNotice('');
        setMessage(forget===null ? 'Enviando solicitud de captura…' : 'Olvidando las grabaciones perdidas…');
      }
      const attempt = ++sequence, controller = new AbortController(); requests.add(controller);
      let refresh = false;
      try {
        const body = forget!==null ? {expected_carried:forget,no_capture_running:true}
          : {muted:request?.muted,expected_revision:request?.revision};
        const response = await fetch(forget!==null ? '/api/capture/control/forget-carried' : '/api/capture/control',
          {credentials:'same-origin',cache:'no-store',signal:controller.signal,
          ...(change ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
        if (!active || sequence!==attempt) return;
        if (response.status===401) {
          denied = true; setStatus(null); setMessage('La sesión venció. Volvé a iniciar sesión.');
          window.dispatchEvent(new CustomEvent('kza:auth-required'));
          return;
        }
        if (response.status===403) {
          denied = true; setStatus(null); setMessage('Esta sesión no tiene permiso para controlar la captura global.');
          return;
        }
        if (response.status===409) {
          refresh = true;
          setNotice(forget===null ? 'La revisión cambió. Revisá el estado actualizado y volvé a elegir el cambio.'
            : 'La cantidad de grabaciones perdidas cambió. No se olvidó ninguna; revisá el estado actualizado.');
          throw Error('Actualizando el estado de captura…');
        }
        if (!response.ok) {
          if (forget!==null) { refresh = true; setNotice('No se pudieron olvidar las grabaciones perdidas.'); }
          throw Error('No se pudo verificar la captura global.');
        }
        const data = await response.json();
        if (!valid(data)) throw Error('El estado de captura global no se pudo verificar.');
        if (active && sequence===attempt) {
          setStatus(data); setMessage('');
          if (forget!==null) setNotice(data.forgotten_capture_owners>0
            ? `${data.forgotten_capture_owners===1 ? 'Se olvidó' : 'Se olvidaron'} ${lostRecordings(data.forgotten_capture_owners)}. Quedó un registro sin contenido con la cantidad.`
            : 'No quedaban grabaciones perdidas para olvidar.');
        }
      } catch (error) {
        if (active && sequence===attempt) {
          setStatus(null);
          setMessage(error.name==='AbortError' ? 'Consulta cancelada.' : 'No se pudo verificar la captura global.');
        }
      } finally {
        requests.delete(controller); pending = false;
        if (active && sequence===attempt) setBusy(false);
        if (refresh && active && !denied) queueMicrotask(() => readOrChange());
      }
    }
    function retire() {
      active = false; sequence++; clearInterval(timer);
      for (const request of requests) request.abort();
      requests.clear(); action.current = null;
      setStatus(null); setMessage('La sesión ya no está disponible.'); setBusy(false);
    }
    action.current = readOrChange;
    const timer = setInterval(() => readOrChange(),1000);
    window.addEventListener('kza:auth-required',retire);
    window.addEventListener('pagehide',retire);
    readOrChange();
    return () => {
      active = false; sequence++; clearInterval(timer);
      window.removeEventListener('kza:auth-required',retire);
      window.removeEventListener('pagehide',retire);
      for (const request of requests) request.abort();
      requests.clear(); action.current = null;
    };
  },[]);
  const labels = {
    initializing:'Preparando el control de captura global',
    running:'Entrada de audio habilitada',
    muting:'Pausando: esperando el trabajo y las grabaciones iniciadas',
    muted:'Entrada sellada; esperando la confirmación completa de la pausa',
    resuming:'Reactivando: esperando la confirmación de los participantes',
    failed:'No se pudo confirmar el cambio de captura',
    closed:'Control de captura cerrado',
  };
  const confirmed = status?.total_mute_confirmed===true;
  const waiting = busy || !status || !status.supported || ['initializing','muting','resuming','closed'].includes(status.phase);
  const lost = status?.carried_capture_owners ?? 0;
  // A confirmation is only for the count it showed; a new count closes it.
  React.useEffect(() => { if (forgetShown!==null && lost!==forgetShown) { setForgetShown(null); setAffirmed(false); } },[lost,forgetShown]);
  const statusLine = message || (!status?.supported ? 'Captura global sin confirmar'
    : confirmed ? 'Captura global pausada y confirmada'
    : status?.phase==='muted' && lost>0 ? 'Entrada sellada; la pausa no se confirma mientras haya grabaciones perdidas'
    : labels[status?.phase]);
  return <section className="card" aria-label="Captura global">
    <h2>Captura global</h2>
    <p role="status">{statusLine}</p>
    {notice && <p role="alert">{notice}</p>}
    {status && !status.supported && <p>El control global no está disponible. No se confirmó la pausa de todos los participantes.</p>}
    {status?.pending_operations>0 && <p className="mute">Hay procesamiento iniciado que todavía debe terminar.</p>}
    {status?.pending_browser_recorders>0 && <p className="mute">Hay grabaciones del navegador pendientes de retiro.</p>}
    {status?.pending_native_clients>0 && <p className="mute">Hay capturas de herramientas pendientes de retiro.</p>}
    {status?.live_capture_owners>0 && <p className="mute">Grabaciones o capturas en curso que una parada del servicio va a esperar: {status.live_capture_owners}.</p>}
    {lost>0 && <div aria-label="Grabaciones perdidas">
      <p className="mute">Grabaciones perdidas: {lost}. Son grabaciones o capturas de un arranque anterior que nadie retiró. No frenan la parada, pero la pausa no se confirma mientras existan.</p>
      {forgetShown===null ? <button type="button" className="btn" disabled={busy || status?.phase==='closed'}
          onClick={() => { setForgetShown(lost); setAffirmed(false); }}>Olvidar grabaciones perdidas</button>
        : <div role="group" aria-label="Confirmar olvido de grabaciones perdidas">
          <p>Olvidarlas no detiene ninguna grabación: sólo deja de esperarlas para confirmar la pausa. Queda un registro sin contenido con la cantidad.</p>
          <label><input type="checkbox" checked={affirmed} onChange={event => setAffirmed(event.target.checked===true)} /> Confirmo que ninguna pestaña ni herramienta sigue grabando.</label>
          <div className="row">
            <button type="button" className="btn" disabled={busy || !affirmed}
              onClick={() => { const shown = forgetShown; setForgetShown(null); setAffirmed(false); action.current?.({forget:shown}); }}>Olvidar {lostRecordings(forgetShown)}</button>
            <button type="button" className="btn" onClick={() => { setForgetShown(null); setAffirmed(false); }}>Cancelar</button>
          </div>
        </div>}
    </div>}
    {status?.ingress_sealed && !confirmed && <p className="mute">La entrada está sellada; el retiro completo sigue sin confirmar.</p>}
    {status?.phase==='failed' && <p>Actualizá el estado antes de volver a solicitar un cambio. No se mostrará una pausa completa sin confirmación.</p>}
    <p>La pausa global incluye la captura de comandos y la captura ambiental. La reactivación se solicita desde este panel autenticado.</p>
    <p>Los datos anteriores conservan sus vencimientos. Pausar no equivale a borrarlos.</p>
    <div className="row">
      <button type="button" className="btn" disabled={waiting || confirmed} onClick={() => action.current?.({muted:true,revision:status.revision})}>Pausar captura global</button>
      <button type="button" className="btn" disabled={waiting || status?.phase==='running'} onClick={() => action.current?.({muted:false,revision:status.revision})}>Reactivar captura global</button>
    </div>
  </section>;
}

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
    <header className="page-head"><div><h1>Privacidad</h1><p className="mute">Control local de la entrada de audio y de la captura ambiental.</p></div></header>
    <CaptureControlCard />
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
  </div>;
}
window.PrivacyView = PrivacyView;
