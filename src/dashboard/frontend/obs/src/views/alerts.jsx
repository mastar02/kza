// =============================================================
// Alerts view
// =============================================================

function AlertCard({ a, canManage }) {
  const kindMap = { info: '', warn: 'warn', critical: 'err' };
  const k = kindMap[a.priority];
  const accentColor = a.priority === 'critical' ? 'var(--err)' : a.priority === 'warn' ? 'var(--warn)' : 'var(--info)';
  return (
    <div className="card" style={{ borderLeft: `2px solid ${accentColor}`, opacity: a.acked ? 0.55 : 1 }}>
      <div className="card-b">
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
              <Pill kind={k}>{a.priority}</Pill>
              <Pill>{a.type}</Pill>
              <span className="mono mute" style={{ fontSize: 10 }}>{a.zone}</span>
              <EventTime at={a.at} ts={a.ts} />
              {a.acked && <span className="mono mute" style={{ fontSize: 10 }}>· atendida</span>}
            </div>
            <div style={{ fontWeight: 500, marginBottom: 4 }}>{a.title}</div>
            <div className="mute" style={{ fontSize: 12 }}>{a.body}</div>
          </div>
          {canManage && !a.acked && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <OperationButton kind="ackAlert" target={a.id} label="Marcar atendida" />
              <OperationButton kind="dismissAlert" target={a.id} label="Descartar" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const ALERT_PRODUCERS = {
  door_open_unexpected: 'Puertas y ventanas',
  battery_low: 'Baterías',
  offline: 'Dispositivos no disponibles',
};

function AlertProducers({ producers, now }) {
  const { alertProducerLabel, formatInstant } = window.KzaObsData;
  return (
    <Card>
      <h3>Fuentes de alertas</h3>
      {producers.length === 0 && <p className="mute">Sin fuentes de alertas disponibles.</p>}
      {producers.map(producer => (
        <div key={producer.id}>
          <b>{ALERT_PRODUCERS[producer.id] || producer.id}</b>
          {' · '}{alertProducerLabel(producer, now)}
          <div className="mono mute">
            Último chequeo {formatInstant(producer.last_check_at)}
            {' · '}Observada {formatInstant(producer.observed_at)}
            {' · '}Vence {formatInstant(producer.expires_at)}
          </div>
          {(producer.uncertain || producer.overflow > 0) && (
            <div className="mono mute">
              {producer.unknown} sin confirmar · {producer.overflow} desbordamientos
            </div>
          )}
        </div>
      ))}
    </Card>
  );
}

const ALERT_MUTES = {
  voice_notifications_disabled: 'Avisos hablados desactivados.',
  alert_global_mute: 'Silencio global activo.',
  alert_mute_unavailable: 'Avisos hablados en pausa. Estado del silencio global no disponible.',
  alert_quiet_hours: 'Horario silencioso activo.',
};

function AlertZoneSelector({ output, obs }) {
  const [zones, setZones] = React.useState(output.zones);
  const held = obs.operationOutcome('setAlertZones', 'zones');
  const unchanged = zones.length === output.zones.length && zones.every(zone => output.zones.includes(zone));

  function save(zones) {
    window.KzaObsData.runOperation({
      fetch: window.fetch.bind(window), store: obs, kind: 'setAlertZones', target: 'zones',
      payload: { zones, expected_revision: output.revision },
    }).then(outcome => {
      if (outcome.state === 'auth' || outcome.sessionExpired) {
        window.dispatchEvent(new CustomEvent('kza:auth-required'));
      }
    });
  }

  return (
    <form onSubmit={event => { event.preventDefault(); save(zones); }}>
      <fieldset disabled={Boolean(held)}>
        <legend>Elegir habitaciones</legend>
        {output.configured_zones.map(zone => (
          <label key={zone} style={{ display: 'block' }}>
            <input type="checkbox" checked={zones.includes(zone)} onChange={event => {
              const { checked } = event.target;
              setZones(previous => checked ? [...previous, zone] : previous.filter(value => value !== zone));
            }} /> {alertZoneName(zone)}
          </label>
        ))}
      </fieldset>
      <button type="submit" className="btn primary" disabled={Boolean(held) || unchanged}>Guardar zonas</button>
      {!output.using_default && (
        <button type="button" className="btn" disabled={Boolean(held)} onClick={() => save(null)}>Usar zona configurada</button>
      )}
    </form>
  );
}

function alertZoneName(zone) {
  return zone.replace(/^zone_/, '').replaceAll('_', ' ');
}

function AlertZones({ output, obs }) {
  return (
    <Card>
      <h3>Avisos hablados</h3>
      {output?.available ? (
        <>
          <p>{output.zones.length ? output.zones.map(alertZoneName).join(', ') : 'Solo en la web'}{output.using_default && ' · según configuración'}</p>
          <p className="mute">Horario silencioso de 23:00 a 07:00, hora de la casa. {ALERT_MUTES[output.mute_reason]}</p>
          {obs.can('admin:manage') && output.writable && (
            <AlertZoneSelector key={output.revision} output={output} obs={obs} />
          )}
        </>
      ) : <p className="mute">Estado de las zonas no disponible.</p>}
    </Card>
  );
}

function AlertsView() {
  const obs = useObs();
  const [tab, setTab] = React.useState('active');
  const alerts = obs.get('ALERTS');
  const available = obs.available('ALERTS');
  const status = obs.entry('ALERTS').alertStatus;
  const canManage = obs.can('admin:manage');
  const active = alerts.filter(a => !a.acked);
  const acked = alerts.filter(a => a.acked);
  const list = tab === 'active' ? active : acked;
  return (
    <div className="view" id="view-alerts">
      <PageHead
        title="Alertas"
        subtitle="Alertas proactivas · seguridad / patrones / dispositivos"
        meta={available ? <><span>{active.length} activas</span> · <span>{acked.length} atendidas</span> · <SourceNote k="ALERTS" /></> : <SourceNote k="ALERTS" />}
      />

      {status && (
        <div className="col" style={{ marginBottom: 16 }}>
          <AlertProducers producers={status.producers} now={obs.now()} />
          <AlertZones output={status.output} obs={obs} />
        </div>
      )}

      <Card>
        <div style={{ display: 'flex', gap: 4 }}>
          <button type="button" className={`btn ${tab === 'active' ? 'primary' : ''}`} aria-pressed={tab === 'active'} onClick={() => setTab('active')}>activas · {active.length}</button>
          <button type="button" className={`btn ${tab === 'acked' ? 'primary' : ''}`} aria-pressed={tab === 'acked'} onClick={() => setTab('acked')}>atendidas · {acked.length}</button>
        </div>
      </Card>

      <div className="col" style={{ marginTop: 16 }}>
        {list.map(a => <AlertCard key={a.id} a={a} canManage={canManage} />)}
        {list.length === 0 && (
          <Card>
            <div className="mute" style={{ textAlign: 'center', padding: 30 }}>
              {available ? 'sin alertas' : 'Alertas no disponibles: no se pudo leer la fuente.'}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

window.AlertsView = AlertsView;
