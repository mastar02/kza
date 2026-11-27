<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session, system = false }: { session: Session; system?: boolean } = $props()
  const ownerSession = untrack(() => session)
  type Field = { key: string; label: string }
  type Collection = { path: string; title: string; fields: Field[]; group: 'system' | 'diagnostics'; isDaemons?: boolean }
  const configurations: Collection[] = [
    { path: '/api/zones', title: 'Zonas', group: 'diagnostics', fields: [
      { key: 'audioState', label: 'Audio' }, { key: 'volume', label: 'Volumen (%)' }, { key: 'ma1260_zone', label: 'Canal MA1260' },
    ] },
    { path: '/api/ha/entities', title: 'Home Assistant', group: 'diagnostics', fields: [
      { key: 'domain', label: 'Dominio' }, { key: 'state', label: 'Estado' }, { key: 'zone', label: 'Ambiente' },
    ] },
    { path: '/api/llm/endpoints', title: 'Endpoints LLM', group: 'diagnostics', fields: [
      { key: 'model', label: 'Endpoint' }, { key: 'kind', label: 'Tipo' }, { key: 'state', label: 'Estado' },
      { key: 'consecutive_errors', label: 'Errores' }, { key: 'cooldown_until', label: 'Cooldown hasta' },
    ] },
    { path: '/api/system/gpus', title: 'GPUs', group: 'system', fields: [
      { key: 'id', label: 'Dispositivo' },
      { key: 'vramUsed', label: 'VRAM usada (GB)' }, { key: 'vramTotal', label: 'VRAM total (GB)' },
      { key: 'temp', label: 'Temperatura (°C)' }, { key: 'util', label: 'Utilización (%)' },
      { key: 'power_w', label: 'Consumo (W)' },
    ] },
    { path: '/api/system/services', title: 'Servicios', group: 'system', fields: [
      { key: 'status', label: 'Estado' }, { key: 'substate', label: 'Subestado' }, { key: 'uptime', label: 'Tiempo activo' }, { key: 'mem', label: 'Memoria' },
    ] },
    { path: '/api/system/daemons', title: 'Daemons y ML', group: 'system', isDaemons: true, fields: [
      { key: 'state', label: 'Estado' }, { key: 'version', label: 'Versión' }, { key: 'uptime', label: 'Tiempo activo' }, { key: 'detail', label: 'Detalle' },
    ] },
  ]
  const group = $derived(system ? 'system' : 'diagnostics')

  function mbToGb(value: unknown): string | number | null {
    return typeof value === 'number' && Number.isFinite(value) ? (value / 1024).toFixed(1) : null
  }

  function formatUptime(ms: unknown): string | null {
    if (typeof ms !== 'number' || !Number.isFinite(ms) || ms <= 0) return null
    const secs = Math.floor(ms / 1000)
    const mins = Math.floor(secs / 60)
    const hours = Math.floor(mins / 60)
    return hours > 0 ? `${hours}h ${mins % 60}m` : mins > 0 ? `${mins}m ${secs % 60}s` : `${secs}s`
  }

  function mapInventoryItem(config: Collection, item: Record<string, unknown>) {
    const normalized: Record<string, unknown> = {
      ...item,
      zone: item.zone ?? item.room_id,
      model: item.model ?? item.name ?? item.id,
      state: item.state ?? (item.in_cooldown === true ? 'cooldown' : item.in_cooldown === false ? 'ready' : undefined),
      cooldown_until: item.cooldown_until ?? item.cooldown_ends,
      vramUsed: item.vramUsed ?? mbToGb(item.memory_used_mb),
      vramTotal: item.vramTotal ?? mbToGb(item.memory_total_mb),
      temp: item.temp ?? item.temperature,
      util: item.util ?? item.utilization,
      power_w: typeof item.power_w === 'number' && Number.isFinite(item.power_w) ? Math.round(item.power_w) : item.power_w,
    }
    const cooldownVal = config.path === '/api/llm/endpoints' ? (normalized.cooldown_until ?? null) : null
    return {
      id: item.id ?? item.entity_id ?? item.name,
      name: item.name ?? item.label ?? item.id ?? item.entity_id,
      cooldown: cooldownVal,
      slots: [] as Record<string, unknown>[],
      fields: config.fields
        .map(({ key, label }) => ({ label, value: normalized[key] }))
        .filter(field => typeof field.value === 'string' || typeof field.value === 'number'),
    }
  }

  const collections = configurations.map(config => ({
    ...config,
    resource: new Resource(ownerSession, async signal => {
      const value = await session.request(config.path, { permissions: ['diagnostics:read'], signal })
      if (config.isDaemons) {
        const daemonsObj = record(record(value)?.daemons)
        if (!daemonsObj) throw new Error('Invalid daemons')
        return Object.entries(daemonsObj).map(([key, raw]) => {
          const item = record(raw) ?? {}
          const slots = Array.isArray(item.slots)
            ? item.slots.map(s => record(s)).filter((s): s is Record<string, unknown> => Boolean(s))
            : []
          const uptime = formatUptime(item.uptime_ms)
          const fields = [
            { label: 'Estado', value: text(item.state, item.available === true ? 'ok' : 'unavailable') },
            { label: 'Versión', value: item.version },
            { label: 'Tiempo activo', value: uptime },
            { label: 'Detalle', value: item.detail },
          ].filter(field => typeof field.value === 'string' || typeof field.value === 'number')
          return {
            id: key,
            name: text(item.service, key),
            cooldown: null as unknown,
            slots,
            fields,
          }
        })
      }
      if (!Array.isArray(value)) throw new Error('Invalid inventory')
      return rows(value).map(item => mapInventoryItem(config, item))
    }),
  }))

  const write = new Mutation(ownerSession)
  const alerts = new Resource(ownerSession, async signal => {
    const value = await session.request('/api/alerts', { permissions: ['home:read'], signal })
    const source = Array.isArray(value) ? value : record(value)?.alerts
    return rows(source).map(item => {
      const state = text(item.state ?? item.status, '')
      const acknowledged = item.acked === true || item.acknowledged === true ||
        ['acknowledged', 'acked', 'resolved'].includes(state)
      return {
        id: item.id ?? item.alert_id,
        title: text(item.title ?? item.name ?? item.message, 'Alerta'),
        detail: text(item.body ?? item.message, ''),
        severity: severityLabel[text(item.priority ?? item.severity, '')] ?? '',
        acknowledged,
      }
    })
  })
  const severityLabel: Record<string, string> = {
    critical: 'Crítica', high: 'Alta', warning: 'Advertencia', medium: 'Media', low: 'Baja', info: 'Informativa',
  }
  const nonRestartableServices = new Set(['nexa-web', 'nexa-dashboard-proxy'])
  let selected = $state<string | null>(null)
  const visible = $derived(collections.filter(config => config.group === group))
  const active = $derived(visible.find(config => config.path === selected) ?? visible[0])
  const list = $derived(active?.resource)

  onMount(() => {
    void active?.resource.load()
    if (!system && session.can('home:read')) void alerts.load()
    const poll = setInterval(() => {
      if (write.busy) return
      void active?.resource.load(true)
      if (!system && session.can('home:read')) void alerts.load(true)
    }, 10000)
    return () => {
      clearInterval(poll)
      write.destroy()
      alerts.destroy()
      for (const item of collections) item.resource.destroy()
    }
  })

  function choose(path: string) {
    if (write.busy || active?.path === path) return
    active?.resource.clear()
    selected = path
    write.clear()
    void active?.resource.load()
  }

  function clearCooldown(id: unknown) {
    if (typeof id !== 'string' || !window.confirm('¿Quitar el cooldown de este endpoint?')) return
    const currentList = active?.resource
    return write.run('/api/llm/endpoints/' + encodeURIComponent(id) + '/clear-cooldown',
      { permissions: ['admin:manage'], method: 'POST' }, async () => { await currentList?.load() })
  }

  function restart(id: unknown, name: unknown) {
    if (typeof id !== 'string' || !window.confirm('¿Reiniciar el servicio ' + text(name, 'seleccionado') + '?')) return
    const currentList = active?.resource
    return write.run('/api/admin/services/' + encodeURIComponent(id) + '/restart',
      { permissions: ['admin:manage'], method: 'POST' }, async () => { await currentList?.load(true) })
  }

  function reloadSlot(slotId: unknown) {
    if (typeof slotId !== 'string' || !window.confirm('¿Recargar el modelo ' + slotId + '?')) return
    const currentList = active?.resource
    return write.run('/api/admin/ml/slots/' + encodeURIComponent(slotId) + '/reload',
      { permissions: ['admin:manage'], method: 'POST', body: {} }, async () => { await currentList?.load(true) })
  }

  function acknowledge(id: unknown) {
    if (typeof id !== 'string') return
    return write.run('/api/admin/alerts/' + encodeURIComponent(id) + '/ack',
      { permissions: ['admin:manage'], method: 'POST' }, () => alerts.load(true))
  }

  function label(value: unknown) {
    return typeof value === 'number' ? value.toLocaleString('es-UY') : text(value)
  }
</script>

<section>
  <div class="heading-row">
    <h2>{system ? 'Sistema' : 'Diagnóstico'}</h2>
    <button disabled={list?.busy || write.busy} onclick={() => list?.load()}>Actualizar</button>
  </div>
  <nav class="button-row" aria-label={system ? 'Inventario del sistema' : 'Inventario de diagnóstico'}>
    {#each visible as config}
      <button aria-pressed={active?.path === config.path} disabled={write.busy} onclick={() => choose(config.path)}>{config.title}</button>
    {/each}
  </nav>
  {#if list?.error}<p class="error" role="alert">{list.error}</p>{/if}
  {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
  <div class="grid">
    {#each list?.data ?? [] as item}
      <article class="card">
        <h3>{label(item.name ?? item.id)}</h3>
        <dl>
          {#each item.fields as field}
            <div><dt>{field.label}</dt><dd>{label(field.value)}</dd></div>
          {/each}
        </dl>
        {#if item.slots.length}
          <div class="ml-slots">
            {#each item.slots as slot}
              <div class="slot-row">
                <span><strong>{text(slot.slot_id)}</strong> · {text(slot.model_name)} ({text(slot.state)})</span>
                {#if session.can('admin:manage') && typeof slot.slot_id === 'string'}
                  <button disabled={write.busy || list?.busy} onclick={() => reloadSlot(slot.slot_id)}>Recargar</button>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
        {#if active?.path === '/api/system/services' && session.can('admin:manage') && typeof item.name === 'string' && !nonRestartableServices.has(item.name)}
          <button disabled={write.busy || list?.busy} onclick={() => restart(item.name, item.name)}>Reiniciar</button>
        {/if}
        {#if item.cooldown && session.can('admin:manage')}
          <button disabled={write.busy || list?.busy} onclick={() => clearCooldown(item.id)}>Quitar cooldown</button>
        {/if}
      </article>
    {:else}<p class="muted">{list?.busy ? 'Consultando…' : 'Sin observaciones disponibles.'}</p>{/each}
  </div>
  {#if !system && session.can('home:read')}
    <section aria-labelledby="alerts-title">
      <h3 id="alerts-title">Alertas</h3>
      {#if alerts.error}<p class="error" role="alert">{alerts.error}</p>{/if}
      <div class="grid">
        {#each alerts.data ?? [] as alert}
          <article class="card">
            <h4>{alert.title}</h4>
            {#if alert.detail}<p>{alert.detail}</p>{/if}
            <p>
              {#if alert.severity}<span class="pill">{alert.severity}</span>{/if}
              <span class="pill">{alert.acknowledged ? 'Reconocida' : 'Pendiente'}</span>
            </p>
            {#if !alert.acknowledged && session.can('admin:manage')}
              <button disabled={write.busy || alerts.busy} onclick={() => acknowledge(alert.id)}>Reconocer</button>
            {/if}
          </article>
        {:else}<p class="muted">{alerts.busy ? 'Consultando alertas…' : 'No hay alertas activas.'}</p>{/each}
      </div>
    </section>
  {/if}
</section>

<style>
  nav { margin-block-end: 1rem; }
  .ml-slots { margin-block-start: .65rem; border-top: 1px solid var(--line); padding-block-start: .5rem; display: grid; gap: .4rem; }
  .slot-row { display: flex; justify-content: space-between; align-items: center; gap: .5rem; flex-wrap: wrap; }
</style>
