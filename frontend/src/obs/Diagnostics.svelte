<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session, system = false }: { session: Session; system?: boolean } = $props()
  const ownerSession = untrack(() => session)
  type Field = { key: string; label: string }
  type Collection = { path: string; title: string; fields: Field[]; group: 'system' | 'diagnostics' }
  const configurations: Collection[] = [
    { path: '/api/zones', title: 'Zonas', group: 'diagnostics', fields: [
      { key: 'audioState', label: 'Audio' }, { key: 'status', label: 'Estado' },
    ] },
    { path: '/api/ha/entities', title: 'Home Assistant', group: 'diagnostics', fields: [
      { key: 'domain', label: 'Dominio' }, { key: 'state', label: 'Estado' }, { key: 'zone', label: 'Zona' },
    ] },
    { path: '/api/llm/endpoints', title: 'Endpoints LLM', group: 'diagnostics', fields: [
      { key: 'model', label: 'Modelo' }, { key: 'state', label: 'Estado' }, { key: 'cooldown_until', label: 'Cooldown hasta' },
    ] },
    { path: '/api/system/gpus', title: 'GPUs', group: 'system', fields: [
      { key: 'id', label: 'Dispositivo' }, { key: 'role', label: 'Asignación declarada' },
      { key: 'vramUsed', label: 'VRAM usada (GB)' }, { key: 'vramTotal', label: 'VRAM total (GB)' },
      { key: 'temp', label: 'Temperatura (°C)' }, { key: 'util', label: 'Utilización (%)' },
    ] },
    { path: '/api/system/services', title: 'Servicios', group: 'system', fields: [
      { key: 'status', label: 'Estado' }, { key: 'substate', label: 'Subestado' }, { key: 'uptime', label: 'Tiempo activo' }, { key: 'mem', label: 'Memoria' },
    ] },
  ]
  const group = $derived(system ? 'system' : 'diagnostics')
  const collections = configurations.map(config => ({
    ...config,
    resource: new Resource(ownerSession, async signal => {
      const value = await session.request(config.path, { permissions: ['diagnostics:read'], signal })
      if (!Array.isArray(value)) throw new Error('Invalid inventory')
      return rows(value).map(item => ({
        id: item.id ?? item.entity_id,
        name: item.name ?? item.label ?? item.id,
        cooldown: config.path === '/api/llm/endpoints' ? item.cooldown_until ?? item.cooldown_ends : null,
        fields: config.fields.map(({ key, label }) => {
          const value = key === 'cooldown_until' ? item.cooldown_until ?? item.cooldown_ends : item[key]
          return { label, value }
        }).filter(field => typeof field.value === 'string' || typeof field.value === 'number'),
      }))
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
    const list = active?.resource
    return write.run('/api/llm/endpoints/' + encodeURIComponent(id) + '/clear-cooldown',
      { permissions: ['admin:manage'], method: 'POST' }, async () => { await list?.load() })
  }

  function restart(id: unknown, name: unknown) {
    if (typeof id !== 'string' || !window.confirm('¿Reiniciar el servicio ' + text(name, 'seleccionado') + '?')) return
    const list = active?.resource
    return write.run('/api/admin/services/' + encodeURIComponent(id) + '/restart',
      { permissions: ['admin:manage'], method: 'POST' }, async () => { await list?.load(true) })
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
        {#if active?.path === '/api/system/services' && session.can('admin:manage')}
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
</style>
