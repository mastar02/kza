<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
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
      { key: 'status', label: 'Estado' }, { key: 'uptime', label: 'Tiempo activo' }, { key: 'mem', label: 'Memoria' },
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
  let selected = $state<string | null>(null)
  const visible = $derived(collections.filter(config => config.group === group))
  const active = $derived(visible.find(config => config.path === selected) ?? visible[0])
  const list = $derived(active?.resource)

  onMount(() => {
    void active?.resource.load()
    const poll = setInterval(() => {
      if (!write.busy) void active?.resource.load(true)
    }, 10000)
    return () => {
      clearInterval(poll)
      write.destroy()
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
        {#if item.cooldown && session.can('admin:manage')}
          <button disabled={write.busy || list?.busy} onclick={() => clearCooldown(item.id)}>Quitar cooldown</button>
        {/if}
      </article>
    {:else}<p class="muted">{list?.busy ? 'Consultando…' : 'Sin observaciones disponibles.'}</p>{/each}
  </div>
</section>

<style>
  nav { margin-block-end: 1rem; }
</style>
