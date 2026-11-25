<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { catalogFrom, commandFor, controlFor } from '../lib/home'
  import type { Operation } from '../lib/home'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session }: { session: Session } = $props()
  const ownerSession = untrack(() => session)
  const list = new Resource(ownerSession, signal => session.request('/api/routines', { permissions: ['home:read'], signal }))
  const catalog = new Resource(ownerSession, async signal =>
    catalogFrom(await session.request('/api/home/catalog', { permissions: ['home:read'], signal })))
  const change = new Mutation(ownerSession)
  const canManage = $derived(session.can('routines:manage'))
  const executionGrant = $derived(canManage ? 'routines:manage' : 'routines:execute')
  const canExecute = $derived(session.can('home:control', executionGrant))
  let selected = $state('')
  let operation = $state<Operation | ''>('')
  const device = $derived(catalog.data?.entities.find(item => item.id === selected))
  const field = $derived(device && operation && controlFor(device, operation))
  let draftError = $state('')
  const routineStatus: Record<string, string> = {
    running: 'En curso', succeeded: 'Completada', failed: 'Falló', uncertain: 'Sin confirmar',
    skipped: 'Omitida', interrupted: 'Interrumpida',
  }

  onMount(() => {
    void list.load()
    if (canManage) void catalog.load()
    const refresh = setInterval(() => { if (!change.busy) void list.load(true) }, 15000)
    const unregister = session.onPurge(() => { selected = ''; operation = ''; draftError = '' })
    return () => { clearInterval(refresh); unregister(); list.destroy(); catalog.destroy(); change.destroy() }
  })

  function modify(routine: Record<string, unknown>, action: 'enable' | 'disable' | 'execute' | 'delete') {
    if (typeof routine.routine_id !== 'string' || !Number.isSafeInteger(routine.revision)) return
    const endpoint = '/api/routines/' + encodeURIComponent(routine.routine_id)
    const revision = routine.revision
    if (action === 'execute') {
      if (!window.confirm('¿Ejecutar esta rutina ahora?')) return
      return change.run(endpoint + '/execute', { permissions: ['home:control', executionGrant], method: 'POST',
        body: { request_id: crypto.randomUUID() } }, () => list.load(true))
    }
    if (action === 'delete') {
      if (!window.confirm('¿Eliminar esta rutina?')) return
      return change.run(endpoint + '?expected_revision=' + String(revision),
        { permissions: ['routines:manage'], method: 'DELETE' }, () => list.load(true))
    }
    return change.run(endpoint + '/' + action, { permissions: ['routines:manage'], method: 'POST',
      body: { expected_revision: revision } }, () => list.load(true))
  }

  async function create(event: SubmitEvent & { currentTarget: HTMLFormElement }) {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    const name = String(data.get('name') ?? '').trim()
    const startTime = data.get('time')
    const value = data.get('value')
    const command = device && operation && device.capabilities.includes(operation)
      ? commandFor(device, operation, typeof value === 'string' ? value : undefined) : null
    if (!name || !command || !device) { draftError = 'Revisá el nombre y la acción de la rutina.'; return }
    draftError = ''
    const saved = await change.run('/api/routines', {
      permissions: ['routines:manage'], method: 'POST',
      body: { name, description: '', enabled: false, timezone: 'America/Montevideo',
        schedule: startTime ? { at: startTime, weekdays: [0, 1, 2, 3, 4, 5, 6] } : null,
        conditions: [], actions: [{ ...command, entity_id: device.id }] },
    }, () => list.load(true))
    if (saved) {
      form.reset()
      selected = ''
      operation = ''
    }
  }
</script>

<section>
  <div class="heading-row">
    <h1>Rutinas</h1>
    <button disabled={list.busy || change.busy} onclick={() => list.load(true)}>Actualizar</button>
  </div>
  {#if list.error}<p class="error" role="alert">{list.error}</p>{/if}
  {#if change.feedback}<p role="status">{change.feedback}</p>{/if}
  <div class="grid">
    {#each rows(list.data) as routine}
      {@const schedule = record(routine.schedule)}
      {@const last = record(routine.latest_run)}
      <article class="card">
        <div class="heading-row">
          <h3>{text(routine.name, 'Rutina')}</h3>
          <span class="pill">{routine.enabled === true ? 'Activa' : 'Desactivada'}</span>
        </div>
        <p>{text(routine.description, schedule ? 'Programada' : 'Ejecución manual')}{schedule ? ' · ' + text(schedule.at) : ''}</p>
        {#if last}<p>Última ejecución · {routineStatus[text(last.status)] ?? 'Sin confirmar'}</p>{/if}
        {#if canExecute && routine.enabled === true}
          <button disabled={change.busy || list.busy || last?.status === 'running'} onclick={() => modify(routine, 'execute')}>Ejecutar</button>
        {/if}
        {#if canManage && typeof routine.enabled === 'boolean'}
          <button disabled={change.busy || list.busy} onclick={() => modify(routine, routine.enabled ? 'disable' : 'enable')}>{routine.enabled ? 'Desactivar' : 'Activar'}</button>
          <button class="danger" disabled={change.busy || list.busy} onclick={() => modify(routine, 'delete')}>Eliminar</button>
        {/if}
      </article>
    {:else}
      <p class="muted">{list.busy ? 'Consultando rutinas…' : 'No hay rutinas para mostrar.'}</p>
    {/each}
  </div>
  {#if canManage && catalog.data}
    <section class="card">
      <h2>Nueva rutina</h2>
      <form class="inline-form" onsubmit={create}>
        <label>Nombre <input name="name" maxlength="80" required /></label>
        <label>Dispositivo
          <select bind:value={selected} onchange={() => operation = ''} required>
            <option value="">Elegí un dispositivo</option>
            {#each catalog.data.entities.filter(item => item.capabilities.length) as item (item.id)}
              <option value={item.id}>{item.name}</option>
            {/each}
          </select>
        </label>
        {#if device}
          <label>Acción
            <select bind:value={operation} required>
              <option value="">Elegí una acción</option>
              {#each device.capabilities as choice}
                {@const control = controlFor(device, choice)}
                {#if control}<option value={choice}>{control.label}</option>{/if}
              {/each}
            </select>
          </label>
        {/if}
        {#if field && field.kind === 'number'}
          <label>{field.label} <input name="value" type="number" min={field.min} max={field.max} step={field.step} required /></label>
        {:else if field && field.kind === 'select'}
          <label>{field.label} <select name="value" required>
            {#each field.options as option}<option value={option}>{option}</option>{/each}
          </select></label>
        {/if}
        <label>Hora diaria (opcional) <input type="time" name="time" /></label>
        <button class="primary" disabled={change.busy || list.busy || !field}>Crear desactivada</button>
      </form>
      {#if draftError}<p class="error" role="alert">{draftError}</p>{/if}
    </section>
  {:else if canManage && catalog.error}
    <p class="error" role="alert">{catalog.error}</p>
  {/if}
</section>
