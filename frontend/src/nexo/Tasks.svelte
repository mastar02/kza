<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import type { RequestOptions, Session } from '../lib/session'
  import { record } from '../lib/session'

  let { session }: { session: Session } = $props()
  const ownerSession = untrack(() => session)
  type Scope = 'shared' | 'personal'
  let scope = $state<Scope>('shared')
  let owner = $state<AbortController | null>(null)
  const personal = $derived(session.can('tasks:personal'))
  const canWrite = $derived(session.can('tasks:write'))
  const canReadNotices = $derived(session.can('home:read', 'tasks:read'))
  const write = new Mutation(ownerSession)
  const tasks = new Resource(ownerSession, async signal => {
    const current = scope
    const permissions = current === 'personal' ? ['tasks:read', 'tasks:personal'] : ['tasks:read']
    const [lists, reminders] = await Promise.all([
      session.request('/api/lists?scope=' + current, { permissions, signal, verifyIdentity: current === 'personal' }),
      session.request('/api/reminders?scope=' + current, { permissions, signal, verifyIdentity: current === 'personal' }),
    ])
    const filtered = rows(lists).filter(row => row.owner_type === (current === 'shared' ? 'shared' : 'user') &&
      typeof row.id === 'string')
    const full = await Promise.all(filtered.map(async row => {
      const id = String(row.id)
      const items = rows(await session.request('/api/lists/' + encodeURIComponent(id) + '/items', { permissions, signal, verifyIdentity: current === 'personal' }))
      return { id, name: text(row.name, 'Lista'), items }
    }))
    return { lists: full, reminders: rows(reminders).filter(row => row.scope === (current === 'shared' ? 'shared' : 'private')) }
  })
  const notices = new Resource(ownerSession, async signal => {
    const value = await session.request('/api/reminders/notices', { permissions: ['home:read', 'tasks:read'], signal })
    if (record(value)?.source !== 'reminder-store') throw new Error('Invalid notices')
    return rows(value, 'notices').filter(row => row.status !== 'confirmed')
  })

  onMount(() => {
    owner = new AbortController()
    const refreshNotices = () => { if (canReadNotices && !write.busy) void notices.load(true) }
    void tasks.load()
    refreshNotices()
    const timer = setInterval(refreshNotices, 5000)
    return () => {
      clearInterval(timer)
      owner?.abort()
      owner = null
      tasks.destroy()
      notices.destroy()
      write.destroy()
    }
  })

  function switchScope(next: Scope) {
    if (write.busy || scope === next || next === 'personal' && !personal) return
    scope = next
    tasks.clear()
    write.clear()
    void tasks.load()
  }

  function change(path: string, method: RequestOptions['method'], body?: unknown, notice = false) {
    const permissions = notice ? ['tasks:write', 'home:read', 'tasks:read']
      : scope === 'personal' ? ['tasks:write', 'tasks:personal'] : ['tasks:write']
    return write.run(path, { permissions, method, body }, () => notice ? notices.load() : tasks.load(), response => {
      const result = record(record(response)?.outcome)
      return result?.status === 'completed' && result.persistence === 'committed'
        ? 'Cambio guardado.' : 'El servidor no confirmó el cambio.'
    })
  }

  function submitText(event: SubmitEvent & { currentTarget: HTMLFormElement }, submit: (value: string, data: FormData) => Promise<boolean>) {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    const value = String(data.get('text') ?? '').trim()
    if (value) void submit(value, data).then(saved => { if (saved && owner && !owner.signal.aborted) form.reset() })
  }

  function addReminder(value: string, data: FormData) {
    const when = new Date(String(data.get('at'))).getTime() / 1000
    if (!Number.isFinite(when) || when * 1000 <= Date.now()) {
      write.feedback = 'Elegí una fecha y hora futura.'
      return Promise.resolve(false)
    }
    return change('/api/reminders', 'POST', { text: value, trigger_at: when, scope: scope === 'shared' ? 'shared' : 'private' })
  }

  function dateLabel(value: unknown) {
    return typeof value === 'number' && Number.isFinite(value) ? new Date(value * 1000).toLocaleString('es-UY') : 'Sin horario'
  }
</script>

<section>
  <div class="heading-row">
    <h1>Tareas</h1>
    <div class="button-row">
      {#if personal}
        <button aria-pressed={scope === 'shared'} disabled={write.busy} onclick={() => switchScope('shared')}>Compartidas</button>
        <button aria-pressed={scope === 'personal'} disabled={write.busy} onclick={() => switchScope('personal')}>Personales</button>
      {/if}
      <button disabled={tasks.busy || write.busy} onclick={() => { void tasks.load(); if (canReadNotices) void notices.load() }}>Actualizar</button>
    </div>
  </div>
  {#if tasks.error}<p class="error" role="alert">{tasks.error}</p>{/if}
  {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
  <div class="grid">
    <article class="card">
      <h2>Listas</h2>
      {#each tasks.data?.lists ?? [] as list (list.id)}
        <h3>{list.name}</h3>
        <ul>
          {#each list.items as item}
            <li>
              {text(item.text)}
              {#if canWrite && typeof item.id === 'string'}
                <button disabled={write.busy} onclick={() => change('/api/lists/' + encodeURIComponent(list.id) + '/items/' + encodeURIComponent(String(item.id)), 'DELETE')}>Quitar</button>
              {/if}
            </li>
          {/each}
        </ul>
        {#if canWrite}
          <form class="inline-form" onsubmit={event => submitText(event, value => change('/api/lists/' + encodeURIComponent(list.id) + '/items', 'POST', { text: value }))}>
            <label>Elemento de {list.name} <input name="text" required maxlength="300" /></label>
            <button disabled={write.busy || tasks.busy}>Agregar</button>
          </form>
        {/if}
      {:else}<p class="muted">{tasks.busy ? 'Consultando listas…' : 'Sin listas.'}</p>{/each}
      {#if canWrite && tasks.data}
        <form class="inline-form" onsubmit={event => submitText(event, name => change('/api/lists', 'POST', { name, shared: scope === 'shared' }))}>
          <label>Nombre de la lista <input name="text" required maxlength="100" /></label>
          <button disabled={write.busy || tasks.busy}>Crear lista</button>
        </form>
      {/if}
    </article>

    <article class="card">
      <h2>Recordatorios</h2>
      <ul>
        {#each tasks.data?.reminders ?? [] as reminder}
          <li>
            {text(reminder.text)} · {dateLabel(reminder.trigger_at)}
            {#if canWrite && typeof reminder.id === 'string'}
              <button disabled={write.busy} onclick={() => {
                if (window.confirm('¿Eliminar este recordatorio?')) void change('/api/reminders/' + encodeURIComponent(String(reminder.id)), 'DELETE')
              }}>Eliminar</button>
            {/if}
          </li>
        {:else}<li class="muted">{tasks.busy ? 'Consultando…' : 'Sin recordatorios.'}</li>{/each}
      </ul>
      {#if canWrite && tasks.data}
        <form class="inline-form" onsubmit={event => submitText(event, addReminder)}>
          <label>Recordatorio <input name="text" required maxlength="500" /></label>
          <label>Fecha y hora <input name="at" type="datetime-local" required /></label>
          <button disabled={write.busy || tasks.busy}>Crear recordatorio</button>
        </form>
      {/if}
    </article>

    {#if canReadNotices && scope === 'shared'}
      <article class="card">
        <h2>Avisos del hogar</h2>
        {#if notices.error}<p class="error" role="alert">{notices.error}</p>{/if}
        <ul>
          {#each notices.data ?? [] as notice}
            <li>
              {text(notice.text, 'Aviso')} · {text(notice.label)} · {dateLabel(notice.trigger_at)}
              {#if canWrite && (notice.status === 'pending' || notice.status === 'uncertain') &&
                typeof notice.reminder_id === 'string' && typeof notice.trigger_at === 'number'}
                <button disabled={write.busy || notices.busy} onclick={() => change('/api/reminders/notices/dismiss', 'POST',
                  { reminder_id: notice.reminder_id, trigger_at: notice.trigger_at }, true)}>Descartar</button>
              {/if}
            </li>
          {:else}<li class="muted">{notices.busy ? 'Consultando…' : 'Sin avisos pendientes.'}</li>{/each}
        </ul>
      </article>
    {/if}
  </div>
</section>
