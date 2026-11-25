<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Role, Session } from '../lib/session'
  import CaptureControls from '../shared/CaptureControls.svelte'

  let { session, role }: { session: Session; role: Role } = $props()
  const ownerSession = untrack(() => session)
  const admin = $derived(session.can('admin:manage'))
  const auth = $derived(role === 'operator' && session.can('auth:manage'))
  const write = new Mutation(ownerSession)
  const consentPath = '/api/admin/cloud-consent'
  const cloud = new Resource(ownerSession, async signal =>
    record(await session.request(consentPath, { permissions: ['admin:manage'], signal })) ?? null)
  const people = new Resource(ownerSession, signal =>
    session.request('/api/admin/users', { permissions: ['admin:manage'], signal }))
  let cursor = ''
  const credentials = new Resource(ownerSession, async signal =>
    record(await session.request('/api/admin/auth/credentials' + cursor, { permissions: ['auth:manage'], signal })) ?? null)
  let cloudClosed = $state(false)

  onMount(() => {
    if (admin) { void cloud.load(); void people.load() }
    if (auth) void credentials.load()
    const unregister = session.onPurge(() => { cursor = ''; cloudClosed = false })
    return () => { unregister(); cloud.destroy(); people.destroy(); credentials.destroy(); write.destroy() }
  })

  function withdraw() {
    if (cloudClosed || cloud.data?.gate_withdrawn !== false ||
        !window.confirm('¿Retirar el consentimiento cloud? Este cambio es unidireccional.')) return
    return write.run(consentPath + '/withdraw', { permissions: ['admin:manage'], method: 'POST' },
      () => cloud.load(), result => {
        if (record(result)?.gate_closed === true) {
          cloudClosed = true
          return 'Compuerta cloud cerrada. Consultá el estado del registro.'
        }
        return 'El servidor no confirmó el retiro.'
      })
  }

  function revoke(entry: Record<string, unknown>) {
    if (typeof entry.id !== 'string' || !Number.isSafeInteger(entry.revision) ||
        !window.confirm('¿Revocar esta credencial y cerrar las sesiones asociadas?')) return
    return write.run('/api/admin/auth/credentials/' + encodeURIComponent(entry.id), {
      permissions: ['auth:manage'], method: 'DELETE',
      body: { request_id: crypto.randomUUID(), expected_revision: entry.revision },
    }, () => credentials.load(), result => record(result)?.applied === true ? 'Credencial revocada.' : 'Revocación sin confirmar.')
  }
</script>

<section>
  <h1>Acceso y sistema</h1>
  {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
  <div class="grid">
    {#if admin}
      <article class="card">
        <h2>Consentimiento cloud</h2>
        <p>{cloudClosed || cloud.data?.gate_withdrawn === true ? 'Compuerta cloud retirada' :
          cloud.data?.gate_withdrawn === false ? 'Sin retiro registrado en esta ejecución' : 'Consultando estado'}</p>
        {#if cloud.data?.pending_write === true}<p>Escritura de configuración pendiente.</p>{/if}
        {#if cloud.data?.registration_failure}<p class="error">El servidor informó un fallo al registrar el retiro.</p>{/if}
        {#if cloud.error}<p role="alert" class="error">{cloud.error}</p>{/if}
        <div class="button-row">
          <button disabled={cloud.busy || write.busy} onclick={() => cloud.load()}>Consultar estado cloud</button>
          {#if cloud.data?.gate_withdrawn === false && !cloudClosed}
            <button class="danger" disabled={cloud.busy || write.busy} onclick={withdraw}>Retirar consentimiento cloud</button>
          {/if}
        </div>
      </article>
      <article class="card">
        <h2>Personas</h2>
        {#if people.error}<p class="error" role="alert">{people.error}</p>{/if}
        <ul>
          {#each rows(people.data) as person}<li>{text(person.name, 'Persona')}</li>
          {:else}<li class="muted">{people.busy ? 'Consultando…' : 'Sin personas registradas.'}</li>{/each}
        </ul>
      </article>
    {/if}
    {#if auth}
      <article class="card">
        <h2>Credenciales de acceso</h2>
        {#if credentials.error}<p class="error" role="alert">{credentials.error}</p>{/if}
        <ul>
          {#each rows(credentials.data?.rows) as entry}
            <li>{text(entry.subject)} · {text(entry.role)}
              {#if entry.active === true && entry.pending === false}
                <button class="danger" disabled={write.busy || credentials.busy} onclick={() => revoke(entry)}>Revocar</button>
              {/if}
            </li>
          {:else}<li class="muted">{credentials.busy ? 'Consultando…' : 'Sin credenciales en esta página.'}</li>{/each}
        </ul>
        {#if typeof credentials.data?.next_cursor === 'string'}
          <button disabled={write.busy || credentials.busy} onclick={() => {
            cursor = '?after=' + encodeURIComponent(String(credentials.data?.next_cursor))
            void credentials.load()
          }}>Más credenciales</button>
        {/if}
        <button disabled={write.busy || credentials.busy} onclick={() => { cursor = ''; void credentials.load() }}>Actualizar</button>
      </article>
    {/if}
  </div>
</section>
<CaptureControls {session} />
