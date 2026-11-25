<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Role, Session } from '../lib/session'
  import VoiceReview from '../shared/VoiceReview.svelte'

  let { session, role }: { session: Session; role: Role } = $props()
  const ownerSession = untrack(() => session)
  const own = $derived(role === 'home' && session.can('memory:own:voice'))
  const administration = $derived(session.can('memory:assign', 'memory:read'))
  const review = $derived(session.can('voices:read', 'voices:review'))
  const ownPath = '/api/memory/mine/voice'
  const switchPath = '/api/memory/voice/switch'
  const write = new Mutation(ownerSession)
  const consent = new Resource(ownerSession, async signal =>
    record(await session.request(ownPath, { permissions: ['memory:own:voice'], signal, verifyIdentity: true })) ?? null)
  const roomList = new Resource(ownerSession, async signal =>
    rows(await session.request('/api/home/catalog', { permissions: ['home:read'], signal }), 'rooms'))
  const global = new Resource(ownerSession, async signal =>
    record(await session.request(switchPath, { permissions: ['memory:read', 'memory:assign'], signal })) ?? null)
  const details = new Resource(ownerSession, async signal =>
    record(await session.request('/api/memory/voice/consents', { permissions: ['memory:read', 'memory:assign'], signal })) ?? null)
  let now = $state(Date.now() / 1000)
  let room = $state('')
  const verification = $derived(record(consent.data?.window))
  const windowCurrent = $derived(typeof verification?.expires_at === 'number' && verification.expires_at > now)
  const statuses: Record<string, string> = {
    pending_verification: 'Pendiente de verificación', active: 'Activo', verification_locked: 'Verificación bloqueada',
    revoked: 'Revocado', invalidated: 'Sin efecto tras un cambio de voz',
  }

  onMount(() => {
    if (own) {
      void consent.load()
      if (session.can('home:read')) void roomList.load()
    }
    if (administration) { void global.load(); void details.load() }
    const clock = setInterval(() => { now = Date.now() / 1000 }, 1000)
    const poll = setInterval(() => { if (own && !write.busy) void consent.load(true) }, 7000)
    const unregister = session.onPurge(() => { room = '' })
    return () => {
      clearInterval(clock); clearInterval(poll); unregister()
      consent.destroy(); roomList.destroy(); global.destroy(); details.destroy(); write.destroy()
    }
  })

  function personalChange(action: 'opt-in' | 'revoke' | 'verify') {
    const revision = consent.data?.revision
    if (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0) return
    if (action === 'revoke' && !window.confirm('¿Revocar tu consentimiento de memoria por voz?')) return
    if (action === 'verify' && !roomList.data?.some(item => item.room_id === room)) return
    return write.run(ownPath + (action === 'verify' ? '/verification' : '/consent'), {
      permissions: ['memory:own:voice'], method: action === 'revoke' ? 'DELETE' : 'POST',
      body: { expected_revision: revision, ...(action === 'verify' ? { room_id: room } : {}) },
    }, () => consent.load())
  }

  function changeSwitch() {
    const current = global.data
    if (typeof current?.enabled !== 'boolean' || !Number.isSafeInteger(current.revision) || current.configured !== true) return
    if (!window.confirm(current.enabled ? '¿Pausar el uso de memoria por voz en todo el hogar?' : '¿Habilitar el uso de memoria por voz para consentimientos vigentes?')) return
    return write.run(switchPath, { permissions: ['memory:assign', 'admin:manage'], method: 'POST',
      body: { enabled: !current.enabled, expected_revision: current.revision } },
    async () => { await Promise.all([global.load(), details.load()]) })
  }
</script>

<section>
  <h1>Voces y consentimientos</h1>
  <p>El consentimiento de memoria por voz requiere una decisión de la persona. Un perfil de voz no otorga acceso ni permisos.</p>
  {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
  <div class="grid">
    {#if own}
      <article class="card">
        <h2>Mi consentimiento</h2>
        {#if consent.error}<p class="error" role="alert">{consent.error}</p>{/if}
        {#if consent.data}
          <p>{statuses[text(consent.data.state)] ?? 'Sin inscripción'}</p>
          {#if typeof consent.data.disclosure === 'string'}<p>{consent.data.disclosure}</p>{/if}
          {#if consent.data.enabled === true && (consent.data.state === null || consent.data.state === 'revoked' || consent.data.state === 'invalidated')}
            <button disabled={write.busy || consent.busy} onclick={() => personalChange('opt-in')}>Solicitar inscripción</button>
          {/if}
          {#if consent.data.state && consent.data.state !== 'revoked'}
            <button class="danger" disabled={write.busy || consent.busy} onclick={() => personalChange('revoke')}>Revocar mi consentimiento</button>
          {/if}
          {#if windowCurrent && verification}
            <p>Decí la frase en la habitación elegida antes de que venza la verificación.</p>
            {#if typeof verification.phrase === 'string'}<p>{verification.phrase}</p>{/if}
          {:else if consent.data.enabled === true && consent.data.state === 'pending_verification' && roomList.data?.length}
            <form class="inline-form" onsubmit={event => { event.preventDefault(); void personalChange('verify') }}>
              <label>Habitación <select bind:value={room} required>
                <option value="">Elegí una habitación</option>
                {#each roomList.data as item}
                  {#if typeof item.room_id === 'string'}<option value={item.room_id}>{text(item.name)}</option>{/if}
                {/each}
              </select></label>
              <button disabled={write.busy || consent.busy || !room}>Abrir verificación</button>
            </form>
          {/if}
        {/if}
      </article>
    {/if}
    {#if administration}
      <article class="card">
        <h2>Memoria por voz del hogar</h2>
        {#if global.error}<p class="error" role="alert">{global.error}</p>{/if}
        {#if global.data}
          <p>{global.data.effective === true ? 'Habilitada' : 'Pausada o sin configurar'}</p>
          {#if session.can('memory:assign', 'admin:manage') && global.data.configured === true}
            <button disabled={write.busy || global.busy} aria-pressed={global.data.enabled === true} onclick={changeSwitch}>
              {global.data.enabled === true ? 'Pausar memoria por voz' : 'Habilitar memoria por voz'}
            </button>
          {/if}
        {/if}
        {#if details.error}<p class="error" role="alert">{details.error}</p>{/if}
        {#if details.data}
          <ul>
            {#each rows(details.data.consents) as entry}
              <li>Persona {text(entry.person_id)} · {statuses[text(entry.state)] ?? 'Estado desconocido'}</li>
            {:else}<li class="muted">Sin consentimientos registrados.</li>{/each}
          </ul>
        {/if}
        <button disabled={write.busy || global.busy || details.busy} onclick={() => { void global.load(); void details.load() }}>Actualizar consentimientos</button>
      </article>
    {/if}
  </div>
</section>
{#if review}<VoiceReview {session} />{/if}
