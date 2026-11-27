<script lang="ts">
  import { onMount, tick } from 'svelte'
  import { SampleMedia, fingerprint, mediaRetired, sampleFrom } from '../lib/media'
  import type { Sample } from '../lib/media'
  import { ReceiptSocket } from '../lib/ws'
  import type { StreamStatus } from '../lib/ws'
  import { messageFor, record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session }: { session: Session } = $props()
  const permissions = ['voices:read', 'voices:review']
  type Profile = { id: string; name: string; revision: number }
  type Capture = { id: string; room: string | null; seconds: number | null; review: boolean }
  const profileId = /^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/
  const media = new SampleMedia()
  let profileRows = $state.raw<Profile[]>([])
  let captureRows = $state.raw<Capture[]>([])
  let sample = $state.raw<Sample | null>(null)
  let player = $state<HTMLDivElement>()
  let status = $state<StreamStatus>('connecting')
  let error = $state('')
  let busy = $state(false)
  let changing = $state(false)
  let selected = $state<string[]>([])
  let profileName = $state('')
  let client: ReceiptSocket | undefined

  function profiles(items: readonly unknown[]): Profile[] {
    return items.map(item => {
      const row = record(item)
      if (typeof row?.id !== 'string' || !profileId.test(row.id) || typeof row.name !== 'string' ||
          !row.name.trim() || row.name.length > 64 || typeof row.revision !== 'number' ||
          !Number.isSafeInteger(row.revision) || row.revision < 1) throw new Error('Invalid profile')
      return { id: row.id, name: row.name, revision: row.revision }
    })
  }

  function captures(items: readonly unknown[]): Capture[] {
    return items.map(item => {
      const row = record(item)
      const id = typeof row?.id === 'string' ? row.id
        : typeof row?.capture_id === 'string' ? row.capture_id
        : typeof row?.fingerprint_id === 'string' ? row.fingerprint_id : null
      if (!id || !fingerprint.test(id)) throw new Error('Invalid capture')
      const seconds = typeof row?.duration_s === 'number' && Number.isFinite(row.duration_s) && row.duration_s > 0
        ? row.duration_s : null
      const review = typeof row?.review_available === 'boolean' ? row.review_available : false
      return { id, room: typeof row?.room_id === 'string' ? row.room_id : null, seconds, review }
    })
  }

  function clearSample() {
    media.clear()
    sample?.audio?.fill(0)
    sample = null
  }

  onMount(() => {
    client = new ReceiptSocket(session, '/ws/voices', {
      status(value) {
        status = value
        if (value === 'ready') void refresh()
      },
      changed(channel, data) {
        if (channel === 'voices:review') {
          clearSample()
          sample = data as Sample | null
        } else if (channel === 'voices:profiles') {
          profileRows = data as Profile[] ?? []
        } else if (channel === 'voices:captures') {
          captureRows = data as Capture[] ?? []
          selected = []
          profileName = ''
          if (!data) client?.discard('voices:review')
        }
      },
      cleared() {
        clearSample()
        selected = []
        profileName = ''
        error = ''
      },
      async committed() { await tick(); await mediaRetired() },
    })
    const expire = () => { client?.expire(); if (document.hidden) client?.discard('voices:review') }
    document.addEventListener('visibilitychange', expire)
    window.addEventListener('focus', expire)
    return () => {
      client?.stop()
      client = undefined
      document.removeEventListener('visibilitychange', expire)
      window.removeEventListener('focus', expire)
    }
  })

  async function refresh() {
    if (!client?.ready || changing) return
    error = ''
    busy = true
    try {
      await Promise.all([
        client.read('voices:profiles', '/api/voices/profiles', { permissions }, profiles),
        client.read('voices:captures', '/api/voices/captures/query', { permissions, method: 'POST',
          body: { limit: 50 } }, captures),
      ])
    } catch (failure) { if (client?.ready) error = messageFor(failure) }
    finally { busy = false }
  }

  async function review(id: string, format: 'text' | 'audio') {
    if (!client?.ready || busy || changing || !captureRows.some(row => row.id === id && (format === 'text' || row.review))) return
    client.discard('voices:review')
    busy = true
    error = ''
    try {
      await client.read('voices:review', '/api/voices/captures/' + id + (format === 'audio' ? '/review' : '/transcript'),
        { permissions }, items => sampleFrom(items, id, format))
    } catch (failure) { if (client?.ready) error = messageFor(failure) }
    finally { busy = false }
  }

  async function change(path: string, method: 'POST' | 'DELETE', body: unknown) {
    if (!client?.ready || busy || changing) return
    changing = true
    for (const channel of ['voices:profiles', 'voices:captures', 'voices:review'] as const) client.discard(channel)
    await tick()
    await mediaRetired()
    try {
      if (!client?.ready || !await session.revalidate() || !client?.ready) return
      await session.request(path, { permissions, method, body })
    } catch (failure) {
      if (client?.ready) error = messageFor(failure)
    } finally {
      changing = false
      if (client?.ready) {
        const previousError = error
        await refresh()
        if (previousError && !error) error = previousError
      }
    }
  }

  function confirmVoice(event: SubmitEvent) {
    event.preventDefault()
    const name = profileName.trim()
    if (selected.length < 3 || selected.length > 32 || !name ||
        !selected.every(id => captureRows.some(row => row.id === id))) return
    void change('/api/voices/profiles', 'POST', { fingerprint_ids: [...selected], name, request_id: crypto.randomUUID() })
  }
</script>

<section>
  <div class="heading-row">
    <h2>Revisión de voces</h2>
    <button disabled={status !== 'ready' || busy || changing} onclick={refresh}>Actualizar voces</button>
  </div>
  {#if status !== 'ready'}<p role="status">{status === 'connecting' ? 'Conectando revisión de voces…' : 'Canal de voces desconectado.'}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  <div class="grid">
    <article class="card">
      <h3>Perfiles de voz</h3>
      <ul>
        {#each profileRows as profile (profile.id)}
          <li>{profile.name}
            <button class="danger" disabled={busy || changing} onclick={() => {
              if (window.confirm('¿Eliminar este perfil de voz?')) void change('/api/voices/profiles/' + profile.id,
                'DELETE', { expected_revision: profile.revision })
            }}>Eliminar perfil</button>
          </li>
        {:else}<li class="muted">Sin perfiles vigentes.</li>{/each}
      </ul>
    </article>
    <article class="card">
      <h3>Capturas vigentes</h3>
      <ul>
        {#each captureRows as capture, index (capture.id)}
          <li>
            <label class="capture-choice"><input type="checkbox" bind:group={selected} value={capture.id} disabled={changing} />
              Captura {index + 1}{capture.seconds !== null ? ' · ' + capture.seconds.toFixed(1) + ' s' : ''}{capture.room ? ' · ' + capture.room : ''}
            </label>
            <button disabled={busy || changing} onclick={() => review(capture.id, 'text')}>Leer</button>
            {#if capture.review}<button disabled={busy || changing} onclick={() => review(capture.id, 'audio')}>Consultar audio</button>{/if}
          </li>
        {:else}<li class="muted">Sin capturas vigentes.</li>{/each}
      </ul>
      {#if selected.length >= 3}
        <form class="inline-form" onsubmit={confirmVoice}>
          <label>Nombre del perfil de voz <input bind:value={profileName} required maxlength="64" /></label>
          <button disabled={busy || changing || selected.length > 32}>Confirmar perfil</button>
        </form>
      {/if}
    </article>
  </div>
  {#if sample}
    <article class="card">
      <div class="heading-row"><h3>Muestra de voz</h3><button onclick={() => client?.discard('voices:review')}>Cerrar muestra</button></div>
      {#if sample.transcript}<p>{sample.transcript}</p>{/if}
      {#if sample.audio}
        <div bind:this={player}></div>
        <button onclick={() => { if (player && sample?.audio && client?.ready) media.play(player, sample.audio) }}>Reproducir</button>
      {/if}
    </article>
  {/if}
</section>

<style>
  .capture-choice { display: flex; flex-direction: row; align-items: center; }
  .capture-choice input { min-block-size: 1.2rem; }
</style>
