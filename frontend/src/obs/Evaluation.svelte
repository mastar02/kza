<script lang="ts">
  import { onMount, tick } from 'svelte'
  import { SampleMedia, fingerprint, mediaRetired, sampleFrom } from '../lib/media'
  import type { Sample } from '../lib/media'
  import { deadline, ReceiptSocket, remaining } from '../lib/ws'
  import type { StreamStatus } from '../lib/ws'
  import { messageFor, record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session }: { session: Session } = $props()
  type Modality = 'audio_preview' | 'transcript_full'
  type Grant = { id: string; modalities: Modality[]; expires: number }
  const permissions = ['evaluation:read']
  const media = new SampleMedia()
  let grants = $state.raw<Grant[]>([])
  let sample = $state.raw<Sample | null>(null)
  let player = $state<HTMLDivElement>()
  let status = $state<StreamStatus>('connecting')
  let error = $state('')
  let busy = $state(false)
  let client: ReceiptSocket | undefined
  let grantController: AbortController | undefined
  let grantDeadline: ReturnType<typeof deadline> = null
  let grantTimer: ReturnType<typeof setTimeout> | undefined

  function clearSample() {
    media.clear()
    sample?.audio?.fill(0)
    sample = null
  }

  function clearGrants() {
    grantController?.abort()
    grantController = undefined
    grantDeadline = null
    clearTimeout(grantTimer)
    grants = []
  }

  function expire() {
    if (grantDeadline && remaining(grantDeadline) <= 0) {
      clearGrants()
      client?.discard('evaluation:sample')
      if (client?.ready) void refresh()
    }
    client?.expire()
  }

  onMount(() => {
    client = new ReceiptSocket(session, '/ws/evaluation', {
      status(value) { status = value; if (value === 'ready') void refresh() },
      changed(_channel, data) { clearSample(); sample = data as Sample | null },
      cleared() { clearGrants(); clearSample(); error = '' },
      retired() { clearGrants() },
      async committed() { await tick(); await mediaRetired() },
    })
    const visibility = () => { if (document.hidden) client?.discard('evaluation:sample'); else expire() }
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('focus', expire)
    return () => {
      client?.stop(); client = undefined
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('focus', expire)
    }
  })

  async function refresh() {
    if (!client?.ready || grantController) return
    client.discard('evaluation:sample')
    clearGrants()
    error = ''
    const owner = new AbortController()
    grantController = owner
    const started = { wall: Date.now(), mono: performance.now() }
    try {
      const value = record(await session.request('/api/evaluation/grants', { permissions, signal: owner.signal }))
      if (owner.signal.aborted || !client?.ready) return
      const end = deadline(value?.expires_at, value?.observed_at, started)
      if (!value || !end || remaining(end) <= 0 || typeof value.expires_at !== 'number' ||
          !Array.isArray(value.grants) || value.grants.length > 64) throw new Error('Invalid catalog')
      const seen = new Set<string>()
      grants = value.grants.map(item => {
        const row = record(item)
        if (typeof row?.fingerprint_id !== 'string' || !fingerprint.test(row.fingerprint_id) ||
            seen.has(row.fingerprint_id) || typeof row.expires_at !== 'number' || !Number.isFinite(row.expires_at) ||
            row.expires_at < Number(value.expires_at) || !Array.isArray(row.modalities) || !row.modalities.length ||
            row.modalities.length > 2 || !row.modalities.every(mode => mode === 'audio_preview' || mode === 'transcript_full')) {
          throw new Error('Invalid grant')
        }
        seen.add(row.fingerprint_id)
        return { id: row.fingerprint_id, expires: row.expires_at, modalities: row.modalities as Modality[] }
      })
      grantDeadline = end
      grantTimer = setTimeout(expire, Math.max(1, Math.min(remaining(end), 2147483647)))
    } catch (failure) {
      if (!owner.signal.aborted && client?.ready) error = messageFor(failure)
    } finally {
      if (grantController === owner) grantController = undefined
    }
  }

  async function access(grant: Grant, choice: Modality) {
    expire()
    if (busy || !client?.ready || !grantDeadline || !grants.includes(grant) || !grant.modalities.includes(choice)) return
    client.discard('evaluation:sample')
    error = ''
    busy = true
    try {
      const end = grantDeadline.wall / 1000
      await client.read('evaluation:sample', '/api/evaluation/samples/' + grant.id + '?modality=' + choice,
        { permissions }, items => sampleFrom(items, grant.id, choice === 'audio_preview' ? 'audio' : 'text', true),
        grant.expires)
      if (Date.now() / 1000 >= end) expire()
    } catch (failure) {
      if (client?.ready) error = messageFor(failure)
    } finally {
      busy = false
    }
  }
</script>

<section>
  <div class="heading-row">
    <h2>Evaluación</h2>
    <button disabled={status !== 'ready' || busy} onclick={refresh}>Actualizar concesiones</button>
  </div>
  {#if status !== 'ready'}<p role="status">{status === 'connecting' ? 'Conectando evaluación…' : 'Canal de evaluación desconectado.'}</p>{/if}
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  <div class="grid">
    {#each grants as grant, index (grant.id)}
      <article class="card">
        <h3>Muestra {index + 1}</h3>
        {#each grant.modalities as choice}
          <button disabled={busy || status !== 'ready'} onclick={() => access(grant, choice)}>{choice === 'audio_preview' ? 'Consultar audio' : 'Leer transcripción'}</button>
        {/each}
      </article>
    {:else}
      <p class="muted">Sin muestras concedidas vigentes.</p>
    {/each}
  </div>
  {#if sample}
    <article class="card">
      <div class="heading-row"><h3>Muestra de evaluación</h3><button onclick={() => client?.discard('evaluation:sample')}>Cerrar muestra</button></div>
      {#if sample.transcript}<p>{sample.transcript}</p>{/if}
      {#if sample.audio}
        <div bind:this={player}></div>
        <button onclick={() => { expire(); if (player && sample?.audio && client?.ready) media.play(player, sample.audio) }}>Reproducir</button>
      {/if}
    </article>
  {/if}
</section>
