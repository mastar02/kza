<script lang="ts">
  import { onMount, tick } from 'svelte'
  import { ReceiptSocket } from '../lib/ws'
  import type { LiveRow, StreamStatus } from '../lib/ws'
  import { messageFor, record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session }: { session: Session } = $props()
  type Turn = { id: string; time: string; room: string; voice: string; transcript: string; response: string }
  let live = $state.raw<LiveRow | null>(null)
  let turns = $state.raw<Turn[]>([])
  let status = $state<StreamStatus>('connecting')
  let error = $state('')
  let busy = $state(false)
  let client: ReceiptSocket | undefined
  const labels: Record<string, string> = { turn: 'Turno', wake: 'Activación', tts: 'Respuesta', alert: 'Alerta', cooldown: 'Cooldown' }

  onMount(() => {
    client = new ReceiptSocket(session, '/ws/live', {
      status(value) { status = value; if (value === 'ready') void refresh() },
      changed(channel, data) {
        if (channel === 'live') live = data as LiveRow | null
        else if (channel === 'conversations') turns = data as Turn[] ?? []
      },
      cleared() { error = '' },
      committed: tick,
    })
    const expire = () => client?.expire()
    window.addEventListener('focus', expire)
    document.addEventListener('visibilitychange', expire)
    return () => {
      client?.stop(); client = undefined
      window.removeEventListener('focus', expire)
      document.removeEventListener('visibilitychange', expire)
    }
  })

  function project(items: readonly unknown[]): Turn[] {
    return items.map(item => {
      const row = record(item)
      if (typeof row?.id !== 'string' || !/^turn_[0-9]+_[0-9a-f-]{36}$/.test(row.id) ||
          typeof row.stt !== 'string' || row.stt.length > 8192 || typeof row.tts !== 'string' || row.tts.length > 8192 ||
          typeof row.ts !== 'string' || row.ts.length > 512 || typeof row.zone !== 'string' || row.zone.length > 512 ||
          typeof row.user !== 'string' || row.user.length > 512) throw new Error('Invalid turn')
      return { id: row.id, time: row.ts, room: row.zone, voice: row.user, transcript: row.stt, response: row.tts }
    })
  }

  async function refresh() {
    if (!client?.ready || busy) return
    error = ''
    busy = true
    try {
      await client.read('conversations', '/api/conversations', { permissions: ['conversations:read'] }, project)
    } catch (failure) { if (client?.ready) error = messageFor(failure) }
    finally { busy = false }
  }
</script>

<section>
  <div class="heading-row">
    <h2>En vivo</h2>
    <span class="pill" role="status">{status === 'ready' ? 'Conectado' : status === 'connecting' ? 'Conectando' : 'Desconectado'}</span>
  </div>
  <article class="card" aria-live="polite">
    {#if live}
      <h3>{labels[live.type]}</h3>
      {#if live.title}<p>{live.title}</p>{/if}
      {#if live.text}<p>{live.text}</p>{/if}
      <p class="muted">{[live.ts, live.zone].filter(Boolean).join(' · ')}</p>
    {:else}
      <p class="muted">Sin eventos vigentes.</p>
    {/if}
  </article>
</section>

<section>
  <div class="heading-row">
    <h2>Conversaciones</h2>
    <button disabled={status !== 'ready' || busy} onclick={refresh}>Actualizar turnos</button>
  </div>
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  <div class="grid">
    {#each turns as turn (turn.id)}
      <article class="card">
        <h3>{[turn.time, turn.room].filter(Boolean).join(' · ') || 'Turno'}</h3>
        {#if turn.voice}<p class="muted">Voz · {turn.voice}</p>{/if}
        <p>{turn.transcript}</p>
        {#if turn.response}<p class="muted">{turn.response}</p>{/if}
      </article>
    {:else}<p class="muted">Sin turnos vigentes.</p>{/each}
  </div>
</section>
