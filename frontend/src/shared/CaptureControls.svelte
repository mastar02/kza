<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session }: { session: Session } = $props()
  const ownerSession = untrack(() => session)
  const captureAllowed = $derived(session.can('admin:manage'))
  const ambientAllowed = $derived(session.can('ambient:control'))
  const capturePath = '/api/capture/control'
  const ambientPath = '/api/ambient/control'
  const capture = new Resource(ownerSession, async signal =>
    record(await session.request(capturePath, { permissions: ['admin:manage'], signal })) ?? null)
  const ambient = new Resource(ownerSession, async signal =>
    record(await session.request(ambientPath, { permissions: ['ambient:control'], signal })) ?? null)
  const write = new Mutation(ownerSession)
  const captureLabels: Record<string, string> = {
    initializing: 'Preparando el control', running: 'Captura habilitada', muting: 'Pausa en curso',
    resuming: 'Reactivación en curso', closed: 'Control cerrado', failed: 'Error en el control de captura',
  }
  const ambientLabels: Record<string, string> = {
    initializing: 'Preparando el control', running: 'Captura ambiental habilitada', pausing: 'Pausa en curso',
    resuming: 'Reactivación en curso', closed: 'Control cerrado', failed: 'Error en el control ambiental',
  }
  const fullPause = $derived(capture.data?.phase === 'muted' && capture.data.total_mute_confirmed === true &&
    capture.data.supported === true && capture.data.target_muted === true && capture.data.persisted_muted === true &&
    capture.data.ingress_sealed === true &&
    ['pending_operations', 'pending_browser_recorders', 'pending_native_clients', 'live_capture_owners', 'carried_capture_owners']
      .every(key => capture.data?.[key] === 0))
  const ambientPause = $derived(ambient.data?.phase === 'paused' && ambient.data.admission_paused === true &&
    ambient.data.persisted_paused === true && ambient.data.pending_operations === 0)

  onMount(() => {
    function refresh() {
      if (write.busy) return
      if (captureAllowed) void capture.load(true)
      if (ambientAllowed) void ambient.load(true)
    }
    refresh()
    const timer = setInterval(refresh, 2000)
    return () => { clearInterval(timer); capture.destroy(); ambient.destroy(); write.destroy() }
  })

  function change(type: 'capture' | 'ambient', value: boolean) {
    const resource = type === 'capture' ? capture : ambient
    const revision = resource.data?.revision
    if (resource.busy || typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0) return
    return write.run(type === 'capture' ? capturePath : ambientPath, {
      permissions: type === 'capture' ? ['admin:manage'] : ['ambient:control'],
      method: 'POST',
      body: type === 'capture' ? { muted: value, expected_revision: revision } : { paused: value, expected_revision: revision },
    }, () => resource.load())
  }
</script>

{#if captureAllowed || ambientAllowed}
  <section>
    <h2>Controles de captura</h2>
    {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
    <div class="grid">
      {#if captureAllowed}
        <article class="card">
          <h3>Captura global</h3>
          <p>{fullPause ? 'Pausa global confirmada' : captureLabels[text(capture.data?.phase)] ?? 'Pausa global sin confirmar'}</p>
          {#if capture.error}<p class="error" role="alert">{capture.error}</p>{/if}
          {#if capture.data?.supported === true && ['running', 'failed', 'muted'].includes(text(capture.data.phase))}
            <button disabled={capture.busy || write.busy || fullPause} onclick={() => change('capture', true)}>Pausar captura global</button>
            <button disabled={capture.busy || write.busy || capture.data.phase === 'running'} onclick={() => change('capture', false)}>Reactivar captura global</button>
          {/if}
        </article>
      {/if}
      {#if ambientAllowed}
        <article class="card">
          <h3>Captura ambiental</h3>
          <p>{ambientPause ? 'Pausa ambiental confirmada' : ambientLabels[text(ambient.data?.phase)] ?? 'Pausa ambiental sin confirmar'}</p>
          {#if ambient.error}<p class="error" role="alert">{ambient.error}</p>{/if}
          {#if ambient.data && ['running', 'failed', 'paused'].includes(text(ambient.data.phase))}
            <button disabled={ambient.busy || write.busy || ambientPause} onclick={() => change('ambient', true)}>Pausar captura ambiental</button>
            <button disabled={ambient.busy || write.busy || ambient.data.phase === 'running'} onclick={() => change('ambient', false)}>Reactivar captura ambiental</button>
          {/if}
        </article>
      {/if}
    </div>
  </section>
{/if}
