<script lang="ts">
  import { onMount } from 'svelte'
  import type { mountScene } from './scene'
  import type { Session } from '../lib/session'

  let { model, plan, session, fallback }: {
    model: ArrayBuffer; plan: Record<string, unknown> | null; session: Session; fallback: () => void
  } = $props()
  let canvas: HTMLCanvasElement
  let scene = $state<Awaited<ReturnType<typeof mountScene>> | null>(null)

  onMount(() => {
    const owner = new AbortController()
    const lost = () => { if (!owner.signal.aborted) fallback() }
    canvas.addEventListener('webglcontextlost', lost)
    const unregister = session.onPurge(() => owner.abort())
    void import('./scene').then(async ({ mountScene }) => {
      if (owner.signal.aborted) return
      const result = await mountScene(canvas, model, plan, owner.signal)
      if (owner.signal.aborted) result.dispose()
      else scene = result
    }).catch(() => { if (!owner.signal.aborted) fallback() })
    return () => {
      canvas.removeEventListener('webglcontextlost', lost)
      owner.abort()
      scene?.dispose()
      scene = null
      unregister()
    }
  })
</script>

<canvas bind:this={canvas} aria-label="Modelo 3D de la casa"></canvas>
{#if scene}
  <div class="button-row">
    <button onclick={() => scene?.rotate(-Math.PI / 8)}>Girar a la izquierda</button>
    <button onclick={() => scene?.rotate(Math.PI / 8)}>Girar a la derecha</button>
    <button onclick={() => scene?.zoom(1.25)}>Acercar</button>
    <button onclick={() => scene?.zoom(0.8)}>Alejar</button>
    <button onclick={() => scene?.top()}>Cambiar perspectiva</button>
  </div>
{:else}
  <p class="muted" role="status">Cargando el visor 3D…</p>
{/if}

<style>
  canvas { display: block; inline-size: 100%; block-size: clamp(16rem, 45vw, 30rem); background: var(--raised); }
</style>
