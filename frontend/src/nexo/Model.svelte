<script lang="ts">
  import { onMount } from 'svelte'
  import type { Catalog } from '../lib/home'
  import type { Session } from '../lib/session'
  import { parseHousePlan } from './house-plan'
  import type { LayerMode, SceneHandle } from './scene'

  let {
    model,
    plan,
    catalog,
    selectedRoomId,
    onSelectRoom,
    session,
    fallback,
  }: {
    model: ArrayBuffer
    plan: Record<string, unknown> | null
    catalog: Catalog | null
    selectedRoomId: string | null
    onSelectRoom: (roomId: string) => void
    session: Session
    fallback: () => void
  } = $props()

  let stage: HTMLDivElement
  let canvas: HTMLCanvasElement
  let pins: HTMLDivElement
  let scene = $state<SceneHandle | null>(null)
  let viewMode = $state<'3d' | 'top'>('3d')
  let layer = $state<LayerMode>('lights')
  const interactive = $derived(parseHousePlan(plan) !== null)

  onMount(() => {
    const owner = new AbortController()
    const lost = () => {
      if (!owner.signal.aborted) fallback()
    }
    canvas.addEventListener('webglcontextlost', lost)
    const unregister = session.onPurge(() => owner.abort())
    void import('./scene')
      .then(async ({ mountScene }) => {
        if (owner.signal.aborted) return
        const result = await mountScene(canvas, model, plan, owner.signal, {
          pins,
          stage,
          onSelectRoom,
        })
        if (owner.signal.aborted) {
          result.dispose()
        } else {
          scene = result
          result.update({ catalog, layer, selectedRoomId })
        }
      })
      .catch(() => {
        if (!owner.signal.aborted) fallback()
      })
    return () => {
      canvas.removeEventListener('webglcontextlost', lost)
      owner.abort()
      scene?.dispose()
      scene = null
      unregister()
    }
  })

  $effect(() => {
    scene?.update({ catalog, layer, selectedRoomId })
  })

  function changeView(next: '3d' | 'top') {
    viewMode = next
    scene?.setView(next)
  }
</script>

<div class="map-card">
  {#if scene}
    <div class="map-toolbar">
      <div class="button-row">
        <button aria-pressed={viewMode === '3d'} onclick={() => changeView('3d')}>3D</button>
        <button aria-pressed={viewMode === 'top'} onclick={() => changeView('top')}>Plano</button>
        {#if interactive}
          <button aria-pressed={layer === 'lights'} onclick={() => { layer = 'lights' }}>Luces</button>
          <button aria-pressed={layer === 'climate'} onclick={() => { layer = 'climate' }}>Clima</button>
        {/if}
      </div>
      <div class="button-row">
        <button aria-label="Girar a la izquierda" onclick={() => scene?.rotate(-1)}>↺</button>
        <button aria-label="Girar a la derecha" onclick={() => scene?.rotate(1)}>↻</button>
        <button aria-label="Acercar" onclick={() => scene?.zoom(1.18)}>+</button>
        <button aria-label="Alejar" onclick={() => scene?.zoom(0.85)}>−</button>
        <button onclick={() => { viewMode = '3d'; scene?.reset() }}>Centrar</button>
      </div>
    </div>
  {:else}
    <p class="muted map-loading" role="status">Cargando el visor 3D…</p>
  {/if}
  <div class="map-viewport" bind:this={stage}>
    <canvas bind:this={canvas} aria-label="Modelo 3D de la casa"></canvas>
    <div class="room-pins" bind:this={pins}></div>
  </div>
</div>

<style>
  .map-card {
    border: 1px solid var(--line);
    border-radius: .8rem;
    background: var(--panel);
    margin-block-end: 1rem;
    overflow: hidden;
  }
  .map-toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: .5rem;
    padding: .65rem .85rem;
    border-bottom: 1px solid var(--line);
    background: var(--raised);
  }
  .map-loading { padding: .65rem .85rem; margin: 0; }
  .map-viewport {
    position: relative;
    block-size: clamp(20rem, 56vh, 36rem);
    overflow: hidden;
    touch-action: none;
    background: radial-gradient(ellipse at center, #213032, var(--bg) 78%);
  }
  canvas {
    display: block;
    inline-size: 100%;
    block-size: 100%;
    touch-action: none;
    cursor: grab;
  }
  canvas:active { cursor: grabbing; }
  canvas:global([data-room-hover]) { cursor: pointer; }
  .room-pins { position: absolute; inset: 0; pointer-events: none; }
  .room-pins :global(.room-pin) {
    position: absolute;
    inset: 0 auto auto 0;
    display: grid;
    place-items: center;
    min-inline-size: 2.5rem;
    min-block-size: 2.5rem;
    background: transparent;
    border: 0;
    padding: 0;
    cursor: pointer;
    pointer-events: auto;
  }
  .room-pins :global(.room-pin .tag) {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .25rem .55rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(24, 34, 36, 0.9);
    color: var(--text);
    font-size: .82rem;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
  }
  .room-pins :global(.room-pin[aria-pressed='true'] .tag) {
    border-color: var(--accent);
    background: var(--accent);
    color: var(--ink);
    font-weight: 600;
  }
  .room-pins :global(.room-pin[data-room-lit] .tag) {
    border-color: #ffd6a0;
  }
  .room-pins :global(.room-pin[data-room-hover] .tag) {
    border-color: var(--accent);
  }
  .room-pins :global(.pin-meta) {
    opacity: .85;
    font-size: .75rem;
  }
</style>
