<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { messageFor, record } from '../lib/session'
  import type { Session } from '../lib/session'
  import { actionLabel, canControl, catalogFrom, commandFor, controlFor, loadHomeAssets, readReceipt, submitAction } from '../lib/home'
  import type { ActionReceipt, Entity, Operation } from '../lib/home'
  import { parseHousePlan } from './house-plan'
  import Model from './Model.svelte'

  let { session }: { session: Session } = $props()
  const ownerSession = untrack(() => session)
  let roomId = $state<string | null>(null)
  let view = $state<'rooms' | 'model'>('rooms')
  let userToggledView = $state(false)
  let modelFailed = $state(false)
  let now = $state(Date.now())
  let feedback = $state('')
  let activeEntity = $state<string | null>(null)
  let receipts = $state<Record<string, ActionReceipt>>({})
  const canReadMics = $derived(session.can('home:read', 'diagnostics:read'))
  const catalog = new Resource(ownerSession, async signal =>
    catalogFrom(await session.request('/api/home/catalog', { permissions: ['home:read'], signal })))
  const assets = new Resource(ownerSession, signal => loadHomeAssets(session, signal))
  const microphones = new Resource(ownerSession, signal =>
    session.request('/api/home/microphones', { permissions: ['home:read', 'diagnostics:read'], signal }))
  const room = $derived(catalog.data?.rooms.find(item => item.id === roomId) ?? catalog.data?.rooms[0] ?? null)
  const devices = $derived(catalog.data?.entities.filter(entity => room?.entityIds.includes(entity.id)) ?? [])
  const others = $derived(catalog.data?.entities.filter(entity =>
    !catalog.data?.rooms.some(item => item.entityIds.includes(entity.id))) ?? [])
  const microphoneRows = $derived(rows(microphones.data, 'microphones'))
  let owned = true

  $effect(() => {
    if (!userToggledView && !modelFailed && assets.data?.model && parseHousePlan(assets.data.plan)) {
      view = 'model'
    }
  })

  onMount(() => {
    void catalog.load()
    void assets.load()
    if (canReadMics) void microphones.load()
    const unregister = session.onPurge(clear)
    const clock = setInterval(() => { now = Date.now() }, 1000)
    const poll = setInterval(() => {
      void catalog.load(true)
      if (canReadMics) void microphones.load(true)
    }, 5000)
    return () => {
      owned = false
      unregister()
      clear()
      clearInterval(clock)
      clearInterval(poll)
      catalog.destroy()
      microphones.destroy()
      assets.destroy()
    }
  })

  function clear() {
    receipts = {}
    feedback = ''
    activeEntity = null
    roomId = null
    view = 'rooms'
    userToggledView = false
  }

  function fallback() {
    modelFailed = true
    view = 'rooms'
    assets.clear()
  }

  function selectView(next: 'rooms' | 'model') {
    userToggledView = true
    view = next
  }

  function waiting(receipt: ActionReceipt | undefined) {
    return receipt?.status === 'in_progress' || receipt?.status === 'uncertain' || receipt?.status === 'missing'
  }

  function currentControlValue(entity: Entity, operation: Operation): string {
    switch (operation) {
      case 'brightness': return entity.brightnessPct !== null ? String(entity.brightnessPct) : ''
      case 'color_temp_kelvin': return entity.colorTempKelvin !== null ? String(entity.colorTempKelvin) : ''
      case 'set_temperature': return entity.temperature !== null ? String(entity.temperature) : ''
      case 'set_hvac_mode': return entity.state || ''
      case 'set_cover_position': return entity.position !== null ? String(entity.position) : ''
      default: return ''
    }
  }

  function telemetryBadges(entity: Entity): string[] {
    const badges: string[] = []
    if (entity.brightnessPct !== null && entity.state === 'on') badges.push(`${entity.brightnessPct}%`)
    if (entity.colorTempKelvin !== null && entity.state === 'on') badges.push(`${entity.colorTempKelvin} K`)
    if (entity.currentTemperature !== null) badges.push(`Actual ${entity.currentTemperature} °C`)
    if (entity.temperature !== null) badges.push(`Objetivo ${entity.temperature} °C`)
    if (entity.currentHumidity !== null) badges.push(`${entity.currentHumidity}% HR`)
    if (entity.hvacAction && entity.hvacAction !== 'off') badges.push(entity.hvacAction)
    if (entity.position !== null) badges.push(`${entity.position}%`)
    return badges
  }

  async function run(entity: Entity, operation: Operation, raw?: string) {
    if (!catalog.data || activeEntity || waiting(receipts[entity.id])) return
    const cmd = commandFor(entity, operation, raw)
    if (!cmd) { feedback = 'Ingresá un valor dentro de los límites del dispositivo.'; return }
    activeEntity = entity.id
    feedback = ''
    try {
      const result = await submitAction(session, catalog.data, entity, cmd)
      if (owned && session.can('home:read')) {
        receipts[entity.id] = result
        await catalog.load(true)
      }
    } catch (error) {
      if (owned && session.can('home:read')) feedback = messageFor(error)
    } finally {
      if (owned) activeEntity = null
    }
  }

  async function reconcile(receipt: ActionReceipt) {
    if (activeEntity) return
    activeEntity = receipt.entity_id
    try {
      const next = await readReceipt(session, receipt)
      if (owned && session.can('home:read')) {
        receipts[receipt.entity_id] = next
        await catalog.load(true)
      }
    } catch (error) {
      if (owned && session.can('home:read')) feedback = messageFor(error)
    } finally {
      if (owned) activeEntity = null
    }
  }
</script>

{#snippet deviceCards(items: Entity[])}
  <div class="grid">
    {#each items as entity (entity.id)}
      {@const receipt = receipts[entity.id]}
      {@const badges = telemetryBadges(entity)}
      <article class="card">
        <div class="heading-row">
          <h3>{entity.name}</h3>
          <span class="pill">{entity.available ? (entity.state || 'Sin estado') : 'No disponible'}</span>
        </div>
        {#if badges.length}
          <p class="muted telemetry">{badges.join(' · ')}</p>
        {/if}
        {#each entity.capabilities as operation (operation)}
          {@const control = controlFor(entity, operation)}
          {@const disabled = !catalog.data || !canControl(session, catalog.data, entity, operation, now) ||
            activeEntity !== null || waiting(receipt)}
          {@const initialValue = currentControlValue(entity, operation)}
          {#if control && session.can('home:control')}
            {#if control.kind === 'button'}
              <button {disabled} onclick={() => run(entity, operation)}>{control.label}</button>
            {:else}
              <form class="inline-form" onsubmit={event => {
                event.preventDefault()
                const value = new FormData(event.currentTarget).get('value')
                if (typeof value === 'string') void run(entity, operation, value)
              }}>
                <label>{control.label}
                  {#if control.kind === 'select'}
                    <select name="value" value={initialValue} required {disabled}>
                      {#each control.options as option}<option value={option}>{option}</option>{/each}
                    </select>
                  {:else}
                    <input type="number" name="value" value={initialValue} min={control.min} max={control.max} step={control.step} required {disabled} />
                  {/if}
                </label>
                <button {disabled}>Aplicar</button>
              </form>
            {/if}
          {/if}
        {/each}
        {#if receipt}
          <p role="status">{actionLabel[receipt.status]}</p>
          {#if waiting(receipt)}
            <button disabled={activeEntity !== null} onclick={() => reconcile(receipt)}>Consultar recibo</button>
          {/if}
        {/if}
      </article>
    {:else}
      <p class="muted">No hay dispositivos en esta selección.</p>
    {/each}
  </div>
{/snippet}

<section aria-labelledby="home-title">
  <div class="heading-row">
    <h1 id="home-title">Casa</h1>
    <div class="button-row">
      {#if assets.data?.model && !modelFailed}
        <button aria-pressed={view === 'rooms'} onclick={() => selectView('rooms')}>Habitaciones</button>
        <button aria-pressed={view === 'model'} onclick={() => selectView('model')}>Vista 3D</button>
      {/if}
      <button disabled={catalog.busy} onclick={() => catalog.load(true)}>Actualizar</button>
    </div>
  </div>
  {#if catalog.error}<p class="error" role="alert">{catalog.error}</p>{/if}
  {#if feedback}<p class="error" role="alert">{feedback}</p>{/if}
  {#if view === 'model' && assets.data?.model}
    <Model
      model={assets.data.model}
      plan={assets.data.plan}
      catalog={catalog.data ?? null}
      selectedRoomId={room?.id ?? null}
      onSelectRoom={id => { roomId = id }}
      {session}
      {fallback}
    />
  {:else if modelFailed}
    <p class="muted">El visor 3D no está disponible.</p>
  {/if}
  {#if catalog.data}
    <nav class="button-row room-nav" aria-label="Habitaciones">
      {#each catalog.data.rooms as item (item.id)}
        <button aria-pressed={item.id === room?.id} onclick={() => roomId = item.id}>{item.name}</button>
      {/each}
    </nav>
    {#if room}
      <div class="heading-row">
        <h2>{room.name}</h2>
        <div class="button-row">
          {#if room.temperature !== null}<span class="pill">{room.temperature} °C</span>{/if}
          {#if room.humidity !== null}<span class="pill">{room.humidity}% HR</span>{/if}
        </div>
      </div>
      {@render deviceCards(devices)}
    {/if}
    {#if others.length}
      <h2>Otros dispositivos</h2>
      {@render deviceCards(others)}
    {/if}
  {:else if catalog.busy}
    <p role="status">Consultando la casa…</p>
  {/if}
</section>

{#if canReadMics}
  <section aria-labelledby="mic-title">
    <h2 id="mic-title">Micrófonos</h2>
    {#if microphones.error}<p class="error" role="alert">{microphones.error}</p>{/if}
    {#if record(microphones.data)?.stale === true}<p class="muted">Lectura de micrófonos vencida.</p>{/if}
    <div class="grid">
      {#each microphoneRows as mic}
        <article class="card">
          <h3>{text(mic.room_name ?? mic.product, 'Micrófono')}</h3>
          <dl>
            <div><dt>Conexión</dt><dd>{text(mic.connection)}</dd></div>
            <div><dt>Captura</dt><dd>{text(record(mic.capture)?.state)}</dd></div>
          </dl>
        </article>
      {:else}
        <p class="muted">{microphones.busy ? 'Consultando micrófonos…' : 'Sin micrófonos observados.'}</p>
      {/each}
    </div>
  </section>
{/if}

<style>
  .room-nav { margin-block: 0 1rem; }
  .telemetry { margin-block: 0 .6rem; font-size: .9rem; }
</style>
