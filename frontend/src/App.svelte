<script lang="ts">
  import { flushSync, onMount } from 'svelte'
  import type { Session, SessionState } from './lib/session'
  import Nexo from './nexo/Nexo.svelte'
  import Obs from './obs/Obs.svelte'

  let { session, pathname }: { session: Session; pathname: string } = $props()
  let state = $state<SessionState>({ status: 'checking' })
  let technical = $derived(pathname === '/obs' || pathname === '/obs/')
  let found = $derived(technical || pathname === '/')

  onMount(() => {
    if (!found) return
    const unsubscribe = session.subscribe(next => flushSync(() => { state = next }))
    void session.start()
    const check = () => { if (!document.hidden) session.checkOnReturn() }
    const hide = () => session.retire()
    window.addEventListener('focus', check)
    window.addEventListener('pagehide', hide)
    document.addEventListener('visibilitychange', check)
    const leaseCheck = setInterval(check, 15000)
    return () => {
      clearInterval(leaseCheck)
      window.removeEventListener('focus', check)
      window.removeEventListener('pagehide', hide)
      document.removeEventListener('visibilitychange', check)
      unsubscribe()
      session.dispose()
    }
  })

  function login(event: SubmitEvent & { currentTarget: HTMLFormElement }) {
    event.preventDefault()
    const field = event.currentTarget.elements.namedItem('token')
    if (!(field instanceof HTMLInputElement) || !field.value.trim()) return
    const token = field.value.trim()
    field.value = ''
    void session.start(token)
  }
</script>

<svelte:head><title>{technical ? 'Observabilidad · KZA' : 'NEXO · KZA'}</title></svelte:head>

{#if !found}
  <main class="login"><h1>Página no encontrada</h1><a href="/">Ir a NEXO</a></main>
{:else if state.status === 'ready'}
  {#key state.generation}
    {#if technical}<Obs {session} role={state.role} />
    {:else}<Nexo {session} role={state.role} />{/if}
  {/key}
{:else}
  <main class="login">
    <div class="brand"><span class="brand-mark" aria-hidden="true"></span>{technical ? 'KZA' : 'NEXO'}</div>
    <h1>{technical ? 'Observabilidad' : 'Tu hogar'}</h1>
    {#if state.status === 'checking'}
      <p role="status">Verificando sesión…</p>
    {:else if state.status === 'login'}
      <form onsubmit={login}>
        <label for="token">Credencial de acceso</label>
        <input id="token" name="token" type="password" autocomplete="current-password" required />
        <button class="primary" type="submit">Ingresar</button>
      </form>
      {#if state.message}<p role="status">{state.message}</p>{/if}
    {:else}
      <p role="alert">{state.message}</p>
      <button onclick={() => session.start()}>Comprobar sesión</button>
    {/if}
  </main>
{/if}
