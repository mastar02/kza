<script lang="ts">
  import type { Role, Session } from '../lib/session'
  import Frame from '../shared/Frame.svelte'
  import Home from './Home.svelte'
  import Routines from './Routines.svelte'
  import Tasks from './Tasks.svelte'
  import Memory from './Memory.svelte'
  import Consents from './Consents.svelte'
  import Access from './Access.svelte'

  let { session, role }: { session: Session; role: Role } = $props()
  const tabs = [
    { id: 'home', label: 'Casa', allowed: () => session.can('home:read') },
    { id: 'routines', label: 'Rutinas', allowed: () => session.can('home:read') },
    { id: 'tasks', label: 'Tareas', allowed: () => session.can('tasks:read') },
    { id: 'memory', label: 'Memoria', allowed: () => session.can('memory:read') },
    { id: 'mine', label: 'Mi Memoria', allowed: () => role === 'home' && session.can('memory:own:read') },
    { id: 'voices', label: 'Voces y consentimientos', allowed: () => session.can('voices:read', 'voices:review') ||
      session.can('memory:read', 'memory:assign') || role === 'home' && session.can('memory:own:voice') },
    { id: 'access', label: 'Acceso y sistema', allowed: () => session.can('admin:manage') ||
      session.can('ambient:control') || role === 'operator' && session.can('auth:manage') },
  ] as const
  let selection = $state<string | null>(null)
  let available = $derived(tabs.filter(tab => tab.allowed()))
  let active = $derived(available.find(tab => tab.id === selection)?.id ?? available[0]?.id)
</script>

<Frame brand="NEXO" {role} logout={() => session.logout()}>
  {#if available.length}
    <nav aria-label="Secciones del hogar" class="tabs">
      {#each available as tab (tab.id)}
        <button aria-current={active === tab.id ? 'page' : undefined} onclick={() => selection = tab.id}>{tab.label}</button>
      {/each}
    </nav>
  {/if}
  {#if active === 'home'}<Home {session} />
  {:else if active === 'routines'}<Routines {session} />
  {:else if active === 'tasks'}<Tasks {session} />
  {:else if active === 'memory' || active === 'mine'}
    {#key active}<Memory {session} personal={active === 'mine'} />{/key}
  {:else if active === 'voices'}<Consents {session} {role} />
  {:else if active === 'access'}<Access {session} {role} />
  {:else}<p>Esta sesión no tiene secciones del hogar habilitadas.</p>{/if}
</Frame>
