<script lang="ts">
  import type { Role, Session } from '../lib/session'
  import Frame from '../shared/Frame.svelte'
  import CaptureControls from '../shared/CaptureControls.svelte'
  import VoiceReview from '../shared/VoiceReview.svelte'
  import Live from './Live.svelte'
  import Diagnostics from './Diagnostics.svelte'
  import Evaluation from './Evaluation.svelte'

  let { session, role }: { session: Session; role: Role } = $props()
  const tabs = [
    { id: 'live', name: 'En vivo', allowed: () => session.can('conversations:read') },
    { id: 'voices', name: 'Voces', allowed: () => session.can('voices:read', 'voices:review') },
    { id: 'evaluation', name: 'Evaluación', allowed: () => role === 'evaluator' && session.can('evaluation:read') },
    { id: 'diagnostics', name: 'Diagnóstico', allowed: () => session.can('diagnostics:read') },
    { id: 'system', name: 'Sistema', allowed: () => session.can('diagnostics:read') },
    { id: 'capture', name: 'Captura', allowed: () => session.can('admin:manage') || session.can('ambient:control') },
  ] as const
  let selection = $state<string | null>(null)
  const visible = $derived(tabs.filter(tab => tab.allowed()))
  const selected = $derived(visible.find(tab => tab.id === selection)?.id ?? visible[0]?.id)
</script>

<Frame brand="KZA" {role} logout={() => session.logout()}>
  <h1>Observabilidad</h1>
  {#if visible.length}
    <nav class="tabs" aria-label="Secciones técnicas">
      {#each visible as tab (tab.id)}
        <button aria-current={selected === tab.id ? 'page' : undefined} onclick={() => selection = tab.id}>{tab.name}</button>
      {/each}
    </nav>
  {/if}
  {#if selected === 'live'}<Live {session} />
  {:else if selected === 'voices'}<VoiceReview {session} />
  {:else if selected === 'evaluation'}<Evaluation {session} />
  {:else if selected === 'diagnostics' || selected === 'system'}
    {#key selected}<Diagnostics {session} system={selected === 'system'} />{/key}
  {:else if selected === 'capture'}<CaptureControls {session} />
  {:else}<p>Esta sesión no tiene acceso a los paneles técnicos.</p>{/if}
</Frame>
