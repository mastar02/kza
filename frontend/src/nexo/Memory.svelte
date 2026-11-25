<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { Mutation } from '../lib/mutation.svelte'
  import { Resource, rows, text } from '../lib/resource.svelte'
  import { record } from '../lib/session'
  import type { Session } from '../lib/session'

  let { session, personal }: { session: Session; personal: boolean } = $props()
  const ownerSession = untrack(() => session)
  let query = $state('')
  let now = $state(Date.now() / 1000)
  const readGrant = $derived(personal ? 'memory:own:read' : 'memory:read')
  const writeGrant = $derived(personal ? 'memory:own:write' : 'memory:review')
  const write = new Mutation(ownerSession)
  const page = new Resource(ownerSession, async signal => {
    const value = record(await session.request('/api/memory' + (personal ? '/mine' : '') + query,
      { permissions: [readGrant], signal, verifyIdentity: personal }))
    if (!value || !Array.isArray(value.facts)) throw new Error('Invalid memory page')
    return {
      facts: project(value.facts), proposals: personal ? [] : project(value.proposals),
      next: personal ? value.next : value.facts_next, nextProposals: value.proposals_next,
    }
  })

  onMount(() => {
    void page.load()
    const timer = setInterval(() => {
      now = Math.max(now, Date.now() / 1000)
      if (!page.data) return
      const { facts, proposals } = page.data
      const liveFacts = facts.filter(current)
      const liveProposals = proposals.filter(current)
      if (liveFacts.length !== facts.length || liveProposals.length !== proposals.length) {
        page.data = { ...page.data, facts: liveFacts, proposals: liveProposals }
      }
    }, 1000)
    const unregister = session.onPurge(() => { query = '' })
    return () => { clearInterval(timer); unregister(); page.destroy(); write.destroy() }
  })

  function navigate(key?: string, after?: unknown) {
    if (write.busy || page.busy) return
    query = key && typeof after === 'string' ? '?' + key + '=' + encodeURIComponent(after) : ''
    void page.load()
  }

  function project(value: unknown) {
    return rows(value).filter(current).map(({ id, kind, content, category, expires_at }) => ({
      id, kind, content: personal && kind === 'voice_proposal_pending' ? null : content, category, expires_at,
    }))
  }

  function current(fact: Record<string, unknown>) {
    return fact.expires_at === null || typeof fact.expires_at === 'number' && fact.expires_at > now
  }

  function change(fact: Record<string, unknown>, operation: 'delete' | 'return' | 'confirm' | 'reject') {
    if (!current(fact) || typeof fact.id !== 'string') return
    const id = encodeURIComponent(fact.id)
    const root = '/api/memory'
    if (operation === 'delete' && !window.confirm('¿Eliminar este recuerdo?')) return
    if (operation === 'return' && !window.confirm('¿Devolver este recuerdo a la administración? El contenido seguirá guardado hasta que se reasigne o se elimine.')) return
    const endpoint = personal ? root + '/mine/facts/' + id + (operation === 'return' ? '/return' : '')
      : operation === 'delete' ? root + '/facts/' + id : root + '/proposals/' + id + '/' + operation
    return write.run(endpoint, { permissions: [writeGrant], method: operation === 'delete' ? 'DELETE' : 'POST' },
      () => page.load())
  }
</script>

<section>
  <div class="heading-row">
    <h1>{personal ? 'Mi Memoria' : 'Memoria'}</h1>
    <button disabled={page.busy || write.busy} onclick={() => navigate()}>Actualizar</button>
  </div>
  {#if page.error}<p class="error" role="alert">{page.error}</p>{/if}
  {#if write.feedback}<p role="status">{write.feedback}</p>{/if}
  <div class="grid">
    {#each page.data?.facts.filter(current) ?? [] as fact}
      {@const proposal = personal && fact.kind === 'voice_proposal_pending'}
      <article class="card">
        <h3>{proposal ? 'Propuesta de voz' : text(fact.category, 'Recuerdo')}</h3>
        <p>{proposal ? 'Pendiente de revisión.' : text(fact.content)}</p>
        {#if session.can(writeGrant)}
          {#if personal}
            <button disabled={write.busy} onclick={() => change(fact, 'return')}>No es mío</button>
          {/if}
          <button class="danger" disabled={write.busy} onclick={() => change(fact, 'delete')}>{proposal ? 'Descartar' : 'Eliminar'}</button>
        {/if}
      </article>
    {:else}
      <p class="muted">{page.busy ? 'Consultando recuerdos…' : 'Sin recuerdos vigentes en esta página.'}</p>
    {/each}
  </div>
  {#if typeof page.data?.next === 'string'}
    <button disabled={page.busy || write.busy} onclick={() => navigate(personal ? 'after' : 'facts_after', page.data?.next)}>Más recuerdos</button>
  {/if}
  {#if !personal}
    <section>
      <h2>Propuestas</h2>
      <div class="grid">
        {#each page.data?.proposals.filter(current) ?? [] as proposal}
          <article class="card">
            <h3>{text(proposal.category, 'Propuesta')}</h3>
            <p>{text(proposal.content)}</p>
            {#if session.can(writeGrant)}
              <button disabled={write.busy} onclick={() => change(proposal, 'confirm')}>Conservar</button>
              <button disabled={write.busy} onclick={() => change(proposal, 'reject')}>Rechazar</button>
            {/if}
          </article>
        {:else}<p class="muted">Sin propuestas vigentes en esta página.</p>{/each}
      </div>
      {#if typeof page.data?.nextProposals === 'string'}
        <button disabled={page.busy || write.busy} onclick={() => navigate('proposals_after', page.data?.nextProposals)}>Más propuestas</button>
      {/if}
    </section>
  {/if}
</section>
