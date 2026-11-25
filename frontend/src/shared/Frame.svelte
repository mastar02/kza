<script lang="ts">
  import type { Snippet } from 'svelte'
  import type { Role } from '../lib/session'
  let { brand, role, logout, children }: {
    brand: string; role: Role; logout: () => void; children: Snippet
  } = $props()
  const labels = { operator: 'Operador', home: 'Hogar', evaluator: 'Evaluador' }
</script>

<header class="app-header">
  <div class="wrap masthead">
    <span class="brand"><span class="brand-mark" aria-hidden="true"></span>{brand}</span>
    <div class="button-row">
      <span class="role-label">{labels[role]}</span>
      <button onclick={logout}>Cerrar sesión</button>
    </div>
  </div>
</header>
<main class="wrap app-body">{@render children()}</main>
