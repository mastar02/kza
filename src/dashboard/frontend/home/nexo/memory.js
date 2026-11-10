/* NEXO Memoria view: DOM only. Cards are keyed by fact id and rebuilt only when
 * their rendered data changes; expiry ticks never rebuild a focused card. */
(function (root) {
  'use strict';
  const CATEGORY = { personal: 'Personal', preference: 'Preferencia', pattern: 'Hábito', fact: 'Dato' };
  const ORIGIN = { ambient: 'Captura ambiental', voice: 'Comando de voz', interaction: 'Conversación' };
  const OP_LABEL = { confirm: 'conservar', reject: 'descartar', delete: 'eliminar' };
  const CLAMP = 280;

  function mountMemory({ container, confirmDialog, owner, icon }) {
    const doc = container.ownerDocument;
    const cards = new Map(), expanded = new Set();
    let active = true, lastAnnouncement = 0, confirmResolve = null;

    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }
    function button(text, key, callback, className = 'identity-secondary') {
      const element = node('button', className, text); element.type = 'button'; element.dataset.key = key;
      element.addEventListener('click', callback); return element;
    }
    function date(seconds) {
      return new Date(seconds * 1000).toLocaleString('es-UY', { timeZone: 'America/Montevideo', dateStyle: 'medium', timeStyle: 'short' });
    }
    // Untrusted ASR/LLM text: textContent only, isolated direction (F15).
    function content(item, clamp = true) {
      const long = item.content.length > CLAMP, open = !clamp || expanded.has(item.id);
      const text = node('p', 'memory-content', long && !open ? item.content.slice(0, CLAMP).trimEnd() + '…' : item.content);
      text.dir = 'auto'; return { text, long };
    }
    function stateLine(item, kept) {
      const kind = item.state === 'proposed' ? 'Propuesta por revisar' : item.state === 'confirmed' ? 'Confirmado · no vence' : 'Automático';
      const expiry = item.expires_at === null ? '' : ' · vence ' + date(item.expires_at);
      return (CATEGORY[item.category] || 'Otro') + ' · ' + kind + expiry + (kept ? ' · No aparece entre los 100 más recientes' : '');
    }
    function confirmedBy(state, item) {
      if (item.state !== 'confirmed' || !item.confirmed_at) return null;
      const who = item.confirmed_by && item.confirmed_by === state.subject ? 'desde esta cuenta' : 'desde otra cuenta de administración';
      return 'Confirmado ' + who + ' · ' + date(item.confirmed_at);
    }
    function focusKey() { const element = doc.activeElement; return element && container.contains(element) ? element.dataset?.key || null : null; }
    function focusByKey(key) {
      if (!key) return false;
      const target = [...container.querySelectorAll('[data-key]')].find(element => element.dataset.key === key);
      if (target && !target.disabled) { target.focus({ preventScroll: true }); return true; }
      return false;
    }

    // Static skeleton.
    const heading = node('div', 'identity-heading memory-heading');
    const intro = node('div');
    const title = node('h2', '', 'Recuerdos del hogar.'); title.id = 'memory-title';
    intro.append(node('p', 'eyebrow', 'RECUERDOS Y PRIVACIDAD'), title,
      // Pinned by tests/unit/memory/test_fact_reader_callers.py: no module outside src/memory reads facts for answers.
      node('p', '', 'KZA guarda automáticamente datos que deduce de lo que escucha en el ambiente; ese texto puede procesarse en la nube. Los conserva hasta que vencen (como máximo 30 días) o los eliminás. En esta versión KZA no usa estos recuerdos para responder. Las propuestas quedan acá hasta que las confirmes o descartes.'),
      node('p', '', 'La memoria es compartida por todo el hogar: las cuentas de administración ven y revisan los recuerdos de todos. Las cuentas del hogar no tienen acceso.'));
    const side = node('div', 'identity-heading-side memory-heading-side');
    // aria-disabled instead of disabled: a disabled focused button drops keyboard focus to <body>.
    const refresh = button('Actualizar', 'refresh', () => { if (refresh.getAttribute('aria-disabled') !== 'true') void owner.reload(); }); refresh.id = 'memory-refresh';
    side.append(refresh); heading.append(intro, side);
    const permission = node('p', 'memory-note memory-permission');
    const sessionNotice = node('p', 'memory-note memory-session-notice'); sessionNotice.setAttribute('role', 'status');
    const freshness = node('p', 'memory-note memory-freshness'); freshness.id = 'memory-freshness';
    const cleanup = node('p', 'memory-note memory-cleanup');
    const indexing = node('p', 'memory-note memory-indexing');
    const quarantine = node('p', 'memory-note');
    // Visible as well as announced: a resolved card may have left the list with its outcome.
    const live = node('p', 'memory-note memory-live'); live.id = 'memory-live'; live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
    const pendingTitle = node('h3', 'memory-section-title', 'Operaciones sin confirmar'); pendingTitle.id = 'memory-pending-title'; pendingTitle.tabIndex = -1;
    const pendingList = node('div', 'memory-grid'); pendingList.id = 'memory-pending';
    const proposalsTitle = node('h3', 'memory-section-title', 'Propuestas pendientes'); proposalsTitle.id = 'memory-proposals-title'; proposalsTitle.tabIndex = -1;
    const proposalsCap = node('p', 'memory-note memory-cap');
    const proposalsGrid = node('div', 'memory-grid'); proposalsGrid.id = 'memory-proposals';
    const proposalsEmpty = node('p', 'memory-note memory-empty', 'No hay propuestas pendientes.');
    const factsTitle = node('h3', 'memory-section-title', 'Recuerdos guardados'); factsTitle.id = 'memory-facts-title'; factsTitle.tabIndex = -1;
    const factsCap = node('p', 'memory-note memory-cap');
    const factsGrid = node('div', 'memory-grid'); factsGrid.id = 'memory-facts';
    const factsEmpty = node('p', 'memory-note memory-empty', 'No hay recuerdos guardados.');
    const sections = node('div', 'memory-sections');
    sections.append(pendingTitle, pendingList, proposalsTitle, proposalsCap, proposalsGrid, proposalsEmpty, factsTitle, factsCap, factsGrid, factsEmpty);
    container.replaceChildren(heading, permission, sessionNotice, freshness, cleanup, indexing, quarantine, live, sections);

    function markerText(marker) {
      if (!marker) return null;
      if (marker.state === 'sending') return 'Enviando…';
      return 'Sin confirmar: ' + OP_LABEL[marker.op];
    }
    function cardSignature(state, item, section) {
      return JSON.stringify([item, section, state.markers[item.id] || null, state.notices[item.id] || null, state.capabilities, state.subject,
        expanded.has(item.id), state.markers[item.id] ? null : owner.deleteBlocked(item.id)]);
    }
    function buildCard(state, item, section, card) {
      const marker = state.markers[item.id], caps = state.capabilities;
      const top = node('div', 'memory-card-top');
      const badge = node('span', 'scene-icon ' + (item.state === 'proposed' ? 'lavender' : 'mint')); badge.append(icon(item.state === 'proposed' ? 'spark' : 'layers'));
      const meta = node('p', 'memory-meta', stateLine(item, section === 'kept'));
      top.append(badge, meta); card.append(top);
      const { text, long } = content(item); card.append(text);
      if (long) {
        const open = expanded.has(item.id);
        const toggle = button(open ? 'Ver menos' : 'Ver completo', item.id + ':expand', () => { if (expanded.has(item.id)) expanded.delete(item.id); else expanded.add(item.id); render(); });
        toggle.setAttribute('aria-expanded', String(open)); card.append(toggle);
      }
      const by = confirmedBy(state, item);
      if (by) card.append(node('p', 'memory-detail', by));
      if (item.sources.some(source => source.depends_on.length)) card.append(node('p', 'memory-detail', 'Derivado de otro recuerdo.'));
      if (item.sources.length) {
        const details = node('details', 'memory-sources'); const summary = node('summary', '', 'Ver procedencia'); summary.dataset.key = item.id + ':sources';
        details.append(summary);
        for (const source of item.sources) details.append(node('p', 'memory-detail', (ORIGIN[source.origin] || 'Origen desconocido') + ' · ' + date(source.captured_at)));
        card.append(details);
      }
      const status = markerText(marker);
      if (status) { const line = node('p', 'memory-state', status); line.dataset.memoryMarker = marker.state; card.append(line); }
      const message = node('p', 'memory-notice', state.notices[item.id] || ''); message.dataset.memoryNotice = item.id; card.append(message);
      const controls = node('div', 'memory-controls');
      if (caps.review) {
        if (marker) {
          const again = button('Reintentar ' + OP_LABEL[marker.op], item.id + ':retry', () => owner.retry(item.id));
          again.disabled = marker.state === 'sending'; again.dataset.memoryWrite = 'retry'; controls.append(again);
        } else if (item.state === 'proposed') {
          const keep = button('Conservar sin vencimiento', item.id + ':confirm', () => owner.confirm(item.id), 'identity-primary'); keep.dataset.memoryWrite = 'confirm';
          const drop = button('Descartar propuesta', item.id + ':reject', () => owner.reject(item.id)); drop.dataset.memoryWrite = 'reject';
          controls.append(keep, drop);
        } else {
          const removal = button('Eliminar…', item.id + ':delete', () => confirmDelete(item.id), 'identity-danger'); removal.dataset.memoryWrite = 'delete';
          removal.setAttribute('aria-label', 'Eliminar este recuerdo');
          removal.disabled = owner.deleteBlocked(item.id);
          controls.append(removal);
        }
      }
      if (controls.childElementCount) card.append(controls);
    }
    // Keyed placement: a card that disappears hands focus to the card now at its position, else the heading.
    function place(grid, entries, state, heading) {
      const key = focusKey(), seen = new Set();
      let previous = null, lostFocus = null;
      for (const [item, section] of entries) {
        seen.add(item.id);
        let entry = cards.get(item.id);
        if (entry && entry.grid !== grid) { if (entry.card.contains(doc.activeElement)) lostFocus = lostFocus || { index: 0, key }; entry.card.remove(); cards.delete(item.id); entry = null; }
        if (!entry) {
          const card = node('article', 'memory-card panel'); card.dataset.memoryId = item.id; card.tabIndex = -1;
          entry = { card, signature: null, grid }; cards.set(item.id, entry);
        }
        const signature = cardSignature(state, item, section);
        if (entry.signature !== signature) {
          const hadFocus = entry.card.contains(doc.activeElement);
          entry.card.replaceChildren(); entry.card.setAttribute('aria-label', (item.state === 'proposed' ? 'Propuesta' : 'Recuerdo') + ' ' + (CATEGORY[item.category] || ''));
          buildCard(state, item, section, entry.card); entry.signature = signature;
          // A control disabled while its write is sent keeps focus on its card, never on the page body.
          if (hadFocus && !focusByKey(key)) (entry.card.querySelector('button:not([disabled]),summary') || entry.card).focus({ preventScroll: true });
        }
        const expected = previous ? previous.nextSibling : grid.firstChild;
        if (expected !== entry.card) grid.insertBefore(entry.card, expected);
        previous = entry.card;
      }
      [...grid.children].forEach((card, index) => {
        const id = card.dataset.memoryId;
        if (seen.has(id)) return;
        if (card.contains(doc.activeElement)) lostFocus = { index, key };
        card.remove(); if (cards.get(id)?.grid === grid) cards.delete(id);
      });
      if (lostFocus) return () => {
        // The moved card (a confirmation) keeps focus on its own heading when it is listed elsewhere.
        const target = grid.children[Math.min(lostFocus.index, grid.children.length - 1)];
        if (target) (target.querySelector('button:not([disabled]),summary') || target).focus({ preventScroll: true });
        else heading.focus({ preventScroll: true });
      };
      return null;
    }

    async function confirmDelete(id) {
      const state = owner.getState();
      const item = [...state.facts, ...state.kept].find(entry => entry.id === id);
      if (!item) return;
      let others = owner.dependents(id), changed = false;
      // The cascade accepted must be the cascade sent: re-ask when a read changed it meanwhile (N5).
      for (let round = 0; round < 5; round++) {
        const accepted = await ask(item, others, state.factsFull || state.proposalsFull, id + ':delete', changed);
        if (!accepted || !active) return;
        const now = owner.getState();
        if (![...now.facts, ...now.kept].some(entry => entry.id === id)) return;
        const current = owner.dependents(id);
        const same = current.length === others.length && current.every(entry => others.some(other => other.id === entry.id));
        if (same) break;
        others = current; changed = true;
        if (round === 4) return;
      }
      {
        const result = await owner.remove(id, { confirmed: true });
        if (active && result?.applied) { const target = factsGrid.querySelector('button:not([disabled]),summary'); (target || factsTitle).focus({ preventScroll: true }); }
      }
    }
    function ask(item, others, full, invoker, changed = false) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        const titleNode = node('h2', '', 'Eliminar este recuerdo'); titleNode.id = 'memory-confirm-title';
        const target = node('div', 'memory-confirm-target');
        target.append(content(item, true).text, node('p', 'memory-detail', stateLine(item, false)));
        const body = node('p', 'modal-description', 'Es definitivo: no se puede deshacer.' + (others.length ? ' También se eliminan los recuerdos derivados de este:' : ' No hay recuerdos derivados visibles.'));
        const parts = [titleNode, target, body];
        if (changed) { const note = node('p', 'memory-detail memory-changed', 'La lista de recuerdos derivados cambió mientras decidías. Revisala antes de confirmar.'); note.setAttribute('role', 'alert'); parts.splice(1, 0, note); }
        if (others.length) {
          const list = node('ul', 'memory-confirm-items');
          for (const other of others) {
            const line = node('li');
            const text = node('span', 'memory-content', other.content.length > 120 ? other.content.slice(0, 120).trimEnd() + '…' : other.content); text.dir = 'auto';
            line.append(text, node('span', 'memory-detail', ' · ' + (CATEGORY[other.category] || 'Otro') + ' · ' + (other.state === 'confirmed' ? 'confirmado' : other.state === 'proposed' ? 'propuesta' : 'automático')));
            list.append(line);
          }
          parts.push(list);
        }
        if (full) parts.push(node('p', 'memory-detail', 'Puede haber otros que no se muestran.'));
        parts.push(node('p', 'memory-detail', 'Las copias en índices de búsqueda se limpian en segundo plano; si la limpieza queda pendiente se indica en esta vista. No borra la conversación de origen (vence sola en 48 horas como máximo) ni el texto que ya se haya enviado a la nube para deducirlo.'));
        const actions = node('div', 'modal-actions memory-confirm-actions');
        const cancelButton = button('Cancelar', 'confirm-cancel', () => finish(false));
        const accept = button('Eliminar definitivamente', 'confirm-accept', () => finish(true), 'identity-danger');
        actions.append(cancelButton, accept); parts.push(actions);
        confirmDialog.replaceChildren(...parts);
        confirmDialog.setAttribute('aria-labelledby', 'memory-confirm-title');
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          if (active && !focusByKey(invoker)) factsTitle.focus({ preventScroll: true });
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish;
        if (!confirmDialog.open) confirmDialog.showModal();
        cancelButton.focus();
      });
    }

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      refresh.setAttribute('aria-disabled', String(state.read.loading));
      permission.textContent = caps.review ? '' : 'Este acceso permite ver la memoria pero no revisarla.';
      permission.hidden = !permission.textContent;
      sessionNotice.textContent = state.notices.session || ''; sessionNotice.hidden = !sessionNotice.textContent;
      const time = state.read.observedWall ? new Date(state.read.observedWall).toLocaleTimeString('es-UY', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }) : null;
      freshness.textContent = state.read.error ? `${state.read.error}${time ? ' Última consulta confirmada a las ' + time + '.' : ''}${state.loaded ? ' Se muestran esos datos.' : ''}`
        : !state.loaded ? (state.read.loading ? 'Consultando la memoria…' : 'Sin consulta confirmada.') : 'Consultada a las ' + time + '.';
      freshness.classList.toggle('is-error', Boolean(state.read.error));
      cleanup.textContent = state.pendingCleanup > 0 ? 'Hay copias en índices de búsqueda pendientes de limpieza.' : '';
      cleanup.hidden = !cleanup.textContent;
      indexing.textContent = state.pendingIndex > 0 ? 'Actualizando el índice de búsqueda.' : ''; indexing.hidden = !indexing.textContent;
      quarantine.textContent = state.legacyPolicy === 'quarantined' ? 'Los registros anteriores a esta política están en cuarentena y no se muestran acá.' : '';
      quarantine.hidden = !quarantine.textContent;
      // Pending operations whose fact is not listed (after a reload, at the cap or once gone).
      const listed = new Set([...state.facts, ...state.proposals, ...state.kept].map(item => item.id));
      const orphan = Object.entries(state.markers).filter(([id]) => !listed.has(id));
      const orphanKeys = new Set(orphan.map(([id]) => 'orphan:' + id));
      for (const element of [...pendingList.children]) if (!orphanKeys.has(element.dataset.orphan)) element.remove();
      for (const [id, marker] of orphan) {
        const keyName = 'orphan:' + id, signature = JSON.stringify([marker, state.notices[id] || null, caps.review]);
        let card = [...pendingList.children].find(element => element.dataset.orphan === keyName);
        if (!card) { card = node('article', 'memory-card panel memory-orphan'); card.dataset.orphan = keyName; card.setAttribute('aria-label', 'Operación sin confirmar'); pendingList.append(card); }
        if (card.dataset.signature !== signature) {
          const hadFocus = card.contains(doc.activeElement);
          card.replaceChildren(node('p', 'memory-state', markerText(marker) + ' un recuerdo que no figura en la lista.'));
          const message = node('p', 'memory-notice', state.notices[id] || ''); message.dataset.memoryNotice = id; card.append(message);
          if (caps.review) { const again = button('Reintentar ' + OP_LABEL[marker.op], id + ':retry', () => owner.retry(id)); again.disabled = marker.state === 'sending'; card.append(again); }
          card.dataset.signature = signature;
          if (hadFocus) focusByKey(id + ':retry');
        }
      }
      pendingTitle.hidden = pendingList.hidden = !orphan.length || !state.loaded;
      const show = state.loaded;
      for (const element of [proposalsTitle, proposalsGrid, factsTitle, factsGrid]) element.hidden = !show;
      proposalsCap.textContent = show && state.proposalsFull ? 'Se muestran las 100 propuestas más recientes. Las anteriores no se pueden revisar desde acá; vencen solas.' : '';
      factsCap.textContent = show && state.factsFull ? 'Se muestran los 100 más recientes. Los anteriores, incluidos recuerdos confirmados que no vencen, no se pueden revisar ni eliminar desde acá.' : '';
      proposalsCap.hidden = !proposalsCap.textContent; factsCap.hidden = !factsCap.textContent;
      const restoreProposals = place(proposalsGrid, show ? state.proposals.map(item => [item, 'proposals']) : [], state, proposalsTitle);
      const moved = show ? [...state.kept.map(item => [item, 'kept']), ...state.facts.map(item => [item, 'facts'])] : [];
      const restoreFacts = place(factsGrid, moved, state, factsTitle);
      // A confirmation moves its card to Recuerdos guardados: focus follows it there (F1c).
      const movedCard = restoreProposals && lastConfirmed ? cards.get(lastConfirmed) : null;
      if (movedCard && movedCard.grid === factsGrid) { (movedCard.card.querySelector('button:not([disabled]),summary') || factsTitle).focus({ preventScroll: true }); }
      else restoreProposals?.();
      if (restoreProposals) lastConfirmed = null;
      restoreFacts?.();
      proposalsEmpty.hidden = !show || state.proposals.length > 0; factsEmpty.hidden = !show || state.facts.length + state.kept.length > 0;
      if (state.announcement && state.announcement.serial !== lastAnnouncement) { lastAnnouncement = state.announcement.serial; live.textContent = state.announcement.text; }
    }
    let lastConfirmed = null;
    container.addEventListener('click', event => {
      const key = event.target.closest?.('[data-key]')?.dataset.key || '';
      if (key.endsWith(':confirm')) lastConfirmed = key.slice(0, -':confirm'.length);
    }, true);
    const stop = owner.subscribe(render);
    if (owner.getState().announcement) lastAnnouncement = owner.getState().announcement.serial;
    render();
    // Private text lives in the dialog too: closing the view empties both (F8).
    return () => {
      active = false; stop();
      if (confirmResolve) confirmResolve(false);
      if (confirmDialog.open) confirmDialog.close();
      confirmDialog.replaceChildren(); container.replaceChildren(); cards.clear(); expanded.clear();
    };
  }
  const exported = { mountMemory };
  root.NexoMemory = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
