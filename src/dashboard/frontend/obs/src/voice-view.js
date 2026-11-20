(function (root) {
  'use strict';
  function mountVoices(container,controller) {
    const state = controller.state, doc = container.ownerDocument, selected = new Set();
    const listeningButtons = new Map(), deleteButtons = new Set();
    let active = true, renderedReview = null, requestedReview = null, startingReview = false;
    let actionPending = false, updateCaptureActions = () => {};
    function node(tag,text='',className='') { const element = doc.createElement(tag); element.textContent = text; element.className = className; return element; }
    function button(text,action) { const value = node('button',text,'btn'); value.type = 'button'; value.addEventListener('click',action); return value; }
    function message(store,empty) {
      return ({ waiting:'Esperando conexión…', loading:'Consultando…', unavailable:'La consulta no está disponible. Podés volver a actualizar.',
        expired:'La vista venció. Actualizá para consultar los datos vigentes.', closed:'Sesión cerrada.',
        forbidden:'Sin permiso para consultar estas muestras.' })[store.status()] || empty;
    }
    const heading = node('header','','page-head');
    const refreshStatus = node('p','','mute');
    const refresh = button('Actualizar',() => runAction(refreshStatus,async () => {
      await controller.refresh(); await controller.settled();
      return ['profiles','captures'].every(key => state[key].status()==='real');
    },'No se pudieron actualizar las voces. Volvé a intentar.','Actualizando…'));
    heading.append(node('h1','Voces'),node('p','Revisá muestras locales antes de confirmar un nombre. El nombre no concede permisos.','mute'),refresh,refreshStatus);
    const profiles = node('section','','card'), captures = node('section','','card'), review = node('section','','card');
    const filterView = root.KzaVoiceFilters.mountVoiceFilters({doc,controller,button}), results = node('div');
    captures.append(node('h2','Muestras de voz'),
      node('p','Audio y transcripción vencen con la captura original, como máximo a las 48 horas.','mute'),
      node('p','Los grupos sugieren una misma voz. La similitud no verifica identidad. Las correcciones manuales conservan el vencimiento de sus fuentes.','mute'),
      node('p','Estas correcciones organizan las muestras; los perfiles guardados se revisan por separado.','mute'),
      node('p','Más recientes primero.','mute'),filterView.element,results);
    const profileStatus = node('p','','mute'), captureStatus = node('p','','mute');
    for (const status of [refreshStatus,profileStatus,captureStatus]) status.setAttribute('role','status');
    profiles.setAttribute('aria-label','Perfiles confirmados'); captures.setAttribute('aria-label','Muestras de voz'); review.setAttribute('aria-label','Escuchar muestra');
    container.replaceChildren(heading,review,profiles,captures);
    function reviewPending() { return requestedReview!==null || state.review.status()==='loading'; }
    function updateActions() {
      const loading = reviewPending();
      refresh.disabled = actionPending || loading;
      for (const control of deleteButtons) control.disabled = actionPending || loading;
      updateCaptureActions();
      for (const [id,{listen,read,available}] of listeningButtons) {
        listen.disabled = read.disabled = !available || loading || actionPending;
        listen.textContent = requestedReview?.id===id && requestedReview.variant==='audio' ? 'Cargando audio…' : 'Escuchar';
        read.textContent = requestedReview?.id===id && requestedReview.variant==='text' ? 'Cargando transcripción…' : 'Leer transcripción';
      }
    }
    function openReview(id,variant) {
      if (!active || doc.hidden || actionPending || reviewPending()) return;
      const [row] = state.review.snapshot();
      if (row?.id!==id || variant==='audio' && row.variant!=='audio') {
        requestedReview = {id,variant}; startingReview = true; updateActions();
        try { if (!(variant==='text' ? controller.transcript(id) : controller.review(id))) requestedReview = null; }
        finally { startingReview = false; updateActions(); }
      }
      review.scrollIntoView({block:'center',behavior:'auto'});
    }
    async function runAction(status, operation, failure, pending = 'Aplicando cambios…') {
      if (!active || actionPending || reviewPending()) return;
      actionPending = true; status.textContent = pending; updateActions();
      let accepted = false;
      try { accepted = await operation(); }
      catch { accepted = false; }
      finally {
        actionPending = false;
        if (active) { status.textContent = accepted ? '' : failure; updateActions(); }
      }
    }
    function drawProfiles() {
      const rows = state.profiles.snapshot(); deleteButtons.clear(); profiles.replaceChildren(node('h2','Perfiles confirmados'));
      profiles.append(node('p','Se conservan hasta que los borres. No habilitan acciones ni permisos.','mute'));
      if (!rows.length) profiles.append(node('p',message(state.profiles,'No hay perfiles confirmados.')));
      for (const row of rows) {
        const id = row.id, revision = row.revision, item = node('div','','row');
        const remove = button('Borrar perfil',() => runAction(profileStatus,() => controller.deleteProfile(id,revision),
          'No se pudo borrar el perfil. Actualizá antes de reintentar.'));
        deleteButtons.add(remove);
        item.append(node('strong',row.name),node('span',new Date(row.confirmed_at*1000).toLocaleString(),'mute'),remove); profiles.append(item);
      }
      if (controller.hasNext('profiles')) profiles.append(button('Más perfiles',() => controller.next('profiles')));
      profiles.append(profileStatus); updateActions();
      state.profiles.committed(state.profiles.deliveryToken());
    }
    function drawCaptures() {
      const rows = state.captures.snapshot().slice().sort((left,right) =>
        right.created_at-left.created_at || (left.id<right.id ? 1 : left.id>right.id ? -1 : 0));
      for (const id of selected) if (!rows.some(row => row.id===id)) selected.delete(id);
      listeningButtons.clear();
      results.replaceChildren();
      if (!rows.length) results.append(node('p',message(state.captures,'Ninguna muestra vigente coincide con estos filtros.')));
      const count = node('p'), input = doc.createElement('input');
      input.type = 'text'; input.maxLength = 64; input.setAttribute('aria-label','Nombre del perfil'); input.autocomplete = 'off';
      const confirm = button('Confirmar perfil',() => runAction(captureStatus,() => {
        const name = input.value.trim(); input.value = '';
        return controller.confirm([...selected],name,crypto.randomUUID());
      },'No se pudo confirmar. Las muestras pueden haber vencido; actualizá antes de reintentar.'));
      function correction(action) {
        return runAction(captureStatus,() => controller.correctGroups([...selected],action,crypto.randomUUID()),
          'No se pudo corregir el grupo. Actualizá para revisar sus muestras y reintentar.');
      }
      const merge = button('Unir grupos seleccionados',() => correction('merge'));
      const split = button('Separar muestras seleccionadas',() => correction('split'));
      const unknown = button('Dejar como desconocido',() => correction('unknown'));
      function update() {
        const chosen = rows.filter(row => selected.has(row.id)), groups = new Map(chosen.map(row => [row.group_id,row]));
        const locked = actionPending || reviewPending();
        count.textContent = `${selected.size} muestras seleccionadas · elegí entre 3 y 32 de la misma voz para confirmar un perfil.`;
        confirm.disabled = locked || selected.size<3 || !input.value.trim();
        merge.disabled = locked || groups.size<2 || [...groups.values()].reduce((total,row) => total+row.group_size,0)>32;
        split.disabled = locked || !selected.size || groups.size!==1 || selected.size>=[...groups.values()][0].group_size;
        unknown.disabled = locked || !selected.size;
      }
      input.addEventListener('input',update);
      const table = node('table','','k'), body = doc.createElement('tbody');
      const groups = new Map();
      for (const row of rows) {
        if (!groups.has(row.group_id)) groups.set(row.group_id,[]);
        groups.get(row.group_id).push(row);
      }
      const shownGroups = new Set();
      for (const row of rows) {
        const groupId = row.group_id, members = groups.get(groupId);
        if (!shownGroups.has(groupId)) {
          shownGroups.add(groupId);
          const group = members[0], title = doc.createElement('tr'), cell = doc.createElement('td');
          title.dataset.voiceGroupId = groupId; cell.colSpan = 6;
          const label = {suggested:'Grupo sugerido',manual:'Agrupación manual',unknown:'Desconocido'}[group.group_state];
          const reasons = {short_audio:'Audio corto',ambiguous:'Similitud ambigua',manual_unknown:'Separado manualmente',unreviewed:'Sin revisión',prior_group_expired:'Venció el grupo anterior'};
          cell.append(node('strong',`${label} · ${groupId}`),node('p',`${members.length} de ${group.group_size} muestras en esta página${group.group_similarity===null?'':` · similitud mínima ${group.group_similarity.toFixed(2)}`}${reasons[group.group_reason]?` · ${reasons[group.group_reason]}`:''}`,'mute'),
            button('Seleccionar grupo visible',() => {
              for (const member of members) if (member.review_available && selected.size<32) selected.add(member.id);
              drawCaptures();
            })); title.append(cell); body.append(title);
        }
        const id = row.id, tr = doc.createElement('tr'), choice = doc.createElement('input');
        tr.dataset.captureId = id;
        choice.type = 'checkbox'; choice.checked = selected.has(id); choice.disabled = !row.review_available;
        choice.setAttribute('aria-label',`Seleccionar muestra ${new Date(row.created_at*1000).toLocaleTimeString()}`);
        choice.addEventListener('change',() => { if (choice.checked && selected.size<32) selected.add(id); else { selected.delete(id); choice.checked = false; } update(); });
        const cell = doc.createElement('td'); cell.append(choice); tr.append(cell);
        tr.append(node('td',new Date(row.created_at*1000).toLocaleString()),node('td',`${row.duration_s.toFixed(1)} s`),
          node('td',row.room_id ?? 'Sin habitación registrada'));
        const listen = button('Escuchar',() => openReview(id,'audio'));
        const read = button('Leer transcripción',() => openReview(id,'text'));
        listeningButtons.set(id,{listen,read,available:row.review_available});
        const action = node('div','','voice-sample-actions'), actionCell = doc.createElement('td');
        action.append(listen,read); actionCell.append(action); tr.append(actionCell);
        tr.append(node('td',`Grupo ${groupId.slice(0,8)} · ${row.suggested_profile ? `Sugerencia · similitud ${row.similarity.toFixed(2)}` : 'Sin asociación revisada'}`));
        body.append(tr);
      }
      const scroll = node('div','','table-scroll');
      scroll.setAttribute('role','region');
      scroll.setAttribute('aria-label','Tabla de muestras de voz');
      scroll.tabIndex = 0;
      table.append(body); scroll.append(table); results.append(scroll,count,
        node('p','Unir incluye todas las muestras de los grupos elegidos, aunque algunas estén en otra página. Separar forma otro grupo con las muestras seleccionadas; dejar como desconocido las separa individualmente.','mute'),
        merge,split,unknown,input,confirm,captureStatus);
      updateCaptureActions = update; updateActions();
      if (controller.hasNext('captures')) results.append(button('Más muestras',() => controller.next('captures')));
      state.captures.committed(state.captures.deliveryToken());
    }
    function drawReview() {
      const [row] = state.review.snapshot();
      const loading = state.review.status()==='loading';
      if (!startingReview && (row || !loading)) requestedReview = null;
      review.setAttribute('aria-busy',String(loading));
      updateActions();
      if (row && row===renderedReview) {
        state.review.committed(state.review.deliveryToken());
        return;
      }
      renderedReview = row ?? null;
      state.review.detachMedia(); review.replaceChildren(node('h2','Escuchar muestra'));
      if (!row) {
        const readingText = requestedReview?.variant==='text';
        const status = node('p',loading ? readingText ? 'Cargando transcripción…' : 'Cargando audio…'
          : message(state.review,'Elegí una muestra para escucharla o leer su transcripción.'));
        status.setAttribute('role','status'); review.append(status);
        if (loading && !readingText) {
          const placeholder = doc.createElement('audio'); placeholder.controls = true;
          placeholder.setAttribute('aria-label','Preparando muestra de audio'); review.append(placeholder);
        }
      }
      else {
        const transcriptDetails = root.KzaVoicePrivacy.metadataDetails('transcript',row.transcript_metadata);
        review.append(node('p',row.room_id ? `Habitación: ${row.room_id}` : 'Sin habitación registrada'));
        if (row.variant==='audio') {
          const player = root.KzaVoicePrivacy.createSamplePlayer({doc,store:state.review,row,preload:'auto',buttonClass:'btn'});
          review.append(player,node('p',row.clip_truncated ? `Primeros ${row.clip_duration_s} segundos de un segmento de ${row.duration_s} segundos.`
            : `Fragmento de ${row.clip_duration_s} segundos.`,'mute'),
            ...root.KzaVoicePrivacy.metadataDetails('audio',row.audio_metadata).map(text => node('p',text,'mute')));
        }
        review.append(node('h3','Transcripción del segmento'),
          ...transcriptDetails.map(text => node('p',text,'mute')),
          node('p',row.transcript || 'Sin transcripción disponible.'));
        if (row.variant==='audio') review.append(node('p','El fragmento de audio y la transcripción pueden corresponder a ventanas y canales distintos.','mute'));
      }
      state.review.committed(state.review.deliveryToken());
    }
    const unsubscribe = [state.profiles.subscribe(drawProfiles),state.captures.subscribe(drawCaptures),state.review.subscribe(drawReview),filterView.stop];
    const check = () => { controller.checkFilters(); for (const key of ['profiles','captures','review']) state[key].check(); };
    const onVisibility = () => { if (doc.hidden) controller.closeReview(); else check(); };
    doc.addEventListener('visibilitychange',onVisibility); root.addEventListener('focus',check);
    drawProfiles(); drawCaptures(); drawReview(); controller.refresh();
    return () => {
      active = false; renderedReview = requestedReview = null; listeningButtons.clear(); deleteButtons.clear();
      for (const stop of unsubscribe) stop(); selected.clear(); controller.leave();
      doc.removeEventListener('visibilitychange',onVisibility); root.removeEventListener('focus',check);
      container.replaceChildren();
      for (const key of ['profiles','captures','review']) state[key].committed(null);
    };
  }
  root.KzaVoiceView = { mountVoices };
})(window);
