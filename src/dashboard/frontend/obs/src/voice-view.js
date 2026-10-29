(function (root) {
  'use strict';
  function mountVoices(container,controller) {
    const state = controller.state, doc = container.ownerDocument, selected = new Set();
    let active = true, renderedReview = null;
    function node(tag,text='',className='') { const element = doc.createElement(tag); element.textContent = text; element.className = className; return element; }
    function button(text,action) { const value = node('button',text,'btn'); value.type = 'button'; value.addEventListener('click',action); return value; }
    function message(store,empty) {
      return ({ waiting:'Esperando conexión…', loading:'Consultando…', unavailable:'La consulta no está disponible. Podés volver a actualizar.',
        expired:'La vista venció. Actualizá para consultar los datos vigentes.', closed:'Sesión cerrada.' })[store.status()] || empty;
    }
    const heading = node('header','','page-head');
    heading.append(node('h1','Voces'),node('p','Revisá muestras locales antes de confirmar un nombre. El nombre no concede permisos.','mute'),
      button('Actualizar',() => controller.refresh()));
    const profiles = node('section','','card'), captures = node('section','','card'), review = node('section','','card');
    profiles.setAttribute('aria-label','Perfiles confirmados'); captures.setAttribute('aria-label','Muestras de voz'); review.setAttribute('aria-label','Escuchar muestra');
    container.replaceChildren(heading,profiles,captures,review);
    function drawProfiles() {
      const rows = state.profiles.snapshot(); profiles.replaceChildren(node('h2','Perfiles confirmados'));
      profiles.append(node('p','Se conservan hasta que los borres. No habilitan acciones ni permisos.','mute'));
      if (!rows.length) profiles.append(node('p',message(state.profiles,'No hay perfiles confirmados.')));
      for (const row of rows) {
        const id = row.id, revision = row.revision, item = node('div','','row');
        item.append(node('strong',row.name),node('span',new Date(row.confirmed_at*1000).toLocaleString(),'mute'),
          button('Borrar perfil',event => {
            event.currentTarget.disabled = true;
            controller.deleteProfile(id,revision).then(accepted => {
              if (!accepted && active) profiles.append(node('p','No se pudo borrar el perfil. Actualizá antes de reintentar.'));
            });
          })); profiles.append(item);
      }
      if (controller.hasNext('profiles')) profiles.append(button('Más perfiles',() => controller.next('profiles')));
      state.profiles.committed(state.profiles.deliveryToken());
    }
    function drawCaptures() {
      const rows = state.captures.snapshot().slice().sort((left,right) =>
        right.created_at-left.created_at || (left.id<right.id ? 1 : left.id>right.id ? -1 : 0));
      for (const id of selected) if (!rows.some(row => row.id===id)) selected.delete(id);
      captures.replaceChildren(node('h2','Muestras de voz'));
      captures.append(node('p','Audio y transcripción vencen con la captura original, como máximo a las 48 horas.','mute'));
      captures.append(node('p','Los grupos sugieren una misma voz. La similitud no verifica identidad. Las correcciones manuales conservan el vencimiento de sus fuentes.','mute'));
      captures.append(node('p','Estas correcciones organizan las muestras; los perfiles guardados se revisan por separado.','mute'));
      captures.append(node('p','Más recientes primero.','mute'));
      if (!rows.length) captures.append(node('p',message(state.captures,'No hay muestras vigentes disponibles.')));
      const count = node('p'), input = doc.createElement('input');
      input.type = 'text'; input.maxLength = 64; input.setAttribute('aria-label','Nombre del perfil'); input.autocomplete = 'off';
      const confirm = button('Confirmar perfil',async () => {
        const ids = [...selected]; let name = input.value.trim(); input.value = ''; confirm.disabled = true;
        const requestId = crypto.randomUUID();
        const sending = controller.confirm(ids,name,requestId); name = null;
        if (!await sending && active) captures.append(node('p','No se pudo confirmar. Las muestras pueden haber vencido; actualizá antes de reintentar.'));
      });
      function correction(action) {
        const sending = controller.correctGroups([...selected],action,crypto.randomUUID());
        for (const control of [merge,split,unknown]) control.disabled = true;
        sending.then(accepted => {
          if (!accepted && active) captures.append(node('p','No se pudo corregir el grupo. Actualizá para revisar sus muestras y reintentar.'));
        });
      }
      const merge = button('Unir grupos seleccionados',() => correction('merge'));
      const split = button('Separar muestras seleccionadas',() => correction('split'));
      const unknown = button('Dejar como desconocido',() => correction('unknown'));
      function update() {
        const chosen = rows.filter(row => selected.has(row.id)), groups = new Map(chosen.map(row => [row.group_id,row]));
        count.textContent = `${selected.size} muestras seleccionadas · elegí entre 3 y 32 de la misma voz para confirmar un perfil.`;
        confirm.disabled = selected.size<3 || !input.value.trim();
        merge.disabled = groups.size<2 || [...groups.values()].reduce((total,row) => total+row.group_size,0)>32;
        split.disabled = !selected.size || groups.size!==1 || selected.size>=[...groups.values()][0].group_size;
        unknown.disabled = !selected.size;
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
          title.dataset.voiceGroupId = groupId; cell.colSpan = 5;
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
        tr.append(node('td',new Date(row.created_at*1000).toLocaleString()),node('td',`${row.duration_s.toFixed(1)} s`));
        const listen = button('Escuchar',() => controller.review(id)); listen.disabled = !row.review_available;
        const action = doc.createElement('td'); action.append(listen); tr.append(action);
        tr.append(node('td',`Grupo ${groupId.slice(0,8)} · ${row.suggested_profile ? `Sugerencia · similitud ${row.similarity.toFixed(2)}` : 'Sin asociación revisada'}`));
        body.append(tr);
      }
      table.append(body); captures.append(table,count,
        node('p','Unir incluye todas las muestras de los grupos elegidos, aunque algunas estén en otra página. Separar forma otro grupo con las muestras seleccionadas; dejar como desconocido las separa individualmente.','mute'),
        merge,split,unknown,input,confirm); update();
      if (controller.hasNext('captures')) captures.append(button('Más muestras',() => controller.next('captures')));
      state.captures.committed(state.captures.deliveryToken());
    }
    function drawReview() {
      const [row] = state.review.snapshot();
      if (row && row.audio_url===renderedReview) {
        state.review.committed(state.review.deliveryToken());
        return;
      }
      renderedReview = row?.audio_url ?? null;
      state.review.detachMedia(); review.replaceChildren(node('h2','Escuchar muestra'));
      if (!row) review.append(node('p',message(state.review,'Elegí una muestra para escucharla y revisar su transcripción.')));
      else {
        const audio = doc.createElement('audio'); audio.controls = true; audio.preload = 'none'; audio.src = row.audio_url;
        review.append(node('p',row.room_id ? `Habitación: ${row.room_id}` : 'Habitación sin registrar'),audio,
          node('p',row.clip_truncated ? `Primeros ${row.clip_duration_s} segundos de un segmento de ${row.duration_s} segundos.` : `Fragmento de ${row.clip_duration_s} segundos.`, 'mute'),
          node('h3','Transcripción del segmento'),node('p',row.transcript || 'Sin transcripción disponible.'));
        state.review.attachMedia(audio);
      }
      state.review.committed(state.review.deliveryToken());
    }
    const unsubscribe = [state.profiles.subscribe(drawProfiles),state.captures.subscribe(drawCaptures),state.review.subscribe(drawReview)];
    const check = () => { for (const key of ['profiles','captures','review']) state[key].check(); };
    doc.addEventListener('visibilitychange',check); root.addEventListener('focus',check);
    drawProfiles(); drawCaptures(); drawReview(); controller.refresh();
    return () => {
      active = false; renderedReview = null;
      for (const stop of unsubscribe) stop(); selected.clear(); controller.leave();
      doc.removeEventListener('visibilitychange',check); root.removeEventListener('focus',check);
      container.replaceChildren();
      for (const key of ['profiles','captures','review']) state[key].committed(null);
    };
  }
  root.KzaVoiceView = { mountVoices };
})(window);
