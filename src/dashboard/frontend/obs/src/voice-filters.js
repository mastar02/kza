(function (root) {
  'use strict';
  function mountVoiceFilters({ doc, controller, button }) {
    const element = doc.createElement('div'), status = doc.createElement('p');
    const day = 86400000;
    element.className = 'voice-filters';
    element.setAttribute('role','group');
    element.setAttribute('aria-label','Filtros de muestras');
    status.className = 'voice-filter-status';
    status.setAttribute('aria-live','polite');
    status.setAttribute('aria-atomic','true');
    function field(tag, text, type) {
      const label = doc.createElement('label'), input = doc.createElement(tag);
      label.textContent = text;
      if (type) input.type = type;
      input.setAttribute('autocomplete','off'); label.append(input); element.append(label);
      return input;
    }
    function options(control, entries) {
      control.replaceChildren(...entries.map(([value,text]) => {
        const option = doc.createElement('option');
        option.value = value; option.textContent = text;
        return option;
      }));
    }
    function select(text, entries) {
      const control = field('select',text);
      options(control,entries);
      return control;
    }
    const room = field('select','Habitación');
    const period = select('Período',[['','Todo el período vigente'],['1','Última hora'],['6','Últimas 6 horas'],
      ['today','Hoy'],['24','Últimas 24 horas'],['custom','Personalizado']]);
    const since = field('input','Desde','datetime-local'), until = field('input','Hasta','datetime-local');
    const group = select('Estado',[['','Todos'],['suggested','Grupo sugerido'],['manual','Agrupación manual'],['unknown','Desconocido']]);
    const association = select('Asociación',[['','Todas'],['suggested','Con sugerencia revisada'],['unknown','Sin asociación']]);
    const listenable = field('input','Solo con audio para escuchar','checkbox');
    const search = controller.searchAllowed() ? field('input','Buscar en transcripciones','search') : null;
    let validation = '', drawnRooms = null;
    function set(changes) {
      validation = '';
      controller.setFilters(changes); draw();
    }
    function localTime(time) {
      return new Date(time-new Date(time).getTimezoneOffset()*60000).toISOString().slice(0,16);
    }
    function applyPeriod() {
      const now = Date.now(), custom = period.value === 'custom';
      since.parentElement.hidden = until.parentElement.hidden = !custom;
      if (custom) {
        since.min = until.min = localTime(now-2*day); since.max = until.max = localTime(now);
        if (!since.value || !until.value) { validation = ''; draw(); return; }
        const start = new Date(since.value).getTime(), end = new Date(until.value).getTime();
        if (!since.validity.valid || !until.validity.valid || !(start < end && end <= now)) {
          validation = 'Elegí un intervalo válido de las últimas 48 horas.'; draw(); return;
        }
        set({window:{since:start/1000,until:end/1000}});
      } else {
        const start = period.value === 'today' ? new Date(now).setHours(0,0,0,0) : now-Number(period.value)*3600000;
        set({window:period.value && start < now ? {since:start/1000,until:now/1000} : null});
      }
    }
    function applySearch() {
      const value = search.value.trim(), normalized = value.normalize('NFKD').replace(/\p{M}/gu,'').trim().replace(/\s+/gu,' ');
      if (value && ([...normalized].length < 2 || [...normalized].length > 64 || normalized.split(' ').length > 8)) {
        validation = 'Ingresá entre 2 y 64 caracteres y hasta 8 palabras.'; draw(); return;
      }
      set({text:value || null});
    }
    if (search) {
      search.maxLength = 128; search.spellcheck = false;
      element.append(button('Buscar',applySearch));
      search.addEventListener('keydown',event => {
        if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); applySearch(); }
      });
    }
    const clear = button('Quitar filtros',() => controller.resetFilters());
    element.append(clear,status);
    room.addEventListener('change',() => set({room:room.value ? room.value === 'unrecorded'
      ? {kind:'unrecorded'} : {kind:'room',id:room.value.slice(5)} : null}));
    for (const control of [period,since,until]) control.addEventListener('change',applyPeriod);
    group.addEventListener('change',() => set({group_state:group.value || null}));
    association.addEventListener('change',() => set({association:association.value || null}));
    listenable.addEventListener('change',() => set({listenable_only:listenable.checked}));
    function draw(reset = false) {
      if (reset) {
        validation = ''; period.value = since.value = until.value = '';
        if (search) search.value = '';
      }
      const filters = controller.captureFilters(), summary = controller.captureSummary();
      const selectedRoom = filters.room?.kind === 'room' ? filters.room.id : null;
      const visibleRooms = selectedRoom !== null && !summary.rooms.includes(selectedRoom)
        ? [...summary.rooms,selectedRoom] : summary.rooms;
      const roomKey = JSON.stringify(visibleRooms);
      if (roomKey !== drawnRooms) {
        drawnRooms = roomKey;
        options(room,[['','Todas'],...visibleRooms.map(id => ['room:'+id,id]),['unrecorded','Sin habitación registrada']]);
      }
      room.value = filters.room?.kind === 'unrecorded' ? 'unrecorded'
        : selectedRoom !== null ? 'room:'+selectedRoom : '';
      group.value = filters.group_state ?? ''; association.value = filters.association ?? '';
      listenable.checked = filters.listenable_only;
      since.parentElement.hidden = until.parentElement.hidden = period.value !== 'custom';
      const total = summary.count === null ? '' : summary.count === 1 ? '1 muestra vigente.' : `${summary.count} muestras vigentes.`;
      status.textContent = [validation,summary.roomNotice,total].filter(Boolean).join(' ');
    }
    const stop = controller.subscribeFilters(draw);
    draw();
    return {element,stop};
  }
  root.KzaVoiceFilters = {mountVoiceFilters};
})(typeof window === 'undefined' ? globalThis : window);
