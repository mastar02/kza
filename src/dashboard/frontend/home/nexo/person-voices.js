/* A bounded, read-only voice page inside one canonical person's detail. */
(function (root) {
  'use strict';
  function mountPersonVoices({ container, controller, personId, current, icon, onManage }) {
    if (!controller.profilesOnly || controller.personId !== personId) throw Error('invalid person voice view');
    const doc = container.ownerDocument, store = controller.state.profiles;
    let active = true, busy = false, invalid = false;
    const valid = () => active && current() === true;
    function node(tag,text='',className='') { const element=doc.createElement(tag); element.textContent=text; element.className=className; return element; }
    function button(text,callback) { const element=node('button',text,'identity-secondary');element.type='button';element.addEventListener('click',callback);return element; }
    const heading=node('div','','person-linked-header'), count=node('span','—'), list=node('div','','person-linked-profiles');
    heading.append(node('h4','Perfiles vocales'),count);
    const connection=node('p','Preparando consulta privada…','person-voice-status'); connection.dataset.voiceConnection='';
    const status=node('p','','person-voice-status'), controls=node('div','','person-voice-controls');status.setAttribute('role','status');
    const refresh=button('Actualizar perfiles',()=>request(false)), next=button('Más perfiles',()=>request(true));
    refresh.dataset.voiceRefresh='';next.dataset.voiceNext='';controls.append(refresh,next);
    const manage=button('Gestionar en Voces',()=>{if(valid())onManage();});manage.dataset.voiceManage='';
    container.replaceChildren(heading,connection,status,list,controls,manage);
    async function request(continuation) {
      if(!valid()||busy)return;busy=true;invalid=false;draw();
      try { if(continuation)controller.next('profiles');else await controller.refresh();await controller.settled(); }
      finally { busy=false;if(valid())draw(); }
    }
    function draw() {
      if(!valid())return;
      const rows=store.snapshot();
      if(rows.some(row=>row.person_link?.user_id!==personId)){invalid=true;store.clear();return;}
      const real=store.status()==='real'&&!invalid;
      list.replaceChildren();count.textContent=real?`${rows.length} en esta página`:'—';
      const messages={waiting:'Esperando consulta privada…',loading:'Consultando perfiles…',expired:'La consulta venció. Actualizá para ver los vínculos vigentes.',unavailable:'No se pudieron consultar los perfiles. Volvé a actualizar.',closed:'La vista se cerró.'};
      status.textContent=invalid?messages.unavailable:real?'Consulta de perfiles confirmados.':messages[store.status()]||messages.unavailable;
      if(real&&!rows.length)list.append(node('p','Sin perfiles en esta consulta.','person-voice-empty'));
      for(const row of rows){const item=node('div','','person-linked-profile');item.dataset.personVoice=row.id;item.append(icon('mic'),node('span',row.name));list.append(item);}
      next.hidden=!real||!controller.hasNext('profiles');refresh.disabled=busy||['waiting','loading'].includes(store.status());next.disabled=busy||!real;
      store.committed(store.deliveryToken());
    }
    const stop=store.subscribe(draw);draw();
    return ()=>{active=false;stop();container.replaceChildren();store.committed(null);};
  }
  const api={mountPersonVoices};root.NexoPersonVoices=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
