const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag,doc) { this.tagName=tag;this.ownerDocument=doc;this.children=[];this.dataset={};this.attributes={};this.listeners={};this.value=''; }
  append(...children) { for (const child of children) child.parentElement=this; this.children.push(...children); }
  replaceChildren(...children) { this.children=[]; this.append(...children); }
  setAttribute(name,value) { this.attributes[name]=value; }
  addEventListener(name,callback) { this.listeners[name]=callback; }
  removeEventListener() {}
  scrollIntoView(options) { this.scrollOptions=options; }
}
function descendants(node) { return [node,...node.children.flatMap(descendants)]; }
const groupA='11111111-1111-4111-8111-111111111111', groupB='22222222-2222-4222-8222-222222222222';
function capture(id,created,group) {
  return {id:'vf_'+id.repeat(64),created_at:created,group_id:group,group_state:'suggested',
    group_reason:'pairwise_similarity',group_similarity:.95,group_size:2,review_available:true,
    duration_s:3,suggested_profile:null,room_id:id==='a' ? null : 'Habitación sintética'};
}
function fixture(rows, admin = false) {
  const doc={createElement(tag){return new Element(tag,this);},addEventListener(){},removeEventListener(){}};
  const root={addEventListener(){},removeEventListener(){},
    KzaVoicePrivacy:require('../../src/dashboard/frontend/obs/src/voice-privacy.js')};
  const store=values=>{
    const subscribers=new Set();let status='real';
    return {snapshot:()=>values,status:()=>status,subscribe:callback=>{subscribers.add(callback);return ()=>subscribers.delete(callback);},
      update(next,state){values=next;status=state;for(const callback of subscribers)callback();},
      committed(){},deliveryToken:()=>null,check(){},detachMedia(){},attachMedia(media){this.media=media;}};
  };
  const state={captures:store(rows),profiles:store([]),review:store([])};
  const container=new Element('main',doc);
  for (const script of ['voice-view','voice-filters']) {
    vm.runInNewContext(fs.readFileSync(require.resolve('../../src/dashboard/frontend/obs/src/'+script+'.js'),'utf8'),
      {window:root,crypto:{randomUUID:()=>groupA}});
  }
  const requested=[], transcripts=[], filters=[], subscribers=new Set();
  let filterState={room:null,window:null,group_state:null,association:null,listenable_only:false,text:null};
  let roomOptions=['Habitación sintética'];
  function setFilters(changes) {
    filters.push(changes); filterState={...filterState,...changes};
    for (const draw of subscribers) draw();
  }
  function setRoomOptions(rooms) {
    roomOptions=rooms;
    for (const draw of subscribers) draw();
  }
  const stop=root.KzaVoiceView.mountVoices(container,{state,refresh(){},hasNext:()=>false,leave(){},
    searchAllowed:()=>admin,captureFilters:()=>filterState,captureSummary:()=>({rooms:roomOptions,count:rows.length}),
    setFilters,
    subscribeFilters(draw){subscribers.add(draw);return ()=>subscribers.delete(draw);},
    review(id){requested.push(id);state.review.update([],'loading');return true;},
    transcript(id){transcripts.push(id);state.review.update([],'loading');return true;}});
  return {container,stop,state,requested,transcripts,filters,setFilters,setRoomOptions};
}

test('capture rows remain newest first across interleaved groups and tied timestamps',()=>{
  const rows=[capture('a',1,groupA),capture('b',3,groupA),capture('d',2,groupB),capture('c',2,groupA)];
  const f=fixture(rows),original=rows.map(row=>row.id);
  const displayed=descendants(f.container).filter(node=>node.dataset.captureId);
  assert.deepEqual(displayed.map(node=>node.dataset.captureId),[rows[1].id,rows[2].id,rows[3].id,rows[0].id]);
  assert.ok(descendants(displayed[0]).some(node=>node.textContent==='Habitación sintética'));
  assert.ok(descendants(displayed[3]).some(node=>node.textContent==='Sin habitación registrada'));
  assert.deepEqual(rows.map(row=>row.id),original);
  assert.equal(descendants(f.container).filter(node=>node.dataset.voiceGroupId).length,2);
  f.stop();
});

test('selecting a visible group still includes its chronologically separated captures',()=>{
  const rows=[capture('a',1,groupA),capture('b',3,groupA),capture('c',2,groupB)];
  const f=fixture(rows);
  const heading=descendants(f.container).find(node=>node.dataset.voiceGroupId===groupA);
  descendants(heading).find(node=>node.tagName==='button').listeners.click();
  const selected=descendants(f.container).filter(node=>node.dataset.captureId)
    .filter(node=>descendants(node).some(child=>child.type==='checkbox'&&child.checked)).map(node=>node.dataset.captureId);
  assert.deepEqual(selected,[rows[1].id,rows[0].id]);
  f.stop();
});

test('listening exposes an empty player immediately and preserves edits while loading',()=>{
  const rows=[capture('a',1,groupA),capture('b',2,groupA)],f=fixture(rows);
  const review=f.container.children.find(node=>node.attributes['aria-label']==='Escuchar muestra');
  const captures=f.container.children.find(node=>node.attributes['aria-label']==='Muestras de voz');
  assert.ok(f.container.children.indexOf(review)<f.container.children.indexOf(captures));
  const input=descendants(captures).find(node=>node.attributes['aria-label']==='Nombre del perfil');
  input.value='Draft name';
  const buttons=descendants(captures).filter(node=>node.tagName==='button'&&node.textContent==='Escuchar');
  buttons[0].listeners.click();
  assert.deepEqual(f.requested,[rows[1].id]);
  assert.equal(review.attributes['aria-busy'],'true');
  assert.equal(buttons[0].textContent,'Cargando audio…');
  assert.ok(buttons.every(button=>button.disabled));
  assert.equal(review.scrollOptions.block,'nearest');
  const placeholder=descendants(review).find(node=>node.tagName==='audio');
  assert.ok(placeholder.controls);
  assert.equal(placeholder.src,undefined);
  assert.equal(f.state.review.media,undefined);
  assert.ok(descendants(captures).includes(input));
  assert.equal(input.value,'Draft name');
  const row={...rows[1],variant:'audio',audio_url:'blob:synthetic-selected',clip_duration_s:3,transcript:'Synthetic phrase',
    audio_metadata:null,transcript_metadata:null};
  f.state.review.update([row],'real');
  assert.equal(review.attributes['aria-busy'],'false');
  assert.ok(buttons.every(button=>!button.disabled&&button.textContent==='Escuchar'));
  const player=f.state.review.media;
  assert.equal(player.src,row.audio_url);
  assert.equal(player.preload,'auto');
  assert.ok(descendants(review).some(node=>node.textContent==='No consta si el texto fue recortado.'));
  buttons[0].listeners.click();
  assert.equal(f.requested.length,1);
  assert.equal(f.state.review.media,player);
  f.state.review.update([row],'real');
  assert.equal(f.state.review.media,player);
  f.state.review.update([],'expired');
  assert.ok(!descendants(review).some(node=>node.tagName==='audio'));
  assert.ok(!descendants(review).some(node=>node.textContent==='No consta si el texto fue recortado.'));
  buttons[0].listeners.click();
  assert.equal(f.requested.length,2);
  f.stop();
});

test('reading text and loading audio are explicit actions with distinct render states',()=>{
  const rows=[capture('a',1,groupA),{...capture('b',2,groupA),review_available:false}],f=fixture(rows);
  const button=(id,text)=>descendants(descendants(f.container).find(node=>node.dataset.captureId===id))
    .find(node=>node.tagName==='button'&&node.textContent===text);
  assert.equal(button(rows[1].id,'Leer transcripción').disabled,true);
  button(rows[0].id,'Leer transcripción').listeners.click();
  assert.deepEqual(f.transcripts,[rows[0].id]); assert.deepEqual(f.requested,[]);
  assert.equal(descendants(f.container).some(node=>node.tagName==='audio'),false);
  f.state.review.update([{id:rows[0].id,variant:'text',transcript:'Synthetic text alone',transcript_metadata:null}],'real');
  assert.ok(descendants(f.container).some(node=>node.textContent==='Synthetic text alone'));
  button(rows[0].id,'Escuchar').listeners.click();
  assert.deepEqual(f.requested,[rows[0].id]);
  const row={...rows[0],variant:'audio',audio_url:'blob:synthetic',clip_duration_s:1,transcript:'Synthetic audio text',
    audio_metadata:null,transcript_metadata:null};
  f.state.review.update([row],'real');
  button(rows[0].id,'Leer transcripción').listeners.click();
  assert.deepEqual(f.transcripts,[rows[0].id]); assert.equal(f.state.review.media.src,row.audio_url);
  f.stop();
});

test('filter controls remain mounted and the admin search only sends explicit submissions',()=>{
  const row=capture('a',1,groupA), reviewer=fixture([row]), admin=fixture([row],true);
  assert.equal(descendants(reviewer.container).some(node=>node.type==='search'),false);
  const all=descendants(admin.container), group=all.find(node=>node.attributes['aria-label']==='Filtros de muestras');
  const search=descendants(group).find(node=>node.type==='search');
  const room=descendants(group).find(node=>node.tagName==='select');
  assert.equal(room.attributes.autocomplete,'off'); assert.equal(search.attributes.autocomplete,'off');
  assert.equal(search.attributes.name,undefined);
  search.value='synthetic phrase'; assert.deepEqual(admin.filters,[]);
  search.listeners.keydown({key:'k',ctrlKey:true}); assert.deepEqual(admin.filters,[]);
  descendants(group).find(node=>node.textContent==='Buscar').listeners.click();
  assert.equal(admin.filters.at(-1).text,'synthetic phrase');
  room.value='unrecorded'; room.listeners.change();
  assert.equal(admin.filters.at(-1).room.kind,'unrecorded');
  admin.state.captures.update([],'real');
  assert.ok(descendants(admin.container).includes(room));
  assert.ok(descendants(admin.container).some(node=>node.attributes['aria-live']==='polite'));
  reviewer.stop(); admin.stop();
});

test('the room selector keeps the active room until the controller clears the filter',t=>{
  const f=fixture([capture('b',1,groupA)]), id='Habitación sintética';
  t.after(f.stop);
  const room=descendants(f.container).find(node=>node.tagName==='select');
  const options=()=>room.children.map(node=>[node.value,node.textContent]);
  const selected=['room:'+id,id], all=['','Todas'], unrecorded=['unrecorded','Sin habitación registrada'];
  room.value=selected[0]; room.listeners.change();
  assert.deepEqual({...f.filters.at(-1).room},{kind:'room',id});
  assert.deepEqual(options(),[all,selected,unrecorded]);
  f.setRoomOptions([]);
  assert.equal(room.value,selected[0]);
  assert.deepEqual(options(),[all,selected,unrecorded]);
  f.setRoomOptions([id,'Otra habitación']);
  assert.equal(room.value,selected[0]);
  assert.deepEqual(options(),[all,selected,['room:Otra habitación','Otra habitación'],unrecorded]);
  f.setRoomOptions([]);
  f.setFilters({room:null});
  assert.equal(room.value,'');
  assert.deepEqual(options(),[all,unrecorded]);
});
