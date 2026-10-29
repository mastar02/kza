const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag,doc) { this.tagName=tag;this.ownerDocument=doc;this.children=[];this.dataset={};this.attributes={};this.listeners={};this.value=''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=[...children]; }
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
    duration_s:3,suggested_profile:null};
}
function fixture(rows) {
  const doc={createElement(tag){return new Element(tag,this);},addEventListener(){},removeEventListener(){}};
  const root={addEventListener(){},removeEventListener(){}};
  const store=values=>{
    const subscribers=new Set();let status='real';
    return {snapshot:()=>values,status:()=>status,subscribe:callback=>{subscribers.add(callback);return ()=>subscribers.delete(callback);},
      update(next,state){values=next;status=state;for(const callback of subscribers)callback();},
      committed(){},deliveryToken:()=>null,check(){},detachMedia(){},attachMedia(media){this.media=media;}};
  };
  const state={captures:store(rows),profiles:store([]),review:store([])};
  const container=new Element('main',doc);
  vm.runInNewContext(fs.readFileSync(require.resolve('../../src/dashboard/frontend/obs/src/voice-view.js'),'utf8'),
    {window:root,crypto:{randomUUID:()=>groupA}});
  const requested=[];
  const stop=root.KzaVoiceView.mountVoices(container,{state,refresh(){},hasNext:()=>false,leave(){},
    review(id){requested.push(id);state.review.update([],'loading');return true;}});
  return {container,stop,state,requested};
}

test('capture rows remain newest first across interleaved groups and tied timestamps',()=>{
  const rows=[capture('a',1,groupA),capture('b',3,groupA),capture('d',2,groupB),capture('c',2,groupA)];
  const f=fixture(rows),original=rows.map(row=>row.id);
  const displayed=descendants(f.container).filter(node=>node.dataset.captureId);
  assert.deepEqual(displayed.map(node=>node.dataset.captureId),[rows[1].id,rows[2].id,rows[3].id,rows[0].id]);
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
  const row={...rows[1],audio_url:'blob:synthetic-selected',clip_duration_s:3,transcript:'Synthetic phrase'};
  f.state.review.update([row],'real');
  assert.equal(review.attributes['aria-busy'],'false');
  assert.ok(buttons.every(button=>!button.disabled&&button.textContent==='Escuchar'));
  const player=f.state.review.media;
  assert.equal(player.src,row.audio_url);
  assert.equal(player.preload,'auto');
  buttons[0].listeners.click();
  assert.equal(f.requested.length,1);
  assert.equal(f.state.review.media,player);
  f.state.review.update([row],'real');
  assert.equal(f.state.review.media,player);
  f.state.review.update([],'expired');
  assert.ok(!descendants(review).some(node=>node.tagName==='audio'));
  buttons[0].listeners.click();
  assert.equal(f.requested.length,2);
  f.stop();
});
