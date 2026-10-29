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
  const store=values=>({snapshot:()=>values,status:()=> 'real',subscribe:()=>()=>{},committed(){},deliveryToken:()=>null,check(){},detachMedia(){}});
  const state={captures:store(rows),profiles:store([]),review:store([])};
  const container=new Element('main',doc);
  vm.runInNewContext(fs.readFileSync(require.resolve('../../src/dashboard/frontend/obs/src/voice-view.js'),'utf8'),
    {window:root,crypto:{randomUUID:()=>groupA}});
  const stop=root.KzaVoiceView.mountVoices(container,{state,refresh(){},hasNext:()=>false,leave(){}});
  return {container,stop};
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
