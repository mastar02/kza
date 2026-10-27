const assert=require('node:assert/strict');
const {test}=require('node:test');
const {A,B,C,review,profile,Media,fixture,flush}=require('../mocks/voice_browser.cjs');

test('voice preview becomes one bounded Blob URL and never caches its base64 response',async()=>{
  const f=fixture(),input=review();
  assert(f.store.receive([input],{mono:0,wall:2000000000000}));
  input.transcript='Changed caller value';
  assert.equal(f.store.snapshot()[0].transcript,'Synthetic private voice phrase');
  assert.equal(f.urls.size,1);assert.equal('audio_base64' in f.store.snapshot()[0],false);
  f.store.clear();assert.equal(f.urls.size,0);assert.equal(f.revoked.length,1);
});
test('native media retirement waits for emptied even after pause, detach and URL revocation',async()=>{
  const f=fixture();f.store.receive([review()],{mono:0,wall:2000000000000});
  const media=new Media();f.store.attachMedia(media);f.store.clear();
  let retired=false;const waiting=f.store.retired().then(()=>{retired=true;});await flush();
  assert.equal(f.urls.size,0);assert.equal(media.paused,true);assert.equal(media.isConnected,false);
  assert.equal(retired,false);assert.equal(f.store.pendingRetirements(),1);
  media.emptied();await waiting;assert.equal(f.store.pendingRetirements(),0);
});
test('an unloaded media element needs no fabricated emptied event',async()=>{
  const f=fixture(),media=new Media(false);f.store.attachMedia(media);f.store.clear();
  await f.store.retired();assert.equal(f.store.pendingRetirements(),0);
});
test('request and decode duration consume the original raw deadline',async()=>{
  const f=fixture();f.advance(10000);f.store.receive([review()],{mono:0,wall:2000000000000});
  f.advance(49999);f.store.check();assert.equal(f.store.snapshot().length,1);
  f.advance(1);f.store.check();assert.equal(f.store.snapshot().length,0);assert.equal(f.urls.size,0);
});
test('a backwards wall clock cannot extend the monotonic listening lifetime',async()=>{
  const f=fixture();f.store.receive([review()],{mono:0,wall:2000000000000});f.advance(60000,-30000);f.store.check();
  assert.equal(f.store.snapshot().length,0);assert.equal(f.urls.size,0);
});
test('confirmed labels use a bounded view lease while their server profile is permanent',async()=>{
  const f=fixture('profiles');assert(f.store.receive([profile()],{mono:0,wall:2000000000000}));
  f.advance(300000);f.store.check();assert.deepEqual(f.store.snapshot(),[]);
});
for(const damage of ['oversize','invalid_wav','long_clip','long_text','expired'])test('invalid listening input is rejected without retaining URLs: '+damage,async()=>{
  const f=fixture(),row=review();
  if(damage==='oversize')row.audio_base64='A'.repeat(256064);
  if(damage==='invalid_wav')row.audio_base64=Buffer.from('bad audio data').toString('base64');
  if(damage==='long_clip')row.clip_duration_s=7;
  if(damage==='long_text')row.transcript='x'.repeat(8193);
  if(damage==='expired')row.expires_at=row.observed_at;
  assert.equal(f.store.receive([row],{mono:0,wall:2000000000000}),false);assert.equal(f.urls.size,0);
});
test('another REST channel cannot retire or acknowledge a voice frame',async()=>{
  const f=fixture();f.channel.open(A);await f.response();await f.body();
  assert.equal(f.channel.retire({type:'rest_retire',channel:'conversations',payload:{event_ids:[B],receipts:[{event_id:B,ticket:B,nonce:C}]}}),false);
  await flush();assert.equal(f.store.snapshot().length,1);assert.equal(f.acks.length,0);
  assert.equal(f.channel.retire({type:'rest_retire',payload:{event_ids:[B],receipts:[{event_id:B,ticket:B,nonce:C}]}}),false);
  f.channel.dispose();
});
test('voice ACK waits for explicit DOM commit and the actual native emptied callback',async()=>{
  const f=fixture('review',true);f.channel.open(A);await f.response();await f.body();f.store.committed(B);
  const media=new Media();f.store.attachMedia(media);f.retire();f.store.committed(null);await flush();
  assert.equal(f.store.snapshot().length,0);assert.equal(f.acks.length,0);assert.equal(f.urls.size,0);
  media.emptied();await f.store.retired();await flush();
  assert.deepEqual(f.acks,[{type:'live_ack',receipts:[{ticket:B,nonce:C}]}]);f.channel.dispose();
});
test('an unseen voice retirement waits for an actual response decoder despite fetch abort',async()=>{
  const f=fixture();f.channel.open(A);await f.response();f.retire();await flush();assert.equal(f.acks.length,0);
  await f.body();assert.equal(f.store.snapshot().length,0);assert.equal(f.acks[0].receipts[0].ticket,B);f.channel.dispose();
});
test('page reset cannot acknowledge until prior media teardown has really finished',async()=>{
  const f=fixture();f.channel.open(A);await f.response();await f.body();const media=new Media();f.store.attachMedia(media);
  let done=false;const resetting=f.channel.reset().then(value=>{done=value;});await flush();assert.equal(done,false);
  media.emptied();await resetting;assert.equal(done,true);assert.deepEqual(f.acks,[]);f.channel.dispose();
});

test('emptied before the final native buffer reset keeps the lease until actual state settles',{timeout:1000},async()=>{
  const f=fixture(),media=new Media();
  media.load=function(){this.networkState=0;this.readyState=0;};
  f.store.attachMedia(media);f.store.clear();media.emptied();await flush();
  assert.equal(f.store.pendingRetirements(),1);
  assert.equal(f.store.retirementStatus()[0].emptied,true);
  media.buffered.length=0;media.duration=NaN;
  await f.store.retired();assert.equal(f.store.pendingRetirements(),0);
});

test('a review mutation owns its actual response body after headers arrive',async()=>{
  const {createVoiceState}=require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
  const {createVoiceController}=require('../../src/dashboard/frontend/obs/src/voice-channel.js');
  const state=createVoiceState(),acks=[];let finishBody;
  const controller=createVoiceController({state,emitAck:value=>{acks.push(value);return true;},fetch:async(path,options)=>{
    if(options.method)return{ok:true,status:200,arrayBuffer:()=>new Promise(resolve=>{finishBody=resolve;})};
    return{ok:true,status:200,json:async()=>({connection_id:A,snapshot:null,channel:path.endsWith('/profiles')?'voices:profiles':'voices:captures'})};
  }});
  controller.open(A);let done=false;
  const saving=controller.confirm(['vf_'+'a'.repeat(64),'vf_'+'b'.repeat(64),'vf_'+'c'.repeat(64)],'Synthetic draft name',B).then(value=>{done=true;return value;});
  await flush();assert.equal(done,false);assert.equal(typeof finishBody,'function');
  finishBody(new ArrayBuffer(0));assert.equal(await saving,true);controller.dispose();
});

test('another read is refused until the actual response decoder settles',async()=>{
  const f=fixture('review',true);f.channel.open(A);await f.response();
  assert.equal(f.channel.refresh(),false);assert.equal(f.requests.length,1);
  await f.body();await f.channel.settled();f.channel.dispose();
});
test('replacing audio waits for native retirement before fetching another body',async()=>{
  const f=fixture('review',true);f.channel.open(A);await f.response();await f.body();await f.channel.settled();
  const media=new Media();f.store.attachMedia(media);assert.equal(f.channel.refresh(),true);await flush();
  assert.equal(f.requests.length,1);assert.equal(f.store.pendingRetirements(),1);
  for(let index=0;index<32;index++)assert.equal(f.channel.refresh(),false);
  assert.equal(f.requests.length,1);assert.equal(f.store.pendingRetirements(),1);
  media.emptied();await flush();assert.equal(f.requests.length,2);
  await f.response();await f.body();await f.channel.settled();f.channel.dispose();
});

function groupedCapture() {
  return {id:'vf_'+'b'.repeat(64),observed_at:2000000000,expires_at:2000000060,duration_s:3,created_at:2000000000,
    review_available:true,suggested_profile:null,similarity:null,group_id:A,group_revision:2,group_size:3,
    group_state:'suggested',group_reason:'pairwise_similarity',group_similarity:.95,group_expires_at:2000000010};
}
test('group metadata is detached and expires at its earliest contributor without renewing raw samples',async()=>{
  const f=fixture('captures'),input=groupedCapture();
  assert(f.store.receive([{...input,private_name:'Not a group field'}],{mono:0,wall:2000000000000}));
  const cached=f.store.snapshot()[0];assert.equal(cached.group_id,A);assert.equal(cached.group_revision,2);
  assert.equal(cached.private_name,undefined);assert(Object.isFrozen(cached));
  f.advance(10000);f.store.check();assert.deepEqual(f.store.snapshot(),[]);
});
for(const [field,value] of [['group_id','unknown'],['group_revision',0],['group_size',33],['group_state','verified_identity'],
  ['group_reason','verified'],['group_similarity',2],['group_expires_at',Infinity]]) {
  test('invalid group metadata is rejected before caching: '+field,async()=>{
    const f=fixture('captures');assert.equal(f.store.receive([{...groupedCapture(),[field]:value}],{mono:0,wall:2000000000000}),false);
    assert.deepEqual(f.store.snapshot(),[]);
  });
}
test('manual corrections carry only selected IDs and their current group revisions',async()=>{
  const {createVoiceState}=require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
  const {createVoiceController}=require('../../src/dashboard/frontend/obs/src/voice-channel.js');
  const state=createVoiceState({monotonic:()=>0,wall:()=>2000000000000,schedule:()=>1,cancel:()=>{}}),requests=[];
  const controller=createVoiceController({state,hasView:()=>false,emitAck:()=>true,
    fetch:async(path,options)=>{
      if(options.method){requests.push(JSON.parse(options.body));return{status:200,ok:true,arrayBuffer:async()=>new ArrayBuffer(0)};}
      return{status:200,ok:true,json:async()=>({connection_id:A,channel:'voices:profiles',snapshot:null})};
    }});
  controller.open(A);const row=groupedCapture();state.captures.receive([row],{mono:0,wall:2000000000000});
  assert(await controller.correctGroups([row.id],'split',B));
  assert.deepEqual(requests,[{fingerprint_ids:[row.id],expected_revisions:{[A]:2},action:'split',request_id:B}]);
  assert.equal(await controller.correctGroups([row.id],'unknown',C),false);assert.equal(requests.length,1);
  controller.dispose();
});
