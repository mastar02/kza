const assert=require('node:assert/strict');
const {test}=require('node:test');
const {A,B,C,review,profile,Media,fixture,flush}=require('../mocks/voice_browser.cjs');
const {createSamplePlayer,createVoiceState}=require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
const {createVoiceController}=require('../../src/dashboard/frontend/obs/src/voice-channel.js');

function samplePlayer() {
  const f=fixture(), audio=new Media();
  class Element extends EventTarget {
    append(...children) { this.children=children; }
    setAttribute() {}
  }
  const doc={hidden:false,createElement:tag=>tag==='audio'?audio:new Element()};
  f.store.receive([review()],{mono:0,wall:2000000000000});
  const player=createSamplePlayer({doc,store:f.store,row:f.store.snapshot()[0],preload:'metadata'});
  const [,button,status]=player.children, click=()=>button.dispatchEvent(new Event('click'));
  return { ...f,audio,button,status,click };
}

test('explicit play locks immediately, ignores readiness and confirms only the playing event',async()=>{
  const {audio,button,status,click,store}=samplePlayer();
  let calls=0,fail;
  audio.play=()=>{
    assert.equal(button.disabled,true);
    assert.equal(status.textContent,'Iniciando reproducción…');
    return ++calls===1 ? new Promise((_,reject)=>{fail=reject;}) : Promise.resolve();
  };
  assert.equal(button.textContent,'Reproducir');
  click();click();assert.equal(calls,1);
  audio.dispatchEvent(new Event('canplay'));
  audio.dispatchEvent(new Event('loadedmetadata'));
  assert.equal(status.textContent,'Iniciando reproducción…');
  fail();await flush();
  assert.equal(button.disabled,false);assert.match(status.textContent,/reintentar/);
  click();await flush();
  assert.equal(calls,2);assert.equal(status.textContent,'Iniciando reproducción…');
  audio.dispatchEvent(new Event('playing'));
  assert.equal(status.textContent,'Reproduciendo…');assert.equal(button.textContent,'Pausar');
  store.clear();audio.emptied();await store.retired();
});

test('expiry revokes the URL and invalidates a pending rejection and media events',async()=>{
  const {audio,status,click,store,advance,urls}=samplePlayer();
  let fail;
  audio.play=()=>new Promise((_,reject)=>{fail=reject;});
  click();advance(60000);store.check();
  assert.equal(urls.size,0);assert.equal(audio.isConnected,false);
  audio.dispatchEvent(new Event('playing'));fail();await flush();
  assert.equal(status.textContent,'Iniciando reproducción…');
  audio.emptied();await store.retired();
});

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
    review_available:true,room_id:'Synthetic room',suggested_profile:null,similarity:null,group_id:A,group_revision:2,group_size:3,
    group_state:'suggested',group_reason:'pairwise_similarity',group_similarity:.95,group_expires_at:2000000010};
}
test('group metadata is detached and expires at its earliest contributor without renewing raw samples',async()=>{
  const f=fixture('captures'),input=groupedCapture();
  assert(f.store.receive([{...input,private_name:'Not a group field'}],{mono:0,wall:2000000000000}));
  const cached=f.store.snapshot()[0];assert.equal(cached.group_id,A);assert.equal(cached.group_revision,2);
  assert.equal(cached.room_id,'Synthetic room');
  assert.equal(cached.expires_at,2000000010);
  assert.equal(cached.private_name,undefined);assert(Object.isFrozen(cached));
  f.advance(10000);f.store.check();assert.deepEqual(f.store.snapshot(),[]);
});
for(const [field,value] of [['group_id','unknown'],['group_revision',0],['group_size',33],['group_state','verified_identity'],
  ['group_reason','verified'],['group_similarity',2],['group_expires_at',Infinity],['room_id',undefined],['room_id','bad\nroom']]) {
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
      if(path.endsWith('/groups/corrections')){requests.push(JSON.parse(options.body));return{status:200,ok:true,arrayBuffer:async()=>new ArrayBuffer(0)};}
      return{status:200,ok:true,json:async()=>({connection_id:A,channel:path.endsWith('/profiles')?'voices:profiles':'voices:captures',snapshot:null})};
    }});
  controller.open(A);const row=groupedCapture();state.captures.receive([row],{mono:0,wall:2000000000000});
  assert(await controller.correctGroups([row.id],'split',B));
  assert.deepEqual(requests,[{fingerprint_ids:[row.id],expected_revisions:{[A]:2},action:'split',request_id:B}]);
  assert.equal(await controller.correctGroups([row.id],'unknown',C),false);assert.equal(requests.length,1);
  controller.dispose();
});

test('enriched profile links are required, copied and deeply immutable',()=>{
  const {createVoiceStore}=require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
  const options={includePerson:true,monotonic:()=>0,wall:()=>2000000000000,schedule:()=>1,cancel:()=>{}};
  const store=createVoiceStore('profiles',options), row={...profile(),person_link:{user_id:'null',revision:2}};
  assert(store.receive([row],{mono:0,wall:2000000000000}));row.person_link.user_id='changed';
  const link=store.snapshot()[0].person_link;assert.deepEqual(link,{user_id:'null',revision:2});assert(Object.isFrozen(link));Reflect.set(link,'user_id','other');assert.equal(link.user_id,'null');store.clear();
  assert.equal(store.receive([profile()],{mono:0,wall:2000000000000}),false);
  assert.equal(store.receive([{...profile(),person_link:{user_id:'null',revision:0}}],{mono:0,wall:2000000000000}),false);
  assert(store.receive([{...profile(),person_link:{user_id:null,revision:0}}],{mono:0,wall:2000000000000}));store.clear();
});

test('a person-scoped profile store rejects another target before caching names',()=>{
  const {createVoiceStore}=require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
  const store=createVoiceStore('profiles',{includePerson:true,personId:'person-A',monotonic:()=>0,wall:()=>2000000000000,schedule:()=>1,cancel:()=>{}});
  assert.equal(store.receive([{...profile(),person_link:{user_id:'person-B',revision:1}}],{mono:0,wall:2000000000000}),false);assert.deepEqual(store.snapshot(),[]);
  assert(store.receive([{...profile(),person_link:{user_id:'person-A',revision:1}}],{mono:0,wall:2000000000000}));assert.equal(store.snapshot()[0].person_link.user_id,'person-A');store.clear();
});

const QUERY='/api/voices/captures/query', NOW=2000000000, CURSOR='vc2_'+'a'.repeat(114);
const DEFAULTS={room:null,window:null,group_state:null,association:null,listenable_only:false,text:null,cursor:null};
const OTHER='vf_'+'b'.repeat(64);
function transcript(id=review().id) {
  const row=review();
  return {id,variant:'text',transcript:row.transcript,room_id:row.room_id,duration_s:row.duration_s,
    transcript_scope:row.transcript_scope,transcript_metadata:row.transcript_metadata};
}
function controllerFixture(makeController=createVoiceController, {wallSkew=0,onAck=()=>{}}={}) {
  let now=0, id=0, unauthorized=0;
  const calls=[], acks=[], revoked=[], timers=new Map();
  const clock={monotonic:()=>now,wall:()=>NOW*1000+now+wallSkew,
    schedule:(fn,ms)=>{timers.set(++id,{fn,at:now+ms});return id;},cancel:token=>timers.delete(token)};
  const state=createVoiceState({...clock,url:{createObjectURL:()=>`blob:synthetic-${++id}`,revokeObjectURL:token=>revoked.push(token)}});
  const controller=makeController({state,...clock,canSearch:()=>true,
    onUnauthorized:()=>unauthorized++,emitAck:ack=>{onAck(ack);acks.push(ack);return true;},
    fetch:(path,options)=>new Promise(resolve=>{
      const channel=path===QUERY ? 'voices:captures' : path.includes('/profiles') ? 'voices:profiles' : 'voices:review';
      calls.push({path,options,respond({rows=null,fields={},status=200,eventId=B,empty=null}={}) {
        const snapshot=rows===null ? null : {channel,event_id:eventId,observed_at:NOW,expires_at:NOW+60,receipt:{ticket:eventId,nonce:C},rows,...fields};
        resolve({ok:status===200,status,json:async()=>({connection_id:A,channel,snapshot,empty_page:empty,next_cursor:fields.next_cursor})});
      }});
    })});
  controller.open(A);
  const latest=()=>calls.at(-1), query=()=>JSON.parse(latest().options.body);
  const advance=ms=>{
    now+=ms;
    for (const [key,timer] of timers) if (timer.at<=now) {timers.delete(key);timer.fn();}
  };
  const retire=(channel,eventId=B)=>controller.retire({type:'rest_retire',channel,
    payload:{event_ids:[eventId],receipts:[{event_id:eventId,ticket:eventId,nonce:C}]}});
  return {controller,state,calls,latest,query,revoked,advance,retire,acks,unauthorized:()=>unauthorized};
}

test('text replacement drains native media, refuses overlapping reads, and can be followed by audio',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  assert(c.review(review().id));
  assert.equal(c.transcript(OTHER),false); assert.equal(f.latest().options.signal.aborted,false);
  f.latest().respond({rows:[review()]}); await c.settled();
  const url=state.review.snapshot()[0].audio_url, media=new Media();
  state.review.attachMedia(media);
  assert(c.transcript(OTHER)); await flush();
  assert.deepEqual(f.revoked,[url]); assert.equal(media.isConnected,false); assert.equal(f.calls.length,1);
  media.emptied(); await flush();
  assert.equal(f.latest().path,'/api/voices/captures/'+OTHER+'/transcript');
  f.latest().respond({rows:[transcript(OTHER)],eventId:C}); await c.settled();
  assert.deepEqual(state.review.snapshot().map(row=>[row.id,row.variant]),[[OTHER,'text']]);
  f.retire('voices:review',B); assert.equal(state.review.snapshot()[0].id,OTHER);
  assert(c.review(OTHER)); f.latest().respond({rows:[{...review(),id:OTHER}],eventId:A}); await c.settled();
  assert.equal(state.review.snapshot()[0].variant,'audio');
  assert(c.transcript(review().id)); f.latest().respond({status:404}); await c.settled();
  assert.deepEqual(state.review.snapshot(),[]); c.dispose();
});

test('the text row expires and is cleared only by a matching review retirement',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  assert(c.transcript(OTHER)); f.latest().respond({rows:[transcript(OTHER)]}); await c.settled();
  f.retire('voices:captures',B); f.retire('voices:review',C);
  assert.equal(state.review.snapshot()[0].id,OTHER);
  f.advance(60000); assert.deepEqual(state.review.snapshot(),[]);
  c.dispose();
  const g=controllerFixture();
  assert(g.controller.transcript(OTHER)); g.latest().respond({rows:[transcript(OTHER)]}); await g.controller.settled();
  g.retire('voices:review'); assert.deepEqual(g.state.review.snapshot(),[]); g.controller.dispose();
});

for (const variant of ['audio','text']) test(`a captures filter keeps open ${variant} while profiles are paging`,async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  const initial=c.refresh(); await flush();
  f.latest().respond({rows:[profile()],fields:{next_cursor:'profiles-cursor'}});
  await initial; f.latest().respond({rows:[groupedCapture()]}); await c.settled();
  assert(c[variant==='audio'?'review':'transcript'](review().id));
  f.latest().respond({rows:[variant==='audio'?review():transcript()]}); await c.settled();
  const opened=state.review.snapshot()[0], media=variant==='audio'?new Media():null;
  if (media) state.review.attachMedia(media);
  assert(c.next('profiles')); const profilePage=f.latest(), marker=f.calls.length;
  await c.setFilters({association:'unknown'});
  assert.deepEqual(f.calls.slice(marker).map(call=>call.path),[QUERY]);
  assert.equal(profilePage.options.signal.aborted,false);
  assert.deepEqual(f.query(),{...DEFAULTS,association:'unknown'});
  f.latest().respond(); profilePage.respond();
  await c.settled();
  assert.equal(state.review.snapshot()[0],opened);
  assert.deepEqual(f.revoked,[]); assert.equal(state.review.pendingRetirements(),0);
  if (media) assert.equal(media.isConnected,true);
  c.dispose();
  if (media) media.emptied();
  await state.retired();
});

test('a filter interrupts only captures paging and applies the latest body after settlement',async()=>{
  const f=controllerFixture(), {controller:c}=f;
  await c.setFilters({listenable_only:true});
  assert.deepEqual(f.query(),{...DEFAULTS,listenable_only:true});
  f.latest().respond({rows:[groupedCapture()],fields:{next_cursor:CURSOR}}); await c.settled();
  assert(c.next('captures')); assert.equal(f.query().cursor,CURSOR);
  const paging=f.latest(), first=c.setFilters({group_state:'manual'}), last=c.setFilters({group_state:'unknown'});
  assert.equal(paging.options.signal.aborted,true); assert.equal(c.next('captures'),false); assert.equal(f.calls.length,2);
  paging.respond({rows:[groupedCapture()]});
  assert.equal(await first,false); assert.equal(await last,true);
  assert.deepEqual(f.query(),{...DEFAULTS,listenable_only:true,group_state:'unknown'});
  assert.equal(f.calls.length,3);
  f.latest().respond({empty:{matching_count:0,scan_complete:true,count_observed_at:NOW}}); await c.settled();
  assert.equal(c.captureSummary().count,0); c.dispose();
});

test('a filter applied during a full refresh takes precedence over its initial captures read',async()=>{
  const f=controllerFixture(), {controller:c}=f;
  const initial=c.refresh(); await flush();
  const profiles=f.latest(); assert.equal(profiles.path,'/api/voices/profiles');
  await c.setFilters({text:'synthetic latest'});
  const active=f.latest(); assert.equal(active.path,QUERY);
  profiles.respond(); await initial;
  assert.equal(f.calls.length,2);
  assert.equal(active.options.signal.aborted,false);
  const changed=c.setFilters({text:'synthetic final'});
  assert.equal(active.options.signal.aborted,true);
  active.respond();
  assert.equal(await changed,true); assert.equal(f.calls.length,3);
  assert.deepEqual(f.query(),{...DEFAULTS,text:'synthetic final'});
  f.latest().respond(); await c.settled(); c.dispose();
});

test('resetting filter controls preserves a live unfiltered page and reloads a filtered page',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  const initial=c.refresh(); await flush();
  f.latest().respond(); await initial;
  const fields={next_cursor:CURSOR,matching_count:1,scan_complete:true,count_expires_at:NOW+8,
    room_options:['Synthetic room'],room_options_expires_at:NOW+10};
  f.latest().respond({rows:[groupedCapture()],fields}); await c.settled();
  const [unfiltered]=state.captures.snapshot(), notifications=[];
  c.subscribeFilters(reset=>notifications.push(reset));
  c.resetFilters(); await flush();
  assert.equal(f.calls.length,2);
  assert.equal(state.captures.snapshot()[0],unfiltered);
  assert.equal(state.captures.deliveryToken(),B);
  assert.equal(c.hasNext('captures'),true);
  assert.equal(c.captureSummary().count,1);
  assert.deepEqual(notifications,[true]);
  await c.setFilters({text:'synthetic phrase'});
  f.latest().respond({eventId:C,rows:[groupedCapture()]}); await c.settled();
  c.resetFilters();
  assert.deepEqual(state.captures.snapshot(),[]);
  assert.equal(c.captureFilters().text,null);
  await flush();
  assert.deepEqual(f.query(),DEFAULTS);
  f.latest().respond(); await c.settled(); c.dispose();
});

test('room options survive an empty filtered page and their original ACK waits for retirement',async()=>{
  const f=controllerFixture(), {controller:c}=f;
  await c.setFilters({listenable_only:true});
  f.latest().respond({rows:[groupedCapture()],fields:{matching_count:1,scan_complete:true,
    count_expires_at:NOW+5,room_options:['Synthetic room'],room_options_expires_at:NOW+10}});
  await c.settled();
  assert.deepEqual(c.captureSummary(),{count:1,rooms:['Synthetic room'],roomNotice:''});
  f.advance(5000); assert.equal(c.captureSummary().count,null);
  await c.setFilters({room:{kind:'room',id:'Synthetic room'}});
  f.latest().respond({empty:{matching_count:0,scan_complete:true,count_observed_at:NOW}}); await c.settled();
  assert.deepEqual(c.captureSummary(),{count:0,rooms:['Synthetic room'],roomNotice:''});
  assert.deepEqual(f.acks,[]);
  f.advance(5000); await flush();
  assert.deepEqual(c.captureSummary().rooms,[]); assert.deepEqual(f.query().room,{kind:'room',id:'Synthetic room'});
  f.latest().respond({empty:{matching_count:0,scan_complete:true,count_observed_at:NOW}}); await flush();
  assert.equal(f.query().room,null);
  assert.match(c.captureSummary().roomNotice,/ya no tiene muestras vigentes/);
  f.latest().respond(); await c.settled();
  assert.equal(f.acks.flatMap(ack=>ack.receipts).some(receipt=>receipt.ticket===B),true);
  c.dispose();
});

test('room options expire with their snapshot when its deadline is earlier',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  const room={kind:'room',id:'Synthetic room'};
  await c.setFilters({room});
  f.latest().respond({rows:[groupedCapture()],fields:{expires_at:NOW+4,matching_count:1,scan_complete:true,
    count_expires_at:NOW+8,room_options:['Synthetic room'],room_options_expires_at:NOW+10}});
  await c.settled();
  f.advance(3999);
  assert.deepEqual(c.captureSummary().rooms,['Synthetic room']);
  assert.equal(state.captures.snapshot().length,1);
  f.advance(1); await flush();
  assert.deepEqual(state.captures.snapshot(),[]);
  assert.deepEqual(c.captureSummary(),{count:null,rooms:[],roomNotice:''});
  assert.equal(f.calls.length,2);
  assert.deepEqual(f.query(),{...DEFAULTS,room});
  f.latest().respond(); await flush();
  assert.deepEqual(f.query(),DEFAULTS);
  f.latest().respond(); await c.settled(); c.dispose();
});

test('an earlier room-options deadline preserves the selected room and its live capture lease',async t=>{
  const f=controllerFixture(), {controller:c,state}=f;
  t.after(()=>c.dispose());
  const room={kind:'room',id:'Synthetic room'};
  const acknowledged=()=>f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket);
  await c.setFilters({room});
  f.latest().respond({rows:[groupedCapture()],fields:{expires_at:NOW+10,next_cursor:CURSOR,
    matching_count:1,scan_complete:true,count_expires_at:NOW+10,
    room_options:[room.id,'Other synthetic room'],room_options_expires_at:NOW+3}});
  await c.settled();
  const [capture]=state.captures.snapshot();
  f.advance(3000); await flush();
  assert.deepEqual(c.captureSummary(),{count:1,rooms:[],roomNotice:''});
  assert.deepEqual(c.captureFilters().room,room);
  assert.equal(state.captures.snapshot()[0],capture);
  assert.equal(state.captures.deliveryToken(),B);
  assert.equal(c.hasNext('captures'),true);
  assert.equal(f.calls.length,1);
  assert.deepEqual(acknowledged(),[]);
  f.advance(6999); await flush();
  assert.equal(state.captures.snapshot()[0],capture);
  assert.deepEqual(acknowledged(),[]);
  f.advance(1); await flush();
  assert.deepEqual(state.captures.snapshot(),[]);
  assert.equal(c.captureSummary().count,null);
  assert.deepEqual(acknowledged(),[B]);
});

for (const reason of ['expired in flight','omitted from the capped options']) test(`live filtered rows survive when the room is ${reason}`,async t=>{
  const f=controllerFixture(), {controller:c,state}=f;
  t.after(()=>c.dispose());
  const room={kind:'room',id:'Synthetic room'}, row=groupedCapture();
  await c.setFilters({room});
  if (reason==='expired in flight') f.advance(3000);
  const options=reason==='expired in flight' ? [room.id]
    : Array.from({length:32},(_,index)=>'Other synthetic room '+index);
  f.latest().respond({rows:[row],fields:{expires_at:NOW+10,matching_count:1,scan_complete:true,
    count_expires_at:NOW+10,room_options:options,room_options_expires_at:NOW+3}});
  await flush();
  assert.equal(f.calls.length,1);
  assert.deepEqual(state.captures.snapshot().map(item=>item.id),[row.id]);
  assert.deepEqual(c.captureFilters().room,room);
  assert.deepEqual(c.captureSummary(),{count:1,rooms:reason==='expired in flight'?[]:options,roomNotice:''});
  assert.equal(state.captures.deliveryToken(),B);
  assert.deepEqual(f.acks,[]);
});

for (const wallSkew of [0,-48*3600000]) {
  test(`a six-second response preserves captures past the options deadline, wall skew ${wallSkew} ms`,async t=>{
    const f=controllerFixture(createVoiceController,{wallSkew}), {controller:c,state}=f;
    t.after(()=>c.dispose());
    const room={kind:'room',id:'Synthetic room'}, row=groupedCapture();
    await c.setFilters({room});
    f.advance(6000);
    f.latest().respond({rows:[row],fields:{expires_at:NOW+10,matching_count:1,scan_complete:true,
      count_expires_at:NOW+10,room_options:[room.id],room_options_expires_at:NOW+3}});
    await c.settled();
    assert.equal(state.captures.status(),'real');
    assert.deepEqual(state.captures.snapshot().map(item=>item.id),[row.id]);
    assert.equal(state.captures.deliveryToken(),B);
    assert.deepEqual(c.captureFilters().room,room);
    assert.deepEqual(c.captureSummary(),{count:1,rooms:[],roomNotice:''});
    assert.equal(f.calls.length,1);
    f.advance(3999); await flush();
    assert.equal(state.captures.snapshot()[0].id,row.id);
    assert.deepEqual(f.acks,[]);
    f.advance(1); await flush();
    assert.deepEqual(state.captures.snapshot(),[]);
    assert.deepEqual(f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket),[B]);
  });

  test(`pagination with a newer observation charges only its own request, wall skew ${wallSkew} ms`,async t=>{
    const f=controllerFixture(createVoiceController,{wallSkew}), {controller:c,state}=f;
    t.after(()=>c.dispose());
    const first={...groupedCapture(),group_expires_at:NOW+40}, second={...groupedCapture(),id:review().id};
    const summary={matching_count:1,scan_complete:true,room_options:['Synthetic room']};
    await c.setFilters({listenable_only:true});
    f.advance(4000);
    f.latest().respond({rows:[first],fields:{...summary,observed_at:NOW+4,expires_at:NOW+40,
      next_cursor:CURSOR,count_expires_at:NOW+40,room_options_expires_at:NOW+40}});
    await c.settled();
    assert.equal(state.captures.snapshot()[0].id,first.id);
    assert.equal(state.captures.deliveryToken(),B);
    f.advance(2000);
    assert.equal(c.next('captures'),true);
    assert.equal(f.query().cursor,CURSOR);
    f.latest().respond({eventId:C,rows:[second],fields:{...summary,observed_at:NOW+6,expires_at:NOW+10,
      count_expires_at:NOW+10,room_options_expires_at:NOW+10}});
    await c.settled();
    assert.equal(state.captures.status(),'real');
    assert.deepEqual(state.captures.snapshot().map(item=>item.id),[second.id]);
    assert.equal(state.captures.deliveryToken(),C);
    assert.equal(c.captureSummary().count,1);
    f.advance(3999); await flush();
    assert.equal(state.captures.snapshot()[0].id,second.id);
    assert.deepEqual(f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket),[B]);
    f.advance(1); await flush();
    assert.deepEqual(state.captures.snapshot(),[]);
    assert.deepEqual(f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket),[B,C]);
  });

  test(`a replacement after a delayed response keeps the original deadline, wall skew ${wallSkew} ms`,async t=>{
    const f=controllerFixture(createVoiceController,{wallSkew}), {controller:c,state}=f;
    t.after(()=>c.dispose());
    const row=groupedCapture(), acknowledged=()=>f.acks.flatMap(ack=>ack.receipts.map(receipt=>receipt.ticket));
    const fields={observed_at:NOW,expires_at:NOW+10,matching_count:1,scan_complete:true,
      count_expires_at:NOW+10,room_options:['Synthetic room'],room_options_expires_at:NOW+10};
    await c.setFilters({listenable_only:true});
    f.advance(6000);
    f.latest().respond({rows:[row],fields}); await c.settled();
    assert.equal(state.captures.deliveryToken(),B);
    assert.deepEqual(acknowledged(),[]);
    f.advance(1000);
    await c.setFilters({association:'unknown'});
    f.latest().respond({eventId:C,rows:[row],fields}); await c.settled();
    assert.deepEqual(state.captures.snapshot().map(item=>item.id),[row.id]);
    assert.equal(state.captures.deliveryToken(),C);
    assert.deepEqual(acknowledged(),[B]);
    f.advance(2999); await flush();
    assert.equal(state.captures.snapshot()[0].id,row.id);
    assert.equal(c.captureSummary().count,1);
    assert.deepEqual(acknowledged(),[B]);
    f.advance(1); await flush();
    assert.deepEqual(state.captures.snapshot(),[]);
    assert.equal(state.captures.deliveryToken(),null);
    assert.deepEqual(c.captureSummary(),{count:null,rooms:[],roomNotice:''});
    assert.deepEqual(acknowledged(),[B,C]);
  });
}

for (const withdrawal of ['expiry','retire']) test(`OBS drops a selected room from the DOM before ACK on ${withdrawal}`,async t=>{
  const {readFileSync}=require('node:fs'), {runInNewContext}=require('node:vm');
  const {createDocument,Element}=require('../mocks/dom_shim.cjs');
  const window={};
  runInNewContext(readFileSync(require.resolve('../../src/dashboard/frontend/obs/src/voice-filters.js'),'utf8'),{window});
  const doc=createDocument();
  Object.defineProperty(doc,'createElement',{value:tag=>{
    const element=new Element(tag,doc);
    Object.defineProperty(element,'parentElement',{get:()=>element.parentNode});
    return element;
  }});
  const observed=[];
  let select;
  const f=controllerFixture(createVoiceController,{onAck(ack){
    for (const receipt of ack.receipts) observed.push({ticket:receipt.ticket,value:select.value,text:select.textContent});
  }});
  const c=f.controller, room={kind:'room',id:'Synthetic room'};
  const filters=window.KzaVoiceFilters.mountVoiceFilters({doc,controller:c,button:(text,click)=>{
    const button=doc.createElement('button'); button.textContent=text;
    button.addEventListener('click',click); return button;
  }});
  select=filters.element.querySelector('select');
  t.after(()=>{filters.stop();c.dispose();});
  await c.setFilters({room});
  f.latest().respond({rows:[groupedCapture()],fields:{expires_at:NOW+10,matching_count:1,scan_complete:true,
    count_expires_at:NOW+10,room_options:[room.id],room_options_expires_at:NOW+3}});
  await c.settled();
  f.advance(3000); await flush();
  assert.equal(select.value,'room:'+room.id);
  assert.ok(select.textContent.includes(room.id));
  assert.deepEqual(observed,[]);
  if (withdrawal==='expiry') f.advance(7000);
  else f.retire('voices:captures',B);
  assert.equal(select.value,'');
  assert.equal(select.textContent.includes(room.id),false);
  await flush();
  assert.deepEqual(observed,[{ticket:B,value:'',text:'TodasSin habitación registrada'}]);
  if (withdrawal==='retire') assert.equal(c.captureFilters().room,null);
});

test('room options can expire without unmounting a live unfiltered capture during review',async t=>{
  const f=controllerFixture(), {controller:c,state}=f;
  t.after(()=>c.dispose());
  const row=groupedCapture();
  await c.setFilters({listenable_only:true});
  f.latest().respond({rows:[row],fields:{matching_count:1,scan_complete:true,
    count_expires_at:NOW+8,room_options:['Synthetic room'],room_options_expires_at:NOW+3}});
  await c.settled();
  const [capture]=state.captures.snapshot();
  assert(c.review(row.id));
  const reading=f.latest();
  f.advance(3000); await flush();
  assert.deepEqual(c.captureSummary(),{count:1,rooms:[],roomNotice:''});
  assert.equal(state.captures.snapshot()[0],capture);
  assert.equal(state.captures.deliveryToken(),B);
  assert.equal(f.calls.length,2);
  assert.equal(reading.options.signal.aborted,false);
  reading.respond({eventId:C,rows:[{...review(),id:row.id}]}); await c.settled();
  const [opened]=state.review.snapshot();
  assert.equal(opened.id,row.id);
  f.advance(7000); await flush();
  assert.deepEqual(state.captures.snapshot(),[]);
  assert.equal(state.review.snapshot()[0],opened);
  assert.equal(f.calls.length,2);
  assert.deepEqual(f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket),[B]);
});

for (const withdrawal of ['expiry','retire']) test(`an older room summary's ${withdrawal} preserves the current page and review`,async t=>{
  const f=controllerFixture(), {controller:c,state}=f;
  t.after(()=>c.dispose());
  const room={kind:'room',id:'Synthetic room'}, row=groupedCapture();
  const summary={matching_count:1,scan_complete:true,count_expires_at:NOW+8};
  await c.setFilters({room});
  f.latest().respond({rows:[row],fields:{...summary,room_options:[room.id],room_options_expires_at:NOW+3}});
  await c.settled();
  await c.setFilters({text:'synthetic phrase'});
  f.latest().respond({eventId:C,rows:[row],fields:summary}); await c.settled();
  const [current]=state.captures.snapshot();
  assert.deepEqual(c.captureSummary().rooms,[room.id]);
  assert.equal(f.acks.length,0); // The old snapshot still owns the room option.
  assert(c.review(row.id));
  const reading=f.latest();
  if (withdrawal==='expiry') f.advance(3000);
  else f.retire('voices:captures',B);
  await flush();
  assert.deepEqual(c.captureSummary(),{count:1,rooms:[],roomNotice:''});
  assert.deepEqual(c.captureFilters().room,room);
  assert.equal(state.captures.snapshot()[0],current);
  assert.equal(state.captures.deliveryToken(),C);
  assert.equal(f.calls.length,3);
  assert.equal(reading.options.signal.aborted,false);
  assert.deepEqual(f.acks.flatMap(ack=>ack.receipts).map(receipt=>receipt.ticket),[B]);
  reading.respond({eventId:A,rows:[{...review(),id:row.id}]}); await c.settled();
  assert.equal(state.review.snapshot()[0].id,row.id);
});

test('a room refresh cannot renew an expired capture or hold its replacement ticket',async t=>{
  const f=controllerFixture(createVoiceController,{wallSkew:-48*3600000}), {controller:c,state}=f;
  t.after(()=>c.dispose());
  const short=NOW+3, row={...groupedCapture(),expires_at:short,group_expires_at:short};
  const fields={expires_at:short,matching_count:1,scan_complete:true,count_expires_at:short,
    room_options:['Synthetic room'],room_options_expires_at:short};
  const room={kind:'room',id:'Synthetic room'};
  const acknowledged=()=>f.acks.flatMap(ack=>ack.receipts.map(receipt=>receipt.ticket));
  await c.setFilters({room});
  f.latest().respond({rows:[row],fields}); await c.settled();
  assert(c.review(row.id));
  f.latest().respond({eventId:C,rows:[{...review(),id:row.id}],fields:{expires_at:short}});
  await c.settled();
  const media=new Media(); state.review.attachMedia(media);
  f.advance(2999); await flush();
  assert.equal(state.captures.deliveryToken(),B);
  assert.equal(c.captureSummary().count,1);
  assert.deepEqual(acknowledged(),[]);
  f.advance(1); await flush();
  assert.equal(state.captures.deliveryToken(),null);
  assert.equal(state.review.pendingRetirements(),1);
  assert.deepEqual(acknowledged(),[B]);
  assert.equal(f.calls.length,3);
  assert.equal(f.latest().path,QUERY);
  // The server clock is still frozen. Its replacement snapshot has a new
  // event ID but the same source, observation time and absolute deadline.
  f.latest().respond({eventId:A,rows:[row],fields}); await c.settled();
  assert.deepEqual(state.captures.snapshot(),[]);
  assert.deepEqual(c.captureSummary(),{count:null,rooms:[],roomNotice:'La habitación elegida ya no tiene muestras vigentes.'});
  assert.equal(c.captureFilters().room,null);
  assert.deepEqual(acknowledged(),[B,A]);
  media.emptied(); await state.review.retired(); await flush();
  assert.deepEqual(acknowledged(),[B,A,C]);
  await c.setFilters({listenable_only:true});
  const fresh=short+3;
  f.latest().respond({rows:[{...row,id:'vf_'+'c'.repeat(64),created_at:short,expires_at:fresh,group_expires_at:fresh}],
    fields:{...fields,observed_at:short,expires_at:fresh,count_expires_at:fresh,room_options_expires_at:fresh}});
  await c.settled();
  assert.equal(state.captures.snapshot().length,1);
  assert.equal(c.captureSummary().count,1);
});

test('retiring an earlier captures snapshot leaves the current summary intact',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  const fields={matching_count:1,scan_complete:true,count_expires_at:NOW+8,room_options_expires_at:NOW+10};
  await c.setFilters({listenable_only:true});
  f.latest().respond({rows:[groupedCapture()],fields:{...fields,room_options:['Old synthetic room']}});
  await c.settled();
  await c.setFilters({room:{kind:'room',id:'Old synthetic room'}});
  f.latest().respond({eventId:C,rows:[{...groupedCapture(),room_id:'Old synthetic room'}],
    fields:{...fields,room_options:['Current synthetic room','Old synthetic room']}});
  await c.settled();
  f.retire('voices:captures',B); await flush();
  assert.deepEqual(c.captureSummary(),{count:1,rooms:['Current synthetic room','Old synthetic room'],roomNotice:''});
  assert.equal(state.captures.snapshot().length,1);
  assert.deepEqual(c.captureFilters().room,{kind:'room',id:'Old synthetic room'});
  assert.equal(f.calls.length,2);
  f.retire('voices:captures',C); await flush();
  assert.deepEqual(c.captureSummary().rooms,[]);
  assert.equal(c.captureFilters().room,null);
  assert.equal(f.calls.length,3);
  f.latest().respond(); await c.settled(); c.dispose();
});

test('retirement tracks retained room options and search count by their own snapshots',async()=>{
  const f=controllerFixture(), {controller:c}=f;
  const fields={matching_count:1,scan_complete:true,count_expires_at:NOW+8};
  await c.setFilters({listenable_only:true});
  f.latest().respond({rows:[groupedCapture()],fields:{...fields,room_options:['Synthetic room'],room_options_expires_at:NOW+10}});
  await c.settled();
  await c.setFilters({text:'synthetic phrase'});
  f.latest().respond({eventId:C,rows:[groupedCapture()],fields}); await c.settled();
  f.retire('voices:captures',B);
  assert.deepEqual(c.captureSummary(),{count:1,rooms:[],roomNotice:''});
  f.retire('voices:captures',C);
  assert.deepEqual(c.captureSummary(),{count:null,rooms:[],roomNotice:''});
  assert.equal(f.calls.length,2);
  c.dispose();
});

test('a cursor conflict restarts captures and a denial does not expire the session',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  await c.setFilters({text:'synthetic phrase'});
  f.latest().respond({rows:[groupedCapture()],fields:{next_cursor:CURSOR}}); await c.settled();
  assert(c.next('captures')); f.latest().respond({status:409}); await flush();
  assert.deepEqual(f.query(),{...DEFAULTS,text:'synthetic phrase'}); assert.equal(f.calls.length,3);
  f.latest().respond({status:403}); await c.settled();
  assert.equal(state.captures.status(),'forbidden'); assert.equal(f.unauthorized(),0);
  c.leave(); assert.deepEqual({...c.captureFilters(),cursor:null},DEFAULTS);
  c.dispose();
});

test('a filters-only 403 checks session authority even when admin search is available',async()=>{
  const f=controllerFixture(), {controller:c,state}=f;
  assert.equal(await c.setFilters({listenable_only:true}),true);
  assert.equal(f.query().text,null);
  f.latest().respond({status:403}); await c.settled();
  assert.equal(f.unauthorized(),1);
  assert.equal(state.captures.status(),'unavailable');
  c.dispose();
});

test('capture retirement and pagehide clear the summary and the browser owner drops filter text',async()=>{
  const {readFileSync}=require('node:fs'), {runInNewContext}=require('node:vm');
  const window=new EventTarget();
  window.KzaConversationChannel=require('../../src/dashboard/frontend/obs/src/conversation-channel.js');
  runInNewContext(readFileSync(require.resolve('../../src/dashboard/frontend/obs/src/voice-channel.js'),'utf8'),{window});
  const f=controllerFixture(window.KzaVoiceChannel.createVoiceController), {controller:c,state}=f;
  await c.setFilters({group_state:'manual'});
  const fields={matching_count:1,scan_complete:true,count_expires_at:NOW+8,room_options:['Synthetic room'],room_options_expires_at:NOW+10};
  f.latest().respond({rows:[groupedCapture()],fields}); await c.settled();
  assert.deepEqual([...c.captureSummary().rooms],['Synthetic room']);
  f.retire('voices:captures');
  assert.deepEqual([...c.captureSummary().rooms],[]); assert.equal(c.captureSummary().count,null);
  await c.setFilters({text:'synthetic phrase'}); f.latest().respond({rows:[groupedCapture()],eventId:C}); await c.settled();
  assert(c.transcript(OTHER)); f.latest().respond({rows:[transcript(OTHER)],eventId:C}); await c.settled();
  window.dispatchEvent(new Event('pagehide'));
  assert.deepEqual({...c.captureFilters(),cursor:null},DEFAULTS);
  assert.deepEqual(state.review.snapshot(),[]);
  assert.deepEqual(state.captures.snapshot(),[]);
  c.dispose();
});
