const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createVoiceOwner, permittedPath } = require('../../src/dashboard/frontend/home/nexo/voice-owner.js');
const { createSession } = require('../../src/dashboard/frontend/home/nexo/state.js');
const UUID = '10000000-0000-4000-8000-000000000001';
const RESET = { type: 'live_reset', payload: { id: UUID, nonce: '20000000-0000-4000-8000-000000000002' }, ts: '' };
const READY = { type: 'rest_ready', connection_id: UUID, channels: ['voices:captures', 'voices:profiles', 'voices:review'], ts: '' };
const deferred = () => { let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve }; };
const turn = async () => { for (let index = 0; index < 12; index++) await new Promise(resolve => setImmediate(resolve)); };
class Socket extends EventTarget {
  constructor() { super(); this.readyState = 0; this.sent = []; }
  event(type, data) { const event = new Event(type); if (data !== undefined) event.data = JSON.stringify(data); this.dispatchEvent(event); }
  open() { this.readyState = 1; this.event('open'); }
  send(data) { assert.equal(this.readyState, 1); this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; this.event('close'); }
}
function setup(options = {}) {
  const session = createSession({ wall: () => 0, monotonic: () => 0 });
  session.accept({ ok: true, role: 'operator', expires_at: 1e12, permissions: ['voices:read', 'voices:review'] });
  const sockets = [], controllers = [], requests = [], mounted = new Set();
  const owner = createVoiceOwner({ session, socketUrl: () => 'ws://synthetic/ws/voices',
    connect() { const socket = new Socket(); sockets.push(socket); return socket; },
    mount(controller) { controllers.push(controller); mounted.add(controller); return () => { mounted.delete(controller); controller.leave(); for (const kind of ['profiles', 'captures', 'review']) controller.state[kind].committed(null); }; },
    fetch: async (path, request) => { requests.push([path, request]); return { ok: true, status: 200, json: async () => ({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }) }; },
    ...options });
  return { session, sockets, controllers, requests, mounted, owner };
}
async function ready(graph) { await graph.owner.open(); const socket = graph.sockets.at(-1); socket.open(); socket.event('message', RESET); await turn(); assert.equal(socket.sent.at(-1).type, 'live_reset_ack'); socket.event('message', READY); await turn(); return socket; }

test('closed voice routes and methods exclude conversation, identity and grant APIs', () => {
  for (const [path, method] of [['/api/voices', 'GET'], ['/api/voices/profiles?cursor=abc%2Fdef', 'GET'], ['/api/voices/profiles', 'POST'], ['/api/voices/profiles/' + UUID, 'DELETE']]) assert.equal(permittedPath(path, method), true);
  for (const [path, method] of [['https://elsewhere/api/voices', 'GET'], ['/api/conversations', 'GET'], ['/api/admin/users', 'GET'], ['/api/voices/profiles', 'PUT'], ['/api/voices?other=1', 'GET'], ['/api/voices?cursor=a?cursor=b', 'GET']]) assert.equal(permittedPath(path, method), false);
});
test('both voice permissions precede socket creation; REST waits for reset ACK and ready', async () => {
  const g = setup(); g.session.accept({ ok: true, role: 'home', expires_at: 1e12, permissions: ['voices:read'] });
  assert.equal(await g.owner.open(), false); assert.equal(g.sockets.length, 0);
  g.session.accept({ ok: true, role: 'operator', expires_at: 1e12, permissions: ['voices:read', 'voices:review'] });
  await g.owner.open(); const socket = g.sockets[0]; socket.open(); socket.event('message', READY); await turn(); assert.equal(g.requests.length, 0);
  socket.event('message', RESET); await turn(); assert.equal(g.requests.length, 0);
  socket.event('message', { ...READY, channels: [...READY.channels, 'conversations'] }); await turn(); assert.equal(g.requests.length, 0);
  socket.event('message', READY); await turn(); assert.equal(g.requests.length, 2);
  assert.ok(g.requests.every(([, request]) => request.headers['X-KZA-Live-Connection'] === UUID));
  await g.owner.leave(); assert.equal(g.mounted.size, 0);
});
test('reentry and new login wait for the actual old JSON decoder; stale callbacks send no ACK', async () => {
  const body = deferred(), entered = deferred(); let requests = 0;
  const g = setup({ fetch: async () => { requests++; return { ok: true, status: 200, json: () => { entered.resolve(); return body.promise; } }; } });
  const first = await ready(g); await entered.promise; const before = first.sent.length;
  const closing = g.owner.leave(); assert.equal(g.mounted.size, 0);
  g.session.clear(); g.session.accept({ ok: true, role: 'operator', expires_at: 1e12, permissions: ['voices:read', 'voices:review'] });
  const reopening = g.owner.open(); await turn(); assert.equal(g.sockets.length, 1); assert.equal(requests, 1);
  first.event('message', RESET); first.event('message', READY); await turn(); assert.equal(first.sent.length, before);
  body.resolve({ connection_id: UUID, channel: 'voices:profiles', snapshot: null });
  assert.equal(await closing, true); assert.equal(await reopening, true); assert.equal(g.sockets.length, 2);
  const second = g.sockets[1]; second.open(); second.event('message', RESET); await turn(); assert.equal(first.sent.length, before); assert.equal(second.sent.at(-1).type, 'live_reset_ack');
  await g.owner.leave();
});
test('reset ACK waits for native media retirement even with a detached DOM', async () => {
  const retired = deferred(); let pending = false;
  const g = setup({ createState() { const { createVoiceState } = require('../../src/dashboard/frontend/obs/src/voice-privacy.js'); const state = createVoiceState(); const original = state.retired; state.retired = () => pending ? retired.promise : original(); return state; } });
  const socket = await ready(g); pending = true; const before = socket.sent.length;
  socket.event('message', RESET); await turn(); assert.equal(g.mounted.size, 0); assert.equal(socket.sent.length, before);
  retired.resolve(); await turn(); assert.equal(socket.sent.at(-1).type, 'live_reset_ack'); await g.owner.leave();
});
test('late HTTP headers after leave retain cancellation and do not decode or refresh', async () => {
  const response = deferred(), cancelled = deferred(), entered = deferred(); let decodes = 0, count = 0;
  const g = setup({ fetch: async () => { count++; entered.resolve(); return response.promise; } });
  const socket = await ready(g); await entered.promise; const before = socket.sent.length;
  let finished = false; const closing = g.owner.leave().then(value => { finished = true; return value; });
  response.resolve({ ok: true, status: 200, body: { cancel: () => cancelled.promise }, json: async () => { decodes++; } });
  await turn(); assert.equal(finished, false); assert.equal(decodes, 0); cancelled.resolve(); assert.equal(await closing, true);
  assert.equal(count, 1); assert.equal(socket.sent.length, before);
});
test('a functional 403 checks the original identity and leaves a valid session active', async () => {
  let checks = 0, retired = 0;
  const g = setup({ fetch: async () => ({ ok: false, status: 403 }), verifyIdentity: async () => { checks++; return true; }, onUnauthorized: () => retired++ });
  await ready(g); assert.ok(checks > 0); assert.equal(retired, 0); assert.equal(g.session.active(), true); await g.owner.leave();
});
test('socket construction failure drains partial owners and allows an explicit retry', async () => {
  let fail = true;
  const g = setup({ connect() { if (fail) throw Error('synthetic constructor failure'); return new Socket(); } });
  assert.equal(await g.owner.open(), false); assert.equal(await g.owner.leave(), true);
  fail = false; assert.equal(await g.owner.open(), true); assert.equal(await g.owner.leave(), true);
});

test('opening under another login seals its predecessor even without an explicit leave', async () => {
  const body = deferred(), entered = deferred();
  const g = setup({ fetch: async () => ({ ok: true, status: 200, json: () => { entered.resolve(); return body.promise; } }) });
  const first = await ready(g); await entered.promise; const sent = first.sent.length;
  g.session.accept({ ok: true, role: 'home', expires_at: 1e12, permissions: [] });
  const opening = g.owner.open(); assert.equal(g.mounted.size, 0); await turn(); assert.equal(g.sockets.length, 1);
  body.resolve({ connection_id: UUID, channel: 'voices:profiles', snapshot: null });
  assert.equal(await opening, false); assert.equal(first.sent.length, sent); assert.equal(await g.owner.leave(), true);
});

test('decoder finishing after expiry cannot copy a private row or emit an ACK', async () => {
  const body = deferred(), entered = deferred(); let now = 0;
  const g = setup({ fetch: async () => ({ ok: true, status: 200, json: () => { entered.resolve(); return body.promise; } }) });
  const originalCan = g.session.can; g.session.can = permission => now === 0 && originalCan(permission);
  const socket = await ready(g); await entered.promise; const sent = socket.sent.length;
  now = 1;
  body.resolve({ connection_id: UUID, channel: 'voices:profiles', snapshot: { channel: 'voices:profiles',
    event_id: '30000000-0000-4000-8000-000000000003', observed_at: Date.now()/1000, expires_at: Date.now()/1000+60,
    receipt: { ticket: '40000000-0000-4000-8000-000000000004', nonce: '50000000-0000-4000-8000-000000000005' },
    rows: [{ id: UUID, name: 'Synthetic old private name', revision: 1, confirmed_at: Date.now()/1000 }] } });
  await turn(); assert.deepEqual(g.controllers[0].state.profiles.snapshot(), []); assert.equal(socket.sent.length, sent);
  await g.owner.leave();
});

test('a mutation body remains owned through leave and cannot start its automatic refresh', async () => {
  const body = deferred(), entered = deferred(); let requests = 0;
  const g = setup({ fetch: async (path, options) => {
    requests++;
    if (options.method === 'POST') return { ok: true, status: 200, arrayBuffer: () => { entered.resolve(); return body.promise; } };
    return { ok: true, status: 200, json: async () => ({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }) };
  } });
  const socket = await ready(g), before = socket.sent.length;
  const mutation = g.controllers[0].confirm([], 'Synthetic mutation', UUID); await entered.promise;
  let done = false; const closing = g.owner.leave().then(value => { done = true; return value; });
  await turn(); assert.equal(done, false); body.resolve(new ArrayBuffer(0));
  assert.equal(await mutation, false); assert.equal(await closing, true); assert.equal(requests, 3); assert.equal(socket.sent.length, before);
});

test('closing a review in flight clears its loading state and cannot restore the old row', async () => {
  const body = deferred(), entered = deferred();
  const g = setup({ fetch: async path => ({ ok: true, status: 200, json: () => {
    if (path.endsWith('/review')) { entered.resolve(); return body.promise; }
    return Promise.resolve({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null });
  } }) });
  await ready(g); const controller = g.controllers[0];
  controller.review('vf_'+'a'.repeat(64)); await entered.promise; controller.closeReview();
  assert.equal(controller.state.review.status(), 'waiting'); body.resolve({ connection_id: UUID, channel: 'voices:review', snapshot: null });
  await turn(); assert.deepEqual(controller.state.review.snapshot(), []); assert.equal(controller.state.review.status(), 'waiting'); await g.owner.leave();
});

test('lost confirmation receipts reuse their command ID across view reentry, independently of login B', async () => {
  const commands = []; let lose = true;
  const g = setup({ fetch: async (path, options) => {
    if (options.method === 'POST') { commands.push(JSON.parse(options.body)); return { ok: true, status: 200, arrayBuffer: async () => { if (lose) { lose = false; throw Error('synthetic lost receipt'); } return new ArrayBuffer(0); } }; }
    if (options.method === 'DELETE') return { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) };
    return { ok: true, status: 200, json: async () => ({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }) };
  } });
  await ready(g); assert.equal(await g.controllers.at(-1).confirm(['c', 'a', 'b'], 'Synthetic retry', UUID), false);
  await g.owner.leave(); await ready(g);
  assert.equal(await g.controllers.at(-1).confirm(['a', 'b', 'c'], 'Synthetic retry', UUID), true);
  assert.equal(commands[0].request_id, commands[1].request_id);
  assert.equal(await g.controllers.at(-1).deleteProfile(UUID, 1), true);
  await g.controllers.at(-1).confirm(['a', 'b', 'c'], 'Synthetic retry', UUID);
  assert.notEqual(commands[2].request_id, commands[1].request_id);
  g.session.accept({ ok: true, role: 'operator', expires_at: 1e12, permissions: ['voices:read', 'voices:review'] });
  await ready(g); await g.controllers.at(-1).confirm(['a', 'b', 'c'], 'Synthetic retry', UUID);
  assert.notEqual(commands[3].request_id, commands[0].request_id); await g.owner.clear();
});

test('manual refresh waits for the superseded decoder before admitting replacement queries', async () => {
  const body = deferred(), entered = deferred(); let count = 0;
  const g = setup({ fetch: async path => {
    count++; const first = count === 1;
    return { ok: true, status: 200, json: () => { if (first) { entered.resolve(); return body.promise; }
      return Promise.resolve({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }); } };
  } });
  await ready(g); await entered.promise; const refresh = g.controllers[0].refresh(); await turn(); assert.equal(count, 1);
  body.resolve({ connection_id: UUID, channel: 'voices:profiles', snapshot: null }); await refresh; await g.controllers[0].settled();
  assert.equal(count, 3); assert.equal(g.controllers[0].state.captures.status(), 'real'); await g.owner.leave();
});

test('confirmation works on the HTTP LAN without crypto.randomUUID', async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto'), nativeCrypto = globalThis.crypto;
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { getRandomValues: values => nativeCrypto.getRandomValues(values) } });
  try {
    let identity;
    const g = setup({ fetch: async (path, options) => {
      if (options.method === 'POST') { identity = JSON.parse(options.body).request_id; return { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) }; }
      return { ok: true, status: 200, json: async () => ({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }) };
    } });
    await ready(g); assert.equal(await g.controllers[0].confirm(['a', 'b', 'c'], 'Synthetic LAN'), true);
    assert.match(identity, /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/); await g.owner.clear();
  } finally { Object.defineProperty(globalThis, 'crypto', descriptor); }
});

test('retiring the view drops its previously delivered page cursors', async () => {
  const g = setup({ fetch: async path => ({ ok: true, status: 200, json: async () => ({ connection_id: UUID,
    channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null, next_cursor: UUID }) }) });
  await ready(g); const controller = g.controllers[0]; assert.equal(controller.hasNext('profiles'), true); assert.equal(controller.hasNext('captures'), true);
  await g.owner.leave(); assert.equal(controller.hasNext('profiles'), false); assert.equal(controller.hasNext('captures'), false);
});

test('uncertain command identities are bounded and clear with the browser session', async () => {
  const commands = [];
  const g = setup({ fetch: async (path, options) => {
    if (options.method === 'POST') { commands.push(JSON.parse(options.body)); return { ok: false, status: 503, arrayBuffer: async () => new ArrayBuffer(0) }; }
    return { ok: true, status: 200, json: async () => ({ connection_id: UUID, channel: path.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null }) };
  } });
  await ready(g); const controller = g.controllers[0];
  for (let index = 0; index < 64; index++) assert.equal(await controller.confirm(['a', 'b', 'c'], 'Synthetic '+index), false);
  assert.equal(await controller.confirm(['a', 'b', 'c'], 'Synthetic overflow'), false); assert.equal(commands.length, 64);
  await controller.confirm(['a', 'b', 'c'], 'Synthetic 0'); assert.equal(commands[0].request_id, commands[64].request_id);
  await g.owner.clear(); await ready(g); await g.controllers.at(-1).confirm(['a', 'b', 'c'], 'Synthetic overflow'); assert.equal(commands.length, 66);
  await g.owner.clear();
});

const adminIdentity = { ok:true,role:'operator',expires_at:1e12,permissions:['voices:read','voices:review','admin:manage'] };
const personRows = [{id:'null',name:'Synthetic person',permission_level:'adult',is_active:true}];
const voiceEnvelope = (path,snapshot=null) => ({connection_id:UUID,channel:path.includes('/profiles')?'voices:profiles':'voices:captures',snapshot});
const profileBatch = () => ({channel:'voices:profiles',event_id:'30000000-0000-4000-8000-000000000003',observed_at:Date.now()/1000,expires_at:Date.now()/1000+60,
  receipt:{ticket:'40000000-0000-4000-8000-000000000004',nonce:'50000000-0000-4000-8000-000000000005'},rows:[{id:UUID,name:'Synthetic profile',revision:1,confirmed_at:Date.now()/1000,person_link:{user_id:'null',revision:1}}]});

test('person metadata uses explicit admin capability and closed query combinations', () => {
  const person = '/api/voices/profiles/'+UUID+'/person';
  assert.equal(permittedPath(person,'PUT'),false); assert.equal(permittedPath(person,'PUT',true),true);
  assert.equal(permittedPath('/api/voices/profiles?include_person=true','GET'),false);
  assert.equal(permittedPath('/api/voices/profiles?include_person=true&cursor=opaque%2Fcursor','GET',true),true);
  for (const path of ['/api/admin/users','/api/voices?include_person=true','/api/voices/profiles?include_person=false','/api/voices/profiles?include_person=true&include_person=true','/api/voices/profiles?cursor=a&cursor=b','/api/voices/profiles?include_person=true&other=1']) assert.equal(permittedPath(path,'GET',true),false);
});

test('fixed people lookup requires admin and returns only a boolean after its consumer', async () => {
  let peopleCalls=0;
  const g=setup({fetch:async path=>{ if(path==='/api/admin/users'){peopleCalls++;return{ok:true,status:200,json:async()=>personRows};} return{ok:true,status:200,json:async()=>voiceEnvelope(path)}; }});
  await ready(g);assert.equal(await g.controllers.at(-1).loadPeople(()=>assert.fail('no admin consumer')),false);assert.equal(peopleCalls,0);await g.owner.leave();
  g.session.accept(adminIdentity);await ready(g);let accepted;
  assert.equal(await g.controllers.at(-1).loadPeople(rows=>{accepted=rows;}),true);
  assert.deepEqual(accepted,[{id:'null',name:'Synthetic person',is_active:true}]);assert(Object.isFrozen(accepted));assert(Object.isFrozen(accepted[0]));assert.equal(peopleCalls,1);
  assert(g.requests.every(([path])=>!path.includes('include_person')||path.includes('include_person=true')));await g.owner.leave();
});

test('late people decoder holds close and login B, then rejects its old consumer', async () => {
  const body=deferred(),entered=deferred();let accepted=0;
  const g=setup({fetch:async path=>path==='/api/admin/users'?{ok:true,status:200,json:()=>{entered.resolve();return body.promise;}}:{ok:true,status:200,json:async()=>voiceEnvelope(path)}});
  g.session.accept(adminIdentity);const socket=await ready(g), before=socket.sent.length;
  const loading=g.controllers[0].loadPeople(()=>accepted++);await entered.promise;
  let closed=false;const closing=g.owner.leave().then(value=>{closed=true;return value;});g.session.clear();g.session.accept(adminIdentity);const reentry=g.owner.open();await turn();
  assert.equal(closed,false);assert.equal(g.sockets.length,1);body.resolve(personRows);
  assert.equal(await loading,false);assert.equal(await closing,true);assert.equal(await reentry,true);assert.equal(g.sockets.length,2);assert.equal(accepted,0);assert.equal(socket.sent.length,before);await g.owner.leave();
});

test('reset aborts auxiliary admission and waits for an actual async consumer', async () => {
  const consumer=deferred(),entered=deferred();let signal;
  const g=setup({fetch:async(path,options)=>{if(path==='/api/admin/users'){signal=options.signal;return{ok:true,status:200,json:async()=>personRows};}return{ok:true,status:200,json:async()=>voiceEnvelope(path)};}});
  g.session.accept(adminIdentity);const socket=await ready(g), before=socket.sent.filter(row=>row.type==='live_reset_ack').length;
  const loading=g.controllers[0].loadPeople(async()=>{entered.resolve();await consumer.promise;});await entered.promise;socket.event('message',RESET);await turn();
  assert.equal(signal.aborted,true);assert.equal(socket.sent.filter(row=>row.type==='live_reset_ack').length,before);consumer.resolve();assert.equal(await loading,false);await turn();assert.equal(socket.sent.filter(row=>row.type==='live_reset_ack').length,before+1);await g.owner.leave();
});

for(const domClearedFirst of [false,true])test('auxiliary completion flush preserves actual DOM commit: '+domClearedFirst,async()=>{
  const body=deferred(),entered=deferred(),batch=profileBatch();
  const g=setup({fetch:async path=>path==='/api/admin/users'?{ok:true,status:200,json:()=>{entered.resolve();return body.promise;}}:{ok:true,status:200,json:async()=>voiceEnvelope(path,path.includes('/profiles')?batch:null)}});
  g.session.accept(adminIdentity);const socket=await ready(g),controller=g.controllers[0],store=controller.state.profiles;
  store.committed(batch.event_id);const loading=controller.loadPeople(()=>{});await entered.promise;
  socket.event('message',{type:'rest_retire',channel:'voices:profiles',payload:{event_ids:[batch.event_id],receipts:[{event_id:batch.event_id,...batch.receipt}]}});await turn();
  if(domClearedFirst)store.committed(null);
  assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,0);body.resolve(personRows);assert.equal(await loading,true);await turn();
  assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,domClearedFirst?1:0);
  if(!domClearedFirst){store.committed(null);await turn();}
  assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,1);await g.owner.leave();
});

test('failed people consumer drains admission and permits an explicit next lookup',async()=>{
  const g=setup({fetch:async path=>({ok:true,status:200,json:async()=>path==='/api/admin/users'?personRows:voiceEnvelope(path)})});g.session.accept(adminIdentity);await ready(g);
  assert.equal(await g.controllers[0].loadPeople(()=>{throw Error('synthetic callback failure');}),false);
  assert.equal(await g.controllers[0].loadPeople(()=>{}),true);await g.owner.leave();
});

test('catalogue, mutation body and native audio are independent retirement barriers',async()=>{
  const {Media,review}=require('../mocks/voice_browser.cjs'), personBody=deferred(),mutationBody=deferred(),personEntered=deferred(),mutationEntered=deferred();
  const now=Date.now()/1000,batch={...profileBatch(),channel:'voices:review',rows:[{...review(),observed_at:now,expires_at:now+60}]};
  const g=setup({fetch:async(path,options)=>{
    if(path==='/api/admin/users')return{ok:true,status:200,json:()=>{personEntered.resolve();return personBody.promise;}};
    if(options.method==='PUT')return{ok:true,status:200,arrayBuffer:()=>{mutationEntered.resolve();return mutationBody.promise;}};
    if(path.endsWith('/review'))return{ok:true,status:200,json:async()=>({connection_id:UUID,channel:'voices:review',snapshot:batch})};
    return{ok:true,status:200,json:async()=>voiceEnvelope(path)};
  }});
  g.session.accept(adminIdentity);const socket=await ready(g),controller=g.controllers[0];assert(controller.review(review().id));await controller.settled();
  const media=new Media();controller.state.review.attachMedia(media);controller.state.review.committed(batch.event_id);
  const loading=controller.loadPeople(()=>{}),saving=controller.setPerson(UUID,'null',1,0);await personEntered.promise;await mutationEntered.promise;
  socket.event('message',{type:'rest_retire',channel:'voices:review',payload:{event_ids:[batch.event_id],receipts:[{event_id:batch.event_id,...batch.receipt}]}});await turn();controller.state.review.committed(null);
  personBody.resolve(personRows);assert.equal(await loading,true);await turn();assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,0);
  mutationBody.resolve(new ArrayBuffer(0));assert.equal(await saving,true);await turn();assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,0);
  media.emptied();await controller.state.retired();await turn();assert.equal(socket.sent.filter(row=>row.type==='live_ack').length,1);await g.owner.leave();
});

test('a deferred identity check cannot start under the next login',async()=>{
  let verify=0, unauthorized;
  const {createVoiceController}=require('../../src/dashboard/frontend/obs/src/voice-channel.js');
  const g=setup({verifyIdentity:async()=>{verify++;return false;},createController:options=>{unauthorized=options.onUnauthorized;return createVoiceController(options);}});
  await ready(g);unauthorized();const closing=g.owner.leave();g.session.clear();g.session.accept(adminIdentity);await turn();assert.equal(verify,0);assert.equal(await closing,true);
});
