const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createConversationState } = require('../../src/dashboard/frontend/obs/src/conversation-privacy.js');
const { createConversationChannel } = require('../../src/dashboard/frontend/obs/src/conversation-channel.js');
const A = '00000000-0000-4000-8000-000000000001';
const B = '00000000-0000-4000-8000-000000000002';
const C = '00000000-0000-4000-8000-000000000003';
const flush = async () => { for (let i=0; i<6; i++) await Promise.resolve(); };
function fixture(view = false) {
  let finishResponse, finishJSON, latest;
  const acks = [], states = [], timers = new Map(); let next=0, unauthorized=0;
  const store = createConversationState({ monotonic:()=>0, wall:()=>2000000000000,
    schedule:(f,ms) => { timers.set(++next,{f,ms}); return next; }, cancel:id=>timers.delete(id) });
  const channel = createConversationChannel({ store, hasView:()=>view, monotonic:()=>0, wall:()=>2000000000000,
    onUnauthorized:()=>{unauthorized++;},
    onState:state=>states.push(state),
    emitAck:message=>{acks.push(message); return true;},
    fetch:(path,options)=>{ latest={path,options}; return new Promise(resolve=>{ finishResponse=resolve; }); } });
  const document = { connection_id:A, snapshot:{ event_id:B, expires_at:2000000060, observed_at:2000000000,
    receipt:{ticket:B,nonce:C}, rows:[{id:`turn_1_${C}`,ts:'12:00:00',user:'Synthetic voice',zone:'Synthetic room',path:'fast',
      stt:'Synthetic private REST turn',tts:'Synthetic response',intent:'light.turn_on',target:'light.synthetic',
      observed_at:2000000000,expires_at:2000000060,success:null,latency_ms:20}] } };
  async function respond(status=200) { finishResponse({ok:status===200,status,json:()=>new Promise(resolve=>{finishJSON=resolve;})}); await flush(); }
  async function body(value=document) { finishJSON(value); await flush(); }
  const retire = () => channel.retire({ type:'rest_retire',payload:{event_ids:[B],receipts:[{event_id:B,ticket:B,nonce:C}]} });
  return {store,channel,acks,states,document,respond,body,retire,latest:()=>latest,unauthorized:()=>unauthorized};
}
test('REST starts only after a confirmed connection and uses its exact header', async () => {
  const f=fixture();
  assert.equal(f.latest(),undefined); assert.equal(f.channel.open('bad'),false);
  f.channel.open(A); assert.equal(f.latest().options.headers['X-KZA-Live-Connection'],A);
  assert.equal(f.channel.open(A),false);
  await f.respond(); await f.body();
  assert.equal(f.store.snapshot()[0].stt,'Synthetic private REST turn');
  assert.equal(f.acks.length,0); f.channel.dispose(); await flush();
});
test('retirement waits for both cache and explicit DOM commit then defers its ACK', async () => {
  const f=fixture(true); f.channel.open(A); await f.respond(); await f.body(); f.store.committed(B);
  f.retire(); await flush(); assert.deepEqual(f.store.snapshot(),[]); assert.equal(f.acks.length,0);
  f.store.committed(null); assert.equal(f.acks.length,0); await flush();
  assert.deepEqual(f.acks,[{type:'live_ack',receipts:[{ticket:B,nonce:C}]}]); f.channel.dispose();
});
test('unseen retirement cannot ACK or re-admit a response still being decoded', async () => {
  const f=fixture(); f.channel.open(A); await f.respond();
  f.retire(); await flush(); assert.equal(f.acks.length,0);
  await f.body(); assert.deepEqual(f.store.snapshot(),[]);
  assert.equal(f.acks[0].receipts[0].ticket,B);
  assert(!JSON.stringify(f.acks).includes('Synthetic private')); f.channel.dispose();
});
test('reset waits for actual decode settlement even after asking fetch to abort', async () => {
  const f=fixture(); f.channel.open(A); await f.respond();
  let settled=false; const resetting=f.channel.reset().then(value=>{settled=true;return value;}); await flush();
  assert(f.latest().options.signal.aborted); assert.equal(settled,false);
  await f.body(); assert.equal(await resetting,true); assert.deepEqual(f.store.snapshot(),[]);
  assert.equal(f.acks.length,0); f.channel.dispose();
});
test('connection mismatch and malformed capabilities never populate the cache', async () => {
  for (const change of [doc=>{doc.connection_id=C;},doc=>{doc.snapshot.receipt.nonce='bad';}]) {
    const f=fixture(); change(f.document); f.channel.open(A); await f.respond(); await f.body();
    assert.deepEqual(f.store.snapshot(),[]); assert.equal(f.acks.length,0); f.channel.dispose();
  }
});
test('one batch uses its earliest source deadline for every cached row', async () => {
  const f=fixture(); f.document.snapshot.rows[0].expires_at+=100;
  f.channel.open(A); await f.respond(); await f.body();
  assert.equal(f.store.snapshot()[0].expires_at,f.document.snapshot.expires_at); f.channel.dispose();
});
test('disposing a mounted cache clears it but cannot acknowledge a decode still running', async () => {
  const f=fixture(); f.channel.open(A); await f.respond(); f.channel.dispose(); await flush();
  assert(f.latest().options.signal.aborted); assert.equal(f.acks.length,0);
  await f.body(); assert.deepEqual(f.store.snapshot(),[]); assert.equal(f.acks.length,0);
});
test('an unauthorized REST read clears the cache and requests session retirement', async () => {
  const f=fixture(); f.channel.open(A); await f.respond(401);
  assert.equal(f.unauthorized(),1); assert.deepEqual(f.store.snapshot(),[]); f.channel.dispose();
});
test('real source status requires the actual response decoder to settle', async () => {
  const f=fixture(); f.channel.open(A); await f.respond();
  assert.deepEqual(f.states,['loading']);
  await f.body(); assert.equal(f.states.at(-1),'real'); f.channel.dispose();
});
test('a stale response cannot publish real source status after reset', async () => {
  const f=fixture(); f.channel.open(A); await f.respond();
  const resetting=f.channel.reset(); await f.body(); await resetting;
  assert.equal(f.states.at(-1),'waiting'); assert(!f.states.includes('real')); f.channel.dispose();
});
