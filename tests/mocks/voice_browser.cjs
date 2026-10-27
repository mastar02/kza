const { createVoiceStore } = require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
const { createConversationChannel } = require('../../src/dashboard/frontend/obs/src/conversation-channel.js');
const A='00000000-0000-4000-8000-000000000001', B='00000000-0000-4000-8000-000000000002', C='00000000-0000-4000-8000-000000000003';
const VF='vf_'+'a'.repeat(64);
function wave(seconds=1) {
  const buffer=Buffer.alloc(44+seconds*32000);
  buffer.write('RIFF'); buffer.writeUInt32LE(buffer.length-8,4); buffer.write('WAVEfmt ',8);
  buffer.writeUInt32LE(16,16); buffer.writeUInt16LE(1,20); buffer.writeUInt16LE(1,22);
  buffer.writeUInt32LE(16000,24); buffer.writeUInt32LE(32000,28); buffer.writeUInt16LE(2,32); buffer.writeUInt16LE(16,34);
  buffer.write('data',36); buffer.writeUInt32LE(buffer.length-44,40); return buffer.toString('base64');
}
function review() { return { id:VF, observed_at:2000000000,expires_at:2000000060,transcript:'Synthetic private voice phrase',
  room_id:'Synthetic room',duration_s:1,clip_duration_s:1,audio_mime:'audio/wav',transcript_scope:'full_segment',audio_base64:wave() }; }
function profile() { return { id:A, name:'Synthetic reviewed profile',revision:1,confirmed_at:2000000000,
  observed_at:2000000000,expires_at:2000000300 }; }
class Media extends EventTarget {
  constructor(active=true) { super(); this.networkState=active?2:0; this.readyState=active?4:0; this.currentSrc=active?'blob:synthetic':'';
    this.isConnected=true; this.paused=!active; this.srcObject=null;this.buffered={length:active?1:0};this.duration=active?1:NaN; }
  pause() { this.paused=true; }
  removeAttribute() {}
  hasAttribute() { return false; }
  querySelector() { return null; }
  remove() { this.isConnected=false; }
  load() { this.networkState=0; this.readyState=0; this.buffered.length=0;this.duration=NaN; }
  emptied() { this.dispatchEvent(new Event('emptied')); }
}
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
function fixture(kind='review',view=false) {
  const urls=new Map(),revoked=[],timers=new Map(),acks=[],requests=[]; let next=0,mono=0,wall=2000000000000,respond,decode;
  const store=createVoiceStore(kind,{monotonic:()=>mono,wall:()=>wall,
    schedule:(fn,ms)=>{timers.set(++next,{fn,ms});return next;},cancel:id=>timers.delete(id),
    url:{createObjectURL:blob=>{const id='blob:'+ ++next;urls.set(id,blob);return id;},revokeObjectURL:id=>{revoked.push(id);urls.delete(id);}}});
  const channel=createConversationChannel({store,path:'/api/voices',channel:'voices:'+kind,
    monotonic:()=>mono,wall:()=>wall,hasView:()=>view,emitAck:message=>{acks.push(message);return true;},
    fetch:()=>{requests.push(true);return new Promise(resolve=>{respond=resolve;});}});
  const document={connection_id:A,channel:'voices:'+kind,snapshot:{event_id:B,observed_at:2000000000,
    expires_at:kind==='profiles'?2000000300:2000000060,channel:'voices:'+kind,receipt:{ticket:B,nonce:C},rows:[kind==='profiles'?profile():review()]}};
  return {store,channel,urls,revoked,timers,acks,requests,document,advance:(m,w=m)=>{mono+=m;wall+=w;},
    async response(){respond({ok:true,status:200,json:()=>new Promise(resolve=>{decode=resolve;})});await flush();},
    async body(){decode(document);await flush();},
    retire:()=>channel.retire({type:'rest_retire',channel:'voices:'+kind,payload:{event_ids:[B],receipts:[{event_id:B,ticket:B,nonce:C}]}})};
}
module.exports={A,B,C,VF,wave,review,profile,Media,fixture,flush};
