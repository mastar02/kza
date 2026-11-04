"""Exercise the actual browser owner in Node with synthetic media leaves."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
COMMON = r"""
const assert = require('node:assert/strict');
const {createModalOwner} = require('./src/dashboard/frontend/obs/src/capture-recorder.js');
function deferred() { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; }
async function until(predicate) { for(let i=0;i<200;i++){if(predicate())return;await Promise.resolve();}throw Error('semantic witness not reached'); }
function fixture(config={}) {
  const events=[], contexts=[], timers=new Map(), token='a'.repeat(43); let serial=0, reads=0;
  const track={readyState:'live',stop(){events.push('track-stop');this.readyState='ended';}};
  const stream={getTracks(){return [track];}};
  const later=(fn,delay)=>{const id=++serial;timers.set(id,{fn,delay});return id;};
  const cancel=id=>timers.delete(id);
  function fire(delay){const row=[...timers].find(([,v])=>v.delay===delay);assert.ok(row,'missing timer '+delay);timers.delete(row[0]);row[1].fn();}
  class Audio {
    constructor(){this.state='running';this.destination={};contexts.push(this);events.push('context-created');}
    createMediaStreamSource(){return {connect(){},disconnect(){events.push('source-disconnected');}};}
    createScriptProcessor(){const proc={connect(){},disconnect(){events.push('processor-disconnected');},onaudioprocess:null};this.proc=proc;return proc;}
    async close(){events.push('context-close-enter');if(config.close)await config.close();this.state='closed';events.push('context-close-return');}
  }
  class Form {constructor(){this.entries=[];}append(...args){this.entries.push(args);}}
  const fetch=async(path,opts)=>{
    const action=path.split('/').at(-1);events.push(action);
    if(config.fetch){const value=await config.fetch(path,opts,action);if(value)return value;}
    const data=action==='register'?{token,state:'reserved',capture_allowed:false}:
      action==='retire'?{retired:true}:{state:'active',capture_allowed:true};
    return {ok:true,status:200,json:async()=>data};
  };
  const owner=createModalOwner({fetch,getUserMedia:async constraints=>{
    events.push('gum');assert.equal(constraints.audio.sampleRate,48000);
    return config.media?await config.media():stream;
  },AudioContext:Audio,FormData:Form,setTimeout:later,clearTimeout:cancel,
  onRetiring(){events.push('retiring');}});
  function pcm(proc){proc.onaudioprocess({inputBuffer:{getChannelData(){reads++;return new Float32Array([0,0.3,0.6,-0.3,-0.6,-0.9]);}}});}
  const encode=(samples,rate)=>{events.push('encode');assert.equal(rate,16000);return {samples:[...samples],rate};};
  return {owner,events,contexts,track,stream,fire,pcm,encode,reads:()=>reads};
}
"""

CASES = {
    "late-acquisition": r"""
const acquisition=deferred(), f=fixture({media:()=>acquisition.promise});
const recording=f.owner.recordWavSample(3,f.encode);recording.catch(()=>{});
await until(()=>f.events.includes('gum'));
assert.ok(f.events.indexOf('activate')<f.events.indexOf('gum'));
let stopped=false;const stop=f.owner.stop().then(()=>stopped=true);
await Promise.resolve();assert.equal(stopped,false);assert.equal(f.events.includes('retire'),false);
acquisition.resolve(f.stream);await assert.rejects(recording,/capture_muted/);await stop;
assert.equal(f.track.readyState,'ended');assert.equal(f.contexts.length,0);
assert.equal(f.owner.sampleCount(),0);assert.equal(f.events.filter(x=>x==='retire').length,1);
assert.ok(f.events.indexOf('track-stop')<f.events.indexOf('retire'));
""",
    "context-cleanup-receipt": r"""
const closing=deferred(), f=fixture({close:()=>closing.promise});
const recording=f.owner.recordWavSample(3,f.encode);recording.catch(()=>{});
await until(()=>f.contexts[0]?.proc?.onaudioprocess);f.pcm(f.contexts[0].proc);f.fire(3000);
await until(()=>f.events.includes('context-close-enter'));
const stop=f.owner.stop();await Promise.resolve();assert.equal(f.events.includes('retire'),false);
closing.resolve();await assert.rejects(recording,/capture_muted/);await stop;
assert.equal(f.events.filter(x=>x==='context-close-enter').length,1);
assert.ok(f.events.indexOf('context-close-return')<f.events.indexOf('retire'));
assert.equal(f.owner.sampleCount(),0);
""",
    "samples-heartbeat-late-callback": r"""
const f=fixture({fetch:async(_p,_o,action)=>action==='poll'?{ok:true,status:200,json:async()=>({state:'active',capture_allowed:false})}:null});
const recording=f.owner.recordWavSample(3,f.encode);
await until(()=>f.contexts[0]?.proc?.onaudioprocess);const late=f.contexts[0].proc.onaudioprocess;
f.pcm(f.contexts[0].proc);f.fire(3000);const blob=await recording;
assert.deepEqual(blob.samples,[9830,-19660]);assert.equal(f.owner.sampleCount(),1);
const before=f.reads();f.fire(500);await until(()=>f.events.includes('retire'));
await f.owner.stop();late({inputBuffer:{getChannelData(){throw Error('late PCM copied');}}});
assert.equal(f.reads(),before);assert.equal(f.owner.sampleCount(),0);
assert.equal(f.events.filter(x=>x==='retiring').length,1);
""",
    "upload-abort-receipt": r"""
const cleanup=deferred();let requestBody=null,aborted=false;
const f=fixture({fetch:async(path,opts)=>{
 if(!path.includes('/api/admin/'))return null;
 requestBody=opts.body;await new Promise(resolve=>opts.signal.addEventListener('abort',()=>{aborted=true;resolve();},{once:true}));
 await cleanup.promise;throw Error('upload aborted');
}});
const recording=f.owner.recordWavSample(3,f.encode);
await until(()=>f.contexts[0]?.proc?.onaudioprocess);f.pcm(f.contexts[0].proc);f.fire(3000);await recording;
const upload=f.owner.uploadSamples('/api/admin/users/private/enroll');upload.catch(()=>{});
await until(()=>requestBody);assert.equal(requestBody.entries.length,1);
const stopping=f.owner.stop();await until(()=>aborted);
assert.equal(f.owner.sampleCount(),0);assert.equal(f.events.includes('retire'),false);
cleanup.resolve();await assert.rejects(upload,/upload aborted/);await stopping;
assert.equal(f.events.filter(x=>x==='retire').length,1);
""",
    "failed-cleanup-no-ack": r"""
const f=fixture({close:async()=>{throw Error('synthetic close failure');}});
const recording=f.owner.recordWavSample(3,f.encode);recording.catch(()=>{});
await until(()=>f.contexts[0]?.proc?.onaudioprocess);f.fire(3000);
await assert.rejects(recording,/capture_retirement_unconfirmed/);
await assert.rejects(f.owner.stop(),/capture_retirement_unconfirmed/);
assert.equal(f.events.includes('retire'),false);assert.equal(f.owner.sampleCount(),0);
assert.equal(f.events.filter(x=>x==='context-close-enter').length,1);
""",
}


@pytest.mark.parametrize("case", CASES)
def test_actual_browser_owner(case):
    node = shutil.which("node")
    assert node, "Node is required to verify the shipped browser owner"
    script = COMMON + "\nconst watchdog=setTimeout(()=>{console.error('watchdog');process.exit(2);},10000);\n"
    script += "(async()=>{\n" + CASES[case] + "\n})().then(()=>{clearTimeout(watchdog);console.log('CAPTURE_CASE_COMPLETE');},error=>{clearTimeout(watchdog);console.error(error);process.exitCode=1;});"
    result = subprocess.run([node, "-e", script], cwd=ROOT, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("CAPTURE_CASE_COMPLETE") == 1
