(function (root) {
  'use strict';
  const channelsAPI = root?.KzaConversationChannel || require('./conversation-channel.js');
  const PATHS = { profiles:'/api/voices/profiles', captures:'/api/voices', review:'/api/voices/captures/' };
  function createVoiceController({ state, fetch, emitAck, hasView = () => false, onUnauthorized = () => {}, onStatus = () => {} }) {
    let closed = false, sequence = 0;
    const pages = {}, actions = new Set(), aborts = new Set();
    const channels = Object.fromEntries(Object.keys(PATHS).map(kind => [kind,channelsAPI.createConversationChannel({
      store:state[kind], fetch, path:PATHS[kind], channel:`voices:${kind}`, loadOnOpen:false, hasView,
      emitAck:message => actions.size===0 && emitAck(message), onUnauthorized,
      onPage:cursor => { pages[kind] = cursor; },
      onState:value => { state[kind].setStatus(value); onStatus(value); },
    })]));
    async function refresh() {
      const attempt = ++sequence;
      for (const kind of Object.keys(channels)) channels[kind].clearView();
      if (!channels.profiles.refresh()) return;
      await channels.profiles.settled();
      if (closed || attempt!==sequence) return;
      channels.captures.refresh();
    }
    async function mutate(path,body,method) {
      if (closed || actions.size) return false;
      const abort = new AbortController(); aborts.add(abort);
      let encoded = JSON.stringify(body), response = null;
      const task = (async () => {
        try {
          response = await fetch(path,{ method, credentials:'same-origin', signal:abort.signal,
            headers:{'Content-Type':'application/json'}, body:encoded });
          if (response.status===401 || response.status===403) onUnauthorized();
          // Mutation responses carry status/IDs only. Drain their real body so
          // an error echo or delayed transport cannot outlive action ownership.
          await response.arrayBuffer();
          return response.ok && !abort.signal.aborted;
        } catch { return false; }
        finally { encoded = body = response = null; }
      })();
      actions.add(task);
      let accepted = false;
      try { accepted = await task; }
      finally {
        actions.delete(task); aborts.delete(abort);
        for (const kind of Object.keys(channels)) state[kind].committed(state[kind].deliveryToken());
      }
      if (!closed && accepted) await refresh();
      return accepted;
    }
    function leave() {
      sequence++;
      for (const abort of aborts) abort.abort();
      for (const channel of Object.values(channels)) channel.clearView();
      state.clear();
    }
    return {
      state,refresh,leave,
      open(identity) { for (const channel of Object.values(channels)) channel.open(identity); if (hasView()) refresh(); },
      async reset() {
        sequence++; for (const abort of aborts) abort.abort(); state.clear();
        await Promise.allSettled([...actions]);
        const done = await Promise.all(Object.values(channels).map(channel => channel.reset()));
        await state.retired(); return done.every(Boolean);
      },
      retire(frame) { for (const channel of Object.values(channels)) channel.retire(frame); },
      review(identity) { return /^vf_[0-9a-f]{64}$/.test(identity) && channels.review.refresh(PATHS.review+identity+'/review'); },
      next(kind) { const cursor = pages[kind]; if (cursor && kind!=='review') channels[kind].refresh(PATHS[kind]+'?cursor='+encodeURIComponent(cursor)); },
      hasNext:kind => !!pages[kind],
      confirm(identities,name,requestId) { return mutate('/api/voices/profiles',
        { fingerprint_ids:identities, name, request_id:requestId },'POST'); },
      correctGroups(identities,action,requestId) {
        const rows = state.captures.snapshot().filter(row => identities.includes(row.id));
        if (!identities.length || rows.length!==identities.length || new Set(identities).size!==identities.length) return Promise.resolve(false);
        return mutate('/api/voices/groups/corrections',{fingerprint_ids:identities,
          expected_revisions:Object.fromEntries(rows.map(row => [row.group_id,row.group_revision])),action,request_id:requestId},'POST');
      },
      deleteProfile(identity,revision) { return mutate('/api/voices/profiles/'+identity,{expected_revision:revision},'DELETE'); },
      dispose() { closed = true; leave(); for (const channel of Object.values(channels)) channel.dispose(); },
    };
  }
  const api = { createVoiceController };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaVoiceChannel = api;
})(typeof window === 'undefined' ? null : window);
