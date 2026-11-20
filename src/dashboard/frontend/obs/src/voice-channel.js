(function (root) {
  'use strict';
  const channelsAPI = root?.KzaConversationChannel || require('./conversation-channel.js');
  const PATHS = { profiles:'/api/voices/profiles', captures:'/api/voices/captures/query', review:'/api/voices/captures/' };
  const DEFAULT_FILTERS = Object.freeze({room:null,window:null,group_state:null,association:null,listenable_only:false,text:null});
  const CAPTURE_QUERY_KEYS = Object.freeze([...Object.keys(DEFAULT_FILTERS),'cursor']);
  const CAPTURE_ID = /^vf_[0-9a-f]{64}$/;
  const ROOM_GONE = 'La habitación elegida ya no tiene muestras vigentes.';
  function createVoiceController({ state, fetch, emitAck, hasView = () => false, onUnauthorized = () => {}, onStatus = () => {},
      includePerson = false, profilesOnly = false, personId = null, canSearch = () => false,
      monotonic = () => performance.now(), wall = Date.now, schedule = setTimeout, cancel = clearTimeout }) {
    if (typeof includePerson !== 'boolean') throw Error('invalid voice person scope');
    if (typeof profilesOnly !== 'boolean' || profilesOnly && (!includePerson || typeof personId !== 'string' || !personId.length || personId.length > 128)
        || !profilesOnly && personId !== null) throw Error('invalid voice profile scope');
    const profilePath = cursor => PATHS.profiles + (includePerson ? '?include_person=true' : '')
      + (personId !== null ? '&person_id='+encodeURIComponent(personId) : '')
      + (cursor ? (includePerson ? '&' : '?') + 'cursor=' + encodeURIComponent(cursor) : '');
    let closed = false, sequence = 0, capturesSequence = 0, reapplying = false;
    let filters = DEFAULT_FILTERS, count = null, rooms = null, summaryTimer = null, roomNotice = '';
    let queryCursor = null, queryHasSearch = false;
    const pages = {}, actions = new Set(), aborts = new Set();
    const filterListeners = new Set();
    const searchAllowed = () => !profilesOnly && canSearch() === true;
    const expired = entry => entry && (entry.mono <= monotonic() || entry.wall <= wall());
    const notifyFilters = (reset = false) => { for (const listener of filterListeners) listener(reset); };
    function timed(value, deadline, observed, started, page) {
      if (!Number.isFinite(observed) || !Number.isFinite(deadline) || deadline <= observed || deadline-observed > 172800) return null;
      const expiresAt = Math.min(deadline,page.expires_at);
      const elapsed = Math.max(0,monotonic()-started.mono,wall()-started.wall);
      const budget = (expiresAt-observed)*1000-elapsed;
      if (budget <= 0 || expiresAt*1000 <= wall()) return null;
      return {value,token:page.event_id,mono:monotonic()+budget,wall:Math.min(expiresAt*1000,wall()+budget)};
    }
    function armSummary() {
      if (summaryTimer !== null) cancel(summaryTimer);
      const deadlines = [count,rooms].filter(Boolean).flatMap(entry => [entry.mono-monotonic(),entry.wall-wall()]);
      summaryTimer = deadlines.length && Number.isFinite(Math.min(...deadlines))
        ? schedule(checkSummary,Math.max(1,Math.min(...deadlines,2147483647))) : null;
    }
    function clearSummary(reset = false) {
      count = rooms = null; armSummary(); notifyFilters(reset);
    }
    function resetMissingRoom() {
      if (filters.room?.kind !== 'room') return false;
      filters = Object.freeze({...filters,room:null}); roomNotice = ROOM_GONE;
      return true;
    }
    function checkSummary() {
      const roomExpired = expired(rooms), countExpired = expired(count);
      if (countExpired) count = null;
      if (roomExpired) {
        rooms = null;
        // Keep the selection for the refresh only. The view drops the expired
        // label immediately, and a search cannot supply a replacement list.
        if (filters.text !== null) resetMissingRoom();
        void reapplyCaptures();
      }
      if (countExpired || roomExpired) notifyFilters();
      if (countExpired || roomExpired) channels.captures.flush();
      armSummary();
    }
    function capturePage(cursor,page,started) {
      pages.captures = cursor;
      const observed = page?.count_observed_at ?? page?.observed_at, token = page?.event_id;
      const value = page?.matching_count;
      count = page?.scan_complete === true && Number.isSafeInteger(value) && value >= 0 && value <= 4096
        ? token ? timed(value,page.count_expires_at,observed,started,page)
          : value === 0 && Number.isFinite(observed) ? {value,mono:Infinity,wall:Infinity} : null
        : null;
      const options = page?.room_options;
      let resetRoom = false;
      if (token && filters.text === null && Array.isArray(options) && options.length <= 32
          && options.every(room => typeof room === 'string' && room.length && [...room].length <= 128 && !/[\x00-\x1f\x7f]/.test(room))) {
        rooms = options.length ? timed(Object.freeze([...options]),page.room_options_expires_at,observed,started,page) : null;
        if (!rooms?.value.includes(filters.room?.id)) resetRoom = resetMissingRoom();
      } else if (!token && !rooms) resetRoom = resetMissingRoom();
      armSummary(); notifyFilters();
      if (resetRoom) void reapplyCaptures();
    }
    function retireSummary(tokens) {
      const countRetired = tokens.includes(count?.token), roomsRetired = tokens.includes(rooms?.token);
      if (!countRetired && !roomsRetired) return;
      if (countRetired) count = null;
      if (roomsRetired) rooms = null;
      if (roomsRetired && resetMissingRoom()) void reapplyCaptures();
      else { armSummary(); notifyFilters(); }
    }
    const channels = Object.fromEntries(Object.keys(PATHS).map(kind => [kind,channelsAPI.createConversationChannel({
      store:state[kind], fetch, path:PATHS[kind], channel:`voices:${kind}`, loadOnOpen:false, hasView,
      emitAck:message => actions.size===0 && emitAck(message), onUnauthorized,
      monotonic, wall,
      onPage:kind === 'captures' ? capturePage : cursor => { pages[kind] = cursor; },
      onRetire:kind === 'captures' ? retireSummary : undefined,
      retains:kind === 'captures' ? token => rooms?.token === token || count?.token === token : undefined,
      onHttpError:kind === 'captures' ? status => {
        if (status === 409 && queryCursor !== null) void reapplyCaptures();
        if (status === 403) { resetMissingRoom(); clearSummary(); return queryHasSearch; }
        return false;
      } : undefined,
      onState:value => {
        if (kind === 'captures' && value === 'unavailable' && !rooms && resetMissingRoom()) notifyFilters();
        state[kind].setStatus(value); onStatus(value);
      },
    })]));
    function queryCaptures(cursor = null) {
      const text = searchAllowed() ? filters.text : null;
      const body = JSON.stringify({...filters,text,cursor});
      const accepted = channels.captures.refresh(PATHS.captures,{method:'POST',body});
      if (accepted) { queryCursor = cursor; queryHasSearch = text !== null; }
      return accepted;
    }
    async function reapplyCaptures() {
      if (closed || profilesOnly) return false;
      const attempt = ++capturesSequence;
      reapplying = true; delete pages.captures; count = null;
      try {
        do {
          channels.captures.clearView();
          armSummary(); notifyFilters();
          await channels.captures.settled();
          if (closed || attempt !== capturesSequence) return false;
          if (queryCaptures()) return true;
        } while (channels.captures.busy());
        return false;
      } finally { if (attempt === capturesSequence) reapplying = false; }
    }
    async function refresh() {
      const attempt = ++sequence;
      const captureAttempt = ++capturesSequence;
      reapplying = false;
      for (const kind of Object.keys(pages)) delete pages[kind];
      count = null; armSummary(); notifyFilters();
      for (const kind of Object.keys(channels)) channels[kind].clearView();
      await Promise.all(Object.values(channels).map(channel => channel.settled()));
      if (closed || attempt!==sequence) return;
      if (!channels.profiles.refresh(profilePath())) return;
      await channels.profiles.settled();
      if (closed || attempt!==sequence) return;
      if (!profilesOnly && captureAttempt === capturesSequence) queryCaptures();
    }
    async function mutate(path,body,method) {
      if (closed || profilesOnly || actions.size) return false;
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
        for (const channel of Object.values(channels)) channel.flush();
      }
      if (!closed && accepted) await refresh();
      return accepted;
    }
    function clearFilters() {
      capturesSequence++; reapplying = false; queryCursor = null; queryHasSearch = false;
      filters = DEFAULT_FILTERS; roomNotice = ''; clearSummary(true);
    }
    function leave() {
      sequence++; clearFilters();
      for (const kind of Object.keys(pages)) delete pages[kind];
      for (const abort of aborts) abort.abort();
      for (const channel of Object.values(channels)) channel.clearView();
      state.clear();
    }
    if (!profilesOnly) root?.addEventListener?.('pagehide',leave);
    return {
      state,refresh,leave,includePerson,profilesOnly,personId,searchAllowed,
      captureFilters:() => filters,
      captureSummary:() => ({count:expired(count) ? null : count?.value ?? null,
        rooms:expired(rooms) ? [] : rooms?.value ?? [], roomNotice}),
      subscribeFilters(listener) { filterListeners.add(listener); return () => filterListeners.delete(listener); },
      checkFilters:checkSummary,
      setFilters(changes) {
        if (closed || profilesOnly) return false;
        const next = Object.fromEntries(Object.keys(DEFAULT_FILTERS).map(key =>
          [key,Object.hasOwn(changes,key) ? changes[key] : filters[key]]));
        if (!searchAllowed()) next.text = null;
        if (JSON.stringify(next) === JSON.stringify(filters)) return false;
        for (const key of ['room','window']) if (next[key]) next[key] = Object.freeze({...next[key]});
        filters = Object.freeze(next); roomNotice = ''; return reapplyCaptures();
      },
      resetFilters() {
        if (closed || profilesOnly) return;
        clearFilters();
        void reapplyCaptures();
      },
      flush() { for (const channel of Object.values(channels)) channel.flush(); },
      settled:() => Promise.allSettled([...actions,...Object.values(channels).map(channel => channel.settled())]),
      open(identity) { for (const channel of Object.values(channels)) channel.open(identity); if (hasView()) refresh(); },
      async reset() {
        for (const kind of Object.keys(pages)) delete pages[kind];
        sequence++; clearFilters(); for (const abort of aborts) abort.abort(); state.clear();
        await Promise.allSettled([...actions]);
        const done = await Promise.all(Object.values(channels).map(channel => channel.reset()));
        await state.retired(); return done.every(Boolean);
      },
      retire(frame) { for (const channel of Object.values(channels)) channel.retire(frame); },
      review(identity) { return !profilesOnly && CAPTURE_ID.test(identity)
        && channels.review.refresh(PATHS.review+identity+'/review',{variant:'audio',rowId:identity}); },
      transcript(identity) { return !profilesOnly && CAPTURE_ID.test(identity)
        && channels.review.refresh(PATHS.review+identity+'/transcript',{variant:'text',rowId:identity}); },
      closeReview() { channels.review.clearView(); state.review.setStatus('waiting'); },
      next(kind) {
        const cursor = pages[kind];
        if (!cursor) return false;
        if (kind === 'profiles') return channels.profiles.refresh(profilePath(cursor));
        return kind === 'captures' && !profilesOnly && !reapplying && queryCaptures(cursor);
      },
      hasNext:kind => !!pages[kind],
      confirm(identities,name,requestId) { return mutate('/api/voices/profiles',
        { fingerprint_ids:identities, name, request_id:requestId },'POST'); },
      correctGroups(identities,action,requestId) {
        const rows = state.captures.snapshot().filter(row => identities.includes(row.id));
        if (!identities.length || rows.length!==identities.length || new Set(identities).size!==identities.length) return Promise.resolve(false);
        return mutate('/api/voices/groups/corrections',{fingerprint_ids:identities,
          expected_revisions:Object.fromEntries(rows.map(row => [row.group_id,row.group_revision])),action,request_id:requestId},'POST');
      },
      setPerson(identity,userId,profileRevision,linkRevision) {
        if (!includePerson || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(identity)
            || userId !== null && (typeof userId !== 'string' || !userId.length || userId.length > 128 || /[\x00-\x1f\x7f]/.test(userId))
            || !Number.isInteger(profileRevision) || profileRevision < 1 || !Number.isInteger(linkRevision) || linkRevision < 0) return Promise.resolve(false);
        return mutate('/api/voices/profiles/'+identity+'/person', {user_id:userId,
          expected_profile_revision:profileRevision,expected_link_revision:linkRevision},'PUT');
      },
      deleteProfile(identity,revision) { return mutate('/api/voices/profiles/'+identity,{expected_revision:revision},'DELETE'); },
      dispose() { closed = true; root?.removeEventListener?.('pagehide',leave); leave(); for (const channel of Object.values(channels)) channel.dispose(); },
    };
  }
  const api = { createVoiceController, CAPTURE_QUERY_KEYS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaVoiceChannel = api;
})(typeof window === 'undefined' ? null : window);
