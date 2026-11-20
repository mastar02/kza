(function (root) {
  'use strict';
  const channelsAPI = root?.KzaConversationChannel || require('./conversation-channel.js');
  const PATHS = { profiles:'/api/voices/profiles', captures:'/api/voices/captures/query', review:'/api/voices/captures/' };
  const DEFAULT_FILTERS = Object.freeze({room:null,window:null,group_state:null,association:null,listenable_only:false,text:null});
  const CAPTURE_QUERY_KEYS = Object.freeze([...Object.keys(DEFAULT_FILTERS),'cursor']);
  const CAPTURE_ID = /^vf_[0-9a-f]{64}$/;
  const ROOM_GONE = 'La habitación elegida ya no tiene muestras vigentes.';
  function mutationRejection(status) {
    if (status === 401 || status === 403) return 'denied';
    if (status === 409) return 'conflict';
    if (status >= 500 && status !== 503) return 'uncertain';
    return 'unavailable';
  }
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
    let capturePageToken;
    let obsView = false, mutationHold = null;
    const pages = {}, actions = new Set(), aborts = new Set();
    const filterListeners = new Set(), mutationListeners = new Set();
    function setMutationHold(value) {
      mutationHold = value;
      for (const listener of mutationListeners) listener();
    }
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
    function canRefreshRoom(token) {
      // Recheck the selected room once no live rows remain. An older summary
      // cannot invalidate the page or the request that replaced it.
      return filters.room?.kind === 'room' && !channels.captures.busy()
        && (capturePageToken === token || capturePageToken === null)
        && state.captures.snapshot().length === 0;
    }
    function checkSummary() {
      const expiredRooms = expired(rooms) ? rooms : null, countExpired = expired(count);
      if (countExpired) count = null;
      if (expiredRooms) rooms = null;
      if (countExpired || expiredRooms) channels.captures.flush();
      const refreshRoom = expiredRooms && canRefreshRoom(expiredRooms.token);
      if (refreshRoom && filters.text !== null) resetMissingRoom();
      armSummary();
      if (countExpired || expiredRooms) notifyFilters();
      if (refreshRoom) void reapplyCaptures();
    }
    function capturePage(cursor,page,started) {
      pages.captures = cursor;
      const observed = page?.count_observed_at ?? page?.observed_at, token = page?.event_id;
      capturePageToken = token ?? null;
      const value = page?.matching_count;
      count = page?.scan_complete === true && Number.isSafeInteger(value) && value >= 0 && value <= 4096
        ? token ? timed(value,page.count_expires_at,observed,started,page)
          : value === 0 && Number.isFinite(observed) ? {value,mono:Infinity,wall:Infinity} : null
        : null;
      const options = page?.room_options;
      let roomMissing = false;
      if (token && filters.text === null && Array.isArray(options) && options.length <= 32
          && options.every(room => typeof room === 'string' && room.length && [...room].length <= 128 && !/[\x00-\x1f\x7f]/.test(room))) {
        rooms = options.length ? timed(Object.freeze([...options]),page.room_options_expires_at,observed,started,page) : null;
        roomMissing = !rooms?.value.includes(filters.room?.id);
      } else roomMissing = !token && !rooms;
      const resetRoom = roomMissing && state.captures.snapshot().length === 0 && resetMissingRoom();
      armSummary(); notifyFilters();
      if (resetRoom) void reapplyCaptures();
    }
    function retireSummary(tokens) {
      const countRetired = tokens.includes(count?.token), roomsRetired = tokens.includes(rooms?.token);
      const resetRoom = !rooms && tokens.includes(capturePageToken) && resetMissingRoom();
      if (!countRetired && !roomsRetired && !resetRoom) return;
      const refreshRoom = roomsRetired && canRefreshRoom(rooms.token) && resetMissingRoom();
      if (countRetired) count = null;
      if (roomsRetired) rooms = null;
      armSummary(); notifyFilters(); channels.captures.flush();
      if (refreshRoom) void reapplyCaptures();
    }
    // Clear the selector's selected-room fallback before the channel retires
    // the capture ticket and queues its ACK.
    const unsubscribeCaptures = state.captures.subscribe(() => {
      if (state.captures.snapshot().length === 0) notifyFilters();
    });
    const channels = Object.fromEntries(Object.keys(PATHS).map(kind => [kind,channelsAPI.createConversationChannel({
      store:state[kind], fetch, path:PATHS[kind], channel:`voices:${kind}`, loadOnOpen:false, hasView,
      emitAck:message => actions.size===0 && emitAck(message), onUnauthorized,
      monotonic, wall, trackObservedAt:kind === 'captures',
      onPage:kind === 'captures' ? capturePage : cursor => { pages[kind] = cursor; },
      onRetire:kind === 'captures' ? retireSummary : undefined,
      retains:kind === 'captures' ? token => rooms?.token === token || count?.token === token : undefined,
      onHttpError:status => {
        if (kind === 'captures') {
          if (status === 409 && queryCursor !== null) void reapplyCaptures();
          if (status === 403) {
            resetMissingRoom(); clearSummary();
            if (queryHasSearch) return true;
          }
        }
        return status === 403 && obsView;
      },
      onState:value => {
        if (kind === 'captures' && value === 'unavailable' && !rooms && resetMissingRoom()) notifyFilters();
        state[kind].setStatus(value); onStatus(value);
      },
    })]));
    function queryCaptures(cursor = null) {
      const text = searchAllowed() ? filters.text : null;
      const body = JSON.stringify({...filters,text,cursor});
      const accepted = channels.captures.refresh(PATHS.captures,{method:'POST',body});
      if (accepted) { queryCursor = cursor; queryHasSearch = text !== null; capturePageToken = undefined; }
      return accepted;
    }
    async function reapplyCaptures() {
      if (closed || profilesOnly) return false;
      const attempt = ++capturesSequence;
      reapplying = true; delete pages.captures; count = null; capturePageToken = undefined;
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
    async function refresh({ surface } = {}) {
      if (surface === 'obs') obsView = true;
      // An older refresh must never release a hold added while its reads run.
      const hold = mutationHold;
      const attempt = ++sequence;
      const captureAttempt = ++capturesSequence;
      reapplying = false;
      for (const kind of Object.keys(pages)) delete pages[kind];
      count = null; capturePageToken = undefined; armSummary(); notifyFilters();
      for (const kind of Object.keys(channels)) channels[kind].clearView();
      await Promise.all(Object.values(channels).map(channel => channel.settled()));
      if (closed || attempt!==sequence) return;
      if (!channels.profiles.refresh(profilePath())) return;
      await channels.profiles.settled();
      if (closed || attempt!==sequence) return;
      const capturesStarted = !profilesOnly && captureAttempt === capturesSequence && queryCaptures();
      if (!hold) return;
      if (hold.includes('profiles') && state.profiles.status() !== 'real') return;
      if (hold.includes('captures')) {
        if (!capturesStarted) return;
        await channels.captures.settled();
        if (captureAttempt !== capturesSequence || state.captures.status() !== 'real') return;
      }
      if (!closed && attempt === sequence && hold === mutationHold) {
        setMutationHold(null);
      }
    }
    async function mutate(path,body,method,typed,refreshKeys) {
      if (mutationHold) return {state:'uncertain'};
      if (closed || profilesOnly || actions.size) return {state:'unavailable'};
      const abort = new AbortController(); aborts.add(abort);
      let encoded = JSON.stringify(body), response = null;
      const task = (async () => {
        let rejection;
        try {
          response = await fetch(path,{ method, credentials:'same-origin', signal:abort.signal,
            headers:{'Content-Type':'application/json'}, body:encoded });
          if (!response.ok) rejection = mutationRejection(response.status);
          if (response.status===401 || (response.status===403 && !typed)) onUnauthorized();
          // Mutation responses carry status/IDs only. Drain their real body so
          // an error echo or delayed transport cannot outlive action ownership.
          await response.arrayBuffer();
          return {state:rejection ?? (abort.signal.aborted ? 'uncertain' : 'accepted')};
        } catch { return {state:rejection ?? 'uncertain'}; }
        finally { encoded = body = response = null; }
      })();
      actions.add(task);
      let outcome;
      try {
        outcome = await task;
        if (!closed && typed && outcome.state === 'uncertain') setMutationHold(refreshKeys);
      }
      finally {
        actions.delete(task); aborts.delete(abort);
        for (const channel of Object.values(channels)) channel.flush();
      }
      if (!closed && outcome.state === 'accepted') await refresh();
      return outcome;
    }
    function confirm(identities,name,requestId,typed) {
      return mutate('/api/voices/profiles',{fingerprint_ids:identities,name,request_id:requestId},
        'POST',typed,['profiles','captures']);
    }
    function correctGroups(identities,action,requestId,typed) {
      const rows = state.captures.snapshot().filter(row => identities.includes(row.id));
      if (!identities.length || rows.length!==identities.length || new Set(identities).size!==identities.length) {
        return Promise.resolve({state:'unavailable'});
      }
      return mutate('/api/voices/groups/corrections',{fingerprint_ids:identities,
        expected_revisions:Object.fromEntries(rows.map(row => [row.group_id,row.group_revision])),action,request_id:requestId},
        'POST',typed,['captures']);
    }
    function deleteProfile(identity,revision,typed) {
      return mutate('/api/voices/profiles/'+identity,{expected_revision:revision},'DELETE',typed,['profiles']);
    }
    function clearFilters() {
      capturesSequence++; reapplying = false; queryCursor = null; queryHasSearch = false; capturePageToken = undefined;
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
        if (Object.keys(DEFAULT_FILTERS).every(key => filters[key] === DEFAULT_FILTERS[key])) {
          roomNotice = ''; notifyFilters(true);
          return;
        }
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
      // NEXO uses the boolean methods and revalidates its session on both 401 and 403.
      async confirm(identities,name,requestId) { return (await confirm(identities,name,requestId)).state === 'accepted'; },
      async correctGroups(identities,action,requestId) { return (await correctGroups(identities,action,requestId)).state === 'accepted'; },
      async setPerson(identity,userId,profileRevision,linkRevision) {
        if (!includePerson || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(identity)
            || userId !== null && (typeof userId !== 'string' || !userId.length || userId.length > 128 || /[\x00-\x1f\x7f]/.test(userId))
            || !Number.isInteger(profileRevision) || profileRevision < 1 || !Number.isInteger(linkRevision) || linkRevision < 0) return false;
        return (await mutate('/api/voices/profiles/'+identity+'/person', {user_id:userId,
          expected_profile_revision:profileRevision,expected_link_revision:linkRevision},'PUT')).state === 'accepted';
      },
      async deleteProfile(identity,revision) { return (await deleteProfile(identity,revision)).state === 'accepted'; },
      confirmOutcome(identities,name,requestId) { return confirm(identities,name,requestId,true); },
      correctGroupsOutcome(identities,action,requestId) { return correctGroups(identities,action,requestId,true); },
      deleteProfileOutcome(identity,revision) { return deleteProfile(identity,revision,true); },
      mutationHeld:() => mutationHold !== null,
      subscribeMutationHold(listener) { mutationListeners.add(listener); return () => mutationListeners.delete(listener); },
      dispose() {
        closed = true; root?.removeEventListener?.('pagehide',leave);
        leave(); unsubscribeCaptures();
        for (const channel of Object.values(channels)) channel.dispose();
      },
    };
  }
  const api = { createVoiceController, CAPTURE_QUERY_KEYS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaVoiceChannel = api;
})(typeof window === 'undefined' ? null : window);
