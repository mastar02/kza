(function (root) {
  'use strict';
  const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const VF = /^vf_[0-9a-f]{64}$/;
  const OPAQUE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
  const SOURCES = new Set(['live_input', 'consented_recording']);
  const LANES = new Set(['command', 'ambient', 'evaluation']);
  function closed(value, names) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
      && Object.getPrototypeOf(value) === Object.prototype
      && Object.keys(value).length === names.length && names.every(name => Object.hasOwn(value, name));
  }
  function integer(value, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
    // Native counters are int64; the browser admits only exactly representable
    // integers. Unsafe values are rejected, never rounded or made unknown.
    return Number.isSafeInteger(value) && value >= minimum && value <= maximum;
  }
  function revision(value) { return typeof value === 'string' && OPAQUE.test(value); }
  function boundedText(value, maximum, maximumBytes, minimum = 0) {
    if (typeof value !== 'string' || value.length > maximum * 2) return false;
    let count = 0;
    for (const character of value) {
      const code = character.codePointAt(0);
      if (++count > maximum || code >= 0xd800 && code <= 0xdfff) return false;
    }
    return count >= minimum && new TextEncoder().encode(value).length <= maximumBytes;
  }
  function copyProvenance(value) {
    const names = ['source', 'source_ref', 'lane', 'stream_generation', 'segment_generation',
      'source_channels', 'selected_channel', 'sample_rate_hz', 'frame_count', 'dsp'];
    if (!closed(value, names) || !SOURCES.has(value.source) || !revision(value.source_ref)
        || !LANES.has(value.lane) || !integer(value.stream_generation) || !integer(value.segment_generation)
        || !integer(value.source_channels, 1, 64) || !integer(value.selected_channel, 0, value.source_channels - 1)
        || !integer(value.sample_rate_hz, 8000, 192000) || !integer(value.frame_count, 1)
        || !closed(value.dsp, ['certainty', 'declared_revision', 'observed_revision'])) throw Error('invalid capture metadata');
    const dsp = value.dsp, declared = dsp.declared_revision, observed = dsp.observed_revision;
    if (declared !== null && !revision(declared) || observed !== null && !revision(observed)
        || !['unknown', 'declared', 'observed'].includes(dsp.certainty)
        || dsp.certainty === 'unknown' && (declared !== null || observed !== null)
        || dsp.certainty === 'declared' && (declared === null || observed !== null)
        || dsp.certainty === 'observed' && observed === null) throw Error('invalid capture metadata');
    return Object.freeze({ source: value.source, source_ref: value.source_ref, lane: value.lane,
      stream_generation: value.stream_generation, segment_generation: value.segment_generation,
      source_channels: value.source_channels, selected_channel: value.selected_channel,
      sample_rate_hz: value.sample_rate_hz, frame_count: value.frame_count,
      dsp: Object.freeze({ certainty: dsp.certainty, declared_revision: declared, observed_revision: observed }) });
  }
  function copyCaptureMetadata(value, transcript, { previewFrames, durationS } = {}) {
    if (value === null) return null;
    const names = ['schema_version', 'provenance', 'span', ...(transcript ? ['text_truncated'] : [])];
    if (!closed(value, names) || value.schema_version !== 1 || !Number.isInteger(value.schema_version)
        || transcript && value.text_truncated !== null && typeof value.text_truncated !== 'boolean') throw Error('invalid capture metadata');
    let provenance = null, span = null;
    if (value.provenance === null || value.span === null) {
      if (!transcript || value.provenance !== null || value.span !== null || value.text_truncated === null) throw Error('invalid capture metadata');
    } else {
      provenance = copyProvenance(value.provenance);
      if (!closed(value.span, ['start_frame', 'end_frame']) || !integer(value.span.start_frame)
          || !integer(value.span.end_frame, 1, provenance.frame_count)
          || value.span.start_frame >= value.span.end_frame) throw Error('invalid capture metadata');
      span = Object.freeze({ start_frame: value.span.start_frame, end_frame: value.span.end_frame });
      if (span.start_frame !== 0 || span.end_frame !== (transcript ? provenance.frame_count : Math.min(provenance.frame_count, 96000))) throw Error('invalid capture metadata');
      if (!transcript && (provenance.sample_rate_hz !== 16000 || !integer(provenance.frame_count, 12800, 480000)
          || previewFrames !== undefined && (!integer(previewFrames, 12800, 96000) || span.end_frame !== previewFrames)
          || durationS !== undefined && (!Number.isFinite(durationS) || durationS !== provenance.frame_count / 16000))) throw Error('invalid capture metadata');
    }
    const copy = { schema_version: 1, provenance, span, ...(transcript ? { text_truncated: value.text_truncated } : {}) };
    if (new TextEncoder().encode(JSON.stringify(copy)).length > 4096) throw Error('invalid capture metadata');
    return Object.freeze(copy);
  }
  function copyAudioMetadata(value, options) { return copyCaptureMetadata(value, false, options); }
  function copyTranscriptMetadata(value) { return copyCaptureMetadata(value, true); }
  function metadataDetails(kind, value) {
    const text = kind === 'transcript', metadata = text ? copyTranscriptMetadata(value) : copyAudioMetadata(value);
    const label = text ? 'transcripción' : 'audio', lines = [];
    if (metadata?.provenance) {
      const p = metadata.provenance, span = metadata.span;
      const number = value => value.toLocaleString('es-UY');
      lines.push(text ? `Entrada completa del STT: ${number(span.end_frame)} muestras.`
        : `Vista previa: ${number(span.end_frame)} muestras de una entrada de ${number(p.frame_count)} muestras.`);
      const lane = { command: 'comandos', ambient: 'ambiente', evaluation: 'evaluación' }[p.lane];
      lines.push(`Canal ${p.selected_channel} · ${number(p.sample_rate_hz)} Hz · ${lane}.`);
      lines.push({ unknown: 'Estado DSP desconocido.', declared: 'DSP declarado; aplicación sin verificar.', observed: 'Estado DSP observado.' }[p.dsp.certainty]);
    } else lines.push(`Ventana y canal de ${label}: desconocidos.`);
    if (text) lines.push(metadata?.text_truncated === true ? 'El texto fue recortado para revisión.'
      : metadata?.text_truncated === false ? 'Texto conservado sin recorte.' : 'No consta si el texto fue recortado.');
    return lines;
  }
  function createVoiceStore(kind, { monotonic = () => performance.now(), wall = Date.now,
      schedule = setTimeout, cancel = clearTimeout, url = root?.URL || URL, includePerson = false, personId = null } = {}) {
    if (typeof includePerson !== 'boolean' || personId !== null && (!includePerson || typeof personId !== 'string' || !personId.length || personId.length > 128)) throw Error('invalid voice person scope');
    let entries = [], timer = null, privacy = null, media = null, status = 'waiting';
    const listeners = new Set(), retiring = new Map();
    function notify() { for (const listener of listeners) listener(); }
    function retireMedia() {
      if (!media) return;
      const element = media; media = null;
      const needsEvent = element.networkState !== 0 || element.readyState !== 0 || !!element.currentSrc;
      let emptied = false, finished = false, verify = null, finish;
      const pending = new Promise(resolve => { finish = resolve; });
      function check() {
        if (finished) return;
        // Chromium can retain the revoked currentSrc string after load(). The
        // load/emptied contract resets the actual resource, tracks and ranges:
        // https://html.spec.whatwg.org/multipage/media.html#media-element-load-algorithm
        if ((!needsEvent || emptied) && !element.isConnected && element.paused && !element.hasAttribute('src')
            && element.srcObject === null && !element.querySelector('source')
            && element.readyState === 0 && element.networkState === 0
            && element.buffered.length === 0 && Number.isNaN(element.duration)) {
          finished = true;
          if (verify !== null) clearTimeout(verify);
          element.removeEventListener('emptied', onEmpty); finish();
        } else if ((!needsEvent || emptied) && verify === null) {
          // The event and final public media state can settle on different
          // native turns. Observe progress; elapsed time never authorizes ACK.
          verify = setTimeout(() => { verify = null; check(); },5);
        }
      }
      function onEmpty() { emptied = true; check(); }
      element.addEventListener('emptied', onEmpty);
      retiring.set(pending,() => ({needs_event:needsEvent,emptied,connected:element.isConnected,paused:element.paused,
        has_source:element.hasAttribute('src'),has_provider:element.srcObject!==null,has_child:!!element.querySelector('source'),
        ready:element.readyState,network:element.networkState,buffers:element.buffered.length,empty_duration:Number.isNaN(element.duration)}));
      pending.then(() => { retiring.delete(pending); });
      element.pause(); element.removeAttribute('src'); element.remove(); element.load();
      queueMicrotask(check);
    }
    function release(rows) { for (const entry of rows) if (entry.row.audio_url) url.revokeObjectURL(entry.row.audio_url); }
    function clear() {
      retireMedia(); release(entries); entries = [];
      if (timer !== null) cancel(timer); timer = null; notify();
    }
    function prune() {
      const now = monotonic(), clock = wall();
      const expired = entries.filter(entry => now >= entry.mono || clock >= entry.wall);
      if (expired.length) {
        retireMedia(); release(expired);
        entries = entries.filter(entry => !expired.includes(entry)); status = 'expired'; notify();
      }
    }
    function arm() {
      if (timer !== null) cancel(timer); timer = null;
      if (!entries.length) return;
      const delay = Math.min(...entries.map(entry => Math.min(entry.mono-monotonic(),entry.wall-wall())));
      timer = schedule(() => { timer = null; prune(); arm(); },Math.max(1,Math.min(delay,2147483647)));
    }
    function copy(value) {
      if (kind === 'profiles') {
        if (!ID.test(value.id) || typeof value.name !== 'string' || !value.name.trim() || value.name.length>64
            || !Number.isInteger(value.revision) || value.revision<1 || !Number.isFinite(value.confirmed_at)) throw Error('invalid voice profile');
        const row = { id:value.id, name:value.name, revision:value.revision, confirmed_at:value.confirmed_at };
        if (includePerson) {
          const link = value.person_link;
          if (!link || Object.keys(link).sort().join(',') !== 'revision,user_id'
              || personId !== null && link.user_id !== personId
              || !Number.isInteger(link.revision) || link.revision < (link.user_id === null ? 0 : 1)
              || link.user_id !== null && (typeof link.user_id !== 'string' || !link.user_id.length
                || link.user_id.length > 128 || /[\x00-\x1f\x7f]/.test(link.user_id))) throw Error('invalid voice person link');
          row.person_link = Object.freeze({ user_id:link.user_id, revision:link.revision });
        }
        return row;
      }
      if (!VF.test(value.id)) throw Error('invalid voice capture');
      if (kind === 'captures') {
        if (!Number.isFinite(value.duration_s) || value.duration_s<.8 || value.duration_s>30
            || !Number.isFinite(value.created_at) || typeof value.review_available !== 'boolean'
            || value.suggested_profile !== null && !ID.test(value.suggested_profile)
            || value.similarity !== null && (!Number.isFinite(value.similarity) || Math.abs(value.similarity)>1)
            || !ID.test(value.group_id) || !Number.isInteger(value.group_revision) || value.group_revision<1
            || !Number.isInteger(value.group_size) || value.group_size<1 || value.group_size>32
            || !['suggested','manual','unknown'].includes(value.group_state)
            || !['new_voice','pairwise_similarity','short_audio','ambiguous','manual','manual_unknown','unreviewed','prior_group_expired'].includes(value.group_reason)
            || !Number.isFinite(value.group_expires_at) || value.group_expires_at<=value.observed_at
            || value.group_expires_at-value.observed_at>172800
            || value.group_similarity!==null && (!Number.isFinite(value.group_similarity) || Math.abs(value.group_similarity)>1)) throw Error('invalid capture metadata');
        return { id:value.id, duration_s:value.duration_s, created_at:value.created_at, review_available:value.review_available,
          expires_at:Math.min(value.expires_at,value.group_expires_at),
          suggested_profile:value.suggested_profile, similarity:value.similarity, group_id:value.group_id,
          group_revision:value.group_revision, group_size:value.group_size, group_state:value.group_state,
          group_reason:value.group_reason, group_similarity:value.group_similarity, group_expires_at:value.group_expires_at };
      }
      if (kind !== 'review' || !closed(value, ['id', 'observed_at', 'expires_at', 'transcript', 'room_id',
          'duration_s', 'clip_duration_s', 'clip_truncated', 'transcript_scope', 'audio_mime', 'audio_base64',
          'audio_metadata', 'transcript_metadata'])
          || !boundedText(value.transcript, 8192, 32768)
          || value.room_id !== null && !boundedText(value.room_id, 128, 512, 1)
          || value.audio_mime !== 'audio/wav' || value.transcript_scope !== 'full_segment'
          || !Number.isFinite(value.duration_s) || value.duration_s<.8 || value.duration_s>30
          || !Number.isFinite(value.clip_duration_s) || value.clip_duration_s<.8 || value.clip_duration_s>Math.min(value.duration_s,6)
          || typeof value.clip_truncated !== 'boolean' || value.clip_truncated !== (value.clip_duration_s<value.duration_s)
          || typeof value.audio_base64 !== 'string' || value.audio_base64.length>256060
          || !/^[A-Za-z0-9+/]+={0,2}$/.test(value.audio_base64)) throw Error('invalid listening sample');
      const transcriptMetadata = copyTranscriptMetadata(value.transcript_metadata);
      const raw = atob(value.audio_base64), bytes = Uint8Array.from(raw,c => c.charCodeAt(0));
      const view = new DataView(bytes.buffer), text = (offset,size) => String.fromCharCode(...bytes.slice(offset,offset+size));
      if (bytes.length<44 || bytes.length>192044 || text(0,4)!=='RIFF' || text(8,4)!=='WAVE'
          || text(12,4)!=='fmt ' || view.getUint32(16,true)!==16 || view.getUint16(20,true)!==1
          || view.getUint32(4,true)!==bytes.length-8 || view.getUint16(22,true)!==1
          || view.getUint32(24,true)!==16000 || view.getUint32(28,true)!==32000
          || view.getUint16(32,true)!==2 || view.getUint16(34,true)!==16
          || text(36,4)!=='data' || view.getUint32(40,true)!==bytes.length-44
          || (bytes.length-44)%2 || value.clip_duration_s!==(bytes.length-44)/32000) throw Error('invalid listening WAV');
      const audioMetadata = copyAudioMetadata(value.audio_metadata, { previewFrames:(bytes.length-44)/2, durationS:value.duration_s });
      return { id:value.id, transcript:value.transcript, room_id:value.room_id, duration_s:value.duration_s,
        clip_duration_s:value.clip_duration_s, clip_truncated:value.clip_duration_s<value.duration_s,
        audio_metadata:audioMetadata, transcript_metadata:transcriptMetadata,
        audio_url:url.createObjectURL(new Blob([bytes],{type:'audio/wav'})) };
    }
    return {
      receive(rows,started) {
        const next = [];
        try {
          const elapsed = started && Math.max(0,monotonic()-started.mono,wall()-started.wall);
          if (!Array.isArray(rows) || rows.length>(kind==='review'?1:50) || !Number.isFinite(elapsed)) throw Error('invalid voice snapshot');
          const ids = new Set();
          for (const value of rows) {
            if (!value || !Number.isFinite(value.observed_at) || !Number.isFinite(value.expires_at)
                || !Number.isFinite(value.expires_at*1000) || value.expires_at<=value.observed_at
                || value.expires_at-value.observed_at>(kind==='profiles'?300:172800) || ids.has(value.id)) throw Error('invalid voice deadline');
            const deadline = kind==='captures' ? Math.min(value.expires_at,value.group_expires_at) : value.expires_at;
            if (!Number.isFinite(deadline)) throw Error('invalid voice group deadline');
            const budget = (deadline-value.observed_at)*1000-elapsed;
            if (budget<=0 || deadline*1000<=wall()) continue;
            const row = copy(value); ids.add(row.id);
            next.push({ row:Object.freeze(row), mono:monotonic()+budget, wall:Math.min(deadline*1000,wall()+budget) });
          }
          clear(); entries = next; status = 'real'; prune(); arm(); notify(); return true;
        } catch { release(next); clear(); status = 'unavailable'; notify(); return false; }
      },
      clear, check() { prune(); arm(); }, snapshot() { prune(); return entries.map(entry => entry.row); },
      status:() => status, setStatus(value) { status = value; notify(); },
      attachMedia(element) { retireMedia(); media = element; }, detachMedia:retireMedia,
      pendingRetirements:() => retiring.size, retired:() => Promise.allSettled([...retiring.keys()]),
      retirementStatus:() => [...retiring.values()].map(snapshot => snapshot()),
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      bindPrivacy(value) { privacy = value; }, committed(token) { privacy?.committed(token); },
      deliveryToken:() => privacy?.token() ?? null,
    };
  }
  function createVoiceState(options) {
    const state = Object.fromEntries(['profiles','captures','review'].map(kind => [kind,createVoiceStore(kind,options)]));
    return Object.assign(state,{ clear() { for (const key of ['profiles','captures','review']) state[key].clear(); },
      retired:() => Promise.all(['profiles','captures','review'].map(key => state[key].retired())) });
  }
  const api = { createVoiceStore,createVoiceState,copyAudioMetadata,copyTranscriptMetadata,metadataDetails };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.KzaVoicePrivacy = api;
})(typeof window === 'undefined' ? null : window);
