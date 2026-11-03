'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const privacy = require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
const { copyAudioMetadata, copyTranscriptMetadata, metadataDetails, createVoiceStore } = privacy;
const FP = 'vf_' + 'a'.repeat(64);
function provenance(overrides = {}) {
  return { source: 'live_input', source_ref: 'private-source-canary', lane: 'ambient',
    stream_generation: 7, segment_generation: 11, source_channels: 2, selected_channel: 0,
    sample_rate_hz: 16000, frame_count: 12800,
    dsp: { certainty: 'declared', declared_revision: 'private-dsp-canary', observed_revision: null }, ...overrides };
}
function audioMetadata(overrides = {}) {
  return { schema_version: 1, provenance: provenance(), span: { start_frame: 0, end_frame: 12800 }, ...overrides };
}
function textMetadata(overrides = {}) {
  return { schema_version: 1, provenance: provenance({ selected_channel: 1, frame_count: 160001 }),
    span: { start_frame: 0, end_frame: 160001 }, text_truncated: false, ...overrides };
}
function clone(value) { return JSON.parse(JSON.stringify(value)); }
function wav(frames = 12800) {
  const size = frames * 2, value = Buffer.alloc(size + 44);
  value.write('RIFF', 0); value.writeUInt32LE(size + 36, 4); value.write('WAVEfmt ', 8);
  value.writeUInt32LE(16, 16); value.writeUInt16LE(1, 20); value.writeUInt16LE(1, 22);
  value.writeUInt32LE(16000, 24); value.writeUInt32LE(32000, 28); value.writeUInt16LE(2, 32);
  value.writeUInt16LE(16, 34); value.write('data', 36); value.writeUInt32LE(size, 40);
  return value.toString('base64');
}
function review() {
  return { id: FP, observed_at: 1000, expires_at: 1100, transcript: 'Synthetic full-window text', room_id: null,
    duration_s: .8, clip_duration_s: .8, clip_truncated: false, transcript_scope: 'full_segment',
    audio_mime: 'audio/wav', audio_base64: wav(), audio_metadata: audioMetadata(), transcript_metadata: textMetadata() };
}
function store() {
  const allocated = [], revoked = [];
  return { allocated, revoked, value: createVoiceStore('review', { monotonic: () => 0, wall: () => 1000000,
    schedule: () => 1, cancel() {}, url: { createObjectURL() { const url = 'blob:' + allocated.length; allocated.push(url); return url; },
      revokeObjectURL(value) { revoked.push(value); } } }) };
}
const started = { mono: 0, wall: 1000000 };

test('legacy NULL is explicit unknown, including independently known text truncation', () => {
  assert.equal(copyAudioMetadata(null), null); assert.equal(copyTranscriptMetadata(null), null);
  for (const value of [undefined, '', false, {}, []]) {
    assert.throws(() => copyAudioMetadata(value)); assert.throws(() => copyTranscriptMetadata(value));
  }
  for (const text_truncated of [false, true]) {
    const metadata = copyTranscriptMetadata({ schema_version: 1, provenance: null, span: null, text_truncated });
    assert.equal(metadata.text_truncated, text_truncated);
  }
  assert.throws(() => copyAudioMetadata({ schema_version: 1, provenance: null, span: null }));
  assert.throws(() => copyTranscriptMetadata({ schema_version: 1, provenance: null, span: null, text_truncated: null }));
});
test('metadata is copied and frozen without inferring common audio and STT windows', () => {
  const input = audioMetadata(), audio = copyAudioMetadata(input, { previewFrames: 12800, durationS: .8 });
  const text = copyTranscriptMetadata(textMetadata({ text_truncated: true }));
  input.provenance.source_ref = 'changed'; input.provenance.dsp.declared_revision = 'changed';
  assert.equal(audio.provenance.source_ref, 'private-source-canary');
  assert.equal(audio.provenance.dsp.declared_revision, 'private-dsp-canary');
  assert.ok(Object.isFrozen(audio) && Object.isFrozen(audio.span) && Object.isFrozen(audio.provenance.dsp));
  assert.equal(text.span.end_frame, 160001); assert.equal(text.provenance.selected_channel, 1);
  assert.equal(text.text_truncated, true); assert.equal(audio.span.end_frame, 12800);
});
test('versions, extra keys and malformed null pairs fail closed at every nested level', () => {
  for (const version of [0, 2, true, '1', null]) {
    assert.throws(() => copyAudioMetadata(audioMetadata({ schema_version: version })));
  }
  for (const mutate of [value => { value.extra = null; }, value => { value.provenance.extra = null; },
    value => { value.provenance.dsp.extra = null; }, value => { value.span.extra = null; },
    value => { delete value.provenance.stream_generation; }, value => { value.span = null; },
    value => { value.provenance = null; }]) {
    const value = audioMetadata(); mutate(value); assert.throws(() => copyAudioMetadata(value));
  }
  for (const value of [0, 1, 'false', {}, []]) assert.throws(() => copyTranscriptMetadata(textMetadata({ text_truncated: value })));
});
test('all native scalar and DSP invariants are strict and bounded', () => {
  const changes = [ { source: 'room' }, { lane: 'speaker' }, { source_ref: 'unsafe/source' },
    { source_ref: 'a'.repeat(129) }, { source_ref: '\ud800' }, { stream_generation: true },
    { segment_generation: -1 }, { frame_count: 1.5 }, { frame_count: Number.MAX_SAFE_INTEGER + 1 },
    { source_channels: 65 }, { selected_channel: 2 }, { sample_rate_hz: 7999 },
    { dsp: { certainty: 'unknown', declared_revision: 'intent', observed_revision: null } },
    { dsp: { certainty: 'declared', declared_revision: 'intent', observed_revision: 'actual' } },
    { dsp: { certainty: 'observed', declared_revision: null, observed_revision: null } } ];
  for (const change of changes) assert.throws(() => copyAudioMetadata(audioMetadata({ provenance: provenance(change) })));
});
test('half-open spans and actual WAV frame counts cannot be rounded into agreement', () => {
  for (const span of [{ start_frame: 1, end_frame: 12800 }, { start_frame: 0, end_frame: 12799 },
    { start_frame: 0, end_frame: 12801 }, { start_frame: false, end_frame: 12800 }]) {
    assert.throws(() => copyAudioMetadata(audioMetadata({ span })));
  }
  assert.throws(() => copyAudioMetadata(audioMetadata(), { previewFrames: 12801 }));
  assert.throws(() => copyAudioMetadata(audioMetadata(), { durationS: .8 + .25 / 16000 }));
  assert.throws(() => copyTranscriptMetadata(textMetadata({ span: { start_frame: 0, end_frame: 12800 } })));
});
test('operator store keeps independent metadata and clears it with its Blob', () => {
  const g = store(); assert.equal(g.value.receive([review()], started), true);
  const [row] = g.value.snapshot();
  assert.equal(row.audio_metadata.provenance.selected_channel, 0);
  assert.equal(row.transcript_metadata.provenance.selected_channel, 1);
  assert.equal(row.transcript_metadata.span.end_frame, 160001);
  g.value.clear(); assert.deepEqual(g.value.snapshot(), []); assert.deepEqual(g.revoked, g.allocated);
});
test('operator rejects missing/foreign metadata and mismatched canonical WAV before Blob allocation', () => {
  for (const mutate of [value => { delete value.audio_metadata; }, value => { value.extra = null; },
    value => { value.audio_metadata = {}; }, value => { value.audio_base64 = wav(12801); value.clip_duration_s = value.duration_s = 12801 / 16000; },
    value => { value.clip_duration_s += .25 / 16000; }]) {
    const g = store(), value = review(); mutate(value);
    assert.equal(g.value.receive([value], started), false);
    assert.deepEqual(g.allocated, []); assert.deepEqual(g.value.snapshot(), []); g.value.clear();
  }
});
test('operator text uses bounded Unicode code points with native UTF8 limits', () => {
  for (const [transcript, room_id] of [['', null], ['😀'.repeat(8192), '😀'.repeat(128)],
    ['x'.repeat(8191) + '\0', '\0'], ['😀', 'x'.repeat(128)]]) {
    const g = store(), value = { ...review(), transcript, room_id };
    assert.equal(g.value.receive([value], started), true);
    const [row] = g.value.snapshot();
    assert.equal(row.transcript, transcript); assert.equal(row.room_id, room_id); g.value.clear();
  }
  for (const [transcript, room_id] of [['😀'.repeat(8193), null], ['x'.repeat(16385), null],
    ['\ud800', null], ['\udfff', null], ['x', ''], ['x', '😀'.repeat(129)],
    ['x', 'x'.repeat(257)], ['x', '\ud800'], ['x', '\udfff'], ['x', 1]]) {
    const g = store(), value = { ...review(), transcript, room_id };
    assert.equal(g.value.receive([value], started), false);
    assert.deepEqual(g.allocated, []); assert.deepEqual(g.value.snapshot(), []); g.value.clear();
  }
});
test('user-facing details preserve unknown/truncation without showing opaque IDs or implying DSP observation', () => {
  const lines = [...metadataDetails('audio', audioMetadata()), ...metadataDetails('transcript', textMetadata({ text_truncated: true }))];
  assert.ok(lines.some(value => value.includes('recortado')));
  assert.ok(lines.some(value => value.includes('sin verificar')));
  assert.ok(!lines.join(' ').includes('private-source-canary') && !lines.join(' ').includes('private-dsp-canary'));
  assert.ok(metadataDetails('transcript', null).some(value => value.includes('No consta')));
  assert.ok(metadataDetails('audio', null).some(value => value.includes('desconocidos')));
});
