// =============================================================
// Mock data — KZA dashboard
// All data here mimics the contract described in the brief.
// To wire to real backend: replace these constants with fetch()s
// against the /api/* routes; shapes already match.
// =============================================================

const ZONES = [
  {
    id: 'sala', name: 'Sala', icon: '◰',
    user: { id: 'u_marco', name: 'Marco', present: true, ble_rssi: -52 },
    lastUtterance: { text: 'subí dos grados la calefacción', ts: '14:32:08', user: 'Marco' },
    audioState: 'speaking',
    volume: 42,
    ma1260_zone: 'A',
    spotify: { playing: true, song: 'Sun Setters', artist: 'Khruangbin', progress: 142, total: 218 },
  },
  {
    id: 'dormitorio', name: 'Dormitorio', icon: '▢',
    user: { id: 'u_lu', name: 'Lucía', present: true, ble_rssi: -64 },
    lastUtterance: { text: 'apagá la luz del techo', ts: '14:28:55', user: 'Lucía' },
    audioState: 'idle',
    volume: 18,
    ma1260_zone: 'B',
    spotify: { playing: false },
  },
  {
    id: 'cocina', name: 'Cocina', icon: '◐',
    user: null,
    lastUtterance: { text: 'temporizador de quince minutos', ts: '13:55:12', user: 'Marco' },
    audioState: 'idle',
    volume: 0,
    ma1260_zone: 'C',
    spotify: { playing: false },
  },
  {
    id: 'estudio', name: 'Estudio', icon: '◧',
    user: { id: 'u_marco', name: 'Marco', present: false, ble_rssi: null },
    lastUtterance: { text: 'cuántos pull requests abiertos tengo', ts: '12:11:41', user: 'Marco' },
    audioState: 'listening',
    volume: 30,
    ma1260_zone: 'D',
    spotify: { playing: true, song: 'Lost in the Dream', artist: 'The War on Drugs', progress: 88, total: 354 },
  },
  {
    id: 'patio', name: 'Patio', icon: '◔',
    user: null,
    lastUtterance: null,
    audioState: 'idle',
    volume: 0,
    ma1260_zone: 'E',
    spotify: { playing: false },
  },
];

// 5min latency sparkline data (fast path) — values in ms
const LATENCY_FAST = {
  p50: [180, 192, 188, 175, 201, 195, 188, 192, 210, 188, 195, 182, 178, 188, 195, 200, 198, 192, 188, 195, 188, 184, 192, 198, 195, 188, 192, 200, 195, 188],
  p95: [245, 260, 252, 248, 270, 265, 258, 262, 280, 258, 265, 250, 248, 260, 265, 272, 268, 262, 258, 265, 258, 254, 262, 268, 265, 258, 262, 270, 265, 258],
  p99: [288, 305, 295, 292, 318, 310, 302, 308, 325, 302, 308, 295, 292, 305, 308, 318, 312, 308, 302, 308, 302, 298, 308, 312, 308, 302, 308, 318, 308, 302],
};

const LATENCY_BREAKDOWN = [
  { stage: 'Wake',   ms: 12,  pct: 6  },
  { stage: 'STT',    ms: 78,  pct: 41 },
  { stage: 'Router', ms: 32,  pct: 17 },
  { stage: 'Vector', ms: 18,  pct: 9  },
  { stage: 'HA',     ms: 22,  pct: 11 },
  { stage: 'TTS',    ms: 30,  pct: 16 },
];

const WAKE_EVENTS_24H = {
  total: 142,
  false_positive: 6,
  last_trigger: { ts: '14:32:08', confidence: 0.91, zone: 'sala' },
  hourly: [4,2,1,0,0,1,3,8,11,9,7,6,8,12,14,11,9,7,6,5,8,4,3,3],
};

const LLM_ENDPOINTS = [
  {
    id: 'vllm-7b',     name: 'vLLM Qwen 7B AWQ',
    url: 'http://localhost:8100', priority: 1,
    role: 'fast / routing',
    state: 'healthy',
    tps: 82, ttft_ms: 142,
    last_check: '14:32:31',
    failures_7d: { timeout: 1, billing: 0, rate_limit: 0, idle: 2 },
    cooldown_ends: null,
  },
  {
    id: 'ik-30b-cpu',  name: 'ik_llama Qwen3-30B-A3B',
    url: 'http://localhost:8200', priority: 2,
    role: 'slow / reasoning',
    state: 'cooldown',
    tps: 63, ttft_ms: 1840,
    last_check: '14:30:02',
    failures_7d: { timeout: 4, billing: 0, rate_limit: 0, idle: 0 },
    cooldown_ends: '14:35:02',
    cooldown_step: '5m',
  },
  {
    id: 'embed-bge',   name: 'BGE-M3 (cuda:0)',
    url: 'local',     priority: 0,
    role: 'embeddings',
    state: 'healthy',
    tps: null, ttft_ms: 8,
    last_check: '14:32:30',
    failures_7d: { timeout: 0, billing: 0, rate_limit: 0, idle: 0 },
    cooldown_ends: null,
  },
];

const COOLDOWN_HISTORY_7D = [
  { day: 'Mié', count: 0 }, { day: 'Jue', count: 1 },
  { day: 'Vie', count: 0 }, { day: 'Sáb', count: 2 },
  { day: 'Dom', count: 0 }, { day: 'Lun', count: 1 },
  { day: 'Mar', count: 3 },
];

const CONVERSATIONS = [
  {
    id: 'turn_8821', ts: '14:32:08', user: 'Marco', zone: 'sala',
    path: 'fast', stt: 'subí dos grados la calefacción',
    intent: 'climate.set_temperature', target: 'climate.living_room',
    args: { temperature: 22.5 },
    tts: 'Subí la sala a veintidós con cinco.',
    latency_ms: 192, success: true,
  },
  {
    id: 'turn_8820', ts: '14:28:55', user: 'Lucía', zone: 'dormitorio',
    path: 'fast', stt: 'apagá la luz del techo',
    intent: 'light.turn_off', target: 'light.bedroom_ceiling',
    args: {},
    tts: 'Listo.',
    latency_ms: 178, success: true,
  },
  {
    id: 'turn_8819', ts: '14:21:02', user: 'Marco', zone: 'estudio',
    path: 'slow', stt: 'cuántos pull requests abiertos tengo en el repo de kza',
    intent: 'reasoning.dev_query', target: null,
    args: { query: 'github_open_prs', repo: 'kza' },
    tts: 'Tenés ocho pull requests abiertos: cuatro tuyos y cuatro de revisión pendiente.',
    latency_ms: 12480, success: true,
  },
  {
    id: 'turn_8818', ts: '14:15:33', user: 'Marco', zone: 'sala',
    path: 'music', stt: 'poné algo tranquilo en la sala',
    intent: 'music.play_mood', target: 'media_player.living_room',
    args: { mood: 'chill', zone: 'A' },
    tts: 'Poniendo Khruangbin en la sala.',
    latency_ms: 312, success: true,
  },
  {
    id: 'turn_8817', ts: '13:55:12', user: 'Marco', zone: 'cocina',
    path: 'fast', stt: 'temporizador de quince minutos',
    intent: 'timer.start', target: 'timer.kitchen',
    args: { duration: 900 },
    tts: 'Quince minutos en la cocina.',
    latency_ms: 168, success: true,
  },
  {
    id: 'turn_8816', ts: '13:42:08', user: 'Lucía', zone: 'dormitorio',
    path: 'fast', stt: 'cerrá las persianas',
    intent: 'cover.close', target: 'cover.bedroom_blinds',
    args: {},
    tts: 'Listo.',
    latency_ms: 205, success: true,
  },
  {
    id: 'turn_8815', ts: '13:30:55', user: 'Marco', zone: 'sala',
    path: 'fast', stt: 'pausá la música',
    intent: 'music.pause', target: 'media_player.living_room',
    args: {},
    tts: '', // no TTS, ack-only
    latency_ms: 142, success: true,
  },
  {
    id: 'turn_8814', ts: '13:18:21', user: 'Lucía', zone: 'cocina',
    path: 'fast', stt: 'qué temperatura hace afuera',
    intent: 'sensor.read', target: 'sensor.outdoor_temp',
    args: {},
    tts: 'Afuera dieciocho grados.',
    latency_ms: 188, success: true,
  },
  {
    id: 'turn_8813', ts: '12:58:07', user: 'Marco', zone: 'estudio',
    path: 'slow', stt: 'resumime los cambios del último commit',
    intent: 'reasoning.dev_query', target: null,
    args: { query: 'git_last_commit' },
    tts: 'Refactor del LLMRouter: cooldown exponencial con persistencia y nuevo health check pasivo.',
    latency_ms: 8920, success: true,
  },
  {
    id: 'turn_8812', ts: '12:41:18', user: 'Marco', zone: 'sala',
    path: 'fast', stt: 'apagá todo',
    intent: 'scene.activate', target: 'scene.away',
    args: {},
    tts: 'Apagando todo en la sala.',
    latency_ms: 224, success: false, error: 'HA timeout en cover.living_blind_2',
  },
];

const HA_ENTITIES = [
  { id: 'light.living_room_ceiling',   domain: 'light',   name: 'Luz techo sala',         state: 'on',  attrs: { brightness: 180, color: '#fff2c2' }, score: 0.94, lastSeen: '14:32' },
  { id: 'light.bedroom_ceiling',       domain: 'light',   name: 'Luz techo dormitorio',   state: 'off', attrs: { brightness: 0 }, score: 0.92, lastSeen: '14:28' },
  { id: 'climate.living_room',         domain: 'climate', name: 'Calefacción sala',       state: 'heat', attrs: { current: 22.4, target: 22.5 }, score: 0.97, lastSeen: '14:32' },
  { id: 'climate.bedroom',             domain: 'climate', name: 'Calefacción dormitorio', state: 'off', attrs: { current: 19.8, target: 19 }, score: 0.88, lastSeen: '14:00' },
  { id: 'cover.bedroom_blinds',        domain: 'cover',   name: 'Persianas dormitorio',   state: 'closed', attrs: { position: 0 }, score: 0.91, lastSeen: '13:42' },
  { id: 'cover.living_blind_1',        domain: 'cover',   name: 'Persiana sala 1',        state: 'open', attrs: { position: 100 }, score: 0.86, lastSeen: '12:41' },
  { id: 'cover.living_blind_2',        domain: 'cover',   name: 'Persiana sala 2',        state: 'unavailable', attrs: {}, score: 0.79, lastSeen: '11:02' },
  { id: 'media_player.living_room',    domain: 'media',   name: 'Sonos sala',             state: 'playing', attrs: { volume: 42, source: 'spotify' }, score: 0.95, lastSeen: '14:30' },
  { id: 'media_player.studio',         domain: 'media',   name: 'Sonos estudio',          state: 'playing', attrs: { volume: 30 }, score: 0.93, lastSeen: '14:31' },
  { id: 'sensor.outdoor_temp',         domain: 'sensor',  name: 'Temp exterior',          state: '18.2', attrs: { unit: '°C' }, score: 0.89, lastSeen: '14:32' },
  { id: 'sensor.power_total',          domain: 'sensor',  name: 'Consumo total',          state: '842', attrs: { unit: 'W' }, score: 0.83, lastSeen: '14:32' },
  { id: 'lock.front_door',             domain: 'lock',    name: 'Puerta de entrada',      state: 'locked', attrs: {}, score: 0.96, lastSeen: '13:14' },
  { id: 'binary_sensor.motion_living', domain: 'sensor',  name: 'Movimiento sala',        state: 'on', attrs: {}, score: 0.85, lastSeen: '14:32' },
  { id: 'switch.heater_balcony',       domain: 'switch',  name: 'Calefactor balcón',      state: 'off', attrs: {}, score: 0.72, lastSeen: '08:21' },
];

const HA_ACTIONS = [
  { id: 'act_4421', ts: '14:32:09', idem: 'idm_8821_climate', user: 'Marco', service: 'climate.set_temperature', target: 'climate.living_room', args: '{temperature:22.5}', ok: true, lat_ms: 22 },
  { id: 'act_4420', ts: '14:28:56', idem: 'idm_8820_light',   user: 'Lucía', service: 'light.turn_off',          target: 'light.bedroom_ceiling', args: '{}', ok: true, lat_ms: 14 },
  { id: 'act_4419', ts: '14:15:33', idem: 'idm_8818_music',   user: 'Marco', service: 'music_assistant.play_media', target: 'media_player.living_room', args: '{mood:chill}', ok: true, lat_ms: 88 },
  { id: 'act_4418', ts: '13:42:09', idem: 'idm_8816_cover',   user: 'Lucía', service: 'cover.close_cover',       target: 'cover.bedroom_blinds', args: '{}', ok: true, lat_ms: 18 },
  { id: 'act_4417', ts: '13:30:56', idem: 'idm_8815_music',   user: 'Marco', service: 'media_player.media_pause', target: 'media_player.living_room', args: '{}', ok: true, lat_ms: 9 },
  { id: 'act_4416', ts: '12:41:19', idem: 'idm_8812_scene',   user: 'Marco', service: 'scene.activate',         target: 'scene.away', args: '{}', ok: false, lat_ms: 5012, err: 'cover.living_blind_2 unavailable' },
  { id: 'act_4415', ts: '12:18:42', idem: 'idm_8810_lock',    user: 'Marco', service: 'lock.lock',              target: 'lock.front_door', args: '{}', ok: true, lat_ms: 188 },
];

const USERS = [
  {
    id: 'u_marco', name: 'Marco', samples: 24, lastEnroll: '2026-03-12',
    emotions: { neutral: 62, happy: 18, focused: 14, frustrated: 4, tired: 2 },
    topCommands: [
      { cmd: 'climate.set_temperature', n: 412 },
      { cmd: 'music.play_mood', n: 188 },
      { cmd: 'reasoning.dev_query', n: 142 },
      { cmd: 'light.turn_off', n: 98 },
      { cmd: 'scene.activate', n: 64 },
    ],
    permissions: { climate: true, lights: true, security: true, music: true, scenes: true },
    pca: [
      [-0.4, 0.2], [-0.42, 0.18], [-0.38, 0.22], [-0.36, 0.15],
      [-0.45, 0.25], [-0.41, 0.21], [-0.39, 0.19], [-0.43, 0.17],
    ],
  },
  {
    id: 'u_lu', name: 'Lucía', samples: 18, lastEnroll: '2026-04-02',
    emotions: { neutral: 58, happy: 22, calm: 14, tired: 4, frustrated: 2 },
    topCommands: [
      { cmd: 'light.turn_off', n: 245 },
      { cmd: 'cover.close', n: 142 },
      { cmd: 'climate.set_temperature', n: 88 },
      { cmd: 'sensor.read', n: 64 },
      { cmd: 'timer.start', n: 38 },
    ],
    permissions: { climate: true, lights: true, security: false, music: true, scenes: true },
    pca: [
      [0.32, -0.18], [0.34, -0.16], [0.30, -0.20], [0.36, -0.14],
      [0.33, -0.19], [0.35, -0.17],
    ],
  },
  {
    id: 'u_guest', name: 'Invitado', samples: 4, lastEnroll: '2026-04-21',
    emotions: { neutral: 80, happy: 12, tired: 8 },
    topCommands: [
      { cmd: 'light.turn_on', n: 12 },
      { cmd: 'sensor.read', n: 8 },
    ],
    permissions: { climate: false, lights: true, security: false, music: true, scenes: false },
    pca: [
      [0.05, 0.42], [0.08, 0.45], [0.04, 0.40],
    ],
  },
];

const ALERTS = [
  { id: 'al_201', ts: '14:31:55', priority: 'warn',     type: 'pattern',  zone: 'sala',        title: 'Temperatura objetivo cambiada 4 veces en 2min', body: 'Marco modificó climate.living_room múltiples veces. Posible loop.', acked: false },
  { id: 'al_200', ts: '14:18:21', priority: 'critical', type: 'security', zone: 'patio',       title: 'Movimiento detectado con todos ausentes', body: 'binary_sensor.motion_patio activo. Cámara: ningún rostro reconocido.', acked: false },
  { id: 'al_199', ts: '13:55:08', priority: 'info',     type: 'device',   zone: 'sala',        title: 'cover.living_blind_2 unavailable >2h', body: 'Última heartbeat ZHA: 11:02. Reintentando reconexión.', acked: false },
  { id: 'al_198', ts: '12:41:19', priority: 'warn',     type: 'device',   zone: 'sala',        title: 'Escena "away" parcialmente fallida', body: '1 de 8 entidades no respondió (cover.living_blind_2). Idempotency: idm_8812_scene.', acked: true },
  { id: 'al_197', ts: '11:08:42', priority: 'info',     type: 'pattern',  zone: 'estudio',     title: 'Slow path 30B → cooldown 1m', body: 'Timeout consecutivo nro 1. Próximo step: 5m.', acked: true },
  { id: 'al_196', ts: '08:12:00', priority: 'warn',     type: 'device',   zone: 'cocina',      title: 'Mic ReSpeaker XVF3800 reinició', body: 'USB renegociation. 2 wake-words perdidos.', acked: true },
];

const GPUS = [
  {
    id: 0, name: 'RTX 3070 (cuda:0)', role: 'STT • TTS • SpeakerID • Emotion • Embeddings',
    util: 64, vramUsed: 7.4, vramTotal: 8, temp: 71, power: 168,
    procs: [
      { name: 'whisper-v3-turbo',   vram: 2.8 },
      { name: 'kokoro-tts',         vram: 1.4 },
      { name: 'ecapa-speakerid',    vram: 0.8 },
      { name: 'wav2vec2-emotion',   vram: 1.2 },
      { name: 'bge-m3-embeddings',  vram: 1.2 },
    ],
  },
  {
    id: 1, name: 'RTX 3070 (cuda:1)', role: 'vLLM Qwen 7B AWQ (compartido)',
    util: 38, vramUsed: 6.1, vramTotal: 8, temp: 64, power: 142,
    procs: [
      { name: 'vllm-qwen-7b-awq', vram: 6.1 },
    ],
  },
];

const SERVICES = [
  { name: 'kza-voice',         status: 'active', uptime: '6d 14h', mem: '2.1 GB', cpu: '8%',  pid: 18221 },
  { name: 'kza-llm-ik',        status: 'active', uptime: '6d 14h', mem: '38.4 GB', cpu: '64%', pid: 18244 },
  { name: 'vllm-shared',       status: 'active', uptime: '12d 02h', mem: '6.1 GB', cpu: '12%', pid: 9802 },
  { name: 'chromadb',          status: 'active', uptime: '12d 02h', mem: '1.2 GB', cpu: '2%',  pid: 9821 },
  { name: 'home-assistant',    status: 'active', uptime: '21d 08h', mem: '684 MB',  cpu: '3%',  pid: 4112 },
  { name: 'ma1260-bridge',     status: 'restarting', uptime: '2m 04s', mem: '88 MB',   cpu: '1%',  pid: 31882 },
];

const LOG_LINES = [
  { ts: '14:32:31', lvl: 'INFO', svc: 'router',     msg: 'health_check vllm-7b OK ttft=142ms' },
  { ts: '14:32:08', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8821 fast lat=192ms zone=sala user=u_marco' },
  { ts: '14:32:08', lvl: 'OK',   svc: 'ha',         msg: 'climate.set_temperature target=22.5 idem=idm_8821_climate' },
  { ts: '14:30:02', lvl: 'WARN', svc: 'router',     msg: 'ik-30b-cpu timeout (4400ms) → cooldown 5m' },
  { ts: '14:28:55', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8820 fast lat=178ms zone=dormitorio user=u_lu' },
  { ts: '14:21:02', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8819 slow lat=12480ms zone=estudio user=u_marco' },
  { ts: '14:15:33', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8818 music lat=312ms zone=sala' },
  { ts: '14:08:42', lvl: 'INFO', svc: 'speakerid',  msg: 'enrolled embedding for u_lu sample #18 (sim=0.94)' },
  { ts: '14:02:11', lvl: 'WARN', svc: 'wake',       msg: 'false positive zone=cocina conf=0.62' },
  { ts: '13:55:12', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8817 fast lat=168ms zone=cocina user=u_marco' },
  { ts: '13:42:08', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8816 fast lat=205ms zone=dormitorio user=u_lu' },
  { ts: '13:30:55', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8815 fast lat=142ms zone=sala' },
  { ts: '13:18:21', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8814 fast lat=188ms zone=cocina user=u_lu' },
  { ts: '12:58:07', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8813 slow lat=8920ms zone=estudio user=u_marco' },
  { ts: '12:41:19', lvl: 'ERR',  svc: 'ha',         msg: 'scene.activate FAIL cover.living_blind_2 unavailable idem=idm_8812_scene' },
  { ts: '12:41:18', lvl: 'INFO', svc: 'pipeline',   msg: 'turn_8812 fast lat=224ms zone=sala user=u_marco' },
  { ts: '12:18:42', lvl: 'OK',   svc: 'ha',         msg: 'lock.lock front_door idem=idm_8810_lock' },
  { ts: '11:08:42', lvl: 'WARN', svc: 'router',     msg: 'ik-30b-cpu cooldown 1m (consecutive_failures=1)' },
  { ts: '08:12:00', lvl: 'WARN', svc: 'voice',      msg: 'ReSpeaker XVF3800 USB renegotiation, 2 wake-words lost' },
  { ts: '07:00:00', lvl: 'INFO', svc: 'system',     msg: 'daily heartbeat OK gpus=2 services=6/6' },
];

const PIPELINE_STAGES = [
  { id: 'wake',   label: 'Wake',     hw: 'CPU',     ms: 12 },
  { id: 'stt',    label: 'STT',      hw: 'cuda:0',  ms: 78 },
  { id: 'router', label: 'Router',   hw: 'cuda:1',  ms: 32 },
  { id: 'action', label: 'Action',   hw: 'HA REST', ms: 40 },
  { id: 'tts',    label: 'TTS',      hw: 'cuda:0',  ms: 30 },
];

window.MOCKS = {
  ZONES, LATENCY_FAST, LATENCY_BREAKDOWN, WAKE_EVENTS_24H,
  LLM_ENDPOINTS, COOLDOWN_HISTORY_7D,
  CONVERSATIONS, HA_ENTITIES, HA_ACTIONS,
  USERS, ALERTS, GPUS, SERVICES, LOG_LINES, PIPELINE_STAGES,
};
