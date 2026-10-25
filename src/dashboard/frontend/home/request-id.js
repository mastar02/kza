(function (host) {
  'use strict';
  function newRequestId() {
    if (typeof host.crypto?.randomUUID === 'function') return host.crypto.randomUUID();
    // getRandomValues is available on the HTTP LAN dashboard too.
    if (typeof host.crypto?.getRandomValues !== 'function') throw new Error('Secure request identity unavailable');
    const bytes = host.crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(value => value.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { newRequestId };
  host.HomeRequestIdentity = { newRequestId };
})(typeof globalThis !== 'undefined' ? globalThis : this);
