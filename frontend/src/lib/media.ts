import { record } from './session'

export const fingerprint = /^(?:vf_[0-9a-f]{64}|[0-9a-f]{32})$/
export const strictFingerprint = /^vf_[0-9a-f]{64}$/
export type Sample = { id: string; transcript: string | null; audio: Uint8Array<ArrayBuffer> | null }

export function sampleFrom(rows: readonly unknown[], id: string, format: 'text' | 'audio', evaluation = false): Sample {
  const row = record(rows[0])
  if (rows.length !== 1 || row?.id !== id || !fingerprint.test(id) ||
      evaluation && (!strictFingerprint.test(id) || row.modality !== (format === 'audio' ? 'audio_preview' : 'transcript_full'))) {
    throw new Error('Invalid review sample')
  }
  const transcript = row.transcript
  const validScope = evaluation ? row.transcript_scope === 'full_segment'
    : (row.transcript_scope === 'full_segment' || row.transcript_scope === 'preview')
  if ((!evaluation || format === 'text') && (typeof transcript !== 'string' || transcript.length > 16384 ||
      !validScope || new TextEncoder().encode(transcript).length > 32768)) {
    throw new Error('Invalid transcript')
  }
  if (format === 'text') return { id, transcript: transcript as string, audio: null }

  const base64 = row.audio_base64
  if (row.audio_mime !== 'audio/wav' || typeof base64 !== 'string' ||
      base64.length > 256060 || !/^[A-Za-z0-9+/]+={0,2}$/.test(base64)) throw new Error('Invalid audio')
  const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0))
  const data = new DataView(bytes.buffer)
  const tag = (offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4))
  if (bytes.length < 25644 || bytes.length > 192044 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE' ||
      tag(12) !== 'fmt ' || tag(36) !== 'data' || data.getUint32(4, true) !== bytes.length - 8 ||
      data.getUint32(16, true) !== 16 || data.getUint16(20, true) !== 1 || data.getUint16(22, true) !== 1 ||
      data.getUint32(24, true) !== 16000 || data.getUint32(28, true) !== 32000 ||
      data.getUint16(32, true) !== 2 || data.getUint16(34, true) !== 16 ||
      data.getUint32(40, true) !== bytes.length - 44 || (bytes.length - 44) % 2 ||
      row.clip_duration_s !== (bytes.length - 44) / 32000) throw new Error('Invalid review WAV')
  return { id, transcript: evaluation ? null : transcript as string, audio: bytes }
}

const retirements = new Set<Promise<void>>()

export async function mediaRetired() {
  while (retirements.size) await Promise.all(retirements)
}

/** Media resources may outlive their element. The receipt waits for native retirement. */
export class SampleMedia {
  #player: HTMLAudioElement | null = null
  #url: string | null = null
  #retire: Promise<void> = Promise.resolve()
  #pending = false

  play(container: HTMLElement, bytes: Uint8Array<ArrayBuffer>) {
    if (this.#pending || this.#player || document.hidden) return
    const player = document.createElement('audio')
    player.controls = true
    player.preload = 'none'
    this.#url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
    this.#player = player
    player.src = this.#url
    container.append(player)
    void player.play().catch(() => {}) // The native controls remain available if autoplay is denied.
  }

  clear() {
    const player = this.#player
    const url = this.#url
    if (!player || !url) return
    this.#player = null
    this.#url = null
    this.#pending = true
    const needsEvent = player.networkState !== 0 || player.readyState !== 0 || !!player.currentSrc
    this.#retire = new Promise<void>(resolve => {
      let emptied = !needsEvent
      let finished = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const check = () => {
        if (finished) return
        if (emptied && !player.isConnected && !player.hasAttribute('src') && player.srcObject === null &&
            player.paused && player.readyState === 0 && player.networkState === 0 &&
            player.buffered.length === 0 && Number.isNaN(player.duration)) {
          finished = true
          clearTimeout(timer)
          player.removeEventListener('emptied', onEmpty)
          this.#pending = false
          resolve()
        } else if (emptied && timer === undefined) {
          // The event and final native media state can settle on different turns.
          timer = setTimeout(() => { timer = undefined; check() }, 25)
        }
      }
      const onEmpty = () => { emptied = true; queueMicrotask(check) }
      player.addEventListener('emptied', onEmpty)
      player.pause()
      player.removeAttribute('src')
      player.remove()
      player.load()
      URL.revokeObjectURL(url)
      queueMicrotask(check)
    })
    const retirement = this.#retire
    retirements.add(retirement)
    void retirement.then(() => retirements.delete(retirement))
  }
}
