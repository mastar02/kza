import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { access } from 'node:fs/promises'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { stripVTControlCharacters } from 'node:util'
import { setTimeout as delay } from 'node:timers/promises'

export default async function setup() {
  const root = path.resolve(__dirname, '../..')
  const target = process.env.CARGO_TARGET_DIR || path.join(root, 'target')
  const bin = process.env.NEXA_WEB_BIN || path.join(target, 'debug/nexa-web')
  const dist = path.join(root, 'frontend/dist')
  await access(path.join(dist, 'index.html'))
  const child = spawn(bin, ['--test-mode', '--allow-insecure-cookies', '--bind',
    '127.0.0.1:0', '--static-dir', dist], { stdio: ['ignore', 'pipe', 'inherit'] })
  const closed = new Promise<void>(resolve => child.once('close', () => resolve()))
  const lines = createInterface({ input: child.stdout })
  const timeout = AbortSignal.timeout(10_000)
  async function start() {
    for await (const line of lines) {
      const address = /\baddress=(127\.0\.0\.1:\d+)\b/.exec(stripVTControlCharacters(line))
      if (!address) continue
      const origin = 'http://' + address[1]
      while (!timeout.aborted) {
        const response = await fetch(origin + '/api/health', { signal: timeout })
        await response.body?.cancel()
        if (response.status === 200) {
          process.env.NEXA_E2E_ORIGIN = origin
          return
        }
        await delay(50)
      }
    }
    throw new Error('nexa-web exited before readiness')
  }
  async function stop() {
    child.kill('SIGTERM')
    const force = setTimeout(() => child.kill('SIGKILL'), 3000)
    try { await closed } finally { clearTimeout(force) }
  }
  try {
    await Promise.race([start(), once(child, 'error').then(([err]) => Promise.reject(err)),
      once(timeout, 'abort').then(() => { throw new Error('nexa-web startup timed out') })])
    return stop
  } catch (error) {
    await stop()
    throw error
  }
}
