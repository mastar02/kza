"""Analyze all currently reviewable dashboard WAVs over its authenticated API.

Uses the same version-three receipt protocol as the dashboard. Only waveform
statistics leave the process: no WAV, transcript, token or capture ID is saved.
The runtime account can run a compiled copy with --runtime-credential; it reads
only the existing dashboard token and never changes a service or DSP setting.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
import json
import os
import time
from uuid import uuid4

import aiohttp
import soundfile as sf

# A compiled operator bundle places both modules next to one another.
try:
    from tools.analyze_audio_quality import measure_channel, summarize
except ModuleNotFoundError:
    from analyze_audio_quality import measure_channel, summarize


BASE_URL = 'http://127.0.0.1:9500'


def measure_preview(row: dict) -> dict:
    """Measure a bounded live preview without copying transcript or identity."""
    raw = audio = None
    try:
        if (row.get('audio_mime') != 'audio/wav' or not isinstance(row.get('audio_base64'), str)
                or len(row['audio_base64']) > 256060 or row.get('expires_at', 0) <= time.time()):
            raise ValueError('invalid_or_expired_preview')
        raw = base64.b64decode(row['audio_base64'], validate=True)
        if not 44 < len(raw) <= 192044:
            raise ValueError('invalid_preview_size')
        with sf.SoundFile(BytesIO(raw)) as recording:
            if recording.samplerate != 16000 or recording.channels != 1 or recording.subtype != 'PCM_16':
                raise ValueError('invalid_preview_format')
            audio = recording.read(dtype='float64', always_2d=True)
        if row.get('expires_at', 0) <= time.time():
            raise ValueError('expired_preview')
        return {'kind': 'current_dashboard_preview', 'room': row.get('room_id'),
                'sample_rate': 16000, 'subtype': 'PCM_16', 'channels': 1, 'filename_date': None,
                'measurements': [measure_channel(audio[:, 0], 16000)]}
    finally:
        raw = audio = row = None


async def _ack(socket, receipts: list[dict]) -> None:
    for start in range(0, len(receipts), 16):
        await socket.send_json({'type': 'live_ack', 'receipts': receipts[start:start + 16]})


async def analyze_dashboard(credential: str, *, base_url: str = BASE_URL) -> dict:
    """Read every available page and preview, acknowledging only released bodies."""
    rows, outcomes, cursors, identities = [], Counter(), set(), set()
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + credential}, timeout=timeout,
                                     trust_env=False) as session:
        async with session.ws_connect(base_url + '/ws/live', max_msg_size=1_048_576) as socket:
            await socket.send_json({'type': 'live_hello', 'version': 3, 'client_id': str(uuid4())})
            connection = asyncio.get_running_loop().create_future()
            busy = False
            pending = {}

            async def acknowledge(receipts=()):
                for receipt in receipts:
                    pending[receipt['ticket']] = {'ticket': receipt['ticket'], 'nonce': receipt['nonce']}
                if not busy and pending:
                    batch = list(pending.values())
                    pending.clear()
                    await _ack(socket, batch)

            async def controls():
                try:
                    async for incoming in socket:
                        if incoming.type != aiohttp.WSMsgType.TEXT:
                            continue
                        frame = json.loads(incoming.data)
                        kind = frame.get('type')
                        if kind == 'live_reset':
                            if connection.done():
                                raise RuntimeError('unexpected_dashboard_reset')
                            payload = frame['payload']
                            frame = incoming = None
                            await socket.send_json({'type': 'live_reset_ack', 'id': payload['id'], 'nonce': payload['nonce']})
                        elif kind == 'rest_ready':
                            if not connection.done():
                                connection.set_result(frame['connection_id'])
                        else:
                            receipts = []
                            if frame.get('receipt'):
                                receipts.append(frame['receipt'])
                            if kind == 'retire':
                                receipts.extend({'ticket': item['ticket'], 'nonce': item['nonce']}
                                                for item in frame.get('payload', {}).get('receipts', []))
                            # Live event bodies are discarded before their ACK.
                            # REST preview receipts are owned by the caller below.
                            frame = incoming = None
                            await acknowledge(receipts)
                        frame = incoming = None
                    raise RuntimeError('dashboard_socket_closed')
                except BaseException as error:
                    if not connection.done():
                        connection.set_exception(error)
                    raise

            receiver = asyncio.create_task(controls())
            try:
                identity = await asyncio.wait_for(asyncio.shield(connection), timeout=20)
                headers = {'X-KZA-Live-Connection': identity}

                async def get(path):
                    async with session.get(base_url + path, headers=headers, allow_redirects=False) as response:
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            body.extend(chunk)
                            if len(body) > 1_048_576:
                                raise ValueError('dashboard_response_too_large')
                        if response.status in (404, 409):
                            return None
                        if response.status != 200:
                            raise RuntimeError('dashboard_http_' + str(response.status))
                        return json.loads(body)

                cursor = None
                while True:
                    if receiver.done():
                        receiver.result()
                    busy = True
                    receipt = None
                    try:
                        document = await get('/api/voices' + ('?cursor=' + cursor if cursor else ''))
                        if not document:
                            raise RuntimeError('capture_page_unavailable')
                        batch = document.get('snapshot')
                        cursor = (batch or {}).get('next_cursor') or document.get('next_cursor')
                        candidates = [(item['id'], item.get('review_available') is True)
                                      for item in (batch or {}).get('rows', [])]
                        receipt = (batch or {}).get('receipt')
                    finally:
                        document = batch = None
                        busy = False
                        await acknowledge([receipt] if receipt else [])
                    for capture, available in candidates:
                        if capture in identities:
                            continue
                        identities.add(capture)
                        if len(identities) > 4096:
                            raise ValueError('capture_capacity_exceeded')
                        if not available:
                            outcomes['no_preview'] += 1
                            continue
                        busy = True
                        receipt = None
                        try:
                            document = await get('/api/voices/captures/' + capture + '/review')
                            batch = (document or {}).get('snapshot')
                            receipt = (batch or {}).get('receipt')
                            preview = (batch or {}).get('rows', [])
                            if len(preview) == 1:
                                rows.append(measure_preview(preview[0]))
                                outcomes['analyzed'] += 1
                            else:
                                outcomes['unavailable_or_expired'] += 1
                        except ValueError:
                            outcomes['invalid_or_expired'] += 1
                        finally:
                            preview = document = batch = None
                            busy = False
                            await acknowledge([receipt] if receipt else [])
                    if not cursor:
                        break
                    if cursor in cursors:
                        raise RuntimeError('repeated_capture_cursor')
                    cursors.add(cursor)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    return {'checked_at': datetime.now(UTC).isoformat(), 'source': 'authenticated_dashboard_v3',
            'listed_captures': len(identities), 'outcomes': dict(outcomes), 'summary': summarize(rows),
            'raw_audio_transcripts_credentials_or_capture_ids_exported': False,
            'scope': 'Only previews currently authorized by the running dashboard; expired captures are not recovered.'}


def main() -> None:
    """Print aggregate waveform metrics, using an existing configured credential."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-credential', action='store_true',
                        help='Read only KZA_DASHBOARD_TOKEN from the fixed runtime secret file')
    args = parser.parse_args()
    credential = os.environ.get('KZA_DASHBOARD_TOKEN')
    if args.runtime_credential:
        from dotenv import dotenv_values
        credential = dotenv_values('/home/kza/secrets/.env').get('KZA_DASHBOARD_TOKEN')
    try:
        if not credential:
            raise RuntimeError('dashboard_credential_unavailable')
        result = asyncio.run(analyze_dashboard(credential))
    except Exception as error:
        # Errors can include request headers or private server bodies. Export
        # neither exception messages nor tracebacks from the runtime account.
        result = {'available': False, 'error_type': type(error).__name__}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
