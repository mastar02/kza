"""Measure eight HTTPS voice-preview reads, exporting timings and byte counts only."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import ssl
from statistics import median

try:
    from analyze_dashboard_audio import analyze_dashboard
except ModuleNotFoundError:
    from tools.analyze_dashboard_audio import analyze_dashboard


def summarize_timings(rows: list[dict]) -> dict:
    """Aggregate network/body/JSON times without retaining capture identifiers."""
    result = {}
    for kind in ('captures', 'review'):
        group = [row for row in rows if row['kind'] == kind]
        if not group:
            continue
        metrics = {}
        for name in ('headers_ms', 'body_ms', 'json_ms', 'total_ms', 'bytes'):
            values = sorted(row[name] for row in group)
            metrics[name] = {'min': values[0], 'p50': median(values),
                             'p95': values[math.ceil(.95*len(values))-1], 'max': values[-1]}
        result[kind] = {'requests': len(group), **metrics}
    return result


def main() -> None:
    """Use the existing runtime credential, keeping WAVs and transcripts in memory."""
    from dotenv import dotenv_values

    try:
        token = dotenv_values('/home/kza/secrets/.env').get('KZA_DASHBOARD_TOKEN')
        if not token:
            raise ValueError('dashboard_credential_unavailable')
        context = ssl.create_default_context(cafile=str(Path(__file__).resolve().parent/'dashboard-public.crt'))
        result = asyncio.run(analyze_dashboard(token, base_url='https://192.168.1.2:9540',
                                              max_captures=8, timings=True, tls_context=context))
        result = {'checked_at': result['checked_at'], 'outcomes': result['outcomes'],
                  'timings': summarize_timings(result['request_timings']),
                  'raw_audio_transcripts_credentials_or_capture_ids_exported': False,
                  'scope': 'HTTPS reads from runtime host; does not measure the user device network, browser decode or native player retirement.'}
    except Exception as error:
        result = {'available': False, 'error_type': type(error).__name__}
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
