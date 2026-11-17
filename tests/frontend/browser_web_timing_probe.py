"""Harness checks of the T046-U1 probe on server headless Chromium (never a measurement).

Runs the probe's real run lifecycle against the fixture backend's harness pages
(tools/web_timing_pages) through ``web_timing_probe.fixture_page_checks`` and checks the
browser-level items of t046/preregistration.json unit_checks_required: one injection per
document and removal on reused targets (R5-1), busy loops inside and after the timed window
(R3-7, R4-2), sub-threshold feedback (R5-10), detached target (R4-1), missing target (R5-4),
play-gesture lookup and pipelined dispatch with a long task (R5-8, R5-13) and the
post-window command log (R4-2). Values come from fixture pages, not from NEXO.

Usage (server, as yo)::

    taskset -c 24-47 nice -n 10 .venv/bin/python -m tests.frontend.browser_web_timing_probe
"""

import asyncio
import json
import sys

from tests.frontend.browser_support import browser_binary, browser_profile_root, evidence_directory
from tools import web_timing_probe as probe


async def main() -> int:
    if sys.platform == 'darwin':
        raise SystemExit('server headless Chromium only; the Mac runs `smoke-item fixture-pages`')
    out_dir = probe.checked_output('harness', evidence_directory('KZA_T046_HARNESS_EVIDENCE', str(
        probe.HARNESS_ROOT.relative_to(probe.REPOSITORY))))
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    summary = await probe.fixture_page_checks('harness', out_dir, binary=browser_binary(),
                                              profile_root=browser_profile_root(mode=0o700))
    sys.stdout.write(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
