"""Exercise NEXO's real WebGL model with a separate nine-room catalog and CDP input."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import tempfile

from aiohttp import ClientSession, web
from websockets.asyncio.client import connect

from tests.frontend.browser_support import Page, browser_binary, browser_profile_root, press_enter

FRONTEND = Path(__file__).resolve().parents[2] / 'src/dashboard/frontend'
VIEWPORTS = ((1440, 1000), (768, 1024), (390, 844), (320, 760))

CONTEXT_LOSS_PROBE = """(() => {
    const phase = new URL(location.href).searchParams.get('context-loss');
    if (!phase) return;
    Error.stackTraceLimit = 40;
    const probe = window.contextLossProbe = {};
    const canvas = () => document.querySelector('#house-canvas canvas');
    document.addEventListener('webglcontextlost', event => {
        if (event.target === canvas()) probe.eventDelivered = true;
    }, true);
    const lose = gl => {
        const extension = gl.getExtension('WEBGL_lose_context');
        if (!extension) throw new Error('Chromium must expose WEBGL_lose_context');
        probe.attempted = true;
        extension.loseContext();
        probe.lostBeforeEvent = gl.isContextLost() && !probe.eventDelivered;
    };
    const request = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = callback => {
        if (callback.name !== 'frame') return request(callback);
        return request(time => {
            const firstFrame = time >= 30 && window.houseProbe?.ready && !probe.attempted;
            if (firstFrame && phase === 'before-frame') lose(canvas().getContext('webgl2'));
            try {
                callback(time);
            } finally {
                if (firstFrame && probe.attempted) {
                    probe.pausedBeforeEvent = !window.houseProbe.ready && !probe.eventDelivered;
                }
            }
        });
    };
    const originalLog = WebGL2RenderingContext.prototype.getProgramInfoLog;
    WebGL2RenderingContext.prototype.getProgramInfoLog = function (program) {
        if (phase === 'during-render' && this.canvas === canvas() && !probe.attempted) {
            lose(this);
            const log = originalLog.call(this, program);
            probe.nullProgramLog = log === null;
            return log;
        }
        return originalLog.call(this, program);
    };
})()"""


async def exercise_context_loss(page: Page, url: str) -> dict:
    """Interrupt a real Three.js frame on either side of the render call."""
    await page.command('Page.enable')
    await page.command('Runtime.enable')
    await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': CONTEXT_LOSS_PROBE})
    await page.command('Emulation.setDeviceMetricsOverride', {
        'width': 768, 'height': 1024, 'deviceScaleFactor': 1, 'mobile': False,
    })
    trials = {}
    for phase in ('before-frame', 'during-render'):
        trials[phase] = []
        for trial in range(20):
            start = len(page.errors)
            await page.command('Page.navigate', {'url': f'{url}?context-loss={phase}&trial={trial}'})
            await page.wait_for(f"location.search === '?context-loss={phase}&trial={trial}' "
                                "&& window.contextLossProbe?.eventDelivered")
            state = await page.evaluate('window.contextLossProbe')
            state['errors'] = page.errors[start:]
            trials[phase].append(state)
    summary = {phase: {
        'trials': len(states),
        'lost_before_event': sum(state.get('lostBeforeEvent', False) for state in states),
        'paused_before_event': sum(state.get('pausedBeforeEvent', False) for state in states),
        'null_program_log': sum(state.get('nullProgramLog', False) for state in states),
        'trials_with_uncaught_exceptions': sum(bool(state['errors']) for state in states),
        'uncaught_exceptions': sum(len(state['errors']) for state in states),
    } for phase, states in trials.items()}
    passed = all(counts['lost_before_event'] == counts['paused_before_event'] == counts['trials']
                 and not counts['uncaught_exceptions'] for counts in summary.values())
    passed = passed and summary['during-render']['null_program_log'] == summary['during-render']['trials']
    return {'passed': passed, 'summary': summary, 'trials': trials}


async def click(page: Page, point: dict, *, touch: bool = False) -> None:
    """Click at CSS viewport coordinates through the browser's native input path."""
    if touch:
        await page.command('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [point]})
        await page.command('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        return
    for event, buttons in (('mousePressed', 1), ('mouseReleased', 0)):
        await page.command('Input.dispatchMouseEvent', {
            'type': event, 'button': 'left', 'buttons': buttons, 'clickCount': 1, **point,
        })


async def screenshot(page: Page, path: Path) -> None:
    capture = await page.command('Page.captureScreenshot', {'format': 'png'})
    path.write_bytes(base64.b64decode(capture['data']))
    path.chmod(0o600)


async def check_gestures(page: Page, width: int, inspection: dict, occupancy: float) -> dict:
    point = inspection['samples']['living'][0]['point']
    miss = inspection['miss']
    for touch in (False, True):
        await page.evaluate('houseProbe.clearSelection()')
        await click(page, miss, touch=touch)
        assert await page.evaluate('houseProbe.selections') == [], ('far miss', width)
    near = inspection.get('nearMiss')
    view = '3D'
    if near is None:
        # At this scale the exterior walls can reach the capture band in the 3D view.
        await page.evaluate("houseProbe.house.setView('top');houseProbe.house.zoom(-.25);houseProbe.settle()")
        plan = await page.evaluate('houseProbe.samples()')
        near = plan.get('nearMiss')
        assert near, ('no air within capture radius', width, inspection.get('airDistance'), plan.get('airDistance'))
        view = 'Plano'
    for touch in (False, True):
        await page.evaluate('houseProbe.clearSelection()')
        await click(page, near['point'], touch=touch)
        assert await page.evaluate('houseProbe.selections') == [near['room']], ('near miss', width, near)
    await page.evaluate('houseProbe.house.reset();houseProbe.settle()')
    await page.evaluate('houseProbe.clearSelection()')
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', **point})
    assert await page.evaluate('houseProbe.hovered()') == 'living', ('hover', width)
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', **miss})
    assert await page.evaluate('houseProbe.hovered()') is None, ('hover clear', width)
    yaw = await page.evaluate('houseProbe.house.yaw()')
    await page.command('Input.dispatchMouseEvent', {
        'type': 'mousePressed', 'button': 'left', 'buttons': 1, 'clickCount': 1, **point,
    })
    drag_end = {**point, 'x': point['x'] + 6}
    await page.command('Input.dispatchMouseEvent', {
        'type': 'mouseMoved', 'buttons': 1, **drag_end,
    })
    await page.command('Input.dispatchMouseEvent', {
        'type': 'mouseReleased', 'button': 'left', 'buttons': 0, 'clickCount': 1, **drag_end,
    })
    assert await page.evaluate('houseProbe.selections') == []
    assert await page.evaluate('houseProbe.house.yaw()') != yaw
    await page.evaluate('houseProbe.house.reset();houseProbe.settle()')
    await page.command('Input.dispatchTouchEvent', {
        'type': 'touchStart', 'touchPoints': [{**point, 'id': 0}, {**point, 'x': point['x'] - 28, 'id': 1}],
    })
    await page.command('Input.dispatchTouchEvent', {
        'type': 'touchMove', 'touchPoints': [{**point, 'id': 0}, {**point, 'x': point['x'] - 65, 'id': 1}],
    })
    await page.command('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
    await page.evaluate('houseProbe.settle()')
    assert await page.evaluate('houseProbe.selections') == [], ('pinch', width)
    assert await page.evaluate('houseProbe.measure().occupancy') > occupancy, ('pinch zoom', width)
    await page.evaluate('houseProbe.house.reset();houseProbe.settle()')
    await page.command('Input.dispatchMouseEvent', {
        'type': 'mouseWheel', 'deltaX': 0, 'deltaY': -60, 'modifiers': 2, **point,
    })
    await page.evaluate('houseProbe.settle()')
    assert await page.evaluate('houseProbe.selections') == [], ('wheel', width)
    assert await page.evaluate('houseProbe.measure().occupancy') > occupancy, ('wheel zoom', width)
    return {**near, 'view': view}


async def exercise(page: Page, url: str, evidence: Path, mode: str) -> dict:
    await page.command('Page.enable')
    await page.command('Runtime.enable')
    result = {'passed': False, 'widths': {}, 'failures': []}
    await page.command('Emulation.setEmulatedMedia', {
        'features': [{'name': 'prefers-reduced-motion', 'value': 'reduce'}],
    })
    for width, height in VIEWPORTS:
        await page.command('Emulation.setDeviceMetricsOverride', {
            'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': width <= 390,
        })
        await page.command('Page.navigate', {'url': url})
        await page.wait_for('window.houseProbe?.ready === true')
        await page.evaluate('houseProbe.settle()')
        stats = await page.evaluate('houseProbe.measure()')
        result['widths'][width] = stats
        if width in (390, 1440):
            await screenshot(page, evidence / f'nueve-ambientes-{width}.png')
        if mode == 'measure':
            continue
        minimum_height = {1440: 520, 768: 460, 390: 420, 320: 420}[width]
        assert stats['canvas']['height'] >= minimum_height and .85 <= stats['occupancy'] <= 1, stats
        assert not stats['pinOverlaps'] and not stats['coveredAnchors'] and len(stats['pins']) == 9, stats
        assert all(pin['visible'] and pin['inside'] and min(pin['w'], pin['h']) >= 44
                   and pin['font'] >= 14 and pin['label'] == pin['name'] for pin in stats['pins']), stats
        if width <= 390:
            hallway = next(pin for pin in stats['pins'] if pin['room'] == 'pasillo')
            assert hallway['anchorGap'] <= 44, (width, hallway)
        if width == 320:
            continue

        # Verify the visible click targets before isolating the canvas hit tests.
        for pin in stats['pins']:
            await page.evaluate('houseProbe.clearSelection()')
            await click(page, pin['center'], touch=width <= 390)
            assert await page.evaluate('houseProbe.selections') == [pin['room']], pin

        # Uncover the model so these samples exercise the canvas rather than the separate pin buttons.
        await page.evaluate("document.querySelector('#room-pins').style.visibility = 'hidden'")
        inspection = await page.evaluate('houseProbe.samples()')
        stats.update(inspection)
        for room, samples in inspection['samples'].items():
            assert any(sample['kind'] == 'centro' for sample in samples), (width, room, samples)
            assert any(sample['kind'] == 'pared' for sample in samples), (width, room, samples)
            assert sum(sample['kind'][0] in 'xz' for sample in samples) == 4, (width, room, samples)
            if room not in {'pasillo', 'escalera', 'balcon'}:
                assert any(sample['kind'] == 'mueble' for sample in samples), (width, room, samples)
            for sample in samples:
                sample['results'] = {}
                for method in ('mouse', 'touch'):
                    await page.evaluate('houseProbe.clearSelection()')
                    await click(page, sample['point'], touch=method == 'touch')
                    actual = await page.evaluate('houseProbe.selections')
                    sample['results'][method] = actual
                    if actual != [room]:
                        result['failures'].append({'width': width, 'expected': room, 'method': method, **sample})

        if mode == 'geometry':
            continue
        await page.evaluate("document.querySelector('#room-pins').style.visibility = ''")
        padding_probes = 0
        for room, samples in inspection['samples'].items():
            for sample in samples:
                if not await page.evaluate(f'houseProbe.coveredByPadding({json.dumps(sample["point"])}, {json.dumps(room)})'):
                    continue
                padding_probes += 1
                await page.evaluate('houseProbe.clearSelection()')
                await click(page, sample['point'], touch=width <= 390)
                assert await page.evaluate('houseProbe.selections') == [room], ('pin padding', width, room, sample)
        stats['paddingProbes'] = padding_probes
        await page.evaluate("document.querySelector('#room-pins').style.visibility = 'hidden'")
        stats['verifiedNearMiss'] = await check_gestures(page, width, inspection, stats['occupancy'])
        await page.evaluate('houseProbe.house.reset();houseProbe.unlink();houseProbe.settle()')
        await click(page, inspection['unlinked'], touch=width <= 390)
        assert await page.evaluate('houseProbe.selections') == [], ('unlinked scan', width)

        assert await page.evaluate("""(() => {
            const pins = document.querySelector('#room-pins');
            pins.style.visibility = '';
            const pin = [...pins.querySelectorAll('.room-pin')].find(button => button.textContent === 'Sala de estar');
            pin.focus({ preventScroll: true });
            houseProbe.clearSelection();
            return document.activeElement === pin;
        })()"""), ('pin focus', width)
        await press_enter(page)
        assert await page.evaluate('houseProbe.selections') == ['living'], ('pin Enter', width)
        assert await page.evaluate("getComputedStyle(document.activeElement.firstChild).outlineStyle === 'solid'"), ('pin focus ring', width)
        assert await page.evaluate('houseProbe.destroy()')
        assert not page.errors, page.errors

    result['passed'] = not result['failures']
    return result


async def main(frontend: Path, evidence: Path, mode: str) -> None:
    private = frontend / 'home/nexo/private'
    required = ('casa.glb', 'house-plan.js', 'house-picking-fixture.js')
    if not all((private / name).is_file() for name in required):
        raise FileNotFoundError('Real house picking requires the local plan, model and picking fixture')
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Use the production viewport markup and CSS, mounting only its independent viewer.
    html = (frontend / 'home/index.html').read_text()
    html = html[:html.index('<section id="actions-view"')] + '</section></main></div>'
    html = html.replace('<script type="module" src="/home/nexo/app.js"></script>',
                        '<script type="module" src="/probe.js"></script>')
    html = html.replace('id="login-panel"', 'id="login-panel" hidden')
    html = html.replace('id="nexo-workspace" hidden', 'id="nexo-workspace"')
    html = html.replace('class="casa-grid" hidden', 'class="casa-grid"')
    probe = Path(__file__).with_suffix('.js')

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=html, content_type='text/html')

    async def script(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(probe)

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/probe.js', script)
    app.router.add_static('/home/', frontend / 'home')
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        with tempfile.TemporaryDirectory(prefix='nexo-picking-', dir=browser_profile_root()) as temporary:
            profile = Path(temporary)
            with (evidence / 'chromium.log').open('w') as log:
                browser = await asyncio.to_thread(subprocess.Popen, [
                    browser_binary(), '--headless', '--disable-gpu', '--use-angle=swiftshader',
                    '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking',
                    '--disable-component-update', '--remote-debugging-port=0',
                    f'--user-data-dir={profile}', 'about:blank',
                ], stdout=log, stderr=log)
                try:
                    active_port = profile / 'DevToolsActivePort'
                    for _ in range(400):
                        if active_port.exists():
                            break
                        assert browser.poll() is None, 'Chromium exited'
                        await asyncio.sleep(.05)
                    debugging = int(active_port.read_text().splitlines()[0])
                    async with ClientSession() as client:
                        async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            targets = await response.json()
                    target = next(target for target in targets if target['type'] == 'page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = Page(connection)
                        url = f'http://127.0.0.1:{port}/'
                        if mode == 'context-loss':
                            result = await exercise_context_loss(page, url)
                        else:
                            result = await exercise(page, url, evidence, mode)
                    (evidence / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                    if mode == 'context-loss':
                        print(json.dumps({'passed': result['passed'], 'summary': result['summary']}))
                        assert result['passed'], f"Context-loss regression. See {evidence / 'result.json'}"
                        return
                    geometry = [item for item in result['failures'] if item['kind'] in ('pared', 'mueble')]
                    assert result['passed'], f"{len(result['failures'])} selection failures, including {len(geometry)} walls/furniture; first: {geometry[:3]}"
                    print(json.dumps({'passed': True, 'mode': mode, 'widths': list(result['widths'])}))
                finally:
                    if browser.poll() is None:
                        browser.terminate()
                        try:
                            await asyncio.to_thread(browser.wait, 10)
                        except subprocess.TimeoutExpired:
                            browser.kill()
                            await asyncio.to_thread(browser.wait, 5)
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', type=Path, default=FRONTEND, help='Serve a separate baseline or mutation copy')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'geometry', 'measure', 'context-loss'), default='full')
    args = parser.parse_args()
    asyncio.run(main(args.frontend, args.evidence, args.mode))
