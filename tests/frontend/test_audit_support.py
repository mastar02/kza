"""Exercise the WCAG probes against synthetic HTML and real Chromium CDP events."""

import asyncio
import base64
import json
from pathlib import Path
import tempfile

import aiohttp
import pytest
import pytest_asyncio
from websockets.asyncio.client import connect

from tests.frontend.browser_support import (
    Page,
    WCAG_SCRIPT,
    audit_modal_focus,
    audit_modal_return,
    audit_status,
    audit_view,
    audit_wcag,
    browser_binary,
    browser_profile_root,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def page():
    try:
        binary = browser_binary()
    except RuntimeError as exc:
        if exc.args != ('browser_binary_missing',):
            raise
        pytest.skip('Chromium is required for the DOM and keyboard audit probes')

    with tempfile.TemporaryDirectory(prefix='kza-audit-test-', dir=browser_profile_root()) as profile:
        browser = await asyncio.create_subprocess_exec(
            binary, '--headless', '--no-first-run', '--disable-background-networking',
            '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank',
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            port_file = Path(profile) / 'DevToolsActivePort'
            async with asyncio.timeout(15):
                while not port_file.exists() or not port_file.read_text().splitlines():
                    assert browser.returncode is None, 'Chromium exited before exposing CDP'
                    await asyncio.sleep(.05)
            port = int(port_file.read_text().splitlines()[0])
            async with aiohttp.ClientSession() as client:
                async with client.get(f'http://127.0.0.1:{port}/json/list') as response:
                    response.raise_for_status()
                    target = next(item for item in await response.json() if item['type'] == 'page')
            async with connect(target['webSocketDebuggerUrl']) as connection:
                page = Page(connection)
                await page.command('Page.bringToFront')
                yield page
        finally:
            if browser.returncode is None:
                browser.terminate()
            await browser.wait()


async def load(page: Page, html: str) -> None:
    await page.evaluate(f'document.body.innerHTML = {json.dumps(html)}')


async def test_contrast_uses_wcag_luminance_and_computed_paint(page: Page):
    await load(page, '''
        <style>body {background:white; color:black}
          input {appearance:none; border:1px solid #aaa; background:white}
        </style>
        <p style="color:#777">Normal text fails</p>
        <p style="color:#777;font-size:24px">Large text passes</p>
        <p style="color:#767676">Normal text passes</p>
        <p style="color:rgba(0,0,0,.5)">Composited text fails</p>
        <input aria-label="Name">
        <button aria-label="Icon"><svg width="20" height="20"><circle cx="10" cy="10" r="8" fill="#aaa"/></svg></button>
        <p style="background:linear-gradient(white,gray)">Review gradient</p>
    ''')
    await page.evaluate(WCAG_SCRIPT)
    assert await page.evaluate('window.__kzaWcag.contrast([0,0,0], [255,255,255])') == 21
    assert await page.evaluate('window.__kzaWcag.contrast([119,119,119], [255,255,255])') == pytest.approx(4.478, abs=.001)
    assert await page.evaluate('window.__kzaWcag.contrast([118,118,118], [255,255,255])') == pytest.approx(4.542, abs=.001)
    findings = await audit_wcag(page)
    small_text = [item for item in findings if item['criterion'] == '1.4.3' and item['issue'] == 'low_contrast']
    assert [item['element'] for item in small_text] == ['body > p:nth-of-type(1)', 'body > p:nth-of-type(4)']
    assert all(item['minimum'] == 4.5 for item in small_text)
    assert {item['part'] for item in findings if item['criterion'] == '1.4.11'
            and item['issue'] == 'low_contrast'} == {'control_boundary', 'fill'}
    assert any(item['criterion'] == '1.4.3' and item['issue'] == 'paint_needs_review' for item in findings)


async def test_target_minimum_spacing_and_product_goal(page: Page):
    await load(page, '''
        <style>button {position:absolute; box-sizing:border-box; left:10px; padding:0}</style>
        <button style="width:20px;height:20px;top:10px">A</button>
        <button style="width:20px;height:20px;top:31px">B</button>
        <button style="width:20px;height:20px;top:70px">C</button>
        <button style="width:24px;height:24px;top:120px">D</button>
        <button style="width:44px;height:44px;top:175px">E</button>
        <button disabled style="width:18px;height:18px;top:80px">F</button>
    ''')
    findings = await audit_wcag(page)
    minimum = [item for item in findings if item['criterion'] == '2.5.8']
    product = [item for item in findings if item['criterion'] == 'product']
    assert [(item['element'], item['width'], item['minimum']) for item in minimum] == [
        ('body > button:nth-of-type(1)', 20, 24), ('body > button:nth-of-type(2)', 20, 24),
    ]
    assert [(item['width'], item['minimum']) for item in product] == [(20, 44)] * 3 + [(24, 44)]


async def test_spacing_and_text_resize_report_new_clipping_and_restore_styles(page: Page):
    await load(page, '''
        <style>p {font:16px/16px Arial;margin:0}</style>
        <p style="height:17px;overflow:hidden">A line that fits before text changes.</p>
        <p>A flexible paragraph stays readable.</p>
    ''')
    original = await page.evaluate('[...document.querySelectorAll("p")].map(el=>el.getAttribute("style"))')
    findings = await audit_wcag(page)
    clipping = [item for item in findings if item['issue'] == 'text_newly_clipped']
    assert {(item['criterion'], item['element']) for item in clipping} == {
        ('1.4.12', 'body > p:nth-of-type(1)'), ('1.4.4', 'body > p:nth-of-type(1)'),
    }
    assert await page.evaluate('[...document.querySelectorAll("p")].map(el=>el.getAttribute("style"))') == original


async def test_status_messages_check_live_semantics_without_recording_payload(page: Page, tmp_path: Path):
    await load(page, '''
        <div role="status"><span id="progress"></span></div>
        <div role="alert"><span aria-live="off" id="off">No announcement</span></div>
        <p id="plain">Private synthetic status payload</p>
        <div aria-hidden="true"><p id="hidden" role="alert">Hidden from AT</p></div>
    ''')
    await page.evaluate('document.querySelector("#progress").textContent = "Request pending"')
    findings = await audit_wcag(page, status_messages=(
        ('loading', '#progress'), ('stale', '#off'), ('409', '#plain'),
        ('503', '#hidden'), ('empty', '#absent'),
    ))
    assert {item['state']: item['issue'] for item in findings if item['criterion'] == '4.1.3'} == {
        'stale': 'live_semantics_missing', '409': 'live_semantics_missing',
        '503': 'status_hidden_from_at', 'empty': 'status_message_missing',
    }
    await audit_status(page, tmp_path, 'synthetic', '409', '#plain')
    content = (tmp_path / 'a11y-synthetic-409.json').read_text()
    assert json.loads(content)['findings'][0]['criterion'] == '4.1.3'
    assert 'Private synthetic status payload' not in content


async def test_audit_view_keeps_new_findings_nonblocking(page: Page, tmp_path: Path):
    await load(page, '<p style="color:#aaa;background:white">Texto de prueba</p>')
    await page.evaluate('document.documentElement.lang = "es"')

    async def screenshot(page: Page, path: Path, *, full: bool) -> None:
        image = await page.command('Page.captureScreenshot', {'captureBeyondViewport': full})
        path.write_bytes(base64.b64decode(image['data']))

    await audit_view(page, tmp_path, 'synthetic', (), screenshot, widths=(768,),
                     focus_dialog='#missing-dialog', status_messages=(('loading', '#missing-status'),))
    evidence = tmp_path / 'a11y-synthetic.json'
    report = json.loads(evidence.read_text())
    assert report['passed'] is True
    assert report['modal_focus']['findings'][0]['issue'] == 'dialog_missing'
    assert any(item['criterion'] == '1.4.3' and item['issue'] == 'low_contrast'
               for item in report['widths']['768']['wcag_findings'])
    assert any(item['issue'] == 'status_message_missing' for item in report['widths']['768']['wcag_findings'])

    await page.evaluate('document.documentElement.lang = "en"')
    with pytest.raises(AssertionError, match='html lang'):
        await audit_view(page, tmp_path, 'synthetic-legacy', (), screenshot, widths=(768,))
    assert json.loads((tmp_path / 'a11y-synthetic-legacy.json').read_text())['passed'] is False


async def test_modal_focus_detects_escapes_and_proves_wrap_and_return(page: Page, tmp_path: Path):
    await load(page, '''
        <button id="opener">Open</button>
        <div role="dialog" aria-modal="true" id="modal">
            <button id="first">First</button><button id="last">Last</button>
        </div>
        <button id="outside">Outside</button>
    ''')
    await page.evaluate('document.querySelector("#opener").focus()')
    broken = await audit_modal_focus(page, '#modal', initial='#first')
    assert [item['issue'] for item in broken['findings']] == [
        'initial_focus_incorrect', 'focus_escaped', 'focus_escaped',
    ]
    assert await page.evaluate('document.activeElement.id') == 'opener'

    await page.evaluate('''(() => {
        const modal = document.querySelector('#modal'), first = modal.firstElementChild, last = modal.lastElementChild;
        first.focus();
        modal.addEventListener('keydown', event => {
            if (event.key !== 'Tab' || event.target !== (event.shiftKey ? first : last)) return;
            event.preventDefault();
            (event.shiftKey ? last : first).focus();
        });
    })()''')
    working = await audit_modal_focus(page, '#modal', initial='#first')
    assert working['findings'] == []
    assert await page.evaluate('document.activeElement.id') == 'first'
    await page.evaluate('document.querySelector("#modal").hidden = true; document.querySelector("#opener").focus()')
    await audit_modal_return(page, tmp_path, 'modal', working, dialog='#modal', opener='#opener')
    evidence = tmp_path / 'a11y-modal.json'
    assert json.loads(evidence.read_text())['focus_return_ok'] is True
    failed_return = {'findings': []}
    await page.evaluate('document.querySelector("#outside").focus()')
    await audit_modal_return(page, tmp_path, 'modal-broken', failed_return, dialog='#modal', opener='#opener')
    assert failed_return['findings'][0]['issue'] == 'focus_not_returned'
