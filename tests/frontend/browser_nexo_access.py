"""Actual Chromium coverage for NEXO access and exact evaluator UI.

Synthetic credentials remain in process memory. Evidence never serializes a
credential, a request body, an input value, or raw CDP exception details.
"""

import asyncio
from contextlib import AsyncExitStack
from functools import partial
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import aiohttp
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices
from src.users.user_manager import PermissionLevel
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_nexo_voices import capture_personas_matrix
from tests.frontend.browser_support import (
    FixtureRows,
    Page, WCAG_SCRIPT, assert_frame_consistent, audit_current_accessibility, audit_modal_focus,
    audit_modal_return, audit_status, audit_view,
    browser_binary, browser_profile_root, evidence_directory, press_enter,
)
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO, Encoder


EVIDENCE = evidence_directory('KZA_NEXO_ACCESS_EVIDENCE', 'data/autonomous-roadmap-20261102/t013/nexo-access-browser-runs')
INSTRUMENT = r"""(() => {
  window.__emptyCallbacks=[];window.__holdEmpty=false;window.__urls=new Set();
  window.__credentialPosts=[];window.__loseIssueReceipt=false;window.__copyRequests=0;window.__copiedShape=false;
  window.__evaluationReads=0;window.__accessFixture=null;window.__accessCallbacks=[];
  window.__personRequests=[];window.__personSockets=[];
  const NativeSocket=WebSocket;
  window.WebSocket=class extends NativeSocket { constructor(...args){super(...args);window.__personSockets.push(new URL(args[0]).pathname);} };
  Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async value=>{
    window.__copyRequests++;window.__copiedShape=/^kza_[A-Za-z0-9_-]{43}$/.test(value);
    window.__copiedExact=value===document.querySelector('#access-secret-dialog input').value;
  }}});
  const add=HTMLMediaElement.prototype.addEventListener,remove=HTMLMediaElement.prototype.removeEventListener,wrappers=new WeakMap();
  HTMLMediaElement.prototype.addEventListener=function(type,callback,options){
    if(type==='emptied'){const wrapped=event=>window.__holdEmpty?window.__emptyCallbacks.push(()=>callback(event)):callback(event);wrappers.set(callback,wrapped);return add.call(this,type,wrapped,options);}
    return add.call(this,type,callback,options);
  };
  HTMLMediaElement.prototype.removeEventListener=function(type,callback,options){return remove.call(this,type,wrappers.get(callback)||callback,options);};
  const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
  URL.createObjectURL=blob=>{const value=create(blob);window.__urls.add(value);return value;};
  URL.revokeObjectURL=value=>{window.__urls.delete(value);return revoke(value);};
  const fetch=window.fetch.bind(window);
  window.fetch=async(path,options={})=>{
    if(typeof path==='string')window.__personRequests.push(new URL(path,location.href).pathname);
    if(path==='/api/admin/auth/credentials' && options.method==='GET'){
      const response=await fetch(path,options);
      if(window.__accessFixture==='hold')await new Promise(resolve=>window.__accessCallbacks.push(resolve));
      if(window.__accessFixture==='error')return new Response(null,{status:503});
      return response;
    }
    if(typeof path==='string' && path.startsWith('/api/evaluation/samples/'))window.__evaluationReads++;
    if(path==='/api/admin/auth/credentials' && options.method==='POST'){
      const command=JSON.parse(options.body);window.__credentialPosts.push(command.request_id);
      const response=await fetch(path,options);
      if(window.__loseIssueReceipt){window.__loseIssueReceipt=false;return new Proxy(response,{get(target,key){
        if(key==='json')return async()=>{let document=await target.json();document=null;throw Error('synthetic lost access receipt');};
        const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
      }});}
      return response;
    }
    return fetch(path,options);
  };
})()"""


def redacted(value):
    return re.sub(r'kza_[A-Za-z0-9_-]{43}', '[redacted]', json.dumps(value, ensure_ascii=False))


async def click_text(page, selector, text):
    await page.evaluate("(()=>{const target=[...document.querySelectorAll(" + json.dumps(selector)
        + ")].find(node=>node.textContent.trim()===" + json.dumps(text)
        + ");if(!target)throw Error('missing named control');target.click()})()")


async def escape(page):
    await page.command('Input.dispatchKeyEvent', {'type': 'keyDown', 'key': 'Escape', 'code': 'Escape', 'windowsVirtualKeyCode': 27})
    await page.command('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': 'Escape', 'code': 'Escape', 'windowsVirtualKeyCode': 27})


async def assert_no_system_ids(page: Page, selector: str, private_ids=()) -> None:
    leaks = await page.evaluate(r"""((selector, privateIds) => {
        // textContent can join a receipt and the next button without a separator.
        const ids = /(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|vf_[0-9a-f]{64})/i;
        const exactIds = [...privateIds, ...window.__credentialPosts];
        const containsId = value => ids.test(value) || exactIds.some(id => value.includes(id));
        const region = document.querySelector(selector);
        return {text: containsId(region.textContent), attributes: [region, ...region.querySelectorAll('*')]
            .some(element => [...element.attributes].some(attribute => containsId(attribute.value))),
            values: [...region.querySelectorAll('input[type=text], textarea')].some(element =>
                element.checkVisibility({visibilityProperty: true}) && containsId(element.value))};
    })(""" + json.dumps(selector) + ', ' + json.dumps(private_ids) + ')')
    assert leaks == {'text': False, 'attributes': False, 'values': False}, f'{selector}: identifiers in DOM: {leaks}'


async def watch_secret_dialog(page: Page) -> None:
    """Observe even receipt nodes removed before Chromium renders the dialog."""
    await page.evaluate(r"""(() => {
        const dialog = document.querySelector('#access-secret-dialog');
        const uuid = /[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}/i;
        let leaked = false, openings = 0;
        const inspect = records => {
            for (const record of records) {
                if (record.type === 'attributes' && record.target === dialog
                    && record.attributeName === 'open' && record.oldValue === null) openings++;
                const nodes = [...record.addedNodes, record.target];
                leaked ||= uuid.test(record.oldValue ?? '') || nodes.some(node =>
                    uuid.test(node.textContent) || [node, ...node.querySelectorAll?.('*') ?? []].some(element =>
                        [...element.attributes ?? []].some(attribute => uuid.test(attribute.value))));
            }
        };
        const observer = new MutationObserver(inspect);
        observer.observe(dialog, {subtree: true, childList: true, characterData: true, attributes: true,
            characterDataOldValue: true, attributeOldValue: true});
        window.__stopSecretDialogWatch = () => {
            inspect(observer.takeRecords());
            observer.disconnect();
            delete window.__stopSecretDialogWatch;
            return {leaked, openings};
        };
    })()""")


async def assert_safe_secret_delivery(page: Page) -> None:
    """Check the dialog opened once and introduced no UUID in its mutations."""
    observed = await page.evaluate('window.__stopSecretDialogWatch()')
    assert observed == {'leaked': False, 'openings': 1}, f'Credential dialog mutations: {observed}'
    assert await page.evaluate("document.querySelectorAll('#access-secret-dialog input').length === 1")


async def assert_local_sample_labels(page: Page, count: int) -> None:
    """Check the live field values and the separation from manual references."""
    await assert_no_system_ids(page, '#access-dialog')
    samples = await page.evaluate("""[...document.querySelectorAll('.access-sample input[type=text]')]
        .map(({value, readOnly, pattern}) => ({value, readOnly, pattern}))""")
    assert samples == [{'value': f'Muestra {n}', 'readOnly': True, 'pattern': ''}
                       for n in range(1, count + 1)], 'Expected local labels for the selected samples'


async def choose_person_option(page: Page, name: str) -> None:
    await page.evaluate("""(name => {
        const select = document.querySelectorAll('#access-dialog form select')[1];
        const option = [...select.options].find(option => option.textContent.split(' · Referencia ')[0] === name);
        if (!option) throw Error('missing option');
        select.value = option.value;
        select.dispatchEvent(new Event('change', {bubbles: true}));
    })(""" + json.dumps(name) + ')')


async def assert_system_homonyms(page: Page, expected: tuple[str, ...]) -> None:
    result = await page.evaluate("""(() => {
        const person = document.querySelectorAll('#access-dialog form select')[1];
        return [...person.options].slice(1).map(option => {
            person.value = option.value;
            person.dispatchEvent(new Event('change', {bubbles: true}));
            const reference = document.getElementById(person.getAttribute('aria-describedby'));
            return {label: option.textContent, localDescription: reference?.textContent || '',
                sameDialog: !reference || person.closest('dialog').contains(reference)};
        });
    })()""")
    for row in result:
        name, _, reference = row['label'].partition(' · Referencia ')
        assert row['sameDialog'] and (name in expected) == bool(reference), f'Person reference: {result}'
        assert row['localDescription'] == (f'Referencia {reference}' if reference else ''), f'Person description: {result}'
    assert len({row['localDescription'] for row in result if row['localDescription']}) == len(expected), result


async def audit_system_matrix(page: Page, name: str, selector: str) -> None:
    await capture_personas_matrix(page, EVIDENCE, name, audit_text=False)
    await page.evaluate(WCAG_SCRIPT)
    change_theme = """(value => {
        const select = document.querySelector('#theme-slot select');
        select.value = value; select.dispatchEvent(new Event('change'));
    })"""
    previous = await page.evaluate("document.querySelector('#theme-slot select').value")
    try:
        for theme in ('claro', 'oscuro'):
            await page.evaluate(change_theme + '(' + json.dumps(theme) + ')')
            if selector in ('#access-view', '#evaluation-view'):
                await assert_frame_consistent(page, selector, theme)
            for width in (1440, 768, 390, 320):
                await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900,
                    'deviceScaleFactor': 1, 'mobile': width <= 390})
                await page.evaluate("Promise.all(document.getAnimations().filter(animation => animation.constructor.name === 'CSSTransition').map(animation => animation.finished.catch(() => {})))")
                result = await page.evaluate(r"""(selector => {
                    const region = document.querySelector(selector);
                    const color = value => value.match(/[0-9.]+/g).map(Number);
                    const findings = [];
                    for (const el of region.querySelectorAll('h1,h2,h3,p,span,strong,pre,button,input,select,label')) {
                        if (!el.checkVisibility() || el.closest(':disabled,[aria-disabled=true]')) continue;
                        const style = getComputedStyle(el), size = parseFloat(style.fontSize);
                        const text = [...el.childNodes].some(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
                        const control = el.matches('button,select,input:not([type=checkbox])');
                        const rect = el.getBoundingClientRect();
                        if (control && (rect.width < 43.95 || rect.height < 43.95)) findings.push([el.localName, 'target']);
                        if (!text && !control) continue;
                        let background;
                        for (let current = el; current && !background; current = current.parentElement) {
                            const fill = getComputedStyle(current).backgroundColor;
                            if (fill.startsWith('rgb(')) background = color(fill);
                        }
                        const ratio = background ? window.__kzaWcag.contrast(color(style.color), background) : 0;
                        const large = size >= 24 || size >= 18.67 && Number(style.fontWeight) >= 700;
                        if (size < 14 || ratio < (large ? 3 : 4.5)) findings.push([el.localName, el.className, 'text', size, ratio, style.color, background]);
                    }
                    return {width: innerWidth, theme: document.documentElement.dataset.theme, findings};
                })(""" + json.dumps(selector) + ')')
                assert result == {'width': width, 'theme': theme, 'findings': []}, f'{name}: {result}'
                if selector.endswith('-dialog') and width in {390, 1440}:
                    await audit_current_accessibility(page, EVIDENCE, name, selector)
                if width == 320:
                    zoom = await page.evaluate('window.__kzaWcag.beginTextZoom()')
                    try:
                        assert zoom['document'] == 0 and zoom['dialog'] in (None, 0), f'{name} {theme} 200%: {zoom}'
                        await screenshot(page, EVIDENCE/'matrix'/f'{name}-{theme}-320-texto-200.png', full=False)
                        await audit_current_accessibility(page, EVIDENCE, f'{name}-text200',
                                                          selector if selector.endswith('-dialog') else None)
                    finally:
                        await page.evaluate('window.__kzaWcag.endTextZoom()')
    finally:
        await page.evaluate(change_theme + '(' + json.dumps(previous) + ')')
        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180,
            'deviceScaleFactor': 1, 'mobile': False})


async def assert_credential_references(page: Page) -> None:
    references = await page.evaluate("""(() => {
        const rows = [...document.querySelectorAll('#access-view [data-credential-id]')];
        return rows.filter(row => row.querySelector('strong').textContent === 'Synthetic duplicate access')
            .map(row => ({expires: row.querySelector('.row-meta').textContent,
                references: [...row.querySelectorAll('button')].map(button => {
                    const ref = document.getElementById(button.getAttribute('aria-describedby'));
                    return ref && row.contains(ref) ? ref.textContent : null;
                })}));
    })()""")
    for item in references:
        duplicate = sum(other['expires'] == item['expires'] for other in references) == 2
        assert all(bool(ref) == duplicate for ref in item['references']), references
    labels = [tuple(item['references']) for item in references]
    assert len(references) == 3 and len(set(labels)) == 3, references


async def assert_access_connection(page: Page) -> None:
    block = "document.querySelector('#access-view [aria-label=Conexión]')"
    reading = f"{block}.querySelectorAll('.row-meta')[1].textContent"
    freshness = f"{block}.querySelector('.tag').textContent"
    assert await page.evaluate(f"{block}.textContent.includes('Administración · Vence el')"
        f" && {freshness} === 'Datos disponibles' && {reading} !== 'Todavía sin lectura confirmada'")
    last_reading = await page.evaluate(reading)
    await page.evaluate("window.__accessFixture='hold'")
    await click_text(page, '#access-view button', 'Actualizar accesos')
    await page.wait_for("window.__accessCallbacks.length===1")
    assert await page.evaluate(f"{freshness} === 'Actualizando' && {reading} === {json.dumps(last_reading)}")
    await page.evaluate("window.__accessFixture='error';window.__accessCallbacks.splice(0).forEach(resolve=>resolve())")
    await page.wait_for(f"{freshness} === 'Consulta no disponible'")
    assert await page.evaluate(f"{reading} === {json.dumps(last_reading)} && document.querySelector('#add-access').disabled")
    await page.evaluate("window.__accessFixture=null")
    await click_text(page, '#access-view button', 'Actualizar accesos')
    await page.wait_for(f"{freshness} === 'Datos disponibles' && !document.querySelector('#add-access').disabled")


async def select_person(page: Page, name: str) -> None:
    await page.evaluate("""[...document.querySelectorAll('#people-grid [data-person]')]
        .find(card => card.querySelector('.row-title').textContent === """ + json.dumps(name) + """).click()""")


async def assert_linked_form_guard(page: Page, person: str, duplicate: str, scenario: str) -> None:
    await page.evaluate("document.querySelector('#rail-nav [data-section=people]').click()")
    await page.evaluate("document.querySelector('#refresh-people').click()")
    await page.wait_for("[...document.querySelectorAll('#people-grid .row-title')].some(title => title.textContent === " + json.dumps(duplicate) + ")")
    await select_person(page, person)
    await page.wait_for("""document.querySelector('[data-person-access] .rows:not([hidden])')
        ?.textContent.includes('Synthetic homonym access')""")
    marker = await page.evaluate('window.__credentialPosts.length')
    controls = await page.evaluate('''(() => {
        const scope = document.querySelector('[data-person-access]');
        const actions = [...scope.querySelectorAll('[data-credential-id] button')].map(button => button.textContent);
        scope.querySelector('#add-person-access')?.click();
        return { reason: scope.textContent.includes('Hay otra persona con el mismo nombre'),
            system: scope.querySelector('a')?.getAttribute('href') === '#accesos',
            remove: actions.includes('Retirar'), modify: actions.includes('Modificar'),
            create: !!scope.querySelector('#add-person-access'), form: !!document.querySelector('#access-dialog form') };
    })()''')
    assert controls == {'reason': True, 'system': True, 'remove': True,
        'modify': False, 'create': False, 'form': False}, f'{scenario}: unexpected homonym controls: {controls}'
    assert await page.evaluate(f'window.__credentialPosts.length === {marker}')


async def audit_linked_access(page: Page, evidence: Path, first: str, second: str) -> None:
    scope = '#person-detail [data-person-access]'
    await page.evaluate("document.querySelector('#rail-nav [data-section=people]').click()")
    await page.wait_for("!!document.querySelector('#people-grid [data-person][aria-pressed=true]')")
    await select_person(page, first)
    await page.wait_for("!!document.querySelector('#add-person-access:enabled')")
    await page.wait_for(f"document.querySelector('{scope}').textContent.includes('Esta persona no tiene credenciales')")
    await capture_personas_matrix(page, evidence, 'access-person-empty')
    await page.evaluate("window.__accessFixture='hold'")
    await click_text(page, f'{scope} button', 'Actualizar accesos')
    await page.wait_for(f"window.__accessCallbacks.length===1 && document.querySelector('{scope}').textContent.includes('Consultando accesos')")
    await capture_personas_matrix(page, evidence, 'access-person-loading')
    await page.evaluate("window.__accessFixture='error';window.__accessCallbacks.splice(0).forEach(resolve=>resolve())")
    await page.wait_for(f"document.querySelector('{scope}').textContent.includes('No pudimos consultar los accesos')")
    await capture_personas_matrix(page, evidence, 'access-person-unavailable')
    await page.evaluate("window.__accessFixture=null")
    await click_text(page, f'{scope} button', 'Actualizar accesos')
    await page.wait_for(f"!document.querySelector('#add-person-access').disabled && !document.querySelector('{scope}').textContent.includes('No pudimos consultar')")

    await page.evaluate("document.querySelector('#add-person-access').focus()")
    await press_enter(page)
    await page.wait_for("!!document.querySelector('#access-dialog[open] form')")
    assert await page.evaluate("!document.querySelector('#access-dialog select, #access-dialog .access-visit-preset')")
    assert await page.evaluate("document.querySelector('#access-dialog button[type=submit]').disabled")
    await page.evaluate('''(() => {
        const name = document.querySelector('#access-dialog input[type=text]');
        name.value = 'Synthetic linked access';
        name.dispatchEvent(new Event('input', { bubbles: true }));
    })()''')
    await page.wait_for("!!document.querySelector('#access-dialog[open] form button[type=submit]:enabled')")
    await capture_personas_matrix(page, evidence, 'access-person-create-dialog', dialog='#access-dialog')
    before_issue = await page.evaluate('window.__credentialPosts.length')
    await watch_secret_dialog(page)
    await page.evaluate("document.querySelector('#access-dialog form').requestSubmit()")
    await page.wait_for("!!document.querySelector('#access-secret-dialog[open] input[type=password]')")
    assert await page.evaluate("document.querySelector('#access-secret-dialog').textContent.includes('Se muestra una sola vez')")
    assert await page.evaluate(f'window.__credentialPosts.length === {before_issue} + 1')
    await assert_safe_secret_delivery(page)
    leaks = await page.evaluate(r'''(() => {
        const uuid = /\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b/i;
        const posted = window.__credentialPosts.at(-1);
        const dialog = document.querySelector('#access-secret-dialog');
        const includesId = value => value.includes(posted) || uuid.test(value);
        return { postIsUuid: uuid.test(posted), text: includesId(dialog.textContent),
            attributes: [dialog, ...dialog.querySelectorAll('*')].some(element =>
                [...element.attributes].some(attribute => includesId(attribute.value))) };
    })()''')
    assert leaks == {'postIsUuid': True, 'text': False, 'attributes': False}, f'Scoped secret dialog UUID check: {leaks}'
    assert await page.evaluate("!document.querySelector('#access-secret-dialog > .row-meta')")
    # This dialog still uses the System typography; capture its layout without applying Personas text tokens.
    await capture_personas_matrix(page, evidence, 'access-person-secret-dialog', audit_text=False,
                                  dialog='#access-secret-dialog')
    focus = await audit_modal_focus(page, '#access-secret-dialog')
    await escape(page)
    assert await page.evaluate("!document.querySelector('#access-secret-dialog').children.length")
    await audit_modal_return(page, evidence, 'access-person-secret-return', focus,
                             dialog='#access-secret-dialog', opener='#add-person-access')
    await page.wait_for(f"document.querySelector('{scope} [data-credential-id]')?.textContent.includes('Synthetic linked access')")
    await capture_personas_matrix(page, evidence, 'access-person-data')
    await audit_view(page, evidence, 'personas-accesos', (('#add-person-access', None),),
                     screenshot, widths=(1440, 390))
    await select_person(page, second)
    await page.wait_for(f"document.querySelector('{scope}').textContent.includes('Esta persona no tiene credenciales')")
    assert await page.evaluate(f"!document.querySelector('{scope} [data-credential-id]') && !document.querySelector('#access-secret-dialog').children.length")
    await select_person(page, first)
    await page.wait_for(f"!!document.querySelector('{scope} [data-credential-id] button:enabled')")
    remove = f'{scope} [data-credential-id] button.danger'
    await page.evaluate(f'document.querySelector({json.dumps(remove)}).focus()')
    await press_enter(page)
    await page.wait_for("!!document.querySelector('#access-dialog[open] .danger:enabled')")
    assert await page.evaluate("!document.querySelector('#access-dialog').textContent.includes('Para guardar, la cuenta debe estar activa')")
    await capture_personas_matrix(page, evidence, 'access-person-remove-dialog', dialog='#access-dialog')
    focus = await audit_modal_focus(page, '#access-dialog')
    await escape(page)
    assert await page.evaluate(f"!document.querySelector('#access-dialog').open && !!document.querySelector('{scope} [data-credential-id]')")
    await audit_modal_return(page, evidence, 'access-person-remove-return', focus,
                             dialog='#access-dialog', opener=remove)


async def main():
    os.umask(0o077)
    EVIDENCE.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-nexo-access-root'
    requests, checks = [], []
    phase = 'setup'

    class SafePage(Page):
        async def wait_for(self, expression):
            try:
                return await super().wait_for(expression)
            except AssertionError:
                state = await self.evaluate("({text:document.body.innerText,ready:document.readyState,dialogs:[...document.querySelectorAll('dialog')].map(d=>({id:d.id,open:d.open})),copyRequests:window.__copyRequests,posts:window.__credentialPosts?.length,reads:window.__evaluationReads,urls:window.__urls?.size,emptyCallbacks:window.__emptyCallbacks?.length})")
                (EVIDENCE/'failure.json').write_text(redacted({'phase': phase, 'condition': expression, 'state': state,
                    'requests': requests, 'runtime_error_count': len(self.errors)})+'\n')
                # Inputs containing secrets always remain masked in evidence.
                await self.evaluate("for(const input of document.querySelectorAll('input'))if(/^kza_/.test(input.value))input.type='password'")
                await screenshot(self, EVIDENCE/'failure.png')
                raise AssertionError('Browser condition failed in phase: '+phase) from None

    binary = browser_binary()
    browser_data = browser_profile_root()
    with tempfile.TemporaryDirectory(prefix='kza-nexo-access-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            users = None
            authority = DashboardAuthority(directory/'auth.db', session_ttl_s=600,
                account_provider=lambda: users.active_accounts() if users is not None else None)
            stack.push_async_callback(authority.aclose)
            await authority.initialize()
            root = authority.resolve('synthetic-nexo-access-root')
            operator = await authority.issue_credential(root, subject='synthetic-access-operator',
                role=DashboardRole.OPERATOR, expires_at=time.time()+600)
            clock = Clock(time.time())
            privacy = PrivacyServices(str(directory/'privacy.db'), clock=clock)
            stack.push_async_callback(privacy.aclose)
            await privacy.acall(privacy.initialize)
            owner = ErasureCoordinator(privacy)
            stack.push_async_callback(owner.aclose)
            bus = await build_live_services({}, privacy=privacy, owner=owner, shutdown=stack)
            voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
            users = await voices.load_accounts(str(directory/'legacy-users.json'),
                retirement_proof=lambda identities: authority.users_retired(identities)
                    and voices.ledger.people.user_links_cleared(identities))
            person_a, _ = await voices.acall(users.add_user, 'Synthetic homonym', PermissionLevel.ADMIN)
            person_b, _ = await voices.acall(users.add_user, 'Synthetic second person', PermissionLevel.ADMIN)
            assert person_a is not None and person_b is not None
            variant = '  SÝNTHETIC   HOMÓNÝM  '
            assert (await voices.acall(partial(users.update_user, person_b.user_id, name=variant)))[0]
            person_unique, _ = await voices.acall(users.add_user, 'Synthetic unique person', PermissionLevel.ADULT)
            assert person_unique is not None
            duplicates_expiry = time.time() + 480
            for expiry in (duplicates_expiry, duplicates_expiry, duplicates_expiry + 10):
                await authority.issue_credential(root, subject='Synthetic duplicate access', role=DashboardRole.HOME, expires_at=expiry)
            await authority.issue_credential(root, subject='Synthetic existing linked access', role=DashboardRole.HOME,
                expires_at=duplicates_expiry, user_id=person_unique.user_id)
            encoder = Encoder(); voices.register_encoder(encoder)
            for index in range(3):
                source = memory_source(now=clock.now, ttl=600)
                assert voices.observe(encoder, np.tile(AUDIO, 3 + index), (source,),
                    transcript=f'Synthetic evaluator phrase {index}', room_id='Synthetic private room')
                async with asyncio.timeout(5):
                    while voices.resource_status()['pending'] or voices.resource_status()['operations']:
                        await asyncio.sleep(.001)
            captures = (await voices.acall(voices.ledger.snapshot))['fingerprints']
            identities = [row['id'] for row in captures]
            assert len(identities) == 3
            dashboard = DashboardAPI(authority=authority, event_bus=bus, voice_registry=voices, user_manager=users)
            @dashboard.app.middleware('http')
            async def record(request, call_next):
                response = await call_next(request)
                requests.append({'method': request.method, 'path': request.url.path, 'status': response.status_code})
                return response
            listener = socket.socket(); listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            browser = None
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(.02)
                profile = directory/'chromium'
                with (EVIDENCE/'chromium.log').open('w') as output:
                    browser = subprocess.Popen([binary, '--headless', '--disable-gpu', '--use-angle=swiftshader',
                        '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking',
                        '--autoplay-policy=no-user-gesture-required', '--remote-debugging-port=0',
                        f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            assert browser.poll() is None
                            await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as client:
                        async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type'] == 'page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = SafePage(connection)
                        await page.command('Runtime.enable'); await page.command('Page.enable')
                        capture_rows = FixtureRows('#voice-dialog .review-sample', '.row-meta')
                        labels = [(capture['id'], f"{capture['duration_s']:.1f} s · ") for capture in captures]
                        await capture_rows.bind_many(page, labels, prefix=True)
                        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': INSTRUMENT})
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        await page.command('Page.navigate', {'url': f'http://127.0.0.1:{port}/#accesos'})
                        phase = 'operator login and home credential'
                        await page.wait_for("document.querySelector('#login-error')?.textContent.length>0")
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("!document.querySelector('#access-view').hidden && !document.querySelector('#add-access').disabled")
                        await assert_access_connection(page)
                        assert await page.evaluate("""!window.__personRequests.includes('/api/admin/users')
                            && document.querySelector('#access-view').textContent.includes('Persona vinculada')
                            && !document.querySelector('#access-view').textContent.includes('Synthetic unique person')""")
                        await page.evaluate("(()=>{const trigger=document.querySelector('#add-access');trigger.focus();trigger.click()})()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===4")
                        access_focus = await audit_modal_focus(page, '#access-dialog',
                            initial='#access-dialog .sistema-dialog-top button')
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-dialog').open")
                        await audit_modal_return(page, EVIDENCE, 'accesos-alta', access_focus,
                                                 dialog='#access-dialog', opener='#add-access')
                        assert not access_focus['findings'], access_focus['findings']
                        await page.evaluate("(()=>{const trigger=document.querySelector('#add-access');trigger.focus();trigger.click()})()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===4")
                        await assert_system_homonyms(page, ('Synthetic homonym', variant))
                        await assert_no_system_ids(page, '#access-dialog', [person_a.user_id, person_b.user_id, person_unique.user_id])
                        await audit_system_matrix(page, 'access-form', '#access-dialog')
                        await choose_person_option(page, variant)
                        await watch_secret_dialog(page)
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic browser home';form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open && document.querySelector('#access-secret-dialog input').type==='password'")
                        await assert_safe_secret_delivery(page)
                        await assert_no_system_ids(page, '#access-secret-dialog')
                        await assert_no_system_ids(page, '#access-view .access-receipts')
                        assert await page.evaluate(r"""(() => {
                            const receipt = document.querySelector('#access-secret-dialog > .row-meta');
                            return !!receipt && /^Operación \d+$/.test(receipt.textContent)
                                && [...document.querySelectorAll('#access-view .access-receipts .row-meta')]
                                    .some(row => row.textContent === receipt.textContent);
                        })()"""), 'The one-time delivery must match the operation in Receipts'
                        secret_focus = await audit_modal_focus(page, '#access-secret-dialog')
                        assert await page.evaluate("window.__copyRequests===0 && !document.querySelector('#token').value")
                        await screenshot(page, EVIDENCE/'access-secret-masked.png')
                        await audit_system_matrix(page, 'access-secret', '#access-secret-dialog')
                        await click_text(page, '#access-secret-dialog button', 'Copiar credencial')
                        await page.wait_for("window.__copyRequests===1 && window.__copiedShape && window.__copiedExact && document.querySelector('.access-secret-status').textContent.includes('copiada')")
                        await audit_status(page, EVIDENCE, 'accesos-credencial', 'copied', '.access-secret-status')
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#access-secret-dialog').children.length")
                        await audit_modal_return(page, EVIDENCE, 'accesos-credencial', secret_focus,
                                                 dialog='#access-secret-dialog', opener='#add-access')
                        assert not secret_focus['findings'], secret_focus['findings']
                        catalog = authority.commands.catalog(root)['rows']
                        home = next(row for row in catalog if row['subject'] == 'Synthetic browser home')
                        assert home['user_id'] == person_b.user_id and home['role'] == 'home'
                        checks.append('home grant uses selected canonical homonym; explicit masked secret copy and Escape retirement')
                        await screenshot(page, EVIDENCE/'access-desktop.png')
                        await assert_no_system_ids(page, '#access-view',
                            [home['id'], person_a.user_id, person_b.user_id, person_unique.user_id])
                        await assert_credential_references(page)
                        assert await page.evaluate("document.querySelector('#access-view').textContent.includes('Synthetic unique person')")
                        await audit_system_matrix(page, 'access-system', '#access-view')
                        checks.append('focus return and local receipts; normalized homonyms and references only on indistinguishable rows; System theme and text-zoom matrix')
                        phase = 'access reflow and accessibility audit'
                        audits = {'accesos': await audit_view(page, EVIDENCE, 'accesos', (
                            ('#app-nav [data-view=access]', None), ('#add-access', None),
                            ('[data-credential-id] button', 'Modificar'), ('#logout', None)), screenshot)}
                        for width in (390, 320):
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, EVIDENCE/f'access-{width}.png', full=False)
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        phase = 'modify and revoke'
                        row = "[...document.querySelectorAll('[data-credential-id] strong')].find(node => node.textContent === 'Synthetic browser home')"
                        await page.wait_for(f'!!({row})')
                        await page.evaluate(f"({row}).closest('.row').querySelector('button').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===4")
                        assert await page.evaluate("!document.querySelector('#access-dialog .access-visit-preset')")
                        await audit_system_matrix(page, 'access-edit', '#access-dialog')
                        await choose_person_option(page, 'Synthetic homonym')
                        await page.evaluate("document.querySelector('#access-dialog form').requestSubmit()")
                        await page.wait_for("!document.querySelector('#access-dialog').open && !document.querySelector('#add-access').disabled")
                        changed = next(row for row in authority.commands.catalog(root)['rows'] if row['id'] == home['id'])
                        assert changed['revision'] == home['revision']+1 and changed['user_id'] == person_a.user_id
                        await page.evaluate(f"({row}).closest('.row').querySelector('.btn.danger').click()")
                        removal_focus = await audit_modal_focus(page, '#access-dialog', initial='#access-dialog .row-actions button')
                        assert not removal_focus['findings'], removal_focus['findings']
                        await audit_system_matrix(page, 'access-remove', '#access-dialog')
                        await click_text(page, '#access-dialog button', 'Retirar acceso')
                        await page.wait_for(f'!({row}) && !document.querySelector("#add-access").disabled')
                        checks.append('real CAS modification and revocation')
                        phase = 'operator credential form'
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.evaluate("(()=>{const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic browser operator';const role=form.querySelector('select');role.value='operator';role.dispatchEvent(new Event('change'));form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open")
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#add-access').disabled")
                        checks.append('HOME and OPERATOR submit with irrelevant sample fields disabled')
                        phase = 'lost receipt'
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open")
                        before = await page.evaluate('window.__credentialPosts.length')
                        await page.evaluate("(()=>{window.__loseIssueReceipt=true;const form=document.querySelector('#access-dialog form');form.querySelector('input[type=text]').value='Synthetic lost receipt';form.dispatchEvent(new Event('input',{bubbles:true}));form.requestSubmit()})()")
                        await page.wait_for("document.querySelector('.access-receipts').textContent.includes('Cambio confirmado') && !document.querySelector('#add-access').disabled")
                        await click_text(page, '#access-view button', 'Actualizar accesos')
                        await page.wait_for("!document.querySelector('#add-access').disabled")
                        assert await page.evaluate('window.__credentialPosts.length==='+str(before+1)+" && !document.querySelector('#access-secret-dialog').open")
                        assert sum(row['subject'] == 'Synthetic lost receipt' for row in authority.commands.catalog(root)['rows']) == 1
                        checks.append('lost JSON receipt reconciles without second POST or secret')
                        phase = 'seven-day visitor preset'
                        await page.evaluate("document.querySelector('#add-access').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('#access-dialog form select')[1].options.length===4")
                        await choose_person_option(page, 'Synthetic homonym')
                        for width in (320, 390, 1440):
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900,
                                'deviceScaleFactor': 1, 'mobile': width <= 390})
                            assert await page.evaluate("""(() => {
                              const {width, height} = document.querySelector('.access-visit-preset').getBoundingClientRect();
                              return width >= 44 && height >= 44;
                            })()""")
                        # visitExpiry uses the browser's millisecond Date.now(): bound it with the same clock.
                        before_preset = await page.evaluate('Date.now()') / 1000
                        await click_text(page, '#access-dialog button', 'Visita (7 días)')
                        assert await page.evaluate("""(() => {
                          const form = document.querySelector('#access-dialog form');
                          const [role, person] = form.querySelectorAll('select');
                          const subject = form.querySelector('input[type=text]');
                          if (role.value !== 'home' || person.value !== '' || subject.value !== 'Visita') return false;
                          subject.value = 'Visita sintética';
                          form.querySelector('.access-visit-preset').click();
                          if (subject.value !== 'Visita sintética') return false;
                          form.requestSubmit();
                          return true;
                        })()""")
                        await page.wait_for("document.querySelector('#access-secret-dialog').open")
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#add-access').disabled")
                        visit = next(row for row in authority.commands.catalog(root)['rows'] if row['subject'] == 'Visita sintética')
                        assert visit['role'] == 'home' and visit['user_id'] is None
                        after_preset = await page.evaluate('Date.now()') / 1000
                        assert before_preset + 7 * 86400 <= visit['expires_at'] <= after_preset + 7 * 86400
                        checks.append('seven-day visitor preset clears the person, preserves a custom name and issues a real home grant')
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        phase = 'voice selection and native retirement before import'
                        await page.evaluate("document.querySelector('#rail-nav [data-section=people]').click();document.querySelector('#section-views [data-view=voices]').click()")
                        await page.wait_for("document.querySelectorAll('.review-sample').length===3")
                        await click_text(page, '#voices-content button', 'Revisar muestras')
                        for identity in identities[:2]:
                            await page.evaluate(f"document.querySelector({json.dumps(capture_rows.selector(identity, 'input'))}).click()")
                        await page.evaluate("document.querySelector('.review-sample button').click()")
                        await page.wait_for("document.querySelector('#voice-dialog audio')?.readyState>=2")
                        await page.evaluate("window.__player=document.querySelector('#voice-dialog audio');window.__player.muted=true;window.__player.play();for(const input of document.querySelectorAll('.voice-evaluation-selection input')){input.checked=true;input.dispatchEvent(new Event('change'))}window.__holdEmpty=true")
                        await click_text(page, '.voice-evaluation-selection button', 'Preparar acceso de evaluación')
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("document.querySelector('#access-view').hidden && !document.querySelector('#access-dialog').open")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('.access-sample').length===2")
                        await assert_local_sample_labels(page, 2)
                        await screenshot(page, EVIDENCE/'evaluation-scope-dialog.png')
                        await audit_system_matrix(page, 'access-evaluation-form', '#access-dialog')
                        phase = 'issue exact evaluator grant'
                        # Access orders selections by fingerprint before assigning the local labels.
                        audio_only_index = sorted(identities[:2]).index(identities[1])
                        await page.evaluate("""(audioOnly => {
                            const form = document.querySelector('#access-dialog form');
                            form.querySelector('input[type=text]').value = 'Synthetic evaluator access';
                            form.querySelectorAll('.access-sample')[audioOnly]
                                .querySelector('input[value=transcript_full]').checked = false;
                            form.dispatchEvent(new Event('input', {bubbles: true}));
                            form.requestSubmit();
                        })(""" + str(audio_only_index) + ')')
                        await page.wait_for("document.querySelector('#access-secret-dialog').open")
                        evaluator_token = await page.evaluate("document.querySelector('#access-secret-dialog input').value")
                        assert isinstance(evaluator_token, str) and evaluator_token.startswith('kza_')
                        granted = next(row for row in authority.commands.catalog(root)['rows'] if row['subject'] == 'Synthetic evaluator access')
                        assert {row['fingerprint_id']: row['modalities'] for row in granted['samples']} == {
                            identities[0]: ['audio_preview', 'transcript_full'], identities[1]: ['audio_preview']}
                        await escape(page)
                        await page.wait_for("!document.querySelector('#access-secret-dialog').open && !document.querySelector('#add-access').disabled")
                        phase = 'edit evaluator grant and validate an explicit manual reference'
                        evaluator_row = """[...document.querySelectorAll('[data-credential-id]')]
                            .find(row => row.querySelector('strong').textContent === 'Synthetic evaluator access')"""
                        await page.evaluate(f"({evaluator_row}).querySelector('button').click()")
                        await page.wait_for("document.querySelector('#access-dialog').open && document.querySelectorAll('.access-sample').length === 2")
                        await assert_local_sample_labels(page, 2)
                        await audit_system_matrix(page, 'access-evaluation-edit', '#access-dialog')
                        manual = granted['samples'][1]
                        await page.evaluate("""(() => {
                            const form = document.querySelector('#access-dialog form');
                            const sample = form.querySelectorAll('.access-sample')[1];
                            const deadline = sample.querySelector('input[type=datetime-local]').value;
                            sample.querySelector('button').click();
                            form.querySelector('.access-selections > button').click();
                            form.querySelectorAll('.access-sample')[1].querySelector('input[type=datetime-local]').value = deadline;
                        })()""")
                        assert await page.evaluate("""(modalities => {
                            const sample = document.querySelectorAll('.access-sample')[1];
                            const input = sample.querySelector('input[type=text]');
                            for (const mode of modalities) sample.querySelector('input[value=' + mode + ']').checked = true;
                            input.value = 'Muestra 2';
                            input.dispatchEvent(new Event('input', {bubbles: true}));
                            return !input.readOnly && input.required && input.validity.patternMismatch
                                && input.form.querySelector('button[type=submit]').disabled;
                        })(""" + json.dumps(manual['modalities']) + ')')
                        await page.evaluate("""(fingerprint => {
                            const form = document.querySelector('#access-dialog form');
                            const input = form.querySelectorAll('.access-sample')[1].querySelector('input[type=text]');
                            input.value = fingerprint;
                            input.dispatchEvent(new Event('input', {bubbles: true}));
                            form.requestSubmit();
                        })(""" + json.dumps(manual['fingerprint_id']) + ')')
                        await page.wait_for("!document.querySelector('#access-dialog').open && !document.querySelector('#add-access').disabled")
                        revised = next(row for row in authority.commands.catalog(root)['rows'] if row['id'] == granted['id'])
                        assert revised['revision'] == granted['revision'] + 1 and revised['samples'] == granted['samples']
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        checks.append('local labels for imported and existing samples; exact server grant on issue and edit; explicit manual validation')
                        phase = 'evaluator automatic manifest and audio'
                        evaluator_start = len(requests)
                        grant_order = [grant['fingerprint_id'] for grant in granted['samples']]
                        await login(page, evaluator_token, 'Evaluación temporal')
                        evaluator_token = None
                        await page.wait_for("!document.querySelector('#evaluation-view').hidden && document.querySelector('#evaluation-fingerprint')?.options.length===2 && !document.querySelector('#evaluation-fingerprint').disabled")
                        assert await page.evaluate(r"""(() => {
                            const view = document.querySelector('#evaluation-view');
                            const connection = view.querySelector('[aria-label=Conexión]');
                            const [session, reading] = connection.querySelectorAll('.row-meta');
                            return !view.querySelector('.sistema-evaluation-fallback').checkVisibility()
                                && session.textContent.startsWith('Evaluación temporal · Vence el')
                                && /\d/.test(reading.textContent)
                                && connection.querySelector('.tag').textContent === 'Datos disponibles';
                        })()"""), 'Confirmed Evaluation must show the active session and actual reading'
                        assert not any(row['path'].startswith(('/api/home', '/api/voices', '/api/admin/users', '/api/conversations')) for row in requests[evaluator_start:])
                        await page.evaluate("(()=>{const select=document.querySelector('#evaluation-fingerprint');select.value="+json.dumps(str(grant_order.index(identities[0]) + 1))+";select.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-modality').disabled && document.querySelector('#evaluation-modality').options.length===2")
                        await page.evaluate("(()=>{const mode=document.querySelector('#evaluation-modality');mode.value='audio_preview';mode.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-content button[type=submit]').disabled")
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("!!document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("window.__player=document.querySelector('[data-evaluation-sample] audio');window.__player.muted=true;window.__player.play()")
                        await page.wait_for('window.__player.readyState>=2 && !window.__player.paused')
                        assert await page.evaluate("!document.querySelector('.evaluation-transcript') && window.__urls.size===1")
                        phase = 'modality change holds real media retirement'
                        await page.evaluate("window.__holdEmpty=true;const mode=document.querySelector('#evaluation-modality');mode.value='transcript_full';mode.dispatchEvent(new Event('change'))")
                        await page.wait_for('window.__emptyCallbacks.length>0')
                        assert await page.evaluate("!document.querySelector('[data-evaluation-sample] audio') && document.querySelector('#evaluation-content button[type=submit]').disabled")
                        await page.evaluate("window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())")
                        await page.wait_for("!document.querySelector('#evaluation-content button[type=submit]').disabled && document.querySelector('#evaluation-modality').value==='transcript_full'")
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("document.querySelector('.evaluation-transcript')?.textContent.startsWith('Synthetic evaluator phrase')")
                        assert await page.evaluate("!document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        checks.append('audio-only payload; modality change drains native audio before separate transcript query')
                        await screenshot(page, EVIDENCE/'evaluation-desktop.png')
                        await assert_no_system_ids(page, '#evaluation-view', identities)
                        await audit_system_matrix(page, 'evaluation-system', '#evaluation-view')
                        phase = 'evaluation reflow and accessibility audit'
                        audits['evaluacion'] = await audit_view(page, EVIDENCE, 'evaluacion', (
                            ('#evaluation-fingerprint', None), ('#evaluation-modality', None),
                            ('#evaluation-content button[type=submit]', None), ('#logout', None)), screenshot)
                        phase = 'mobile evaluator layout'
                        for width in (390, 320):
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': True})
                            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                            await screenshot(page, EVIDENCE/f'evaluation-{width}.png', full=False)
                        phase = 'visibility retirement and no automatic payload replay'
                        reads_before = await page.evaluate('window.__evaluationReads')
                        await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'))")
                        await page.wait_for("!document.querySelector('.evaluation-transcript') && !document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        await page.evaluate("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
                        await page.wait_for("document.querySelector('#evaluation-fingerprint')?.options.length===2 && !document.querySelector('#evaluation-fingerprint').disabled")
                        assert await page.evaluate('window.__evaluationReads==='+str(reads_before)+" && !document.querySelector('.evaluation-transcript') && !document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("(()=>{const select=document.querySelector('#evaluation-fingerprint');select.value="+json.dumps(str(grant_order.index(identities[1]) + 1))+";select.dispatchEvent(new Event('change'))})()")
                        await page.wait_for("!document.querySelector('#evaluation-modality').disabled && document.querySelector('#evaluation-modality').options.length===1")
                        assert await page.evaluate("document.querySelector('#evaluation-modality').value==='audio_preview'")
                        phase = 'native revocation during playback'
                        await page.evaluate("document.querySelector('#evaluation-content form').requestSubmit()")
                        await page.wait_for("!!document.querySelector('[data-evaluation-sample] audio')")
                        await page.evaluate("window.__player=document.querySelector('[data-evaluation-sample] audio');window.__player.muted=true;window.__player.play()")
                        await page.wait_for('window.__player.readyState>=2 && !window.__player.paused')
                        await page.evaluate('window.__holdEmpty=true')
                        revoking = asyncio.create_task(authority.revoke_credential(root, granted['id']))
                        await page.wait_for("window.__emptyCallbacks.length>0 && !document.querySelector('[data-evaluation-sample] audio') && window.__urls.size===0")
                        assert await page.evaluate('window.__player.paused && !window.__player.isConnected')
                        await page.evaluate('window.__holdEmpty=false;window.__emptyCallbacks.splice(0).forEach(resolve=>resolve())')
                        assert await asyncio.wait_for(revoking, 5)
                        checks.append('native credential revocation removes playback and Blob while decoder retirement is held')
                        await page.evaluate('''window.__personRequests=[];window.__personSockets=[];
                            const target = location.hash;
                            history.replaceState({}, '', '#personas');
                            dispatchEvent(new PopStateEvent('popstate'));
                            history.replaceState({}, '', target);
                        ''')
                        assert await page.evaluate('''!document.querySelector('#people-view').children.length &&
                            !window.__personRequests.some(path => ['/api/admin/users', '/api/admin/auth/credentials', '/api/voices/profiles'].includes(path)) &&
                            !window.__personSockets.includes('/ws/voices')''')
                        await page.evaluate("document.querySelector('#logout').click()")
                        await page.wait_for("document.querySelector('#login-error').textContent==='La sesión ya no está vigente.' && !document.querySelector('#token').disabled")
                        await audit_status(page, EVIDENCE, 'accesos-sesion', '401', '#login-error')
                        assert await page.evaluate("!document.querySelector('#evaluation-content').children.length && window.__urls.size===0 && document.querySelector('#token').value===''")
                        assert not any(row['path'].startswith(('/api/home', '/api/voices', '/api/admin/users', '/api/conversations')) for row in requests[evaluator_start:])
                        checks.append('mobile 390/320; hidden view clears payload, show loads metadata only; audio-only second grant; logout clears')
                        await login(page, operator.token, 'Administración')
                        await page.wait_for("document.querySelector('#add-access') && !document.querySelector('#add-access').disabled")
                        assert await page.evaluate("!document.querySelector('#evaluation-content').children.length")
                        checks.append('logout confirms an already revoked session without blocking a new authorized login')
                        phase = 'person-linked access and isolation'
                        first_name, second_name = 'Synthetic access check A', 'Synthetic access check B'
                        person_audit, _ = await voices.acall(users.add_user, first_name, PermissionLevel.ADULT)
                        person_other, _ = await voices.acall(users.add_user, second_name, PermissionLevel.ADULT)
                        assert person_audit is not None and person_other is not None
                        await authority.issue_credential(root, subject='Synthetic homonym access', role=DashboardRole.HOME,
                            expires_at=time.time()+300, user_id=person_a.user_id)
                        for level, variant in ((PermissionLevel.ADMIN, 'Sýnthetic homónym'),
                                               (PermissionLevel.ADULT, '  SÝNTHETIC   HOMÓNÝM  ')):
                            assert (await voices.acall(partial(users.update_user, person_b.user_id,
                                name=variant, permission_level=level)))[0]
                            await assert_linked_form_guard(page, 'Synthetic homonym', variant, f'homonym at level {level.name}')
                        await audit_linked_access(page, EVIDENCE, first_name, second_name)
                        # System can reconcile the same operation through the original owner.
                        await click_text(page, '#person-detail [data-person-access] a', 'Ver todos los accesos y recibos')
                        await page.wait_for("document.querySelector('#access-view .access-receipts')?.textContent.includes('Crear')")
                        assert await page.evaluate("!document.querySelector('#access-secret-dialog').children.length")
                        checks.append('person-scoped issue, normalized homonyms at both levels, canonical isolation, one-time secret without UUID, removal confirmation and shared receipt owner')
                        assert not page.errors, 'Unexpected browser runtime errors'
                        result = {'passed': True, 'checks': checks, 'request_count': len(requests),
                            'credential_posts': await page.evaluate('window.__credentialPosts.length'),
                            'runtime_error_count': len(page.errors),
                            'visual_matrix': sorted(path.name for path in (EVIDENCE/'matrix').glob('*.png')),
                            'screenshots': ['access-secret-masked.png', 'evaluation-390.png', 'evaluation-320.png'],
                            'reflow_widths': {view: sorted(audit['widths'], key=int) for view, audit in audits.items()},
                            'a11y_evidence': [f'a11y-{view}.json' for view in audits]}
                        (EVIDENCE/'result.json').write_text(json.dumps(result, indent=2)+'\n')
                        print(json.dumps(result))
            finally:
                if browser is not None and browser.poll() is None:
                    browser.terminate(); await asyncio.to_thread(browser.wait, 10)
                server.should_exit = True
                await serving
                await dashboard.stop()


def test_nexo_access_browser(monkeypatch):
    """Run the existing browser workflow when this file is selected in pytest."""
    monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'synthetic-nexo-access-root')
    mask = os.umask(0o077)
    try:
        asyncio.run(main())
    finally:
        os.umask(mask)


if __name__ == '__main__':
    asyncio.run(main())
