"""Offline bootstrap retains exact pinned bytes and distribution notices."""

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT/'src/dashboard/frontend/obs/index.html'
MANIFEST = json.loads((ROOT/'docs/research/2026-10-27-dashboard-vendor.json').read_text())


class Resources(HTMLParser):
    """Collect active bootstrap requests, excluding ordinary navigation links."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def handle_starttag(self,tag,attrs):
        values = dict(attrs)
        if tag=='script' and values.get('src'):
            self.requests.append(values)
        if tag=='link' and values.get('href') and values.get('rel') in ('stylesheet','preconnect','preload'):
            self.requests.append(values)


@pytest.mark.parametrize('package',MANIFEST['packages'],ids=lambda package:package['package'])
def test_local_bundle_matches_original_sri_and_retains_its_license(package):
    parser = Resources()
    parser.feed(INDEX.read_text())
    asset = ROOT/package['asset']
    relative = asset.relative_to(INDEX.parent).as_posix()
    entry = next(item for item in parser.requests if item.get('src')==relative)
    data = asset.read_bytes()
    digest = base64.b64encode(hashlib.sha384(data).digest()).decode()
    assert entry['integrity']=='sha384-'+package['sha384']=='sha384-'+digest
    assert hashlib.sha256(data).hexdigest()==package['sha256']
    license_data = (ROOT/package['license']).read_bytes()
    assert hashlib.sha256(license_data).hexdigest()==package['license_sha256']
    assert b'Copyright' in license_data and b'Permission is hereby granted' in license_data


def test_bootstrap_has_no_external_script_stylesheet_or_font_requests():
    parser = Resources()
    parser.feed(INDEX.read_text())
    assert parser.requests
    assert all(not urlsplit(item.get('src') or item.get('href')).netloc for item in parser.requests)
