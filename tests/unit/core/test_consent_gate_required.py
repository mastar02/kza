"""OQ-8 N4: every client constructor of a consented CLOUD destination requires the gate.

One test per construction site/class, plus a static check that every
in-repository construction passes ``consent_gate=``. Evaluation-only
EgressPolicy objects (src/llm/cloud_consent.py) are not clients.
"""
import ast
from pathlib import Path

import pytest

from src.core.egress import CloudConsentGate

ROOT = Path(__file__).resolve().parents[3]
CLOUD = "https://gateway.test/v1"
LOCAL = "http://127.0.0.1:8101/v1"


def _required(factory):
    with pytest.raises(ValueError, match="^consent_gate_required$"):
        factory(None)
    gated = factory(CloudConsentGate())
    return gated


def test_http_reasoner_requires_the_gate():
    from src.llm.reasoner import HttpReasoner
    value = _required(lambda gate: HttpReasoner(base_url=CLOUD, cloud_consent=True, consent_gate=gate))
    assert value.policy.consent_gate is not None
    assert HttpReasoner(base_url=CLOUD, cloud_consent=False).policy.consent_gate is None
    assert HttpReasoner(base_url=LOCAL, destination="local", cloud_consent=True).policy.allowed


def test_fast_router_requires_the_gate():
    from src.llm.reasoner import FastRouter
    _required(lambda gate: FastRouter(base_url=CLOUD, cloud_consent=True, consent_gate=gate))
    assert FastRouter(base_url=LOCAL, destination="local", cloud_consent=True).policy.allowed


def test_hermes_reasoner_requires_the_gate():
    from src.llm.hermes_reasoner import HermesCliReasoner
    _required(lambda gate: HermesCliReasoner(cloud_consent=True, consent_gate=gate))


@pytest.mark.parametrize("name", ["RemoteLLMProvider", "RemoteRouterProvider", "RemoteSTTProvider",
                                  "RemoteTTSProvider"])
def test_remote_providers_require_the_gate(name):
    from src.providers import remote
    config = _required(lambda gate: remote.RemoteProviderConfig(url=CLOUD, cloud_consent=True, consent_gate=gate))
    provider = getattr(remote, name)(config)
    assert provider._config.egress_policy.consent_gate is config.consent_gate


@pytest.mark.parametrize("factory", ["create_llm_provider", "create_router_provider"])
def test_provider_factory_sites_require_the_gate(factory):
    from src.providers import factory as module
    config = {"providers": {"llm": {"mode": "remote", "url": CLOUD}, "router": {"mode": "remote", "url": CLOUD}},
              "reasoner": {"cloud": {"consent": True}}}
    with pytest.raises(ValueError, match="^consent_gate_required$"):
        getattr(module, factory)(config)
    gate = CloudConsentGate()
    assert getattr(module, factory)(config, consent_gate=gate)._config.consent_gate is gate


def test_provider_factory_local_fast_router_requires_the_gate():
    from src.providers import factory as module
    config = {"router": {"base_url": CLOUD, "destination": "cloud"}, "reasoner": {"cloud": {"consent": True}}}
    with pytest.raises(ValueError, match="^consent_gate_required$"):
        module.create_router_provider(config)
    gate = CloudConsentGate()
    assert module.create_router_provider(config, consent_gate=gate).policy.consent_gate is gate


def test_code_index_card_generator_requires_the_gate():
    from src.code_index.cards import CardGenerator
    _required(lambda gate: CardGenerator(base_url=CLOUD, model="m", cloud_consent=True, consent_gate=gate))


def test_operator_script_client_requires_the_gate(monkeypatch):
    import openai
    from scripts import sync_ha_to_chroma
    from types import SimpleNamespace
    listing = SimpleNamespace(data=[SimpleNamespace(id="m")])
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(list=lambda: listing)))
    _required(lambda gate: sync_ha_to_chroma.VLLMClient(CLOUD, "m", cloud_consent=True, consent_gate=gate))


CLIENTS = {"HttpReasoner", "FastRouter", "HermesCliReasoner", "CardGenerator", "VLLMClient",
           "RemoteProviderConfig"}


def test_every_in_repository_construction_passes_the_gate():
    """A future caller that omits the gate fails here (and at runtime when consented)."""
    missing, seen = [], []
    for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name not in CLIENTS:
                continue
            where = f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
            seen.append(where)
            if not any(keyword.arg == "consent_gate" for keyword in node.keywords):
                missing.append(where)
    assert missing == []
    assert any(site.startswith("src/main.py") for site in seen) and len(seen) >= 9
