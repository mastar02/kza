"""Controllable external classifier/index boundaries for private job races."""

from copy import deepcopy
import json
import threading


class Classifier:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.prompts = []

    def generate(self, prompts, *args):
        self.prompts.extend(prompts)
        self.entered.set()
        assert self.release.wait(3)
        return [json.dumps({'is_command': True, 'intent': 'turn_on', 'entity_hint': 'luz sintética',
            'confidence': .99, 'slots': {}})]


class Index:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.command = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.synthetic',
            'description': 'Luz sintética encendida', 'similarity': .99, 'data': None}

    def search_command(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(3)
        return deepcopy(self.command)
