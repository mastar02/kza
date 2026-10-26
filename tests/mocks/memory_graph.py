"""Real temporary Chroma, with deterministic local embeddings and I/O gates."""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
import numpy as np


@dataclass
class Clock:
    now: float = 2_000_000_000.0

    def __call__(self):
        return self.now


class SyntheticEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        return [np.array([1, len(text) / 100, sum(map(ord, text)) % 100 / 100], dtype=np.float32) for text in input]

    @staticmethod
    def name():
        return 'kza-synthetic-memory-embedding'

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return SyntheticEmbedding()


class ObservedIndex:
    def __init__(self, collection):
        self.collection = collection
        self.before_upsert = None
        self.delete_error = None
        self.query_error = None
        self.upsert_error = None

    def upsert(self, **kwargs):
        if self.before_upsert:
            self.before_upsert()
        if self.upsert_error:
            raise self.upsert_error
        return self.collection.upsert(**kwargs)

    def delete(self, **kwargs):
        if self.delete_error:
            raise self.delete_error
        return self.collection.delete(**kwargs)

    def query(self, **kwargs):
        if self.query_error:
            raise self.query_error
        return self.collection.query(**kwargs)

    def count(self):
        return self.collection.count()

    def get(self, **kwargs):
        return self.collection.get(**kwargs)


class MemoryClient:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collections = {}
        self.requested = []

    def get_or_create_collection(self, name, **kwargs):
        self.requested.append(name)
        if name not in self.collections:
            collection = self.client.get_or_create_collection(name=name, embedding_function=SyntheticEmbedding(), **kwargs)
            self.collections[name] = ObservedIndex(collection)
        return self.collections[name]
