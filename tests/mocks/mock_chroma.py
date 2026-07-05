"""Mock in-memory de una colección Chroma (API mínima usada por code_index)."""

import math


class MockChromaCollection:
    """Colección Chroma fake: almacenamiento dict + cosine por fuerza bruta."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._store: dict[str, dict] = {}  # id -> {embedding, document, metadata}

    def add(self, ids, embeddings, documents, metadatas):
        """Agregar ids nuevos. Espeja Chroma real: rechaza duplicados (no upsert)."""
        seen: set[str] = set()
        for id_ in ids:
            if id_ in seen or id_ in self._store:
                raise ValueError("Expected IDs to be unique")
            seen.add(id_)
        for i, id_ in enumerate(ids):
            self._store[id_] = {
                "embedding": list(embeddings[i]),
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def delete(self, ids=None, where=None):
        if ids is not None:
            for id_ in ids:
                self._store.pop(id_, None)
        if where is not None:
            matches = [
                id_ for id_, e in self._store.items()
                if all(e["metadata"].get(k) == v for k, v in where.items())
            ]
            for id_ in matches:
                self._store.pop(id_)

    def get(self, ids=None, include=None):
        found = [i for i in (ids or list(self._store)) if i in self._store]
        return {
            "ids": found,
            "documents": [self._store[i]["document"] for i in found],
            "metadatas": [self._store[i]["metadata"] for i in found],
        }

    def query(self, query_embeddings, n_results=8, include=None):
        q = query_embeddings[0]

        def dist(id_):
            e = self._store[id_]["embedding"]
            dot = sum(a * b for a, b in zip(q, e))
            nq = math.sqrt(sum(a * a for a in q)) or 1.0
            ne = math.sqrt(sum(a * a for a in e)) or 1.0
            return 1.0 - dot / (nq * ne)

        ranked = sorted(self._store, key=dist)[:n_results]
        return {
            "ids": [ranked],
            "documents": [[self._store[i]["document"] for i in ranked]],
            "metadatas": [[self._store[i]["metadata"] for i in ranked]],
            "distances": [[dist(i) for i in ranked]],
        }

    def count(self):
        return len(self._store)
