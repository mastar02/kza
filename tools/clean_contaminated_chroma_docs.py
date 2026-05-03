"""Detecta y (opcional) borra docs contaminados en home_assistant_commands.

Decisión 3-A (sesión 2026-05-03): el sync genera variantes sintéticas de
frases ("data augmentation"). Para entities específicas de room (light.cuarto,
light.escritorio, etc.) algunas variantes salen SIN mención del room
("Prendé las luces, tú."). Esos docs se vuelven attractors para queries
genéricas y enrutan acciones al room equivocado.

Este script:
  1. Lista todas las entities con `area` en metadata.
  2. Para cada entity, encuentra docs cuyo texto NO mentions
     - su `area`/`friendly_name`
     - ni ninguno de los aliases conocidos del room.
  3. Imprime un reporte de los candidatos.
  4. Si --apply, los borra (manda confirm explícito).

Uso:
    .venv/bin/python tools/clean_contaminated_chroma_docs.py        # dry-run
    .venv/bin/python tools/clean_contaminated_chroma_docs.py --apply
    .venv/bin/python tools/clean_contaminated_chroma_docs.py --chroma-path /path
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict


# Aliases por area conocidos. Mantener sincronizado con
# src/orchestrator/dispatcher._ROOM_ALIASES_TO_AREA y
# src/rooms/room_context.create_default_rooms().
AREA_ALIASES: dict[str, tuple[str, ...]] = {
    "Living": ("living", "sala", "salon"),
    "Escritorio": ("escritorio", "oficina", "estudio"),
    "Hall": ("hall", "pasillo", "entrada"),
    "Cocina": ("cocina", "kitchen"),
    "Baño": ("bano", "bathroom"),
    "Cuarto": ("cuarto", "dormitorio", "habitacion"),
    "Balcon": ("balcon",),
    "Pasillo": ("pasillo",),
    "Escalera": ("escalera", "escaleras"),
    "Escaleras": ("escaleras", "escalera"),
}


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _doc_mentions_area(doc: str, area: str, friendly: str) -> bool:
    """True si el doc menciona literal el area, friendly_name, o algún alias."""
    norm = _normalize(doc)
    candidates: list[str] = []
    if friendly:
        candidates.append(_normalize(friendly))
    if area:
        candidates.append(_normalize(area))
    candidates.extend(AREA_ALIASES.get(area, ()))
    for c in candidates:
        if not c:
            continue
        if re.search(rf"\b{re.escape(c)}\b", norm):
            return True
    return False


def find_contaminated(client, collection_name: str) -> list[tuple[str, str, dict, str]]:
    """Returns list of (id, entity_id, metadata, doc) for contaminated entries."""
    col = client.get_collection(collection_name)
    data = col.get(limit=10000, include=["metadatas", "documents"])
    contaminated: list[tuple[str, str, dict, str]] = []
    for doc_id, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        meta = meta or {}
        area = meta.get("area")
        entity_id = meta.get("entity_id", "")
        # Solo entities asignadas a un area específico — entities sin area
        # (genéricas tipo light.hogar) no aplican.
        if not area or not entity_id:
            continue
        friendly = meta.get("friendly_name", "")
        if not _doc_mentions_area(doc, area, friendly):
            contaminated.append((doc_id, entity_id, meta, doc))
    return contaminated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chroma-path",
        default="./data/chroma_db",
        help="Path al ChromaDB persistente",
    )
    parser.add_argument(
        "--collection",
        default="home_assistant_commands",
        help="Nombre de la collection",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Borrar los docs contaminados (sin esto, dry-run).",
    )
    args = parser.parse_args()

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb no está instalado. Activá .venv del proyecto.")
        return 1

    client = chromadb.PersistentClient(path=args.chroma_path)
    contaminated = find_contaminated(client, args.collection)

    if not contaminated:
        print("No se encontraron docs contaminados. ✓")
        return 0

    by_entity: dict[str, list] = defaultdict(list)
    for doc_id, entity_id, meta, doc in contaminated:
        by_entity[entity_id].append((doc_id, meta, doc))

    total_docs = sum(len(v) for v in by_entity.values())
    print(f"Encontrados {total_docs} docs contaminados en {len(by_entity)} entities:\n")

    for entity_id, items in sorted(by_entity.items()):
        area = items[0][1].get("area", "?")
        print(f"  {entity_id} (area={area}) — {len(items)} doc(s):")
        for _doc_id, _meta, doc in items[:5]:
            print(f"    · {doc[:120]}")
        if len(items) > 5:
            print(f"    ... y {len(items) - 5} más")

    if not args.apply:
        print(f"\n[DRY-RUN] Pasá --apply para borrar los {total_docs} docs.")
        return 0

    col = client.get_collection(args.collection)
    ids_to_delete = [doc_id for doc_id, _, _, _ in contaminated]
    print(f"\nBorrando {len(ids_to_delete)} docs de {args.collection}...")
    col.delete(ids=ids_to_delete)
    print("✓ Borrados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
