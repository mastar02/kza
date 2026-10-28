"""
Vector Database Sync Module
Sincronización de comandos de Home Assistant con ChromaDB
"""

import asyncio
import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import Context
from datetime import datetime
from functools import wraps
import threading

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.core.owned_work import ClosePhase, OwnedBlockingWork, WorkCloseTimeout

logger = logging.getLogger(__name__)


def _owned_operation(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with self._operation():
            return function(self, *args, **kwargs)
    return invoke


class _CollectionView:
    """Expose operations through the owner, never an unleased SDK collection."""

    def __init__(self, owner, attribute):
        self._owner, self._attribute = owner, attribute

    def _call(self, method, *args, **kwargs):
        with self._owner._operation():
            collection = getattr(self._owner, self._attribute)
            if collection is None or not self._owner._ready:
                self._owner.initialize()
                collection = getattr(self._owner, self._attribute)
            return getattr(collection, method)(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self._call('query', *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._call('get', *args, **kwargs)

    def add(self, *args, **kwargs):
        return self._call('add', *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._call('delete', *args, **kwargs)

    def count(self):
        return self._call('count')


class _EmbedderView:
    """Keep encoding inside the same native lifetime as the SDK."""

    def __init__(self, owner):
        self._owner = owner

    def encode(self, *args, **kwargs):
        with self._owner._operation():
            if self._owner._embedder is None or not self._owner._ready:
                self._owner.initialize()
            return self._owner._embedder.encode(*args, **kwargs)


class ChromaSync:
    """Sincronización de Home Assistant con ChromaDB"""

    # Bonus aplicado a la similarity cuando metadata.area matchea prefer_area.
    # Calibrado para que un candidato con area-match supere por margen claro a
    # otros con base_sim hasta ~0.10 más alta, pero no rescate candidatos con
    # base_sim mucho menor. Valor 0.15 ≈ 7.5% del rango cosine [0,2] mapeado a
    # similarity. Ver bug 2026-05-03 (light.cuarto vs light.escritorio,
    # diferencia de distancias 0.34 vs 0.30 → similarities 0.83 vs 0.85).
    # 0.15 → 0.35 (2026-06-04): el garble far-field del STT ('prender a luz',
    # 'la luz de la vida') matcheaba docs de OTRAS rooms con gap > 0.15 y
    # prendía living/balcón desde el escritorio. Si hay candidato del área
    # del mic sobre threshold, debe ganar. El cross-room explícito no se
    # rompe: cuando el texto menciona la room, prefer_area sale del TEXTO
    # (prioridad en _resolve_prefer_area), no de la zona del mic.
    PREFER_AREA_BOOST: float = 0.35

    def __init__(
        self,
        chroma_path: str,
        embedder_model: str,
        embedder_device: str,
        commands_collection: str = "home_assistant_commands",
        routines_collection: str = "home_assistant_routines",
        excluded_entities: list[str] | None = None,
        excluded_patterns: list[str] | None = None,
    ):
        self.chroma_path = chroma_path
        self.embedder_model = embedder_model
        self.embedder_device = embedder_device
        self.commands_collection_name = commands_collection
        self.routines_collection_name = routines_collection

        # Entidades excluidas del vector search. Default: light.hogar (grupo
        # global que prendería TODA la casa con "la luz" sin room). Configurable
        # desde settings.yaml (vectordb.exclude_entities / exclude_patterns) —
        # ej. ocultar bombillas miembro de un grupo Z2M (light.escritorio\d+)
        # dejando solo el grupo (light.escritorio). Ver memoria
        # project_lights_zigbee2mqtt_migration_2026-05-31.
        self._excluded_entities = set(
            excluded_entities if excluded_entities is not None else ["light.hogar"]
        )
        self._excluded_patterns = []
        for pat in (excluded_patterns or []):
            try:
                self._excluded_patterns.append(re.compile(pat))
            except re.error as e:
                logger.warning(f"Patrón de exclusión inválido, ignorado: {pat!r} ({e})")

        self._client = None
        self._commands_collection = None
        self._routines_collection = None
        self._embedder = None
        self._ready = False
        self._admission_closed = False
        self._active = 0
        self._condition = threading.Condition()
        self._local = threading.local()
        self._initialize_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._client_closed = False
        self._client_failed = False
        self._closing = None
        self._phase = ClosePhase.IDLE
        self._async_work = OwnedBlockingWork()
        self._commands_view = _CollectionView(self, '_commands_collection')
        self._routines_view = _CollectionView(self, '_routines_collection')
        self._embedder_view = _EmbedderView(self)

    def _is_excluded(self, entity_id: str) -> bool:
        """True si la entidad no debe indexarse en el vector search (lista
        explícita o algún patrón regex de exclusión)."""
        if entity_id in self._excluded_entities:
            return True
        return any(p.search(entity_id) for p in self._excluded_patterns)

    @contextmanager
    def _operation(self):
        # The condition guards only counters; no SDK, encoder or I/O holds it.
        depth = getattr(self._local, 'depth', 0)
        with self._condition:
            if depth == 0:
                if self._admission_closed:
                    raise RuntimeError('chroma_sync_closed')
                self._active += 1
            self._local.depth = depth + 1
        try:
            yield
        finally:
            with self._condition:
                self._local.depth = depth
                if depth == 0:
                    self._active -= 1
                    self._condition.notify_all()

    @_owned_operation
    def initialize(self) -> None:
        """Initialize once, closing our client on a partial construction failure."""
        failed = False
        with self._initialize_lock:
            if self._ready:
                return
            # An admitted operation can still finish while shutdown waits for it.
            # A failed initializer must never overwrite its surviving SDK handle.
            if self._admission_closed:
                raise RuntimeError('chroma_sync_closed')
            try:
                Context().run(self._construct)
                self._ready = True
            except Exception:
                failed = True
                self._admission_closed = True
            if failed:
                # Still inside the initializer's native ownership. Other admitted
                # initializers wait on this lock and cannot acquire a second SDK.
                self._close_client()
        if failed:
            raise RuntimeError('chroma_sync_initialization_failed') from None

    def _construct(self) -> None:
        self._client = chromadb.PersistentClient(path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False))
        self._commands_collection = self._client.get_or_create_collection(
            name=self.commands_collection_name, metadata={"hnsw:space": "cosine"})
        self._routines_collection = self._client.get_or_create_collection(
            name=self.routines_collection_name, metadata={"hnsw:space": "cosine"})
        self._embedder = SentenceTransformer(self.embedder_model, device=self.embedder_device)

    @property
    def commands(self):
        """Command operations retained through actual native completion."""
        return self._commands_view

    @property
    def routines(self):
        """Routine operations retained through actual native completion."""
        return self._routines_view

    @property
    def embedder(self):
        """Encoding operations retained through actual native completion."""
        return self._embedder_view

    def _close_client(self) -> None:
        failed = False
        if self._client_failed:
            raise RuntimeError('chroma_sync_sdk_close_failed')
        if self._client is not None and not self._client_closed:
            try:
                self._client.close()
            except Exception:
                self._client_failed = failed = True
            if failed:
                raise RuntimeError('chroma_sync_sdk_close_failed') from None
            self._client_closed = True
        self._client = self._commands_collection = self._routines_collection = self._embedder = None
        self._ready = False

    def close(self) -> None:
        """Drain real native users and close our SDK reference on a worker thread."""
        if getattr(self._local, 'depth', 0):
            raise RuntimeError('chroma_sync_close_inside_operation')
        self._admission_closed = True
        with self._close_lock:
            with self._condition:
                while self._active:
                    self._condition.wait()
            self._phase = ClosePhase.RESOURCES
            self._close_client()
            self._phase = ClosePhase.COMPLETE

    def begin_close(self) -> asyncio.Task:
        """Close admission now and retain actual cleanup independently of callers."""
        self._admission_closed = True
        if self._closing is None:
            draining = self._async_work.begin_close()
            self._phase = ClosePhase.DRAIN
            async def finish():
                await draining
                await asyncio.to_thread(self.close)
            self._closing = asyncio.create_task(finish(), context=Context())
            self._closing.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while keeping actual SDK shutdown owned."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                async with asyncio.timeout(1):
                    await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout('chroma_sync_still_running', phase=self._phase,
                    diagnostics=self.resource_status()) from error
        closing.result()
        if cancelled:
            raise asyncio.CancelledError

    def resource_status(self) -> dict:
        """Report counters and phases without private paths, queries or SDK reprs."""
        return {'closed': self._admission_closed, 'operations': self._active,
            'async_operations': self._async_work.status()['operations'],
            'client_closed': self._client_closed, 'client_close_failed': self._client_failed,
            'close_phase': self._phase.value}

    @_owned_operation
    def warmup_embedder(self) -> float:
        """Forzar la inicialización lazy del embedder y compilar kernels.

        Usa la property `embedder` (que llama initialize() si hace falta) en
        vez de leer `_embedder` directo — así el warmup NO se saltea cuando el
        embedder todavía no fue materializado (bug 2026-05-29: el guard lazy en
        main.py daba False y el primer comando pagaba el cold start de ~48ms).

        Returns:
            Latencia del encode dummy en ms (para logging del warmup).
        """
        start = time.perf_counter()
        self.embedder.encode(["warmup"])
        return (time.perf_counter() - start) * 1000

    # ==================== Sincronización de Comandos ====================
    
    @_owned_operation
    def sync_commands(
        self,
        ha_client,
        llm_reasoner,
        max_entities: int = 100
    ) -> int:
        """
        Sincronizar comandos de Home Assistant
        
        Args:
            ha_client: Cliente de Home Assistant
            llm_reasoner: LLM para generar descripciones
            max_entities: Máximo de entidades a procesar por batch
        
        Returns:
            Número de frases indexadas
        """
        logger.info(f"Iniciando sincronización ({datetime.now().strftime('%H:%M:%S')})")
        start_time = time.time()
        
        # 1. Obtener datos de HA
        logger.info("Consultando Home Assistant...")
        entities = ha_client.get_domotics_entities()
        services_by_domain = ha_client.get_services_by_domain()
        
        logger.info(f"Encontradas {len(entities)} entidades de domótica")
        
        # 2. Limpiar colección anterior
        existing = self.commands.get()
        if existing["ids"]:
            logger.info(f"Eliminando {len(existing['ids'])} registros anteriores...")
            self.commands.delete(ids=existing["ids"])
        
        # 3. Generar descripciones con LLM
        logger.info("Generando descripciones con LLM...")
        
        all_phrases = []

        for i, entity in enumerate(entities[:max_entities]):
            entity_id = entity["entity_id"]
            domain = entity_id.split(".")[0]
            services = services_by_domain.get(domain, [])

            if not services:
                continue

            # Excluidas por config (lista + patrones). Default light.hogar
            # (grupo global). Ver _is_excluded / settings.yaml vectordb.
            if self._is_excluded(entity_id):
                logger.info(f"Skip {entity_id} (excluido del vector search)")
                continue
            
            logger.debug(f"[{i+1}/{len(entities)}] {entity_id}")
            
            # Generar descripciones
            commands = self._generate_command_descriptions(
                entity, services, llm_reasoner
            )
            
            for cmd in commands:
                for phrase in cmd.get("phrases", []):
                    all_phrases.append({
                        "phrase": phrase,
                        "entity_id": cmd["entity_id"],
                        "domain": cmd["domain"],
                        "service": cmd["service"],
                        "description": cmd["description"],
                        "data": cmd.get("data", {})
                    })
        
        logger.info(f"Generadas {len(all_phrases)} frases")
        
        # 4. Crear embeddings y guardar
        logger.info("Generando embeddings...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, item in enumerate(all_phrases):
            embedding = self.embedder.encode(item["phrase"]).tolist()
            
            ids.append(f"cmd_{i}")
            embeddings.append(embedding)
            documents.append(item["phrase"])
            metadatas.append({
                "entity_id": item["entity_id"],
                "domain": item["domain"],
                "service": item["service"],
                "description": item["description"],
                "data": json.dumps(item.get("data", {}))
            })
        
        # Guardar en batches
        BATCH_SIZE = 500
        for i in range(0, len(ids), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(ids))
            self.commands.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                documents=documents[i:batch_end]
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Sincronización completada en {elapsed:.1f}s - {len(ids)} frases")
        
        return len(ids)
    
    def _generate_command_descriptions(
        self,
        entity: dict,
        services: list[str],
        llm_reasoner
    ) -> list[dict]:
        """Generar descripciones de comandos usando LLM"""
        entity_id = entity["entity_id"]
        domain = entity_id.split(".")[0]
        friendly_name = entity["attributes"].get("friendly_name", entity_id)
        
        prompt = f"""Genera comandos de voz para esta entidad de domótica.

Entidad: {entity_id}
Nombre: {friendly_name}
Dominio: {domain}
Servicios: {', '.join(services)}

Genera JSON con comandos posibles. Cada comando tiene:
- entity_id, domain, service, description
- phrases: 5-8 frases naturales en español

Ejemplo:
[{{"entity_id": "light.living", "domain": "light", "service": "turn_on", 
   "description": "Encender luz living", 
   "phrases": ["prende la luz del living", "enciende el living", "luz del living"]}}]

Solo JSON, sin explicaciones:"""

        result = llm_reasoner(prompt, max_tokens=1024, temperature=0.3)
        response = result["choices"][0]["text"]
        
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            logger.warning(f"Error parseando JSON para {entity_id}")
        
        return []
    
    # ==================== Búsqueda de Comandos ====================
    
    @_owned_operation
    def search_command(
        self,
        query: str,
        threshold: float = 0.65,
        service_filter: str | None = None,
        query_slots: dict | None = None,
        hint_entities: list[str] | None = None,
        prefer_area: str | None = None,
    ) -> dict | None:
        """
        Buscar comando más similar con filtro de intent + merge de slots.

        Args:
            query: Texto del usuario (post-STT).
            threshold: Similitud mínima (0-1).
            service_filter: Si viene "turn_on" o "turn_off", filtra la búsqueda
                a ese service (evita que dense retrieval confunda antónimos —
                ver memoria feedback_dense_retrieval_antonyms.md).
            query_slots: Slots extraídos por el NLU (brightness_pct, rgb_color,
                color_temp_kelvin). Sobrescriben el service_data default del preset.
            hint_entities: Lista opcional de entity_ids preservados del
                contexto previo del usuario (plan #2 OpenClaw). Hoy solo
                se loguea; el consumo real (boost de score) queda para un
                follow-up cuando haya señal empírica de mejora.
            prefer_area: Área requerida (matchea metadata.area en Chroma).
                Cuando viene, solo se aceptan candidatos de esa área.
                Un área sin coincidencias válidas devuelve None; nunca se
                sustituye por una entidad de otro ambiente.
                Crítico para el caso 'bajá la luz al cincuenta por ciento'
                desde el escritorio: sin esto, docs contaminados sin room
                de otras entities ganan por similitud pura. Bug 2026-05-03.

        Returns:
            dict con {entity_id, domain, service, data, matched_phrase,
                      similarity, capability, value_label} o None.
        """
        from src.nlu.slot_extractor import merge_service_data

        if hint_entities:
            logger.debug(
                f"[VectorSearch] hint_entities count={len(hint_entities)}"
            )

        start = time.perf_counter()
        query_embedding = self.embedder.encode(query).tolist()

        where = {"service": service_filter} if service_filter else None

        # PASE 1 — área del mic primero (2026-06-04, ronda 3): con queries
        # genéricas ("prende la luz") el top-10 GLOBAL se llena de docs cortos
        # de otras rooms y ningún doc del área del mic entra al pool — el
        # boost no tiene a quién boostear (medido: top-10 sin Escritorio,
        # 'prende la luz fría'@Pasillo sim=0.945 → prendía el pasillo).
        # Buscamos primero SOLO dentro del área preferida; si hay match sobre
        # threshold, gana. Si no, rechazamos sin buscar en otro ambiente.
        results = None
        best_idx: int | None = None
        area_locked = False
        if prefer_area:
            area_clauses: list[dict] = [{"area": prefer_area}]
            if service_filter:
                area_clauses.append({"service": service_filter})
            where_area = (
                {"$and": area_clauses} if len(area_clauses) > 1 else area_clauses[0]
            )
            res_area = self.commands.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where=where_area,
                include=["metadatas", "distances", "documents"],
            )
            if res_area["ids"][0]:
                best_sim_a = -1.0
                for i, (dist, meta) in enumerate(
                    zip(res_area["distances"][0], res_area["metadatas"][0])
                ):
                    m = meta or {}
                    if self._is_excluded(m.get("entity_id", "")):
                        continue
                    # Cinturón: el where ya filtró en Chroma, pero validamos
                    # el área igual (defensa contra metadata inconsistente).
                    if m.get("area") != prefer_area:
                        continue
                    sim = 1 - (dist / 2)
                    if sim >= threshold and sim > best_sim_a:
                        best_sim_a = sim
                        best_idx = i
                if best_idx is not None:
                    results = res_area
                    area_locked = True

        if results is None and prefer_area:
            logger.info(f"[VectorSearch] No verified match in requested area {prefer_area!r}; refusing cross-room fallback")
            return None

        if results is None:
            # PASE 2 — global. Con prefer_area traemos más candidatos para
            # tener pool real para re-puntuar; sin él, N=3 como antes.
            best_idx = None
            n_results = 10 if prefer_area else 3
            results = self.commands.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["metadatas", "distances", "documents"],
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results["ids"][0]:
            logger.debug(f"Vector search ({elapsed_ms:.0f}ms): No results")
            return None

        # Filtro de entidades excluidas en QUERY-time (2026-06-04): el índice
        # puede venir contaminado — el sync del 31-05 indexó 76 docs genéricos
        # de light.hogar pese al default del runtime, y "prende la luz" desde
        # el escritorio prendió TODA la casa. La exclusión al indexar no
        # alcanza si el índice ya está sucio; acá es la defensa definitiva.
        # (Con area_locked, el pase 1 ya filtró excluidos y validó threshold.)
        if not area_locked:
            _metas_all = results["metadatas"][0]
            valid_idx = [
                i for i, m in enumerate(_metas_all)
                if not self._is_excluded((m or {}).get("entity_id", ""))
            ]
            if not valid_idx:
                logger.info(
                    f"Vector search ({elapsed_ms:.0f}ms): todos los candidatos "
                    f"están excluidos (índice contaminado?) — sin match"
                )
                return None

            # Re-ranking por prefer_area. El gate de threshold se aplica
            # sobre la similarity ORIGINAL (no boosteada) — el bonus solo
            # desempata, no rescata candidatos lejos del query.
            best_idx = valid_idx[0]
        if not area_locked and prefer_area:
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            best_score = -1.0
            for i in valid_idx:
                dist, meta = distances[i], metadatas[i]
                base_sim = 1 - (dist / 2)
                # Threshold gate sobre similarity base — protege contra rescue
                # artificial. Si no pasa, no es candidato para ningún ranking.
                if base_sim < threshold:
                    continue
                area_bonus = (
                    self.PREFER_AREA_BOOST
                    if (meta or {}).get("area") == prefer_area
                    else 0.0
                )
                score = base_sim + area_bonus
                if score > best_score:
                    best_score = score
                    best_idx = i
            # Si NINGÚN candidato pasa threshold, retornar None
            base_sim_top = 1 - (results["distances"][0][best_idx] / 2)
            if base_sim_top < threshold:
                logger.debug(
                    f"Vector search ({elapsed_ms:.0f}ms): no candidates above "
                    f"threshold (top base_sim={base_sim_top:.2f}, "
                    f"prefer_area={prefer_area!r})"
                )
                return None

        best_distance = results["distances"][0][best_idx]
        similarity = 1 - (best_distance / 2)

        if similarity < threshold:
            logger.debug(f"Vector search ({elapsed_ms:.0f}ms): Below threshold (sim={similarity:.2f})")
            return None

        metadata = results["metadatas"][0][best_idx]
        # Merge: preset service_data (de la frase indexada) + slots reales del usuario
        preset_data = {}
        if metadata.get("service_data"):
            try:
                preset_data = json.loads(metadata["service_data"])
            except (json.JSONDecodeError, TypeError):
                preset_data = {}
        # Legacy support: campos viejos ("data")
        if not preset_data and metadata.get("data"):
            try:
                preset_data = json.loads(metadata["data"])
            except (json.JSONDecodeError, TypeError):
                preset_data = {}
        final_data = merge_service_data(preset_data, query_slots or {})

        capability = metadata.get("capability", "onoff")
        value_label = metadata.get("value_label", "")
        friendly = metadata.get("friendly_name", metadata["entity_id"])
        description = f"{metadata['service']} {friendly}"
        if value_label and value_label not in ("prender", "apagar"):
            description += f" ({value_label})"

        result = {
            "entity_id": metadata["entity_id"],
            "domain": metadata["domain"],
            "service": metadata["service"],
            "description": description,
            "data": final_data,
            "matched_phrase": results["documents"][0][best_idx],
            "similarity": similarity,
            "capability": capability,
            "value_label": value_label,
        }
        logger.debug(
            f"Vector search ({elapsed_ms:.0f}ms): {description} "
            f"(sim={similarity:.2f}, cap={capability}, data={final_data})"
        )
        return result

    async def asearch_command(
        self,
        query: str,
        threshold: float = 0.65,
        service_filter: str | None = None,
        query_slots: dict | None = None,
        hint_entities: list[str] | None = None,
        prefer_area: str | None = None,
    ) -> dict | None:
        """Variante async de search_command para el fast path.

        Delega el trabajo síncrono (encode BGE-M3 en CPU + query a Chroma) a un
        thread con ownership retenido para no bloquear el event loop. Misma
        firma y retorno que search_command. Usar en contextos async calientes
        (dispatcher fast path); los callers sync siguen usando search_command.
        """
        return await self._async_work.run(
            self.search_command,
            query,
            threshold,
            service_filter=service_filter,
            query_slots=query_slots,
            hint_entities=hint_entities,
            prefer_area=prefer_area,
        )

    # ==================== Rutinas ====================
    
    @_owned_operation
    def save_routine(self, routine: dict):
        """Guardar rutina en ChromaDB"""
        routine_id = routine["id"]
        
        # Crear frases para buscar la rutina
        phrases = [
            routine["name"],
            routine["description"],
            f"rutina {routine['name']}",
            f"automatización {routine['name']}"
        ]
        
        # Agregar frases de triggers
        for trigger in routine.get("triggers", []):
            phrases.append(f"rutina de {trigger.get('description', '')}")
        
        for i, phrase in enumerate(phrases):
            if not phrase.strip():
                continue
                
            embedding = self.embedder.encode(phrase).tolist()
            
            self.routines.add(
                ids=[f"{routine_id}_{i}"],
                embeddings=[embedding],
                documents=[phrase],
                metadatas=[{
                    "routine_id": routine_id,
                    "name": routine["name"],
                    "description": routine["description"],
                    "created_at": routine.get("created_at", ""),
                    "full_config": json.dumps(routine)
                }]
            )
    
    @_owned_operation
    def search_routine(
        self,
        query: str,
        threshold: float = 0.5
    ) -> dict | None:
        """Buscar rutina por descripción"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.routines.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            return None
        
        distance = results["distances"][0][0]
        similarity = 1 - (distance / 2)
        
        if similarity >= threshold:
            metadata = results["metadatas"][0][0]
            return {
                "routine_id": metadata["routine_id"],
                "name": metadata["name"],
                "description": metadata["description"],
                "similarity": similarity,
                "full_config": json.loads(metadata["full_config"])
            }
        
        return None
    
    @_owned_operation
    def delete_routine(self, routine_id: str):
        """Eliminar rutina de ChromaDB"""
        existing = self.routines.get(where={"routine_id": routine_id})
        if existing["ids"]:
            self.routines.delete(ids=existing["ids"])
            logger.info(f"Rutina eliminada de ChromaDB: {routine_id}")
    
    @_owned_operation
    def list_routines(self) -> list[dict]:
        """Listar todas las rutinas"""
        all_data = self.routines.get(include=["metadatas"])
        
        # Deduplicar por routine_id
        seen = set()
        routines = []
        
        for metadata in all_data["metadatas"]:
            routine_id = metadata["routine_id"]
            if routine_id not in seen:
                seen.add(routine_id)
                routines.append({
                    "id": routine_id,
                    "name": metadata["name"],
                    "description": metadata["description"]
                })
        
        return routines
    
    # ==================== Estadísticas ====================
    
    @_owned_operation
    def get_stats(self) -> dict:
        """Obtener estadísticas de la base de datos"""
        commands_count = self.commands.count()
        routines_count = self.routines.count()
        
        return {
            "commands_phrases": commands_count,
            "routines_phrases": routines_count,
            "chroma_path": self.chroma_path
        }
