"""
Vector Database Sync Module
Sincronización de comandos de Home Assistant con ChromaDB
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaSync:
    """Sincronización de Home Assistant con ChromaDB"""
    
    def __init__(
        self,
        chroma_path: str,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        embedder_device: str = "cuda:1",
        commands_collection: str = "home_assistant_commands",
        routines_collection: str = "home_assistant_routines"
    ):
        self.chroma_path = chroma_path
        self.embedder_model = embedder_model
        self.embedder_device = embedder_device
        self.commands_collection_name = commands_collection
        self.routines_collection_name = routines_collection
        
        self._client = None
        self._commands_collection = None
        self._routines_collection = None
        self._embedder = None
    
    def initialize(self):
        """Inicializar ChromaDB y embedder"""
        logger.info("Inicializando ChromaDB...")
        
        # Cliente persistente
        self._client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Colecciones
        self._commands_collection = self._client.get_or_create_collection(
            name=self.commands_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self._routines_collection = self._client.get_or_create_collection(
            name=self.routines_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Embedder
        logger.info(f"Cargando embedder: {self.embedder_model}")
        self._embedder = SentenceTransformer(
            self.embedder_model,
            device=self.embedder_device
        )
        
        logger.info("ChromaDB inicializado")
    
    @property
    def commands(self):
        """Colección de comandos"""
        if self._commands_collection is None:
            self.initialize()
        return self._commands_collection
    
    @property
    def routines(self):
        """Colección de rutinas"""
        if self._routines_collection is None:
            self.initialize()
        return self._routines_collection
    
    @property
    def embedder(self):
        """Modelo de embeddings"""
        if self._embedder is None:
            self.initialize()
        return self._embedder
    
    # ==================== Sincronización de Comandos ====================
    
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
    
    def search_command(
        self,
        query: str,
        threshold: float = 0.65,
        service_filter: str | None = None,
        query_slots: dict | None = None,
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

        Returns:
            dict con {entity_id, domain, service, data, matched_phrase,
                      similarity, capability, value_label} o None.
        """
        from src.nlu.slot_extractor import merge_service_data

        start = time.perf_counter()
        query_embedding = self.embedder.encode(query).tolist()

        where = {"service": service_filter} if service_filter else None
        results = self.commands.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results["ids"][0]:
            logger.debug(f"Vector search ({elapsed_ms:.0f}ms): No results")
            return None

        best_distance = results["distances"][0][0]
        similarity = 1 - (best_distance / 2)

        if similarity < threshold:
            logger.debug(f"Vector search ({elapsed_ms:.0f}ms): Below threshold (sim={similarity:.2f})")
            return None

        metadata = results["metadatas"][0][0]
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
            "matched_phrase": results["documents"][0][0],
            "similarity": similarity,
            "capability": capability,
            "value_label": value_label,
        }
        logger.debug(
            f"Vector search ({elapsed_ms:.0f}ms): {description} "
            f"(sim={similarity:.2f}, cap={capability}, data={final_data})"
        )
        return result
    
    # ==================== Rutinas ====================
    
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
    
    def delete_routine(self, routine_id: str):
        """Eliminar rutina de ChromaDB"""
        existing = self.routines.get(where={"routine_id": routine_id})
        if existing["ids"]:
            self.routines.delete(ids=existing["ids"])
            logger.info(f"Rutina eliminada de ChromaDB: {routine_id}")
    
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
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de la base de datos"""
        commands_count = self.commands.count()
        routines_count = self.routines.count()
        
        return {
            "commands_phrases": commands_count,
            "routines_phrases": routines_count,
            "chroma_path": self.chroma_path
        }
