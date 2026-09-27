from typing import List, Dict, Any
from src.config import AppConfig, setup_logger
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore
from src.layer1_learning import CategoryBasedMemoryLearning
from src.layer2_construction import ChainBasedMemoryConstruction
from src.layer3_updating import SelfMonitoringMemoryUpdating
from src.layer4_retrieval import SelfReflectionMemoryRetrieval

logger = setup_logger("m4.pipeline")

class MetaMemoryPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_client = OllamaClient(config)
        self.graph_store = Neo4jGraphStore(config)
        self.layer1 = CategoryBasedMemoryLearning(config, self.llm_client)
        self.layer2 = ChainBasedMemoryConstruction(config, self.graph_store)
        self.layer3 = SelfMonitoringMemoryUpdating(config)
        self.layer4 = SelfReflectionMemoryRetrieval(config)

    def startup(self):
        self.graph_store.connect()

    def shutdown(self):
        self.graph_store.close()

    def ingest_session(self, session_id: str, conversation_text: str) -> Dict[str, Any]:
        logger.info("=== Ingesting Session [%s] ===", session_id)
        # Layer 1: Learn & Disambiguate Clues
        clues = self.layer1.process_session(conversation_text)
        if not clues:
            return {"session_id": session_id, "clues_extracted": 0, "nodes_created": []}

        # Layer 2: Construct Memory Chains & Cross-Graph Connections
        node_ids = self.layer2.build_memory_chains(clues, session_id)

        return {
            "session_id": session_id,
            "clues_extracted": len(clues),
            "nodes_created": node_ids,
            "status": "Committed to Neo4j"
        }
