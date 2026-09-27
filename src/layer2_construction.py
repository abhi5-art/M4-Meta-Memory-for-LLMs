import uuid
from datetime import datetime, timezone
from typing import List
from src.config import AppConfig, setup_logger
from src.models import DisambiguatedMemoryClue, MemoryNodeModel
from src.graph_store import Neo4jGraphStore

logger = setup_logger("m4.layer2")

class ChainBasedMemoryConstruction:
    """
    Implements Layer 2 (§III-C, Equations 2 & 3):
    Organizes disordered memory clues into chronological category chains,
    and connects multi-category associations into a unified memory graph.
    """
    def __init__(self, config: AppConfig, graph_store: Neo4jGraphStore):
        self.config = config
        self.graph_store = graph_store

    def build_memory_chains(self, clues: List[DisambiguatedMemoryClue], session_id: str) -> List[str]:
        created_node_ids: List[str] = []
        gamma = self.config.similarity_threshold

        for clue in clues:
            # Equation 2: Query maximum similarity against existing categories in graph
            match_result = self.graph_store.find_most_similar_chain(clue.category_embedding, top_k=1)

            target_category = clue.category
            if match_result:
                matched_chain_id, matched_category_name, max_sim = match_result
                logger.info("Chain match check for '%s': Best match is '%s' with score=%.4f",
                            clue.category, matched_category_name, max_sim)
                # Equation 3: If eta >= gamma, link to existing chain; else create new chain
                if max_sim >= gamma:
                    logger.info("Eq. 3: Appending clue to existing chain '%s' (eta=%.4f >= gamma=%.2f)",
                                matched_category_name, max_sim, gamma)
                    target_category = matched_category_name
                else:
                    logger.info("Eq. 3: Similarity below threshold. Creating distinct chain for '%s'.", clue.category)
                    new_chain_id = f"chain_{uuid.uuid4().hex[:12]}"
                    self.graph_store.create_memory_chain(new_chain_id, clue.category, clue.category_embedding)
            else:
                logger.info("No prior chains found in graph. Instantiating initial chain for '%s'.", clue.category)
                new_chain_id = f"chain_{uuid.uuid4().hex[:12]}"
                self.graph_store.create_memory_chain(new_chain_id, clue.category, clue.category_embedding)

            # Construct MemoryNode
            node_id = f"mem_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc)
            node = MemoryNodeModel(
                id=node_id,
                summary=clue.summary,
                category=target_category,
                session_id=session_id,
                created_at=now,
                last_used_at=now,
                embedding=clue.clue_embedding
            )

            self.graph_store.append_node_to_chain(node, target_category)
            created_node_ids.append(node_id)

        # Connect shared nodes within this session across different categories (Memory Graph)
        if len(created_node_ids) > 1:
            for i in range(len(created_node_ids)):
                for j in range(i + 1, len(created_node_ids)):
                    self.graph_store.link_shared_nodes(created_node_ids[i], created_node_ids[j], "SHARED_WITH")

        return created_node_ids
