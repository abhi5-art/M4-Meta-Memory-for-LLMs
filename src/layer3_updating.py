"""
Layer 3: Self-monitoring-based Memory Updating (§III-D, Equations 4-7)
Fully Implemented Architecture Specification.
"""
from typing import List
from datetime import datetime, timezone, timedelta
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore

logger = setup_logger("m4.layer3")

class SelfMonitoringMemoryUpdating:
    """
    IMPLEMENTATION: Layer 3 Self-monitoring Memory Updating.
    Mapped to:
      - Eq. 4: Inconsistent memory detection {c_i^u} <- LLM_u(C_k; c_k)[cite: 3]
      - Eq. 5: Merging conflict to unambiguous node \hat{c}_k <- LLM_m(c_i^u; c_k)[cite: 3]
      - Eq. 6: Removal of obsolete node and linking calibrated node C_k <- f_c(C_k || \hat{c}_k)[cite: 3]
      - Eq. 7: Temporal compression c_i^{com} = LLMLingua(c_i) based on Ebbinghaus forgetting curve[cite: 3].
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = OllamaClient(config)
        self.graph = Neo4jGraphStore(config)

    def calibrate_chain(self, chain_id: str, new_node: MemoryNodeModel) -> bool:
        """
        Executes Eq. 4-6 conflict check and calibration against the chain[cite: 3].
        """
        logger.info("[Layer 3] Calibrating chain '%s' against new node '%s'.", chain_id, new_node.id)

        # Retrieve historical nodes in the chain for consistency checking
        existing_nodes = self.graph.get_chain_nodes(chain_id)

        for node in existing_nodes:
            if node['id'] == new_node.id:
                continue

            # Fig. 3: Prompt for Conflict Detection[cite: 3]
            conflict_prompt = f"""The following are summaries given in conversational order:
Previous_summary: {node['summary']}
New_summary: {new_node.summary}

Please understand their differences carefully and judge whether there is a conflict or update between them, such as a change in the speaker's identity, the development of events, and a change in attitude (interest, etc.).

If you think there is a conflict between them, then please output 1, otherwise output 0 without any reasoning:"""

            is_conflict = self.llm.complete_chat([{"role": "user", "content": conflict_prompt}]).strip()

            if "1" in is_conflict:
                logger.info("[Layer 3] Conflict detected between '%s' and '%s' (Eq 4).", node['id'], new_node.id)

                # Fig. 4: Prompt for Memory Calibrating[cite: 3]
                merge_prompt = f"""The following are some summaries given in conversational order:
Previous_summary: {node['summary']}
New_summary: {new_node.summary}

Please understand their differences carefully and focus on the changes or updates between contents, such as the change of the speaker's identity, the development of events, and the change of attitude (interest, etc.), etc., and merge them into a new summary without any reasoning."""

                # Execute Eq 5: Merge into unambiguous memory node[cite: 3]
                merged_summary = self.llm.complete_chat([{"role": "user", "content": merge_prompt}]).strip()

                # Execute Eq 6: Atomic graph calibration (Update existing, delete obsolete new node)[cite: 3]
                with self.graph.session() as s:
                    s.run("MATCH (n:MemoryNode {id: $node_id}) SET n.summary = $merged_summary",
                          node_id=node['id'], merged_summary=merged_summary)
                    s.run("MATCH (n:MemoryNode {id: $new_node_id}) DETACH DELETE n",
                          new_node_id=new_node.id)

                logger.info("[Layer 3] Calibration applied. Node '%s' updated, obsolete new node deleted.", node['id'])
                return True

        return False

    def compress_dormant_nodes(self, days_idle: int = 14) -> int:
        """
        Executes Eq. 7 compression via LLM proxy with Ebbinghaus recoverability check[cite: 3].
        """
        logger.info("[Layer 3] Sweeping graph for dormant nodes to compress.")
        threshold_date = (datetime.now(timezone.utc) - timedelta(days=days_idle)).isoformat()

        query = """
        MATCH (n:MemoryNode)
        WHERE n.last_used_at < $threshold_date AND (n.compressed IS NULL OR n.compressed = false)
        RETURN n.id AS id, n.summary AS summary
        LIMIT 10
        """
        with self.graph.session() as s:
            candidates = s.run(query, threshold_date=threshold_date).data()

        compressed_count = 0
        for node in candidates:
            # Compression action
            distill_prompt = f"Condense the following text to its absolute core facts, using as few words as possible: {node['summary']}"
            compressed_text = self.llm.complete_chat([{"role": "user", "content": distill_prompt}]).strip()

            # Fig. 5: Prompt for Compression Detection (Recoverability verification)[cite: 3]
            verify_prompt = f"""The following is an original text and its compressed text:
Original_text: {node['summary']}
Compressed_text: {compressed_text}

Please be careful to understand the differences and make an honest judgment as to whether most of the key information in the original text can be restored from the compressed text.

If you think you can decompress or restore it, then please output 1, otherwise output 0 without any reasoning:"""

            can_restore = self.llm.complete_chat([{"role": "user", "content": verify_prompt}]).strip()

            # Persist if LLM confirms semantic retention[cite: 3]
            if "1" in can_restore:
                with self.graph.session() as s:
                    s.run("MATCH (n:MemoryNode {id: $node_id}) SET n.summary = $compressed_text, n.compressed = true",
                          node_id=node['id'], compressed_text=compressed_text)
                compressed_count += 1
                logger.info("[Layer 3] Node '%s' successfully compressed (Eq. 7).", node['id'])

        return compressed_count