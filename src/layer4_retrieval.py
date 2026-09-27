# """
# Layer 4: Self-reflection-based Memory Retrieval (§III-E, Equations 8-9, Algorithm 1)
# Architecture Specification & Stubs for Next Sprint.
# """
# from typing import List, Tuple
# from src.config import AppConfig, setup_logger

# logger = setup_logger("m4.layer4")

# class SelfReflectionMemoryRetrieval:
#     """  
#     STUB IMPLEMENTATION: Reserved for Layer 4.
#     Mapped to:
#       - §III-E Category Retrieval: Prompting for T^q relevant categories.
#       - Eq. 8: Bidirectional chain expansion M <- f_c(M || c_{p_f}^q) if LLM_re(c_{p_f}^q)
#       - Algorithm 1: Dynamic loop checking LLM_confident(q, M) to stop search.
#       - Eq. 9: Context-aware response generation r = LLM(q, M).
#     """
#     def __init__(self, config: AppConfig):
#         self.config = config

#     def retrieve_memory(self, query: str) -> Tuple[List[str], str]:
#         """
#         Executes Algorithm 1 bidirectional traversal.
#         Docstring: Identifies category, seeds search with closest node, expands forward and backward
#         until self-evaluation indicates sufficient context, then synthesizes answer.
#         """
#         logger.info("[STUB Layer 4] retrieve_memory called for query: '%s'. Skipping.", query)
#         return ([], "Layer 4 stub: retrieval will be implemented in subsequent sprint.")






#  The below code is for layer 1,2,3,4 combined ----------------------------------------------------------






"""
Layer 4: Self-reflection-based Memory Retrieval (§III-E, Equations 8-9, Algorithm 1)
Fully Implemented Architecture Specification.
"""
import json
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.config import AppConfig, setup_logger
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore

logger = setup_logger("m4.layer4")

class SelfReflectionMemoryRetrieval:
    """  
    IMPLEMENTATION: Layer 4 Self-reflection Memory Retrieval[cite: 3].
    Mapped to:
      - §III-E Category Retrieval: Prompting for T^q relevant categories[cite: 3].
      - Eq. 8: Bidirectional chain expansion M <- f_c(M || c_{p_f}^q) if LLM_re(c_{p_f}^q)[cite: 3].
      - Algorithm 1: Dynamic loop checking LLM_confident(q, M) to stop search[cite: 3].
      - Eq. 9: Context-aware response generation r = LLM(q, M)[cite: 3].
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = OllamaClient(config)
        self.graph = Neo4jGraphStore(config)

    def retrieve_memory(self, query: str) -> Tuple[List[str], str]:
        """
        Executes Algorithm 1 bidirectional traversal[cite: 3].
        """
        logger.info("[Layer 4] Initiating Self-Reflection Retrieval for query: '%s'", query)

        # Fig. 6: Prompt for Category Retrieval[cite: 3]
        cat_prompt = f"""Given the current utterance/query: {query}
Please analyze the utterance and determine which categories it is relevant to.
Return a JSON array of objects, where each object represents a category that is relevant to the utterance.
Example JSON Response:
[
  {{"Category": "YOUR_CATEGORY"}},
  {{"Category": "YOUR_CATEGORY"}}
]
Based on the utterance, please provide the JSON response with the relevant categories without any reasoning:"""

        try:
            categories_json = self.llm.complete_chat([{"role": "user", "content": cat_prompt}])
            match = re.search(r"\[.*\]", categories_json.strip(), re.DOTALL)
            categories = json.loads(match.group(0)) if match else []
            cat_names = [c.get("Category") for c in categories if "Category" in c]
        except Exception as e:
            logger.error("[Layer 4] Failed to extract categories: %s", e)
            cat_names = []

        memory_context = []
        q_emb = self.llm.get_embedding(query)

        for cat in cat_names:
            # Algorithm 1: Evaluate LLM_confident(q, M)[cite: 3]
            if self._is_confident(query, memory_context):
                logger.info("[Layer 4] Confidence threshold met. Halting traversal.")
                break

            # Locate sequence nodes for bidirectional traversal[cite: 3]
            chain_nodes = self._get_nodes_with_embeddings(cat)
            if not chain_nodes:
                continue

            # Algorithm 1: Obtain the index of the most relevant memory clue (k <- argmax(sim))[cite: 3]
            best_node_idx = -1
            max_sim = -1.0
            for idx, node in enumerate(chain_nodes):
                sim = self._cosine_sim(node['embedding'], q_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_node_idx = idx

            if best_node_idx == -1:
                continue

            best_node = chain_nodes[best_node_idx]
            memory_context.append(best_node['summary'])

            # Algorithm 1: Initialize bidirectional traversal pointers[cite: 3]
            pf = best_node_idx + 1
            pb = best_node_idx - 1

            while not self._is_confident(query, memory_context):
                B_f = False
                if pf < len(chain_nodes):
                    # Check LLM_re for forward nodes[cite: 3]
                    B_f = self._improves_response(query, chain_nodes[pf]['summary'], memory_context)
                
                B_b = False
                if pb >= 0:
                    # Check LLM_re for backward nodes[cite: 3]
                    B_b = self._improves_response(query, chain_nodes[pb]['summary'], memory_context)

                # Integrate approved nodes into memory array (Eq. 8)[cite: 3]
                if B_f:
                    memory_context.append(chain_nodes[pf]['summary'])
                if B_b:
                    memory_context.append(chain_nodes[pb]['summary'])

                # Handling cross-category dependencies (Shared Nodes Fallback)[cite: 3]
                # "When the memory chains... have been traversed but cannot answer... LLM will automatically locate the shared node"[cite: 3]
                if not B_f and not B_b:
                    shared_node = self._get_shared_node(best_node['id'], memory_context)
                    if shared_node and self._improves_response(query, shared_node['summary'], memory_context):
                        memory_context.append(shared_node['summary'])
                        break  # Shift traversal context to associated multi-hop graph
                    
                # Algorithm 1: Terminate when all finite nodes are traversed[cite: 3]
                if pf >= len(chain_nodes) - 1 and pb <= 0:
                    break
                    
                if pf < len(chain_nodes) - 1:
                    pf += 1
                if pb > 0:
                    pb -= 1

        # Fig. 8: Prompt for Response Generation (Eq. 9)[cite: 3]
        final_prompt = f"""The following are user's query/utterance and related memories:
Query/Utterance: {query}
Memories: {chr(10).join(memory_context)}

Generate the most plausible answer based memories without any reasoning. Each line in the memory represents a timeline chain of topics. Please pay attention to the sequence, the evolution of these memories."""

        response = self.llm.complete_chat([{"role": "user", "content": final_prompt}]).strip()
        logger.info("[Layer 4] Generated Context-Aware Response.")

        return memory_context, response

    def _is_confident(self, query: str, memories: List[str]) -> bool:
        """Fig. 7: Prompt for Self-Evaluation of LLM (LLM_confident)[cite: 3]."""
        if not memories:
            return False
        prompt = f"""The following are user's query/utterance and candidate memories:
Query/Utterance: {query}
Memories: {memories}

Please be very honest in your judgment as to whether you can generate the best response based on the known memories.

If you can, please output 1 and conversely if you need more memories then output 0 without any reasoning:"""
        return "1" in self.llm.complete_chat([{"role": "user", "content": prompt}]).strip()

    def _improves_response(self, query: str, candidate: str, current_memories: List[str]) -> bool:
        """LLM_re function: evaluates if adding a specific node improves response quality[cite: 3]."""
        test_memories = current_memories + [candidate]
        return self._is_confident(query, test_memories)

    def _get_nodes_with_embeddings(self, category_name: str) -> List[Dict[str, Any]]:
        """Extracts ordered chain data to construct algorithm search path."""
        query = """
        MATCH (c:MemoryChain {category_name: $category_name})-[:HEAD_NODE]->(head:MemoryNode)
        MATCH path = (head)-[:NEXT_IN_CHAIN*0..]->(tail:MemoryNode)
        WHERE NOT (tail)-[:NEXT_IN_CHAIN]->()
        RETURN [n in nodes(path) | {id: n.id, summary: n.summary, embedding: n.embedding}] AS chain_nodes
        """
        with self.graph.session() as s:
            res = s.run(query, category_name=category_name).single()
            return res["chain_nodes"] if res and res["chain_nodes"] else []

    def _get_shared_node(self, node_id: str, current_memories: List[str]) -> Optional[Dict[str, Any]]:
        """Identifies shared graph connections across category boundaries[cite: 3]."""
        query = """
        MATCH (n:MemoryNode {id: $node_id})-[:SHARED_WITH]-(shared:MemoryNode)
        RETURN shared.id AS id, shared.summary AS summary
        LIMIT 1
        """
        with self.graph.session() as s:
            res = s.run(query, node_id=node_id).single()
            if res and res["summary"] not in current_memories:
                return {"id": res["id"], "summary": res["summary"]}
            return None

    def _cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Helper to process mathematical similarity for base anchor selection."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        n_a = np.linalg.norm(a)
        n_b = np.linalg.norm(b)
        if n_a == 0 or n_b == 0:
            return 0.0
        return float(np.dot(a, b) / (n_a * n_b))