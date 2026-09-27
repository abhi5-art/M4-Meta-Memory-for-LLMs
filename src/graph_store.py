from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from neo4j import GraphDatabase, Driver, Session
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel, MemoryChainModel

logger = setup_logger("m4.graph_store")

class Neo4jGraphStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self._driver: Optional[Driver] = None

    def connect(self):
        if not self._driver:
            self._driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_pool_size=50,
                connection_acquisition_timeout=30.0
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j graph store successfully at %s", self.config.neo4j_uri)

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver connection pool closed.")

    @contextmanager
    def session(self) -> Session:
        if not self._driver:
            self.connect()
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def find_most_similar_chain(self, category_embedding: List[float], top_k: int = 1) -> Optional[Tuple[str, str, float]]:
        """
        Uses Neo4j's native Vector Index to retrieve the closest existing category chain.
        Returns Tuple: (chain_id, category_name, similarity_score).
        """
        query = """
        CALL db.index.vector.queryNodes('category_chain_embeddings', $top_k, $category_embedding)
        YIELD node, score
        RETURN node.id AS chain_id, node.category_name AS category_name, score
        LIMIT 1
        """
        with self.session() as s:
            result = s.run(query, top_k=top_k, category_embedding=category_embedding).single()
            if result:
                return (result["chain_id"], result["category_name"], float(result["score"]))
            return None

    def create_memory_chain(self, chain_id: str, category_name: str, embedding: List[float]) -> str:
        query = """
        MERGE (c:MemoryChain {category_name: $category_name})
        ON CREATE SET c.id = $chain_id,
                      c.embedding = $embedding,
                      c.created_at = datetime()
        RETURN c.id AS id
        """
        with self.session() as s:
            record = s.run(query, chain_id=chain_id, category_name=category_name, embedding=embedding).single()
            return record["id"]

    def append_node_to_chain(
        self,
        node: MemoryNodeModel,
        chain_category: str
    ):
        """
        Atomically appends node to chain via [:NEXT_IN_CHAIN], preserving chronological order.
        Links chain to head node if first, or extends current tail node.
        """
        query = """
        MATCH (c:MemoryChain {category_name: $category_name})
        CREATE (n:MemoryNode {
            id: $node_id,
            summary: $summary,
            category: $category,
            session_id: $session_id,
            created_at: $created_at,
            last_used_at: $last_used_at,
            embedding: $embedding
        })
        WITH c, n
        OPTIONAL MATCH (c)-[:HEAD_NODE]->(head:MemoryNode)
        FOREACH (_ IN CASE WHEN head IS NULL THEN [1] ELSE [] END |
            CREATE (c)-[:HEAD_NODE]->(n)
        )
        WITH c, n
        OPTIONAL MATCH (c)-[old_tail:TAIL_NODE]->(current_tail:MemoryNode)
        FOREACH (_ IN CASE WHEN current_tail IS NOT NULL THEN [1] ELSE [] END |
            CREATE (current_tail)-[:NEXT_IN_CHAIN]->(n)
            DELETE old_tail
        )
        CREATE (c)-[:TAIL_NODE]->(n)
        RETURN n.id AS created_node_id
        """
        with self.session() as s:
            s.run(
                query,
                category_name=chain_category,
                node_id=node.id,
                summary=node.summary,
                category=node.category,
                session_id=node.session_id,
                created_at=node.created_at.isoformat(),
                last_used_at=node.last_used_at.isoformat(),
                embedding=node.embedding
            )
            logger.info("Node [%s] committed to chain '%s'.", node.id, chain_category)

    def link_shared_nodes(self, node_id_a: str, node_id_b: str, relation_type: str = "SHARED_WITH"):
        """Establishes cross-category graph link (Memory Graph construction from paper)."""
        query = f"""
        MATCH (a:MemoryNode {{id: $id_a}}), (b:MemoryNode {{id: $id_b}})
        MERGE (a)-[r:{relation_type}]->(b)
        RETURN count(r) AS links
        """
        with self.session() as s:
            s.run(query, id_a=node_id_a, id_b=node_id_b)
            logger.debug("Cross-category edge created: (%s)-[:%s]->(%s)", node_id_a, relation_type, node_id_b)

    def get_chain_nodes(self, category_name: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (c:MemoryChain {category_name: $category_name})-[:HEAD_NODE]->(head:MemoryNode)
        MATCH path = (head)-[:NEXT_IN_CHAIN*0..]->(tail:MemoryNode)
        WHERE NOT (tail)-[:NEXT_IN_CHAIN]->()
        RETURN [n in nodes(path) | {id: n.id, summary: n.summary, category: n.category, session_id: n.session_id}] AS chain_nodes
        """
        with self.session() as s:
            res = s.run(query, category_name=category_name).single()
            return res["chain_nodes"] if res and res["chain_nodes"] else []
