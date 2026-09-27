#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# Meta-Memory (M4) Production Bootstrap Script
# Implements: Layer 1 (§III-B, Eq. 1) & Layer 2 (§III-C, Eq. 2-3)
# =====================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }

# ---------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------
log_info "Executing pre-flight platform validation..."

command -v docker >/dev/null 2>&1 || { log_err "Docker is not installed or not in PATH. Aborting."; exit 1; }
docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1 || {
    log_err "Neither 'docker compose' nor 'docker-compose' is available. Aborting.";
    exit 1;
}

DOCKER_COMPOSE_CMD="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
fi

check_port() {
    local port=$1
    if python3 -c "import socket; s = socket.socket(); s.bind(('0.0.0.0', $port)); s.close()" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

for port in 7474 7687; do
    if ! check_port "$port"; then
        log_err "Port $port is currently occupied on host. Free this port to allow Neo4j binding."
        exit 1
    fi
done

# ---------------------------------------------------------------------
# Project Directory Scaffolding
# ---------------------------------------------------------------------
PROJECT_ROOT="m4_metamemory_engine"
log_info "Scaffolding project structure in ./${PROJECT_ROOT}..."
mkdir -p "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/tests"
mkdir -p "${PROJECT_ROOT}/neo4j_init"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------
# File Generation via Heredocs
# ---------------------------------------------------------------------

log_info "Writing project configuration files..."

cat << 'EOF' > .env.example
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=StrongPassword123!
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=qwen2.5:7b-instruct
EMBEDDING_MODEL=all-minilm
SIMILARITY_THRESHOLD=0.5
LOG_LEVEL=INFO
EOF

cp .env.example .env

cat << 'EOF' > requirements.txt
pydantic==2.8.2
pydantic-settings==2.3.4
neo4j==5.23.0
httpx==0.27.0
numpy==1.26.4
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-mock==3.14.0
tenacity==8.5.0
python-dotenv==1.0.1
EOF

cat << 'EOF' > neo4j_init/init_schema.cypher
// Uniqueness constraints
CREATE CONSTRAINT unique_memory_node_id IF NOT EXISTS
FOR (n:MemoryNode) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT unique_memory_chain_id IF NOT EXISTS
FOR (c:MemoryChain) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT unique_memory_chain_category IF NOT EXISTS
FOR (c:MemoryChain) REQUIRE c.category_name IS UNIQUE;

// Indexes for fast lookups
CREATE INDEX index_memory_node_category IF NOT EXISTS
FOR (n:MemoryNode) ON (n.category);

CREATE INDEX index_memory_node_session IF NOT EXISTS
FOR (n:MemoryNode) ON (n.session_id);

// Native 384-dimensional Vector Index for Category Chains
CREATE VECTOR INDEX category_chain_embeddings IF NOT EXISTS
FOR (c:MemoryChain) ON (c.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 384,
 `vector.similarity_function`: 'cosine'
}};

// Native 384-dimensional Vector Index for Memory Nodes
CREATE VECTOR INDEX memory_node_embeddings IF NOT EXISTS
FOR (n:MemoryNode) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 384,
 `vector.similarity_function`: 'cosine'
}};
EOF

cat << 'EOF' > Dockerfile
# Multi-stage production build
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
RUN addgroup --system appgroup && adduser --system --group appuser

COPY --from=builder /root/.local /home/appuser/.local
COPY . /app

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

ENTRYPOINT ["python"]
CMD ["demo.py"]
EOF

cat << 'EOF' > docker-compose.yml
services:
  neo4j:
    image: neo4j:5.23.0-community
    container_name: m4_neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/StrongPassword123!
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_db_tx__log_rotation_retention__policy=2 files
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p StrongPassword123! 'RETURN 1' || exit 1"]
      interval: 5s
      timeout: 5s
      retries: 20
      start_period: 10s
    volumes:
      - neo4j_data:/data
      - ./neo4j_init:/var/lib/neo4j/import

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: m4_app
    depends_on:
      neo4j:
        condition: service_healthy
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - .:/app
    command: ["demo.py"]

volumes:
  neo4j_data:
EOF

# ---------------------------------------------------------------------
# Python Core Implementation
# ---------------------------------------------------------------------

log_info "Writing Python domain logic and core layers..."

cat << 'EOF' > src/__init__.py
"""Meta-Memory (M4) Engine Package."""
EOF

cat << 'EOF' > src/config.py
import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    neo4j_uri: str = Field(default="bolt://neo4j:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(default="StrongPassword123!", validation_alias="NEO4J_PASSWORD")
    ollama_base_url: str = Field(default="http://host.docker.internal:11434", validation_alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="LLM_MODEL")
    embedding_model: str = Field(default="all-minilm", validation_alias="EMBEDDING_MODEL")
    similarity_threshold: float = Field(default=0.5, validation_alias="SIMILARITY_THRESHOLD")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def setup_logger(name: str = "m4") -> logging.Logger:
    config = AppConfig()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(config.log_level.upper())
    return logger
EOF

cat << 'EOF' > src/models.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class RawMemoryItem(BaseModel):
    category: str = Field(..., alias="Category")
    summary: str = Field(..., alias="Summary")

class DisambiguatedMemoryClue(BaseModel):
    category: str
    summary: str
    category_embedding: List[float]
    clue_embedding: List[float]

class MemoryNodeModel(BaseModel):
    id: str
    summary: str
    category: str
    session_id: str
    created_at: datetime
    last_used_at: datetime
    embedding: List[float]

class MemoryChainModel(BaseModel):
    id: str
    category_name: str
    embedding: List[float]
    node_count: int = 0
EOF

cat << 'EOF' > src/llm_client.py
import json
import re
import httpx
from typing import List, Dict, Any
from src.config import AppConfig, setup_logger
from src.models import RawMemoryItem

logger = setup_logger("m4.llm_client")

class LLMClientException(Exception):
    """Raised when interaction with Ollama LLM / Embedding endpoints fails."""
    pass

class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.http_client = httpx.Client(timeout=45.0)

    def get_embedding(self, text: str) -> List[float]:
        """Calls native Ollama /api/embeddings endpoint."""
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/embeddings"
        payload = {
            "model": self.config.embedding_model,
            "prompt": text
        }
        try:
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise LLMClientException(f"Invalid embedding payload returned: {data}")
            return data["embedding"]
        except Exception as e:
            logger.error("Failed to generate embedding for input text: %s", str(e))
            raise LLMClientException(f"Embedding generation failure: {e}") from e

    def complete_chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Calls OpenAI-compatible /v1/chat/completions endpoint on Ollama."""
        url = f"{self.config.ollama_base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature
        }
        try:
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Chat completion request failed: %s", str(e))
            raise LLMClientException(f"Chat completion failure: {e}") from e

    def extract_structured_json(self, prompt: str) -> List[RawMemoryItem]:
        """Requests extraction and validates JSON against schema with 1 retry."""
        messages = [
            {"role": "system", "content": "You are a specialized memory extraction engine. Respond strictly with a JSON array."},
            {"role": "user", "content": prompt}
        ]
        raw_text = self.complete_chat(messages, temperature=0.0)
        parsed = self._parse_json_array(raw_text)
        if parsed is not None:
            return parsed

        logger.warn("Initial JSON parse failed. Retrying with error feedback prompt...")
        retry_messages = [
            {"role": "system", "content": "You must correct invalid JSON. Output ONLY valid JSON array with objects matching keys 'Category' and 'Summary'."},
            {"role": "user", "content": f"The following output was invalid JSON:\n{raw_text}\nFix it and output ONLY the valid JSON array."}
        ]
        second_attempt = self.complete_chat(retry_messages, temperature=0.0)
        parsed_second = self._parse_json_array(second_attempt)
        if parsed_second is not None:
            return parsed_second

        raise LLMClientException(f"Failed to extract valid JSON memory clues after 1 retry. Output: {second_attempt}")

    def _parse_json_array(self, text: str) -> Optional[List[RawMemoryItem]]:
        clean_text = text.strip()
        # Strip markdown block ticks if present
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
            if array_match:
                clean_text = array_match.group(0)

        try:
            data = json.loads(clean_text)
            if not isinstance(data, list):
                return None
            items = [RawMemoryItem.model_validate(item) for item in data]
            return items
        except Exception as ex:
            logger.debug("JSON array parse attempt rejected: %s", str(ex))
            return None
EOF

cat << 'EOF' > src/graph_store.py
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
EOF

cat << 'EOF' > src/layer1_learning.py
import numpy as np
from typing import List
from src.config import AppConfig, setup_logger
from src.llm_client import OllamaClient
from src.models import RawMemoryItem, DisambiguatedMemoryClue

logger = setup_logger("m4.layer1")

def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two dense vectors (Eq. 1)."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class CategoryBasedMemoryLearning:
    """
    Implements Layer 1 (§III-B, Equation 1):
    Adaptively extracts memory clues and merges ambiguous/redundant categories
    using vector similarity before chain formation.
    """
    def __init__(self, config: AppConfig, client: OllamaClient):
        self.config = config
        self.client = client

    def format_extraction_prompt(self, conversation_text: str) -> str:
        return f"""Here is a conversation:
{conversation_text}

Please analyze the conversation and identify distinct pieces of information to memorize.
Focus exclusively on details specific to the speakers or the situation (e.g., personal preferences, plans, specific events, relationships) and avoid summarizing common knowledge you already possess. This enables you to autonomously identify what needs to be memorized.
For each identified piece of memory, classify it into a distinct and natural memory category and provide a concise summary.
Return a JSON array of objects, where each object represents a memory item.

Example JSON Response:
[
  {{"Category": "Personal Preferences", "Summary": "The user prefers black coffee without sugar."}},
  {{"Category": "Life Events", "Summary": "The user adopted a golden retriever named Max last Friday."}}
]

Now, for the given conversation, provide the JSON response without any reasoning:"""

    def process_session(self, conversation_text: str) -> List[DisambiguatedMemoryClue]:
        logger.info("Extracting candidate memory categories and clues...")
        prompt = self.format_extraction_prompt(conversation_text)
        raw_items: List[RawMemoryItem] = self.client.extract_structured_json(prompt)
        if not raw_items:
            logger.warn("No memory clues identified by LLM in session.")
            return []

        logger.info("Extracted %d raw candidate items. Computing embeddings...", len(raw_items))
        category_embeddings = [self.client.get_embedding(item.category) for item in raw_items]
        clue_embeddings = [self.client.get_embedding(item.summary) for item in raw_items]

        # Intra-session category disambiguation and merge (Eq. 1)
        # If sim(t_i, t_j) >= gamma, merge category t_j into t_i
        gamma = self.config.similarity_threshold
        canonical_items: List[DisambiguatedMemoryClue] = []
        visited = [False] * len(raw_items)

        for i in range(len(raw_items)):
            if visited[i]:
                continue
            canonical_cat = raw_items[i].category
            canonical_cat_emb = category_embeddings[i]
            visited[i] = True

            # Check subsequent items for category merge
            for j in range(i + 1, len(raw_items)):
                if not visited[j]:
                    sim = compute_cosine_similarity(canonical_cat_emb, category_embeddings[j])
                    if sim >= gamma:
                        logger.info("Eq. 1 Merge: Merging category '%s' into canonical '%s' (sim=%.4f >= gamma=%.2f)",
                                    raw_items[j].category, canonical_cat, sim, gamma)
                        visited[j] = True
                        canonical_items.append(DisambiguatedMemoryClue(
                            category=canonical_cat,
                            summary=raw_items[j].summary,
                            category_embedding=canonical_cat_emb,
                            clue_embedding=clue_embeddings[j]
                        ))

            canonical_items.append(DisambiguatedMemoryClue(
                category=canonical_cat,
                summary=raw_items[i].summary,
                category_embedding=canonical_cat_emb,
                clue_embedding=clue_embeddings[i]
            ))

        logger.info("Disambiguation complete: produced %d consolidated clues.", len(canonical_items))
        return canonical_items
EOF

cat << 'EOF' > src/layer2_construction.py
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
EOF

cat << 'EOF' > src/layer3_updating.py
"""
Layer 3: Self-monitoring-based Memory Updating (§III-D, Equations 4-7)
Architecture Specification & Stubs for Next Sprint.
"""
from typing import List
from src.config import AppConfig, setup_logger
from src.models import MemoryNodeModel

logger = setup_logger("m4.layer3")

class SelfMonitoringMemoryUpdating:
    """
    STUB IMPLEMENTATION: Reserved for Layer 3.
    Mapped to:
      - Eq. 4: Inconsistent memory detection {c_i^u} <- LLM_u(C_k; c_k)
      - Eq. 5: Merging conflict to unambiguous node \hat{c}_k <- LLM_m(c_i^u; c_k)
      - Eq. 6: Removal of obsolete node and linking calibrated node C_k <- f_c(C_k || \hat{c}_k)
      - Eq. 7: Temporal compression c_i^{com} = LLMLingua(c_i) based on Ebbinghaus forgetting curve.
    """
    def __init__(self, config: AppConfig):
        self.config = config

    def calibrate_chain(self, chain_id: str, new_node: MemoryNodeModel) -> bool:
        """
        Executes Eq. 4-6 conflict check and calibration against the chain.
        Docstring: Will prompt LLM to detect contradictions between new_node and existing chain nodes,
        rewrite the fact, and atomically update the graph.
        """
        logger.info("[STUB Layer 3] calibrate_chain called for chain %s with node %s. Skipping (unimplemented).",
                    chain_id, new_node.id)
        return True

    def compress_dormant_nodes(self, days_idle: int = 14) -> int:
        """
        Executes Eq. 7 compression via LLMLingua-2 with LLM recoverability check.
        Docstring: Finds nodes with (now - last_used_at) > days_idle, applies token pruning,
        verifies reconstruction, and persists compressed representation.
        """
        logger.info("[STUB Layer 3] compress_dormant_nodes called for threshold %d days. Skipping.", days_idle)
        return 0
EOF

cat << 'EOF' > src/layer4_retrieval.py
"""
Layer 4: Self-reflection-based Memory Retrieval (§III-E, Equations 8-9, Algorithm 1)
Architecture Specification & Stubs for Next Sprint.
"""
from typing import List, Tuple
from src.config import AppConfig, setup_logger

logger = setup_logger("m4.layer4")

class SelfReflectionMemoryRetrieval:
    """
    STUB IMPLEMENTATION: Reserved for Layer 4.
    Mapped to:
      - §III-E Category Retrieval: Prompting for T^q relevant categories.
      - Eq. 8: Bidirectional chain expansion M <- f_c(M || c_{p_f}^q) if LLM_re(c_{p_f}^q)
      - Algorithm 1: Dynamic loop checking LLM_confident(q, M) to stop search.
      - Eq. 9: Context-aware response generation r = LLM(q, M).
    """
    def __init__(self, config: AppConfig):
        self.config = config

    def retrieve_memory(self, query: str) -> Tuple[List[str], str]:
        """
        Executes Algorithm 1 bidirectional traversal.
        Docstring: Identifies category, seeds search with closest node, expands forward and backward
        until self-evaluation indicates sufficient context, then synthesizes answer.
        """
        logger.info("[STUB Layer 4] retrieve_memory called for query: '%s'. Skipping.", query)
        return ([], "Layer 4 stub: retrieval will be implemented in subsequent sprint.")
EOF

cat << 'EOF' > src/pipeline.py
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
EOF

# ---------------------------------------------------------------------
# Unit and Integration Tests
# ---------------------------------------------------------------------

log_info "Writing automated pytest test suite..."

cat << 'EOF' > tests/test_unit.py
import pytest
from unittest.mock import MagicMock
from src.config import AppConfig
from src.models import RawMemoryItem, DisambiguatedMemoryClue
from src.layer1_learning import CategoryBasedMemoryLearning, compute_cosine_similarity
from src.layer2_construction import ChainBasedMemoryConstruction

def test_cosine_similarity_identity():
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert pytest.approx(compute_cosine_similarity(vec1, vec2), 0.0001) == 1.0

def test_cosine_similarity_orthogonal():
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    assert pytest.approx(compute_cosine_similarity(vec1, vec2), 0.0001) == 0.0

def test_layer1_category_merge_eq1():
    """Validates that categories with similarity >= gamma are merged (Eq. 1)."""
    config = AppConfig(similarity_threshold=0.7)
    mock_client = MagicMock()

    # Raw extracted items: two categories are semantically identical
    mock_client.extract_structured_json.return_value = [
        RawMemoryItem(Category="Dietary Preference", Summary="Drinks green tea"),
        RawMemoryItem(Category="Dietary Choices", Summary="Avoids red meat")
    ]

    # Return vectors: similarity = 0.99 (Dietary Preference vs Dietary Choices)
    mock_client.get_embedding.side_effect = [
        [1.0, 0.0, 0.0],  # Cat 1
        [0.99, 0.01, 0.0], # Cat 2
        [0.1, 0.2, 0.3],  # Clue 1
        [0.4, 0.5, 0.6],  # Clue 2
    ]

    layer1 = CategoryBasedMemoryLearning(config, mock_client)
    result = layer1.process_session("Dialogue text placeholder")

    assert len(result) == 2
    # Both items must be collapsed to the canonical category
    assert result[0].category == "Dietary Preference"
    assert result[1].category == "Dietary Preference"

def test_layer2_chain_linking_eq2_3():
    """Validates chain lookup and conditional creation (Eq. 2-3)."""
    config = AppConfig(similarity_threshold=0.5)
    mock_store = MagicMock()

    # Simulate existing chain with similarity 0.85 (above threshold 0.5)
    mock_store.find_most_similar_chain.return_value = ("chain_123", "Fitness Routine", 0.85)

    clues = [
        DisambiguatedMemoryClue(
            category="Workout Habits",
            summary="User goes swimming every Tuesday.",
            category_embedding=[0.8, 0.2],
            clue_embedding=[0.1, 0.9]
        )
    ]

    layer2 = ChainBasedMemoryConstruction(config, mock_store)
    created_ids = layer2.build_memory_chains(clues, session_id="test_sess_01")

    assert len(created_ids) == 1
    # Must append to existing matched chain "Fitness Routine"
    mock_store.append_node_to_chain.assert_called_once()
    appended_node, target_cat = mock_store.append_node_to_chain.call_args[0]
    assert target_cat == "Fitness Routine"
    assert appended_node.summary == "User goes swimming every Tuesday."
EOF

cat << 'EOF' > tests/test_integration.py
import pytest
from datetime import datetime, timezone
from src.config import AppConfig
from src.graph_store import Neo4jGraphStore
from src.models import MemoryNodeModel

@pytest.fixture(scope="module")
def graph_store():
    config = AppConfig()
    store = Neo4jGraphStore(config)
    try:
        store.connect()
    except Exception as e:
        pytest.skip(f"Neo4j container not accessible for integration test: {e}")
    yield store
    store.close()

def test_neo4j_chain_topology(graph_store):
    """
    Verifies live Neo4j:
      1. Creation of unique MemoryChain
      2. Appending 2 ordered nodes via [:NEXT_IN_CHAIN]
      3. Validating sequential retrieval order
    """
    category_name = f"TestCategory_{int(datetime.now().timestamp())}"
    chain_id = f"test_chain_{int(datetime.now().timestamp())}"
    dummy_embedding = [0.05] * 384

    # 1. Create chain
    c_id = graph_store.create_memory_chain(chain_id, category_name, dummy_embedding)
    assert c_id == chain_id

    # 2. Append node 1
    now = datetime.now(timezone.utc)
    node1 = MemoryNodeModel(
        id=f"n1_{chain_id}",
        summary="First sequential memory clue.",
        category=category_name,
        session_id="sess_integration_01",
        created_at=now,
        last_used_at=now,
        embedding=dummy_embedding
    )
    graph_store.append_node_to_chain(node1, category_name)

    # 3. Append node 2
    node2 = MemoryNodeModel(
        id=f"n2_{chain_id}",
        summary="Second sequential memory clue.",
        category=category_name,
        session_id="sess_integration_02",
        created_at=now,
        last_used_at=now,
        embedding=dummy_embedding
    )
    graph_store.append_node_to_chain(node2, category_name)

    # 4. Verify ordered traversal along [:NEXT_IN_CHAIN]
    nodes = graph_store.get_chain_nodes(category_name)
    assert len(nodes) == 2
    assert nodes[0]["id"] == node1.id
    assert nodes[1]["id"] == node2.id
    assert nodes[0]["summary"] == "First sequential memory clue."
    assert nodes[1]["summary"] == "Second sequential memory clue."
EOF

# ---------------------------------------------------------------------
# End-to-End Demo Script
# ---------------------------------------------------------------------

log_info "Writing end-to-end multi-session evaluation demonstration..."

cat << 'EOF' > demo.py
import sys
from src.config import AppConfig, setup_logger
from src.pipeline import MetaMemoryPipeline
from src.models import RawMemoryItem

logger = setup_logger("m4.demo")

# Multi-session benchmark dialogues modeling persona evolution and shared nodes
SESSION_1 = """
User: Hi! I am a software engineer working in robotics. I usually start my mornings with a cup of strong black coffee.
Assistant: Nice to meet you! Coffee and robotics sound like an energetic start to the day.
User: Yeah, and next weekend I'm planning to adopt a German Shepherd puppy.
Assistant: That's a big step! Puppies bring so much joy and require lots of activity.
"""

SESSION_2 = """
User: Hey again! Quick update: I ended up adopting the German Shepherd! We named him Buster.
Assistant: Congratulations on Buster! How has the first week been?
User: He has so much energy. I also swapped my morning black coffee for green tea this week to lower caffeine.
Assistant: That is a healthy switch. How is Buster adjusting to the daily routine?
"""

SESSION_3 = """
User: Buster loved our morning walk today. By the way, at work we started using ROS 2 for our autonomous navigation stack.
Assistant: That's awesome. Buster sounds like a great companion, and ROS 2 is great for distributed robotics systems!
"""

def main():
    logger.info("Initializing Meta-Memory (M4) Production Pipeline...")
    config = AppConfig()
    pipeline = MetaMemoryPipeline(config)

    try:
        pipeline.startup()
    except Exception as e:
        logger.error("Failed to connect to Neo4j. Verify container is healthy. Error: %s", str(e))
        sys.exit(1)

    # Graceful fallback mock if local Ollama service is not currently active on host
    try:
        pipeline.llm_client.get_embedding("health check probe")
        logger.info("Local Ollama connection verified (/api/embeddings & /v1/chat/completions active).")
    except Exception as e:
        logger.warn("Ollama not currently reachable at %s. Engaging deterministic mock provider for demo run.",
                    config.ollama_base_url)
        # Monkey patch with predictable mocked embeddings & extraction for dry verification
        mock_categories = [
            [RawMemoryItem(Category="Daily Habits", Summary="Prefers strong black coffee in the morning."),
             RawMemoryItem(Category="Pets & Family", Summary="Planning to adopt a German Shepherd puppy.")],
            [RawMemoryItem(Category="Daily Habits", Summary="Switched from black coffee to green tea."),
             RawMemoryItem(Category="Pets & Family", Summary="Adopted German Shepherd puppy named Buster.")],
            [RawMemoryItem(Category="Professional Tech", Summary="Using ROS 2 for robotics navigation stack."),
             RawMemoryItem(Category="Pets & Family", Summary="Buster enjoys regular morning walks.")]
        ]
        
        def mock_extract(prompt: str):
            if "Buster loved our morning walk" in prompt:
                return mock_categories[2]
            elif "Quick update: I ended up adopting" in prompt:
                return mock_categories[1]
            return mock_categories[0]

        def deterministic_vector(text: str):
            val = float(sum(ord(c) for c in text) % 100) / 100.0
            vec = [0.01 * val] * 384
            vec[0] = 1.0 if "Habits" in text or "coffee" in text or "tea" in text else 0.05
            vec[1] = 1.0 if "Pets" in text or "Shepherd" in text or "Buster" in text else 0.05
            vec[2] = 1.0 if "Tech" in text or "robotics" in text or "ROS" in text else 0.05
            return vec

        pipeline.llm_client.extract_structured_json = mock_extract
        pipeline.llm_client.get_embedding = deterministic_vector

    # Execute 3 consecutive sessions to demonstrate Layer 1 & 2 evolution
    sessions = [
        ("sess_001_morning_routine", SESSION_1),
        ("sess_002_puppy_and_diet", SESSION_2),
        ("sess_003_work_and_pets", SESSION_3)
    ]

    for sess_id, dialogue in sessions:
        res = pipeline.ingest_session(sess_id, dialogue)
        logger.info("Session Result: %s", res)

    # Inspect the resulting Graph Database
    logger.info("=== Graph Store Inspection (Verification of Memory Chains) ===")
    with pipeline.graph_store.session() as s:
        chains = s.run("MATCH (c:MemoryChain) RETURN c.category_name AS name ORDER BY name").data()
        logger.info("Created Chains: %s", [c['name'] for c in chains])

        for c in chains:
            cat_name = c['name']
            nodes = pipeline.graph_store.get_chain_nodes(cat_name)
            logger.info("Chain ['%s'] (Length=%d):", cat_name, len(nodes))
            for idx, n in enumerate(nodes):
                logger.info("  [%d] (Session: %s) -> %s", idx, n['session_id'], n['summary'])

        shared_count = s.run("MATCH ()-[r:SHARED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]
        logger.info("Total cross-category [:SHARED_WITH] graph relationships: %d", shared_count)

    pipeline.shutdown()
    logger.info("M4 Demonstration Completed Successfully.")

if __name__ == "__main__":
    main()
EOF

# ---------------------------------------------------------------------
# Readme Documentation
# ---------------------------------------------------------------------

cat << 'EOF' > README.md
# Meta-Memory for Large Language Models ($M^4$) — Layers 1 & 2

Production implementation of Layer 1 (Category-based Memory Learning) and Layer 2 (Chain-based Memory Construction) based on Liang et al. (IEEE TASLP 2026).

## Architecture Stack
- **Graph Engine:** Neo4j 5.23.0 Community Edition
- **Native Vector Index:** 384-dimensional cosine index on `:MemoryChain` and `:MemoryNode`
- **Embeddings & LLM:** Local Ollama running `all-minilm` (384-dim) and `qwen2.5:7b-instruct`
- **Configuration & Validation:** Pydantic 2.8 + Pydantic-Settings

## Quickstart
1. Ensure Ollama is running on the host machine:
   ```bash
   ollama run qwen2.5:7b-instruct
   ollama run all-minilm