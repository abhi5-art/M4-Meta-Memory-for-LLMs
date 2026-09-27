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
