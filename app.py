
# import streamlit as st
# import time
# from src.config import AppConfig
# from src.pipeline import MetaMemoryPipeline

# # Page Configuration
# st.set_page_config(
#     page_title="M4 Meta-Memory Engine Inspector",
#     page_icon="🧠",
#     layout="wide"
# )

# # Initialize App Configuration & Pipeline
# @st.cache_resource
# def get_pipeline():
#     config = AppConfig()
#     pipeline = MetaMemoryPipeline(config)
#     pipeline.startup()
#     return pipeline

# try:
#     pipeline = get_pipeline()
# except Exception as e:
#     st.error(f"Failed to connect to Neo4j or initialize pipeline: {e}")
#     st.stop()

# # Preset Benchmark Dialogues
# SESSION_1 = """
# User: Hi! I am a backend developer working on distributed API gateways using Python and FastAPI. I usually start my mornings reviewing database logs.
# Assistant: Sounds like a high-performance start to the day. FastAPI is great for asynchronous workloads.
# User: Yeah, and next weekend I'm planning to migrate our caching layer to Redis.
# Assistant: That will definitely speed up response times. Make sure to handle eviction policies properly.
# """

# SESSION_2 = """
# User: Hey again! Quick update: I finished the Redis migration for the caching layer. 
# Assistant: Excellent work! How did the performance metrics look after deployment?
# User: Latency dropped by 40%. I also switched my morning review routine to focus on error tracking dashboards instead of raw logs.
# Assistant: That is a much more efficient way to monitor health. How are the API endpoints holding up under peak load?
# """

# SESSION_3 = """
# User: The API endpoints handled peak traffic smoothly today. By the way, we started containerizing our microservices using Docker Compose.
# Assistant: That's awesome. Docker Compose will make local orchestration and testing much cleaner!
# """


# SESSION_OFF_CONTEXT_1 = """
# User: Hi! I am trying to master authentic Neapolitan pizza dough and sourdough fermentation this weekend.
# Assistant: Baking is a wonderful science! Are you using a preferment or a direct dough method?
# User: Using a 24-hour cold fermentation poolish with Caputo 00 flour. Also picked up a cast-iron skillet for baking.
# Assistant: That will give you a fantastic crust and char. Don't forget high hydration ratios!
# """

# SESSION_OFF_CONTEXT_2 = """
# User: Hey! Switching gears completely—I'm learning fingerstyle acoustic guitar and practicing alternate tunings like DADGAD.
# Assistant: DADGAD opens up wonderful sonic textures for Celtic and folk arrangements. How are your chord transitions coming along?
# User: Slowly but steadily. Also bought a solid mahogany acoustic guitar to get a warmer tone.
# Assistant: Mahogany bodies provide a gorgeous woody mid-range response. Keep at it!
# """


# st.title("🧠 Meta-Memory ($M^4$) Engine Dashboard")
# st.markdown("### Final-Year B.Tech Project Demonstration (Layers 1 & 2 Verification)")

# # --- SIDEBAR CONTROLS ---
# with st.sidebar:
#     st.header("⚙️ Control Panel")
    
#     # Clear Database Button
#     if st.button("🗑️ Clear Database (Reset Graph)", type="primary"):
#         try:
#             with pipeline.graph_store.session() as s:
#                 s.run("MATCH (n0) DETACH DELETE n0;")
#                 # Re-create vector index
#                 s.run("""
#                     CREATE VECTOR INDEX category_chain_embeddings IF NOT EXISTS
#                     FOR (c:MemoryChain)
#                     ON (c.embedding)
#                     OPTIONS { indexConfig: {
#                       `vector.dimensions`: 384,
#                       `vector.similarity_function`: 'cosine'
#                     }};
#                 """)
#             st.success("Database wiped and vector index reinitialized successfully!")
#             time.sleep(1)
#             st.rerun()
#         except Exception as err:
#             st.error(f"Error clearing database: {err}")

#     st.markdown("---")
#     st.markdown("### 📊 System Status")
#     st.info(f"**Neo4j URI:** `{pipeline.config.neo4j_uri}`")
#     st.info(f"**Ollama Host:** `{pipeline.config.ollama_base_url}`")

# # --- MAIN INTERFACE TABS ---
# tab1, tab2 = st.tabs(["🚀 Live Ingestion & Simulation", "🔍 Graph Store Inspector"])

# with tab1:
#     st.subheader("Interactive Multi-Session Ingestion")
#     st.markdown("Run preset sessions sequentially or input custom dialogue to observe Layer 1 extraction and Layer 2 chain linking.")

#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("Run Session 1 (Backend Ops)"):
#             with st.spinner("Ingesting Session 1..."):
#                 res = pipeline.ingest_session("sess_001_backend_ops", SESSION_1)
#                 st.success("Session 1 Ingested!")
#                 st.json(res)

#     with col2:
#         if st.button("Run Session 2 (Cache Migration)"):
#             with st.spinner("Ingesting Session 2..."):
#                 res = pipeline.ingest_session("sess_002_cache_migration", SESSION_2)
#                 st.success("Session 2 Ingested!")
#                 st.json(res)

#     with col3:
#         if st.button("Run Session 3 (Containerization)"):
#             with st.spinner("Ingesting Session 3..."):
#                 res = pipeline.ingest_session("sess_003_containerization", SESSION_3)
#                 st.success("Session 3 Ingested!")
#                 st.json(res)

#     st.markdown("---")
#     st.markdown("### Or Ingest Custom Dialogue")
#     custom_session_id = st.text_input("Session ID", value="sess_custom_001")
#     custom_dialogue = st.text_area("Dialogue Transcript", value="User: Hi, I'm testing custom memory extraction.\nAssistant: Let's see how it behaves!")
    
#     if st.button("Ingest Custom Session"):
#         if custom_dialogue.strip():
#             with st.spinner("Processing through Layer 1 & Layer 2..."):
#                 res = pipeline.ingest_session(custom_session_id, custom_dialogue)
#                 st.success("Custom session processed and committed to Neo4j graph!")
#                 st.json(res)
#         else:
#             st.warning("Please provide dialogue text.")

# with tab2:
#     st.subheader("Neo4j Memory Chain Inspection")
#     st.markdown("Visualizing active category chains, chronological node sequences (`[:NEXT_IN_CHAIN]`), and cross-category edges.")

#     if st.button("Refresh Graph View"):
#         st.rerun()

#     try:
#         with pipeline.graph_store.session() as s:
#             chains = s.run("MATCH (c:MemoryChain) RETURN c.category_name AS name ORDER BY name").data()
            
#             if not chains:
#                 st.warning("No memory chains found in the database. Run a session in Tab 1 or check if the database was cleared.")
#             else:
#                 for c in chains:
#                     cat_name = c['name']
#                     nodes = pipeline.graph_store.get_chain_nodes(cat_name)
                    
#                     with st.expander(f"📁 Category Chain: `{cat_name}` (Total Nodes: {len(nodes)})", expanded=True):
#                         for idx, n in enumerate(nodes):
#                             st.markdown(f"**[{idx}] Session:** `{n.get('session_id', 'N/A')}`")
#                             st.markdown(f"> *Summary:* {n.get('summary', '')}")
#                             st.caption(f"Node ID: `{n.get('id', '')}` | Created: `{n.get('created_at', '')}`")
#                             if idx < len(nodes) - 1:
#                                 st.markdown("⬇️ `[:NEXT_IN_CHAIN]`")

#                 shared_count = s.run("MATCH ()-[r:SHARED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]
#                 st.metric(label="Total Cross-Category `[:SHARED_WITH]` Relationships", value=shared_count)

#     except Exception as err:
#         st.error(f"Error querying Neo4j graph store: {err}")














#  The below code is for layer 1,2,3,4 combined ----------------------------------------------------------









import streamlit as st
import time
from src.config import AppConfig
from src.pipeline import MetaMemoryPipeline

# Page Configuration
st.set_page_config(
    page_title="M4 Meta-Memory Engine Inspector",
    page_icon="🧠",
    layout="wide"
)

# Initialize App Configuration & Pipeline
@st.cache_resource
def get_pipeline():
    config = AppConfig()
    pipeline = MetaMemoryPipeline(config)
    pipeline.startup()
    return pipeline

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to connect to Neo4j or initialize pipeline: {e}")
    st.stop()

# Preset Benchmark Dialogues
SESSION_1 = """
User: Hi! I am a backend developer working on distributed API gateways using Python and FastAPI. I usually start my mornings reviewing database logs.
Assistant: Sounds like a high-performance start to the day. FastAPI is great for asynchronous workloads.
User: Yeah, and next weekend I'm planning to migrate our caching layer to Redis.
Assistant: That will definitely speed up response times. Make sure to handle eviction policies properly.
"""

SESSION_2 = """
User: Hey again! Quick update: I finished the Redis migration for the caching layer. 
Assistant: Excellent work! How did the performance metrics look after deployment?
User: Latency dropped by 40%. I also switched my morning review routine to focus on error tracking dashboards instead of raw logs.
Assistant: That is a much more efficient way to monitor health. How are the API endpoints holding up under peak load?
"""

SESSION_3 = """
User: The API endpoints handled peak traffic smoothly today. By the way, we started containerizing our microservices using Docker Compose.
Assistant: That's awesome. Docker Compose will make local orchestration and testing much cleaner!
"""

SESSION_OFF_CONTEXT_1 = """
User: Hi! I am trying to master authentic Neapolitan pizza dough and sourdough fermentation this weekend.
Assistant: Baking is a wonderful science! Are you using a preferment or a direct dough method?
User: Using a 24-hour cold fermentation poolish with Caputo 00 flour. Also picked up a cast-iron skillet for baking.
Assistant: That will give you a fantastic crust and char. Don't forget high hydration ratios!
"""

SESSION_OFF_CONTEXT_2 = """
User: Hey! Switching gears completely—I'm learning fingerstyle acoustic guitar and practicing alternate tunings like DADGAD.
Assistant: DADGAD opens up wonderful sonic textures for Celtic and folk arrangements. How are your chord transitions coming along?
User: Slowly but steadily. Also bought a solid mahogany acoustic guitar to get a warmer tone.
Assistant: Mahogany bodies provide a gorgeous woody mid-range response. Keep at it!
"""

st.title("🧠 Meta-Memory ($M^4$) Engine Dashboard")
st.markdown("### Full System Verification: Layers 1–4[cite: 2]")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.button("🗑️ Clear Database (Reset Graph)", type="primary"):
        try:
            with pipeline.graph_store.session() as s:
                s.run("MATCH (n0) DETACH DELETE n0;")
                s.run("""
                    CREATE VECTOR INDEX category_chain_embeddings IF NOT EXISTS
                    FOR (c:MemoryChain)
                    ON (c.embedding)
                    OPTIONS { indexConfig: {
                      `vector.dimensions`: 384,
                      `vector.similarity_function`: 'cosine'
                    }};
                """)
            st.success("Database wiped and vector index reinitialized successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as err:
            st.error(f"Error clearing database: {err}")

    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.info(f"**Neo4j URI:** `{pipeline.config.neo4j_uri}`")
    st.info(f"**Ollama Host:** `{pipeline.config.ollama_base_url}`")

# --- MAIN INTERFACE TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Ingestion & Update (Layers 1-3)", "💬 Self-Reflection Retrieval (Layer 4)", "🔍 Graph Store Inspector"])

with tab1:
    st.subheader("Interactive Multi-Session Ingestion")
    st.markdown("Triggers Layer 1 Extraction, Layer 2 Chain Linking, and Layer 3 Calibration (Eq. 4-6) & Compression (Eq. 7)[cite: 2].")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Ingest Session 1 (Backend Ops)"):
            with st.spinner("Ingesting Session 1..."):
                res = pipeline.ingest_session("sess_001_backend", SESSION_1)
                st.success("Session 1 Ingested!")
                st.json(res)
        if st.button("Ingest Off-Context 1 (Baking)"):
            with st.spinner("Ingesting Off-Context 1..."):
                res = pipeline.ingest_session("sess_off_001_baking", SESSION_OFF_CONTEXT_1)
                st.success("Off-Context 1 Ingested!")
                st.json(res)

    with col2:
        if st.button("Ingest Session 2 (Cache Migration)"):
            with st.spinner("Calibrating updates against Session 1..."):
                res = pipeline.ingest_session("sess_002_cache", SESSION_2)
                st.success("Session 2 Ingested (Calibration applied)!")
                st.json(res)
        if st.button("Ingest Off-Context 2 (Guitar)"):
            with st.spinner("Ingesting Off-Context 2..."):
                res = pipeline.ingest_session("sess_off_002_guitar", SESSION_OFF_CONTEXT_2)
                st.success("Off-Context 2 Ingested!")
                st.json(res)

    with col3:
        if st.button("Ingest Session 3 (Containerization)"):
            with st.spinner("Ingesting Session 3..."):
                res = pipeline.ingest_session("sess_003_docker", SESSION_3)
                st.success("Session 3 Ingested!")
                st.json(res)

    st.markdown("---")
    st.markdown("### Or Ingest Custom Dialogue")
    custom_session_id = st.text_input("Session ID", value="sess_custom_001")
    custom_dialogue = st.text_area("Dialogue Transcript", value="User: I started using PostgreSQL today.\nAssistant: Great choice for relational data.")
    
    if st.button("Ingest Custom Session"):
        if custom_dialogue.strip():
            with st.spinner("Processing through Layers 1, 2, and 3..."):
                res = pipeline.ingest_session(custom_session_id, custom_dialogue)
                st.success("Custom session processed, calibrated, and committed to Neo4j graph!")
                st.json(res)
        else:
            st.warning("Please provide dialogue text.")

with tab2:
    st.subheader("Memory-Driven Chat Interface")
    st.markdown("Executes Algorithm 1 bidirectional traversal with $LLM_{re}$ and $LLM_{confident}$ stopping criteria (Eq. 8-9)[cite: 2].")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    query = st.chat_input("Ask a question about the ingested memories (e.g., 'What is my morning routine?')")
    if query:
        st.chat_message("user").write(query)
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("Traversing memory topology (Algorithm 1)..."):
            try:
                response = pipeline.layer4.retrieve_memory(query)
                st.chat_message("assistant").write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Retrieval Error: {e}")

with tab3:
    st.subheader("Neo4j Memory Chain Inspection")
    st.markdown("Visualizing chronological sequences (`[:NEXT_IN_CHAIN]`) and Layer 3 token compression status.")

    if st.button("Refresh Graph View"):
        st.rerun()

    try:
        with pipeline.graph_store.session() as s:
            chains = s.run("MATCH (c:MemoryChain) RETURN c.category_name AS name ORDER BY name").data()
            
            if not chains:
                st.warning("No memory chains found. Run a session in Tab 1.")
            else:
                for c in chains:
                    cat_name = c['name']
                    nodes = pipeline.graph_store.get_chain_nodes(cat_name)
                    
                    with st.expander(f"📁 Category Chain: `{cat_name}` (Total Nodes: {len(nodes)})", expanded=True):
                        for idx, n in enumerate(nodes):
                            st.markdown(f"**[{idx}] Session:** `{n.get('session_id', 'N/A')}`")
                            st.markdown(f"> *Summary:* {n.get('summary', '')}")
                            
                            # Validates Eq. 7 Token Compression state[cite: 2]
                            is_compressed = n.get('compressed', False)
                            status_badge = "✅ Compressed" if is_compressed else "❌ Raw"
                            st.caption(f"Node ID: `{n.get('id', '')}` | State: `{status_badge}`")
                            
                            if idx < len(nodes) - 1:
                                st.markdown("⬇️ `[:NEXT_IN_CHAIN]`")

                shared_count = s.run("MATCH ()-[r:SHARED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]
                st.metric(label="Total Cross-Category `[:SHARED_WITH]` Relationships", value=shared_count)

    except Exception as err:
        st.error(f"Error querying Neo4j graph store: {err}")