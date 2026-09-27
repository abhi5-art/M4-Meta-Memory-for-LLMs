# import sys
# from src.config import AppConfig, setup_logger
# from src.pipeline import MetaMemoryPipeline

# logger = setup_logger("m4.demo")

# # Dynamic, generic test sessions to let the LLM categorize organically
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

# def main():
#     logger.info("Initializing Meta-Memory (M4) Production Pipeline...")
#     config = AppConfig()
#     pipeline = MetaMemoryPipeline(config)

#     try:
#         pipeline.startup()
#     except Exception as e:
#         logger.error("Failed to connect to Neo4j. Verify container is healthy. Error: %s", str(e))
#         sys.exit(1)

#     # Verify that local Ollama service is active and accessible
#     try:
#         pipeline.llm_client.get_embedding("health check probe")
#         logger.info("Local Ollama connection verified (/api/embeddings & /v1/chat/completions active).")
#     except Exception as e:
#         logger.error("Ollama is not reachable at %s. Ensure Ollama is running locally. Error: %s",
#                      config.ollama_base_url, str(e))
#         sys.exit(1)

#     # Execute consecutive sessions to demonstrate Layer 1 & 2 dynamic extraction and chaining
#     sessions = [
#         ("sess_001_backend_ops", SESSION_1),
#         ("sess_002_cache_migration", SESSION_2),
#         ("sess_003_containerization", SESSION_3)
#     ]

#     for sess_id, dialogue in sessions:
#         logger.info("=== Ingesting Session [%s] ===", sess_id)
#         res = pipeline.ingest_session(sess_id, dialogue)
#         logger.info("Session Result: %s", res)

#     # Inspect the resulting Graph Database dynamically
#     logger.info("=== Graph Store Inspection (Verification of Dynamic Memory Chains) ===")
#     with pipeline.graph_store.session() as s:
#         chains = s.run("MATCH (c:MemoryChain) RETURN c.category_name AS name ORDER BY name").data()
#         logger.info("Created Chains: %s", [c['name'] for c in chains])

#         for c in chains:
#             cat_name = c['name']
#             nodes = pipeline.graph_store.get_chain_nodes(cat_name)
#             logger.info("Chain ['%s'] (Length=%d):", cat_name, len(nodes))
#             for idx, n in enumerate(nodes):
#                 logger.info("  [%d] (Session: %s) -> %s", idx, n['session_id'], n['summary'])

#         shared_count = s.run("MATCH ()-[r:SHARED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]
#         logger.info("Total cross-category [:SHARED_WITH] graph relationships: %d", shared_count)

#     pipeline.shutdown()
#     logger.info("M4 Demonstration Completed Successfully.")

# if __name__ == "__main__":
#     main()







#  The below code is for layer 1,2,3,4 combined ----------------------------------------------------------







import sys
from src.config import AppConfig, setup_logger
from src.pipeline import MetaMemoryPipeline

logger = setup_logger("m4.demo")

# Dynamic Benchmark Sessions for Layer 1 & 2 organic categorization[cite: 2]
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

# Off-context domains to verify distinct chain instantiation
SESSION_OFF_CONTEXT_1 = """
User: Hi! I am trying to master authentic Neapolitan pizza dough and sourdough fermentation this weekend.
Assistant: Baking is a wonderful science! Are you using a preferment or a direct dough method?
User: Using a 24-hour cold fermentation poolish with Caputo 00 flour. Also picked up a cast-iron skillet for baking.
"""

SESSION_OFF_CONTEXT_2 = """
User: Hey! Switching gears completely—I'm learning fingerstyle acoustic guitar and practicing alternate tunings like DADGAD.
Assistant: DADGAD opens up wonderful sonic textures for Celtic and folk arrangements. 
User: Slowly but steadily. Also bought a solid mahogany acoustic guitar to get a warmer tone.
"""

def main():
    logger.info("Initializing Meta-Memory (M4) Production Pipeline[cite: 2]...")
    config = AppConfig()
    pipeline = MetaMemoryPipeline(config)

    try:
        pipeline.startup()
    except Exception as e:
        logger.error("Failed to connect to Neo4j. Verify container is healthy. Error: %s", str(e))
        sys.exit(1)

    # Verify live connections - no hardcoded mocks
    try:
        pipeline.llm_client.get_embedding("health check probe")
        logger.info("Local Ollama connection verified (/api/embeddings & /v1/chat/completions active).")
    except Exception as e:
        logger.error("Ollama is not reachable at %s. Ensure Ollama is running locally. Error: %s",
                     config.ollama_base_url, str(e))
        sys.exit(1)

    # 1. Execute consecutive sessions to trigger Learning, Construction, and Updating (Layers 1-3)
    sessions = [
        ("sess_001_backend", SESSION_1),
        ("sess_002_cache", SESSION_2),
        ("sess_003_docker", SESSION_3),
        ("sess_off_001_baking", SESSION_OFF_CONTEXT_1),
        ("sess_off_002_guitar", SESSION_OFF_CONTEXT_2)
    ]

    for sess_id, dialogue in sessions:
        logger.info("=== Ingesting Session [%s] ===", sess_id)
        # Handles Eq. 1-7 (Extraction, Construction, Calibration, Compression)[cite: 2]
        res = pipeline.ingest_session(sess_id, dialogue)
        logger.info("Session Result: %s", res)

    # 2. Execute Self-Reflection Memory Retrieval (Layer 4)
    logger.info("=== Testing Layer 4: Self-Reflection Retrieval (Algorithm 1)[cite: 2] ===")
    test_queries = [
        "What is my morning review routine?",
        "Did I make any updates to the caching layer recently?",
        "What type of guitar did I buy and what tuning am I practicing?"
    ]

    for q in test_queries:
        logger.info("Querying M4: '%s'", q)
        # Evaluates LLM_re and LLM_confident stopping criteria (Eq. 8-9)[cite: 2]
        ans = pipeline.layer4.retrieve_memory(q)
        logger.info("M4 Final Response: %s", ans)

    # 3. Inspect final database topology
    logger.info("=== Graph Store Inspection (Verification of Dynamic Memory Chains) ===")
    with pipeline.graph_store.session() as s:
        chains = s.run("MATCH (c:MemoryChain) RETURN c.category_name AS name ORDER BY name").data()
        logger.info("Created Chains: %s", [c['name'] for c in chains])

        for c in chains:
            cat_name = c['name']
            nodes = pipeline.graph_store.get_chain_nodes(cat_name)
            logger.info("Chain ['%s'] (Length=%d):", cat_name, len(nodes))
            for idx, n in enumerate(nodes):
                compression_state = n.get('compressed', False)
                logger.info("  [%d] (Session: %s) -> %s (Compressed: %s)", 
                            idx, n['session_id'], n['summary'], compression_state)

        shared_count = s.run("MATCH ()-[r:SHARED_WITH]->() RETURN count(r) AS cnt").single()["cnt"]
        logger.info("Total cross-category [:SHARED_WITH] graph relationships: %d", shared_count)

    pipeline.shutdown()
    logger.info("M4 End-to-End Demonstration Completed Successfully.")

if __name__ == "__main__":
    main()