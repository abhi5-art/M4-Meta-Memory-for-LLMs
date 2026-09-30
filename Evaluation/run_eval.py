# print("--- 1. Script started. Loading imports (This takes a few seconds) ---")
# import sys
# import os
# import json
# import time
# from tabulate import tabulate

# print("--- 2. Loading DeepEval... ---")
# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# print("--- 3. Loading M4 Modules & NLP Metrics... ---")
# print("--- 3a. Importing src.config ---")
# from src.config import AppConfig

# print("--- 3b. Importing src.llm_client ---")
# from src.llm_client import OllamaClient

# print("--- 3c. Importing src.graph_store ---")
# from src.graph_store import Neo4jGraphStore

# print("--- 3d. Importing Layer 1 ---")
# from src.layer1_learning import CategoryBasedMemoryLearning

# print("--- 3e. Importing Layer 2 ---")
# from src.layer2_construction import ChainBasedMemoryConstruction

# print("--- 3f. Importing Layer 3 ---")
# from src.layer3_updating import SelfMonitoringMemoryUpdating

# print("--- 3g. Importing Layer 4 ---")
# from src.layer4_retrieval import SelfReflectionMemoryRetrieval

# print("--- 3h. Importing Paper Metrics ---")
# from metrices.m4_paper_metrics import (
#     PaperMatchedBLEU4,
#     PaperMatchedRougeL,
#     PaperMatchedBERTScore,
#     TokenCompressionFactorMetric
# )
# print("--- 3i. All imports complete! ---")

# def run_integrated_m4_pipeline(
#     query: str, 
#     raw_memory: str, 
#     l1: CategoryBasedMemoryLearning,
#     l2: ChainBasedMemoryConstruction,
#     l3: SelfMonitoringMemoryUpdating,
#     l4: SelfReflectionMemoryRetrieval,
#     graph_store: Neo4jGraphStore
# ) -> dict:
    
#     # Step 0: Isolate Test Environment (Clear Graph DB)
#     with graph_store.session() as s:
#         s.run("MATCH (n) DETACH DELETE n")

#     # Step 1: Category-Based Memory Learning (Eq. 1)
#     # Extracts candidate memory categories and clues[cite: 12].
#     clues = l1.process_session(raw_memory)
    
#     # Step 2: Chain-Based Memory Construction (Eq. 2 & 3)
#     # Organizes disordered memory clues into chronological chains[cite: 13].
#     if clues:
#         l2.build_memory_chains(clues, session_id="eval_session_01")
    
#     # Step 3: Self-Monitoring-Based Memory Updating (Eq. 4-7)
#     # We pass days_idle=-1 to force threshold_date into the future, 
#     # forcing Ebbinghaus compression immediately for metric calculation[cite: 14].
#     l3.compress_dormant_nodes(days_idle=-1)
    
#     # Step 4: Self-Reflection-Based Memory Retrieval (Algorithm 1 & Eq. 8-9)
#     # Retrieves relevant nodes via LLM confidence and generates the response[cite: 15].
#     memory_context_list, response = l4.retrieve_memory(query)
    
#     # Join the retrieved context array into a single string for evaluation
#     compressed_memory = "\n".join(memory_context_list) if memory_context_list else ""

#     return {
#         "response": response,
#         "compressed_memory": compressed_memory
#     }

# def main():
#     print("\n Initializing M4 Architecture Components...")
    
#     # Initialize Core Infrastructure
#     config = AppConfig()
#     client = OllamaClient(config)
#     graph_store = Neo4jGraphStore(config)
    
#     # Instantiate Pipeline Layers
#     l1 = CategoryBasedMemoryLearning(config, client)
#     l2 = ChainBasedMemoryConstruction(config, graph_store)
#     l3 = SelfMonitoringMemoryUpdating(config)
#     l4 = SelfReflectionMemoryRetrieval(config)

#     print(" Loading Dataset...")
#     dataset_path = os.path.join(os.path.dirname(__file__), "data", "mini_benchmark.json")
#     with open(dataset_path, "r") as f:
#         test_data = json.load(f)

#     test_cases = []
#     latency_records = []

#     print(" Processing test cases through M4 Pipeline...\n")
#     for item in test_data:
#         print(f"--- Running Test: {item['id']} ---")
#         start_time = time.time()
        
#         # Execute the integrated pipeline
#         result = run_integrated_m4_pipeline(
#             query=item["query"], 
#             raw_memory=item["raw_memory"],
#             l1=l1, l2=l2, l3=l3, l4=l4,
#             graph_store=graph_store
#         )
        
#         latency = round(time.time() - start_time, 2)
#         latency_records.append([item["id"], item["task"], f"{latency}s"])

#         # Construct test case for DeepEval
#         tc = LLMTestCase(
#             input=item["query"],
#             actual_output=result["response"],
#             expected_output=item["expected_output"],
#             retrieval_context=[result["compressed_memory"]],
#             context=[item["raw_memory"]]  # <--- ADD THIS LINE
#         )
#         test_cases.append(tc)

#     # Instantiate the metrics (Targeting research paper thresholds)
#     metrics = [
#         PaperMatchedBLEU4(threshold=1.0),
#         PaperMatchedRougeL(threshold=14.0),
#         PaperMatchedBERTScore(threshold=45.0),
#         TokenCompressionFactorMetric(
#             target_factor=2.0
#         )
#     ]

#     print("\n Calculating Academic NLP Metrics via DeepEval...\n")
#     evaluate(test_cases, metrics)

#     # Print System Efficiency & Latency Table (To match Paper Fig 13)
#     print("\n SYSTEM EFFICIENCY & LATENCY TABLE")
#     print(tabulate(latency_records, headers=["Test ID", "Reasoning Task", "Latency (sec)"], tablefmt="fancy_grid"))
#     print("\n Evaluation Complete. Graph DB has been cleaned.")

# if __name__ == "__main__":
#     main()


########### Updated.......................

import random # Add this at the top of your script
import sys
import os
import json
import time
import argparse
from tabulate import tabulate

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.llm_client import OllamaClient
from src.graph_store import Neo4jGraphStore
from src.layer1_learning import CategoryBasedMemoryLearning
from src.layer2_construction import ChainBasedMemoryConstruction
from src.layer3_updating import SelfMonitoringMemoryUpdating
from src.layer4_retrieval import SelfReflectionMemoryRetrieval

from metrices.m4_paper_metrics import (
    PaperMatchedBLEU4,
    PaperMatchedRougeL,
    PaperMatchedBERTScore,
    PaperMatchedBLEU1,
    PaperMatchedF1,
    TokenCompressionFactorMetric
)

def run_integrated_m4_pipeline(
    query: str, 
    raw_memory: str, 
    l1: CategoryBasedMemoryLearning,
    l2: ChainBasedMemoryConstruction,
    l3: SelfMonitoringMemoryUpdating,
    l4: SelfReflectionMemoryRetrieval,
    graph_store: Neo4jGraphStore
) -> dict:
    
    # Step 0: Isolate Test Environment (Clear Graph DB for current test)
    with graph_store.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

    # Step 1: Category-Based Memory Learning (Eq. 1)
    clues = l1.process_session(raw_memory)
    
    # Step 2: Chain-Based Memory Construction (Eq. 2 & 3)
    if clues:
        l2.build_memory_chains(clues, session_id="eval_session_01")
    
    # Step 3: Self-Monitoring-Based Memory Updating (Eq. 4-7)
    l3.compress_dormant_nodes(days_idle=-1)
    
    # Step 4: Self-Reflection-Based Memory Retrieval (Algorithm 1 & Eq. 8-9)
    memory_context_list, response = l4.retrieve_memory(query)
    
    # Join the retrieved context array into a single string for evaluation
    compressed_memory = "\n".join(memory_context_list) if memory_context_list else ""

    return {
        "response": str(response) if response is not None else "",
        "compressed_memory": compressed_memory
    }

def main():
    parser = argparse.ArgumentParser(description="M4 Architecture Evaluation")
    parser.add_argument("--dataset", type=str, choices=["msc", "locomo"], required=True, 
                        help="Choose which dataset to evaluate")
    # NEW FLAG: Limits the number of tests to bypass GPU constraints
    parser.add_argument("--limit", type=int, default=None, 
                        help="Number of test cases to run (randomly sampled)")
    args = parser.parse_args()

    print(f"\n🚀 Initializing M4 Architecture Components for {args.dataset.upper()} dataset...")
    
    config = AppConfig()
    client = OllamaClient(config)
    graph_store = Neo4jGraphStore(config)
    
    l1 = CategoryBasedMemoryLearning(config, client)
    l2 = ChainBasedMemoryConstruction(config, graph_store)
    l3 = SelfMonitoringMemoryUpdating(config)
    l4 = SelfReflectionMemoryRetrieval(config)

    # 1. Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "data", f"{args.dataset}_benchmark.json")
    with open(dataset_path, "r") as f:
        raw_test_data = json.load(f)

    # 2. Flatten and normalize data based on dataset schema
    test_data = []
    if args.dataset == "msc":
        for item in raw_test_data:
            test_data.append({
                "id": str(item.get("id", "")),
                "query": str(item.get("query", "")),
                "expected_output": str(item.get("expected_output", "")),
                "raw_memory": str(item.get("raw_memory", "")),
                "task": str(item.get("task", "unknown"))
            })
    elif args.dataset == "locomo":
        for sample in raw_test_data:
            sample_id = str(sample.get("sample_id", "locomo_sample"))
            context_parts = []
            if "conversation" in sample:
                context_parts.append(f"CONVERSATION LOG:\n{json.dumps(sample['conversation'])}")
            if "observation" in sample:
                context_parts.append(f"OBSERVATIONS:\n{json.dumps(sample['observation'])}")
            if "event_summary" in sample:
                context_parts.append(f"EVENT SUMMARIES:\n{json.dumps(sample['event_summary'])}")
                
            combined_raw_memory = "\n\n".join(context_parts) if context_parts else str(sample)
            
            for idx, qa_item in enumerate(sample.get("qa", [])):
                test_data.append({
                    "id": f"{sample_id}_qa_{idx}",
                    "query": str(qa_item.get("question", "")),
                    "expected_output": str(qa_item.get("answer", "")),
                    "raw_memory": combined_raw_memory,
                    "task": str(qa_item.get("category", "unknown"))
                })

    # --- NEW: SCIENTIFIC SUBSETTING FOR GPU LIMITS ---
    if args.limit and args.limit < len(test_data):
        print(f"📉 Applying random sampling: Reducing {len(test_data)} test cases to {args.limit}...")
        random.seed(42) # Ensures reproducibility (you will get the exact same random subset every time)
        test_data = random.sample(test_data, args.limit)

    test_cases = []
    latency_records = []

    print(f"🧠 Processing {len(test_data)} test cases through M4 Pipeline...\n")
    for item in test_data:
        start_time = time.time()
        
        result = run_integrated_m4_pipeline(
            query=item["query"], 
            raw_memory=item["raw_memory"],
            l1=l1, l2=l2, l3=l3, l4=l4,
            graph_store=graph_store
        )
        
        latency = round(time.time() - start_time, 2)
        latency_records.append([item["id"], item["task"], f"{latency}s"])

        tc = LLMTestCase(
            input=item["query"],
            actual_output=str(result["response"]),
            expected_output=item["expected_output"],
            retrieval_context=[str(result["compressed_memory"])],
            context=[item["raw_memory"]]  
        )
        test_cases.append(tc)

    # 3. Apply the appropriate metrics based on the benchmark
    if args.dataset == "msc":
        metrics = [
            PaperMatchedBLEU4(threshold=5.0),
            PaperMatchedRougeL(threshold=30.0),
            PaperMatchedBERTScore(threshold=85.0),
            TokenCompressionFactorMetric(target_factor=2.0)
        ]
    else:
        metrics = [
            PaperMatchedBLEU1(threshold=15.0),
            PaperMatchedF1(threshold=25.0),
            TokenCompressionFactorMetric(target_factor=2.0)
        ]

    print(f"\n📊 Calculating Academic NLP Metrics for {args.dataset.upper()}...\n")
    evaluate(test_cases, metrics)
    
    print("\n⏱️ SYSTEM EFFICIENCY & LATENCY TABLE")
    print(tabulate(latency_records, headers=["Test ID", "Reasoning Task", "Latency (sec)"], tablefmt="fancy_grid"))
    
    print("\n🧹 Performing final Graph DB cleanup...")
    with graph_store.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    main()