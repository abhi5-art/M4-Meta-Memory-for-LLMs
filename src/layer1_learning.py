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
