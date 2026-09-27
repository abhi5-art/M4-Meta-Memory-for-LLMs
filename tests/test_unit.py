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
