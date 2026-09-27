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
