from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class ChunkScore(BaseModel):
    chunk_id: str
    dense_score: float
    sparse_score: float
    final_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    scores: List[ChunkScore]
    latency_ms: float

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    page_number: int
    metadata: Dict[str, Any]