from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class Source(BaseModel):
    content: str
    metadata: dict
    score: float
    source_type: Optional[str] = None


class Citation(BaseModel):
    source_number: str
    content: str
    source_type: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    citations: Dict[str, Citation] = {}       
    citation_count: int = 0                     
    query_time_ms: float = 0


class DocumentUpload(BaseModel):
    filename: str
    content: str
    metadata: Optional[dict] = {}


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str
