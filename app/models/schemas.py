from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime


class QueryRequest(BaseModel):
    """Request model for research queries."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class Source(BaseModel):
    """Source document information."""
    content: str
    metadata: dict
    score: float
    source_type: str


class QueryResponse(BaseModel):
    """Response model for research queries."""
    answer: str
    sources: List[Source]
    query_time_ms: float


class DocumentUpload(BaseModel):
    """Document upload metadata."""
    filename: str
    content: str
    metadata: Optional[dict] = {}


class DocumentResponse(BaseModel):
    """Response after document indexing."""
    document_id: str
    filename: str
    chunks_created: int
    status: str
