from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import Query as QueryModel
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embedding import get_embedding_service, EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.rag import get_rag_service, RAGService
from app.utils.logger import setup_logger

router = APIRouter(prefix="/query", tags=["query"])
logger = setup_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Answer a research query using RAG.
    
    This endpoint:
    1. Embeds the query
    2. Retrieves relevant documents from vector store
    3. Generates answer using Claude
    4. Returns answer with sources
    """
    
    # Get RAG service
    rag_service = get_rag_service(embedding_service, vector_store)
    
    # Process query
    result = rag_service.answer_query(
        query=request.query,
        top_k=request.top_k
    )
    
    # Save query history
    query_record = QueryModel(
        query_text=request.query,
        answer=result["answer"],
        query_time_ms=result["query_time_ms"],
        sources_count=len(result["sources"])
    )
    db.add(query_record)
    db.commit()
    
    # Return response
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"] if request.include_sources else [],
        query_time_ms=result["query_time_ms"]
    )


@router.get("/history")
async def get_query_history(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent query history."""
    queries = db.query(QueryModel)\
        .order_by(QueryModel.created_at.desc())\
        .limit(limit)\
        .all()
    
    return [
        {
            "id": q.id,
            "query": q.query_text,
            "answer": q.answer,
            "query_time_ms": q.query_time_ms,
            "created_at": q.created_at
        }
        for q in queries
    ]