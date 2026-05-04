from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import Query as QueryModel
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embedding import get_embedding_service, EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.web_search import get_web_search_service, WebSearchService
from app.services.rag import get_rag_service, RAGService
from app.utils.logger import setup_logger

router = APIRouter(prefix="/query", tags=["query"])
logger = setup_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    web_search_service: WebSearchService = Depends(get_web_search_service)
):
    """
    Answer a research query using RAG with web search.
    
    This endpoint:
    1. Retrieves relevant documents from vector store
    2. Searches the web for additional information
    3. Combines and ranks all sources
    4. Generates answer using Claude
    5. Returns answer with citations
    """
    
    # Get RAG service
    rag_service = get_rag_service(
        embedding_service,
        vector_store,
        web_search_service
    )
    
    # Process query
    result = rag_service.answer_query(
        query=request.query,
        top_k=request.top_k,
        use_web_search=True,  # Enable by default
        web_results=3
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