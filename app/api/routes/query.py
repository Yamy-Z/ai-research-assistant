from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database import Query as QueryModel
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embedding import get_embedding_service, EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.bm25_search import get_bm25_service
from app.services.hybrid_search import get_hybrid_search_service
from app.services.web_search import get_web_search_service, WebSearchService
from app.services.reranker import get_reranker_service, RerankerService
from app.services.citation import get_citation_service, CitationService
from app.services.rag import get_rag_service
from app.utils.logger import setup_logger
from app.services.query_analyzer import get_query_analyzer, QueryAnalyzer

router = APIRouter(prefix="/query", tags=["query"])
logger = setup_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    web_search_service: WebSearchService = Depends(get_web_search_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    citation_service: CitationService = Depends(get_citation_service),
    query_analyzer: QueryAnalyzer = Depends(get_query_analyzer)
):
    """Answer query using hybrid RAG with decomposition, re-ranking, and citations."""
    
    # 1. Create services
    bm25_service = get_bm25_service(db)
    
    hybrid_search_service = get_hybrid_search_service(
        embedding_service,
        vector_store,
        bm25_service,
        reranker_service
    )
    
    rag_service = get_rag_service(
        hybrid_search_service,
        web_search_service,
        query_analyzer 
    )
    
    # 2. Process query (analyzer decomposes complex queries automatically)
    result = rag_service.answer_query(
        query=request.query,
        top_k=request.top_k,
        use_web_search=True,
        web_results=3,
        alpha=0.7
    )
    
    # 3. Extract citations
    citation_result = citation_service.extract_citations(
        answer=result["answer"],
        sources=result["sources"]
    )
    
    # 4. Save query history
    query_record = QueryModel(
        query_text=request.query,
        answer=result["answer"],
        query_time_ms=result["query_time_ms"],
        sources_count=len(result["sources"])
    )
    db.add(query_record)
    db.commit()
    
    # 5. Return response
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"] if request.include_sources else [],
        citations=citation_result["citations"],
        citation_count=citation_result["citation_count"],
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
