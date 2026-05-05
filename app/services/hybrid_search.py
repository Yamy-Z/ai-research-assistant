from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.bm25_search import BM25SearchService
from app.services.reranker import RerankerService
from app.utils.logger import setup_logger
from typing import List, Dict, Any

logger = setup_logger(__name__)


class HybridSearchService:
    """Hybrid search with re-ranking."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        bm25_service: BM25SearchService,
        reranker_service: RerankerService
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_service = bm25_service
        self.reranker = reranker_service
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        use_reranking: bool = True,
        rerank_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with optional re-ranking.
        
        Args:
            query: Search query
            top_k: Final number of results
            alpha: Vector search weight
            use_reranking: Whether to apply re-ranking
            rerank_top_k: Number of candidates for re-ranking
        
        Returns:
            List of results sorted by combined/reranked score
        """
        logger.info(f"Hybrid search with reranking={use_reranking}")
        
        # 1. Get candidates (more than final top_k)
        candidate_k = rerank_top_k if use_reranking else top_k
        
        # Vector search
        query_embedding = self.embedding_service.embed_text(query)
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=candidate_k * 2,
            score_threshold=0.0
        )
        
        # BM25 search
        bm25_results = self.bm25_service.search(
            query=query,
            top_k=candidate_k * 2
        )
        
        # 2. Combine using RRF
        combined_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            alpha=alpha
        )
        
        # Take top candidates
        candidates = combined_results[:candidate_k]
        
        # 3. Re-rank if enabled
        if use_reranking and candidates:
            logger.info(f"Re-ranking top {len(candidates)} candidates...")
            final_results = self.reranker.rerank(
                query=query,
                results=candidates,
                top_k=top_k
            )
        else:
            final_results = candidates[:top_k]
        
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        # Track unique documents by chunk_id
        doc_scores = {}
        doc_data = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result["payload"]["chunk_id"]
            rrf_score = alpha / (k + rank)
            
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = 0.0
                doc_data[chunk_id] = {
                    'content': result["payload"]["content"],
                    'document_id': result["payload"]["document_id"],
                    'chunk_index': result["payload"]["chunk_index"],
                    'filename': result["payload"]["filename"],
                    'vector_score': result["score"],
                    'bm25_score': 0.0
                }
            
            doc_scores[chunk_id] += rrf_score
            doc_data[chunk_id]['vector_rank'] = rank
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result["id"]
            rrf_score = (1 - alpha) / (k + rank)
            
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = 0.0
                doc_data[chunk_id] = {
                    'content': result["content"],
                    'document_id': result["document_id"],
                    'chunk_index': result["chunk_index"],
                    'filename': 'unknown',
                    'vector_score': 0.0,
                    'bm25_score': result["score"]
                }
            
            doc_scores[chunk_id] += rrf_score
            doc_data[chunk_id]['bm25_rank'] = rank
            doc_data[chunk_id]['bm25_score'] = result["score"]
        
        # Sort by combined score
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )
        
        # Build result list
        results = []
        for chunk_id in sorted_ids:
            data = doc_data[chunk_id]
            results.append({
                'id': chunk_id,
                'content': data['content'],
                'score': doc_scores[chunk_id],
                'source_type': 'hybrid',
                'metadata': {
                    'document_id': data['document_id'],
                    'chunk_index': data['chunk_index'],
                    'filename': data['filename'],
                    'vector_score': data['vector_score'],
                    'bm25_score': data['bm25_score'],
                    'vector_rank': data.get('vector_rank', None),
                    'bm25_rank': data.get('bm25_rank', None)
                }
            })
        
        logger.info(f"Hybrid fusion: {len(results)} unique results")
        return results


def get_hybrid_search_service(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    bm25_service: BM25SearchService,
    reranker_service: RerankerService
) -> HybridSearchService:
    """Get hybrid search service dependency."""
    return HybridSearchService(
        embedding_service,
        vector_store,
        bm25_service,
        reranker_service
    )
