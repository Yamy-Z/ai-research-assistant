from app.utils.logger import setup_logger
from typing import List, Dict, Any, Optional
import time

logger = setup_logger(__name__)


class RerankerService:
    """Re-rank search results using cross-encoder."""
    
    # Use a lightweight cross-encoder model
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder model: {self.MODEL_NAME}")
        self.model = CrossEncoder(self.MODEL_NAME)
        logger.info("Cross-encoder model loaded")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of top results to return (None = all)
        
        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return []
        
        start_time = time.time()
        
        # Prepare query-document pairs
        pairs = [(query, result['content']) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_result = result.copy()
            reranked_result['original_score'] = result['score']
            reranked_result['rerank_score'] = float(score)
            reranked_result['score'] = float(score)  # Use rerank score as primary
            reranked_results.append(reranked_result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply top_k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        rerank_time = (time.time() - start_time) * 1000
        logger.info(
            f"Re-ranked {len(results)} results in {rerank_time:.2f}ms, "
            f"returned top {len(reranked_results)}"
        )
        
        return reranked_results


# Singleton instance
_reranker_instance = None

def get_reranker_service() -> RerankerService:
    """Get reranker service dependency (singleton)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance
