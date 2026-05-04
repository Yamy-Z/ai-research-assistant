from tavily import TavilyClient
from app.core.config import get_settings
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import time

settings = get_settings()
logger = setup_logger(__name__)


class WebSearchService:
    """Service for searching the web using Tavily."""
    
    def __init__(self):
        self.client = TavilyClient(api_key=settings.tavily_api_key)
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced"
            include_raw_content: Include full page content
        
        Returns:
            List of search results with title, url, content, score
        """
        try:
            start_time = time.time()
            
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content
            )
            
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Web search completed in {search_time:.2f}ms")
            
            # Format results
            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'raw_content': result.get('raw_content', ''),
                    'score': result.get('score', 0.0),
                    'source_type': 'web'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def search_with_context(
        self,
        query: str,
        context: str = "",
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with additional context for better results.
        
        Args:
            query: Search query
            context: Additional context to improve search
            max_results: Maximum results
        
        Returns:
            List of search results
        """
        # Combine query with context for better search
        enhanced_query = f"{query} {context}".strip()
        return self.search(enhanced_query, max_results)


def get_web_search_service() -> WebSearchService:
    """Get web search service dependency."""
    return WebSearchService()