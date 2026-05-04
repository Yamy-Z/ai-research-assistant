from anthropic import Anthropic
from app.core.config import get_settings
from app.services.hybrid_search import HybridSearchService
from app.services.web_search import WebSearchService
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import time

settings = get_settings()
logger = setup_logger(__name__)


class RAGService:
    """Enhanced RAG service with hybrid search."""
    
    def __init__(
        self,
        hybrid_search_service: HybridSearchService,
        web_search_service: WebSearchService
    ):
        self.hybrid_search = hybrid_search_service
        self.web_search = web_search_service
        self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
    
    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_web_search: bool = True,
        web_results: int = 3,
        alpha: float = 0.7  # Weight for vector vs BM25
    ) -> Dict[str, Any]:
        """
        Answer query using hybrid RAG.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            use_web_search: Include web search
            web_results: Number of web results
            alpha: Vector search weight (0.0-1.0)
        
        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()
        
        logger.info(f"Processing query with hybrid search: {query}")
        
        # 1. Hybrid search (vector + BM25)
        hybrid_results = self.hybrid_search.search(
            query=query,
            top_k=top_k,
            alpha=alpha
        )
        
        all_sources = []
        
        # Add hybrid results
        for result in hybrid_results:
            all_sources.append({
                "content": result["content"],
                "score": result["score"],
                "source_type": "document",
                "metadata": result["metadata"]
            })
        
        # 2. Web search if enabled
        if use_web_search:
            logger.info("Performing web search...")
            web_results_list = self.web_search.search(
                query=query,
                max_results=web_results
            )
            
            for result in web_results_list:
                all_sources.append({
                    "content": result["content"],
                    "score": result["score"],
                    "source_type": "web",
                    "metadata": {
                        "title": result["title"],
                        "url": result["url"]
                    }
                })
        
        # 3. Generate answer
        if not all_sources:
            return {
                "answer": "I couldn't find relevant information.",
                "sources": [],
                "query_time_ms": (time.time() - start_time) * 1000
            }
        
        context = self._build_context(all_sources)
        answer = self._generate_answer(query, context)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query completed in {query_time_ms:.2f}ms")
        
        return {
            "answer": answer,
            "sources": all_sources,
            "query_time_ms": query_time_ms
        }
    
    def _build_context(self, sources: List[Dict]) -> str:
        """Build context string from all sources."""
        context_parts = []
        
        for idx, source in enumerate(sources, 1):
            source_type = source["source_type"]
            content = source["content"]
            
            if source_type == "document":
                filename = source["metadata"]["filename"]
                context_parts.append(
                    f"[Source {idx} - Document: {filename}]\n{content}\n"
                )
            elif source_type == "web":
                title = source["metadata"]["title"]
                url = source["metadata"]["url"]
                context_parts.append(
                    f"[Source {idx} - Web: {title}]\nURL: {url}\n{content}\n"
                )
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Claude with enhanced prompting."""
        
        prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided sources.

            Sources (both documents and web pages):
            {context}

            Question: {query}

            Instructions:
            - Answer based on the provided sources
            - Cite sources using [Source N] notation
            - If sources conflict, mention both perspectives
            - If information is insufficient, say so
            - Be accurate and concise
            - Distinguish between document sources and web sources when relevant

            Answer:"""

        try:
            response = self.llm_client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."


def get_rag_service(
    hybrid_search_service: HybridSearchService,
    web_search_service: WebSearchService
) -> RAGService:
    """Get RAG service dependency."""
    return RAGService(hybrid_search_service, web_search_service)
