from anthropic import Anthropic
from app.core.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.web_search import WebSearchService
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import time

settings = get_settings()
logger = setup_logger(__name__)


class RAGService:
    """Enhanced Retrieval-Augmented Generation service with web search."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        web_search_service: WebSearchService
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.web_search_service = web_search_service
        self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
    
    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_web_search: bool = True,
        web_results: int = 3
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG with optional web search.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve from vector DB
            use_web_search: Whether to include web search results
            web_results: Number of web results to include
        
        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query}")
        
        # 1. Retrieve from vector database
        query_embedding = self.embedding_service.embed_text(query)
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.5
        )
        
        all_sources = []
        
        # Add vector DB results
        for result in vector_results:
            all_sources.append({
                "content": result["payload"]["content"],
                "score": result["score"],
                "source_type": "document",
                "metadata": {
                    "filename": result["payload"]["filename"],
                    "chunk_index": result["payload"]["chunk_index"]
                }
            })
        
        # 2. Web search if enabled
        if use_web_search:
            logger.info("Performing web search...")
            web_results_list = self.web_search_service.search(
                query=query,
                max_results=web_results
            )
            
            # Add web results
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
        
        # 3. Generate answer if we have sources
        if not all_sources:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query_time_ms": (time.time() - start_time) * 1000,
                "source_breakdown": {
                    "documents": 0,
                    "web": 0
                }
            }
        
        # 4. Build context and generate answer
        context = self._build_context(all_sources)
        answer = self._generate_answer(query, context)
        
        # 5. Calculate metrics
        query_time_ms = (time.time() - start_time) * 1000
        source_breakdown = {
            "documents": len([s for s in all_sources if s["source_type"] == "document"]),
            "web": len([s for s in all_sources if s["source_type"] == "web"])
        }
        
        logger.info(
            f"Query completed in {query_time_ms:.2f}ms "
            f"(docs: {source_breakdown['documents']}, web: {source_breakdown['web']})"
        )
        
        return {
            "answer": answer,
            "sources": all_sources,
            "query_time_ms": query_time_ms,
            "source_breakdown": source_breakdown
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
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    web_search_service: WebSearchService
) -> RAGService:
    """Get RAG service dependency."""
    return RAGService(embedding_service, vector_store, web_search_service)