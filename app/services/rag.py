from anthropic import Anthropic
from app.core.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import time

settings = get_settings()
logger = setup_logger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
    
    def answer_query(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()
        
        # 1. Generate query embedding
        logger.info(f"Processing query: {query}")
        query_embedding = self.embedding_service.embed_text(query)
        
        # 2. Retrieve relevant documents
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.5
        )
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query_time_ms": (time.time() - start_time) * 1000
            }
        
        # 3. Build context from retrieved chunks
        context = self._build_context(results)
        
        # 4. Generate answer using LLM
        answer = self._generate_answer(query, context)
        
        # 5. Prepare response
        sources = [
            {
                "content": result["payload"]["content"],
                "score": result["score"],
                "metadata": {
                    "filename": result["payload"]["filename"],
                    "chunk_index": result["payload"]["chunk_index"]
                }
            }
            for result in results
        ]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Query completed in {query_time_ms:.2f}ms")
        
        return {
            "answer": answer,
            "sources": sources,
            "query_time_ms": query_time_ms
        }
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from search results."""
        context_parts = []
        for idx, result in enumerate(results, 1):
            content = result["payload"]["content"]
            filename = result["payload"]["filename"]
            context_parts.append(
                f"[Source {idx} - {filename}]\n{content}\n"
            )
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Claude."""
        
        prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based solely on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite which source(s) you used by referring to [Source N]

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
            
            answer = response.content[0].text
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."


def get_rag_service(
    embedding_service: EmbeddingService,
    vector_store: VectorStore
) -> RAGService:
    """Get RAG service dependency."""
    return RAGService(embedding_service, vector_store)