from app.services.hybrid_search import HybridSearchService
from app.services.web_search import WebSearchService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.bm25_search import BM25SearchService
from app.services.reranker import RerankerService
from app.agents.state import AgentState
from app.utils.logger import setup_logger
from anthropic import Anthropic
from app.core.config import get_settings

settings = get_settings()
logger = setup_logger(__name__)


class ResearchAgent:
    """
    Research agent that uses advanced RAG system.
    """
    
    def __init__(
        self,
        hybrid_search: HybridSearchService,
        web_search: WebSearchService
    ):
        self.hybrid_search = hybrid_search
        self.web_search = web_search
        self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute research for the query.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with research results
        """
        query = state["query"]
        logger.info(f"Research agent processing: {query}")
        
        try:
            # 1. Hybrid search (vector + BM25)
            logger.info("Performing hybrid search...")
            hybrid_results = self.hybrid_search.search(
                query=query,
                top_k=5,
                alpha=0.7,
                use_reranking=True,
                rerank_top_k=10
            )
            
            # 2. Web search
            logger.info("Performing web search...")
            web_results = self.web_search.search(
                query=query,
                max_results=3
            )
            
            # 3. Combine sources
            all_sources = []
            
            for result in hybrid_results:
                all_sources.append({
                    "content": result["content"],
                    "score": result["score"],
                    "source_type": "document",
                    "metadata": result.get("metadata", {})
                })
            
            for result in web_results:
                all_sources.append({
                    "content": result["content"],
                    "score": result["score"],
                    "source_type": "web",
                    "metadata": {
                        "title": result["title"],
                        "url": result["url"]
                    }
                })
            
            # 4. Generate answer
            answer = self._generate_answer(query, all_sources)
            
            # 5. Update state
            state["research_result"] = {
                "answer": answer,
                "sources": all_sources,
                "source_count": len(all_sources)
            }
            
            logger.info(f"Research completed with {len(all_sources)} sources")
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            state["research_result"] = {
                "answer": f"Research failed: {e}",
                "sources": [],
                "source_count": 0
            }
            state["errors"].append(f"Research failed: {e}")
        
        return state
    
    def _generate_answer(self, query: str, sources: list) -> str:
        """Generate answer using Claude."""
        
        if not sources:
            return "No relevant information found."
        
        # Build context
        context_parts = []
        for idx, source in enumerate(sources, 1):
            source_type = source["source_type"]
            content = source["content"]
            
            if source_type == "document":
                filename = source["metadata"].get("filename", "Unknown")
                context_parts.append(f"[Source {idx} - Document: {filename}]\n{content}\n")
            elif source_type == "web":
                title = source["metadata"].get("title", "Unknown")
                url = source["metadata"].get("url", "")
                context_parts.append(f"[Source {idx} - Web: {title}]\nURL: {url}\n{content}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a helpful research assistant. Answer the question based on the provided sources.

            Sources:
            {context}

            Question: {query}

            Instructions:
            - Answer based on the sources
            - Cite sources using [Source N]
            - Be accurate and concise
            - If sources conflict, mention both views
            - If information insufficient, say so

            Answer:"""

        try:
            response = self.llm_client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "Error generating answer."


def create_research_agent_node(
    hybrid_search: HybridSearchService,
    web_search: WebSearchService
):
    """Create research agent node for LangGraph."""
    agent = ResearchAgent(hybrid_search, web_search)
    
    def research_node(state: AgentState) -> AgentState:
        state["current_step"] = "research"
        state = agent.execute(state)
        state["steps_completed"].append("research")
        return state
    
    return research_node
